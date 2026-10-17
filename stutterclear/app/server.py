"""StutterClear app server: the Fluency Mirror UI on http://127.0.0.1:8765.

    python scripts/run_app.py

The UI is a Vite + React build in static/ (source in web/: `npm run build` there refreshes it).
"""
import asyncio
import os
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

from .. import nemotron
from ..audio import wav_bytes
from ..config import SAMPLE_RATE as SR
from ..live import DEFAULT_STUDENT, LiveSession, detectors, devices, recommended, student_eval
from . import demo, notes

STATIC = Path(__file__).parent / "static"
# A public deployment: visitors bring their own mic through the browser; the server never opens its
# own audio devices and never writes recordings to disk.
HOSTED = os.environ.get("STUTTERCLEAR_HOSTED") == "1"
UPDATE_S = 0.03  # ~30 updates a second: the orb reacts within a frame or two
# One live session at a time, so on a public deployment a tab left open must not hold it for ever.
LIVE_LIMIT_S = 300.0 if HOSTED else None
MIC_IDLE_S = 15.0  # no audio for this long: the page froze or the laptop went to sleep
# "last" holds the finished session's audio, in memory only, for the play buttons. Starting a new
# session drops it.
state = {"session": None, "last": None}
clients: set[WebSocket] = set()


async def broadcast():
    """Poll the session once per tick and send the same message to every open page.

    poll() hands out what is new since its last call, so polling per connection would split the
    audio between tabs (or an OBS browser source and the main window).
    """
    while True:
        s = state["session"]
        msg = s.poll() if s else {"idle": True}
        for socket in list(clients):
            try:
                await socket.send_json(msg)
            except Exception:  # closed mid-send; its handler cleans up
                clients.discard(socket)
        await asyncio.sleep(UPDATE_S)


def warm_demo():
    """Load the big model and render every demo sample once in the background, so the first live
    session and the first click on a sample start at once. Nemotron's model is checked first:
    the meaning guard runs while the samples render."""
    if nemotron.available():
        print(nemotron.resolve(), flush=True)
    from ..teacher_runtime import get_teacher, live_capable, window_probs
    if live_capable():
        get_teacher()
        window_probs(np.zeros(16000, np.float32))
    for s in demo.available():
        for engine in s["engines"]:
            try:
                demo.render(s["id"], engine)
            except Exception:  # a broken sample must not take the server down; it errors on click
                pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(broadcast())
    threading.Thread(target=warm_demo, daemon=True).start()
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="StutterClear", lifespan=lifespan)


class StartRequest(BaseModel):
    mode: str = "gentle"
    in_device: int | None = None
    out_device: int | None = None
    # Names win over indices: Windows renumbers devices whenever one (say, Bluetooth earbuds)
    # connects or drops, so an index the page remembered can point at nothing, or the wrong mic.
    in_name: str | None = None
    out_name: str | None = None
    detector: str = "student"
    save: bool = False


def resolve(kind: str, index: int | None, name: str | None, available: list[dict]) -> int | None:
    if name:
        for d in available:
            if d["name"] == name:
                return d["index"]
        raise HTTPException(409, f"{name} is not connected any more. Pick a {kind} again.")
    if index is not None and all(d["index"] != index for d in available):
        raise HTTPException(409, f"That {kind} is not available any more. Pick one again.")
    return index


@app.get("/api/devices")
def list_devices():
    return devices()


@app.get("/api/info")
def info():
    # The AI detector needs its downloaded checkpoint; its measured quality (if evaluated) lets the
    # page recommend it only when it is actually better.
    from .. import captions, context_agent
    return {"student": DEFAULT_STUDENT.exists(), "student_eval": student_eval(), "notes": notes.available(),
            "captions": captions.available(), "rescue": nemotron.available(), "nemotron": nemotron.available(),
            "search": context_agent.search_available(), "model": nemotron.MODEL, "hosted": HOSTED,
            "detectors": detectors(), "recommended": recommended()}


class ContextRequest(BaseModel):
    situation: str  # where the person is about to speak
    current: str = ""  # what the context card says already


@app.post("/api/context")
def prepare_context(req: ContextRequest):
    """Nemotron (with Tavily search when a key is set) writes a context card for the situation."""
    from .. import context_agent
    if not nemotron.available():
        raise HTTPException(503, "The context agent needs NEBIUS_API_KEY set where the app runs.")
    try:
        return context_agent.fill(req.situation, req.current)
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


class InsightsRequest(BaseModel):
    kinds: dict[str, int]  # stutters removed, by kind
    duration_s: float
    removed_s: float
    words: list[str] = []  # words the stutters were on
    transcript: str = ""


@app.post("/api/insights")
def session_insights(req: InsightsRequest):
    """Three short, supportive points about a session or voice note, from Nemotron (text only)."""
    from .. import insights
    if not nemotron.available():
        raise HTTPException(503, "Insights need NEBIUS_API_KEY set where the app runs.")
    try:
        return {"points": insights.summarize(req.kinds, req.duration_s, req.removed_s, req.words, req.transcript)}
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


class RescueRequest(BaseModel):
    said: str  # recent caption text
    context: str = ""  # the context card
    onset: str = ""  # the sound the person is stuck on, if known


@app.post("/api/rescue")
def word_rescue(req: RescueRequest):
    """Nemotron's guesses for the word the person is stuck on (opt-in; sends text, never audio)."""
    from .. import rescue
    if not rescue.available():
        raise HTTPException(503, "Word rescue needs NEBIUS_API_KEY set where the app runs.")
    try:
        return {"suggestions": rescue.suggest(req.said, req.context, req.onset), "model": nemotron.MODEL}
    except Exception as e:
        raise HTTPException(502, f"Nemotron did not answer: {e}") from e


@app.post("/api/start")
def start(req: StartRequest):
    if HOSTED:
        raise HTTPException(403, "On the hosted demo, live mode uses your browser's microphone.")
    if state["session"] is not None:
        state["session"].stop()
        state["session"] = None
    state["last"] = None
    devs = devices()
    in_device = resolve("microphone", req.in_device, req.in_name, devs["inputs"])
    out_device = resolve("output", req.out_device, req.out_name, devs["outputs"])
    s = LiveSession(req.mode, in_device, out_device, req.detector, save=req.save)
    try:
        s.start()
    except Exception as e:  # device busy or gone between listing and opening
        with suppress(Exception):
            s.stop()
        raise HTTPException(409, f"Could not open the audio device: {e}") from e
    state["session"] = s
    return {"ok": True, "detector": s.detector_name, "mode": req.mode}


@app.post("/api/stop")
def stop():
    s, state["session"] = state["session"], None
    if s is None:
        return {"ok": True, "summary": None}
    summary = s.stop()
    state["last"] = s.recording()
    return {"ok": True, "summary": summary, "playback": playback_info()}


@app.get("/api/playback")
def playback_info():
    last = state["last"]
    if not last or not len(last["original"]):
        return {"available": False}
    return {"available": True, **{f"{k}_s": round(len(v) / SR, 2) for k, v in last.items()}}


@app.get("/api/playback/{which}.wav")
def playback(which: str):
    last = state["last"]
    if which not in ("original", "clean") or not last:
        raise HTTPException(404, "Nothing to play yet: record a session first.")
    return Response(wav_bytes(last[which]), media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.websocket("/ws/mic")
async def browser_mic(socket: WebSocket):
    """Live from the browser's mic: the first message is the settings (JSON), then 16 kHz mono
    float32 audio as binary messages, then the text "stop" (or just closing). The session shows up
    on /ws like any other; at the end the server replies with the summary and keeps the recording
    for the play buttons. One live session at a time; the server ends it itself after
    LIVE_LIMIT_S or MIC_IDLE_S without audio, and says why ("ended")."""
    await socket.accept()
    if state["session"] is not None:
        await socket.close(code=4409, reason="A live session is already running here.")
        return
    cfg = await socket.receive_json()
    loop = asyncio.get_running_loop()
    s = await loop.run_in_executor(None, lambda: LiveSession(  # may load a model: off the event loop
        cfg.get("mode", "gentle"), detector=cfg.get("detector", "heuristic"), source="browser"))
    s.start()
    state["session"], state["last"] = s, None
    await socket.send_json({"ok": True, "detector": s.detector_name, "limit_s": LIVE_LIMIT_S})
    t0, ended = loop.time(), None
    try:
        while True:
            try:
                msg = await asyncio.wait_for(socket.receive(), timeout=MIC_IDLE_S)
            except asyncio.TimeoutError:
                ended = "No audio arrived for a while, so the session ended."
                break
            if msg["type"] == "websocket.disconnect" or msg.get("text") == "stop":
                break
            if LIVE_LIMIT_S and loop.time() - t0 > LIVE_LIMIT_S:
                ended = f"Live sessions here last up to {LIVE_LIMIT_S / 60:.0f} minutes, so others can try it too."
                break
            if msg.get("bytes"):
                await loop.run_in_executor(None, s.feed, np.frombuffer(msg["bytes"], np.float32))
    except WebSocketDisconnect:
        pass
    finally:
        summary = None
        if state["session"] is s:
            state["session"] = None
            summary = await loop.run_in_executor(None, s.stop)
            state["last"] = s.recording()
        try:  # the page may already be gone
            await socket.send_json({"summary": summary, "playback": playback_info(), "ended": ended})
            await socket.close()
        except Exception:
            pass


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    clients.add(socket)
    try:
        while True:  # the page never sends anything; this just waits for it to go away
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(socket)


@app.get("/api/demo")
def demo_list():
    return {"samples": demo.available(), "credit": demo.CREDIT}


def _demo(sid: str, engine: str) -> dict:
    if sid.startswith("note-"):  # a cleaned voice note plays through the same player
        if notes.ready(sid) is None or engine not in ("natural", "thorough", "live"):
            raise HTTPException(404, "No such voice note (they are kept for an hour).")
        return demo.render(sid, engine)
    s = next((s for s in demo.available() if s["id"] == sid), None)
    if s is None or engine not in s["engines"]:
        raise HTTPException(404, "No such sample.")
    return demo.render(sid, engine)


@app.post("/api/notes")
async def create_note(request: Request):
    """Body: a WAV file (the page decodes recordings and uploads to WAV in the browser)."""
    if not notes.available():
        raise HTTPException(503, "Voice notes need the teacher model, which is not downloaded.")
    data = await request.body()
    if len(data) > 40_000_000:
        raise HTTPException(413, "That file is too large.")
    try:
        return {"id": notes.submit(data)}
    except notes.NoteBusy as e:
        raise HTTPException(429, str(e)) from e
    except notes.NoteError as e:
        raise HTTPException(415, str(e)) from e


@app.get("/api/notes/{nid}")
def note_status(nid: str):
    s = notes.status(nid)
    if s is None:
        raise HTTPException(404, "No such voice note (they are kept for an hour).")
    return s


@app.get("/api/demo/{sid}")
def demo_timeline(sid: str, engine: str = "natural"):
    return _demo(sid, engine)["timeline"]


@app.get("/api/demo/{sid}/{which}.wav")
def demo_audio(sid: str, which: str, engine: str = "natural"):
    if which not in ("original", "clean"):
        raise HTTPException(404, "No such track.")
    return Response(wav_bytes(_demo(sid, engine)[which]), media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=3600"})


# Last, so the API and WebSocket routes above take precedence over the built UI files.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
