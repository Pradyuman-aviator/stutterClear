"""StutterClear app server: the Fluency Mirror UI on http://127.0.0.1:8765.

    python scripts/run_app.py

The UI is a Vite + React build in static/ (source in web/: `npm run build` there refreshes it).
"""
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..audio import wav_bytes
from ..config import SAMPLE_RATE as SR
from ..live import DEFAULT_STUDENT, LiveSession, devices, student_eval
from . import demo

STATIC = Path(__file__).parent / "static"
UPDATE_S = 0.03  # ~30 updates a second: the orb reacts within a frame or two
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(broadcast())
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
    return {"student": DEFAULT_STUDENT.exists(), "student_eval": student_eval()}


@app.post("/api/start")
def start(req: StartRequest):
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
    s = next((s for s in demo.available() if s["id"] == sid), None)
    if s is None or engine not in s["engines"]:
        raise HTTPException(404, "No such sample.")
    return demo.render(sid, engine)


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
