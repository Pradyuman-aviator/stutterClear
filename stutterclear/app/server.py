"""StutterClear app server: the Fluency Mirror UI on http://127.0.0.1:8765.

    python scripts/run_app.py

The UI is a Vite + React build in static/ (source in web/: `npm run build` there refreshes it).
"""
import asyncio
import os
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

from .. import feedback, journal, nemotron
from ..audio import wav_bytes
from ..config import SAMPLE_RATE as SR
from ..live import DEFAULT_STUDENT, LiveSession, detectors, devices, recommended, student_eval
from . import demo, notes, video

STATIC = Path(__file__).parent / "static"
# A public deployment: visitors bring their own mic through the browser; the server never opens its
# own audio devices and never writes recordings to disk.
HOSTED = os.environ.get("STUTTERCLEAR_HOSTED") == "1"
UPDATE_S = 0.03  # ~30 updates a second: the orb reacts within a frame or two
# One live session at a time, so on a public deployment a tab left open must not hold it for ever.
LIVE_LIMIT_S = 300.0 if HOSTED else None
MIC_IDLE_S = 15.0  # no audio for this long: the page froze or the laptop went to sleep
BUSY_HOSTED = "Someone else is live right now (turns last up to 5 minutes). Try again shortly, or play a real recording."
# Visitors' Nemotron calls spend the owner's Nebius credits: a public demo caps them per minute.
NEMOTRON_PER_MIN = 60 if HOSTED else None
_nemotron_calls: deque = deque()
TTS_PER_MIN = 30 if HOSTED else None  # Say it spends the owner's NVIDIA credits the same way
_tts_calls: deque = deque()
# "last" holds the finished session's audio, in memory only, for the play buttons. Starting a new
# session drops it.
# On a public server a session belongs to the page that started it ("owner", a random id the page
# sends): only that page gets its live updates (waveforms, captions) and its listen-back.
state = {"session": None, "last": None, "owner": None, "last_owner": None}
notes.live_running = lambda: state["session"] is not None  # a voice note leaves the live teacher alone
clients: dict[WebSocket, str] = {}  # open pages -> their page id
IDLE = {"idle": True}


def sees_session(page: str) -> bool:
    """Whether a page gets the live session's updates: every page on this computer; on a public
    server only the page that started it."""
    return not (HOSTED and state["session"] and (not page or page != state["owner"]))


async def broadcast():
    """Poll the session once per tick and send the same message to every open page.

    poll() hands out what is new since its last call, so polling per connection would split the
    audio between tabs (or an OBS browser source and the main window).
    """
    while True:
        s = state["session"]
        msg = s.poll() if s else IDLE
        for socket, page in list(clients.items()):
            try:
                await socket.send_json(msg if sees_session(page) else IDLE)
            except Exception:  # closed mid-send; its handler cleans up
                clients.pop(socket, None)
        await asyncio.sleep(UPDATE_S)


def warm_demo():
    """Load the big model and render every demo sample once in the background, so the first live
    session and the first click on a sample start at once. Nemotron's model is checked first:
    the meaning guard runs while the samples render."""
    try:
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
    finally:
        notes.warm.set()  # voice notes may take the GPU now


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(broadcast())
    notes.warm.clear()  # voice notes wait: the warm-up uses the live teacher on the GPU
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
    from .. import captions, context_agent, tts
    return {"student": DEFAULT_STUDENT.exists(), "student_eval": student_eval(), "notes": notes.available(),
            "captions": captions.available(), "rescue": nemotron.available(), "nemotron": nemotron.available(),
            "search": context_agent.search_available(), "model": nemotron.MODEL, "hosted": HOSTED,
            "tts": tts.available(), "tts_voice": tts.VOICE, "tts_label": tts.LABEL, "interview": nemotron.available(),
            "feedback": "browser" if HOSTED else "server", "uclass_demos": demo.uclass_enabled(),
            "video": video.available(),
            "detectors": detectors(), "recommended": recommended()}


class ContextRequest(BaseModel):
    situation: str  # where the person is about to speak
    current: str = ""  # what the context card says already


def _budget(calls: deque, per_min: int | None, busy: str):
    """Count one visitor call against a per-minute cap, or turn it away for a moment."""
    if not per_min:
        return
    now = time.monotonic()
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= per_min:
        raise HTTPException(429, busy)
    calls.append(now)


def nemotron_budget():
    _budget(_nemotron_calls, NEMOTRON_PER_MIN, "Nemotron is busy with other visitors; try again in a moment.")


@app.post("/api/context")
def prepare_context(req: ContextRequest):
    """Nemotron (with Tavily search when a key is set) writes a context card for the situation."""
    from .. import context_agent
    if not nemotron.available():
        raise HTTPException(503, "The context agent needs NEBIUS_API_KEY set where the app runs.")
    nemotron_budget()
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
    nemotron_budget()
    try:
        return {"points": insights.summarize(req.kinds, req.duration_s, req.removed_s, req.words, req.transcript)}
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


class GuardRequest(BaseModel):
    text: str


@app.post("/api/guard")
def guard_try(req: GuardRequest):
    """The meaning guard on typed text: which repeats Nemotron would cut and which it keeps."""
    from .. import guard
    if not nemotron.available():
        raise HTTPException(503, "The meaning guard needs NEBIUS_API_KEY set where the app runs.")
    if not req.text.split():
        raise HTTPException(400, "Type a sentence first.")
    nemotron_budget()
    try:
        return guard.check_text(req.text)
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
    nemotron_budget()
    try:
        return {"suggestions": rescue.suggest(req.said, req.context, req.onset), "model": nemotron.MODEL}
    except Exception as e:
        raise HTTPException(502, f"Nemotron did not answer: {e}") from e


class TTSRequest(BaseModel):
    text: str  # the word or short phrase the user clicked
    voice: str = ""  # a Magpie voice name; empty for the default


@app.post("/api/tts")
def say_it(req: TTSRequest):
    """Say it: NVIDIA Magpie TTS speaks what the user clicked, only ever on that click. The reply is
    a WAV; its X-Voice-Label header is the label the page shows (the voice is synthetic)."""
    from .. import tts
    if not tts.available():
        raise HTTPException(503, "Say it needs NVIDIA_API_KEY set where the app runs.")
    if not req.text.split():
        raise HTTPException(400, "Pick a word first.")
    if len(req.text) > tts.MAX_CHARS or (req.voice and not re.fullmatch(r"[\w.\-]{1,80}", req.voice)):
        raise HTTPException(400, f"Say it takes a word or a short phrase (up to {tts.MAX_CHARS} characters).")
    _budget(_tts_calls, TTS_PER_MIN, "Say it is busy with other visitors; try again in a moment.")
    try:
        y = tts.synthesize(req.text, req.voice or tts.VOICE)
    except tts.TTSError as e:
        raise HTTPException(502, str(e)) from e
    return Response(wav_bytes(y), media_type="audio/wav", headers={"Cache-Control": "no-store", "X-Voice-Label": tts.LABEL})


class InterviewRequest(BaseModel):
    role: str
    company: str = ""
    posting: str = ""  # the job post, optional
    card: str = ""  # the context card from /api/interview/start
    history: list[dict] = []  # [{question, answer}], answers as the captions wrote them


def _interview(fn):
    if not nemotron.available():
        raise HTTPException(503, "The interview partner needs NEBIUS_API_KEY set where the app runs.")
    nemotron_budget()
    try:
        return fn()
    except nemotron.NemotronError as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/interview/start")
def interview_start(req: InterviewRequest):
    """Practice: research the role into a context card, then the first question."""
    from .. import interview
    if not req.role.strip():
        raise HTTPException(400, "Say which role the interview is for.")
    return _interview(lambda: interview.start(req.role, req.company, req.posting))


@app.post("/api/interview/next")
def interview_next(req: InterviewRequest):
    """The next question, following up on the answers so far."""
    from .. import interview
    return _interview(lambda: {"question": interview.next_question(req.role, req.company, req.card, req.history)})


@app.post("/api/interview/summary")
def interview_summary(req: InterviewRequest):
    """Three practice points about the answers (content only, never how they were spoken)."""
    from .. import interview
    if not req.history:
        raise HTTPException(400, "Answer a question first.")
    return _interview(lambda: {"points": interview.summary(req.role, req.company, req.history)})


class FeedbackRequest(BaseModel):
    source: str = "note"  # note | live
    id: str = ""  # the voice note (or session) the cut belongs to
    cut: dict | None = None  # {start, end, kind, words}
    verdict: str | None = None  # accept (the cut was right) | reject (it should not have been cut)
    missed_at_s: float | None = None  # a stutter that was left in, tapped at this time


class FeedbackLog(BaseModel):
    entries: list[dict] = []


@app.post("/api/feedback")
def give_feedback(req: FeedbackRequest):
    """One review decision. The server works out which kind a missed stutter was (from the voice
    note's teacher probabilities); a personal computer logs it, a public server hands it back for
    the page to keep in the browser."""
    entry = req.model_dump()
    if req.missed_at_s is not None and req.id.startswith("note-"):
        note = notes.ready(req.id)
        if note is not None:
            entry["kind"] = feedback.kind_at(note["probs"], req.missed_at_s)
    entry = feedback.clean(entry)
    if entry is None:
        raise HTTPException(400, "Feedback is a reviewed cut (accept or reject) or a missed stutter's time.")
    if HOSTED or not feedback.stored_here():
        return {"entry": entry, "stored": False}
    return {"entry": entry, "stored": True, **feedback.record(entry)}


@app.get("/api/feedback")
def feedback_summary():
    """Multipliers per kind, agreement over the recent decisions, whether auto mode can be offered."""
    entries = [] if HOSTED else feedback.read()
    return {"stored": not HOSTED, **feedback.summary(entries)}


@app.post("/api/feedback/summary")
def feedback_summary_of(req: FeedbackLog):
    """The same summary for a log the page keeps itself (public servers store none)."""
    return feedback.summary([e for e in (feedback.clean(x) for x in req.entries[-2000:]) if e])


@app.delete("/api/feedback")
def forget_feedback():
    """Forget every review decision on this computer; StutterClear goes back to its defaults."""
    if not HOSTED:
        feedback.forget()
    return {"ok": True}


class SayRequest(BaseModel):
    text: str
    voice: str = ""
    page: str = ""  # on a public server, only the page that owns the session


@app.post("/api/say")
def say_in_call(req: SayRequest):
    """Say it in the call: NVIDIA Magpie TTS speaks the clicked word into the live session's output
    (what the listener hears, e.g. through a virtual cable), only ever on that click."""
    from .. import tts
    s = state["session"]
    if s is None:
        raise HTTPException(409, "Start a live session first.")
    if HOSTED and req.page != state["owner"]:
        raise HTTPException(403, "Only the page that started the session can speak into it.")
    if s.out_device is None:
        raise HTTPException(409, "Nothing is sending your voice to a call here, so Say it plays in this browser instead.")
    if not tts.available():
        raise HTTPException(503, "Say it needs NVIDIA_API_KEY set where the app runs.")
    if not req.text.split() or len(req.text) > tts.MAX_CHARS:
        raise HTTPException(400, f"Say it takes a word or a short phrase (up to {tts.MAX_CHARS} characters).")
    _budget(_tts_calls, TTS_PER_MIN, "Say it is busy with other visitors; try again in a moment.")
    try:
        y = tts.synthesize(req.text, req.voice or tts.VOICE)
    except tts.TTSError as e:
        raise HTTPException(502, str(e)) from e
    s.say(y)
    return {"ok": True, "seconds": round(len(y) / SR, 2), "label": tts.LABEL}


@app.get("/api/journal")
def journal_summary():
    """Progress over time: stutters per minute by day, this week against the last, hardest words."""
    return journal.summary([] if HOSTED else journal.read())


@app.post("/api/journal/start")
def journal_start():
    """Start keeping a journal on this computer (counts and words, never audio)."""
    if HOSTED:
        raise HTTPException(403, "The journal is kept only on your own computer; a public server keeps none.")
    journal.start()
    return journal.summary(journal.read())


@app.delete("/api/journal")
def journal_forget():
    """Forget the journal and stop keeping one."""
    if not HOSTED:
        journal.forget()
    return {"ok": True}


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
    notes.release_gpu()  # the live teacher gets the GPU back from a finished voice note
    s = LiveSession(req.mode, in_device, out_device, req.detector, save=req.save, adjust=feedback.current())
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
    if HOSTED:  # browser sessions end over their own connection; nobody else may end them
        raise HTTPException(403, "On the hosted demo, a session stops from the page that started it.")
    s, state["session"] = state["session"], None
    if s is None:
        return {"ok": True, "summary": None}
    summary = s.stop()
    state["last"] = s.recording()
    journal.record(journal.from_session(summary, s.raw.n / SR))  # only if the person started a journal
    return {"ok": True, "summary": summary, "playback": playback_info()}


def _last(page: str):
    """The finished session's audio, if this page may hear it."""
    return None if HOSTED and (not page or page != state["last_owner"]) else state["last"]


@app.get("/api/playback")
def playback_info(page: str = ""):
    last = _last(page)
    if not last or not len(last["original"]):
        return {"available": False}
    return {"available": True, **{f"{k}_s": round(len(v) / SR, 2) for k, v in last.items()}}


@app.get("/api/playback/{which}.wav")
def playback(which: str, page: str = ""):
    last = _last(page)
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
        await socket.close(code=4409, reason=BUSY_HOSTED if HOSTED else "A live session is already running here.")
        return
    cfg = await socket.receive_json()
    page = str(cfg.get("page", ""))
    # The person's feedback: this computer's log, or on a public server what the page kept for them
    adjust = feedback.from_page(cfg.get("adjust")) if HOSTED else feedback.current()
    notes.release_gpu()
    loop = asyncio.get_running_loop()
    s = await loop.run_in_executor(None, lambda: LiveSession(  # may load a model: off the event loop
        cfg.get("mode", "gentle"), detector=cfg.get("detector", "heuristic"), source="browser", adjust=adjust))
    s.start()
    state["session"], state["last"], state["owner"] = s, None, page
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
            state["last"], state["last_owner"] = s.recording(), page
            if not HOSTED:
                journal.record(journal.from_session(summary, s.raw.n / SR))
        try:  # the page may already be gone
            await socket.send_json({"summary": summary, "playback": playback_info(page), "ended": ended})
            await socket.close()
        except Exception:
            pass


@app.websocket("/ws")
async def ws(socket: WebSocket, page: str = ""):
    await socket.accept()
    clients[socket] = page
    try:
        while True:  # the page never sends anything; this just waits for it to go away
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.pop(socket, None)


@app.get("/api/demo")
def demo_list():
    samples = demo.available()
    return {"samples": samples, "credit": demo.credit(samples)}


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


@app.post("/api/video")
async def create_video(request: Request, engine: str = "natural", name: str = "video.mp4"):
    """Body: a video file. Its stutters are cut from picture and sound together (the voice-note
    engines on its soundtrack, ffmpeg for the cuts)."""
    if not video.available():
        raise HTTPException(503, "Cleaning a video needs the offline model and ffmpeg (pip install imageio-ffmpeg).")
    data = await request.body()
    if len(data) > video.MAX_BYTES:
        raise HTTPException(413, "That video is too large (300 MB at most).")
    try:
        return {"id": video.submit(data, name, engine)}
    except notes.NoteBusy as e:
        raise HTTPException(429, str(e)) from e
    except video.VideoError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/video/{jid}")
def video_status(jid: str):
    s = video.status(jid)
    if s is None:
        raise HTTPException(404, "No such video (they are kept for an hour).")
    return s


@app.get("/api/video/{jid}/clean.mp4")
def video_result(jid: str, download: bool = False):
    data = video.result(jid)
    if data is None:
        raise HTTPException(404, "That video is not ready (or was forgotten after an hour).")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = 'attachment; filename="stutterclear-video.mp4"'
    return Response(data, media_type="video/mp4", headers=headers)


@app.get("/api/notes/{nid}")
def note_status(nid: str):
    s = notes.status(nid)
    if s is None:
        raise HTTPException(404, "No such voice note (they are kept for an hour).")
    return s


@app.get("/api/demo/{sid}")
def demo_timeline(sid: str, engine: str = "natural"):
    return _demo(sid, engine)["timeline"]


@app.get("/api/demo/{sid}/cuts.{fmt}")
def demo_cuts(sid: str, fmt: str, engine: str = "natural"):
    """Timestamps of everything removed, as a table (csv) or subtitles (srt), on the original
    recording's clock, for editing or checking a recording elsewhere."""
    if fmt not in ("csv", "srt"):
        raise HTTPException(404, "Timestamps come as cuts.csv or cuts.srt.")
    cuts = _demo(sid, engine)["timeline"]["cuts"]
    body = demo.cuts_csv(cuts) if fmt == "csv" else demo.cuts_srt(cuts)
    return Response(body, media_type="text/csv" if fmt == "csv" else "application/x-subrip",
                    headers={"Content-Disposition": f'attachment; filename="{sid}-{engine}-cuts.{fmt}"'})


@app.get("/api/demo/{sid}/{which}.wav")
def demo_audio(sid: str, which: str, engine: str = "natural"):
    if which not in ("original", "clean"):
        raise HTTPException(404, "No such track.")
    return Response(wav_bytes(_demo(sid, engine)[which]), media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=3600"})


# Last, so the API and WebSocket routes above take precedence over the built UI files.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
