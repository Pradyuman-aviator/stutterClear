"""StutterClear app server: the Fluency Mirror UI on http://127.0.0.1:8765.

    python scripts/run_app.py

The UI is a Vite + React build in static/ (source in web/: `npm run build` there refreshes it).
"""
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..live import DEFAULT_STUDENT, LiveSession, devices

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="StutterClear")
state = {"session": None}


class StartRequest(BaseModel):
    mode: str = "gentle"
    in_device: int | None = None
    out_device: int | None = None
    detector: str = "student"
    save: bool = False


@app.get("/api/devices")
def list_devices():
    return devices()


@app.get("/api/info")
def info():
    return {"student": DEFAULT_STUDENT.exists()}  # the AI detector needs its downloaded checkpoint


@app.post("/api/start")
def start(req: StartRequest):
    if state["session"] is not None:
        state["session"].stop()
    s = LiveSession(req.mode, req.in_device, req.out_device, req.detector, save=req.save)
    s.start()
    state["session"] = s
    return {"ok": True, "detector": s.detector_name, "mode": req.mode}


@app.post("/api/stop")
def stop():
    s, state["session"] = state["session"], None
    return {"ok": True, "summary": s.stop() if s else None}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            s = state["session"]
            await socket.send_json(s.poll() if s else {"idle": True})
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


# Last, so the API and WebSocket routes above take precedence over the built UI files.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
