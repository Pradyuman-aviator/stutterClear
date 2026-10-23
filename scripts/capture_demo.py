"""Record a real demo of the running app for the film, and cut it to ~30 s.

    python scripts/run_app.py --no-browser --port 8766 --strict-port     # in another terminal
    python scripts/capture_demo.py [--mode gentle|strong]                 # writes video/public/generated/demo/

A headless Edge opens the app and its screen is recorded (DevTools screencast). Alex's real recording
(video/public/generated/before.wav, from export_video_data.py) is the microphone, through Chromium's
fake audio capture, so the live pipeline really runs on this GPU: Calls, Best (GPU), "This browser's
mic". Then the Recordings tab cleans the same speaker's clip with Thorough, and the meaning guard
answers live from Nemotron. The build cuts four moments from it (the live session, its summary,
Thorough, the guard's answer) into demo.mp4 (30 fps) and demo.wav (what the app played: the live
session's cleaned voice, then the Thorough result), with demo.json describing the cut.
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video" / "public" / "generated" / "demo"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9334
APP = "http://127.0.0.1:8766"
CONTEXT = "Demo day pitch for my app: it teaches travellers to read Japanese train signs."
SR = 48000
MIC_LEAD_S = 4.0  # silence before the clip in the fake mic file: a live session takes ~3 s to start streaming
GUM_DELAY_S = 0.25  # from the click to the mic opening (getUserMedia), roughly


class Browser:
    """A page over the DevTools protocol; screencast frames are saved as they arrive."""

    def __init__(self, ws_url, frames_dir):
        import websocket
        self.ws = websocket.create_connection(ws_url, timeout=120, suppress_origin=True)
        self.frames_dir = frames_dir
        self.n = 0
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.replies = {}
        self.frames = []
        self.recording = False
        threading.Thread(target=self._read, daemon=True).start()

    def _send(self, msg):
        with self.lock:
            self.ws.send(json.dumps(msg))

    def _read(self):
        while True:
            try:
                msg = json.loads(self.ws.recv())
            except Exception:
                return
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                if self.recording:
                    (self.frames_dir / f"{len(self.frames):05d}.jpg").write_bytes(base64.b64decode(p["data"]))
                    self.frames.append(p["metadata"]["timestamp"])
                self._send({"id": 10_000_000 + len(self.frames), "method": "Page.screencastFrameAck",
                            "params": {"sessionId": p["sessionId"]}})
            elif "id" in msg:
                with self.cond:
                    self.replies[msg["id"]] = msg
                    self.cond.notify_all()

    def call(self, method, **params):
        self.n += 1
        i = self.n
        self._send({"id": i, "method": method, "params": params})
        with self.cond:
            while i not in self.replies:
                self.cond.wait(60)
            msg = self.replies.pop(i)
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    def js(self, expr):
        return self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True).get("result", {}).get("value")

    def has(self, text):
        return bool(self.js(f"document.body.innerText.includes({json.dumps(text)})"))

    def click(self, text, contains=False, sel="button,[role=tab],[role=radio],label,span,div"):
        """Click the smallest visible element whose text is (or contains) `text`."""
        p = self.js(
            "(() => { const t = %s; const els = [...document.querySelectorAll(%s)].filter(e => (%s) && e.getBoundingClientRect().width > 0);"
            " els.sort((a, b) => a.getBoundingClientRect().width * a.getBoundingClientRect().height - b.getBoundingClientRect().width * b.getBoundingClientRect().height);"
            " const e = els[0]; if (!e) return null; e.scrollIntoView({block: 'center', behavior: 'instant'}); const r = e.getBoundingClientRect();"
            " return [r.x + r.width / 2, r.y + r.height / 2]; })()"
            % (json.dumps(text), json.dumps(sel), "e.textContent.includes(t)" if contains else "e.textContent.trim() === t"))
        if not p:
            raise RuntimeError(f"nothing to click: {text!r}")
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=kind, x=p[0], y=p[1], button="left", clickCount=1)

    def wait_for(self, text, timeout):
        t = time.time()
        while time.time() - t < timeout:
            if self.has(text):
                return True
            time.sleep(0.25)
        return False


def mic_file(work: Path) -> Path:
    """Alex's clip at 48 kHz with a short silent lead and tail, as the fake microphone."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sr, x = wavfile.read(ROOT / "video" / "public" / "generated" / "before.wav")
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    y = np.concatenate([np.zeros(int(MIC_LEAD_S * SR)), resample_poly(x, SR // sr, 1), np.zeros(2 * SR)])
    path = (work / "mic.wav").resolve()  # absolute: the browser does not resolve relative paths
    wavfile.write(path, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    return path


def capture(work: Path, mode: str) -> dict:
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mic = mic_file(work)
    edge = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={tempfile.mkdtemp(prefix='edge-demo-')}", "--window-size=1920,1080",
                             "--hide-scrollbars", "--no-first-run", "--autoplay-policy=no-user-gesture-required",
                             "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                             f"--use-file-for-fake-audio-capture={mic}%noloop", "about:blank"])
    marks = {}
    try:
        for _ in range(60):
            try:
                pages = [t for t in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=2).read())
                         if t.get("type") == "page"]
                if pages:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        b = Browser(pages[0]["webSocketDebuggerUrl"], frames_dir)
        b.call("Page.enable")
        b.call("Runtime.enable")
        b.call("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
        b.call("Page.navigate", url=APP)
        time.sleep(3)
        b.js("localStorage.setItem('sc.in.name', '__browser__'); localStorage.setItem('sc.mode', %s);"
             " localStorage.setItem('sc.det', 'teacher'); localStorage.setItem('sc.rescue', 'on');"
             " localStorage.setItem('ctx', %s); true" % (json.dumps(mode), json.dumps(CONTEXT)))
        b.call("Page.reload")
        time.sleep(6)
        # Warm-up: the first live start loads the GPU model, which takes many seconds; start and
        # stop once so the recorded session starts at once (each mic opening replays the file).
        b.click("Start listening", contains=True, sel="button")
        b.wait_for("Waiting for your voice", 90) or b.wait_for("Listening", 5)
        time.sleep(3)
        b.click("Stop", contains=True, sel="button")
        time.sleep(5)
        b.call("Page.reload")
        time.sleep(6)

        b.call("Page.startScreencast", format="jpeg", quality=85, maxWidth=1920, maxHeight=1080, everyNthFrame=1)
        b.recording = True
        t0 = time.time()

        def mark(name):
            marks[name] = time.time()
            print(f"{time.time() - t0:6.1f}s  {name}", flush=True)

        time.sleep(2.5)
        mark("start")
        b.click("Start listening", contains=True, sel="button")
        time.sleep(MIC_LEAD_S + 19.45 + 2.0 + 0.5)  # the clip plays as the mic
        mark("stop")
        b.click("Stop", contains=True, sel="button")
        b.wait_for("Session ended", 10)
        mark("summary")
        marks["summary_text"] = b.js("[...document.querySelectorAll('div,span,p')].map(e => e.innerText).find(t => /stutters? removed/.test(t) && t.length < 120) || ''")
        time.sleep(3.0)
        b.click("Recordings", sel="[role=tab],button")
        time.sleep(1.5)
        b.click("man, 20", contains=True, sel="button")
        time.sleep(6.0)
        mark("thorough")
        b.click("Thorough")
        time.sleep(11.0)
        mark("check")
        b.click("Check", sel="button")
        b.wait_for("cut as stutter", 60)
        mark("answered")
        time.sleep(3.0)
        b.recording = False
        b.call("Page.stopScreencast")
    finally:
        edge.terminate()
    for name, url in (("live_original", f"{APP}/api/playback/original.wav"), ("live_clean", f"{APP}/api/playback/clean.wav"),
                      ("thorough_clean", f"{APP}/api/demo/monologue-20/clean.wav?engine=thorough")):
        (work / f"{name}.wav").write_bytes(urllib.request.urlopen(url, timeout=120).read())
    first = b.frames[0]
    info = {"frames": [t - first for t in b.frames], "marks": {k: (v - first if isinstance(v, float) else v) for k, v in marks.items()},
            "mode": mode, "mic_lead_s": MIC_LEAD_S}
    (work / "capture.json").write_text(json.dumps(info))
    return info


def load(path: Path, sr: int = SR) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    r, y = wavfile.read(path)
    y = y.astype(np.float32) / (32768.0 if y.dtype == np.int16 else 1.0)
    if y.ndim > 1:
        y = y.mean(1)
    return resample_poly(y, sr, r) if r != sr else y


def level(y: np.ndarray, speech_db: float = -21.0) -> np.ndarray:
    """Speech at the film's level (as export_video_data.py sets narration and the clip): the app
    plays recordings at their own, much quieter, level."""
    n = int(0.05 * SR)
    r = np.sqrt((y[: len(y) // n * n].reshape(-1, n) ** 2).mean(1))
    loud = r[r > max(1e-6, np.abs(y).max()) * 10 ** (-30 / 20)]
    if not len(loud):
        return y
    g = min(10 ** ((speech_db - 20 * np.log10(np.sqrt((loud ** 2).mean()))) / 20), 0.89 / np.abs(y).max())
    return (y * g).astype(np.float32)


def build(work: Path, out: Path):
    from scipy.io import wavfile
    from scipy.signal import correlate
    info = json.loads((work / "capture.json").read_text())
    ts, m = info["frames"], info["marks"]
    out.mkdir(parents=True, exist_ok=True)

    # The screen at a constant 30 fps (screencast frames arrive only when the page repaints).
    lines = []
    for i, t in enumerate(ts):
        nxt = ts[i + 1] if i + 1 < len(ts) else t + 1.0
        lines += [f"file '{(work / 'frames' / f'{i:05d}.jpg').as_posix()}'", f"duration {max(0.001, nxt - t):.4f}"]
    lines.append(f"file '{(work / 'frames' / f'{len(ts) - 1:05d}.jpg').as_posix()}'")
    (work / "frames.txt").write_text("\n".join(lines))
    subprocess.run(["npx.cmd", "remotion", "ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(work / "frames.txt"), "-r", "30", "-fps_mode", "cfr", "-pix_fmt", "yuv420p",  # this ffmpeg has no fps filter
                    "-c:v", "libx264", "-crf", "18",
                    "-preset", "medium", "-movflags", "+faststart", str(out / "demo.mp4")], cwd=ROOT / "video", check=True)

    # Where Alex's clip sits in the live session: the session's recording of the mic against the file.
    mic = load(work / "mic.wav", 16000)
    orig = load(work / "live_original.wav", 16000)
    probe = orig[: 16000 * 8]
    x = int(np.argmax(correlate(mic, probe, mode="valid")))  # session sample 0 = mic file sample x
    session0 = m["start"] + GUM_DELAY_S + x / 16000  # screen time of the session's first sample

    cut = [  # source seconds on the screen recording, and what each piece shows
        {"from": m["start"] - 0.6, "to": m["start"] + 1.6, "label": "Calls · Start listening · Gentle"},
        {"from": m["start"] + MIC_LEAD_S, "to": m["start"] + MIC_LEAD_S + 12.5, "label": "live · Alex's recording as the mic"},
        {"from": m["summary"] - 0.2, "to": m["summary"] + 2.6, "label": "the live session's summary"},
        {"from": m["thorough"] + 0.2, "to": m["thorough"] + 8.2, "label": "Recordings · Thorough"},
        {"from": m["check"] - 0.6, "to": m["check"] + 1.0, "label": "meaning guard · asking Nemotron"},
        {"from": m["answered"] - 0.2, "to": m["answered"] + 2.8, "label": "meaning guard · Nemotron's answer"},
    ]
    live, thorough = load(work / "live_clean.wav"), load(work / "thorough_clean.wav")
    sound = []
    for k, c in enumerate(cut):
        n = int(round((c["to"] - c["from"]) * SR))
        seg = np.zeros(n, np.float32)
        if k == 1:  # what the listener heard, live
            src, lo = live, int(round((c["from"] - session0) * SR))
        elif k == 3:  # the Thorough result, which starts playing as Thorough is picked
            src, lo = thorough, int(round((c["from"] - (m["thorough"] + 0.5)) * SR))
        else:
            src = None
        if src is not None:
            idx = np.arange(n) + lo
            ok = (idx >= 0) & (idx < len(src))
            seg[ok] = src[idx[ok]]
            seg = level(seg)
            f = int(0.015 * SR)
            seg[:f] *= np.linspace(0, 1, f)
            seg[-f:] *= np.linspace(1, 0, f)
        sound.append(seg)
    y = np.concatenate(sound)
    wavfile.write(out / "demo.wav", SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    meta = {"video": "generated/demo/demo.mp4", "audio": "generated/demo/demo.wav", "seconds": round(len(y) / SR, 2),
            "cut": [dict(c, **{k: round(c[k], 3) for k in ("from", "to")}) for c in cut],
            "summary": m.get("summary_text", ""), "mode": info["mode"]}
    (out / "demo.json").write_text(json.dumps(meta, indent=1))
    print(f"demo: {meta['seconds']} s, session starts at {session0:.2f} s of the screen (mic offset {x / 16000:.2f} s); {meta['summary']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gentle", choices=["gentle", "strong"])
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "stutterclear-demo"))
    ap.add_argument("--build-only", action="store_true", help="cut an earlier capture again")
    args = ap.parse_args()
    work = Path(args.work)
    if not args.build_only:
        capture(work, args.mode)
    build(work, OUT)


if __name__ == "__main__":
    sys.exit(main())
