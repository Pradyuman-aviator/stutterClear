"""Start the StutterClear app (Fluency Mirror) and open it in the browser.

    python scripts/run_app.py [--port 8765] [--no-browser]

If the port is taken by something else, the next free one is used.
"""
import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.app.server import app  # noqa: E402

PORT = 8765


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:  # something already answers here
                continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"no free port in {start}-{start + 49}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    print(f"StutterClear on {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
