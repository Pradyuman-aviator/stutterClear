"""Start the StutterClear app (Fluency Mirror) and open it in the browser.

    python scripts/run_app.py [--port 8765] [--no-browser] [--host 0.0.0.0] [--strict-port]

If the port is taken by something else, the next free one is used (--strict-port: stop instead,
for a tunnel or proxy that points at that exact port). In a container, listen on
0.0.0.0 and set STUTTERCLEAR_HOSTED=1 (see docs/DEPLOY.md).
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
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve beyond this computer (a container)")
    ap.add_argument("--strict-port", action="store_true", help="exit if --port is taken instead of moving on")
    args = ap.parse_args()
    port = free_port(args.port)
    if args.strict_port and port != args.port:
        sys.exit(f"port {args.port} is taken (another StutterClear?); stop it or pick another --port")
    url = f"http://127.0.0.1:{port}"
    print(f"StutterClear on {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # The largest WebSocket message anyone sends here is a GPU worker's 4 MiB (uvicorn would buffer 16)
    uvicorn.run(app, host=args.host, port=port, log_level="warning", ws_max_size=4 * 1024 * 1024)
