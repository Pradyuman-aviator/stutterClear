"""Start the StutterClear app (Fluency Mirror) and open it in the browser.

    python scripts/run_app.py
"""
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stutterclear.app.server import app  # noqa: E402

PORT = 8765

if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
