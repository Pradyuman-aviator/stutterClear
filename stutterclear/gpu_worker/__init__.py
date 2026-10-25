"""StutterClear's GPU worker: the program that does the app server's GPU work somewhere else (a free
Kaggle notebook, or the team's laptop) and connects out to the server for it, so nothing has to be
opened where it runs. `python -m stutterclear.gpu_worker --help` shows how to start one.

    worker.py    the connection: hello, jobs, live windows, heartbeats, idle sleep, standing by
    engines.py   the GPU work itself (the teachers and CrisperWhisper), and stand-ins for tests
    kaggle.py    the launcher the server uses to start a worker on Kaggle, one wake at a time
    local.py     the launcher that starts one on this computer (end-to-end tests)

The wire between the server and a worker is stutterclear/gpu_protocol.py. Importing this package
imports nothing else: the server imports the launchers without torch or websockets.
"""
