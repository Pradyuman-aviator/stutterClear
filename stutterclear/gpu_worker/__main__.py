"""python -m stutterclear.gpu_worker: run a GPU worker for an app server (worker.py says what it does).

    # the team's laptop: the server's GPU secret, typed at the prompt (or set in this terminal only)
    python -m stutterclear.gpu_worker --server https://stutterclear-<n>.asia-south1.run.app --where laptop

On Kaggle the server's bootstrap (kernel_template/run.py) starts it with that wake's token in
STUTTERCLEAR_GPU_TOKEN. Exit codes: 0 done, 1 an unexpected error, 2 usage or settings, 3 refused by
the server, 4 a different protocol or code version, 5 the models could not load.

--help and argument errors never import torch.
"""
import argparse
import asyncio
import getpass
import re
import sys
import time

from .. import gpu_protocol as gp

DEVICE_RE = re.compile(r"cpu|cuda(:\d)?")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m stutterclear.gpu_worker",
                                 description="Do StutterClear's GPU work for an app server, connecting out to it.")
    ap.add_argument("--server", required=True, help="the app server, e.g. https://stutterclear-123.asia-south1.run.app")
    ap.add_argument("--token-env", default="STUTTERCLEAR_GPU_TOKEN", metavar="NAME",
                    help="the variable that holds this worker's token (Kaggle); without one, the laptop signs its "
                         "own from STUTTERCLEAR_GPU_SECRET or the secret typed at the prompt")
    ap.add_argument("--where", choices=gp.ROLES,
                    help="where this worker runs (default: what its token says, else laptop)")
    ap.add_argument("--idle-exit", type=float, metavar="S", help="leave after this long without work (the server's "
                                                                  "setting replaces it once connected)")
    ap.add_argument("--deadline", type=float, metavar="EPOCH", help="this machine's time limit (Kaggle)")
    ap.add_argument("--first", choices=("live", "notes"), default="notes", help="on one GPU, which engine loads first")
    ap.add_argument("--no-live", action="store_true", help="do not load the live teacher (the laptop also leaves it "
                                                           "unloaded when the server does not use a worker for live)")
    ap.add_argument("--no-notes", action="store_true", help="do not load the recordings engines")
    ap.add_argument("--live-device", metavar="DEV", help="e.g. cuda:0 (default: from the GPUs found)")
    ap.add_argument("--notes-device", metavar="DEV", help="e.g. cuda:1, or cpu (default: from the GPUs found)")
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)  # stand-in engines, no torch
    ap.add_argument("--delay-ms", type=int, default=0, help=argparse.SUPPRESS)  # wait before each reply
    ap.add_argument("--crash-after", type=int, default=0, help=argparse.SUPPRESS)  # die at the Nth recording
    return ap


def _fail(message: str) -> int:
    print(message, file=sys.stderr, flush=True)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        server = gp.server_url(args.server)
    except ValueError as e:
        return _fail(str(e))
    if args.no_live and args.no_notes:
        return _fail("--no-live and --no-notes together leave nothing to do.")
    for flag, dev in (("--live-device", args.live_device), ("--notes-device", args.notes_device)):
        if dev is not None and not DEVICE_RE.fullmatch(dev):
            return _fail(f"{flag} is cpu, cuda or cuda:N.")
    if args.idle_exit is not None and not args.idle_exit > 0:
        return _fail("--idle-exit is a number of seconds.")
    if args.deadline is not None and args.deadline <= time.time() + gp.DRAIN_S:
        print("This machine's time limit is too close to start any work.", flush=True)
        return 0
    role, key_id = args.where, None
    token = gp.read_secret(args.token_env)
    if token:
        claims = gp.peek_token(token)
        if claims is None:  # never sent anywhere: it might be something else pasted by mistake
            return _fail("That is not a StutterClear worker token.")
        if role and role != claims.role:
            return _fail(f"That token is for a {claims.role} worker, not --where {role}.")
        role = claims.role
        if claims.expires < time.time() - 300:
            return _fail("That worker token has expired.")
        tokens = lambda: token  # noqa: E731
    else:
        role = role or "laptop"
        secret = gp.read_secret("STUTTERCLEAR_GPU_SECRET") if role != "kaggle" else None
        if not secret and role != "kaggle" and sys.stdin.isatty():
            secret = getpass.getpass("StutterClear GPU secret: ").strip() or None
        if not secret:
            return _fail("No worker token: set STUTTERCLEAR_GPU_TOKEN (Kaggle) or STUTTERCLEAR_GPU_SECRET "
                         "(the team's laptop).")

        def tokens() -> str:  # a fresh wake each time: one per connection, and one per standby poll
            return gp.make_token(secret, role, gp.new_wake(f"{role}-"), int(time.time()) + 86400, server)

        try:
            tokens()
        except ValueError as e:
            return _fail(str(e))
        key_id = gp.key_id(secret)
        print(f"key id {key_id} (the server prints its own at startup: they must match)", flush=True)

    from .worker import Options, Worker, code_commit  # websockets; the engines import torch only when loading
    opts = Options(server=server, role=role, token=tokens, commit=code_commit(), idle_exit_s=args.idle_exit,
                   deadline=args.deadline, first=args.first, notes=not args.no_notes, live=not args.no_live,
                   fake=args.fake, live_device=args.live_device, notes_device=args.notes_device,
                   delay_ms=max(0, args.delay_ms), crash_after=max(0, args.crash_after), key_id=key_id)
    try:
        return asyncio.run(Worker(opts).run())
    except KeyboardInterrupt:  # Ctrl+C: the worker said goodbye if it was connected
        print("stopped", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
