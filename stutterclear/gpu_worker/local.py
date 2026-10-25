"""A launcher that starts the GPU worker as a process on this computer (gpu_protocol.Launcher): the
end-to-end runs of docs/DEPLOY.md, where a server that hides the GPU from itself wakes a worker that
uses it, exactly as it would wake one on Kaggle.

The worker gets the server's environment minus what would hide the GPU from it, point it at other
checkpoints, or hand it a secret it does not need (the GPU secret, the API keys, Kaggle's), plus its
wake's token, which never goes on its command line.
"""
import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import gpu_protocol as gp

ROOT = Path(__file__).resolve().parents[2]
DROP = {"CUDA_VISIBLE_DEVICES", "STUTTERCLEAR_TEACHER", "STUTTERCLEAR_RECORDINGS_TEACHER",
        "STUTTERCLEAR_RECORDINGS_DEVICE", "STUTTERCLEAR_GPU_SECRET", "STUTTERCLEAR_GPU_LAUNCHER", "NEBIUS_API_KEY",
        "NVIDIA_API_KEY", "TAVILY_API_KEY"}


class LocalLauncher:
    name = "local"
    role = "local"
    where = "this computer"
    token_ttl_s = 86400.0

    def __init__(self, server: str, extra_args: list[str] | tuple = (), quota_left_h: float | None = None,
                 live: bool = False):
        self.server = gp.server_url(server)
        self.extra = list(extra_args)
        self.quota_left_h = quota_left_h  # fault injection: pretend Kaggle's quota has this much left
        self.live = live
        self.proc: subprocess.Popen | None = None

    @classmethod
    def from_env(cls, environ=None) -> "LocalLauncher":
        """STUTTERCLEAR_PUBLIC_URL (where the worker connects), STUTTERCLEAR_GPU_LOCAL_ARGS (extra worker
        flags, e.g. "--fake --delay-ms 1200"), STUTTERCLEAR_GPU_LOCAL_QUOTA_H (fault injection)."""
        env = os.environ if environ is None else environ
        quota = env.get("STUTTERCLEAR_GPU_LOCAL_QUOTA_H", "").strip()
        return cls(env.get("STUTTERCLEAR_PUBLIC_URL", ""), shlex.split(env.get("STUTTERCLEAR_GPU_LOCAL_ARGS", "")),
                   float(quota) if quota else None, live=env.get("STUTTERCLEAR_GPU_LIVE") == "1")

    def launch(self, wake: str, token: str, reason: str, retired: frozenset = frozenset()) -> gp.Launch:
        if self.proc is not None and self.proc.poll() is None:  # still running: it reconnects by itself
            return gp.Launch("running", ref=f"pid {self.proc.pid}", adopted=True)
        env = {k: v for k, v in os.environ.items() if k not in DROP and not k.startswith("KAGGLE_")}
        env["STUTTERCLEAR_GPU_TOKEN"] = token
        args = [sys.executable, "-m", "stutterclear.gpu_worker", "--server", self.server, "--where", "local",
                "--token-env", "STUTTERCLEAR_GPU_TOKEN", f"--first={gp.first_engine(reason)}", *self.extra]
        if not self.live and "--no-live" not in self.extra:
            args.append("--no-live")  # live through the worker is off on this server: keep the GPU for notes
        # Its own process group: Ctrl+C on the server (a restart) does not stop it, as a restart of the
        # app does not stop a Kaggle worker; it reconnects and is adopted, or leaves once idle.
        apart = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"start_new_session": True})
        self.proc = subprocess.Popen(args, env=env, cwd=ROOT, **apart)
        return gp.Launch("running", ref=f"pid {self.proc.pid}")

    def poll(self) -> gp.Launch:
        if self.proc is None:
            return gp.Launch("done")
        ref, rc = f"pid {self.proc.pid}", self.proc.poll()
        if rc is None:
            return gp.Launch("running", ref=ref)
        if rc == 0:
            return gp.Launch("done", ref=ref)
        return gp.Launch("failed", f"The worker exited with code {rc}.", ref=ref, retry_s=60)

    def quota(self) -> gp.Quota | None:
        if self.quota_left_h is None:
            return None
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        saturday = midnight + timedelta(days=(5 - now.weekday()) % 7 or 7)  # the next Saturday 00:00 UTC
        return gp.Quota(used_h=30.0 - self.quota_left_h, total_h=30.0, refresh_at=saturday.timestamp())

    def stop(self):
        """End the worker process (tests, and a server that stops)."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
