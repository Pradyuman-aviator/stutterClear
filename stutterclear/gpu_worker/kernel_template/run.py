"""StutterClear GPU worker on Kaggle. The app server pushes this script as a new version of a PRIVATE
kernel whenever a visitor needs the GPU, with that wake's settings written into CONFIG. It fetches the
public code at the server's exact commit, installs the pinned packages the Kaggle image lacks, finds the
private weights at their exact mount paths and runs the worker, which connects out to the server and
leaves when it is no longer used.

Nothing is written to /kaggle/working (Kaggle keeps it as the notebook's output): code, packages and
model caches go to /tmp, and the folder is emptied before the run ends. CONFIG and the token are never
printed and never put on a command line. Pushed by stutterclear/gpu_worker/kaggle.py, never by hand.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time

TMP, WORK = "/tmp/stutterclear", "/kaggle/working"
RESTARTS = 2  # a worker that crashes (a CUDA error, out of memory) gets a fresh process this often
DONE = (0, 2, 3, 4, 5)  # done, usage, refused, version, models: a fresh process would not help


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_config() -> dict:
    try:
        return json.loads(base64.b64decode("__CONFIG__"))
    except Exception:  # never let a traceback print the encoded settings
        log("this kernel's settings are unreadable: it must be pushed by the StutterClear server")
        sys.exit(1)


CONFIG = load_config()


def run(args: list, **kw) -> int:
    """One command, without a shell. Its arguments never hold the token."""
    log("+", " ".join(args))
    try:
        return subprocess.run(args, **kw).returncode
    except OSError as e:  # the command is missing (nvidia-smi on a machine without a GPU)
        log(" ", type(e).__name__)
        return 127


def fetch_code() -> str:
    """The public repository at exactly the server's commit. There is no fallback: a worker never runs
    other code than the server's."""
    sha, repo = CONFIG["commit"], CONFIG["repo"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+\.git", repo):
        log("refusing: the commit or the repository is not pinned")
        sys.exit(1)
    code = f"{TMP}/code"
    for attempt in range(3):
        shutil.rmtree(code, ignore_errors=True)
        if (run(["git", "init", "-q", code]) == 0 and run(["git", "-C", code, "remote", "add", "origin", repo]) == 0
                and run(["git", "-C", code, "fetch", "-q", "--depth", "1", "origin", sha]) == 0
                and run(["git", "-C", code, "checkout", "-q", "FETCH_HEAD"]) == 0):
            return code
        if attempt < 2:
            time.sleep(10)
    log("could not fetch commit", sha)
    sys.exit(1)


def find(paths: list) -> str | None:
    """The first of these exact paths that is a file. The weights are never searched for."""
    return next((p for p in paths if os.path.isfile(p)), None)


def clean_output():
    """Leave nothing in /kaggle/working: Kaggle keeps it, next to the run's log."""
    for name in os.listdir(WORK) if os.path.isdir(WORK) else []:
        p = os.path.join(WORK, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass


def main():
    t0 = time.time()
    try:
        os.makedirs(TMP, exist_ok=True)
        os.chdir(TMP)
        # Kaggle's own session variables stay out; every cache goes to /tmp
        env = {k: v for k, v in os.environ.items() if not k.startswith("KAGGLE_")}
        env.update(HF_HOME=f"{TMP}/hf", TORCH_HOME=f"{TMP}/torch", XDG_CACHE_HOME=f"{TMP}/cache",
                   PIP_CACHE_DIR=f"{TMP}/pip", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                   PIP_DISABLE_PIP_VERSION_CHECK="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                   TRANSFORMERS_VERBOSITY="warning")
        run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])  # information only
        code = fetch_code()
        log(f"code at {CONFIG['commit'][:12]} after {time.time() - t0:.0f} s")
        # exactly the files hashed in the lock: nothing published later under the same versions
        if run([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", "--only-binary=:all:",
                "--no-deps", "--require-hashes", "-r", f"{code}/stutterclear/gpu_worker/requirements.lock"], env=env) != 0:
            log("could not install the pinned packages")
            sys.exit(1)
        log(f"packages installed after {time.time() - t0:.0f} s")
        live, rec = find(CONFIG["weights"]["live"]), find(CONFIG["weights"]["recordings"])
        log("weights:", "live ok" if live else "live missing", "/", "recordings ok" if rec else "recordings missing")
        # the worker reports missing files itself ("could not find its model files"); the token goes in
        # its environment only
        worker_env = dict(env, PYTHONPATH=code, STUTTERCLEAR_TEACHER=live or "missing",
                          STUTTERCLEAR_RECORDINGS_TEACHER=rec or "missing", STUTTERCLEAR_COMMIT=CONFIG["commit"],
                          STUTTERCLEAR_GPU_TOKEN=CONFIG["token"])
        deadline = int(t0 + int(CONFIG["timeout_s"]) - 300)  # five minutes for a clean goodbye
        args = [sys.executable, "-m", "stutterclear.gpu_worker", "--server", CONFIG["server"], "--where", "kaggle",
                "--token-env", "STUTTERCLEAR_GPU_TOKEN", f"--idle-exit={float(CONFIG['idle_exit_s']):g}",
                f"--deadline={deadline}", "--first=" + ("live" if CONFIG["first"] == "live" else "notes")]
        if not CONFIG["live"]:
            args.append("--no-live")
        rc = 1
        for _ in range(1 + RESTARTS):
            rc = subprocess.run(args, env=worker_env, cwd=code).returncode
            log("worker exited", rc)
            if rc in DONE or time.time() > deadline:
                break
        sys.exit(0 if rc == 0 else 1)
    finally:
        clean_output()


if __name__ == "__main__":
    main()
