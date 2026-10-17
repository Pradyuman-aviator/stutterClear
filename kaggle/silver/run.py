"""Kaggle job: CrisperWhisper verbatim word timings ("silver labels") for real stuttered speech.

Both T4s work in parallel, one shard each. UCLASS goes first, then an evenly spaced SEP-28k /
FluencyBank subset. Results are flushed line by line to /kaggle/working/silver, so a run cut short
still keeps everything finished so far.
Pushed with:  kaggle kernels push -p kaggle/silver
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/Pradyuman-aviator/stutterClear.git"
MAX_SEP = 12000


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


tmp = "/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else "/tmp"
code, data = f"{tmp}/stutterClear", f"{tmp}/data"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"rm -rf {code} && git clone --depth 1 {REPO} {code}")
os.chdir(code)
# CrisperWhisper's CTranslate2 backend needs its own fork of ctranslate2; fall back to PyTorch if it fails.
backend = "ct2"
try:
    sh("pip uninstall -y -q ctranslate2 || true")
    sh('pip install -q "crisperwhisper[ct2]" soundfile')
    sh("python -c \"from crisperwhisper import CrisperWhisperModel; CrisperWhisperModel('large', backend='ct2')\"")
except subprocess.CalledProcessError:
    backend = "transformers"
    sh('pip install -q "crisperwhisper[transformers]" soundfile')
print("backend:", backend, flush=True)

sh(f"python scripts/download_uclass.py --root {data}")
sh(f"python scripts/download_data.py --root {data} --sep-only")

n = max(1, int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout))
procs = []
for i in range(n):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
    cmd = [sys.executable, "scripts/crisper_label.py", "--uclass", f"{data}/uclass_all",
           "--sep-dir", f"{data}/sep28k-fluencybank", "--max-sep", str(MAX_SEP), "--backend", backend,
           "--out", "/kaggle/working/silver", "--shard", str(i), "--nshard", str(n)]
    print("+", " ".join(cmd), flush=True)
    procs.append(subprocess.Popen(cmd, env=env))
codes = [p.wait() for p in procs]
print("shard exit codes:", codes, flush=True)
sh("wc -l /kaggle/working/silver/*.jsonl")
