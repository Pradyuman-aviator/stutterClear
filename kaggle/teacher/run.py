"""Kaggle job: full teacher training (WavLM-Large) on LibriSpeech-100 synthetic stutters + SEP-28k/FluencyBank.

Pushed with:  kaggle kernels push -p kaggle/teacher
Outputs (/kaggle/working/teacher): best.pt, metrics.jsonl, config.json
"""
import os
import shutil
import subprocess

REPO = "https://github.com/Pradyuman-aviator/stutterClear.git"
STEPS = 10000


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def scratch():
    # Keep the ~10 GB of data out of /kaggle/working, which is saved as output (20 GB cap).
    for d in ("/kaggle/tmp", "/tmp"):
        if os.path.isdir(d) and shutil.disk_usage(d).free > 15e9:
            return d
    raise SystemExit("no scratch folder with 15 GB free")


tmp = scratch()
code, data = f"{tmp}/stutterClear", f"{tmp}/data"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"rm -rf {code} && git clone --depth 1 {REPO} {code}")
os.chdir(code)
sh("pip install -q soundfile")
sh(f"HF_HUB_ENABLE_HF_TRANSFER=0 python scripts/download_data.py --root {data}")
sh(
    "python scripts/train_teacher.py"
    f" --libri-train '{data}/librispeech-alignments/data/train_clean_100-*.parquet'"
    f" --libri-val '{data}/librispeech-alignments/data/dev_clean-*.parquet'"
    f" --sep-dir {data}/sep28k-fluencybank"
    " --backbone microsoft/wavlm-large --grad-ckpt"
    f" --steps {STEPS} --batch 6 --sep-batch 6 --workers 4 --eval-every 500 --val-n 400"
    " --max-hours 10.5 --out /kaggle/working/teacher"
)
