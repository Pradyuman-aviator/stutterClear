"""Kaggle job: teacher v2 on whatever GPUs Kaggle gives (one P100, or two T4s with DDP).

Trained from the pretrained WavLM-Large, not from teacher v1: v1 used a different split and saw
speakers of the official SEP-28k-E test set, which would leak into v2's test score. Adds
generator v2 (rapid repeats, phrase repeats, fillers, clusters), UCLASS recordings and SEP-28k
clips with CrisperWhisper silver frame labels; SEP-28k is split by the official SEP-28k-E
partition. Checkpoints are chosen on real speech only.
Input (kernel_sources): stutterclear-silver-labels (words_*.jsonl).
Pushed with:  kaggle kernels push -p kaggle/teacher_v2
"""
import glob
import os
import subprocess

REPO = "https://github.com/Pradyuman-aviator/stutterClear.git"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


silver = sorted(glob.glob("/kaggle/input/**/words_*.jsonl", recursive=True))
assert silver, "silver labels not found under /kaggle/input"
print("silver:", silver, flush=True)

tmp = "/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else "/tmp"
code, data = f"{tmp}/stutterClear", f"{tmp}/data"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"rm -rf {code} && git clone --depth 1 {REPO} {code}")
os.chdir(code)
sh("pip install -q soundfile")
sh(f"python scripts/download_data.py --root {data}")
sh(f"python scripts/download_uclass.py --root {data}")
gpus = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout)
sh(
    f"torchrun --nproc_per_node={gpus} scripts/train_teacher.py"
    f" --libri-train '{data}/librispeech-alignments/data/train_clean_100-*.parquet'"
    f" --libri-val '{data}/librispeech-alignments/data/dev_clean-*.parquet'"
    f" --sep-dir {data}/sep28k-fluencybank --uclass-dir {data}/uclass_all"
    f" --silver {' '.join(silver)}"
    " --backbone microsoft/wavlm-large --grad-ckpt"
    " --steps 10000 --batch 6 --sep-batch 6 --uclass-batch 6"
    " --freeze-steps 500 --warmup 500"
    " --workers 3 --eval-every 1000 --val-n 300 --max-hours 11.0 --out /kaggle/working/teacher_v2"
)
