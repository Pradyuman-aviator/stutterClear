"""Kaggle job: (1) teacher v2 thresholds on validation speakers + one report on the official
SEP-28k-E test set, (2) distil teacher v2 into the causal student, now also on unlabelled UCLASS
recordings of training speakers, (3) per-lookahead student report on test speakers.

Input (kernel_sources): stutterclear-teacher-v2 (teacher_v2/best.pt).
Pushed with:  kaggle kernels push -p kaggle/student_v2
Outputs (/kaggle/working): teacher_v2_eval/, student_v2/
"""
import glob
import os
import shutil
import subprocess

REPO = "https://github.com/Pradyuman-aviator/stutterClear.git"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


ckpts = glob.glob("/kaggle/input/**/teacher_v2/best.pt", recursive=True) or glob.glob("/kaggle/input/**/best.pt", recursive=True)
assert ckpts, "teacher v2 checkpoint not found under /kaggle/input"
teacher = ckpts[0]
tmp = "/kaggle/tmp" if os.path.isdir("/kaggle/tmp") and shutil.disk_usage("/kaggle/tmp").free > 15e9 else "/tmp"
code, data = f"{tmp}/stutterClear", f"{tmp}/data"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"rm -rf {code} && git clone --depth 1 {REPO} {code}")
os.chdir(code)
sh("pip install -q soundfile")
sh(f"python scripts/download_data.py --root {data}")
sh(f"python scripts/download_uclass.py --root {data}")
sh(f"python scripts/eval_teacher.py --ckpt {teacher} --sep-dir {data}/sep28k-fluencybank --out /kaggle/working/teacher_v2_eval")
sh(
    "python scripts/train_student.py"
    f" --teacher-ckpt {teacher}"
    f" --libri-train '{data}/librispeech-alignments/data/train_clean_100-*.parquet'"
    f" --libri-val '{data}/librispeech-alignments/data/dev_clean-*.parquet'"
    f" --sep-dir {data}/sep28k-fluencybank --uclass-dir {data}/uclass_all"
    " --steps 20000 --batch 16 --sep-batch 16 --uclass-batch 16 --workers 4 --eval-every 1000 --val-n 400"
    " --max-hours 10 --out /kaggle/working/student_v2"
)
