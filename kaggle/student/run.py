"""Kaggle job: (1) teacher thresholds on validation speakers + one honest report on test speakers,
(2) distil the teacher into the causal student, (3) per-lookahead student report on test speakers.

Input: the output of the stutterclear-teacher kernel (kernel_sources), mounted under /kaggle/input.
Pushed with:  kaggle kernels push -p kaggle/student
Outputs (/kaggle/working): teacher_eval/, student/
"""
import glob
import os
import shutil
import subprocess

REPO = "https://github.com/Pradyuman-aviator/stutterClear.git"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def scratch():
    for d in ("/kaggle/tmp", "/tmp"):
        if os.path.isdir(d) and shutil.disk_usage(d).free > 15e9:
            return d
    raise SystemExit("no scratch folder with 15 GB free")


ckpts = glob.glob("/kaggle/input/**/best.pt", recursive=True)
assert ckpts, "teacher checkpoint not found under /kaggle/input"
teacher = ckpts[0]
tmp = scratch()
code, data = f"{tmp}/stutterClear", f"{tmp}/data"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"rm -rf {code} && git clone --depth 1 {REPO} {code}")
os.chdir(code)
sh("pip install -q soundfile")
sh(f"python scripts/download_data.py --root {data}")
sh(f"python scripts/eval_teacher.py --ckpt {teacher} --sep-dir {data}/sep28k-fluencybank --out /kaggle/working/teacher_eval")
sh(
    "python scripts/train_student.py"
    f" --teacher-ckpt {teacher}"
    f" --libri-train '{data}/librispeech-alignments/data/train_clean_100-*.parquet'"
    f" --libri-val '{data}/librispeech-alignments/data/dev_clean-*.parquet'"
    f" --sep-dir {data}/sep28k-fluencybank"
    " --steps 15000 --batch 16 --sep-batch 16 --workers 4 --eval-every 1000 --val-n 400"
    " --max-hours 9.5 --out /kaggle/working/student"
)
