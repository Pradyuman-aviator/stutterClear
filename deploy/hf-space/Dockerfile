# StutterClear on a free Hugging Face Space (CPU basic: 2 vCPUs, 16 GB RAM): the always-on backup to
# the laptop link. Live cleaning with the rule-based detector, the synthetic demo clips (their caches
# ship in git, so no GPU model runs here), NVIDIA FastConformer captions on the CPU, and the Nemotron
# and Magpie features from the Space's secrets. Voice notes and video are off: they need a GPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
# Behind the Space's proxy: each visitor is told apart by the X-Forwarded-For entry it adds, so the
# per-visitor limits of a public server never lump everyone together.
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    STUTTERCLEAR_HOSTED=1 STUTTERCLEAR_NOTES=0 STUTTERCLEAR_TRUST_PROXY=1
WORKDIR /home/user/app

# The code from GitHub (rebuild the Space to update); CPU PyTorch, and everything else except what
# needs a GPU (transformers, CrisperWhisper) or audio devices (sounddevice).
ARG REPO=https://github.com/Pradyuman-aviator/stutterClear.git
RUN git clone --depth 1 "$REPO" .
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E "^(torch|transformers|crisperwhisper|sounddevice|pytest)" requirements.txt > /tmp/req.txt \
    && pip install --user -r /tmp/req.txt
RUN python scripts/download_asr.py

EXPOSE 7860
CMD ["python", "scripts/run_app.py", "--no-browser", "--host", "0.0.0.0", "--port", "7860", "--strict-port"]
