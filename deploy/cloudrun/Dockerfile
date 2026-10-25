# StutterClear's main app on Google Cloud Run (CPU, asia-south1): live cleaning, captions, the sample
# clips, Nemotron and Magpie, and the GPU features through a worker the app wakes on Kaggle
# (stutterclear/app/gpu.py). Cloud Build builds it from the allowlisted upload (.gcloudignore) with
# --build-arg COMMIT=<the full git sha>: see scripts/deploy_cloudrun.ps1 and docs/DEPLOY.md.
# No checkpoints and nothing from runs/ except the measured results: the GPU models stay on the worker.
FROM python:3.12.15-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Exactly the files in requirements.lock, each checked against its sha256: requirements.txt without
# what needs a GPU (transformers, CrisperWhisper), audio devices (sounddevice), training data (pyarrow)
# or tests, and CPU PyTorch (notes import the teacher's runtime, which imports torch; the GPU work runs
# on the worker). Nothing published later reaches this image, which holds the keys (scripts/lock_hashes.py
# makes the lock; pip check proves it is whole).
COPY deploy/cloudrun/requirements.lock deploy/cloudrun/asr.sha256 ./
RUN pip install --require-hashes --only-binary=:all: --no-deps --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.lock \
    && pip check && pip freeze

COPY scripts/download_asr.py ./scripts/
COPY stutterclear ./stutterclear
COPY data/demo_clips ./data/demo_clips
COPY runs/eval ./runs/eval
# The three real recordings the demo plays (stutterclear/app/demo_samples.json, source uclass) with their
# model caches: never in git (UCLASS is free for research and teaching, with credit), uploaded by the
# deploy and checked against the list this commit carries, so a commit's image always plays the same audio.
COPY deploy/cloudrun/uclass.sha256 ./
COPY data/external ./data/external
RUN tr -d '\r' < uclass.sha256 | sha256sum -c -
# The captions model (NVIDIA FastConformer for sherpa-onnx, 106 MB), fetched at build time and
# checked against the laptop's tested copy (tr: a Windows checkout gives the list CRLF line endings);
# compiled bytecode makes a cold start a little faster.
RUN python scripts/download_asr.py && tr -d '\r' < asr.sha256 | sha256sum -c - \
    && rm -rf models/asr/*/test_wavs && python -m compileall -q stutterclear

ARG COMMIT
RUN echo "$COMMIT" | grep -Eq '^[0-9a-f]{40}$' || (echo "build with --build-arg COMMIT=<the full git sha>" && exit 1)
LABEL org.opencontainers.image.source=https://github.com/Pradyuman-aviator/stutterClear \
      org.opencontainers.image.revision=$COMMIT
ENV STUTTERCLEAR_HOSTED=1 STUTTERCLEAR_TRUST_PROXY=1 STUTTERCLEAR_COMMIT=$COMMIT PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 7860
# Cloud Run sets PORT (the deploy passes --port 7860). One process: every session, note and GPU job
# lives in its memory. --ws-max-size: the GPU worker's largest message is 4 MiB. Cloud Run allows 10 s
# after SIGTERM; uvicorn closes the sockets (the GPU worker reconnects to the next instance).
CMD ["sh", "-c", "exec python -m uvicorn stutterclear.app.server:app --host 0.0.0.0 --port \"${PORT:-7860}\" --ws-max-size 4194304 --timeout-graceful-shutdown 8 --log-level warning"]
