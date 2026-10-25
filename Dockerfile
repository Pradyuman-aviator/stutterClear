# StutterClear server for a GPU VM (Nebius AI Cloud). Models and data are mounted, not baked in:
# see docs/DEPLOY.md and docker-compose.yml.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Behind Caddy: each visitor is told apart by the X-Forwarded-For entry Caddy adds
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STUTTERCLEAR_HOSTED=1 STUTTERCLEAR_TRUST_PROXY=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY stutterclear ./stutterclear
COPY scripts ./scripts

EXPOSE 8765
CMD ["python", "scripts/run_app.py", "--no-browser", "--host", "0.0.0.0", "--port", "8765"]
