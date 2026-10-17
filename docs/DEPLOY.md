# Deploy the hosted demo on Nebius AI Cloud

The hosted demo runs the same app on a GPU VM: visitors use their browser's microphone, voice notes
are cleaned on the server's GPU, and Nemotron runs on Nebius Token Factory. Nothing a visitor says
is written to disk.

## 1. Create the VM

In the Nebius AI Cloud console create a VM with one NVIDIA GPU (the app needs about 5 GB of GPU
memory; any current data-centre GPU is plenty), the "Ubuntu with CUDA / NVIDIA drivers" image, a
public IP, and ports 22, 80 and 443 open.

On the VM, Docker and the NVIDIA Container Toolkit must be present (Nebius GPU images usually
include them; otherwise follow docs.docker.com and the NVIDIA Container Toolkit guide). Check:

```
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 2. Code and assets

```
git clone https://github.com/Pradyuman-aviator/stutterClear.git && cd stutterClear
```

Models and demo data are not in git. On the laptop that has them:

```
python scripts/pack_assets.py                 # writes stutterclear_assets.tar
scp stutterclear_assets.tar <user>@<vm-ip>:~/stutterClear/
```

On the VM: `tar xf stutterclear_assets.tar`.

## 3. Keys and domain

```
cp .env.example .env
```

Fill in `.env`: `DOMAIN` (your VM's IP with dashes plus `.sslip.io` gives a free name with HTTPS,
e.g. `203-0-113-7.sslip.io`), `NEBIUS_API_KEY`, and optionally `TAVILY_API_KEY`. Keep `.env` on the
server only.

## 4. Start

```
docker compose up -d --build
docker compose logs -f app        # first start downloads WavLM and CrisperWhisper weights once
```

Open `https://<DOMAIN>`. Caddy gets the HTTPS certificate on the first visit.

Update after a `git push`: `git pull && docker compose up -d --build`.

## What runs where

| Part | Where |
|---|---|
| Live detection (WavLM teacher on sliding windows), voice notes, captions | the VM's GPU / CPU |
| Word rescue, meaning guard, context agent, insights | NVIDIA Nemotron on Nebius Token Factory |
| Microphone | the visitor's browser (streamed over a WebSocket) |

One live session runs at a time; others can use the samples and voice notes meanwhile.

## Plan B: serve the demo from a laptop

The hackathon asks for Nebius Token Factory *or* AI Cloud: Nemotron on Token Factory already meets
it, so if the VM is not ready, the laptop with the GPU can serve the same hosted build. Browsers
only allow the microphone on HTTPS (or localhost), so put it behind a tunnel, for example a free
Cloudflare quick tunnel:

```
set STUTTERCLEAR_HOSTED=1                          # PowerShell: $env:STUTTERCLEAR_HOSTED = "1"
python scripts/run_app.py --no-browser --port 8766
cloudflared tunnel --url http://localhost:8766      # prints an https://....trycloudflare.com link
```

The link works only while the laptop is on and the app is running, and it changes on every start,
so this is for a judging session or a recorded demo, not for weeks of judging.
