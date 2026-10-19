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

## Free: serve the demo from the laptop (recommended until credits cover a VM)

The hackathon asks for Nebius Token Factory *or* AI Cloud: Nemotron on Token Factory already meets
it, so the laptop with the GPU can serve the hosted build with every feature, for nothing.
Tailscale Funnel gives it a stable HTTPS link (browsers only allow the microphone on HTTPS).

One-time setup:

1. Install Tailscale (https://tailscale.com/download), run `tailscale up` and log in (free plan).
2. Set the keys once, in a terminal that is then closed: `setx NEBIUS_API_KEY "..."` and, for Say it,
   `setx NVIDIA_API_KEY "nvapi-..."` (build.nvidia.com, free). Never put keys in files or chat.
3. Power settings: plugged in, sleep off, while people use the link.

Every time:

```
powershell -ExecutionPolicy Bypass -File scripts\serve_public.ps1            # public link: synthetic clips
powershell -ExecutionPolicy Bypass -File scripts\serve_public.ps1 -UclassDemos  # private judging link only
```

The first run prints a link to allow Funnel for the machine; after that it prints the public
`https://<machine>.<tailnet>.ts.net` link, the same on every start. Ctrl+C stops the app and the
link together. The script starts the app with `STUTTERCLEAR_HOSTED=1` on an exact port
(`--strict-port`), so the link never points at another program.

What the public mode changes: visitors use their own microphone, one live session at a time (up to
5 minutes), nothing they say is written to disk, their feedback stays in their browser, Nemotron and
Say it calls are capped per minute, and UCLASS clips are left out (research licence) unless
`-UclassDemos` is given for a private judging link.

Without Tailscale, a Cloudflare quick tunnel works too, but its link changes on every start:

```
$env:STUTTERCLEAR_HOSTED = "1"
python scripts/run_app.py --no-browser --port 8766 --strict-port
cloudflared tunnel --url http://localhost:8766      # prints an https://....trycloudflare.com link
```

## Free backup: a Hugging Face Space (always on, CPU)

When the laptop is off, a free Space keeps a link alive: live cleaning with the rule-based detector,
the three synthetic demo clips (their caches are in git, so no GPU model runs), FastConformer
captions on the CPU, and the Nemotron and Say it features from the Space's secrets. Voice notes,
Dictate and Clean my video are off there (they need a GPU).

1. On huggingface.co: New Space → name `stutterclear`, SDK **Docker**, template **Blank**, hardware
   **CPU basic (free)**, public.
2. Upload the two files in `deploy/hf-space/` (`Dockerfile` and `README.md`) to the Space's root.
3. Settings → Variables and secrets: add the secrets `NEBIUS_API_KEY` and `NVIDIA_API_KEY` (never
   in a file). The Dockerfile already sets `STUTTERCLEAR_HOSTED=1` and `STUTTERCLEAR_NOTES=0`.
4. The Space builds from the GitHub repo (about 10 minutes). After a `git push`, use
   Settings → Factory rebuild to update it.

The UCLASS clips never go to the Space (research licence); `STUTTERCLEAR_UCLASS_DEMOS` stays unset.

## Switches for a public server

| Variable | Effect |
|---|---|
| `STUTTERCLEAR_HOSTED=1` | Public mode (see above) |
| `STUTTERCLEAR_NOTES=0` | No voice notes: for a server without a GPU, where they would take minutes |
| `STUTTERCLEAR_UCLASS_DEMOS=1` / `=0` | Show UCLASS clips on a public server (private judging) / hide them anywhere |
| `NVIDIA_API_KEY` | Say it (NVIDIA Magpie TTS); `MAGPIE_VOICE` picks the voice |
