# Deploy StutterClear

For the submission the hosted demo runs in three layers, at no cost to the team (a teammate's Google
Cloud free-trial credit, Kaggle's free GPU hours, GitHub Pages). The link is frozen on Oct 30 (22:30
IST) and must keep working, free and self-serve, until judging ends on Dec 16 (01:30 IST); judging
runs Dec 1-15.

| Layer | What it does | Where | Section |
|---|---|---|---|
| Front door | The frozen link, https://pradyuman-aviator.github.io/stutterClear/. Checks the servers and sends visitors to the first that answers; otherwise shows the demo video, the measured results and a status, never an error | GitHub Pages (`frontdoor/`) | [Front door](#front-door-github-pages) |
| Main app | The app in hosted CPU mode: live cleaning (rule-based detector), captions, the sample clips, Nemotron and Magpie | Google Cloud Run, asia-south1 (Mumbai) | [Google Cloud Run](#google-cloud-run-the-main-app) |
| GPU on demand | Voice notes, Dictate and Clean my video (and Best (GPU) live, phase P2): a worker the app starts only when a visitor asks for it; it connects out to the app and leaves when idle | A private Kaggle notebook (2× T4), or the team's laptop | [Kaggle](#gpu-on-demand-the-kaggle-worker), [laptop](#the-team-laptop-as-the-gpu-backup) |
| Backup | The same CPU app | The Azure VM | [Azure](#always-on-free-for-students-a-small-azure-server-cpu) |

Nothing a visitor says is written to disk on any of them: recordings are cleaned in memory, on the app
server and, for the GPU features, on the worker, and forgotten within an hour.

## Google Cloud Run: the main app

The image (`deploy/cloudrun/Dockerfile`) is the app with CPU PyTorch and no checkpoints: the GPU models
live on the worker. Cloud Build builds it from an allowlisted upload (`.gcloudignore`: the code, the
sample clips and the two measured-results files; never `runs/`, `models/`, recordings, `.env` or keys),
fetches the captions model and checks it against `deploy/cloudrun/asr.sha256`, and labels the image
with the commit it was built from. Its packages are exactly the files in `deploy/cloudrun/requirements.lock`,
each checked against its sha256 (`pip install --require-hashes`), so nothing published later reaches the
process that holds the keys; after changing `requirements.txt`, make the lock again with the command at
its top (`tests/test_locks.py` fails until then). The service then runs that image by digest:

| Setting | Value | Why |
|---|---|---|
| Region | asia-south1 (Mumbai) | close to the team and the judges |
| Size | 2 vCPU, 4 GiB, startup CPU boost, second-generation runtime | peak memory measured at 0.57-0.74 GB; live captions take about a tenth of a core per session |
| Instances | at most 1; at least 0, or 1 during judging | every session, note and GPU job lives in one process's memory |
| Concurrency | 80 requests at once (each open page holds one: its WebSocket) | request bodies are capped per route and at most 4 large uploads arrive at once, so 80 fit in 4 GiB |
| Request timeout | 3600 s, session affinity | live sessions and the GPU worker are WebSockets |
| Account | `stutterclear-run@<project>.iam.gserviceaccount.com` | may read only its own secrets |
| Settings | `deploy/cloudrun/env.yaml` (no secrets) | one reviewed file; the deploy sets exactly these |
| Secrets | Secret Manager, each pinned to a version | `NEBIUS_API_KEY`, `NVIDIA_API_KEY`, `KAGGLE_*`, `STUTTERCLEAR_GPU_SECRET` |

### Once: tools and access

1. Install the Google Cloud CLI (https://cloud.google.com/sdk/docs/install) and run `gcloud auth login`
   with the Google account that has Editor (or Owner) on the project. The project is
   `project-37b67880-f654-4dbd-8a8` (a free trial: never press "Upgrade", so nothing can be charged).
2. The project Owner sets budget alerts (Billing → Budgets & alerts) at 10, 25, 50, 75 and 90 % of
   $300, sent to the team, and checks Billing → Overview: the trial must have started on or after
   about 2026-09-17 to last until Dec 16.

### Deploy

```powershell
# See every check and change first, then do it (phase P0: the CPU app without the GPU worker)
powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1 -Project project-37b67880-f654-4dbd-8a8 -NoGpu -DryRun
powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1 -Project project-37b67880-f654-4dbd-8a8 -NoGpu
```

`scripts/deploy_cloudrun.ps1` works with Editor rights and is safe to run again. It:

1. checks that what will be built is exactly what is committed (and, for the Kaggle worker, pushed:
   the worker fetches that exact commit from GitHub), and that the upload holds only committed files,
   never recordings, checkpoints or keys;
2. enables the APIs and creates what is missing: the `stutterclear` image repository and the
   service's own account;
3. asks for each secret Secret Manager does not have yet at a hidden prompt (`NEBIUS_API_KEY` and
   `NVIDIA_API_KEY` may be skipped with Enter: their features stay hidden). Never type a secret into
   a `$env:NAME = "..."` line: PowerShell's history file (`ConsoleHost_history.txt`) keeps every typed
   line in plain text. Values go to Secret Manager's API as base64 and are never printed or piped into
   `gcloud` (Windows PowerShell would add a byte-order mark and a line ending);
4. checks that the service's account may read each secret, builds the image once per commit (about 10
   minutes), deploys it by digest, sends traffic to it, stops keeping request logs below status 500
   (cost, and no visitor addresses in the logs) and checks `/api/health`;
5. prints the URL, `https://stutterclear-<PROJECT_NUMBER>.asia-south1.run.app`. Put it first in
   `frontdoor/backends.json` (see [Front door](#front-door-github-pages)).

After the first deploy, check once where Google's front end puts the visitor's address (every
per-visitor limit relies on it). On a fresh instance (right after a deploy), open `<url>/api/info` in a
browser, then run `curl.exe -H "X-Forwarded-For: 203.0.113.9" <url>/api/info`, and read the log lines
`proxy: a request with N X-Forwarded-For entries` (one per distinct count; a count, never an address):

- the browser's request shows **1** and the curl request **2**: Google appends the visitor's address
  after whatever the client sent, so the last entry is the visitor. `STUTTERCLEAR_TRUST_PROXY: "1"`
  (in `env.yaml`) is right;
- the browser's request already shows **2**: something appends its own address after the visitor's (a
  load balancer in front of the service). Set `STUTTERCLEAR_TRUST_PROXY` to `"2"` (the visitor is the
  second entry from the end); with `"1"` every visitor would share one address, and so one set of
  limits (one recording at a time, ten sockets) for the whole site;
- the curl request shows **1**: the header is replaced rather than appended to, which is fine too.

Phase P1, with the GPU worker: push the commit, then run the same command without `-NoGpu`; the first
time it asks for the GPU secret ([GPU on demand](#gpu-on-demand-the-kaggle-worker)). Other switches: `-LaptopOnly` (the GPU only through the team's laptop, nothing started on Kaggle),
`-Live` (Best (GPU) live, phase P2), `-SkipBuild` (never build; fail if this commit has no image),
`-WarmDuringJudging` / `-MinInstances N`, `-UpdateSecrets NAME,...` (new versions from this
terminal's variables, for example after rotating a key).

### The Owner's one-time commands

An Editor cannot change who may do what, so where that is needed the script stops (or, at the end,
warns) and prints the exact commands with the real names, for the project Owner to run once:

```
gcloud secrets add-iam-policy-binding <SECRET> --project=<project> --member=serviceAccount:stutterclear-run@<project>.iam.gserviceaccount.com --role=roles/secretmanager.secretAccessor
gcloud run services add-iam-policy-binding stutterclear --project=<project> --region=asia-south1 --member=allUsers --role=roles/run.invoker
gcloud projects add-iam-policy-binding <project> --member=serviceAccount:<the build account the error names> --role=roles/cloudbuild.builds.builder
```

The first is needed once per secret the service reads (it replaces the usual shortcut of letting the
default Compute Engine account, an Editor, read every secret), the second makes the demo public, and
the third only if the build fails for want of a permission. When the person running the script is the
Owner, it grants the first two itself.

### Update, roll back, switch off

- **Update**: commit, push, run the script again with the same switches. Each commit is built once;
  `deploy/cloudrun/env.yaml` is part of the commit, so a settings change is a new commit too.
- **During judging**, do not rebuild. Change a setting in place (a new revision, same image):
  `gcloud run services update stutterclear --region asia-south1 --update-env-vars KEY=VALUE`, and put
  the same change in `env.yaml` afterwards so the next deploy keeps it.
- **Roll back**: `gcloud run revisions list --service stutterclear --region asia-south1`, then
  `gcloud run services update-traffic stutterclear --region asia-south1 --to-revisions <revision>=100`.
  Old revisions are kept; the next deploy sends traffic to the newest again.
- **Kill switch** (abuse, or a bill growing faster than expected):
  `gcloud run services update stutterclear --region asia-south1 --ingress internal` refuses every
  public request at once; the front door then sends visitors to the backup or shows its information
  page. `--ingress all` undoes it.
- **Logs**: `gcloud run services logs read stutterclear --region asia-south1 --limit 100`. The app logs
  one line per GPU event (`gpu: ...`), never a token, an address or what anyone said.

### Cost (the trial's $300 credit; nothing out of pocket)

| Item | Estimate Oct 30 - Dec 16 |
|---|---|
| Cloud Run while in use (about 4 hours a day) | ≈ $13 |
| One instance always on, Dec 1-15 | ≈ $19 |
| Network out (budgeted by the app: 1 GB an hour, 8 GB a day; pages are gzipped and cached) | under $1 normally |
| Cloud Build, Secret Manager, request logs | free tier |
| Artifact Registry (about 1.5 GB of images) | ≈ $0.15 a month |
| **Total** | **≈ $35** |

Pages pause their connection when hidden or idle, the server closes idle page connections after 30
minutes and the GPU worker leaves when unused, so an idle demo costs nothing.

## GPU on demand: the Kaggle worker

When a visitor records a voice note, uses Dictate or picks a video, the page asks the app to wake the
GPU. The app (`stutterclear/app/gpu.py`) first gives the team's laptop 40 s to answer if it is
standing by; otherwise it pushes a new version of the private Kaggle kernel
`pradyumansharmaaa/stutterclear-gpu-worker` (2× T4, internet on, at most 3 hours). The kernel fetches
this exact commit from GitHub, installs `stutterclear/gpu_worker/requirements.lock`, finds the weights
in the outputs of the two private training kernels, and runs `python -m stutterclear.gpu_worker`,
which connects out to `wss://<app>/ws/gpu` with a token signed for that wake only. Waking takes about
2-4 minutes; the page shows each step and keeps the CPU features usable meanwhile. The worker leaves
after 4 minutes if no work came, or 10 minutes after its last job. Nothing is written to
`/kaggle/working` (Kaggle keeps it), and the kernel holds no secret other than that wake's token.

### Once: the GPU secret

```powershell
python -c "import secrets; print(secrets.token_urlsafe(36))"   # keep it in the team's password manager
powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1 -Project project-37b67880-f654-4dbd-8a8 -SetGpuSecret
```

This is a full deploy with the GPU worker (the first deploy without `-NoGpu` asks for the secret even
without `-SetGpuSecret`). Paste it at the hidden prompt (never into a typed `$env:` line, which
PowerShell's history keeps). The script stores it and prints its **key id**, a short fingerprint that is safe to
share: the service's log shows `gpu: worker secret key id <id>` when it starts, and the laptop worker
prints the same, so a mismatch shows at once. To rotate it, run `-SetGpuSecret` again with a new
value: the deploy pins the new version, and every old token and the laptop's old copy stop working.

### Once: Kaggle access for the app

The app starts the worker through the Kaggle API, as the account that owns the weight kernels
(`pradyumansharmaaa`). Prefer a named API token, which can be revoked on its own: kaggle.com → Settings
→ API → create a token that expires after Dec 16, then run the script and paste it at its hidden
prompt (`-UpdateSecrets KAGGLE_API_TOKEN` replaces it the same way). While no token is stored, the
`KAGGLE_USERNAME` and `KAGGLE_KEY` secrets already in Secret Manager are used.

### The weights: never re-run the training kernels

The worker reads the checkpoints from the outputs of `pradyumansharmaaa/stutterclear-teacher` (the live
teacher) and `pradyumansharmaaa/stutterclear-teacher-v2` (the recordings teacher), mounted read-only.
**Never re-run those two kernels before Dec 16**: a run replaces its output. Keep offline copies of both
checkpoints (`runs/` on the laptop and one more disk). The checkpoints are research-licensed: they
must never become public, so the launcher refuses to run if the worker kernel or a weight kernel is
public. Fallback, only if mounting the outputs fails: `python scripts/upload_weights_kaggle.py --dataset
pradyumansharmaaa/stutterclear-weights` (a private dataset of two files, `teacher-v1-best.pt` and
`teacher-v2-best.pt`, where the launcher looks for them; `--dry-run` lists them first), then set
`STUTTERCLEAR_KAGGLE_WEIGHTS` to that dataset in `env.yaml`.

### Once: the smoke run, and pinning what it shows

Before relying on Kaggle, push one small private test kernel by hand (two files in a scratch folder
outside the repo; `kaggle kernels push -p <folder>` with the laptop's own Kaggle CLI) that prints only
versions, paths and timings: the image's Python and PyTorch versions and `pip freeze`, where the two
weight outputs are mounted, how long the pinned downloads and model loads take, one live teacher window
and one 30 s transcription on a T4, and whether the Cloud Run URL answers. Then:

- `stutterclear/gpu_worker/requirements.lock`: only what `pip check` says the image lacks, pinned, then
  its hashes refreshed with `python scripts/lock_hashes.py stutterclear/gpu_worker/requirements.lock`
  (the kernel installs it with `--require-hashes`: exactly those files, whatever is published later);
- `env.yaml`: `STUTTERCLEAR_KAGGLE_DOCKER_IMAGE` set to the image the smoke kernel ran on (its
  `docker_image` in the Kaggle API), so a Kaggle image update cannot break the worker; and
  `STUTTERCLEAR_GPU_TYPICAL_S` to the measured wake times, if they differ from 2-4 minutes;
- the quota before and after the run (`time_used`), to learn how fast 2× T4 spends it.

The very first push of the worker kernel needs `STUTTERCLEAR_KAGGLE_PIN=latest` (there is no "original"
image yet): `gcloud run services update stutterclear --region asia-south1 --update-env-vars
STUTTERCLEAR_KAGGLE_PIN=latest`, wake the GPU once from the page, then the same command with
`--remove-env-vars STUTTERCLEAR_KAGGLE_PIN` (the next deploy would drop it anyway). **Never force-push or
delete the deployed commit before Dec 16**: the worker fetches it by its hash.

### Quota, limits and the Kaggle terms

Kaggle gives about 30 GPU hours a week. The app reads Kaggle's own count before every start and rests
once 20 hours are used (`STUTTERCLEAR_GPU_WEEK_HOURS`, keeping 2 spare), and once a quarter of that is
used in one UTC day (`STUTTERCLEAR_GPU_DAY_HOURS`, 5 h), so a few busy visitors cannot use up the week
in a day; the page then says until when the GPU rests (a standing-by laptop still serves), and
everything else keeps working. A worker takes no new work 90 minutes after it was ready
(`STUTTERCLEAR_GPU_MAX_WAKE_S`): the next recording wakes a fresh one, which counts against the
limits, so steady use cannot keep one machine up for hours. Visitors are limited too: one recording at
a time, 10 minutes of GPU audio an hour and 30 a day each (a recording that never reached a GPU is not
counted), at most 3 GPU starts a day per visitor, 4 an hour and 10 a day in total (only runs actually
started count). Waking the standing-by laptop costs no GPU hours, but each wake holds Cloud Run up for a
while: 4 an hour per visitor and 12 an hour in all (`STUTTERCLEAR_GPU_LAPTOP_WAKES_PER_HOUR`), and a
laptop woken by the Wake button alone sleeps again after 1 minute if no work comes (after 5 minutes when
Record was pressed or a file chosen). Kaggle's terms are a grey area for this ("for the benefit of any
third party"): keep use modest and on demand, let the worker leave by itself, and use the laptop first
during judging. If Kaggle objects, redeploy with `-LaptopOnly`: nothing is started on Kaggle any more.

What else protects the public server: request bodies are capped per route (a voice note at 5.8 MB, the
32 kB-per-second WAV the page sends; a video part at 16 MiB; JSON at 256 kB), and the server reads a
voice note's WAV itself (no audio library ever parses what a visitor sends) and at most three minutes
of a video's sound. Downloads are budgeted per visitor (250 MB an hour) and in total (1 GB an hour, 8 GB
a day): past a budget only the audio and video downloads wait, never the page, and `/api/health` says
`ok: false` meanwhile, so the front door sends visitors to the backup. One visitor may hold live
cleaning for 20 minutes an hour (the slot is shared), and open ten pages and mic sockets at once.

## The team laptop as the GPU (backup)

The laptop (RTX 4050) runs the same worker and needs no tunnel: it connects out to the app.

```powershell
# It asks for the GPU secret at a hidden prompt: paste the value Cloud Run has
python -m stutterclear.gpu_worker --server https://stutterclear-<PROJECT_NUMBER>.asia-south1.run.app --where laptop
```

Never set `$env:STUTTERCLEAR_GPU_SECRET = "..."` by typing it: PowerShell's history file keeps typed lines
in plain text (this laptop's PSReadLine 2.0 filters nothing). If a secret was ever typed that way, run
`Set-PSReadLineOption -HistorySaveStyle SaveNothing` in that window and delete the line from
`%APPDATA%\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt` (or rotate the secret).

- It prints the key id (compare it with the service's log), loads the models, then stands by: no
  open connection, one small request every 15 s, so it costs nothing. A wake uses it first and it is
  ready within seconds; Kaggle is started only if it does not connect within 40 s.
- After the usual idle time it is told to sleep and goes back to standing by. Ctrl+C says goodbye and
  exits; a lost connection (sleep, Wi-Fi) is handled like a crash: the recording goes back in the
  queue and the next wake uses Kaggle.
- A laptop that stops while standing by (Ctrl+C, lid closed) cannot say so: for 45 s after its last
  request the app still counts on it, so a wake in that window waits its 40 s before starting Kaggle.
- During the team's judging slots: plugged in, sleep off, on a stable connection.
- `STUTTERCLEAR_GPU_ROLES=kaggle` in `env.yaml` refuses laptops; `-LaptopOnly` makes the laptop the
  only GPU.

## Try the GPU worker on one computer

The whole path (wake, worker, voice notes, Dictate, video, Best (GPU) live) runs on one computer with
a GPU: the app hides the GPU from itself and wakes the worker as a process, exactly as it would wake
one on Kaggle. PowerShell, from the repository folder:

```powershell
$env:STUTTERCLEAR_GPU_SECRET = python -c "import secrets; print(secrets.token_urlsafe(36))"  # a test value only
$env:STUTTERCLEAR_HOSTED = "1"; $env:STUTTERCLEAR_GPU_LAUNCHER = "local"
$env:STUTTERCLEAR_PUBLIC_URL = "http://127.0.0.1:8801"
$env:CUDA_VISIBLE_DEVICES = "-1"                # the app is a CPU server (on Windows -1; an empty value does not hide it)
$env:STUTTERCLEAR_RECORDINGS_TEACHER = "none"   # so recordings go to the worker
python -m uvicorn stutterclear.app.server:app --host 127.0.0.1 --port 8801 --ws-max-size 4194304
```

Open http://127.0.0.1:8801 and record a voice note. Optional, before starting the app:
`$env:STUTTERCLEAR_GPU_LIVE = "1"` (Best (GPU) live), `$env:STUTTERCLEAR_GPU_LOCAL_ARGS = "--fake"`
(stand-in models: no GPU needed, seconds per wake), `$env:STUTTERCLEAR_GPU_LOCAL_QUOTA_H = "1"` (the
"weekly hours used up" state), `$env:STUTTERCLEAR_GPU_IDLE_S = "60"` (sleeps sooner). The worker runs in
a process group of its own, like a Kaggle worker: Ctrl+C on the app leaves it running, and it
reconnects when the app is back, or leaves by itself once idle. Measured on the team laptop (RTX 4050,
Oct 25): the GPU ready 30-41 s after the wake (11-23 s for live); a 35 s voice note cleaned in 29-50 s
(on a 6 GB card the recordings teacher runs on the CPU, next to CrisperWhisper on the GPU); Best (GPU)
live 0.64 s behind (gentle), with the same cuts as the teacher running in the app itself.

The automated version, with the real models: `$env:STUTTERCLEAR_E2E_GPU = "1"; python -m pytest
tests/test_gpu_worker.py -k end_to_end -s`.

## Front door: GitHub Pages

`frontdoor/` is a static page published to https://pradyuman-aviator.github.io/stutterClear/ by
`.github/workflows/pages.yml` on every push to `main` that changes `frontdoor/`, the measured results
(`runs/eval/*.json`) or the screenshot (`docs/img/app.jpg`), or by hand (Actions → Front door → Run
workflow). It checks every server in `frontdoor/backends.json` at once (`GET /api/health`) and sends the
visitor to the first one in list order that answers, keeping the `#tab`. If none answers, it stays
and shows each server's status, the demo video, the measured results and the screenshot, and checks
again every minute; a server that answers but fails its health check (an older version) is offered as
"Open it anyway", never opened by itself. It never wakes the GPU and holds no secrets.

- **Once** (the repository owner): Settings → Pages → Source: **GitHub Actions**. The address is
  case-sensitive: never rename the repository or make it private.
- **After the first Cloud Run deploy**: replace `https://stutterclear-<PROJECT_NUMBER>.asia-south1.run.app`
  in `backends.json` with the URL the script printed, commit and push to `main`. A placeholder is
  shown as "Not set up yet" and skipped. The workflow refuses to publish a broken `backends.json`.
- **`backends.json`**: `backends` in order of preference, each with a `name`, a `url` (`https://<host>`
  only) and `first_timeout_s` (how long to wait for it, 3-120 s: 45 for Cloud Run, which may be
  starting from cold); `video_url` (a YouTube link is embedded in privacy-enhanced mode; `null` hides
  it); `repo`. Re-point the link by editing this file only.
- **Addresses**: `?stay=1` shows the page without leaving (the app links to it as its status page);
  `?to=1` opens the second server whatever the first does; unknown paths come back to the page.
- **The servers** must answer `/api/health` with CORS for `https://pradyuman-aviator.github.io`
  (`STUTTERCLEAR_CORS_ORIGINS`, set in `env.yaml`). The Azure backup gains `/api/health` when it is
  rebuilt from `main` after the merge; until then it is offered as "Open it anyway".
- **Never release the Azure VM's IP address or DNS name while `backends.json` lists it**: remove the
  entry first, or someone else could claim the name and receive the judges.
- **Try it locally**: `python -m http.server 8831 -d frontdoor`, then open http://127.0.0.1:8831/ (to
  see the results and the screenshot, copy `runs/eval/live_quality.json` and `recordings.json` into
  `frontdoor/results/` and `docs/img/app.jpg` into `frontdoor/img/`; both folders are git-ignored).
  A local server listed as `http://127.0.0.1:<port>` works too; the published list accepts https only.

## Judging checklist (Dec 1-15)

- Dec 1: one instance always on, so judges never wait for a cold start:
  `gcloud run services update stutterclear --region asia-south1 --min=1` (≈ $0.054 an hour).
- Every morning: the front door opens the app (a private window, on a phone and a laptop);
  `https://pradyuman-aviator.github.io/stutterClear/?stay=1` shows both servers running; one voice note
  goes through end to end; the Kaggle hours used this week (`week` in `<url>/api/gpu`, refreshed
  while the GPU is up); the bill against the expected ≈ $1-2 a day.
- The laptop stands by during the team's judging slots.
- No rebuilds: change settings in place or roll back (see Update, roll back, switch off).
- If the app misbehaves: roll back. If it is abused: the kill switch, then put the backup first in
  `backends.json`.

## After Dec 16

- `gcloud run services update stutterclear --region asia-south1 --min=0`, or delete the service.
- Disable the GPU secret (`gcloud secrets versions disable <version> --secret STUTTERCLEAR_GPU_SECRET`)
  or delete it: every token and the laptop's copy stop working.
- Revoke the Kaggle API token (kaggle.com → Settings → API) and delete the worker kernel: its versions
  keep expired tokens and logs.
- Delete the old images: `gcloud artifacts docker images list asia-south1-docker.pkg.dev/<project>/stutterclear/app`,
  then `gcloud artifacts docker images delete <image>@<digest>` for each one no longer used.

## Nebius AI Cloud: a GPU VM

The hosted demo can also run the same app on a GPU VM: visitors use their browser's microphone, voice
notes are cleaned on the server's GPU, and Nemotron runs on Nebius Token Factory. Nothing a visitor
says is written to disk.

### 1. Create the VM

In the Nebius AI Cloud console create a VM with one NVIDIA GPU (the app needs about 5 GB of GPU
memory; any current data-centre GPU is plenty), the "Ubuntu with CUDA / NVIDIA drivers" image, a
public IP, and ports 22, 80 and 443 open.

On the VM, Docker and the NVIDIA Container Toolkit must be present (Nebius GPU images usually
include them; otherwise follow docs.docker.com and the NVIDIA Container Toolkit guide). Check:

```
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 2. Code and assets

```
git clone https://github.com/Pradyuman-aviator/stutterClear.git && cd stutterClear
```

Models and demo data are not in git. On the laptop that has them:

```
python scripts/pack_assets.py                 # writes stutterclear_assets.tar
scp stutterclear_assets.tar <user>@<vm-ip>:~/stutterClear/
```

On the VM: `tar xf stutterclear_assets.tar`.

### 3. Keys and domain

```
cp .env.example .env
```

Fill in `.env`: `DOMAIN` (your VM's IP with dashes plus `.sslip.io` gives a free name with HTTPS,
e.g. `203-0-113-7.sslip.io`), `NEBIUS_API_KEY`, and optionally `TAVILY_API_KEY`. Keep `.env` on the
server only.

### 4. Start

```
docker compose up -d --build
docker compose logs -f app        # first start downloads WavLM and CrisperWhisper weights once
```

Open `https://<DOMAIN>`. Caddy gets the HTTPS certificate on the first visit.

Update after a `git push`: `git pull && docker compose up -d --build`.

### What runs where

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

## Always on, free for students: a small Azure server (CPU)

The public demo runs here: https://stutterclear-demo.centralindia.cloudapp.azure.com. An Azure for
Students subscription ($100 credit, no card) pays for a Standard_B2s VM (2 vCPU, 4 GB, about $30 a
month), with the same CPU build as the Hugging Face Space below, behind Caddy for HTTPS
(`deploy/azure/`).

```
az login --use-device-code                      # the account with the student subscription
az group create -n stutterclear -l centralindia
az vm create -g stutterclear -n stutterclear --image Ubuntu2204 --size Standard_B2s \
  --admin-username stutter --ssh-key-values ~/.ssh/<key>.pub --public-ip-sku Standard \
  --public-ip-address-dns-name stutterclear-demo --os-disk-size-gb 64
az vm open-port -g stutterclear -n stutterclear --port 80,443 --priority 1010
```

On the VM: install Docker (`curl -fsSL https://get.docker.com | sudo sh`), add a 4 GB swap file,
copy `deploy/azure/*` to `~/stutterclear/`, write `~/stutterclear/.env` there (`DOMAIN=<the VM's
name>.centralindia.cloudapp.azure.com`, `NEBIUS_API_KEY`, `NVIDIA_API_KEY`; chmod 600), then
`sudo docker compose up -d --build`. Update after a `git push`: `sudo docker compose build --no-cache
app && sudo docker compose up -d`. Stop paying: delete the resource group `stutterclear`.

## Free backup: a Hugging Face Space (always on, CPU)

Since 2026 a Docker Space on free CPU needs a Hugging Face PRO subscription ("Static Spaces are free
for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription").
With PRO, `python scripts/deploy_hf_space.py` creates it.

A Space keeps a link alive: live cleaning with the rule-based detector,
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
| `NEBIUS_API_KEY` | Nemotron on Nebius Token Factory: word rescue, the meaning guard, the context agent, insights |
| `STUTTERCLEAR_TRUST_PROXY=N` | Behind N proxies that append to `X-Forwarded-For` (Cloud Run or Caddy alone: 1): the visitor is the N-th entry from the end. How to check it: [Google Cloud Run](#google-cloud-run-the-main-app) |
| `STUTTERCLEAR_CORS_ORIGINS` | Origins that may read `/api/health` (comma-separated; default `https://pradyuman-aviator.github.io`, the front door) |
| `STUTTERCLEAR_VIDEO_MAX_MB` | Largest video accepted (default 300; Cloud Run 150) |
| `STUTTERCLEAR_EGRESS_MB_PER_CLIENT` / `STUTTERCLEAR_EGRESS_MB` / `STUTTERCLEAR_EGRESS_MB_PER_DAY` | Download budget per visitor an hour / in total an hour / in total a day, public mode only (default 250 / 1000 / 8000); past one, only the audio and video downloads wait |
| `STUTTERCLEAR_COMMIT` | The code version (`/api/health` shows its first 12 digits; the Cloud Run image sets it; otherwise `git rev-parse HEAD`, else `dev`) |
| `STUTTERCLEAR_CALIBRATION_JSON` | Optional: the recordings teacher's `live_cuts.json` content, for a server without `runs/` |

The GPU worker, on the app server (all off unless `STUTTERCLEAR_GPU_SECRET` is set; then
`/api/info` gains `gpu`):

| Variable | Default | Effect |
|---|---|---|
| `STUTTERCLEAR_GPU_SECRET` | unset | At least 32 characters: turns the GPU worker on and signs its tokens. **Secret** |
| `STUTTERCLEAR_GPU_LAUNCHER` | unset | `kaggle` starts workers on Kaggle; `local` on this computer (tests); unset: only the laptop can serve |
| `STUTTERCLEAR_PUBLIC_URL` | unset | This server's `https://` address; workers connect to it and tokens are signed for it. Needed with a launcher |
| `STUTTERCLEAR_GPU_ROLES` | the launcher's + `laptop` | Which workers may connect (`kaggle` alone refuses the laptop) |
| `STUTTERCLEAR_GPU_LIVE` | `0` | `1` offers Best (GPU) live through the worker (phase P2) |
| `STUTTERCLEAR_GPU_IDLE_S` | 600 | The worker sleeps this long after its last job (if no work came: 4 minutes; 5 when Record was pressed or a file chosen; 1 for the laptop woken by the button; never longer than this) |
| `STUTTERCLEAR_GPU_MAX_WAKE_S` | 5400 | A worker takes no new work this long after it was ready: the next work wakes a fresh one |
| `STUTTERCLEAR_GPU_WAKE_TIMEOUT_S` | 720 | Give up waiting for a worker to connect |
| `STUTTERCLEAR_GPU_WEEK_HOURS` / `_RESERVE_HOURS` | 20 / 2 | Our cap on Kaggle's weekly GPU hours, and what is always left unused |
| `STUTTERCLEAR_GPU_DAY_HOURS` | a quarter of the week's | Our cap on Kaggle's GPU hours in one UTC day |
| `STUTTERCLEAR_GPU_WAKES_PER_HOUR` / `_PER_DAY` | 4 / 10 | GPU starts allowed in total |
| `STUTTERCLEAR_GPU_CLIENT_WAKES_PER_DAY` | 3 | GPU starts one visitor may cause |
| `STUTTERCLEAR_GPU_LAPTOP_WAKES_PER_HOUR` | 12 | Wakes of the standing-by laptop without work yet, in total (4 per visitor) |
| `STUTTERCLEAR_GPU_CLIENT_AUDIO_S_PER_HOUR` / `_PER_DAY` | 600 / 1800 | GPU audio one visitor may send an hour / a day |
| `STUTTERCLEAR_GPU_CLIENT_LIVE_PER_HOUR` | 3 | Best (GPU) live sessions one visitor may start an hour (P2) |
| `STUTTERCLEAR_GPU_TYPICAL_S` | `120,240` | The "usually 2-4 minutes" the page shows while waking |
| `STUTTERCLEAR_GPU_LOCAL_ARGS` / `_LOCAL_QUOTA_H` | unset | Local launcher only (tests): extra worker flags / pretend this many quota hours are left |

The Kaggle launcher (with `STUTTERCLEAR_GPU_LAUNCHER=kaggle`):

| Variable | Default | Effect |
|---|---|---|
| `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY` | unset | The Kaggle account that owns the weight kernels (a named token is preferred). **Secret** |
| `STUTTERCLEAR_KAGGLE_KERNEL` | `pradyumansharmaaa/stutterclear-gpu-worker` | The private worker kernel (the first start creates it) |
| `STUTTERCLEAR_KAGGLE_WEIGHT_KERNELS` | `pradyumansharmaaa/stutterclear-teacher,pradyumansharmaaa/stutterclear-teacher-v2` | The live and the recordings teacher's training kernels, whose outputs hold the weights. Never re-run them before Dec 16 |
| `STUTTERCLEAR_KAGGLE_WEIGHTS` | unset | Fallback: a private dataset from `scripts/upload_weights_kaggle.py` |
| `STUTTERCLEAR_KAGGLE_ACCELERATOR` | `NvidiaTeslaT4` | 2× T4 |
| `STUTTERCLEAR_KAGGLE_TIMEOUT_S` | 10800 | Hard limit per worker run |
| `STUTTERCLEAR_KAGGLE_PIN` | `original` | `latest` for the very first start only |
| `STUTTERCLEAR_KAGGLE_DOCKER_IMAGE` | unset | The Kaggle image the smoke run used, once known |
| `STUTTERCLEAR_REPO_URL` | `https://github.com/Pradyuman-aviator/stutterClear.git` | The public code the worker fetches, at `STUTTERCLEAR_COMMIT` |

On the worker itself: `STUTTERCLEAR_GPU_SECRET` on the laptop (it signs its own tokens with it, or
asks for it), `STUTTERCLEAR_GPU_TOKEN` on Kaggle (that wake's token, set by the kernel), and
`STUTTERCLEAR_TEACHER` / `STUTTERCLEAR_RECORDINGS_TEACHER` for the checkpoint paths (the kernel sets
them; the laptop uses `runs/`).
