<#
Deploy StutterClear's main app to Google Cloud Run (CPU, asia-south1) from this checkout: Cloud Build
builds the image at the current commit (deploy/cloudrun/Dockerfile), the service runs it by digest
with the settings in deploy/cloudrun/env.yaml, and the URL is printed. Safe to run again: it creates
only what is missing and builds each commit once. It never prints a secret, and secret values go to
Secret Manager's API as base64, never through gcloud's input or command line (Windows PowerShell
adds a byte-order mark and a line ending to text it pipes).

    powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1 [-Project ID] [-NoGpu | -LaptopOnly]
        [-Live] [-SetGpuSecret] [-UpdateSecrets NAME,...] [-SkipBuild] [-WarmDuringJudging | -MinInstances N]
        [-DryRun]

  -NoGpu              the CPU app alone (phase P0): no GPU worker, no Kaggle or GPU secret
  -LaptopOnly         the GPU features only through the team's laptop: nothing is started on Kaggle
  -Live               also offer Best (GPU) live through the worker (phase P2)
  -SetGpuSecret       store a new STUTTERCLEAR_GPU_SECRET (typed in at a hidden prompt)
  -UpdateSecrets      store new versions of these secrets (typed in at a hidden prompt each)
  -SkipBuild          never build: deploy the image built earlier for this commit (settings changes)
  -WarmDuringJudging  keep one instance always on (Dec 1-15: no cold start for judges, about $0.05/h)
  -MinInstances N     set the always-on instances explicitly (0 after judging); otherwise left as they are
  -DryRun             check everything and print each change (gcloud command) instead of making it

Needs the Google Cloud CLI logged in (gcloud auth login) as a project Editor or Owner, and, unless
-NoGpu or -LaptopOnly, this commit pushed to GitHub: the Kaggle worker fetches exactly this commit.
What an Editor may not do (public access, letting the service's account read its secrets, the
build's permissions) is printed for the project Owner to run once. Secrets that Secret Manager does
not have yet (NEBIUS_API_KEY, NVIDIA_API_KEY, KAGGLE_API_TOKEN and STUTTERCLEAR_GPU_SECRET) are asked
for at a hidden prompt: never type them into a `$env:` line, which PowerShell's history file keeps in
plain text. (A variable of the same name already set in this terminal is used instead.) The steps,
the Owner's commands and every setting: docs/DEPLOY.md.
#>
param(
    [string]$Project = "",
    [string]$Region = "asia-south1",
    [string]$Service = "stutterclear",
    [switch]$NoGpu,
    [switch]$LaptopOnly,
    [switch]$Live,
    [switch]$SetGpuSecret,
    [ValidateSet("NEBIUS_API_KEY", "NVIDIA_API_KEY", "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY",
        "STUTTERCLEAR_GPU_SECRET")]
    [string[]]$UpdateSecrets = @(),
    [switch]$SkipBuild,
    [switch]$WarmDuringJudging,
    [int]$MinInstances = -1,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$newGpuSecret = $SetGpuSecret -or ($UpdateSecrets -contains "STUTTERCLEAR_GPU_SECRET")
if ($NoGpu -and ($LaptopOnly -or $Live -or $newGpuSecret)) {
    Write-Host "-NoGpu deploys without the GPU worker: leave out -LaptopOnly, -Live and the GPU secret." -ForegroundColor Red
    exit 2
}
$gpu = -not $NoGpu
$kaggle = $gpu -and -not $LaptopOnly
$min = $MinInstances
if ($WarmDuringJudging) { $min = 1 }
$accessor = "roles/secretmanager.secretAccessor"
$optional = "NEBIUS_API_KEY", "NVIDIA_API_KEY"  # without them only their features stay hidden

function Find-GCloud {
    # gcloud.ps1 hands its arguments to gcloud's Python as an array; gcloud.cmd goes through cmd.exe,
    # which mangles ^ ( ) & and < in them. Prefer the .ps1 next to whichever gcloud is installed.
    $dirs = @()
    $cmd = Get-Command gcloud -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd -and $cmd.Source) { $dirs += Split-Path -Parent $cmd.Source }
    $dirs += Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin"
    if (${env:ProgramFiles(x86)}) { $dirs += Join-Path ${env:ProgramFiles(x86)} "Google\Cloud SDK\google-cloud-sdk\bin" }
    foreach ($dir in $dirs) {
        foreach ($name in "gcloud.ps1", "gcloud.cmd") {
            $path = Join-Path $dir $name
            if (Test-Path $path) { return $path }
        }
    }
    throw "The Google Cloud CLI is not installed: https://cloud.google.com/sdk/docs/install, then 'gcloud auth login'."
}

function Invoke-GCloud {
    # gcloud's standard output (its progress and errors stay on the screen); throws if it fails.
    # Nothing passed to gcloud by this script is ever a secret.
    $ErrorActionPreference = "Continue"
    $out = & $script:gcloud @args --quiet
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($args -join ' ') failed (see above)." }
    $out
}

function Invoke-Change {
    # gcloud for anything that changes the project; with -DryRun only shown
    if ($DryRun) { Write-Host "  would run: gcloud $($args -join ' ')" -ForegroundColor Cyan; return }
    Invoke-GCloud @args | Out-Null
}

function Test-GCloud {
    # True when gcloud succeeds, quietly: does this thing exist (and may we see it)?
    $ErrorActionPreference = "Continue"
    & $script:gcloud @args --quiet 2>$null | Out-Null
    $LASTEXITCODE -eq 0
}

function Get-GCloudJson {
    $text = (Invoke-GCloud @args --format=json) -join "`n"
    if (-not $text.Trim()) { return }
    $parsed = ConvertFrom-Json -InputObject $text
    $parsed  # from a variable, so a JSON list comes out item by item (ConvertFrom-Json emits it whole)
}

function Show-OwnerCommands([string[]]$Commands, [string]$Why) {
    Write-Host ""
    Write-Host "The project Owner needs to run this once ($Why). Send it to them, then run this script again:" -ForegroundColor Yellow
    foreach ($c in $Commands) { Write-Host "  $c" }
    Write-Host ""
}

function Test-Role($Policy, [string]$Role, [string]$Member) {
    foreach ($b in @($Policy.bindings)) {
        if ($b -and $b.role -eq $Role -and @($b.members) -contains $Member) { return $true }
    }
    $false
}

function Get-LocalSecret([string]$Name) {
    # This terminal's value, tidied as gpu_protocol.read_secret does it; $null when unset
    $v = [Environment]::GetEnvironmentVariable($Name)
    if ($null -ne $v) { $v = $v.Trim().TrimStart([char]0xFEFF).Trim() }
    if ($v) { $v } else { $null }
}

function Read-Secret([string]$Name, [string]$Hint = "hidden") {
    # Typed at a hidden prompt: never on a command line, in a file or in PowerShell's history (which
    # keeps every typed line, `$env: assignments included, in plain text). $null when left empty.
    $secure = Read-Host -AsSecureString "$Name ($Hint)"
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $v = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    $v = $v.Trim().TrimStart([char]0xFEFF).Trim()
    if ($v) { $v } else { $null }
}

function Read-GpuSecret {
    $v = Get-LocalSecret "STUTTERCLEAR_GPU_SECRET"
    if (-not $v) { $v = Read-Secret "STUTTERCLEAR_GPU_SECRET" "hidden; the same value the laptop worker uses" }
    if (-not $v -or $v.Length -lt 32) {
        throw ("The GPU secret must be at least 32 characters. Make one with " +
            "python -c `"import secrets; print(secrets.token_urlsafe(36))`" and keep it in a password manager.")
    }
    $v
}

function Get-KeyId([string]$Secret) {
    # gpu_protocol.key_id: the first 8 hex digits of HMAC-SHA256(secret, "sc-key-id"). Safe to print.
    $hmac = New-Object System.Security.Cryptography.HMACSHA256 (, [Text.Encoding]::UTF8.GetBytes($Secret))
    $hash = $hmac.ComputeHash([Text.Encoding]::ASCII.GetBytes("sc-key-id"))
    ($hash[0..3] | ForEach-Object { $_.ToString("x2") }) -join ""
}

function Save-Secret([string]$Name, [string]$Value) {
    # A new version: base64 of the UTF-8 bytes to Secret Manager's API, so nothing is added to the value
    if ($DryRun) { Write-Host "  would store a new version of $Name" -ForegroundColor Cyan; return }
    if (-not $existing.ContainsKey($Name)) {
        Invoke-GCloud secrets create $Name --replication-policy=automatic "--project=$Project" | Out-Null
        $existing[$Name] = $true
    }
    $token = Invoke-GCloud auth print-access-token | Select-Object -First 1
    $body = @{ payload = @{ data = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Value)) } } |
        ConvertTo-Json -Compress
    try {
        $r = Invoke-RestMethod -Method Post -ContentType "application/json" -Body $body `
            -Uri "https://secretmanager.googleapis.com/v1/projects/$Project/secrets/${Name}:addVersion" `
            -Headers @{ Authorization = "Bearer $token" }
    }
    catch { throw "Could not store $Name in Secret Manager: $($_.Exception.Message)" }
    Write-Host "Stored $Name ($($r.name -replace '^.*/versions/', 'version '))."
}

function Get-LatestVersion([string]$Name) {
    # The newest enabled version (pinned in the deploy: a newer one changes nothing until the next deploy)
    if (-not $existing.ContainsKey($Name)) { return $null }
    $v = @(Get-GCloudJson secrets versions list $Name "--filter=state:ENABLED" "--sort-by=~createTime" "--limit=1" "--project=$Project")
    if ($v -and $v[0]) { ($v[0].name -split "/")[-1] } else { $null }
}

function Invoke-Build([string]$Tag, [string]$Sha) {
    # Streams the build's log; when it fails for want of a permission, prints the Owner's command
    $build = "builds", "submit", "--config=deploy/cloudrun/cloudbuild.yaml", "--project=$Project",
        "--substitutions=_TAG=$Tag,_COMMIT=$Sha,_REGION=$Region", "."
    if ($DryRun) { Write-Host "  would run: gcloud $($build -join ' ')" -ForegroundColor Cyan; return }
    $log = New-Object System.Collections.Generic.List[string]
    $ErrorActionPreference = "Continue"
    & $script:gcloud @build --quiet 2>&1 | ForEach-Object { $line = "$_"; $log.Add($line); Write-Host $line }
    if ($LASTEXITCODE -eq 0) { return }
    $text = $log -join "`n"
    $accounts = @([regex]::Matches($text, "[\w.-]+@[\w.-]+\.gserviceaccount\.com") | ForEach-Object { $_.Value } |
        Where-Object { $_ -notmatch "@gcp-sa-" } | Sort-Object -Unique)
    if ($accounts -and $text -match "PERMISSION_DENIED|[Pp]ermission|denied") {
        Show-OwnerCommands ($accounts | ForEach-Object {
                "gcloud projects add-iam-policy-binding $Project --member=serviceAccount:$_ --role=roles/cloudbuild.builds.builder"
            }) "the build's account may not read the upload, write logs or push the image"
    }
    throw "The build failed (see above)."
}

$script:gcloud = Find-GCloud
Push-Location $root
$envFile = Join-Path ([IO.Path]::GetTempPath()) "stutterclear-cloudrun-env-$PID.yaml"
try {
    if ($DryRun) { Write-Host "Dry run: checks only; every change is printed instead of made." -ForegroundColor Cyan }

    # 1. Who, which project, where the service will be
    $active = @(Get-GCloudJson auth list) | Where-Object { $_ -and $_.status -eq "ACTIVE" }
    if (-not $active) { throw "Log in first: gcloud auth login" }
    if (-not $Project) { $Project = "$(Invoke-GCloud config get-value project | Select-Object -First 1)".Trim() }
    if (-not $Project) { throw "Say which project: -Project <id> (or gcloud config set project <id>)." }
    $number = "$((Get-GCloudJson projects describe $Project).projectNumber)"
    if ($number -notmatch "^\d+$") { throw "Could not read the number of project $Project." }
    $url = "https://{0}-{1}.{2}.run.app" -f $Service, $number, $Region
    $sa = "stutterclear-run@$Project.iam.gserviceaccount.com"
    $image = "$Region-docker.pkg.dev/$Project/stutterclear/app"
    Write-Host "Project $Project as $(@($active)[0].account): $url"

    # 2. The commit: what is built is exactly what is committed (and, for Kaggle, pushed)
    $deployed = "requirements.txt", "scripts/download_asr.py", "stutterclear", "deploy/cloudrun", "data/demo_clips",
        "runs/eval/live_quality.json", "runs/eval/recordings.json", ".gcloudignore"
    $dirty = @(git status --porcelain --untracked-files=all -- $deployed ":(exclude)stutterclear/app/web")
    if ($LASTEXITCODE -ne 0) { throw "git status failed: run this from the StutterClear repository." }
    if ($dirty) {
        $dirty | ForEach-Object { Write-Host "  $_" }
        throw "Commit these first: the image is built from this folder and says which commit it runs."
    }
    $sha = "$(git rev-parse HEAD)".Trim()
    $tag = $sha.Substring(0, 12)
    if (-not @(git branch -r --contains $sha)) {
        if ($kaggle) { throw "Push this commit first (git push): the Kaggle worker fetches exactly $sha from GitHub." }
        Write-Warning "Commit $tag is not on GitHub yet. Push it before Kaggle uses it."
    }

    # 3. The upload: only committed files, never recordings, checkpoints or keys (.gcloudignore), except
    # the UCLASS recordings the demo plays: never in git, each checked against the hash the commit lists
    $tracked = @{}
    git ls-files | ForEach-Object { $tracked[$_] = $true }
    $private = @{}
    foreach ($line in @(Get-Content deploy/cloudrun/uclass.sha256 | Where-Object { $_.Trim() })) {
        $hash, $file = -split $line
        if ($tracked.ContainsKey($file)) { throw "$file is in git: UCLASS audio must never be committed (git rm --cached)." }
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "$file is missing: the sample clips need the UCLASS download and its caches (data/external)."
        }
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $hash) {
            throw "$file is not the file deploy/cloudrun/uclass.sha256 lists: update the list and commit it."
        }
        $private[$file] = $true
    }
    $upload = @(Invoke-GCloud meta list-files-for-upload . | ForEach-Object { $_ -replace "\\", "/" })
    $stray = @($upload | Where-Object { -not $tracked.ContainsKey($_) -and -not $private.ContainsKey($_) })
    $stray += @($upload | Where-Object {
            $_ -match "\.(ogg|opus|m4a|mp3|wav|mp4|mov|webm|pt|onnx|pem|key)$|(^|/)\.env|kaggle\.json$" -and
            $_ -notmatch "^data/demo_clips/[\w.-]+\.wav$" -and -not $private.ContainsKey($_) })
    if ($stray) {
        $stray | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }
        throw "These would be uploaded to the build: commit them or leave them out (.gcloudignore)."
    }
    $left = @($private.Keys | Where-Object { $upload -notcontains $_ })
    if ($left) {
        $left | ForEach-Object { Write-Host "  $_" }
        throw "These must go up for the sample clips: let them in (.gcloudignore)."
    }
    Write-Host "Upload checked: $($upload.Count - $private.Count) committed files and $($private.Count) UCLASS files."

    # 4. The project's parts: APIs, the image repository, the service's own account
    Invoke-Change services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com `
        secretmanager.googleapis.com iam.googleapis.com "--project=$Project"
    if (-not (Test-GCloud artifacts repositories describe stutterclear "--location=$Region" "--project=$Project")) {
        Invoke-Change artifacts repositories create stutterclear --repository-format=docker "--location=$Region" `
            "--description=StutterClear app images" "--project=$Project"
    }
    if (-not (Test-GCloud iam service-accounts describe $sa "--project=$Project")) {
        try { Invoke-Change iam service-accounts create stutterclear-run "--display-name=StutterClear on Cloud Run" "--project=$Project" }
        catch {
            Show-OwnerCommands @("gcloud iam service-accounts create stutterclear-run --project=$Project --display-name=StutterClear") `
                "the service runs as its own account, which may read only its secrets"
            exit 1
        }
    }

    # 5. Secrets: created from this terminal's variables when missing; new versions only when asked
    $existing = @{}
    foreach ($s in @(Get-GCloudJson secrets list "--project=$Project")) {
        if ($s) { $existing[($s.name -split "/")[-1]] = $true }
    }
    foreach ($name in @($UpdateSecrets | Where-Object { $_ -ne "STUTTERCLEAR_GPU_SECRET" })) {
        $v = Get-LocalSecret $name
        if (-not $v -and -not $DryRun) { $v = Read-Secret $name "hidden; its new value" }
        if (-not $v -and -not $DryRun) { throw "-UpdateSecrets ${name}: nothing was typed." }
        Save-Secret $name $v
        $v = $null
    }
    if ($gpu -and ($newGpuSecret -or -not (Get-LatestVersion "STUTTERCLEAR_GPU_SECRET"))) {
        if ($DryRun) { Save-Secret "STUTTERCLEAR_GPU_SECRET" "" }
        else {
            $v = Read-GpuSecret
            Save-Secret "STUTTERCLEAR_GPU_SECRET" $v
            Write-Host "GPU secret key id $(Get-KeyId $v): the service's log and the laptop worker print the same."
            $v = $null
        }
    }
    $wanted = @() + $optional
    if ($gpu) { $wanted += "STUTTERCLEAR_GPU_SECRET" }
    if ($kaggle) {
        # a named token (revocable on its own) is preferred over the account's legacy key, which is
        # used only while it is stored and no token is
        if (-not $existing.ContainsKey("KAGGLE_API_TOKEN") -and -not (Get-LocalSecret "KAGGLE_API_TOKEN") -and
            $existing.ContainsKey("KAGGLE_USERNAME") -and $existing.ContainsKey("KAGGLE_KEY")) {
            $wanted += "KAGGLE_USERNAME", "KAGGLE_KEY"
        }
        else { $wanted += "KAGGLE_API_TOKEN" }
    }
    $mounts = [ordered]@{}
    foreach ($name in $wanted) {
        $version = Get-LatestVersion $name
        if (-not $version) {
            $v = Get-LocalSecret $name
            if (-not $v -and -not $DryRun) {
                $v = Read-Secret $name $(if ($optional -contains $name) { "hidden; Enter to skip: its features stay hidden" } else { "hidden" })
            }
            if ($v) { Save-Secret $name $v; $v = $null; $version = Get-LatestVersion $name }
            elseif ($optional -contains $name) {
                Write-Warning "$name is not in Secret Manager: the features that need it stay hidden."
                continue
            }
            elseif (-not $DryRun) { throw "$name is not in Secret Manager yet, and nothing was typed for it." }
        }
        if (-not $version) { $version = "<new>" }  # a dry run that would store it
        $mounts[$name] = $version
    }
    Write-Host ("Secrets: " + (($mounts.Keys | ForEach-Object { "$_ v$($mounts[$_])" }) -join ", "))

    # 6. The service's account must be able to read each of them: an Owner running this grants it
    # here; for an Editor it becomes one of the Owner's commands
    $member = "serviceAccount:$sa"
    $everywhere = $false
    try { $everywhere = Test-Role (Get-GCloudJson projects get-iam-policy $Project) $accessor $member } catch { }
    $grants = @()
    if (-not $everywhere) {
        foreach ($name in @($mounts.Keys)) {
            if ($existing.ContainsKey($name) -and
                (Test-Role (Get-GCloudJson secrets get-iam-policy $name "--project=$Project") $accessor $member)) { continue }
            $grant = "secrets", "add-iam-policy-binding", $name, "--project=$Project", "--member=$member", "--role=$accessor"
            if ($DryRun -or -not (Test-GCloud @grant)) { $grants += "gcloud " + ($grant -join " ") }
        }
    }
    if ($grants) {
        Show-OwnerCommands $grants "the service's account may read only the secrets it needs"
        if (-not $DryRun) { exit 1 }
    }

    # 7. The image: built once per commit, deployed by digest
    $digest = "sha256:<the new image's digest>"
    if (Test-GCloud artifacts docker images describe "${image}:$tag" "--project=$Project") {
        Write-Host "The image for commit $tag is already built: deploying it."
    }
    elseif ($SkipBuild) { throw "No image for commit $tag yet: run without -SkipBuild." }
    else {
        Write-Host "Building ${image}:$tag with Cloud Build (about 10 minutes)..."
        Invoke-Build $tag $sha
    }
    if (-not $DryRun) {
        $digest = (Get-GCloudJson artifacts docker images describe "${image}:$tag" "--project=$Project").image_summary.digest
        if ($digest -notmatch "^sha256:[0-9a-f]{64}$") { throw "Could not read the digest of ${image}:$tag." }
    }

    # 8. Deploy: deploy/cloudrun/env.yaml plus what this script works out (no secrets in the file)
    $settings = @(Get-Content (Join-Path $root "deploy/cloudrun/env.yaml"))
    foreach ($k in "STUTTERCLEAR_PUBLIC_URL", "STUTTERCLEAR_GPU_LAUNCHER", "STUTTERCLEAR_GPU_LIVE", "STUTTERCLEAR_NOTES") {
        if ($settings -match "^\s*${k}\s*:") { throw "deploy/cloudrun/env.yaml must not set ${k}: this script does." }
    }
    $extra = [ordered]@{ STUTTERCLEAR_PUBLIC_URL = $url }
    if ($kaggle) { $extra.STUTTERCLEAR_GPU_LAUNCHER = "kaggle" }
    if ($gpu) { $extra.STUTTERCLEAR_GPU_LIVE = $(if ($Live) { "1" } else { "0" }) }
    else { $extra.STUTTERCLEAR_NOTES = "0" }
    $settings += $extra.Keys | ForEach-Object { "${_}: `"$($extra[$_])`"" }
    [IO.File]::WriteAllText($envFile, ($settings -join "`n") + "`n", (New-Object Text.UTF8Encoding $false))
    Write-Host ("Settings: " + (($settings | Where-Object { $_ -match "^\s*[A-Z_]+\s*:" } |
                ForEach-Object { ($_ -split ":")[0].Trim() }) -join ", "))
    $deploy = @("run", "deploy", $Service, "--image=$image@$digest", "--region=$Region", "--project=$Project",
        "--port=7860", "--cpu=2", "--memory=4Gi", "--cpu-boost", "--execution-environment=gen2",
        "--concurrency=80", "--timeout=3600", "--session-affinity", "--max-instances=1",
        "--service-account=$sa", "--env-vars-file=$envFile")
    if ($mounts.Count) { $deploy += "--set-secrets=" + (($mounts.Keys | ForEach-Object { "$_=${_}:$($mounts[$_])" }) -join ",") }
    else { $deploy += "--clear-secrets" }
    if ($min -ge 0) { $deploy += "--min=$min" }  # the service's minimum: it changes without a new revision
    Write-Host "Deploying commit $tag ($digest)..."
    Invoke-Change @deploy
    # a rollback (update-traffic --to-revisions) pins traffic: send it to what was just deployed
    Invoke-Change run services update-traffic $Service --to-latest "--region=$Region" "--project=$Project"

    # 9. Request logs below 500 are not kept: cost, and no visitor addresses in logs for 30 days
    $owner = @()
    $filter = "resource.type=cloud_run_revision AND resource.labels.service_name=$Service AND " +
        "logName=projects/$Project/logs/run.googleapis.com%2Frequests AND httpRequest.status<500"
    try {
        $sink = Get-GCloudJson logging sinks describe _Default "--project=$Project"
        if (-not (@($sink.exclusions) | Where-Object { $_ -and $_.name -eq "run-requests-ok" })) {
            Invoke-Change logging sinks update _Default "--add-exclusion=name=run-requests-ok,filter=$filter" "--project=$Project"
        }
    }
    catch { $owner += "gcloud logging sinks update _Default --project=$Project `"--add-exclusion=name=run-requests-ok,filter=$filter`"" }

    # 10. Is it up, public, and running this commit?
    if (-not $DryRun) {
        Write-Host "Checking $url/api/health (a cold start takes up to a minute)..."
        $health = $null
        $status = 0
        foreach ($attempt in 1..5) {
            try { $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 90; break }
            catch {
                $status = 0
                if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
                if ($status -eq 403 -or $status -eq 404) { break }
                Start-Sleep -Seconds 10
            }
        }
        if ($status -eq 403) {
            $public = "run", "services", "add-iam-policy-binding", $Service, "--project=$Project", "--region=$Region",
                "--member=allUsers", "--role=roles/run.invoker"
            if (Test-GCloud @public) { Write-Host "Made the service public: it takes a minute to apply." }
            else { $owner += "gcloud " + ($public -join " ") }
        }
        elseif ($status -eq 404) {
            $null = Invoke-RestMethod -Uri "$url/api/info" -TimeoutSec 60
            Write-Warning "Up, but this version has no /api/health, so the front door cannot check it."
        }
        elseif ($health -and $health.ok -eq $true) {
            Write-Host "Up: version $($health.version), GPU $(if ($health.gpu) { $health.gpu } else { 'not set up' })." -ForegroundColor Green
            if ($health.version -ne $tag) { Write-Warning "It answers as $($health.version), not ${tag}: is traffic on another revision?" }
        }
        else { Write-Warning "$url did not answer its health check: see the service's log in the Cloud console." }
    }
    if ($owner) { Show-OwnerCommands $owner "public access, and request logs not kept" }

    Write-Host ""
    Write-Host "StutterClear: $url  (commit $tag)" -ForegroundColor Green
    Write-Host "- Front door: this URL is the first backend in frontdoor/backends.json (commit, push to main)."
    if ($gpu) {
        Write-Host "- GPU: the service's log says 'gpu: worker secret key id ...'; the laptop worker must print the same:"
        Write-Host "    python -m stutterclear.gpu_worker --server $url --where laptop"
    }
    Write-Host "- Judging, no new revision: Dec 1  gcloud run services update $Service --region=$Region --min=1"
    Write-Host "                            Dec 16 gcloud run services update $Service --region=$Region --min=0"
    Write-Host "- Roll back: gcloud run revisions list --service=$Service --region=$Region, then"
    Write-Host "    gcloud run services update-traffic $Service --region=$Region --to-revisions=<revision>=100"
}
catch {
    Write-Host "Stopped: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Remove-Item $envFile -ErrorAction SilentlyContinue
    Pop-Location
}
