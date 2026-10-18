<#
Serve StutterClear to the public for free from this laptop: the app in public mode plus Tailscale
Funnel, which gives a stable https://<machine>.<tailnet>.ts.net link (browsers only allow the
microphone on HTTPS pages). Full features on this laptop's GPU; nothing to pay.

    powershell -ExecutionPolicy Bypass -File scripts\serve_public.ps1 [-Port 8766] [-UclassDemos]

Needs Tailscale installed and logged in (tailscale up); the first run prints a link to allow Funnel
for this machine. Keys (NEBIUS_API_KEY, NVIDIA_API_KEY) come from your environment, never from this
file. Ctrl+C stops both. Keep the laptop plugged in with sleep off while people use the link.
#>
param([int]$Port = 8766, [switch]$UclassDemos)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command tailscale -ErrorAction SilentlyContinue)) {
    throw "Tailscale is not installed: get it from https://tailscale.com/download, then run 'tailscale up'."
}
foreach ($k in "NEBIUS_API_KEY", "NVIDIA_API_KEY") {
    if (-not [Environment]::GetEnvironmentVariable($k)) {
        Write-Warning "$k is not set here: the features that need it stay hidden."
    }
}

$env:STUTTERCLEAR_HOSTED = "1"  # visitors use their own mic; nothing they say is written to disk
if ($UclassDemos) {
    $env:STUTTERCLEAR_UCLASS_DEMOS = "1"  # UCLASS is research-only: private judging links only
}

$app = Start-Process python -ArgumentList "scripts/run_app.py", "--no-browser", "--port", $Port, "--strict-port" `
    -WorkingDirectory $root -NoNewWindow -PassThru
Start-Sleep -Seconds 3
if ($app.HasExited) {
    throw "The app did not start (is port $Port taken by another StutterClear?)."
}
try {
    Write-Host "StutterClear (public mode) on http://127.0.0.1:$Port. Opening the funnel: the public link follows."
    tailscale funnel $Port
}
finally {
    Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
