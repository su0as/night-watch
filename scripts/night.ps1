# Opens the four processes for a watched night in separate PowerShell windows.
# Usage (from the repo root, venv already created):
#   .\scripts\night.ps1 -Name night-1 -Fault flip_bug -Until 07:30
# The windows inherit this shell's environment (PATH, caches), so run your env script first
# if you have one. `temporal` and `ollama` must be on PATH.
param(
  [string]$Name = "night-$(Get-Date -Format 'MMdd')",
  [string]$Fault = "none",
  [string]$Until = "07:30",
  [int]$Steps = 3000000,
  [string]$Preset = "zoo",
  [string]$Watcher = "nightwatch"    # rules | thresholds | tabpfn | vision | nightwatch
)
# ($Profile is a built-in PowerShell variable, so the watcher profile is called -Watcher here.)

$root = (Resolve-Path "$PSScriptRoot\..").Path
$activate = Join-Path $root ".venv\Scripts\Activate.ps1"
$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }

function Open-Window($title, $command) {
  $cmd = "`$host.UI.RawUI.WindowTitle = '$title'; Set-Location '$root'; . '$activate'; $command"
  Start-Process $shell -ArgumentList "-NoExit", "-Command", $cmd
}

foreach ($tool in "temporal", "ollama") {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
    Write-Warning "$tool not found on PATH. See docs/TESTING_WINDOWS.md, step 0."
  }
}

# Gemma needs Ollama. Start it if nothing answers on its port.
try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 3 | Out-Null }
catch {
  Write-Host "Starting Ollama..."
  Start-Process ollama -ArgumentList "serve" -WindowStyle Minimized
  Start-Sleep -Seconds 5
}

Open-Window "temporal" "temporal server start-dev --db-filename nightwatch.db"
Start-Sleep -Seconds 8
Open-Window "worker" "python -m nightwatch worker"
Open-Window "train $Name" "python -m nightwatch train --preset $Preset --steps $Steps --fault $Fault --name $Name"

Write-Host "Waiting for $Name to write its first metrics..."
$metrics = Join-Path $root "runs\$Name\metrics.jsonl"
while (-not (Test-Path $metrics)) { Start-Sleep -Seconds 5 }
Start-Sleep -Seconds 5
. $activate
python -m nightwatch start "runs\$Name" --until $Until --profile $Watcher
Write-Host "Watching. Temporal UI: http://localhost:8233   Good night."
