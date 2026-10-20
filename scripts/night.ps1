# Opens the four processes for a watched night in separate PowerShell windows.
# Usage (from the repo root, venv already created):
#   .\scripts\night.ps1 -Name night-1 -Fault flip_bug -Until 07:30
param(
  [string]$Name = "night-$(Get-Date -Format 'MMdd')",
  [string]$Fault = "none",
  [string]$Until = "07:30",
  [int]$Steps = 3000000,
  [string]$Preset = "zoo",
  [string]$Profile = "nightwatch"
)

$root = (Resolve-Path "$PSScriptRoot\..").Path
$activate = Join-Path $root ".venv\Scripts\Activate.ps1"

function Open-Window($title, $command) {
  $cmd = "`$host.UI.RawUI.WindowTitle = '$title'; Set-Location '$root'; . '$activate'; $command"
  Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd
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
python -m nightwatch start "runs\$Name" --until $Until --profile $Profile
Write-Host "Watching. Temporal UI: http://localhost:8233   Good night."
