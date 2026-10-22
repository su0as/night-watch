# Try to break the overnight watch, on purpose, during a real training run.
# Each scenario is logged with timestamps; the Temporal history is exported to docs/evidence/.
#   pwsh -File scripts\break_tests.ps1            (Temporal dev server, Ollama and the venv must be set up)
# Scenarios:
#   A  kill the worker mid-watch, restart it 3 minutes later
#   B  kill the Temporal server mid-watch (on-disk history), restart it 1 minute later
#   C  kill Ollama while Gemma is looking, restart it 90 s later
param(
  [string]$Name = "break-$(Get-Date -Format 'MMdd-HHmm')",
  [string]$TemporalCmd = "temporal server start-dev --db-filename nightwatch.db",
  [string]$OllamaCmd = "ollama serve",
  [string]$EnvScript = ""   # optional script that sets PATH/caches for child processes (they start clean)
)
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $root
$ev = Join-Path $root "docs\evidence"
New-Item -ItemType Directory -Force $ev | Out-Null
$log = Join-Path $ev "$Name.log"
$pw = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pw) { $pw = (Get-Command powershell).Source }
$activate = Join-Path $root ".venv\Scripts\Activate.ps1"
function Say($m) { "[$(Get-Date -Format HH:mm:ss)] $m" | Tee-Object -Append -FilePath $log }
function Detached($tag, $cmd) {
  $pre = if ($EnvScript) { ". '$EnvScript' | Out-Null; " } else { "" }
  $full = "`"$pw`" -NoProfile -Command `"$pre Set-Location '$root'; . '$activate'; `$env:PYTHONUNBUFFERED='1'; $cmd *>> '$ev\$Name-$tag.log'`""
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $full; CurrentDirectory = $root }
  Say "started $tag (pid $($r.ProcessId))"
}
function Kill-Matching($pattern, $what) {
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pattern } | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Say "killed $what ($($_.Name) $($_.ProcessId))" }
}
function Kill-Name($names, $what) {
  Get-Process $names -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue; Say "killed $what ($($_.Name) $($_.Id))" }
}
function Start-Service-Detached($cmd, $what) {
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c $cmd"; CurrentDirectory = $root }
  Say "restarted $what (pid $($r.ProcessId))"
}
. $activate
$wf = "nightwatch-$Name"

Say "=== break tests on run $Name (workflow $wf) ==="
Detached "train" "python -m nightwatch train --steps 500000 --fault flip_bug --seed 31 --name $Name --frame-every 20000 --threads 2"
while (-not (Test-Path "runs\$Name\metrics.jsonl")) { Start-Sleep 5 }
Detached "worker1" "python -m nightwatch worker"
Start-Sleep 60
python -m nightwatch start "runs\$Name" --until +35m --every 60 --profile nightwatch 2>&1 | ForEach-Object { Say $_ }

Start-Sleep 240
Say "--- A: kill the worker ---"
python -m nightwatch status "runs\$Name" 2>&1 | ForEach-Object { Say "status before A: $_" }
Kill-Matching '*nightwatch worker*' 'worker'
Start-Sleep 180
Detached "worker2" "python -m nightwatch worker"
Start-Sleep 150

Say "--- B: kill the Temporal server ---"
Kill-Name @('temporal') 'Temporal server'
Start-Sleep 60
Start-Service-Detached $TemporalCmd 'Temporal server'
Start-Sleep 30
if (-not (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*nightwatch worker*' })) {
  Say "worker exited while the server was down; starting it again"
  Detached "worker3" "python -m nightwatch worker"
}
Start-Sleep 150
python -m nightwatch status "runs\$Name" 2>&1 | ForEach-Object { Say "status after B: $_" }

Say "--- C: kill Ollama while Gemma is looking ---"
Kill-Name @('ollama', 'llama-server') 'Ollama'
Start-Sleep 90
Start-Service-Detached $OllamaCmd 'Ollama'
Start-Sleep 240
python -m nightwatch status "runs\$Name" 2>&1 | ForEach-Object { Say "status after C: $_" }

Say "--- waiting for the watch to finish ---"
for ($i = 0; $i -lt 60; $i++) {
  $st = temporal workflow describe --workflow-id $wf 2>$null | Select-String 'Status' | Select-Object -First 1
  if ($st -and $st.Line -notmatch 'Running') { break }
  Start-Sleep 30
}
temporal workflow show --workflow-id $wf --output json > "$ev\$Name-history.json" 2>$null
temporal workflow show --workflow-id $wf > "$ev\$Name-history.txt" 2>$null
if (Test-Path "runs\$Name\report\morning.md") { Copy-Item "runs\$Name\report\morning.md" "$ev\$Name-morning.md" -Force }
Kill-Matching '*nightwatch worker*' 'worker (end of test)'
Say "=== done; evidence in docs\evidence\$Name-* ==="
