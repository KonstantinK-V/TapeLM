# If main ablation queue looks for decision_reexam_frozen.json but new code
# writes decision_reexam_frozen_frozen.json, launch hid when frozen recovers.
$ErrorActionPreference = "Continue"
$log = Join-Path $PSScriptRoot "_queue_286_ablation_arm3fix.out"
"===== ARM3 FIX WATCHER $(Get-Date -Format o) =====" | Set-Content $log
Set-Location (Split-Path $PSScriptRoot -Parent)

$fj = "results\stage286_decision_reexam_frozen_frozen.json"
$alt = "results\stage286_decision_reexam_frozen.json"

while (-not (Test-Path $fj) -and -not (Test-Path $alt)) {
  $alive = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match '_stage286_evidence|_queue_286_ablation' }
  if (-not $alive) {
    Start-Sleep -Seconds 45
    break
  }
  Start-Sleep -Seconds 60
}

$path = if (Test-Path $fj) { $fj } elseif (Test-Path $alt) { $alt } else { $null }
if (-not $path) {
  "NO_FROZEN_JSON" | Tee-Object -FilePath $log -Append
  exit 1
}
"FOUND $path" | Tee-Object -FilePath $log -Append
$dec = Get-Content $path -Raw | ConvertFrom-Json
$g = $dec.gates
$recovered = [bool]($g.G_learns_evidence -and $g.G_survives_lie)
"FROZEN_RECOVERED:$recovered learns=$($g.G_learns_evidence) lie=$($g.G_survives_lie) overall=$($dec.overall)" |
  Tee-Object -FilePath $log -Append

while (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'frozen-trunk' }) {
  Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 15

$hid_running = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match '_stage286_evidence.py' -and $_.CommandLine -match 'reexam_hid' }
if ($hid_running) {
  "HID_ALREADY_STARTED_BY_MAIN_QUEUE" | Tee-Object -FilePath $log -Append
  exit 0
}
if (Test-Path "results\stage286_decision_reexam_hid.json") {
  "HID_ALREADY_DONE" | Tee-Object -FilePath $log -Append
  exit 0
}
if (-not $recovered) {
  "ARM3_SKIP_OK" | Tee-Object -FilePath $log -Append
  exit 0
}

"ARM3_LAUNCH_HID" | Tee-Object -FilePath $log -Append
python -u _stage286_evidence.py --train-steps 6000 --min-mentions 2 --run-tag reexam_hid 2>&1 |
  Tee-Object -FilePath results\_stage286_reexam_hid.out |
  Tee-Object -FilePath $log -Append
"EXIT_HID:$LASTEXITCODE" | Tee-Object -FilePath $log -Append
