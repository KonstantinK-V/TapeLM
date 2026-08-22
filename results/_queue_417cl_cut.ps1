$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_417cl_cut.out"
$wait_log = "results\_queue_417_densepin.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "417cl cut: wait for 417 densepin DONE. Decoy ceiling; read HARD then SIGNAL then ROOM."

for ($i = 0; $i -lt 240; $i++) {
  if (Test-Path $wait_log) {
    $tail = Get-Content $wait_log -Tail 5 -ErrorAction SilentlyContinue
    if ($tail -match "DONE 417 densepin") { break }
  }
  Start-Sleep -Seconds 20
}
if (Test-Path $wait_log) {
  $tail = Get-Content $wait_log -Tail 3 -ErrorAction SilentlyContinue
  L "417 log tail: $($tail -join ' | ')"
}

L "==== _check417cl_cut ===="
python _check417cl_cut.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 417cl cut seed $s ===="
  python -u _audit417cl_cut.py --seed $s
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict (HARD -> SIGNAL -> ROOM, per seed) ===="
python results\_read417cl_cut.py
L "DONE 417cl cut"
