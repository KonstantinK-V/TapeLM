$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_417_densepin.out"
$wait_log = "results\_queue_416_stream.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "417 densepin: wait for 416 DONE, then teacher-ceiling only (no Phi)."

# Wait until 416 queue finished (or proceed if already done / missing).
for ($i = 0; $i -lt 180; $i++) {
  if (Test-Path $wait_log) {
    $tail = Get-Content $wait_log -Tail 3 -ErrorAction SilentlyContinue
    if ($tail -match "DONE 416 stream") { break }
  }
  Start-Sleep -Seconds 20
}
if (Test-Path $wait_log) {
  $tail = Get-Content $wait_log -Tail 5 -ErrorAction SilentlyContinue
  L "416 log tail: $($tail -join ' | ')"
}

L "==== _check417_densepin ===="
python _check417_densepin.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 417 densepin seed $s ===="
  python -u _audit417_densepin.py --seed $s
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict ===="
python results\_read417_densepin.py
L "DONE 417 densepin"
