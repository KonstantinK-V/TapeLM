$ErrorActionPreference = "Continue"
$log = Join-Path $PSScriptRoot "_queue_289_pair_289c.out"
Set-Location (Split-Path $PSScriptRoot -Parent)
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
"" | Set-Content $log -Encoding utf8
L "QUEUE start 289 pair then 289c"

L "STEP4a 289 full"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 > results\_stage289_full_w150.out 2>&1"
L "EXIT_4a=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP"; exit $LASTEXITCODE }

L "STEP4b 289 no-derivation"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 --no-derivation > results\_stage289_full_w150_noderiv.out 2>&1"
L "EXIT_4b=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP"; exit $LASTEXITCODE }

L "STEP5 289c"
cmd /c "python -u _stage289c_audit.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 > results\_stage289c_full_w150.out 2>&1"
L "EXIT_5=$LASTEXITCODE"
L "QUEUE_DONE"
exit $LASTEXITCODE
