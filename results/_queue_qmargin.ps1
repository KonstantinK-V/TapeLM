$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_qmargin.out"
$env:PYTHONIOENCODING = "utf-8"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "qmargin v28: confidence feature (scale-free gap); same corpus as qrank_big; decide high-margin z vs -3.61"

# same recipe as qrank_big so the only variable is the new node feature / verdict
$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100"

L "289 qmargin"
cmd /c "python -u _stage289_derivation.py $base --run-tag qmargin > results\_stage289_full_qmargin.out 2>&1"
L "EXIT_qmargin=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
L "QUEUE_DONE"
exit 0
