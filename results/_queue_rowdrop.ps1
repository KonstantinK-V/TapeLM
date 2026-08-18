$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_mean_matched.out"
$env:PYTHONIOENCODING = "utf-8"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

# Winning body = maxpool (mean_matched showed capacity alone does not move high-margin).
# Same corpus; row-dropout only at train time.
$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100"

L "289 rowdrop04 --row-dropout 0.4 (on maxpool body; after mean_matched closed capacity confound)"
cmd /c "python -u _stage289_derivation.py $base --row-dropout 0.4 --run-tag rowdrop04 > results\_stage289_full_rowdrop04.out 2>&1"
L "EXIT_rowdrop04=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
L "QUEUE_DONE_ROWDROP"
exit 0
