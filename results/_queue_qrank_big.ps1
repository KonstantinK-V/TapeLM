$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_qrank_big.out"
$env:PYTHONIOENCODING = "utf-8"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "qrank_big v27: underpowered-gate guard + resample overlap; full wiki; train/eval caps 200k/100k; period 100"

# full wikitext file so train-lines/eval-lines are not silently truncated by the 150MB slice
$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100"

L "289 qrank_big"
cmd /c "python -u _stage289_derivation.py $base --run-tag qrank_big > results\_stage289_full_qrank_big.out 2>&1"
L "EXIT_qrank_big=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
L "QUEUE_DONE"
exit 0
