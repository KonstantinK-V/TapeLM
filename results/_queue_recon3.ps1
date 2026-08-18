$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_recon3.out"
$env:PYTHONIOENCODING = "utf-8"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "recon3 v32: views=3 row-dropout=0.35; read held_pooled_vs_single -> held_d_auc -> refusal"

$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100 --views 3 --row-dropout 0.35"

L "289 recon3"
cmd /c "python -u _stage289_derivation.py $base --run-tag recon3 > results\_stage289_full_recon3.out 2>&1"
L "EXIT_recon3=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
L "QUEUE_DONE"
exit 0
