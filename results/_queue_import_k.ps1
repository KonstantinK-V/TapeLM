$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_import_k.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "import-k ablation: exact2 (k=0 control) then import2 (k=2); 289 only, --no-ladder"

L "289 exact2 --import-k 0"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --no-ladder --import-k 0 --run-tag exact2 > results\_stage289_full_exact2.out 2>&1"
L "EXIT_exact2=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP"; exit $LASTEXITCODE }

L "289 import2 --import-k 2"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --no-ladder --import-k 2 --run-tag import2 > results\_stage289_full_import2.out 2>&1"
L "EXIT_import2=$LASTEXITCODE"
L "QUEUE_DONE"
exit $LASTEXITCODE
