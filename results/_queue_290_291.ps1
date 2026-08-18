$ErrorActionPreference = "Continue"
$log = "results\_queue_290_291.out"
"===== 290/291 queue $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100 --fp hash --write-fp hash --tau-mode density --neighbours 3"

L "289/290_sparse $base --run-tag 290_sparse"
cmd /c "python -u _stage289_derivation.py $base --run-tag 290_sparse > results\_stage289_full_290_sparse.out 2>&1"
L "EXIT_290_sparse=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after 290_sparse"; exit $LASTEXITCODE }

L "289/291_refuse $base --refuse --run-tag 291_refuse"
cmd /c "python -u _stage289_derivation.py $base --refuse --run-tag 291_refuse > results\_stage289_full_291_refuse.out 2>&1"
L "EXIT_291_refuse=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after 291_refuse"; exit $LASTEXITCODE }

L "QUEUE_DONE"
exit 0
