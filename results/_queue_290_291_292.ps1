$ErrorActionPreference = "Continue"
$log = "results\_queue_290_291_292.out"
"===== 290/291/292 $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

# --- offline ---
L "check290_sparse"
cmd /c "python -u _check290_sparse.py > results\_check290_sparse.out 2>&1"
L "EXIT_check=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after check"; exit $LASTEXITCODE }

# --- smokes (full path, tiny corpus) ---
$smoke = "--train-steps 1 --addresses 200 --train-lines 20000 --eval-lines 10000 --fp hash --write-fp hash --address-tau 0.4712 --no-ladder"

L "smoke290"
cmd /c "python -u _stage289_derivation.py $smoke --min-mentions 1 --import-k 0 --neighbours 3 --run-tag smoke290 > results\_stage289_full_smoke290.out 2>&1"
L "EXIT_smoke290=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after smoke290"; exit $LASTEXITCODE }

L "smoke291"
cmd /c "python -u _stage289_derivation.py $smoke --min-mentions 1 --import-k 0 --neighbours 3 --refuse --run-tag smoke291 > results\_stage289_full_smoke291.out 2>&1"
L "EXIT_smoke291=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after smoke291"; exit $LASTEXITCODE }

L "smoke292"
cmd /c "python -u _stage289_derivation.py --train-steps 1 --addresses 200 --train-lines 20000 --eval-lines 10000 --fp hash --write-fp hash --address-tau 0.4712 --min-mentions 2 --import-k 2 --run-tag smoke292 > results\_stage289_full_smoke292.out 2>&1"
L "EXIT_smoke292=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after smoke292"; exit $LASTEXITCODE }

L "check290_sparse (again after smokes)"
cmd /c "python -u _check290_sparse.py > results\_check290_sparse_after.out 2>&1"
L "EXIT_check2=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after check2"; exit $LASTEXITCODE }

# --- full wiki ---
$full = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --probe-period 100 --fp hash --write-fp hash --address-tau 0.4712"

L "290_sparse"
cmd /c "python -u _stage289_derivation.py $full --min-mentions 1 --import-k 0 --neighbours 3 --no-ladder --run-tag 290_sparse > results\_stage289_full_290_sparse.out 2>&1"
L "EXIT_290=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after 290"; exit $LASTEXITCODE }

L "291_refuse"
cmd /c "python -u _stage289_derivation.py $full --min-mentions 1 --import-k 0 --neighbours 3 --no-ladder --refuse --run-tag 291_refuse > results\_stage289_full_291_refuse.out 2>&1"
L "EXIT_291=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after 291"; exit $LASTEXITCODE }

L "292_open"
cmd /c "python -u _stage289_derivation.py $full --min-mentions 2 --import-k 2 --run-tag 292_open > results\_stage289_full_292_open.out 2>&1"
L "EXIT_292=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after 292"; exit $LASTEXITCODE }

L "QUEUE_DONE"
exit 0
