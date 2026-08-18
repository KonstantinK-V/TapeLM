$ErrorActionPreference = "Continue"
$log = "results\_queue_night_291_292.out"
"===== night after 291 skip $(Get-Date -Format o) =====" | Add-Content $log -Encoding utf8
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

# Confirmed: smoke291b unanswerable_rate=0.931 >= 0.75 → do NOT run full 291 tonight.
L "SKIP 291_refuse (unanswerable_rate 0.931). Morning: reward-280 objective."

L "smoke292b --open"
cmd /c "python -u _stage289_derivation.py --train-steps 1 --addresses 200 --train-lines 20000 --eval-lines 10000 --fp hash --write-fp hash --min-mentions 2 --address-tau 0.4712 --import-k 2 --open --run-tag smoke292b > results\_stage289_full_smoke292b.out 2>&1"
L "EXIT_smoke292b=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after smoke292b"; exit $LASTEXITCODE }

$sum = (python results\_gate_read.py results\stage289_decision_smoke292b.json open_summary).Trim()
$nVal = (python results\_gate_read.py results\stage289_decision_smoke292b.json open_n).Trim()
L "smoke292b $sum  open.n=$nVal"

$go = (python -c "n=int('$nVal' or 0); print('YES' if n>0 else 'NO')").Trim()
if ($go -ne "YES") {
  L "SKIP full 292_open: open.n=$nVal"
  L "QUEUE_DONE"
  exit 0
}

L "GO full 292_open: open.n=$nVal"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --probe-period 100 --fp hash --write-fp hash --min-mentions 2 --address-tau 0.4712 --import-k 2 --open --run-tag 292_open > results\_stage289_full_292_open.out 2>&1"
L "EXIT_292=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after 292"; exit $LASTEXITCODE }

L "QUEUE_DONE"
exit 0
