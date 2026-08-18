$ErrorActionPreference = "Continue"
$log = "results\_queue_289_tau_batch.out"
"===== 289 tau batch from ink_hash $(Get-Date -Format o) =====" | Add-Content $log -Encoding utf8
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100"

# region3 clean already EXIT=0; resume from ink_hash
L "289 ink_hash --fp hash"
cmd /c "python -u _stage289_derivation.py $base --fp hash --run-tag ink_hash > results\_stage289_full_ink_hash.out 2>&1"
L "EXIT_ink_hash=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after ink_hash"; exit $LASTEXITCODE }

L "289 ink_bigram --ink bigram"
cmd /c "python -u _stage289_derivation.py $base --ink bigram --run-tag ink_bigram > results\_stage289_full_ink_bigram.out 2>&1"
L "EXIT_ink_bigram=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after ink_bigram"; exit $LASTEXITCODE }

L "289 tau_smoke --addresses 200 --train-lines 20000 --train-steps 1 --fp hash --write-fp hash --tau-mode density"
cmd /c "python -u _stage289_derivation.py --addresses 200 --train-lines 20000 --train-steps 1 --fp hash --write-fp hash --tau-mode density --run-tag tau_smoke > results\_stage289_full_tau_smoke.out 2>&1"
L "EXIT_tau_smoke=$LASTEXITCODE"
$smoke = Get-Content "results\_stage289_full_tau_smoke.out" -Encoding utf8 -ErrorAction SilentlyContinue
$cal = $smoke | Select-String -Pattern "tau calibrated:" | Select-Object -Last 1
if ($cal) { L "SMOKE_CAL: $($cal.Line.Trim())" } else { L "STOP: no tau calibrated line in smoke"; exit 2 }
if ($smoke | Select-String -Pattern "density NOT decreasing|outside the bracket") {
  L "STOP: tau smoke reported NOT decreasing or outside the bracket"
  exit 3
}
L "tau smoke mechanism OK"

L "289 write_hash --fp hash --write-fp hash --tau-mode density"
cmd /c "python -u _stage289_derivation.py $base --fp hash --write-fp hash --tau-mode density --run-tag write_hash > results\_stage289_full_write_hash.out 2>&1"
L "EXIT_write_hash=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after write_hash"; exit $LASTEXITCODE }

L "289 write_bigram --ink bigram --write-ink bigram --tau-mode density"
cmd /c "python -u _stage289_derivation.py $base --ink bigram --write-ink bigram --tau-mode density --run-tag write_bigram > results\_stage289_full_write_bigram.out 2>&1"
L "EXIT_write_bigram=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after write_bigram"; exit $LASTEXITCODE }

L "QUEUE_DONE"
exit 0
