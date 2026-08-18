$ErrorActionPreference = "Continue"
$log = "results\_queue_289_tau_batch.out"
"===== 289 tau batch resume $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100"

# --- wait for in-flight region3 (with row-dropout 0.35), if still running ---
$waitPid = $env:REGION3_WAIT_PID
if ($waitPid) {
  L "wait for region3 pid=$waitPid (row-dropout run) to finish"
  try {
    Wait-Process -Id ([int]$waitPid) -ErrorAction SilentlyContinue
  } catch {}
  L "region3 pid=$waitPid done"
}

# archive the dropout run's decision so the clean tag can overwrite
$src = "results\stage289_decision_region3.json"
$dst = "results\stage289_decision_region3_rowdrop35.json"
if (Test-Path $src) {
  Copy-Item -Force $src $dst
  L "copied $src -> $dst"
} else {
  L "WARN: no $src to copy (dropout run may have failed before writing decision)"
}
if (Test-Path "results\_stage289_full_region3.out") {
  Copy-Item -Force "results\_stage289_full_region3.out" "results\_stage289_full_region3_rowdrop35.out"
  L "copied region3 out -> region3_rowdrop35.out"
}
if (Test-Path "results\_stage289_log_region3.txt") {
  Copy-Item -Force "results\_stage289_log_region3.txt" "results\_stage289_log_region3_rowdrop35.txt"
}

# 1) region3 clean — same as prior comparable run (rate 0.0), no row-dropout
L "289 region3 --views 3 --view-mode region (no row-dropout)"
cmd /c "python -u _stage289_derivation.py $base --views 3 --view-mode region --run-tag region3 > results\_stage289_full_region3.out 2>&1"
L "EXIT_region3=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after region3"; exit $LASTEXITCODE }

# 2-3) read-ink arms — same tape as maxpool, one variable each
L "289 ink_hash --fp hash"
cmd /c "python -u _stage289_derivation.py $base --fp hash --run-tag ink_hash > results\_stage289_full_ink_hash.out 2>&1"
L "EXIT_ink_hash=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after ink_hash"; exit $LASTEXITCODE }

L "289 ink_bigram --ink bigram"
cmd /c "python -u _stage289_derivation.py $base --ink bigram --run-tag ink_bigram > results\_stage289_full_ink_bigram.out 2>&1"
L "EXIT_ink_bigram=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP after ink_bigram"; exit $LASTEXITCODE }

# 4) tau calibration smoke before write arms (~5 min)
L "289 tau_smoke --addresses 200 --train-lines 20000 --train-steps 1 --fp hash --write-fp hash --tau-mode density"
cmd /c "python -u _stage289_derivation.py --addresses 200 --train-lines 20000 --train-steps 1 --fp hash --write-fp hash --tau-mode density --run-tag tau_smoke > results\_stage289_full_tau_smoke.out 2>&1"
L "EXIT_tau_smoke=$LASTEXITCODE"
# smoke may exit non-zero on too-few questions; mechanism line still required
$smoke = Get-Content "results\_stage289_full_tau_smoke.out" -Encoding utf8 -ErrorAction SilentlyContinue
$cal = $smoke | Select-String -Pattern "tau calibrated:" | Select-Object -Last 1
if ($cal) { L "SMOKE_CAL: $($cal.Line.Trim())" } else { L "STOP: no tau calibrated line in smoke"; exit 2 }
if ($smoke | Select-String -Pattern "density NOT decreasing|outside the bracket") {
  L "STOP: tau smoke reported NOT decreasing or outside the bracket"
  exit 3
}
L "tau smoke mechanism OK (density convergence logged); tau value NOT reused on full corpus"

# 5-6) write arms — compare frag to 0.0785 / 1.0900 and to matching ink arm; not Phi vs scoreboard
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
