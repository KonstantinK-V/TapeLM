$ErrorActionPreference = "Continue"
$log = "results\_queue_night_291_292.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

L 'night queue v43: wait 292 then audit1200, 290b_dense, audit2400 (no 291)'

# --- wait for 292 ---
$pid292 = $null
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
  if ($_.CommandLine -match 'run-tag 292_open') { $script:pid292 = $_.ProcessId }
}
if ($pid292) {
  L "waiting 292_open pid=$pid292"
  try { Wait-Process -Id $pid292 -ErrorAction SilentlyContinue } catch {}
} else {
  L 'no live 292 python'
}
L '292 done or absent'

$baseHash = "--fp hash --write-fp hash --address-tau 0.4712 --import-k 0 --no-ladder --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000"

# 1) audit on 290 tape (~5 min tape build)
L 'audit1200'
cmd /c "python -u _stage289_derivation.py --train-steps 1 --addresses 1200 $baseHash --min-mentions 1 --neighbours 3 --run-tag audit1200 > results\_stage289_full_audit1200.out 2>&1"
L "EXIT_audit1200=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L 'STOP after audit1200'; exit $LASTEXITCODE }

# 2) 290b dense min-mentions 2 (~1h) — powered population
L '290b_dense'
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --probe-period 100 $baseHash --min-mentions 2 --neighbours 3 --run-tag 290b_dense > results\_stage289_full_290b_dense.out 2>&1"
L "EXIT_290b_dense=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L 'STOP after 290b_dense'; exit $LASTEXITCODE }

# 3) audit scale at 2400 addresses (~20-30 min)
L 'audit2400'
cmd /c "python -u _stage289_derivation.py --train-steps 1 --addresses 2400 $baseHash --min-mentions 1 --neighbours 3 --run-tag audit2400 > results\_stage289_full_audit2400.out 2>&1"
L "EXIT_audit2400=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L 'STOP after audit2400'; exit $LASTEXITCODE }

L 'QUEUE_DONE night v43'
exit 0
