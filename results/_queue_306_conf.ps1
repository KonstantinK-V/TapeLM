$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly

# Wait until the live 304 queue (if any) is no longer training.
while ($true) {
  $busy = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -eq "python.exe" -and
      $_.CommandLine -and
      ($_.CommandLine -like "*_stage289_derivation.py*")
    }
  if (-not $busy) { break }
  Write-Host ("==== wait 304 still running pid={0} ====" -f ($busy.ProcessId -join ","))
  Start-Sleep -Seconds 60
}

Write-Host "==== preflight _check301_wiring ===="
python _check301_wiring.py
if ($LASTEXITCODE -ne 0) { throw "wiring failed" }

$seeds = @(1337, 8642, 5200, 2718)
foreach ($s in $seeds) {
  Write-Host "==== 306_conf seed $s ===="
  & python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region `
    --reach-confirm --conf-window 1 `
    --objective reward --addresses 1500 --reach-max-q 2000 `
    --probe-period 1000 --probe-size 60 --cpu `
    --train-steps 4000 --seed $s --run-tag "306_conf_s$s"
  if ($LASTEXITCODE -ne 0) { throw "seed $s failed exit $LASTEXITCODE" }
}
Write-Host "==== _read299 306_conf ===="
python _read299.py (Get-ChildItem results\stage289_decision_306_conf_s*.json | ForEach-Object { $_.FullName }) --held
Write-Host "==== 306_conf done ===="
