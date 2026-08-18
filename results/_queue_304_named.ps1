$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
python _check301_wiring.py
if ($LASTEXITCODE -ne 0) { throw "wiring failed" }
$seeds = @(1337, 8642, 5200, 2718)
foreach ($s in $seeds) {
  Write-Host "==== 304_line seed $s ===="
  & python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region `
    --reach-line `
    --objective reward --addresses 1500 --reach-max-q 2000 `
    --probe-period 1000 --probe-size 60 --cpu `
    --train-steps 4000 --seed $s --run-tag "304_line_s$s"
  if ($LASTEXITCODE -ne 0) { throw "seed $s failed exit $LASTEXITCODE" }
}
Write-Host "==== _read299 304_line ===="
python _read299.py (Get-ChildItem results\stage289_decision_304_line_s*.json | ForEach-Object { $_.FullName }) --held
Write-Host "==== 304_line named-report done ===="
