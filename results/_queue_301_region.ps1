$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$seeds = @(1337, 8642, 5200, 2718)
foreach ($s in $seeds) {
  Write-Host "==== 301_region seed $s ===="
  python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers `
    --tape-sample region `
    --objective reward --addresses 3000 --reach-max-q 2000 `
    --probe-period 1000 --probe-size 60 --cpu `
    --train-steps 4000 --seed $s --run-tag "301_region_s$s"
  if ($LASTEXITCODE -ne 0) { throw "seed $s failed exit $LASTEXITCODE" }
}
Write-Host "==== _read299 301_region ===="
python _read299.py (Get-ChildItem results\stage289_decision_301_region_s*.json | ForEach-Object { $_.FullName }) --held
