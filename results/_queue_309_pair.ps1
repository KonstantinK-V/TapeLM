$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
New-Item -ItemType Directory -Force -Path out | Out-Null

Write-Host "==== preflight ===="
python _check301_wiring.py
if ($LASTEXITCODE -ne 0) { throw "wiring failed" }
python _check309_pair.py
if ($LASTEXITCODE -ne 0) { throw "pair check failed" }

# --out is not a flag in this tree; write via --run-tag then copy to the path HANDOFF/_read309 expect.
$seeds = @(1337, 8642, 2890, 4711)
foreach ($s in $seeds) {
  Write-Host "==== 309_pair seed $s ===="
  # Standing arm's speed flags: --frame-fp fillers (probe pack ~18s, not ~800s on address-fp)
  # and probe-period/size as in 304. --write-fp hash is required or pack crashes under --fp hash.
  & python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --tape-sample region --addresses 1500 --pair --frame-fp fillers `
    --probe-period 1000 --probe-size 60 --cpu `
    --train-steps 4000 --seed $s --run-tag "309_s$s"
  if ($LASTEXITCODE -ne 0) { throw "seed $s failed exit $LASTEXITCODE" }
  Copy-Item -Force "results\stage289_decision_309_s$s.json" "out\_stage289_decision_309_s$s.json"
}
Write-Host "==== _read309 ===="
python _read309.py (Get-ChildItem out\_stage289_decision_309_s*.json | ForEach-Object { $_.FullName }) --held
Write-Host "==== 309 done ===="
