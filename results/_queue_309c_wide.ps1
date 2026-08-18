$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
New-Item -ItemType Directory -Force -Path out | Out-Null
Write-Host "==== 309c_wide seed 1337 ===="
& python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
  --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
  --tape-sample region --addresses 1500 --pair --frame-fp fillers `
  --pair-cands 32 `
  --probe-period 1000 --probe-size 60 --cpu `
  --train-steps 4000 --seed 1337 --run-tag 309c_wide_s1337
if ($LASTEXITCODE -ne 0) { throw "309c failed $LASTEXITCODE" }
Copy-Item -Force results\stage289_decision_309c_wide_s1337.json out\_stage289_decision_309c_wide_s1337.json
Write-Host "==== _read309 309c vs 309b seq 1337 ===="
python _read309.py results\stage289_decision_309b_seq_s1337.json results\stage289_decision_309c_wide_s1337.json --held
Write-Host "==== 309c done ===="
