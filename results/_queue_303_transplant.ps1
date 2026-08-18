$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$seeds = @(1337, 8642, 5200, 2718)
$wiki2 = "data\_stage254_news.txt"
New-Item -ItemType Directory -Force -Path results\minds | Out-Null

foreach ($s in $seeds) {
  Write-Host "==== 303_train seed $s ===="
  python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region `
    --objective reward --addresses 1500 --reach-max-q 2000 `
    --probe-period 1000 --probe-size 60 --cpu `
    --train-steps 4000 --seed $s `
    --save-mind "results\minds\mind_a1500_s$s.pt" `
    --run-tag "303_train_s$s"
  if ($LASTEXITCODE -ne 0) { throw "train seed $s failed exit $LASTEXITCODE" }
}

foreach ($s in $seeds) {
  Write-Host "==== 303_transplant news seed $s ===="
  python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region `
    --objective reward --addresses 1500 --reach-max-q 2000 --cpu `
    --wiki $wiki2 `
    --load-mind "results\minds\mind_a1500_s$s.pt" `
    --run-tag "303_transplant_news_s$s"
  if ($LASTEXITCODE -ne 0) { throw "transplant seed $s failed exit $LASTEXITCODE" }
}

Write-Host "==== _read299 303_transplant_news ===="
python _read299.py (Get-ChildItem results\stage289_decision_303_transplant_news_s*.json | ForEach-Object { $_.FullName }) --held
