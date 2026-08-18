$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$seeds = @(1337, 8642, 5200, 2718)
$wiki2 = "data\_stage254_news.txt"

foreach ($s in $seeds) {
  Write-Host "==== 303b_transplant news seed $s ===="
  python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region `
    --objective reward --addresses 1500 --reach-max-q 2000 --cpu `
    --wiki $wiki2 --seed $s `
    --load-mind "results\minds\mind_a1500_s$s.pt" `
    --run-tag "303b_transplant_news_s$s"
  if ($LASTEXITCODE -ne 0) { throw "transplant seed $s failed exit $LASTEXITCODE" }
}

Write-Host "==== _read299 303b_transplant_news ===="
python _read299.py (Get-ChildItem results\stage289_decision_303b_transplant_news_s*.json | ForEach-Object { $_.FullName }) --held
