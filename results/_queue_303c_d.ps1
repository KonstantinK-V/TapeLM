$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$seeds = @(1337, 8642, 5200, 2718)

# 1) TinyStories transplant — fiction domain, same wiki-trained minds
foreach ($s in $seeds) {
  Write-Host "==== 303c_transplant tinystories seed $s ===="
  python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region `
    --objective reward --addresses 1500 --reach-max-q 2000 --cpu `
    --wiki data\_tinystories_raw_scale.txt --seed $s --train-steps 0 `
    --load-mind "results\minds\mind_a1500_s$s.pt" `
    --run-tag "303c_transplant_stories_s$s"
  if ($LASTEXITCODE -ne 0) { throw "stories transplant seed $s failed exit $LASTEXITCODE" }
}

# 2) Negative: fresh untrained Phi on news
foreach ($s in $seeds) {
  Write-Host "==== 303d_random news seed $s ===="
  python -u _stage289_derivation.py --tape frames --frame-max 3 --min-mentions 1 `
    --fp hash --write-fp hash --ink mean --write-ink mean --words ascii `
    --reach --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region `
    --objective reward --addresses 1500 --reach-max-q 2000 --cpu `
    --wiki data\_stage254_news.txt --seed $s --train-steps 0 `
    --run-tag "303d_random_news_s$s"
  if ($LASTEXITCODE -ne 0) { throw "random news seed $s failed exit $LASTEXITCODE" }
}

Write-Host "==== _read299 303c stories ===="
python _read299.py (Get-ChildItem results\stage289_decision_303c_transplant_stories_s*.json | ForEach-Object { $_.FullName }) --held
Write-Host "==== _read299 303d random news ===="
python _read299.py (Get-ChildItem results\stage289_decision_303d_random_news_s*.json | ForEach-Object { $_.FullName }) --held
