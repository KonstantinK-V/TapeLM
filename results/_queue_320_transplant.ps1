$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_320_transplant.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
New-Item -ItemType Directory -Force -Path minds | Out-Null

L "320 transplant: wiki save-mind x4, then news + tinystories load-mind (1 mind : 1 seed)"

$common = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu'
)
$seeds = @(1337, 8642, 2890, 4711)
$news = "data\_stage254_news.txt"
$tiny = "data\_tinystories_raw_scale.txt"

foreach ($s in $seeds) {
  $tag = "320wiki_s$s"
  $mind = "minds/320_wiki_s$s.pt"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag save-mind $mind ===="
  python -u _stage289_derivation.py @common --train-steps 4000 --seed $s --run-tag $tag --save-mind $mind --out $out
  if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
  L "OK $tag"
}

foreach ($s in $seeds) {
  $tag = "320news_s$s"
  $mind = "minds/320_wiki_s$s.pt"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag load-mind $mind wiki=$news ===="
  python -u _stage289_derivation.py @common --seed $s --run-tag $tag --load-mind $mind --wiki $news --out $out
  if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
  L "OK $tag"
}

foreach ($s in $seeds) {
  $tag = "320tiny_s$s"
  $mind = "minds/320_wiki_s$s.pt"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag load-mind $mind wiki=$tiny ===="
  python -u _stage289_derivation.py @common --seed $s --run-tag $tag --load-mind $mind --wiki $tiny --out $out
  if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
  L "OK $tag"
}

L "==== _read299 wiki ===="
python _read299.py (Get-ChildItem out\_stage289_decision_320wiki_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "==== _read299 news ===="
python _read299.py (Get-ChildItem out\_stage289_decision_320news_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "==== _read299 tiny ===="
python _read299.py (Get-ChildItem out\_stage289_decision_320tiny_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "DONE"
