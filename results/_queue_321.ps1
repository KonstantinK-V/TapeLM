$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_321.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "321: bisect x4 wiki, then finetune news x4 + tiny x4 (separate arm, not pooled with 320)"

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
  $tag = "321bis_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag --bisect ===="
  python -u _stage289_derivation.py @common --bisect --train-steps 4000 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
  L "OK $tag"
}

foreach ($s in $seeds) {
  $tag = "321ft_news_s$s"
  $mind = "minds/320_wiki_s$s.pt"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag --finetune load $mind wiki=$news ===="
  python -u _stage289_derivation.py @common --load-mind $mind --wiki $news --finetune --train-steps 4000 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
  L "OK $tag"
}

foreach ($s in $seeds) {
  $tag = "321ft_tiny_s$s"
  $mind = "minds/320_wiki_s$s.pt"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag --finetune load $mind wiki=$tiny ===="
  python -u _stage289_derivation.py @common --load-mind $mind --wiki $tiny --finetune --train-steps 4000 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
  L "OK $tag"
}

L "==== _read299 bisect ===="
python _read299.py (Get-ChildItem out\_stage289_decision_321bis_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "==== _read299 ft news (own arm) ===="
python _read299.py (Get-ChildItem out\_stage289_decision_321ft_news_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "==== _read299 ft tiny (own arm) ===="
python _read299.py (Get-ChildItem out\_stage289_decision_321ft_tiny_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "DONE"
