$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_345_cons64.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "345 capacity retry: --constrain dim 64, 4 seeds (after both gates FAIL at d=32)"

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--constrain','--dim','64','--train-steps','4000'
)
$seeds = @(1337, 8642, 2890, 4711)
$news = "data\_stage254_news.txt"

foreach ($s in $seeds) {
  $tag = "345cons64_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $news --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

$f = @(Get-ChildItem out\_stage289_decision_345cons64_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($f.Count -gt 0) {
  L "==== _read345_cons (d64) ===="
  python _read345_cons.py @f | Tee-Object -FilePath results\_read345_cons64.txt
}

L "DONE 345 constrain capacity retry d64"
