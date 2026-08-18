$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_372_compass.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "372 compass: standing arm + --connect --reach-compass m, m in share1/rare/cover/jaccard, 4 seeds; ctrl=365conn (cos). common only if one wins. READ hit_rate then hit/reach; watch reachable"

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1','--connect'
)
$seeds = @(1337, 8642, 2890, 4711)
$members = @('share1', 'rare', 'cover', 'jaccard')
$wiki = "data\_wikitext103_train.txt"

foreach ($m in $members) {
  foreach ($s in $seeds) {
    $tag = "372${m}_s$s"
    $out = "out/_stage289_decision_$tag.json"
    L "==== $tag ===="
    python -u _stage289_derivation.py @base --reach-compass $m --wiki $wiki --seed $s --run-tag $tag --out $out
    if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
  }
  L "==== _read299 372$m ===="
  $f = @(Get-ChildItem out\_stage289_decision_372${m}_s*.json,results\stage289_decision_372${m}_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
  if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_372$m.txt }
}

L "==== _read299 365conn control ===="
$f = @(Get-ChildItem out\_stage289_decision_365conn_s*.json,results\stage289_decision_365conn_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_365conn_ctrl372.txt }

L "DONE 372 compass"
