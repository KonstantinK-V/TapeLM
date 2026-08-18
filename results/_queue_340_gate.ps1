$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_340_gate.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "340: same four-seed 336news reach - GATE-WO, RANK right, composition, gain/random (no new flag)"

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu','--train-steps','4000'
)
$seeds = @(1337, 8642, 2890, 4711)
$news = "data\_stage254_news.txt"

L "==== BLOCK 340: re-run 336news x4 (native news + wiki rival) ===="
foreach ($s in $seeds) {
  $tag = "336news_s$s"
  $out = "out/_stage289_decision_$tag.json"
  $rival = "minds/320_wiki_s$s.pt"
  if (-not (Test-Path $rival)) { L "SKIP $tag no rival $rival"; continue }
  L "==== $tag rival=$rival ===="
  python -u _stage289_derivation.py @base --wiki $news --seed $s --run-tag $tag `
    --rival-mind $rival --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

$f = @(Get-ChildItem out\_stage289_decision_336news_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($f.Count -gt 0) {
  L "==== _read299 340 GATE-WO / RANK right (held) ===="
  python _read299.py @f --held | Tee-Object -FilePath results\_read340_held.txt
}

L "DONE 340"
