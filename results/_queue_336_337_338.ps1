$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_336_337_338.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "336 native-news vs wiki-rival; 337 free on those runs; then 338 retain x4 rules on news with wiki mind"

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
# retain budget: half the usual 1500-place tape - matched N across rules
$RN = 750

# --- 336: native train on NEWS, wiki transplant as paired rival (depth 1 only) ---
L "==== BLOCK 336: native news + --rival-mind wiki320 (also yields 337 RANK) ===="
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
  L "==== _read299 336/337 ===="
  python _read299.py @f --held | Tee-Object -FilePath results\_read336_held.txt
}

# --- 338: retain rules on NEWS; mind rule uses frozen wiki mind ---
L "==== BLOCK 338: --retain $RN on news, rules random/own/share/mind ===="
foreach ($rule in @('random','own','share','mind')) {
  foreach ($s in $seeds) {
    $tag = "338${rule}_s$s"
    $out = "out/_stage289_decision_$tag.json"
    $mind = "minds/320_wiki_s$s.pt"
    L "==== $tag retain=$RN by=$rule ===="
    $extra = @('--retain', "$RN", '--retain-by', $rule, '--wiki', $news, '--train-steps', '0')
    if ($rule -eq 'mind') {
      if (-not (Test-Path $mind)) { L "SKIP $tag no mind"; continue }
      $extra += @('--load-mind', $mind)
    }
    # random/own/share need a mind to answer after the walk - train 0 with no load is empty Phi.
    # 338's reachable_rate is a TAPE property (walk offer); hit_rate needs SOME mind.
    # For non-mind rules: load same wiki mind frozen so pick is comparable; retention is the walk.
    if ($rule -ne 'mind') {
      if (-not (Test-Path $mind)) { L "SKIP $tag no mind for exam"; continue }
      $extra += @('--load-mind', $mind)
    }
    python -u _stage289_derivation.py @base --seed $s --run-tag $tag --out $out @extra
    if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
  }
}
L "==== _read338 retention ===="
python _read338_retention.py (Get-ChildItem out\_stage289_decision_338*_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held | Tee-Object -FilePath results\_read338_held.txt

L "DONE 336+337+338"
