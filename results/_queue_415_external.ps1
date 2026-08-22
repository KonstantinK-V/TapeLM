$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_415_external.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "415 EXTERNAL as-is: frozen 412ref_ce minds on NEWS tape. No retrain. No new levers."
L "Corpus: data/_stage254_news.txt. Same rawlit readout (ge2 / one). Seeds 1337 8642 2890."

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32',
  '--min-fillers','1','--connect',
  '--pick-teacher','ce'
)

$seeds = @(1337, 8642, 2890)
$news = "data\_stage254_news.txt"

foreach ($s in $seeds) {
  $mind = "out/_mind_constr_412ref_ce_s$s.pt"
  if (-not (Test-Path $mind)) {
    L "MISSING mind $mind - abort seed"
    continue
  }
  L "==== EXTERNAL 415ext_news_s$s  load $mind ===="
  python -u _stage289_derivation.py @base --wiki $news --seed $s --train-steps 0 `
    --load-mind $mind `
    --run-tag "415ext_news_s$s" --out "out/_stage289_decision_415ext_news_s$s.json"
  if ($LASTEXITCODE -ne 0) { L "EXAM EXIT $LASTEXITCODE s$s" } else { L "OK 415ext_news_s$s" }
}

L "==== external verdict ===="
python results\_read415_rawlit.py (Get-ChildItem out\_stage289_decision_415ext_news_s*.json | ForEach-Object FullName)
L "DONE 415 external news"
