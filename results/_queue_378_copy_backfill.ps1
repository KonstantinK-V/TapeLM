$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_378_copy_backfill.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8

L "378 copy-backfill lane: standing arm + --copy --copy-backfill --copy-d 4, 4 seeds; ctrl=365conn."
L "GATE (declared): reach >= 377copy on 4/4; hit >= 365conn on 4/4; cand_places not below 365conn."

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1','--connect','--copy','--copy-backfill','--copy-d','4'
)

$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $tag = "378copybf_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read299 378copybf ===="
$f = @(Get-ChildItem out\_stage289_decision_378copybf_s*.json,results\stage289_decision_378copybf_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) {
  python "C:\Users\Kostya\Downloads\_read299(11).py" --held @f | Tee-Object -FilePath results\_read299_378copybf.txt
}

L "==== _read299 365conn control ===="
$f = @(Get-ChildItem out\_stage289_decision_365conn_s*.json,results\stage289_decision_365conn_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) {
  python "C:\Users\Kostya\Downloads\_read299(11).py" --held @f | Tee-Object -FilePath results\_read299_365conn_ctrl378copybf.txt
}

L "DONE 378copybf"

