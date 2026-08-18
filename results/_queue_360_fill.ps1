$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_360_fill.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "360 min-fillers 1 on standard arm (depth 2, two-way), 4 seeds; ctrl=353twoway (mf=2)"

# Same arm as 353twoway / _arm_ctrl_next (max-q 8000), wiki corpus; only --min-fillers 1 is new
$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1'
)
$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $tag = "360fill_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read299 360fill ===="
$f = @(Get-ChildItem out\_stage289_decision_360fill_s*.json,results\stage289_decision_360fill_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_360fill.txt }

L "==== _read299 353twoway control ===="
$f = @(Get-ChildItem results\stage289_decision_353twoway_s*.json,out\_stage289_decision_353twoway_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_353twoway_ctrl360.txt }

L "DONE 360 fill"
