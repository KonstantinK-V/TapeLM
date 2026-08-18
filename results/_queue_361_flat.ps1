$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_361_flat.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "361 flat: 360 arm minus --reach-depth 2, min-fillers 1, 4 seeds; ctrl=360fill"

# Same as 360fill except no --reach-depth 2 (default depth 1). --two-way stays: it is the
# stay-vs-go comparison, which is exactly the over-stepping question.
$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1'
)
$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $tag = "361flat_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read299 361flat ===="
$f = @(Get-ChildItem out\_stage289_decision_361flat_s*.json,results\stage289_decision_361flat_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_361flat.txt }

L "==== _read299 360fill control ===="
$f = @(Get-ChildItem results\stage289_decision_360fill_s*.json,out\_stage289_decision_360fill_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_360fill_ctrl361.txt }

L "DONE 361 flat"
