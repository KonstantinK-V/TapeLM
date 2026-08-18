$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_353_twoway.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "353 two-way @ depth 2: max arm + margin arm, 4 seeds each; ctrl=352deep"

# Same arm as 352deep / _arm_ctrl_next (max-q 8000), wiki corpus
$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--dim','32','--train-steps','4000'
)
$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($arm in @(
  @{ tag = '353twoway'; extra = @('--two-way') },
  @{ tag = '353margin'; extra = @('--two-way','--two-way-by','margin') }
)) {
  foreach ($s in $seeds) {
    $tag = "$($arm.tag)_s$s"
    $out = "out/_stage289_decision_$tag.json"
    L "==== $tag ===="
    python -u _stage289_derivation.py @base @($arm.extra) --wiki $wiki --seed $s --run-tag $tag --out $out
    if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
  }
}

L "==== _read299 353twoway ===="
$f = @(Get-ChildItem out\_stage289_decision_353twoway_s*.json,results\stage289_decision_353twoway_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_353twoway.txt }

L "==== _read299 353margin ===="
$f = @(Get-ChildItem out\_stage289_decision_353margin_s*.json,results\stage289_decision_353margin_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) { python _read299.py --held @f | Tee-Object -FilePath results\_read299_353margin.txt }

L "==== gate vs 352deep (prereg: CONFIRM z>=0 AND walk-only z>=+10) ===="
python results\_gate353.py | Tee-Object -FilePath results\_gate353.txt

L "DONE 353"
