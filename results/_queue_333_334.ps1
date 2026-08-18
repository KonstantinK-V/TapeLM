$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_333_334.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "333root then 334tw then 334tw32 (read order: 333 vs 332, 334a vs ctrl, 334b arrive vs 331)"

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

# --- 333: fixed deep root (tape property again) vs existing 332 mind-root ---
L "==== BLOCK 333: --reach-depth 2 --deep-root first x4 ===="
foreach ($s in $seeds) {
  $tag = "333root_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --reach-depth 2 --deep-root first --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$f = @(Get-ChildItem out\_stage289_decision_333root_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($f.Count -gt 0) {
  L "==== _read299 333 (reachable vs 332 mind-root) ===="
  python _read299.py @f --held | Tee-Object -FilePath results\_read333_held.txt
}

# --- 334a: two-way against ctrl ---
L "==== BLOCK 334a: --two-way x4 (vs ctrl) ===="
foreach ($s in $seeds) {
  $tag = "334tw_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --two-way --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$f = @(Get-ChildItem out\_stage289_decision_334tw_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($f.Count -gt 0) {
  L "==== _read299 334a ===="
  python _read299.py @f --held | Tee-Object -FilePath results\_read334tw_held.txt
}

# --- 334b: two-way + cands 32 against 331 (dilution test: arrive) ---
L "==== BLOCK 334b: --two-way --reach-cands 32 x4 (vs 331; watch arrive) ===="
foreach ($s in $seeds) {
  $tag = "334tw32_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --two-way --reach-cands 32 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$f = @(Get-ChildItem out\_stage289_decision_334tw32_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($f.Count -gt 0) {
  L "==== _read299 334b (arrive vs 331) ===="
  python _read299.py @f --held | Tee-Object -FilePath results\_read334tw32_held.txt
}

L "DONE 333+334"
