$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_night_326.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Add-Content -Path $log -Value "" -Encoding utf8
L "NIGHT RESUME continue-on-fail: BOOKKEEPING exit 1 is intentional, keep going; always DONE"

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu'
)
$seeds = @(1337, 8642, 2890, 4711)
$news = "data\_stage254_news.txt"
$tiny = "data\_tinystories_raw_scale.txt"
New-Item -ItemType Directory -Force -Path minds | Out-Null

function Out-Exists([string]$p) {
  return ((Test-Path $p) -and ((Get-Item $p).Length -gt 1000))
}

# 327
if (-not (Test-Path "results\_stage327_presence.json")) {
  L "==== 327 presence audit ===="
  python -u _audit327_presence.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
  if ($LASTEXITCODE -ne 0) { L "FAIL 327 exit $LASTEXITCODE continuing" } else { L "OK 327" }
} else { L "SKIP 327" }

# BLOCK 1
L "==== BLOCK 1: 322b depth+save+coherence ===="
foreach ($s in $seeds) {
  $tag = "322b_s$s"
  $out = "out/_stage289_decision_$tag.json"
  $mind = "minds/322_wiki_s$s.pt"
  if (Out-Exists $out) { L "SKIP $tag"; continue }
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --reach-depth 2 --coherence 400 --train-steps 4000 --seed $s --run-tag $tag --save-mind $mind --out $out
  if ($LASTEXITCODE -ne 0) {
    L "ARM EXIT $LASTEXITCODE $tag (BOOKKEEPING=intentional if that message; continuing)"
  } else { L "OK $tag -> $mind" }
}
$files = @(Get-ChildItem out\_stage289_decision_322b_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($files.Count -gt 0) {
  L "==== _read299 block1 ===="
  python _read299.py @files --held 2>&1 | Tee-Object -FilePath results\_read322b_held.txt
  python -c "import json,glob
for p in sorted(glob.glob('out/_stage289_decision_322b_s*.json')):
 d=json.load(open(p,encoding='utf-8')); h=((d.get('coherence') or {}).get('held_out') or {})
 print(p.split('decision_')[-1],'real_higher',h.get('real_higher'),'z',h.get('binomial_z'),'n',h.get('n'))"
}

# BLOCK 2
L "==== BLOCK 2: 326 both ===="
foreach ($s in $seeds) {
  $tag = "326_s$s"
  $out = "out/_stage289_decision_$tag.json"
  if (Out-Exists $out) { L "SKIP $tag"; continue }
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --reach-compass both --train-steps 4000 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$files = @(Get-ChildItem out\_stage289_decision_326_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($files.Count -gt 0) {
  L "==== _read299 block2 ===="
  python _read299.py @files --held 2>&1 | Tee-Object -FilePath results\_read326_held.txt
}

# BLOCK 3
L "==== BLOCK 3: transplant ===="
foreach ($corp in @(@{name='news'; path=$news}, @{name='tiny'; path=$tiny})) {
  foreach ($s in $seeds) {
    $tag = "322t_$($corp.name)_s$s"
    $out = "out/_stage289_decision_$tag.json"
    $mind = "minds/322_wiki_s$s.pt"
    if (-not (Test-Path $mind)) { L "SKIP $tag no mind"; continue }
    if (Out-Exists $out) { L "SKIP $tag"; continue }
    L "==== $tag ===="
    python -u _stage289_derivation.py @base --reach-depth 2 --load-mind $mind --wiki $corp.path --train-steps 0 --seed $s --run-tag $tag --out $out
    if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
  }
}
$fn = @(Get-ChildItem out\_stage289_decision_322t_news_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
$ft = @(Get-ChildItem out\_stage289_decision_322t_tiny_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($fn.Count -gt 0) { python _read299.py @fn --held 2>&1 | Tee-Object -FilePath results\_read322t_news.txt }
if ($ft.Count -gt 0) { python _read299.py @ft --held 2>&1 | Tee-Object -FilePath results\_read322t_tiny.txt }

# BLOCK 4
L "==== BLOCK 4: flat+depth ===="
foreach ($s in $seeds) {
  $tag = "322flat_s$s"
  $out = "out/_stage289_decision_$tag.json"
  if (Out-Exists $out) { L "SKIP $tag"; continue }
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --flat --reach-depth 2 --train-steps 4000 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$files = @(Get-ChildItem out\_stage289_decision_322flat_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($files.Count -gt 0) { python _read299.py @files --held 2>&1 | Tee-Object -FilePath results\_read322flat_held.txt }

# 328
L "==== BLOCK 328: shuffle null ===="
foreach ($s in $seeds) {
  $tag = "328null_s$s"
  $out = "out/_stage289_decision_$tag.json"
  if (Out-Exists $out) { L "SKIP $tag"; continue }
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --shuffle-tape --train-steps 4000 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}
$files = @(Get-ChildItem out\_stage289_decision_328null_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($files.Count -gt 0) { python _read299.py @files --held 2>&1 | Tee-Object -FilePath results\_read328_held.txt }

L "DONE night 326-329 (continue-on-fail; 331 waiter may proceed)"
