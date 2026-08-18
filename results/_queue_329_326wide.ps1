$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_329_326wide.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "329ctrl (coherence on clean mind) then 326wide (compass both + cands 32)"

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

# --- 329: coherence on CLEAN ctrl (no depth, no compass, no flat) ---
L "==== BLOCK 329ctrl: --coherence 400 on clean ctrl x4 ===="
foreach ($s in $seeds) {
  $tag = "329ctrl_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag --coherence 400 ===="
  python -u _stage289_derivation.py @base --coherence 400 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) {
    L "ARM EXIT $LASTEXITCODE $tag continuing"
  } else { L "OK $tag" }
}
$f329 = @(Get-ChildItem out\_stage289_decision_329ctrl_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($f329.Count -gt 0) {
  L "==== _read299 329ctrl ===="
  python _read299.py @f329 --held | Tee-Object -FilePath results\_read329ctrl_held.txt
  python -c @"
import json,glob
print('=== COHERENCE held_out (clean ctrl) ===')
for p in sorted(glob.glob('out/_stage289_decision_329ctrl_s*.json')):
    d=json.load(open(p,encoding='utf-8'))
    h=((d.get('coherence') or {}).get('held_out') or {})
    print(p.split('decision_')[-1], 'real_higher', h.get('real_higher'), 'ties', h.get('ties'), 'gap', h.get('mean_gap'), 'z', h.get('binomial_z'), 'n', h.get('n'))
"@
}

# --- 326: second compass, now with room (cands 32) ---
L "==== BLOCK 326wide: --reach-compass both --reach-cands 32 x4 ===="
foreach ($s in $seeds) {
  $tag = "326wide_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag compass=both cands=32 ===="
  python -u _stage289_derivation.py @base --reach-compass both --reach-cands 32 --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) {
    L "ARM EXIT $LASTEXITCODE $tag continuing"
  } else { L "OK $tag" }
}
$f326 = @(Get-ChildItem out\_stage289_decision_326wide_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object { $_.FullName })
if ($f326.Count -gt 0) {
  L "==== _read299 326wide (reachable / PICK / vs COUNT) ===="
  python _read299.py @f326 --held | Tee-Object -FilePath results\_read326wide_held.txt
}

L "DONE 329ctrl+326wide"
