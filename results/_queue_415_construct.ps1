$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_415_construct.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "415 construction: DECLARED mind = 412ref_ce freeze. Paths below. Gate before numbers."
L "  out/_mind_constr_412ref_ce_s1337.pt"
L "  out/_mind_constr_412ref_ce_s8642.pt"
L "  out/_mind_constr_412ref_ce_s2890.pt"
L "GATE ge2: mind_hit > count_hit on 3/3. GATE one: refuse > refuse(ge2) on 3/3. No GATE-WO."

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
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $mind = "out/_mind_constr_412ref_ce_s$s.pt"
  if (-not (Test-Path $mind)) {
    L "==== TRAIN+SAVE $mind ===="
    python -u _stage289_derivation.py @base --wiki $wiki --seed $s --train-steps 4000 `
      --run-tag "constr_train_ce_s$s" --out "out/_stage289_decision_constr_train_ce_s$s.json" `
      --save-mind $mind
    if ($LASTEXITCODE -ne 0) { L "TRAIN EXIT $LASTEXITCODE s$s"; continue }
    L "OK saved $mind"
  } else {
    L "==== reuse existing $mind ===="
  }

  L "==== FREEZE EXAM 415raw_s$s ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --train-steps 0 `
    --load-mind $mind `
    --run-tag "415raw_s$s" --out "out/_stage289_decision_415raw_s$s.json"
  if ($LASTEXITCODE -ne 0) { L "EXAM EXIT $LASTEXITCODE s$s" } else { L "OK 415raw_s$s" }
}

L "==== verdict (per seed, no pool) ===="
python -c @"
import json
from pathlib import Path
seeds = (1337, 8642, 2890)
print('seed  ge2_n  mind  count  dlt   one_n  ref1  ref_ge2  beat  ref_gate')
ok_beat = ok_ref = 0
for s in seeds:
    p = Path(f'out/_stage289_decision_415raw_s{s}.json')
    if not p.exists():
        print(s, 'MISSING'); continue
    r = (json.loads(p.read_text(encoding='utf-8')).get('reach') or {}).get('rawlit') or {}
    h = r.get('held_out') or {}
    g, o = h.get('ge2') or {}, h.get('one') or {}
    beat = bool(h.get('gate_mind_beats_count'))
    rg = bool(h.get('gate_refuse_one_gt_ge2'))
    ok_beat += int(beat); ok_ref += int(rg)
    dlt = (g.get('mind_hit') or 0) - (g.get('count_hit') or 0)
    print(f\"{s:4d}  {g.get('n'):5}  {g.get('mind_hit'):.3f}  {g.get('count_hit'):.3f}  {dlt:+.3f}  {o.get('n'):5}  {o.get('refuse'):.3f}  {g.get('refuse'):.3f}  {beat!s:5}  {rg!s}\")
print(f'GATE mind>count: {ok_beat}/3   refuse_one>ge2: {ok_ref}/3')
"@

L "DONE 415 construction"
