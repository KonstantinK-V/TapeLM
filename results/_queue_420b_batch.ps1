$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_420b_batch.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "420b COMPARE: batch 50/50 + letter-hash fillers. Separate JSON. Do not clobber 420."

if (Test-Path results\_stage420b_batch.json) {
  Remove-Item results\_stage420b_batch.json -Force
}

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 420b batch seed $s ===="
  python -u _audit420b_batch.py --seed $s --steps 3000 --cpu --balance batch `
    --save-mind "out/_mind_420b_batch_s$s.pt" `
    --out "results/_stage420b_batch.json"
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "DONE 420b batch compare"
python -c "
import json
from pathlib import Path
p=Path('results/_stage420b_batch.json')
d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
print('seed  mind_live  rnd_live  dlt    mind   always_r  pins  ge_ar')
gp=ga=0
for s in ('1337','8642','2890'):
 h=d.get(s) or {}
 pins=bool(h.get('gate_pins')); ar=bool(h.get('gate_mind_ge_always_refuse'))
 gp+=pins; ga+=ar
 print(f\"{s}  {h.get('mind_live'):.3f}  {h.get('random_live'):.3f}  {h.get('mind_live_minus_random_live'):+.3f}  {h.get('mind_pin'):.3f}  {h.get('always_refuse'):.3f}  {pins}  {ar}\")
print(f'GATE pins: {gp}/3   mind>=always_refuse: {ga}/3')
"
L "DONE 420b"
