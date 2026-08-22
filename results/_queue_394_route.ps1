$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_394_route.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

function Read-Held([string]$glob, [string]$dest) {
  $f = @(Get-ChildItem $glob -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
  if ($f.Count -gt 0) {
    python -c "from pathlib import Path; import subprocess, sys; p=Path(sys.argv[1]); files=sys.argv[2:]; r=subprocess.run([sys.executable,'_read299.py','--held',*files], capture_output=True, text=True, encoding='utf-8'); p.write_text(r.stdout+(r.stderr or ''), encoding='utf-8'); sys.exit(r.returncode)" $dest @f
  }
}

Set-Content -Path $log -Value "" -Encoding utf8
L "394 --route-on walk_only. Void checks on dumps did not fire on standing 365r3. Control is 365r3 (--route-on all)."
L "GATE 35.3: walk_only hit(step) > hit(stay)+0.05; full-pop hop1 must not fall >0.03 vs 365r3; report route_on_live."

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1','--connect','--route-on','walk_only'
)

$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $tag = "394route_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read299 394route ===="
Read-Held "out\_stage289_decision_394route_s*.json" "results\_read299_394route.txt"
L "==== _read299 365r3 control ===="
Read-Held "out\_stage289_decision_365r3_s*.json" "results\_read299_365r3_ctrl394.txt"
L "==== _read394_walkonly 394route ===="
$rf = @(Get-ChildItem "out\_stage289_decision_394route_s*.json" -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($rf.Count -gt 0) {
  python _read394_walkonly.py @rf --held | Tee-Object -FilePath "results\_read394_walkonly_394route.txt"
}

L "DONE 394route"
