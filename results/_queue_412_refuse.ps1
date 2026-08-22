$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_412_refuse.out"

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
L "412 fair race: CE vs reward WITH refuse (no --reach-no-refuse). Tags 412ref_*."
L "VOID first: pick_target ~1. Read per seed; do NOT pool 1337 with 4711."
L "Ask after: place-policy awake, or again eight names."

# Same geometry as standing 412, but REFUSE stays on the offer (default REACH_NO_REFUSE=False).
$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1','--connect'
)

$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"
$arms = @(
  @{ teacher = "ce";     prefix = "412ref_ce" },
  @{ teacher = "reward"; prefix = "412ref_rw" }
)

foreach ($arm in $arms) {
  foreach ($s in $seeds) {
    $tag = "$($arm.prefix)_s$s"
    $out = "out/_stage289_decision_$tag.json"
    L "==== $tag  pick-teacher=$($arm.teacher) ===="
    python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out `
      --pick-teacher $arm.teacher
    if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
  }
}

L "==== _read299 412ref_ce ===="
Read-Held "out\_stage289_decision_412ref_ce_s*.json" "results\_read299_412ref_ce.txt"
L "==== _read299 412ref_rw ===="
Read-Held "out\_stage289_decision_412ref_rw_s*.json" "results\_read299_412ref_rw.txt"

L "DONE 412 fair refuse race"
