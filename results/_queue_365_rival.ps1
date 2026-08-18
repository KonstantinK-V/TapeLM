$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_365_rival.out"

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
L "365r RECAST on 382: standing arm, honest counting rival sees connect cands. Tag 365r2 keeps 365r JSON (PICK 1132/1, inflated)."
L "Not a lever. Question: does PICK hold against a rival that can NAME the same offer. Control=365r."

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1','--connect'
)

$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $tag = "365r2_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read299 365r2 ===="
Read-Held "out\_stage289_decision_365r2_s*.json" "results\_read299_365r2.txt"
L "==== _read299 365r control ===="
Read-Held "out\_stage289_decision_365r_s*.json" "results\_read299_365r_ctrl382.txt"

L "DONE 365r2"
