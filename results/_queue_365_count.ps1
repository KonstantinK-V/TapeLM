$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_365_count.out"

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
L "365r3: standing arm + 383 count tie-break on the counting rival. Tag 365r3 keeps 365r2. Read PICK vs COUNT and ties next to top_share."
L "GATE (declared): count-rival rises on at least one seed. ties>1 and count-rival up = old rule arbitrary; ties=1.00 and count-rival up = old rule blunt; ties=1.00 and count-rival flat = need a different rival."

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
  $tag = "365r3_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read299 365r3 ===="
Read-Held "out\_stage289_decision_365r3_s*.json" "results\_read299_365r3.txt"
L "==== _read299 365r2 control ===="
Read-Held "out\_stage289_decision_365r2_s*.json" "results\_read299_365r2_ctrl383.txt"

L "DONE 365r3"
