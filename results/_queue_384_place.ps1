$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_384_place.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "384 cons-resolve place on standing arm: --constrain --cons-resolve place --min-fillers 1 --reach-depth 2 --two-way --connect. Tag 384place."
L "GATE (345, unsoftened): (a) lens choice beats rare/frequent/decisive paired beyond-own, z>=+1.645 pooled AND ahead on >=3/4; (b) answerable > walk_answerable on the same question, >=3/4. Fail = seeker does not build on this tape."

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1','--connect',
  '--constrain','--cons-resolve','place'
)

$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $tag = "384place_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read345_cons 384place --held ===="
$f = @(Get-ChildItem out\_stage289_decision_384place_s*.json,results\stage289_decision_384place_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) {
  python -c "from pathlib import Path; import subprocess, sys; p=Path(r'results\_read345_384place.txt'); files=sys.argv[1:]; r=subprocess.run([sys.executable,'_read345_cons.py','--held',*files], capture_output=True, text=True, encoding='utf-8'); p.write_text(r.stdout+(r.stderr or ''), encoding='utf-8'); sys.exit(r.returncode)" @f
}

L "DONE 384place"
