$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_379_rerun.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "379 channel RERUN on 380 deep-root. Same offer as 377copy: standing + --copy --copy-d 4 --reach-channel. Tag 379r keeps 379ch JSON as the buggy-root record."
L "GATE (declared, vs 377copy): hit >= 377copy on 4/4; reach not below 377copy; PICK pooled not below 1258/1572. If hit does not move, provenance is unused and interleave can be frozen."

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','8000',
  '--import-k','1','--gamma','1.0',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--reach-depth','2','--two-way','--dim','32','--train-steps','4000',
  '--min-fillers','1','--connect','--copy','--copy-d','4','--reach-channel'
)

$seeds = @(1337, 8642, 2890, 4711)
$wiki = "data\_wikitext103_train.txt"

foreach ($s in $seeds) {
  $tag = "379r_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag ===="
  python -u _stage289_derivation.py @base --wiki $wiki --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "ARM EXIT $LASTEXITCODE $tag continuing" } else { L "OK $tag" }
}

L "==== _read299 379r ===="
$f = @(Get-ChildItem out\_stage289_decision_379r_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) {
  python -c "from pathlib import Path; import subprocess, sys; p=Path(r'results\_read299_379r.txt'); files=sys.argv[1:]; r=subprocess.run([sys.executable,'_read299.py','--held',*files], capture_output=True, text=True, encoding='utf-8'); p.write_text(r.stdout+(r.stderr or ''), encoding='utf-8'); sys.exit(r.returncode)" @f
}

L "==== _read299 377copy control ===="
$f = @(Get-ChildItem out\_stage289_decision_377copy_s*.json -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
if ($f.Count -gt 0) {
  python -c "from pathlib import Path; import subprocess, sys; p=Path(r'results\_read299_377copy_ctrl379r.txt'); files=sys.argv[1:]; r=subprocess.run([sys.executable,'_read299.py','--held',*files], capture_output=True, text=True, encoding='utf-8'); p.write_text(r.stdout+(r.stderr or ''), encoding='utf-8'); sys.exit(r.returncode)" @f
}

L "DONE 379r"
