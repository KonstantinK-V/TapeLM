$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_322.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "322 RERUN after 325 arrival: depth option on arrival only (not reason to leave); read step first"

$common = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1','--gamma','1.0','--reach-depth','2',
  '--probe-period','1000','--probe-size','60','--cpu','--train-steps','4000'
)
$seeds = @(1337, 8642, 2890, 4711)

foreach ($s in $seeds) {
  $tag = "322_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag --reach-depth 2 (325 bookkeeping) ===="
  python -u _stage289_derivation.py @common --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
  L "OK $tag"
}

L "==== _read299 held ===="
python _read299.py (Get-ChildItem out\_stage289_decision_322_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held | Tee-Object -FilePath results\_read322_held.txt
L "DONE"
