$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_316_pick.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "316 pick: ctrl(gamma=1) then 313a(gamma=0.9); seeds 1337 8642 2890 4711; arrive vs pick"

$common = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1',
  '--probe-period','1000','--probe-size','60','--cpu',
  '--train-steps','4000'
)
$seeds = @(1337, 8642, 2890, 4711)

foreach ($arm in @(
  @{ name = '316ctrl'; gamma = '1.0' },
  @{ name = '316_313a'; gamma = '0.9' }
)) {
  foreach ($s in $seeds) {
    $tag = "$($arm.name)_s$s"
    $out = "out/_stage289_decision_$tag.json"
    L "==== $tag gamma=$($arm.gamma) ===="
    python -u _stage289_derivation.py @common --gamma $arm.gamma --seed $s --run-tag $tag --out $out
    if ($LASTEXITCODE -ne 0) { L "FAIL $tag exit $LASTEXITCODE"; exit $LASTEXITCODE }
    L "OK $tag"
  }
}

L "==== _read299 ctrl ===="
python _read299.py (Get-ChildItem out\_stage289_decision_316ctrl_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "==== _read299 313a ===="
python _read299.py (Get-ChildItem out\_stage289_decision_316_313a_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held
L "DONE"
