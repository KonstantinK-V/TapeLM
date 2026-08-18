$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$nightLog = "results\_queue_night_326.out"
$log = "results\_queue_331.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "331: wait for night DONE, then --reach-cands 32 x4 (fork: widen offer vs constraint interface)"

# Wait until night queue writes its DONE line (running process already started without 331)
L "waiting for night queue ($nightLog) ..."
while ($true) {
  if (Test-Path $nightLog) {
    $tail = Get-Content $nightLog -Tail 5 -ErrorAction SilentlyContinue
    if ($tail -match 'DONE night') { break }
  }
  Start-Sleep -Seconds 60
}
L "night DONE - starting 331wide"

$base = @(
  '--tape','frames','--frame-max','3','--min-mentions','1',
  '--fp','hash','--write-fp','hash','--ink','mean','--write-ink','mean','--words','ascii',
  '--reach','--reach-no-refuse','--reach-lookahead','--frame-fp','fillers',
  '--tape-sample','region',
  '--objective','reward','--addresses','1500','--reach-max-q','2000',
  '--import-k','1','--gamma','1.0','--reach-cands','32',
  '--probe-period','1000','--probe-size','60','--cpu','--train-steps','4000'
)
$seeds = @(1337, 8642, 2890, 4711)

foreach ($s in $seeds) {
  $tag = "331wide_s$s"
  $out = "out/_stage289_decision_$tag.json"
  L "==== $tag --reach-cands 32 ===="
  python -u _stage289_derivation.py @base --seed $s --run-tag $tag --out $out
  if ($LASTEXITCODE -ne 0) {
    L "ARM EXIT $LASTEXITCODE $tag — if REACH BOOKKEEPING BROKEN, intentional; continuing"
    continue
  }
  L "OK $tag"
}

L "==== _read299 331 (reachable / PICK / PICK vs COUNT) ===="
python _read299.py (Get-ChildItem out\_stage289_decision_331wide_s*.json | Sort-Object Name | ForEach-Object { $_.FullName }) --held | Tee-Object -FilePath results\_read331_held.txt
L "DONE 331"
