$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_after6_then_exact.out"
$dl = "C:\Users\Kostya\Downloads"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "WAIT for current 289(6) - code untouched until done"

while ($true) {
  $alive = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and (
      $_.CommandLine -match 'queue_289_pair_289c' -or
      ($_.CommandLine -match 'stage289_derivation\.py' -and $_.CommandLine -notmatch '--run-tag (exact|ladder)') -or
      ($_.CommandLine -match 'stage289c_audit\.py' -and $_.CommandLine -notmatch '--run-tag (exact|ladder)')
    )
  }
  if (-not $alive) { break }
  Start-Sleep -Seconds 30
}
L "CURRENT_RUN_DONE"

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$bak = Join-Path $PSScriptRoot ("backup_289v6_" + $stamp)
New-Item -ItemType Directory -Force -Path $bak | Out-Null
L "BACKUP_DIR $bak"
$copy = @(
  '_stage289_derivation.py',
  '_stage289c_audit.py',
  'results\_stage289_full_w150.out',
  'results\_stage289_full_w150_noderiv.out',
  'results\_stage289c_full_w150.out',
  'results\_queue_289_pair_289c.out',
  'results\stage289_decision_w150.json',
  'results\stage289_decision_w150_noderiv.json',
  'results\stage289c_decision_w150.json'
)
foreach ($rel in $copy) {
  $src = Join-Path $root $rel
  if (Test-Path $src) {
    Copy-Item -Force $src (Join-Path $bak (Split-Path $rel -Leaf))
    L "backed $rel"
  } else { L "missing $rel" }
}
Copy-Item -Force (Join-Path $root '_stage289_derivation.py') (Join-Path $bak '_stage289_derivation_v6.py')

# 1) soft algebra exact
Copy-Item -Force (Join-Path $dl '_stage289_derivation(8).py') (Join-Path $root '_stage289_derivation.py')
L "INSTALLED derivation(8)"
L "STEP 289 exact"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag exact > results\_stage289_full_exact.out 2>&1"
L "EXIT_289_exact=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP"; exit $LASTEXITCODE }

# 2) 289c audit(5) on exact
Copy-Item -Force (Join-Path $dl '_stage289c_audit(5).py') (Join-Path $root '_stage289c_audit.py')
L "INSTALLED audit(5)"
L "STEP 289c exact"
cmd /c "python -u _stage289c_audit.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag exact > results\_stage289c_full_exact.out 2>&1"
L "EXIT_289c_exact=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP"; exit $LASTEXITCODE }

# 3) ladder derivation + 289c (tags keep exact files intact)
Copy-Item -Force (Join-Path $dl '_stage289_derivation_lest.py') (Join-Path $root '_stage289_derivation.py')
L "INSTALLED derivation_lest"
L "STEP 289 ladder"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag ladder > results\_stage289_full_ladder.out 2>&1"
L "EXIT_289_ladder=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { L "STOP"; exit $LASTEXITCODE }

L "STEP 289c ladder"
cmd /c "python -u _stage289c_audit.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag ladder > results\_stage289c_full_ladder.out 2>&1"
L "EXIT_289c_ladder=$LASTEXITCODE"
L "QUEUE_DONE"
exit $LASTEXITCODE
