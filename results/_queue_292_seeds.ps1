$ErrorActionPreference = "Continue"
$log = "results\_queue_night_291_292.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

L 'seed queue: wait night v43 (290b/audit2400), then 292_s4001 and 292_s7919 only'

# Wait until the night after_292 script and its python jobs are gone
while ($true) {
  $busy = $false
  Get-CimInstance Win32_Process | ForEach-Object {
    $c = $_.CommandLine
    if (-not $c) { return }
    if ($c -match '_queue_after_292_audit') { $busy = $true }
    if ($c -match 'run-tag 290b_dense') { $busy = $true }
    if ($c -match 'run-tag audit2400') { $busy = $true }
  }
  if (-not $busy) { break }
  Start-Sleep -Seconds 30
}
L 'night queue clear; start seed replicas'

$common = @(
  '--train-steps', '6000',
  '--addresses', '1200',
  '--wiki-bytes', '600000000',
  '--train-lines', '200000',
  '--eval-lines', '100000',
  '--probe-period', '100',
  '--fp', 'hash',
  '--write-fp', 'hash',
  '--min-mentions', '2',
  '--address-tau', '0.4712',
  '--import-k', '2',
  '--open'
)

foreach ($pair in @(@(4001, '292_s4001'), @(7919, '292_s7919'))) {
  $seed = $pair[0]; $tag = $pair[1]
  L "$tag seed=$seed"
  $out = "results\_stage289_full_$tag.out"
  $arg = $common + @('--seed', "$seed", '--run-tag', $tag)
  & python -u _stage289_derivation.py @arg *> $out
  $code = $LASTEXITCODE
  L "EXIT_$tag=$code"
  if ($code -ne 0) { L "STOP after $tag"; exit $code }
}

L 'QUEUE_DONE seed replicas'
exit 0
