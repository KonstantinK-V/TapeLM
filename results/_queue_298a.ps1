$ErrorActionPreference = "Continue"
$log = "results\_queue_298a.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

L '298a: frames LIVE (fps+nfill) + route, frame-max 3'
$arg = @(
  '--tape', 'frames', '--frame-max', '3', '--min-mentions', '1',
  '--open', '--mixed', '--route', '--open-cands', 'uniform',
  '--objective', 'reward', '--import-k', '2',
  '--addresses', '3000', '--train-steps', '6000', '--seed', '1337', '--run-tag', '298a'
)
& python -u _stage289_derivation.py @arg *> results\_stage289_full_298a.out
$code = $LASTEXITCODE
L "EXIT_298a=$code"
exit $code
