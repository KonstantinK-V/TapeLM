$ErrorActionPreference = "Continue"
$log = "results\_queue_295a.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

L '295a: mined rules, Phi on 4-row worlds, lift rival (schema from 295.patch)'
$arg = @(
  '--patterns', '--addresses', '1200',
  '--wiki-bytes', '600000000', '--train-lines', '200000', '--eval-lines', '100000',
  '--fp', 'hash', '--write-fp', 'hash', '--address-tau', '0.4712', '--min-mentions', '2',
  '--train-steps', '3000', '--seed', '1337', '--run-tag', '295a'
)
& python -u _stage289_derivation.py @arg *> results\_stage289_full_295a.out
$code = $LASTEXITCODE
L "EXIT_295a=$code"
exit $code
