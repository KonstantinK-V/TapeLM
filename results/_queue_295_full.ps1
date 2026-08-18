$ErrorActionPreference = "Continue"
$log = "results\_queue_295_full.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

L '295_full: patterns schema, addresses 5000, train-steps 6000 (supply after 295a starved at 1200)'
$arg = @(
  '--patterns', '--addresses', '5000',
  '--wiki-bytes', '600000000', '--train-lines', '200000', '--eval-lines', '100000',
  '--fp', 'hash', '--write-fp', 'hash', '--address-tau', '0.4712', '--min-mentions', '2',
  '--train-steps', '6000', '--seed', '1337', '--run-tag', '295_full'
)
& python -u _stage289_derivation.py @arg *> results\_stage289_full_295_full.out
$code = $LASTEXITCODE
L "EXIT_295_full=$code"
exit $code
