$ErrorActionPreference = "Continue"
$log = "results\_queue_295_wide.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

# Real x4 on local wiki: drop line-max 400 (~761k filtered vs ~182k). Keep addresses high so N grows.
L '295_wide: patterns, --line-max 0 (~x4 lines), addresses 4800, test 78vs56 at 300+ rules'
$arg = @(
  '--patterns', '--line-max', '0', '--addresses', '4800',
  '--wiki-bytes', '600000000', '--train-lines', '800000', '--eval-lines', '400000',
  '--fp', 'hash', '--write-fp', 'hash', '--address-tau', '0.4712', '--min-mentions', '2',
  '--train-steps', '6000', '--seed', '1337', '--run-tag', '295_wide'
)
& python -u _stage289_derivation.py @arg *> results\_stage289_full_295_wide.out
$code = $LASTEXITCODE
L "EXIT_295_wide=$code"
exit $code
