$ErrorActionPreference = "Continue"
$log = "results\_queue_294_s1337.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

L '294_s1337: open + anchor + uniform'
$arg = @(
  '--open', '--address-from', 'anchor', '--open-cands', 'uniform',
  '--import-k', '2', '--addresses', '1200',
  '--wiki-bytes', '600000000', '--train-lines', '200000', '--eval-lines', '100000',
  '--fp', 'hash', '--write-fp', 'hash', '--address-tau', '0.4712', '--min-mentions', '2',
  '--train-steps', '6000', '--probe-period', '100', '--seed', '1337', '--run-tag', '294_s1337'
)
& python -u _stage289_derivation.py @arg *> results\_stage289_full_294_s1337.out
$code = $LASTEXITCODE
L "EXIT_294_s1337=$code"
exit $code
