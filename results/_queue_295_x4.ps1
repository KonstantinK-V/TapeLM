$ErrorActionPreference = "Continue"
$log = "results\_queue_295_x4.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

# 295a was 200k/100k/600MB/addr1200 -> 4 train / 18 eval (birthday starvation).
# Scale corpus x4 so pair-repeats grow ~x16 (~300 eval); addresses x4 so N can grow with it.
L '295_x4: patterns, corpus x4 vs 295a (lines/bytes/addresses), test whether 78vs56 survives'
$arg = @(
  '--patterns', '--addresses', '4800',
  '--wiki-bytes', '2400000000', '--train-lines', '800000', '--eval-lines', '400000',
  '--fp', 'hash', '--write-fp', 'hash', '--address-tau', '0.4712', '--min-mentions', '2',
  '--train-steps', '6000', '--seed', '1337', '--run-tag', '295_x4'
)
& python -u _stage289_derivation.py @arg *> results\_stage289_full_295_x4.out
$code = $LASTEXITCODE
L "EXIT_295_x4=$code"
exit $code
