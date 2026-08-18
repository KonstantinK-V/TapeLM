$ErrorActionPreference = "Continue"
$log = "results\_queue_292_seeds2.out"
Set-Location $PSScriptRoot\..

function L($m) {
  $line = "$(Get-Date -Format o) $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

L '292 seed batch2: 1337, 5200, 8642 (same recipe as 292_open)'

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

foreach ($pair in @(@(1337, '292_s1337'), @(5200, '292_s5200'), @(8642, '292_s8642'))) {
  $seed = $pair[0]; $tag = $pair[1]
  L "$tag seed=$seed"
  $out = "results\_stage289_full_$tag.out"
  $arg = $common + @('--seed', "$seed", '--run-tag', $tag)
  & python -u _stage289_derivation.py @arg *> $out
  $code = $LASTEXITCODE
  L "EXIT_$tag=$code"
  if ($code -ne 0) { L "STOP after $tag"; exit $code }
}

L 'QUEUE_DONE seed batch2'
exit 0
