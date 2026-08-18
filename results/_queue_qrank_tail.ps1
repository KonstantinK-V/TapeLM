$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_qrank.out"
$env:PYTHONIOENCODING = "utf-8"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
L "qrank tail v26: bigram+hash with --probe-period 100 + fast-cos (after qrank)"

$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --import-k 0 --no-ladder --probe-period 100"

$arms = @(
  @{ tag = "qrank_bigram"; extra = "--ink bigram" },
  @{ tag = "qrank_hash";   extra = "--fp hash" }
)

foreach ($a in $arms) {
  $tag = $a.tag
  $extra = $a.extra
  L "289 $tag $extra"
  cmd /c "python -u _stage289_derivation.py $base $extra --run-tag $tag > results\_stage289_full_$tag.out 2>&1"
  L "EXIT_$tag=$LASTEXITCODE"
  if ($LASTEXITCODE -ne 0) { L "STOP after $tag"; exit $LASTEXITCODE }
}
L "QUEUE_DONE"
exit 0
