$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_mean_matched.out"
$env:PYTHONIOENCODING = "utf-8"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "confound queue v30: mean_matched (d=41 no-max) then mean_wide (d=64 no-max); same corpus as maxpool"

$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 600000000 --train-lines 200000 --eval-lines 100000 --import-k 0 --no-ladder --probe-period 100"

$arms = @(
  @{ tag = "mean_matched"; extra = "--no-max-pool --dim 41" },
  @{ tag = "mean_wide";    extra = "--no-max-pool --dim 64" }
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
