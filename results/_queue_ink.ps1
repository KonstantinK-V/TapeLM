$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_ink.out"
$env:PYTHONIOENCODING = "utf-8"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "ink queue v18: ROADMAP/HANDOFF; ink_mean (=exact2) -> ink_bigram -> fp_hash -> fp_hash_bigram"

# same corpus/recipe as exact2 so ink_mean can match bit-for-bit
$base = "--train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --import-k 0 --no-ladder"

$arms = @(
  @{ tag = "ink_mean";       extra = "--ink mean" },
  @{ tag = "ink_bigram";     extra = "--ink bigram" },
  @{ tag = "fp_hash";        extra = "--fp hash" },
  @{ tag = "fp_hash_bigram"; extra = "--fp hash --ink bigram" }
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
