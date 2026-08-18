$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_edge_ablation.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "edge-channel ablation 289-ONLY: e_both (done) / e_rare / e_cos — no 289c"

$arms = @(
  @{ tag = "e_rare"; ch = "same,rare" },
  @{ tag = "e_cos";  ch = "same,cos" }
)
# If e_rare decision already exists, skip to e_cos only
if (Test-Path (Join-Path $root 'results\stage289_decision_e_rare.json')) {
  $arms = @(@{ tag = "e_cos"; ch = "same,cos" })
  L "skip e_rare - decision already on disk"
}

foreach ($a in $arms) {
  $tag = $a.tag
  $ch = $a.ch
  L "289 $tag channels=$ch"
  cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --no-ladder --edge-channels $ch --run-tag $tag > results\_stage289_full_$tag.out 2>&1"
  L "EXIT_289_$tag=$LASTEXITCODE"
  if ($LASTEXITCODE -ne 0) { L "STOP"; exit $LASTEXITCODE }
}
L "QUEUE_DONE"
exit 0
