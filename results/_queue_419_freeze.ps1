$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_419_freeze.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "419 FREEZE standing: exam-only mind_live / always_refuse / refuse_df*. No new lever."

python _check419_densece.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  $mind = "out/_mind_419_densece_s$s.pt"
  if (-not (Test-Path $mind)) { L "MISSING $mind"; exit 1 }
  L "==== freeze exam seed $s ===="
  python -u _audit419_densece.py --seed $s --cpu --load-mind $mind `
    --out "results/_stage419_densece.json"
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== standing verdict ===="
python results\_read419_densece.py
L "DONE 419 freeze"
