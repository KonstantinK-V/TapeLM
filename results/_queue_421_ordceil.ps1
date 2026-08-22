$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_421_ordceil.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "421 ordered-match ceiling. No Phi. GO if ord-bag>0.05 and ord-rnd>0.05."

if (Test-Path results\_stage421_ordceil.json) {
  Remove-Item results\_stage421_ordceil.json -Force
}

python _check421_ordceil.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 421 ordceil seed $s ===="
  python -u _audit421_ordceil.py --seed $s
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict ===="
python results\_read421_ordceil.py
L "DONE 421"
