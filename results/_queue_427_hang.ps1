$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_427_hang.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "427 hang rare<=median. K=8. No Phi. Checker 4/4. 426 untouched."

if (Test-Path results\_stage427_hang.json) {
  Remove-Item results\_stage427_hang.json -Force
}

python _check427_hang.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 427 hang seed $s ===="
  python -u _audit427_hang.py --seed $s
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict ===="
python results\_read427_hang.py
L "DONE 427"
