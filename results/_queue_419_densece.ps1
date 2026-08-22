$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_419_densece.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "419 dense CE on 417h y. Joint>=2. Gate: mind>random pin; refuse df1>df2. Not 418."

if (Test-Path results\_stage419_densece.json) {
  Remove-Item results\_stage419_densece.json -Force
}

python _check419_densece.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 419 densece seed $s ===="
  python -u _audit419_densece.py --seed $s --steps 3000 --cpu `
    --save-mind "out/_mind_419_densece_s$s.pt" `
    --out "results/_stage419_densece.json"
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict ===="
python results\_read419_densece.py
L "DONE 419 densece"
