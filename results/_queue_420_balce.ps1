$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_420_balce.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "420 train: w_ref=n_live/n_ref, structural feats. NOT 420b. NOT 419."

python _check420_balce.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

if (Test-Path results\_stage420_balce.json) {
  Remove-Item results\_stage420_balce.json -Force
}

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 420 train seed $s ===="
  python -u _train420_balce.py --seed $s --steps 3000 --cpu `
    --save-mind "out/_mind_420_balce_s$s.pt" `
    --out "results/_stage420_balce.json"
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict ===="
python results\_read420_balce.py
L "DONE 420 train"
