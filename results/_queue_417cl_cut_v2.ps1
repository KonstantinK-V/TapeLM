$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_417cl_cut_v2.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "417cl v2: matched primary (freq band + illegal in exact frame). PRIOR gate. Not local-leak."

# Fresh results file for the fixed measurement
Remove-Item results\_stage417cl_cut.json -ErrorAction SilentlyContinue

python _check417cl_cut.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 417cl v2 seed $s ===="
  python -u _audit417cl_cut.py --seed $s
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict (PRIOR -> SIGNAL -> ROOM on matched) ===="
python results\_read417cl_cut.py
L "DONE 417cl v2"
