$ErrorActionPreference = "Continue"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\Kostya\sote-letter-assembly"
$log = "results\_queue_416_stream.out"

function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

Set-Content -Path $log -Value "" -Encoding utf8
L "416 stream: place-mind on raw wiki window. Gate before numbers."
L "GATE ge2: mind > cosine rival. GATE one: refuse > refuse(ge2). No GATE-WO."

python _check416_stream.py
if ($LASTEXITCODE -ne 0) { L "CHECK FAIL"; exit 1 }
L "check OK"

$seeds = @(1337, 8642, 2890)
foreach ($s in $seeds) {
  L "==== 416 stream seed $s ===="
  python -u _audit416_stream.py --seed $s --steps 3000 --cpu `
    --save-mind "out/_mind_416_stream_s$s.pt" `
    --out "results/_stage416_stream.json"
  if ($LASTEXITCODE -ne 0) { L "EXIT $LASTEXITCODE s$s" } else { L "OK s$s" }
}

L "==== verdict ===="
python results\_read416_stream.py
L "DONE 416 stream"
