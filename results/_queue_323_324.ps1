$ErrorActionPreference = "Stop"
Set-Location C:\Users\Kostya\sote-letter-assembly
$env:PYTHONIOENCODING = "utf-8"
$log = "results\_queue_323_324.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
Set-Content -Path $log -Value "" -Encoding utf8
L "323/324 audits (torch-free), parallel with 322"

$common = @('--bytes','30000000','--frame-max','3','--sample','region','--window-lines','400')

L "==== 323 shared-filler vs cosine ===="
python -u _audit323_shared.py @common
if ($LASTEXITCODE -ne 0) { L "FAIL 323 exit $LASTEXITCODE"; exit $LASTEXITCODE }
L "OK 323"

L "==== 324 memory ceiling recall=1.0 ===="
python -u _audit324_memory.py @common
if ($LASTEXITCODE -ne 0) { L "FAIL 324 exit $LASTEXITCODE"; exit $LASTEXITCODE }
Copy-Item -Force results\_stage324_memory.json results\_stage324_memory_r1.0.json
L "OK 324 r1.0 -> results\_stage324_memory_r1.0.json"

L "==== 324 memory ceiling recall=0.5 ===="
python -u _audit324_memory.py @common --recall 0.5
if ($LASTEXITCODE -ne 0) { L "FAIL 324 recall0.5 exit $LASTEXITCODE"; exit $LASTEXITCODE }
Copy-Item -Force results\_stage324_memory.json results\_stage324_memory_r0.5.json
L "OK 324 r0.5 -> results\_stage324_memory_r0.5.json"
L "DONE"
