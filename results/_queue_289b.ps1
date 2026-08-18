$ErrorActionPreference = "Continue"
$log = "results\_queue_289b.out"
"===== 289b smoke -> full -> addr $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8

# 288 refuse may still hold the GPU; wait so packing does not thrash.
while (@(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and $_.CommandLine -match '_stage288_repair\.py'
}).Count -gt 0) {
  "WAIT_288 $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
  Start-Sleep -Seconds 60
}

function Invoke-289b([string]$ArgLine, [string]$OutFile) {
  if (Test-Path $OutFile) { Remove-Item -Force $OutFile -ErrorAction SilentlyContinue }
  cmd /c "python -u _stage289b_mind_tape.py $ArgLine > `"$OutFile`" 2>&1"
  return $LASTEXITCODE
}

$code = Invoke-289b "--smoke" "results\_stage289b_smoke.out"
"EXIT_SMOKE:$code" | Tee-Object -FilePath $log -Append

$code = Invoke-289b "" "results\_stage289b_full.out"
"EXIT_FULL:$code" | Tee-Object -FilePath $log -Append

$code = Invoke-289b "--holdout address" "results\_stage289b_addrhold.out"
"EXIT_ADDR:$code" | Tee-Object -FilePath $log -Append

"QUEUE_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
