$ErrorActionPreference = "Continue"
$log = "results\_queue_288_refuse.out"
"===== 288 refuse: smoke -> full -> addr $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8

function Invoke-288([string]$ArgLine, [string]$OutFile) {
  if (Test-Path $OutFile) { Remove-Item -Force $OutFile -ErrorAction SilentlyContinue }
  $cmd = "python -u _stage288_repair.py $ArgLine > `"$OutFile`" 2>&1"
  cmd /c $cmd
  return $LASTEXITCODE
}

$code = Invoke-288 "--smoke --run-tag refuse" "results\_stage288_smoke_refuse.out"
"EXIT_SMOKE:$code" | Tee-Object -FilePath $log -Append

$code = Invoke-288 "--train-steps 6000 --run-tag refuse" "results\_stage288_full_refuse.out"
"EXIT_FULL:$code" | Tee-Object -FilePath $log -Append

$code = Invoke-288 "--train-steps 6000 --holdout address --run-tag refuse" "results\_stage288_addrhold_refuse.out"
"EXIT_ADDR:$code" | Tee-Object -FilePath $log -Append

"QUEUE_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
