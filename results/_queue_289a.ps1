$ErrorActionPreference = "Continue"
$log = "results\_queue_289a.out"
"===== 289a blind_pair restart: 6000 -> addr $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8

function Invoke-289a([string]$ArgLine, [string]$OutFile) {
  if (Test-Path $OutFile) { Remove-Item -Force $OutFile -ErrorAction SilentlyContinue }
  cmd /c "python -u _stage289a_presupposition.py $ArgLine > `"$OutFile`" 2>&1"
  return $LASTEXITCODE
}

# smoke on thin wiki still has no wrong_relation; skip to full (user asked stop holdout + new file)
$code = Invoke-289a "--train-steps 6000" "results\_stage289a_full.out"
"EXIT_FULL:$code" | Tee-Object -FilePath $log -Append

$code = Invoke-289a "--train-steps 6000 --holdout address" "results\_stage289a_addrhold.out"
"EXIT_ADDR:$code" | Tee-Object -FilePath $log -Append

"QUEUE_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
