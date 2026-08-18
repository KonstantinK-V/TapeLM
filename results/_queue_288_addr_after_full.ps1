$log = "results\_queue_288_remin2.out"
"===== wait full min2 alone, then addrhold $(Get-Date -Format o) =====" | Tee-Object -FilePath $log -Append
while ($true) {
  $full = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -match '_stage288_repair\.py --train-steps 6000 --run-tag min2' -and $_.CommandLine -notmatch 'holdout'
  })
  if ($full.Count -eq 0) { break }
  Start-Sleep -Seconds 30
}
"FULL_MIN2_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
# keep decision if written
if (Test-Path results\stage288_decision_min2.json) {
  "have stage288_decision_min2.json" | Tee-Object -FilePath $log -Append
}
python -u _stage288_repair.py --train-steps 6000 --holdout address --run-tag min2 > results\_stage288_addrhold_min2.out 2>&1
"EXIT_ADDR:$LASTEXITCODE" | Tee-Object -FilePath $log -Append
"QUEUE_DONE" | Tee-Object -FilePath $log -Append
