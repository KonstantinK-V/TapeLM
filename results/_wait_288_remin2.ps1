$ErrorActionPreference = "Continue"
$log = "results\_queue_288_remin2.out"
"===== wait old full (fixed), then min2 queue $(Get-Date -Format o) =====" | Tee-Object -FilePath $log -Append

function Get-288 {
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match '_stage288_repair\.py') }
}

while ($true) {
  $procs = @(Get-288)
  $addr = @($procs | Where-Object { $_.CommandLine -match '--holdout' })
  $full = @($procs | Where-Object { $_.CommandLine -notmatch '--holdout' -and $_.CommandLine -match '--train-steps' })

  if ($addr.Count -gt 0) {
    foreach ($a in $addr) {
      "KILL_MIXED_ADDRHOLD pid=$($a.ProcessId) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
      Stop-Process -Id $a.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
    break
  }

  if ($full.Count -eq 0 -and $addr.Count -eq 0) {
    "OLD_FULL_GONE $(Get-Date -Format o) - watch 90s for mixed addrhold" | Tee-Object -FilePath $log -Append
    for ($i = 0; $i -lt 9; $i++) {
      Start-Sleep -Seconds 10
      foreach ($a in @(Get-288 | Where-Object { $_.CommandLine -match '--holdout' })) {
        "KILL_MIXED_ADDRHOLD pid=$($a.ProcessId)" | Tee-Object -FilePath $log -Append
        Stop-Process -Id $a.ProcessId -Force -ErrorAction SilentlyContinue
      }
    }
    break
  }

  Start-Sleep -Seconds 30
}

foreach ($p in @(Get-288)) {
  if ($p.CommandLine -match '--holdout') {
    "STOP leftover addr $($p.ProcessId)" | Tee-Object -FilePath $log -Append
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  }
}

$dec = Get-ChildItem results\stage288_decision.json -ErrorAction SilentlyContinue
if ($dec) {
  Copy-Item -Force $dec.FullName results\stage288_decision_min3subset.json
  "SAVED min3subset decision" | Tee-Object -FilePath $log -Append
}

"===== restart min2 smoke -> full -> addr $(Get-Date -Format o) =====" | Tee-Object -FilePath $log -Append
python -u _stage288_repair.py --smoke > results\_stage288_smoke_min2.out 2>&1
"EXIT_SMOKE:$LASTEXITCODE" | Tee-Object -FilePath $log -Append
python -u _stage288_repair.py --train-steps 6000 --run-tag min2 > results\_stage288_full_min2.out 2>&1
"EXIT_FULL:$LASTEXITCODE" | Tee-Object -FilePath $log -Append
python -u _stage288_repair.py --train-steps 6000 --holdout address --run-tag min2 > results\_stage288_addrhold_min2.out 2>&1
"EXIT_ADDR:$LASTEXITCODE" | Tee-Object -FilePath $log -Append
"QUEUE_DONE" | Tee-Object -FilePath $log -Append
