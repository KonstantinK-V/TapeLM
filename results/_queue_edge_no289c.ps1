$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = Join-Path $PSScriptRoot "_queue_edge_ablation.out"
function L([string]$m) {
  $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}
L "WATCHDOG: no more 289c - wait e_rare 289, kill any 289c, then e_cos 289 only"

# Wait until e_rare decision exists (current 289 finishing under old queue)
while (-not (Test-Path (Join-Path $root 'results\stage289_decision_e_rare.json'))) {
  Start-Sleep -Seconds 30
}
L "e_rare 289 decision present"

# Old queue will start 289c e_rare - kill it and the queue
for ($i = 0; $i -lt 120; $i++) {
  $c289 = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and $_.CommandLine -match 'stage289c_audit' -and $_.CommandLine -match 'e_rare'
  }
  $q = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and $_.CommandLine -match 'queue_edge_ablation\.ps1'
  }
  if ($c289) {
    foreach ($p in $c289) {
      L "KILL 289c $($p.ProcessId)"
      Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
  }
  # also kill cmd parents of 289c
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and $_.CommandLine -match 'stage289c_audit' -and $_.CommandLine -match 'e_rare'
  } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

  if ($q -and -not (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
      $_.CommandLine -and $_.CommandLine -match 'stage289_derivation' -and $_.CommandLine -match 'e_rare'
    })) {
    # e_rare 289 done; stop old queue so it cannot chain e_cos+289c
    foreach ($p in $q) {
      L "KILL old queue $($p.ProcessId)"
      Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    break
  }
  Start-Sleep -Seconds 5
}

# belt: kill any remaining edge 289c / old queue
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
  $_.CommandLine -and (
    $_.CommandLine -match 'queue_edge_ablation\.ps1' -or
    ($_.CommandLine -match 'stage289c_audit' -and $_.CommandLine -match 'e_(rare|cos|both)')
  )
} | ForEach-Object {
  L "KILL leftover $($_.ProcessId)"
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

L "289 e_cos channels=same,cos"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --no-ladder --edge-channels same,cos --run-tag e_cos > results\_stage289_full_e_cos.out 2>&1"
L "EXIT_289_e_cos=$LASTEXITCODE"
L "QUEUE_DONE 289-only"
exit $LASTEXITCODE
