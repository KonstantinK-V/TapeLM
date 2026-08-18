# Overnight: after m2b -> 282 smoke -> (soft OK) full + no-probe ablation.
# Hard fail = Traceback / no decision / exit!=0 on smoke. Soft fail still launches fulls.
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\Kostya\sote-letter-assembly"
$m2b = "results\stage280_decision_fp_m2b.json"
$smokeOut = "results\_stage282_smoke.out"
$review = "results\_stage282_overnight_review.txt"

function Stage280-M2bAlive {
  foreach ($p in @(Get-Process python -ErrorAction SilentlyContinue)) {
    try { $cl = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine } catch { $cl = "" }
    if ($cl -like '*_stage280_raw_exam.py*' -and $cl -like '*m2b*') { return $true }
  }
  return $false
}

Write-Host "=== overnight 282 orchestrator $(Get-Date -Format o) ==="
Write-Host "waiting for m2b..."
while ($true) {
  if (-not (Stage280-M2bAlive) -and (Test-Path $m2b)) { break }
  Start-Sleep -Seconds 90
}
Write-Host "m2b done"
python -c @"
import json
d=json.load(open(r'results/stage280_decision_fp_m2b.json',encoding='utf-8'))
h=d.get('held_out') or {}
t=h.get('tie') or {}
print('m2b overall', d.get('overall'))
print('ceil', d.get('teacher_ceiling_reward'), 'rew', h.get('reward_total'))
print('tie n', t.get('n'), 'teacher_abstain', t.get('teacher_abstain'), 'policy_abstain', t.get('abstain'))
print('stall', h.get('stall_rate'))
"@

Write-Host "starting 282 smoke..."
$env:PYTHONUNBUFFERED = "1"
python _stage282_mind.py --smoke 2>&1 | Tee-Object -FilePath $smokeOut
$smokeExit = $LASTEXITCODE
Write-Host "282 smoke exit=$smokeExit"

$dec = Get-ChildItem results\stage282_decision*.json -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$tail = ""
if (Test-Path $smokeOut) { $tail = (Get-Content $smokeOut -Tail 40) -join "`n" }
$hard = $false
$reason = ""
if ($smokeExit -ne 0) { $hard = $true; $reason = "smoke exit $smokeExit" }
if ($tail -match 'Traceback|CUDA out of memory|No module named') { $hard = $true; $reason = "traceback/oom/import in smoke log" }
if (-not $dec) { $hard = $true; $reason = "no stage282_decision*.json" }

$lines = @()
$lines += "smoke_exit=$smokeExit"
$lines += "decision=$($dec.FullName)"
$lines += "hard_fail=$hard reason=$reason"
if ($dec) {
  $j = Get-Content $dec.FullName -Raw | ConvertFrom-Json
  $lines += "overall=$($j.overall)"
  $lines += "gates=$($j.gates | ConvertTo-Json -Compress)"
}
$lines += "---- smoke tail ----"
$lines += $tail
$lines | Set-Content -Path $review -Encoding utf8
Write-Host ($lines -join "`n")

if ($hard) {
  Write-Host "HARD FAIL — stop before fulls. See $review"
  exit 2
}

Write-Host "soft-ok — launching 282 full (min-mentions 2)"
python _stage282_mind.py --bc-episodes 4000 --rl-episodes 3000 --min-mentions 2 2>&1 | Tee-Object -FilePath results\_stage282_full_m2.out
Write-Host "full exit=$LASTEXITCODE"
Copy-Item -Force (Get-ChildItem results\stage282_decision*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName results\stage282_decision_full_m2.json -ErrorAction SilentlyContinue

Write-Host "launching 282 full --no-probe"
python _stage282_mind.py --bc-episodes 4000 --rl-episodes 3000 --min-mentions 2 --no-probe 2>&1 | Tee-Object -FilePath results\_stage282_full_m2_noprobe.out
Write-Host "no-probe exit=$LASTEXITCODE"
Copy-Item -Force (Get-ChildItem results\stage282_decision*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName results\stage282_decision_full_m2_noprobe.json -ErrorAction SilentlyContinue

Write-Host "=== overnight 282 orchestrator done $(Get-Date -Format o) ==="
