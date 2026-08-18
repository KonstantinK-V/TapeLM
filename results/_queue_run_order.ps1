$ErrorActionPreference = "Continue"
$log = "results\_queue_run_order.out"
"===== RUN_ORDER queue $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8
Set-Location $PSScriptRoot\..

function Tee($m) { $m | Tee-Object -FilePath $log -Append }

# Step 0 already done (IDENTICAL). Start at 1.
Tee "STEP1 289a w150auc $(Get-Date -Format o)"
cmd /c "python -u _stage289a_presupposition.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150auc > results\_stage289a_full_w150auc.out 2>&1"
Tee "EXIT_STEP1:$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Tee "STOP after step1"; exit $LASTEXITCODE }

Tee "STEP2 289a address holdout $(Get-Date -Format o)"
cmd /c "python -u _stage289a_presupposition.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --holdout address --run-tag w150auc > results\_stage289a_full_w150auc_addr.out 2>&1"
Tee "EXIT_STEP2:$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Tee "STOP after step2"; exit $LASTEXITCODE }

Tee "STEP3 289 smoke $(Get-Date -Format o)"
cmd /c "python -u _stage289_derivation.py --smoke > results\_stage289_smoke.out 2>&1"
Tee "EXIT_STEP3:$LASTEXITCODE"
$qline = Select-String -Path results\_stage289_smoke.out -Pattern 'questions \{' | Select-Object -First 1
if ($qline) { Tee "SMOKE_QUESTIONS: $($qline.Line.Trim())" } else { Tee "SMOKE_QUESTIONS: MISSING"; Tee "STOP after step3 — no questions line"; exit 1 }
# Require count/compare/lookup each >= 5 (MIN_ANSWERED)
$txt = $qline.Line
$ok = $true
foreach ($verb in @('count','compare','lookup')) {
  if ($txt -match "`"$verb`":\s*(\d+)") {
    $n = [int]$Matches[1]
    Tee "SMOKE_$verb=$n"
    if ($n -lt 5) { $ok = $false }
  } else {
    Tee "SMOKE_$verb=MISSING"
    $ok = $false
  }
}
if (-not $ok) {
  Tee "STOP after step3 — raise --train-lines; do not touch the model"
  exit 2
}

Tee "STEP4a 289 full $(Get-Date -Format o)"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 > results\_stage289_full_w150.out 2>&1"
Tee "EXIT_STEP4a:$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Tee "STOP after step4a"; exit $LASTEXITCODE }

Tee "STEP4b 289 --no-derivation $(Get-Date -Format o)"
cmd /c "python -u _stage289_derivation.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 --no-derivation > results\_stage289_full_w150_noderiv.out 2>&1"
Tee "EXIT_STEP4b:$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Tee "STOP after step4b"; exit $LASTEXITCODE }

Tee "STEP5 289c $(Get-Date -Format o)"
cmd /c "python -u _stage289c_audit.py --train-steps 6000 --addresses 1200 --wiki-bytes 150000000 --train-lines 120000 --eval-lines 60000 --run-tag w150 > results\_stage289c_full_w150.out 2>&1"
Tee "EXIT_STEP5:$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Tee "STOP after step5"; exit $LASTEXITCODE }

Tee "STEP6a 286 relational $(Get-Date -Format o)"
cmd /c "python -u _stage286_evidence.py --train-steps 6000 --mind relational --run-tag speedcheck > results\_stage286_full_speedcheck.out 2>&1"
Tee "EXIT_STEP6a:$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Tee "STOP after step6a"; exit $LASTEXITCODE }

Tee "STEP6b 288 min3subset $(Get-Date -Format o)"
cmd /c "python -u _stage288_repair.py --train-steps 6000 --run-tag min3subset > results\_stage288_full_min3subset.out 2>&1"
Tee "EXIT_STEP6b:$LASTEXITCODE"

Tee "QUEUE_DONE $(Get-Date -Format o)"
exit $LASTEXITCODE
