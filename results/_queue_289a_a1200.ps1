$ErrorActionPreference = "Continue"
$log = "results\_queue_289a_a1200.out"
"===== 289a addresses 1200 corpus $(Get-Date -Format o) =====" | Set-Content $log -Encoding utf8
if (Test-Path results\_stage289a_full_a1200.out) { Remove-Item -Force results\_stage289a_full_a1200.out }
cmd /c "python -u _stage289a_presupposition.py --train-steps 6000 --addresses 1200 --run-tag a1200 > `"results\_stage289a_full_a1200.out`" 2>&1"
"EXIT_FULL:$LASTEXITCODE" | Tee-Object -FilePath $log -Append
"QUEUE_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
# print wr n from decision if present
if (Test-Path results\stage289a_decision_a1200.json) {
  python -c "import json;from pathlib import Path;d=json.loads(Path('results/stage289a_decision_a1200.json').read_text(encoding='utf-8'));h=d.get('held_out',{});bp=h.get('blind_pair',{});bc=h.get('by_class') or {};print('overall',d.get('overall'));print('blind_pair',bp);wr=(h.get('wrong_relation') or (bc.get('wrong_relation') if isinstance(bc,dict) else None));print('wr',wr);print('gates',d.get('gates'));print('questions',d.get('train_questions') or d.get('n_by_class') or 'see log')"
}
