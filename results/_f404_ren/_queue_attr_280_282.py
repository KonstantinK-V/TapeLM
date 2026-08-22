"""Attribution queue: 280 casefold control -> 282 two-witness full -> --no-probe."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
v0 = v20(v45).v17().v1
v2 = v0 / 'results'

def log(v4: v18) -> None:
    v19(v4, flush=True)

def run(v5: v34[v18], v6: v20) -> v3:
    v21(f"RUN {' '.v55(v5)}")
    v7 = v46.v35.v22()
    v7['PYTHONUNBUFFERED'] = '1'
    with v6.v36('w', encoding='utf-8', errors='replace') as v23:
        v24 = v47.v37(v5, cwd=v18(v0), stdout=v47.v48, stderr=v47.v49, text=True, bufsize=1, env=v7)
        assert v24.v26 is not None
        for v25 in v24.v26:
            v54.v26.v50(v25)
            v54.v26.v51()
            v23.v50(v25)
            v23.v51()
        return v24.v38()

def snap_decision(v8: v18, v9: v20) -> None:
    v10 = v27(v2.v39(v8), key=lambda v24: v24.v56().v52)
    if not v10:
        v21(f'no decision matching {v8}')
        return
    v11 = v10[-1]
    v9.v28(v11.v40(encoding='utf-8'), encoding='utf-8')
    v12 = v41.v29(v9.v40(encoding='utf-8'))
    v13 = v12.v42('held_out') or {}
    v21(f"saved {v9.v53}: overall={v12.v42('overall')} rew={v13.v42('reward_total')} teach={v13.v42('teacher_reward_total') or v12.v42('teacher_ceiling_reward')} tie_abs={(v13.v42('tie') or {}).v42('abstain')} probe={v13.v42('probe_hit_rate')}")

def main() -> v3:
    v21('=== 1/3 control 280 with casefold (same knobs as m2b) ===')
    v14 = v30([v54.v43, '_stage280_raw_exam.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--hop', 'fp', '--min-mentions', '2', '--min-per-family', '8', '--run-tag', 'casefold'], v2 / '_stage280_full_fp_casefold.out')
    v31('stage280_decision_fp_casefold.json', v2 / 'stage280_decision_fp_casefold.json')
    if v14 != 0:
        v21(f'280 casefold exit={v14} — continue to 282 anyway? stopping')
        return v14
    v21('=== 2/3 282 full two-witness ===')
    v14 = v30([v54.v43, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2'], v2 / '_stage282_full_m2_twowitness.out')
    v31('stage282_decision.json', v2 / 'stage282_decision_full_m2_twowitness.json')
    if v14 != 0:
        return v14
    v21('=== 3/3 282 --no-probe ===')
    v14 = v30([v54.v43, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2', '--no-probe'], v2 / '_stage282_full_m2_noprobe.out')
    v15 = v2 / 'stage282_decision_noprobe.json'
    if v15.v32():
        v31('stage282_decision_noprobe.json', v2 / 'stage282_decision_full_m2_noprobe_twowitness.json')
    else:
        v10 = v27(v2.v39('stage282_decision*.json'), key=lambda v24: v24.v56().v52)
        if v10:
            (v2 / 'stage282_decision_full_m2_noprobe_twowitness.json').v28(v10[-1].v40(encoding='utf-8'), encoding='utf-8')
            v21(f'saved noprobe from {v10[-1].v53}')
    v21('=== attribution queue done ===')
    return v14
if v16 == '__main__':
    raise v33(v44())