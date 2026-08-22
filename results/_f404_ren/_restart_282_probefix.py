"""After probe-index fix: 282 smoke -> full -> --no-probe. m2b already done."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
v0 = v9(v45).v17().v1
v2 = v0 / 'results'

def log(v4: v18) -> None:
    v19(v4, flush=True)

def run(v5: v32[v18], v6: v9) -> v3:
    v20(f"RUN {' '.v54(v5)}")
    v7 = v46.v33.v21()
    v7['PYTHONUNBUFFERED'] = '1'
    with v6.v34('w', encoding='utf-8', errors='replace') as v22:
        v23 = v47.v35(v5, cwd=v18(v0), stdout=v47.v48, stderr=v47.v49, text=True, bufsize=1, env=v7)
        assert v23.v25 is not None
        for v24 in v23.v25:
            v53.v25.v50(v24)
            v53.v25.v51()
            v22.v50(v24)
            v22.v51()
        return v23.v36()

def latest_decision() -> v9 | None:
    v8 = v26(v2.v37('stage282_decision*.json'), key=lambda v23: v23.v55().v52)
    return v8[-1] if v8 else None

def summarize(v10: v9 | None, v11: v18) -> None:
    if not v10:
        v20(f'{v11}: no decision')
        return
    v12 = v38.v27(v10.v39(encoding='utf-8'))
    v13 = v12.v40('held_out') or {}
    v20(f"{v11} overall={v12.v40('overall')} rew={v13.v40('reward_total')} teach={v13.v40('teacher_reward_total')} cov={v13.v40('coverage_all')} probe_hit={v13.v40('probe_hit_rate')} typed_conflict={v13.v40('conflict_when_tie')}")
    v20(f"  gates={v38.v41(v12.v40('gates'), ensure_ascii=True)}")
    v14 = v2 / f'stage282_decision_{v11}.json'
    v14.v28(v38.v41(v12, indent=2), encoding='utf-8')

def main() -> v3:
    v20('=== 282 probe-fix restart ===')
    v15 = v29([v53.v42, '_stage282_mind.py', '--smoke'], v2 / '_stage282_smoke.out')
    v30(v43(), 'smoke_probefix')
    if v15 != 0:
        v20(f'smoke hard fail exit={v15}')
        return v15
    v20('launching full')
    v29([v53.v42, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2'], v2 / '_stage282_full_m2.out')
    v30(v43(), 'full_m2_probefix')
    v20('launching --no-probe')
    v29([v53.v42, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2', '--no-probe'], v2 / '_stage282_full_m2_noprobe.out')
    v30(v43(), 'full_m2_noprobe_probefix')
    v20('=== done ===')
    return 0
if v16 == '__main__':
    raise v31(v44())