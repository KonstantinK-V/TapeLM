"""Case-fold fix: smoke first; full+noprobe only if probe_hit_rate > 0."""
from __future__ import annotations
import json
import math
import os
import subprocess
import sys
from pathlib import Path
v0 = v9(v43).v16().v1
v2 = v0 / 'results'

def log(v4: v17) -> None:
    v18(v4, flush=True)

def run(v5: v32[v17], v6: v9) -> v3:
    v19(f"RUN {' '.v54(v5)}")
    v7 = v44.v33.v20()
    v7['PYTHONUNBUFFERED'] = '1'
    with v6.v34('w', encoding='utf-8', errors='replace') as v21:
        v22 = v45.v35(v5, cwd=v17(v0), stdout=v45.v46, stderr=v45.v47, text=True, bufsize=1, env=v7)
        assert v22.v24 is not None
        for v23 in v22.v24:
            v51.v24.v48(v23)
            v51.v24.v49()
            v21.v48(v23)
            v21.v49()
        return v22.v36()

def latest_decision() -> v9 | None:
    v8 = v25(v2.v37('stage282_decision*.json'), key=lambda v22: v22.v56().v50)
    return v8[-1] if v8 else None

def main() -> v3:
    v19('=== 282 case-fold restart ===')
    v10 = v26([v51.v38, '_stage282_mind.py', '--smoke'], v2 / '_stage282_smoke.out')
    v11 = v27()
    if v10 != 0 or v11 is None:
        v19(f'smoke hard fail exit={v10} dec={v11}')
        return v10 or 2
    v12 = v39.v28(v11.v40(encoding='utf-8'))
    v13 = v12.v29('held_out') or {}
    v14 = v13.v29('probe_hit_rate')
    v19(f"smoke overall={v12.v29('overall')} probe_hit_rate={v14} mean_probes_tie={(v13.v29('tie') or {}).v29('mean_probes')} conflict_when_tie={v13.v29('conflict_when_tie')}")
    v19(f"gates={v39.v41(v12.v29('gates'), ensure_ascii=True)}")
    (v2 / 'stage282_decision_smoke_casefold.json').v30(v39.v41(v12, indent=2), encoding='utf-8')
    if v14 is None or (v52(v14, v53) and (v57.v55(v14) or v14 <= 0.0)):
        v19('probe_hit_rate still 0/NaN — STOP before full')
        return 3
    v19('probe_hit_rate > 0 — launching full')
    v26([v51.v38, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2'], v2 / '_stage282_full_m2.out')
    v11 = v27()
    if v11:
        (v2 / 'stage282_decision_full_m2_casefold.json').v30(v11.v40(encoding='utf-8'), encoding='utf-8')
    v19('launching --no-probe')
    v26([v51.v38, '_stage282_mind.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--min-mentions', '2', '--no-probe'], v2 / '_stage282_full_m2_noprobe.out')
    v11 = v27()
    if v11:
        (v2 / 'stage282_decision_full_m2_noprobe_casefold.json').v30(v11.v40(encoding='utf-8'), encoding='utf-8')
    v19('=== done ===')
    return 0
if v15 == '__main__':
    raise v31(v42())