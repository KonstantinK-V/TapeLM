"""Night queue after 278 n16: stage 281 frames, then 280 if ceiling clears 0.375."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path
v0 = v55(v56).v21().v1
v2 = v0 / 'results'
v3 = v6(v57.v39[1]) if v38(v57.v39) > 1 else 0
v4 = 0.375

def alive(v7: v6) -> v5:
    if v7 <= 0:
        return False
    if v40.v22 == 'nt':
        import ctypes
        v23 = 1048576
        v24 = v67.v64.v58.v41(v23, False, v7)
        if v24:
            v67.v64.v58.v59(v24)
            return True
        return False
    try:
        v40.v42(v7, 0)
        return True
    except v25:
        return False

def run(v8: v43[v26], v9: v26) -> v6:
    v10 = v2 / v9
    v27(f"[queue] start {' '.v65(v8)} -> {v10}", flush=True)
    with v10.v44('w', encoding='utf-8') as v28:
        v29 = v60.v33(v8, cwd=v0, stdout=v28, stderr=v60.v61, text=True)
    v27(f'[queue] done {v8[1]} exit={v29.v11}', flush=True)
    return v29.v11

def main() -> v6:
    v2.v30(exist_ok=True)
    if v3 > 0:
        v27(f'[queue] waiting for pid {v3}', flush=True)
        while v45(v3):
            v66.v62(30)
        v27('[queue] predecessor finished', flush=True)
    else:
        v27('[queue] no wait (pid 0)', flush=True)
    v12 = v2 / 'stage278_decisionon.json'
    if v12.v31():
        (v2 / 'stage278_decisionon_n16.json').v46(v12.v63())
        v32 = v2 / 'stage278_minion.md'
        if v32.v31():
            (v2 / 'stage278_minion_n16.md').v46(v32.v63())
    v13 = v33([v57.v47, '-u', '_stage281_frames.py'], '_stage281_full.out')
    v14 = v2 / 'stage281_decision.json'
    if not v14.v31():
        v27('[queue] missing stage281_decision.json', flush=True)
        return v13 or 1
    v15 = v48.v34(v14.v49(encoding='utf-8'))
    v16 = v15.v35('ceiling_before', {})
    v17 = v15.v35('ceiling_after', {})
    v18 = v16.v35('reward') if v50(v16, v51) else None
    v19 = v17.v35('reward') if v50(v17, v51) else None
    v27(f"[queue] 281 {v15.v35('overall')} ceiling {v18} -> {v19}", flush=True)
    if v19 is not None and v19 >= v4:
        v27('[queue] ceiling cleared 0.375 — re-smoke 280', flush=True)
        v33([v57.v47, '-u', '_stage280_raw_exam.py', '--smoke', '--k-gap', '0.35'], '_stage280_smoke_after281.out')
        v36 = v2 / 'stage280_decision_fp.json'
        if v36.v31():
            v52 = v48.v34(v36.v49(encoding='utf-8'))
            v53 = v52.v35('teacher_ceiling_reward')
            v27(f"[queue] 280 smoke ceiling={v53} overall={v52.v35('overall')}", flush=True)
            if v53 is not None and v53 >= v4:
                v27('[queue] starting full 280 --hop fp', flush=True)
                v33([v57.v47, '-u', '_stage280_raw_exam.py', '--bc-episodes', '4000', '--rl-episodes', '3000', '--hop', 'fp'], '_stage280_full_fp.out')
    else:
        v27('[queue] ceiling below 0.375 — no 280 restart (FRAMES wall elsewhere)', flush=True)
    v27('[queue] night queue done', flush=True)
    return 0
if v20 == '__main__':
    raise v37(v54())