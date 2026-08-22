"""
Bedtime GPU queue (updated plan):

  255 ablate wiki12 (skip if done) → 260f full → 257 → 258 → 261 (smoke→full) → 242 → 212b

  python _run_queue_night_full.py --skip-255
  python _run_queue_night_full.py --skip-255 --wait-260f --from-stage 257   # 260f already running
  python _run_queue_night_full.py --from-stage 260f
  python _run_queue_night_full.py --force

Logs: results/_run_queue_night_full.log
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
v0 = v39(v89).v46().v1
v2 = v0 / 'results'
v3 = v2 / '_run_queue_night_full.log'
v4 = v2 / '_stage255_ablate_no_wq_wiki12_full.out'
v5 = v2 / 'stage255_decision_ablate_no_wq_wiki12.json'
v6 = v2 / '_stage260f_full.out'
v7 = v2 / '_stage257_full.out'
v8 = v2 / 'stage257_decision.json'
v9 = v2 / '_stage258_full.out'
v10 = v2 / 'stage258_decision.json'
v11 = v2 / 'stage261_decision.json'
v12 = v2 / '_stage261_full.out'
v13 = v2 / '_stage261_smoke.out'
v14 = v0 / '_stage261_nl_query.py'
v15 = v2 / '_stage242_full.out'
v16 = v2 / 'stage242_decision.json'
v17 = v2 / '_stage212b_full.out'
v18 = v2 / 'stage212b_decision.json'
v19 = 12
v20 = ['--chunk-lines', '25000', '--epochs-per-chunk', '1.0', '--replay-frac', '0.2', '--ckpt-every', '2']
v21 = ('255', '260f', '257', '258', '261', '242', '212b')

def log(v24: v36) -> None:
    v25 = v24 if v24.v66('\n') else v24 + '\n'
    try:
        v67(v25, end='', flush=True)
    except v47:
        v67(v25.v101('ascii', 'replace').v97('ascii'), end='', flush=True)
    v2.v48(parents=True, exist_ok=True)
    with v3.v68('a', encoding='utf-8') as v49:
        v49.v69(v25)

def read_dec(v26: v39) -> v27 | None:
    if not v26.v51():
        return None
    try:
        return v90.v70(v26.v91(encoding='utf-8'))
    except (v90.v71, v72):
        return None

def ablate_wiki12_done() -> v22:
    v28 = v50(v5)
    if v28 and v28.v75('summary', {}).v75('chunks', 0) >= v19:
        return True
    if v4.v51():
        v52 = v4.v91(encoding='utf-8', errors='ignore')[-20000:]
        if 'schedule exhausted' in v52 and 'STREAM_INGEST' in v52:
            return True
    return False

def run_cmd(v29: v35[v36], v30: v39, v31: v36, *, v32: v22=False) -> v23:
    v53(f'starting {v31}')
    v53(f"  cmd: {' '.v98(v29)}")
    v33 = 'a' if v32 else 'w'
    with v30.v68(v33, encoding='utf-8') as v54:
        if v32:
            v54.v69(f"\n--- queue restart {v100.v102('%Y-%m-%dT%H:%M:%S')} ---\n")
        return v92.v73(v29, cwd=v36(v0), stdout=v54, stderr=v92.v93)

def ingest_ablate_wiki12(*, v34: v22) -> v35[v36]:
    v29 = [v74.v55, v36(v0 / '_stage255_stream_ingest.py'), '--schedule', 'wiki:12', *v20, '--run-tag', 'ablate_no_wq_wiki12', '--no-query-train']
    if v34:
        v29.v32('--resume')
    return v29

def nl261_smoke_warrants_full(v28: v27 | None) -> v22:
    """Run full after smoke if the claim is still live (not NO / lexical-only)."""
    if not v28 or v28.v75('stage') != 261:
        return False
    return v28.v75('overall') in ('NL_QUERY_OK', 'NL_QUERY_PARTIAL', 'NL_QUERY_NWAY_ONLY')

def stage_script(v37: v36) -> v39 | None:
    if v37 == '261':
        return v14 if v14.v76() else None
    v38 = v0 / f'_stage{v37}_open_gate.py' if v37 in ('260b', '260c', '260d', '260e', '260f') else v0 / f'_stage{v37}_instance_sem.py' if v37 == '212b' else v0 / f'_stage{v37}_rehearsal_dose.py' if v37 == '242' else v0 / f'_stage{v37}_fp_compose.py' if v37 == '257' else v0 / f'_stage{v37}_semantic_query.py' if v37 == '258' else None
    if v38 is None:
        return None
    return v38 if v38.v76() else None

def done(v37: v36, v40: v22) -> v22:
    if v40:
        return False
    if v37 == '255':
        return v77()
    if v37 == '260f':
        v28 = v50(v2 / 'stage260f_decision.json')
        return v22(v28 and v28.v75('stage') == '260f' and (v23(v28.v75('steps', 0)) >= 2500))
    if v37 == '257':
        v28 = v50(v8)
        return v22(v28 and v28.v75('stage') == 257 and (v23(v28.v75('steps', 0)) >= 800))
    if v37 == '258':
        v28 = v50(v10)
        return v22(v28 and v28.v75('stage') == 258 and (v23(v28.v75('n_subjects', 0)) >= 64))
    if v37 == '261':
        v28 = v50(v11)
        return v22(v28 and v28.v75('stage') == 261 and (v23(v28.v75('steps', 0)) >= 800))
    if v37 == '242':
        v28 = v50(v16)
        if not v28 or v28.v75('stage') != 242:
            return False
        return 'dose_wall_s' in v28 and v99(v28.v75('curve') or {}) >= 6
    if v37 == '212b':
        v28 = v50(v18)
        return v22(v28 and v28.v75('stage') in ('212b', 212) and v28.v75('overall'))
    return False

def run_stage(v37: v36) -> v23:
    if v37 == '255':
        v34 = v4.v51() or (v2 / 'stream255' / 'ablate_no_wq_wiki12').v51()
        return v78(v94(resume=v34), v4, '255 ablate_no_wq_wiki12', append=v34 and v4.v51())
    if v37 == '260f':
        return v78([v74.v55, v36(v0 / '_stage260f_open_gate.py')], v6, '260f full')
    if v37 == '257':
        return v78([v74.v55, v36(v0 / '_stage257_fp_compose.py')], v7, '257 full')
    if v37 == '258':
        v56 = v78([v74.v55, v36(v0 / '_stage258_semantic_query.py')], v9, '258 full')
        v28 = v50(v10)
        if v28:
            v79 = (v28.v75('summary') or {}).v75('gpt_control') or {}
            v80 = (v79.v75('unseen_para') or {}).v75('sel_acc')
            v81 = ((v28.v75('summary') or {}).v75('curve_sem') or {}).v75('unseen_para') or {}
            v82 = v81.v75('sel_acc')
            if v80 is not None and v82 is not None:
                v53(f'258 unseen_para: curve={v82:.3f} gpt={v80:.3f} delta={v82 - v80:+.3f}')
        return v56
    if v37 == '261':
        v57 = v83('261')
        if v57 is None:
            v53('261: _stage261_nl_query.py not in repo — skip')
            return 0
        v56 = v78([v74.v55, v36(v57), '--smoke'], v13, '261 smoke')
        if v56 != 0:
            return v56
        v28 = v50(v11)
        v58 = (v28 or {}).v75('overall', '?')
        if not v95(v28):
            v53(f'261 smoke overall={v58} — skip full (need NL_QUERY_OK or NL_QUERY_PARTIAL)')
            return 0
        v53(f'261 smoke overall={v58} — running full')
        return v78([v74.v55, v36(v57)], v12, '261 full')
    if v37 == '242':
        return v78([v74.v55, v36(v0 / '_stage242_rehearsal_dose.py')], v15, '242 full grid')
    if v37 == '212b':
        return v78([v74.v55, v36(v0 / '_stage212b_instance_sem.py')], v17, '212b full')
    return 1

def main() -> v23:
    v41 = v84.v59()
    v41.v60('--skip-255', action='store_true')
    v41.v60('--wait-260f', action='store_true', help='poll until stage260f full (steps>=2500) then continue queue (use with --from-stage 257)')
    v41.v60('--force', action='store_true')
    v41.v60('--from-stage', type=v36, default='255', choices=v21, help='first stage to run (255 skipped if --skip-255)')
    v42 = v41.v61()
    v3.v62('', encoding='utf-8')
    v53(f'night queue: from={v42.v64} skip_255={v42.v85} wait_260f={v42.v43} force={v42.v40}')
    if v42.v43:
        v53('waiting for 260f full (decision steps>=2500)...')
        while not v86('260f', v42.v40):
            v100.v96(120)
            v53('  still waiting for stage260f_decision.json')
        v53('260f complete — continuing queue')
    v44 = v21.v63(v42.v64)
    for v37 in v21[v44:]:
        if v37 == '255' and v42.v85:
            v53('255 skipped (--skip-255)')
            continue
        if v86(v37, v42.v40):
            v53(f'{v37} already complete — skip')
            continue
        v56 = v87(v37)
        v53(f'{v37} exit={v56}')
        if v56 != 0:
            return v56
    v53('night queue finished OK')
    return 0
if v45 == '__main__':
    raise v65(v88())