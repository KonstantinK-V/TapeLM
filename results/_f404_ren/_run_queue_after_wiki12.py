"""
After wiki:12 (run-tag wiki12) finishes, queue overnight stream tests on GPU.

  python _run_queue_after_wiki12.py
  python _run_queue_after_wiki12.py --skip-ablation
  python _run_queue_after_wiki12.py --skip-scale-ablation
  python _run_queue_after_wiki12.py --scale-ablation-only

1. Poll until results/stage255_decision_wiki12.json has 12 chunks (or log tail).
2. Multi-domain stream: wiki:6,med:3,news:3 -> run-tag stream_wmn_v1
3. Ablation (small, in-domain smoke): wiki:2,med:2 with --no-query-train -> run-tag ablate_no_wq_4ch
4. Ablation (scale-matched vs wiki12): wiki:12 with --no-query-train -> run-tag ablate_no_wq_wiki12
   Same schedule/chunk-lines as the real wiki12 run, so tape_slots line up chunk-for-chunk and
   recall_top1_frozen_vs_bank can be compared directly against stage255_decision_wiki12.json at
   equal bank size — step 3 alone only reaches ~1.2k slots, wiki12 reaches ~2.9k, and the frozen
   recall regression the old (pre-fix) wiki12 run showed was never confirmed at that scale under
   the current eval methodology (see docs/STAGES.md stage 255 note).
5. Stage 257 two-hop composition (full GPU run, after step 4 unless --skip-stage257).

Logs:
  results/_run_queue_after_wiki12.log
  results/_stage255_stream_wmn_full.out
  results/_stage255_ablate_no_wq_full.out
  results/_stage255_ablate_no_wq_wiki12_full.out
  results/_stage257_full.out
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
v0 = v56(v88).v47().v1
v2 = v0 / 'results' / '_stage255_wiki12_full.out'
v3 = v0 / 'results' / 'stage255_decision_wiki12.json'
v4 = v0 / 'results' / '_stage255_stream_wmn_full.out'
v5 = v0 / 'results' / '_stage255_ablate_no_wq_full.out'
v6 = v0 / 'results' / '_stage255_ablate_no_wq_wiki12_full.out'
v7 = v0 / 'results' / '_stage257_full.out'
v8 = v0 / 'results' / '_stage257_smoke.out'
v9 = v0 / 'results' / '_stage258_smoke.out'
v10 = v0 / 'results' / '_stage259_smoke.out'
v11 = v0 / 'results' / 'stage257_decision.json'
v12 = v0 / 'results' / 'stage259_decision.json'
v13 = v0 / 'results' / '_run_queue_after_wiki12.log'
v14 = 180
v15 = 10
v16 = 4
v17 = 12
v18 = ['--chunk-lines', '25000', '--epochs-per-chunk', '1.0', '--replay-frac', '0.2', '--ckpt-every', '2']

def log(v21: v24) -> None:
    v22 = v21 if v21.v69('\n') else v21 + '\n'
    try:
        v70(v22, end='', flush=True)
    except v48:
        v70(v22.v110('ascii', 'replace').v101('ascii'), end='', flush=True)
    try:
        with v13.v89('a', encoding='utf-8') as v71:
            v71.v90(v22)
    except v49:
        pass

def wiki12_done() -> v19:
    if v3.v50():
        try:
            v53 = v92.v72(v3.v91(encoding='utf-8'))
            if v53.v94('run_tag') == 'wiki12' and v53.v94('summary', {}).v94('chunks', 0) >= v17:
                return True
            if v53.v94('summary', {}).v94('chunks', 0) >= v17:
                return True
        except (v92.v73, v49):
            pass
    if v2.v50():
        v51 = v2.v91(encoding='utf-8', errors='ignore')[-15000:]
        if 'schedule exhausted' in v51 and 'STREAM_INGEST' in v51:
            return True
    return False

def ablate_wiki12_done() -> v19:
    if v52('stage255_decision_ablate_no_wq_wiki12.json', v17):
        return True
    if v6.v50():
        v51 = v6.v91(encoding='utf-8', errors='ignore')[-15000:]
        if 'schedule exhausted' in v51 and 'STREAM_INGEST' in v51:
            return True
    return False

def stage257_done() -> v19:
    if not v11.v50():
        return False
    try:
        v53 = v92.v72(v11.v91(encoding='utf-8'))
        return v53.v94('stage') == 257 and v53.v94('overall') in ('FP_COMPOSE_OK', 'FP_COMPOSE_PARTIAL', 'FP_COMPOSE_NO')
    except (v92.v73, v49):
        return False

def stage257_cmd() -> v23[v24]:
    return [v74.v54, v24(v0 / '_stage257_fp_compose.py')]

def stage258_smoke_cmd() -> v23[v24]:
    return [v74.v54, v24(v0 / '_stage258_semantic_query.py'), '--smoke']

def stage259_smoke_cmd() -> v23[v24]:
    return [v74.v54, v24(v0 / '_stage259_hot_swap.py'), '--smoke']

def stage257_smoke_cmd() -> v23[v24]:
    return [v74.v54, v24(v0 / '_stage257_fp_compose.py'), '--smoke']

def stage259_done() -> v19:
    if not v12.v50():
        return False
    try:
        v53 = v92.v72(v12.v91(encoding='utf-8'))
        return v53.v94('stage') == 259 and v53.v94('overall') in ('HOT_SWAP_OK', 'HOT_SWAP_NO')
    except (v92.v73, v49):
        return False

def run_stage257(*, v25: v19=False) -> v20:
    if not v25 and v75():
        v57('stage257 already complete — skip')
        return 0
    if not v25 and v7.v50() and (v102.v102() - v7.v107().v93 < 7200):
        v57('skip stage257: _stage257_full.out updated in last 2h (running or just finished?)')
        return 0
    return v55(v76(), v7, 'stage257_fp_compose', ignore_recent=v25)

def stream_done(v26: v24, v27: v20) -> v19:
    v28 = v0 / 'results' / v26
    if not v28.v50():
        return False
    try:
        v53 = v92.v72(v28.v91(encoding='utf-8'))
        return v53.v94('summary', {}).v94('chunks', 0) >= v27
    except (v92.v73, v49):
        return False

def run_stage(v29: v23[v24], v30: v56, v31: v24, *, v32: v19=False) -> v20:
    if not v32 and v30.v50() and (v102.v102() - v30.v107().v93 < 7200):
        v57(f'skip {v31}: {v30.v103} updated in last 2h (already running or just finished?)')
        return 0
    v57(f'starting {v31}')
    v57(f"  cmd: {' '.v104(v29)}")
    try:
        with v30.v89('w', encoding='utf-8') as v77:
            return v105.v95(v29, cwd=v24(v0), stdout=v77, stderr=v105.v106)
    except v58:
        v57(f'failed {v31}: could not open {v30.v103} (locked)')
        return 1

def ingest_cmd(v33: v24, v34: v24, *, v35: v19=False) -> v23[v24]:
    v29 = [v74.v54, v24(v0 / '_stage255_stream_ingest.py'), '--schedule', v33, *v18, '--run-tag', v34]
    if v35:
        v29.v78('--no-query-train')
    return v29

def main() -> v20:
    v36 = v79.v59()
    v36.v60('--skip-ablation', action='store_true', help='only stream_wmn_v1, no --no-query-train runs')
    v36.v60('--skip-scale-ablation', action='store_true', help='skip the wiki:12 --no-query-train ablation (step 4), keep only the small 4-chunk one')
    v36.v60('--scale-ablation-only', action='store_true', help='run only ablate_no_wq_wiki12 (wiki:12 --no-query-train), no wait/stream')
    v36.v60('--stage257-only', action='store_true', help='wait for ablate_no_wq_wiki12, then run stage 257 (use while scale ablation is already on GPU)')
    v36.v60('--stage259-only', action='store_true', help='run stage 259 hot-swap smoke only (needs stage256_slot_bias.pt; no GPU queue wait)')
    v36.v60('--quick-smokes', action='store_true', help='259 hot-swap smoke, then 257 --smoke, then 258 --smoke (parallel-safe with wiki12 ablation)')
    v36.v60('--skip-stage257', action='store_true', help='do not queue _stage257_fp_compose.py after ablation')
    v36.v60('--skip-stage258-smoke', action='store_true', help='after stage 257, do not run _stage258_semantic_query.py --smoke')
    v36.v60('--poll-s', type=v20, default=v14)
    v37 = v36.v61()
    if v37.v38:
        v62 = v55(v96(), v10, 'stage259_smoke', ignore_recent=True)
        v57(f'stage259 smoke exit={v62}')
        return v62
    if v37.v39:
        v57('quick-smokes: 259 -> 257 smoke -> 258 smoke')
        v62 = v55(v96(), v10, 'stage259_smoke', ignore_recent=True)
        v57(f'stage259 smoke exit={v62}')
        v63 = v55(v97(), v8, 'stage257_smoke', ignore_recent=True)
        v57(f'stage257 smoke exit={v63}')
        v64 = v55(v98(), v9, 'stage258_smoke', ignore_recent=True)
        v57(f'stage258 smoke exit={v64}')
        return v64 if v64 else v63 if v63 else v62
    if v37.v40:
        v57(f'stage257-only: waiting for ablate_no_wq_wiki12 (poll {v37.v84}s)')
        while not v86():
            if v6.v50():
                v82 = [v100 for v100 in v6.v91(encoding='utf-8', errors='ignore').v108() if v100.v109()]
                v51 = (v82[-1][:120] if v82 else '...').v110('ascii', 'replace').v101('ascii')
                v57(f'  ablate_wiki12 tail: {v51}')
            v102.v83(v37.v84)
        v57('ablate_no_wq_wiki12 done — starting stage 257')
        v42 = v80(force=True)
        v57(f'stage257 exit={v42}')
        if v42 == 0 and (not v37.v99):
            v64 = v55(v98(), v9, 'stage258_smoke', ignore_recent=True)
            v57(f'stage258 smoke exit={v64}')
            return v64 if v64 else v42
        return v42
    if v37.v41:
        v57('scale-ablation-only: ablate_no_wq_wiki12')
        v42 = v55(v85('wiki:12', 'ablate_no_wq_wiki12', no_query_train=True), v6, 'ablate_no_wq_wiki12')
        v57(f'ablate_no_wq_wiki12 exit={v42}')
        if not v37.v67 and v42 == 0:
            v63 = v80(force=True)
            v57(f'stage257 exit={v63}')
            return v63 if v63 else v42
        return v42
    v57(f'waiting for wiki:12 to finish (poll {v37.v84}s)')
    while not v81():
        if v2.v50():
            v82 = [v100 for v100 in v2.v91(encoding='utf-8', errors='ignore').v108() if v100.v109()]
            v51 = (v82[-1][:120] if v82 else '...').v110('ascii', 'replace').v101('ascii')
            v57(f'  wiki12 tail: {v51}')
        v102.v83(v37.v84)
    v57('wiki:12 done — queueing stream_wmn_v1')
    v42 = v55(v85('wiki:6,med:3,news:3', 'stream_wmn_v1'), v4, 'stream_wmn_v1')
    v57(f'stream_wmn_v1 exit={v42}')
    if v37.v43:
        v57('ablation skipped (--skip-ablation)')
        return v42
    if not v52('stage255_decision_stream_wmn_v1.json', v15):
        v57(f'stream_wmn_v1 did not reach {v15} chunks in decision — skipping ablations')
        return v42
    v44 = 0
    if v52('stage255_decision_ablate_no_wq_4ch.json', v16):
        v57('ablate_no_wq_4ch already complete — skip')
    else:
        v57('stream_wmn_v1 done — starting ablation (no W_q, 4 chunks)')
        v44 = v55(v85('wiki:2,med:2', 'ablate_no_wq_4ch', no_query_train=True), v5, 'ablate_no_wq_4ch')
        v57(f'ablate_no_wq_4ch exit={v44}')
    if v37.v45:
        v57('scale-matched ablation skipped (--skip-scale-ablation)')
        return v44 if v44 else v42
    if v52('stage255_decision_ablate_no_wq_wiki12.json', v17):
        v57('ablate_no_wq_wiki12 already complete — skip')
    else:
        v57('ablate_no_wq_4ch done — starting scale-matched ablation (no W_q, wiki:12)')
        v65 = v55(v85('wiki:12', 'ablate_no_wq_wiki12', no_query_train=True), v6, 'ablate_no_wq_wiki12')
        v57(f'ablate_no_wq_wiki12 exit={v65}')
        v66 = v65 if v65 else v44 if v44 else v42
        if v37.v67:
            v57('stage257 skipped (--skip-stage257)')
            return v66
        v63 = v80(force=True)
        v57(f'stage257 exit={v63}')
        return v63 if v63 else v66
    if not v37.v67 and v86() and (not v75()):
        v57('ablate_no_wq_wiki12 already done — starting stage 257')
        v63 = v80(force=True)
        v57(f'stage257 exit={v63}')
        return v63
    return v44 if v44 else v42
if v46 == '__main__':
    raise v68(v87())