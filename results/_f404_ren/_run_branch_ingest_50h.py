"""
Autonomous ~45h pipeline for the TapeLM ingest-forks branch (247+).

Runs without an interactive agent: executes jobs, reads verdicts, chooses next
steps, writes journals under results/branch_ingest_50h/.

  python _run_branch_ingest_50h.py [--hours 45] [--resume]

Stop: write results/branch_ingest_50h/STOP (or Ctrl+C — state is saved).
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
v0 = v96(v97).v47().v1
v2 = v0 / 'results' / 'branch_ingest_50h'
v3 = v2 / 'state.json'
v4 = v2 / 'journal.jsonl'
v5 = v2 / 'master.log'
v6 = v2 / 'PLAN.md'
v7 = v2 / 'FINAL_REPORT.md'
v8 = v2 / 'STOP'
v9 = v16.v10

def utc() -> v11:
    return v106.v98(v107.v54).v48()

def log(v17: v11) -> None:
    v18 = v17.v49('≈', '~').v49('—', '-').v49('–', '-').v49('→', '->').v49('…', '...')
    v19 = f'[{v54()}] {v18}'
    try:
        v79(v19, flush=True)
    except v50:
        v79(v19.v115('ascii', 'replace').v108('ascii'), flush=True)
    v2.v51(parents=True, exist_ok=True)
    with v5.v80('a', encoding='utf-8') as v52:
        v52.v81(f'[{v54()}] {v17}\n')

def journal(v20: v12) -> None:
    v20 = {'ts': v54(), **v20}
    with v4.v80('a', encoding='utf-8') as v52:
        v52.v81(v99.v83(v20, ensure_ascii=False) + '\n')

def load_state() -> v12:
    if v3.v53():
        return v99.v82(v3.v100(encoding='utf-8'))
    return {'started': v54(), 'hours_budget': 45.0, 'completed': [], 'failed': [], 'skipped': [], 'decisions': [], 'next_job': None, 't0': v60.v60()}

def save_state(v21: v12) -> None:
    v21['updated'] = v54()
    v3.v55(v99.v83(v21, indent=2), encoding='utf-8')

def elapsed_h(v21: v12) -> v13:
    return (v60.v60() - v21['t0']) / 3600.0

def remaining_h(v21: v12) -> v13:
    return v21['hours_budget'] - v84(v21)

def read_overall(v22: v15 | v11) -> v11 | None:
    v23 = v0 / 'results' / f'stage{v22}_decision.json'
    if not v23.v53():
        return None
    try:
        return v99.v82(v23.v100(encoding='utf-8')).v68('overall')
    except v56:
        return None

def read_decision(v22: v15 | v11) -> v12 | None:
    v23 = v0 / 'results' / f'stage{v22}_decision.json'
    if not v23.v53():
        return None
    try:
        return v99.v82(v23.v100(encoding='utf-8'))
    except v56:
        return None

def run_job(v21: v12, v24: v11, v25: v85[v11], v26: v13, v27: v11) -> v14:
    if v8.v53():
        v57('STOP file present — abort before job ' + v24)
        return False
    if v70(v21) < 0.05:
        v57('budget exhausted — skip ' + v24)
        v21['skipped'].v86(v24)
        v59(v21)
        return False
    if v24 in v21['completed']:
        v57(f'already done {v24} — skip')
        return True
    v57(f'START {v24} est~{v26:.1f}h remain~{v70(v21):.1f}h | {v27}')
    v58({'event': 'start', 'job': v24, 'argv': v25, 'est_h': v26, 'why': v27})
    v21['next_job'] = v24
    v59(v21)
    v28 = v60.v60()
    try:
        v61 = v101.v87([v9, *v25], cwd=v11(v0), check=False)
        v62 = (v60.v60() - v28) / 3600.0
        v63 = v61.v88 == 0
        v64 = {'event': 'end', 'job': v24, 'ok': v63, 'returncode': v61.v88, 'wall_h': v62}
        for v65 in v25:
            if v65.v109('_stage') and v65.v110('.py'):
                v102 = v65.v49('_stage', '').v112('_')[0]
                if v102.v111():
                    v64['overall'] = v66(v15(v102))
                    v64['decision'] = v89(v15(v102))
                break
        v58(v64)
        if v63:
            v21['completed'].v86(v24)
            v57(f"OK {v24} wall={v62:.2f}h overall={v64.v68('overall')}")
        else:
            v21['failed'].v86({'job': v24, 'code': v61.v88, 'wall_h': v62})
            v57(f'FAIL {v24} code={v61.v88} wall={v62:.2f}h')
        v21['next_job'] = None
        v59(v21)
        return v63
    except v56 as e:
        v58({'event': 'exception', 'job': v24, 'error': v11(v113), 'tb': v116.v114()})
        v21['failed'].v86({'job': v24, 'error': v11(v113)})
        v59(v21)
        v57(f'EXCEPTION {v24}: {v113}')
        return False

def write_plan(v29: v13) -> None:
    v6.v55(f'# Branch ingest-forks ~{v29:.0f}h autonomous plan\n\nStarted from 247 `INGEST_FORK_SLOTS_AND_HOP`: bindings→tape, CE without bindings, hop as admission.\n\n## Job ladder (adaptive)\n\n1. **Confirm 247** — second seed / re-run if needed (~0.2h)\n2. **249 hop stream** — precision of hop admission (~0.2–1h)\n3. **248 masked understand** — 4k then 12k then 20k steps if gates hold (~3–12h)\n4. **246 curriculum** — 10k/domain; if wiki mem holds + GPT PPL rises → **30k/domain** (~8h + ~20h)\n5. **248 mega** — remaining budget into masked CE night (up to 25k)\n6. **Multi-seed 247** — seeds 2471/2472 if time (~0.5h)\n7. **FINAL_REPORT.md** — synthesis for the branch\n\n## Stop\n\n`echo stop > results/branch_ingest_50h/STOP`\n\n## Logs\n\n- `master.log` — human timeline\n- `journal.jsonl` — machine events + verdicts\n- `state.json` — resume pointer\n', encoding='utf-8')

def decide_and_run(v21: v12) -> None:
    """Main adaptive loop — decisions encoded here."""
    v29 = v21['hours_budget']
    v30 = v66(247)
    if v30 is None or 'SLOTS' not in (v30 or ''):
        v67(v21, '247_rerun', ['_stage247_ingest_forks.py'], 0.25, '247 missing/weak — re-run full')
        v30 = v66(247)
    else:
        v57(f'247 already {v30} — keep')
        v58({'event': 'decision', 'decision': 'keep_247', 'overall': v30})
    if v8.v53() or v70(v21) < 0.1:
        return
    v67(v21, '249_hop', ['_stage249_hop_stream.py'], 0.3, 'map hop admission precision')
    v31 = v66(249)
    v58({'event': 'decision', 'after': '249', 'overall': v31, 'next': '248_masked_4k'})
    if v8.v53() or v70(v21) < 0.2:
        return
    v67(v21, '248_4k', ['_stage248_masked_understand.py', '--steps', '4000'], 2.5, '247 fork: grow understanding without binding CE')
    v32 = v89(248) or {}
    v33 = v32.v68('overall')
    v34 = (v32.v68('masked') or {}).v68('mem_after_code_W', 0)
    v35 = (v32.v68('masked') or {}).v68('next_tok', 0)
    if v33 and 'NO' not in v33 and (v70(v21) > 4):
        v69 = 12000
        v67(v21, '248_12k', ['_stage248_masked_understand.py', '--steps', v11(v69)], 6.0, f'248 {v33} mem_cf={v34} nt={v35} — scale masked CE')
        v32 = v89(248) or v32
        v33 = v32.v68('overall')
    else:
        v58({'event': 'decision', 'skip': '248_12k', 'reason': v33})
    if v70(v21) > 8 and v33 and ('OK' in (v33 or '')):
        v67(v21, '248_20k', ['_stage248_masked_understand.py', '--steps', '20000'], 10.0, 'masked CE night — deepen understanding arm')
    if v8.v53() or v70(v21) < 1:
        return
    v67(v21, '246_10k', ['_stage246_domain_curriculum.py', '--steps', '10000'], 8.0, 'multi-domain retention matrix at 10k/domain')
    v36 = v89(246) or {}
    v37 = v36.v68('drops_first_to_final') or {}
    v38 = v37.v68('wiki') or {}
    v39 = (v36.v68('matrix') or [{}])[-1].v68('domains', {}).v68('wiki', {}).v68('tape_mem', 0) >= 0.7
    v40 = v38.v68('gpt_ppl_rise', 0) >= 1.0
    v58({'event': 'decision', 'after': '246_10k', 'overall': v36.v68('overall'), 'tape_wiki_ok': v39, 'gpt_ppl_rise': v38.v68('gpt_ppl_rise'), 'escalate_30k': v14(v39 and v40 and (v70(v21) > 12))})
    if v39 and v40 and (v70(v21) > 12):
        v67(v21, '246_30k', ['_stage246_domain_curriculum.py', '--steps', '30000'], 22.0, 'escalate: wiki mem holds & GPT PPL rises at 10k → paper-scale 30k')
    elif v70(v21) > 12:
        v67(v21, '248_fill', ['_stage248_masked_understand.py', '--steps', '15000'], v103(12.0, v70(v21) - 1), '246 did not clear escalate gates — burn budget on masked understand')
    if v8.v53() or v70(v21) < 0.5:
        return
    v67(v21, '249_scaled', ['_stage249_hop_stream.py', '--steps', '8000'], 0.5, 'larger hop/uniform admission stress')
    if v70(v21) > 1.0:
        v67(v21, '247_confirm', ['_stage247_ingest_forks.py'], 0.3, 'confirm fork map still holds')
    v41 = v70(v21)
    if v41 > 2.0:
        v71 = v15(v103(25000, v41 * 3600 / 2.5))
        if v71 >= 3000:
            v67(v21, f'248_leftover_{v71}', ['_stage248_masked_understand.py', '--steps', v11(v71)], v41 - 0.3, f'burn remaining ~{v41:.1f}h into masked CE ({v71} steps)')

def write_report(v21: v12) -> None:
    v42 = [f'# Branch ingest-forks final report', f'', f"- started: {v21.v68('started')}", f'- finished: {v54()}', f"- wall_h: {v84(v21):.2f} / budget {v21['hours_budget']}", f"- completed: {v21.v68('completed')}", f"- failed: {v21.v68('failed')}", f"- skipped: {v21.v68('skipped')}", f'', f'## Verdicts']
    for v43 in (247, 248, 249, 246):
        v72 = v89(v43)
        if v72:
            v42.v86(f"- **{v43}**: `{v72.v68('overall')}`")
        else:
            v42.v86(f'- **{v43}**: (missing)')
    v42 += ['', '## Branch recommendation', '1. Keep unknown bindings out of CE (247/248).', '2. Use hop-similarity as slot admission (247/249), not as CE curriculum.', '3. Domain lenses (246) for eras; backbone understanding via masked CE.', '4. Escalate curriculum steps only when wiki mem holds and GPT PPL rises.', '', 'See journal.jsonl for decision trail.']
    v7.v55('\n'.v104(v42) + '\n', encoding='utf-8')
    v57('wrote ' + v11(v7))

def main() -> v15:
    v44 = v90.v73()
    v44.v74('--hours', type=v13, default=45.0)
    v44.v74('--resume', action='store_true')
    v45 = v44.v75()
    v2.v51(parents=True, exist_ok=True)
    if v45.v76 and v3.v53():
        v21 = v91()
        v21['hours_budget'] = v45.v29
        v57(f"RESUME elapsed={v84(v21):.2f}h completed={v21.v68('completed')}")
    else:
        if v8.v53():
            v8.v105()
        v21 = v91()
        v21['hours_budget'] = v45.v29
        v21['t0'] = v60.v60()
        v21['started'] = v54()
        v21['completed'] = []
        v21['failed'] = []
        v21['skipped'] = []
        v5.v55('', encoding='utf-8')
        v4.v55('', encoding='utf-8')
        v92(v45.v29)
        v59(v21)
        v57(f'START pipeline hours={v45.v29}')
    v58({'event': 'pipeline_start', 'hours': v45.v29, 'resume': v45.v76})
    try:
        v93(v21)
    except v77:
        v57('KeyboardInterrupt — saving state')
        v58({'event': 'interrupt'})
    except v56 as e:
        v57(f'pipeline exception: {v113}')
        v58({'event': 'pipeline_exception', 'error': v11(v113), 'tb': v116.v114()})
    finally:
        v94(v21)
        v59(v21)
        v57(f'PIPELINE END wall={v84(v21):.2f}h')
    return 0
if v46 == '__main__':
    raise v78(v95())