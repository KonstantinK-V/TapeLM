"""~7 hours on THE MISSING CONSTRUCTION: the loop. Self-gating, cheapest decision first.

THE DIAGNOSIS THIS QUEUE IS BUILT ON. The pipeline is assembled and it is assembled FOR ONE
SHOT. Question k+1 knows nothing of question k; the answer is scored and discarded; the reward
is terminal; we choose the hole at random. Every closed result in this project - composition,
generation, revision - was closed AS A SINGLE-SHOT OPERATION, and all three are things a mind
does ACROSS steps.

OLD CONSTRAINTS THAT MAY NO LONGER APPLY, and this queue re-opens exactly two of them:

  324 CLOSED MEMORY. It measured a PERFECT write-back's marginal retrieval gain on INDEPENDENT
      questions and found ~0. It never measured a DEPENDENT CHAIN. Mis-scoped for the question
      we now care about, and the closure does not transfer.

  322 CLOSED DEPTH. It measured reachable 0.54 at depth two against 0.12 at depth one - THE
      LARGEST MOVEMENT OF REACH THIS PROJECT HAS EVER RECORDED - and was closed because CONFIRM
      collapsed and an honest rival read 2-3%. Both are single-shot artefacts: CONFIRM broke
      because ONE decision had to serve "answer at home" and "chase depth" at once, and a
      one-hop rival cannot follow a two-hop path by construction. Neither says the chain does
      not reach.

  342b was demoted as "about the architecture we are leaving". WRONG - we are not leaving it,
      we are extending it in TIME. If a chain needs a second decision type (when to stop), the
      4x price of a second objective is load-bearing again, and 342a showed capacity is not
      binding for ONE objective at d=64, which is not the same question.

WHAT IS NOT RE-OPENED: no heuristics, everything a count, rivals closed and declared, nulls and
transplant riding along. Those are why the results mean anything.

    TIER 1  ~1h, torch-free   the ceiling of a two-step chain (_audit351_chain)
    GATE    tiers 2-3 run only if a chain reaches materially more than one step
    TIER 2  ~3h              depth re-run at 4 seeds, read with the 337/339 machinery that did
                             not exist when 322 closed it
    TIER 3  ~2h              342b: re-price the second objective at d=64

    python _queue352.py
    python _queue352.py --go
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
v0 = '_stage289_derivation.py'
v1 = (v6('results'), v6('out'))
v2 = v6('results/_arm_ctrl_next.txt')
v3 = '--reach --tape frames --frame-max 3 --min-mentions 1 --fp hash --write-fp hash --ink mean --write-ink mean --words ascii --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region --objective reward --addresses 1500 --reach-max-q 2000 --import-k 1 --gamma 1.0 --probe-period 1000 --probe-size 60 --cpu'
v4 = (1337, 2890, 4711, 8642)

def done(v7):
    return v24(((v37 / f'{v50}stage289_decision_{v7}.json').v46() for v37 in v1 for v50 in ('', '_')))

def run(v8, v9, v10):
    v25(f'  [{v10}] ' + ' '.v43(v8[1:]), flush=True)
    if not v9:
        return True
    v11 = v26.v26()
    v12 = v40.v27(v8)
    v25(f'    -> exit {v12.v28} in {v26.v26() - v11:.0f}s', flush=True)
    return v12.v28 == 0

def swap(v13, v14, v15):
    v17, v29, v30 = ([], 0, False)
    v16 = v31(v13)
    while v29 < v41(v16):
        if v16[v29] == v14:
            v17 += [v14, v52(v15)]
            v29 += 2
            v30 = True
        else:
            v17.v51(v16[v29])
            v29 += 1
    if not v30:
        v17 += [v14, v52(v15)]
    return v17

def main() -> v5:
    v18 = v42.v32()
    v18.v33('--base', default=None)
    v18.v33('--wiki', default='data/_wikitext103_train.txt')
    v18.v33('--steps', type=v5, default=4000)
    v18.v33('--go', action='store_true')
    v18.v33('--skip-tier1', action='store_true')
    v19 = v18.v34()
    v13 = (v19.v13 or (v2.v54(encoding='utf-8').v60() if v2.v46() else v3)).v35()
    v25('arm:', ' '.v43(v13))
    v25('PLAN ONLY - add --go\n' if not v19.v9 else 'RUNNING\n')
    v20 = v26.v26()
    if not v19.v36:
        v25('TIER 1  the ceiling of a chain (torch-free, ~1h)')
        for v44, v45 in ((8, 8), (8, 32)):
            v27([v61.v57, '_audit351_chain.py', '--topm', v52(v44), '--branch', v52(v45)], v19.v9, f'351 top{v44} branch{v45}')
    v21 = v6('results/_stage351_chain.json')
    if v19.v9 and v21.v46():
        v37 = v53.v47(v21.v54(encoding='utf-8'))
        v38 = v37.v55('oracle_or_1', 0) > v37.v55('reach1', 0) + 0.05
        v25(f"\nGATE  one step {v37.v55('reach1', 0):.4f} -> a perfect chooser over {v37.v55('paths', 0):.0f} paths {v37.v55('oracle_or_1', 0):.4f}  ->  {('the chain reaches, tiers 2-3 run' if v38 else 'THE CHAIN DOES NOT REACH')}")
        if not v38:
            v25("  Tiers 2 and 3 SKIPPED. A second hop over the same relation reaches no more than the first, so depth has nothing to route to and the second objective has nothing to price. Substitution is closed one step or two, and the project's result is the separation proof.")
            v25(f'\n{(v26.v26() - v20) / 60:.0f} min')
            return 0
    v25('\nTIER 2  depth 2, 4 seeds, read with machinery 322 did not have (~3h)')
    for v22 in v4:
        v7 = f'352deep_s{v22}'
        if v48(v7):
            v25(f'  [{v7}] already present')
            continue
        v8 = [v61.v57, v0] + v58(v13, '--reach-depth', 2) + ['--wiki', v19.v56, '--seed', v52(v22), '--train-steps', v52(v19.v59), '--run-tag', v7]
        v27(v8, v19.v9, v7)
    v25('\nTIER 3  342b: is the 4x price capacity or law, at d=64 (~2h)')
    for v22 in v4:
        v7 = f'352price_s{v22}'
        if v48(v7):
            v25(f'  [{v7}] already present')
            continue
        v8 = [v61.v57, v0] + v58(v13, '--dim', 64) + ['--speak-batch', '8', '--speak-weight', '1.0', '--wiki', v19.v56, '--seed', v52(v22), '--train-steps', v52(v62(1, v19.v59 // 8)), '--run-tag', v7]
        v27(v8, v19.v9, v7)
    v25(f'\n{(v26.v26() - v20) / 60:.0f} min')
    v25('read with:')
    v25('  python _read299.py <352deep reports> --held     # DEPTH, GATE-WO, MARGIN, RANK')
    v25('  python _read299.py <352price reports> --held    # route first: the 4x is the veto')
    v25('  compare 352price against the d=64 ctrl already in hand (342news_d64)')
    return 0
if v23 == '__main__':
    raise v39(v49())