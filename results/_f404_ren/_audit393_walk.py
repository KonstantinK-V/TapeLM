"""393: A WALKER, NOT A SMARTER MIND. One step to a bridge, then read from there.

392 closed bind: addr-N(A)∩addr-N(B) pays, and a B from another window pays the same.
391 closed requirement 4 on the eight. 388 measured hop-2 dump: the truth is often two hops
away (hop2-only ~0.18) and hop2@8 does not beat hop1@8. Dumping hop-2 into eight is 347.

This file asks the remaining question as a count: if the mind STEPS to one hop-1 place and
reads THAT place's walk@8, does the truth land more than staying put? The output of the step
is a place. The eight after the step are content of the arrived place's neighbourhood, not a
merged hop-2 dump.

  hop1          standing offer (step interleaved with connect, cap 8)     TODAY
  committed     walk@8 rooted at the single nearest hop-1 place           A WALKER
  oracle        truth in walk@8 of SOME hop-1 bridge (cap --bridges)      A PERFECT STEP
  dump2         388's hop-2 VALUE dump, cap 8                             347, the rival
  rand          walk@8 rooted at a random place                           the floor

qprof on the question. Bridges are walked from their full profile (they are handles).
Same-line of the question dropped. Section 27: the question place's prof[pid] is never a key.

VOID, read first. oracle - hop1 < 0.02: nowhere to walk, close without a chooser.
GATE, declared before the run, four seeds.
  1. oracle - hop1 > 0.05 on 3/4.
  2. dump2 does not beat oracle (a walker is not a wider offer).
  3. rand below half of oracle.
committed is reported, not a gate: if the top bridge loses, the step is real and the ranking
of bridges is the next lever (386's shape). Nothing here claims the mind got smarter.

    python _check393_walk.py
    python _audit393_walk.py --seed 1337
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import _audit390_address as A
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage393_walk.json')

def dump_hop2(v4, v5, v6, v7, v8, v9):
    """388's hop-2 dump: values at places two hops out, scored by min(overlap) summed over paths."""
    v10 = v50(v8) | {v5}
    v11 = v43()
    v44, v45 = v78.v46(v4, v5, v6, v9)
    for v12 in v8:
        v47 = v44[v12]
        v79, v45 = v78.v46(v4, v12, v4['prof'][v12], v9 | v10)
        for v57, v80 in v79.v81():
            v82 = v102(v47, v80)
            for v83 in v4['prof'][v57]:
                if v83 not in v7:
                    v11[v83] += v82
    return [v48 for v48, v103 in v11.v104()]

def from_bridge(v4, v12, v7, v9, v13, v14):
    """Arrive at b, then the standing walk from b, cap 8. b's full profile: it is a handle."""
    v15 = v78.v49(v4, v12, v4['prof'][v12], v13, v9)
    return v78.v84(v4, v15, v7)[:v14]

def measure(v4, v16, v17, v18):
    v19 = v4['toks']
    v5 = v4['place_of'][v16]
    v20 = v19[v16]
    v7 = {v19[v85] for v85 in v4['places'][v5] if v85 != v16}
    if not v7 or v20 in v7:
        return None
    v6 = v43((v19[v85] for v85 in v4['places'][v5] if v85 != v16))
    v9 = v50(v4['on_line'][v4['owner'][v16]])
    v9.v51(v5)
    v21 = v78.v52(v4, v5, v7, v6, v17.v53, v9)
    v22 = v78.v54(v4, v5, v7, v6, v9, norm_by_places=False)
    v23 = v78.v55(v21, v22, cap=v17.v86)
    v8 = v78.v49(v4, v5, v6, v17.v8, v9)
    v24 = v87(v4, v5, v6, v7, v8, v9)[:v17.v86]
    v25 = [v88(v4, v12, v7, v9 | {v5}, v17.v53, v17.v86) for v12 in v8]
    v26 = v25[0] if v25 else []
    v27 = v56((v20 in v105 for v105 in v25))
    v28 = v20 in v24
    v29 = v20 in v23
    v30 = v20 in v26
    v31 = [v57 for v57 in v106(v89(v4['places'])) if v57 != v5 and v57 not in v9]
    v32 = v88(v4, v31[v18.v97(v89(v31))], v7, v9 | {v5}, v17.v53, v17.v86) if v31 else []
    return {'n': 1, 'hop1': v2(v29), 'committed': v2(v30), 'oracle': v2(v27), 'dump2': v2(v28), 'rand': v2(v20 in v32), 'walk_only': v2(v27 and (not v29)), 'n_bridges': v89(v8), '_lanes': {'hop1': v23, 'committed': v26, 'dump2': v24}}

def run(v4, v17, v18):
    v33 = v43()
    v34 = [v16 for v90 in v4['places'] for v16 in v90]
    v18.v58(v34)
    for v16 in v34:
        if v33['n'] >= v17.v91:
            break
        v59 = v92(v4, v16, v17, v18)
        if v59 is None:
            continue
        for v13, v48 in v59.v81():
            if not v13.v112('_'):
                v33[v13] += v48
    return v33

def main() -> v2:
    v35 = v93.v60()
    v35.v61('--bytes', type=v2, default=30000000)
    v35.v61('--frame-max', type=v2, default=3)
    v35.v61('--min-fillers', type=v2, default=1)
    v35.v61('--lines', type=v2, default=25000)
    v35.v61('--window-lines', type=v2, default=400)
    v35.v61('--places', type=v2, default=8)
    v35.v61('--topm', type=v2, default=8)
    v35.v61('--bridges', type=v2, default=8)
    v35.v61('--max-questions', type=v2, default=3000)
    v35.v61('--seed', type=v2, default=1337)
    v35.v61('--corpus', default=v107(v0))
    v35.v61('--out', default=v107(v1))
    v17 = v35.v62()
    v36 = v3(v17.v113).v108('r', encoding='utf-8', errors='ignore').v63(v17.v64)
    v37 = [v95.v94() for v95 in v36.v109('\n') if v89(v95.v94()) >= 80]
    v38 = v37[:v2(0.7 * v89(v37))][:v17.v38]
    v18 = v96.v65(v17.v66)
    if v17.v67 and v17.v67 < v89(v38):
        v68 = v18.v97(v89(v38) - v17.v67)
        v38 = v38[v68:v68 + v17.v67]
    v4 = v78.v69(v38, v17.v70, v17.v71)
    v33 = v72(v4, v17, v18)
    v39 = v73(1, v33['n'])

    def f(v13):
        return v33[v13] / v39
    v40 = {'seed': v17.v66, 'places': v89(v4['places']), 'n': v33['n'], 'n_bridges': v33['n_bridges'] / v39, 'hop1': v98('hop1'), 'committed': v98('committed'), 'oracle': v98('oracle'), 'dump2': v98('dump2'), 'rand': v98('rand'), 'walk_only': v98('walk_only'), 'oracle_minus_hop1': v98('oracle') - v98('hop1'), 'dump_minus_oracle': v98('dump2') - v98('oracle')}
    v41 = v40['oracle_minus_hop1'] < 0.02
    v74(f"places {v40['places']}  questions {v40['n']}  bridges/q {v40['n_bridges']:.1f}")
    v74(f"VOID CHECK  oracle-hop1 {v40['oracle_minus_hop1']:+.4f}" + ('  NOWHERE TO WALK' if v41 else '  ok'))
    v74(f"HIT        hop1 {v40['hop1']:.4f}  committed {v40['committed']:.4f}  oracle {v40['oracle']:.4f}  dump2 {v40['dump2']:.4f}  rand {v40['rand']:.4f}  walk_only {v40['walk_only']:.4f}")
    v3(v17.v110).v99.v75(parents=True, exist_ok=True)
    v3(v17.v110).v76(v111.v100(v40, indent=2), encoding='utf-8')
    v74(f'wrote {v17.v110}')
    return 0
if v42 == '__main__':
    raise v77(v101())