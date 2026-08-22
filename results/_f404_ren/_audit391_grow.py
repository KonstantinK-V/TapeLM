"""C: SAME QUESTIONS, TAPE 1x/2x/4x/8x. Requirement 4, measured as a count.

369 scaled `--addresses` (more questions at an unchanged walk). That is not this. Section 29.2
asked for the same Phi, the same question set, the tape cut to 1x/2x/4x/8x, accuracy monotone.

No frozen Phi is on disk (365r3 left metrics, not weights). This file therefore freezes the
STANDING OFFER - walk interleaved with connect, cap 8, qprof, same-line drop - which is what Phi
ranks. If the eight do not improve as the tape grows, Phi has nothing to get smarter with, and
requirement 4 is closed by that measurement rather than by substituting 369.

  ONE RUN. Seed 1337. Nested prefixes of the same wiki window: 400 / 800 / 1600 / 3200 lines.
  Questions are drawn on the 1x prefix and replayed by (line, word-in-line) on every larger tape.

  VOID. If a 1x question's hidden token is not at the same slot on a larger tape, the mapping
  is broken and the run is not C.
  GATE. std8 is non-decreasing on 1x -> 2x -> 4x -> 8x. That monotonicity IS the gate.
        A new metric does not decide it.

    python _audit391_grow.py
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import _audit390_address as A
v0 = v4('data/_wikitext103_train.txt')
v1 = v4('results/_stage391_grow.json')
v2 = (1, 2, 4, 8)

def line_starts(v5):
    v6 = [0]
    for v7 in v5:
        v6.v58(v6[-1] + v70(v7.v79()))
    return v6

def slot_at(v5, v8, v9):
    return v34(v5)[v8] + v9

def collect_1x(v10, v5, v11, v12):
    """Questions from the 1x tape, identified independently of place ids."""
    v13 = v34(v5)
    v14 = []
    v15 = [v16 for v59 in v10['places'] for v16 in v59]
    v12.v35(v15)
    for v16 in v15:
        if v70(v14) >= v11.v60:
            break
        v36 = v71.v61(v10, v16, v11, v12)
        if v36 is None:
            continue
        v8 = v10['owner'][v16]
        v9 = v16 - v13[v8]
        v14.v58((v8, v9, v10['toks'][v16]))
    return v14

def replay(v10, v5, v14, v11, v12):
    """The same questions on a (possibly thicker) tape. Broken maps return None."""
    v18, v17 = (v62(), 0)
    for v8, v9, v37 in v14:
        if v8 >= v70(v5):
            v17 += 1
            continue
        v16 = v63(v5, v8, v9)
        if v16 >= v70(v10['toks']) or v10['toks'][v16] != v37:
            v17 += 1
            continue
        v36 = v71.v61(v10, v16, v11, v12)
        if v36 is None:
            continue
        for v28, v64 in v36.v65():
            if not v28.v83('_'):
                v18[v28] += v64
    v18['broken'] = v17
    return v18

def main() -> v3:
    v19 = v66.v38()
    v19.v39('--bytes', type=v3, default=30000000)
    v19.v39('--frame-max', type=v3, default=3)
    v19.v39('--min-fillers', type=v3, default=1)
    v19.v39('--base-lines', type=v3, default=400)
    v19.v39('--places', type=v3, default=8)
    v19.v39('--topm', type=v3, default=8)
    v19.v39('--max-questions', type=v3, default=3000)
    v19.v39('--seed', type=v3, default=1337)
    v19.v39('--corpus', default=v77(v0))
    v19.v39('--out', default=v77(v1))
    v11 = v19.v40()
    v20 = v4(v11.v85).v78('r', encoding='utf-8', errors='ignore').v41(v11.v42)
    v21 = [v68.v67() for v68 in v20.v79('\n') if v70(v68.v67()) >= 80]
    v21 = v21[:v3(0.7 * v70(v21))]
    v12 = v69.v43(v11.v44)
    v22 = v11.v45 * v2[-1]
    if v70(v21) < v22:
        v55('corpus shorter than 8x')
        return 1
    v23 = v12.v46(v70(v21) - v22)
    v24 = v21[v23:v23 + v22]
    v25 = v24[:v11.v45]
    v26 = v71.v47(v25, v11.v48, v11.v49)
    v14 = v50(v26, v25, v11, v69.v43(v11.v44))
    if not v14:
        v55('no 1x questions')
        return 1
    v27 = []
    for v28 in v2:
        v5 = v24[:v11.v45 * v28]
        v10 = v71.v47(v5, v11.v48, v11.v49)
        v18 = v72(v10, v5, v14, v11, v69.v43(v11.v44))
        v51 = v73(1, v18['n'])
        v52 = {'scale': v28, 'lines': v70(v5), 'places': v70(v10['places']), 'n': v18['n'], 'broken': v18['broken'], 'std8': v18['std8'] / v51, 'half8': v18['half8'] / v51, 'half_any': v18['half_any'] / v51}
        v27.v58(v52)
        v55(f"{v28}x  lines {v52['lines']}  places {v52['places']}  n {v52['n']}  broken {v52['broken']}  std8 {v52['std8']:.4f}  half8 {v52['half8']:.4f}  half_any {v52['half_any']:.4f}")
    v29 = [v74['std8'] for v74 in v27]
    v17 = v53((v74['broken'] for v74 in v27))
    v30 = v54((v29[v80] <= v29[v80 + 1] + 1e-12 for v80 in v84(v70(v29) - 1)))
    v31 = v17 > 0
    v32 = {'seed': v11.v44, 'n_1x': v70(v14), 'rows': v27, 'std8': v29, 'monotone': v30, 'void': v31, 'broken': v17}
    v55('VOID mapping broken' if v31 else 'GATE PASS  std8 monotone' if v30 else 'GATE FAIL  std8 not monotone — requirement 4 closed on the offer')
    v4(v11.v81).v56(v82.v75(v32, indent=2), encoding='utf-8')
    v55(f'wrote {v11.v81}')
    return 0
if v33 == '__main__':
    raise v57(v76())