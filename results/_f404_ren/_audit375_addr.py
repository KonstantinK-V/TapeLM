"""THE ATOM OF THE ADDRESS. 373/374 tested the value's atom; the frame's was never tested.

374 on german said both things at once: the frame's function words carry the ending
(pred_func .52 vs null .27) and the material for production is absent (oracle .02). So the
form signal is real and its place is not the VALUE but the ADDRESS. Today an address is a
tuple of EXACT tokens - `singleton_bucket 1.0`, every frame unique - which is why places are
thin, `own` is thin, the offer is thin, and an unreachable population exists at all.

ONE LEVER: generalise the alphabet the frame is written in, leave the value whole.

    kappa(t) = t        if t is one of the corpus's F most frequent tokens (counted, no list)
             = e(t)     otherwise - its ending class, 374's counted inventory

    addr'(i) = (w, kappa(left), kappa(right))

`the walked dog of` and `the crossed cat of` become one address. Frames recur more, places
pool, own/offer thicken - the unreachable set is attacked at the WRITE PATH, where 356 said
coverage lives, instead of teaching production inside it.

  GATES, before the run, exact tape as baseline on the same window:
    G1  unreachable/seen drops by >= 20% relative
    G2  top-1 hit on the offer does not fall by more than 0.02 absolute
  Swept, not tuned: --func 16/32/64.

    python _audit375_addr.py
    python _audit375_addr.py --func 64 --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from _audit374_shape import ending_inventory, make_split
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage375_addr.json')

def build_tape(v4, v5, v6, v7):
    """374's frame cutter with the frame spelled in kappa; the hole's value stays the token."""
    v8 = [v32 for v78 in v4 for v32 in v78.v29()]
    v9 = [v7(v32) for v32 in v8]
    v10 = v33(v8)
    v34, v35 = ({}, v41(v36(1, v10 - 1)))
    for v11 in v36(1, v5 + 1):
        v37 = v40(v41)
        for v38 in v35:
            if v38 - v11 < 0 or v38 + 1 + v11 > v10:
                continue
            v37[v112(v9[v38 - v11:v38]), v112(v9[v38 + 1:v38 + 1 + v11])].v79(v38)
        v39 = []
        for v42, v47 in v37.v43():
            if v33(v47) >= 2:
                for v38 in v47:
                    v34[v38] = (v11,) + v42
                v39.v107(v47)
        v35 = v39
        if not v35:
            break
    v12 = v40(v41)
    for v38, v42 in v34.v43():
        v12[v42].v79(v38)
    v13 = [v80(v47) for v42, v47 in v12.v43() if v33({v8[v38] for v38 in v47}) >= v6]
    return (v13, v8, v10)

def exam(v14, v8, v15, v16, v17):
    v44, v45 = ({}, v40(v41))
    for v46, v47 in v48(v14):
        for v21 in v47:
            v44[v21] = v46
            v45[v8[v21]].v79(v21)
    v18 = {}
    v19 = [v21 for v47 in v14 for v21 in v47]
    v17.v49(v19)
    v20 = v50()
    for v21 in v19:
        if v20['seen'] >= v16:
            break
        v46 = v44[v21]
        v51 = v8[v21]
        v52 = {v8[v99] for v99 in v14[v46] if v99 != v21}
        if not v52:
            continue
        v20['seen'] += 1
        v53 = v50((v8[v99] for v99 in v14[v46]))
        v53[v51] -= 1
        if v53[v51] <= 0:
            del v53[v51]
        v54 = v41(v53)[:6]
        v55 = []
        if v54:
            v81 = v50((v8[v99] for v99 in v14[v46]))
            v82 = v50()
            for v83 in v54:
                v100 = v18.v108(v83)
                if v100 is None:
                    v100 = v50()
                    for v109 in v45[v83]:
                        for v111 in v14[v44[v109]]:
                            if v8[v111] != v83:
                                v100[v8[v111]] += 1
                    v18[v83] = v100
                for v11, v42 in v100.v43():
                    v42 -= v81.v108(v11, 0)
                    if v42 > 0 and v11 != v83:
                        v82[v11] += v42
            v55 = v82.v101(v15)
        v56 = {v11 for v11, v104 in v55}
        if v51 in v52 or v51 in v56:
            v20['reach'] += 1
            if v51 in v56:
                v20['in_off'] += 1
                v20['hit1'] += v55[0][0] == v51
        else:
            v20['unreach'] += 1
    return v20

def main() -> v2:
    v22 = v84.v57()
    v22.v58('--bytes', type=v2, default=30000000)
    v22.v58('--frame-max', type=v2, default=3)
    v22.v58('--min-fillers', type=v2, default=1)
    v22.v58('--lines', type=v2, default=25000)
    v22.v58('--window-lines', type=v2, default=400)
    v22.v58('--topm', type=v2, default=8)
    v22.v58('--endings', type=v2, default=64)
    v22.v58('--suffix-max', type=v2, default=4)
    v22.v58('--func', type=v2, default=32)
    v22.v58('--max-questions', type=v2, default=4000)
    v22.v58('--seed', type=v2, default=1337)
    v22.v58('--corpus', default=v102(v0))
    v23 = v22.v59()
    v24 = v3(v23.v67).v103('r', encoding='utf-8', errors='ignore').v60(v23.v61)
    v25 = [v86.v85() for v86 in v24.v29('\n') if v33(v86.v85()) >= 80]
    v4 = v25[:v2(0.7 * v33(v25))][:v23.v4]
    v17 = v87.v62(v23.v63)
    if v23.v64 and v23.v64 < v33(v4):
        v65 = v17.v88(v33(v4) - v23.v64)
        v4 = v4[v65:v65 + v23.v64]
    v26 = [v32 for v86 in v4 for v32 in v86.v29()]
    v27 = v50(v26)
    v28 = {v11 for v11, v104 in v27.v101(v23.v28)}
    v29 = v66(v89(v80(v110(v26)), v23.v90, v23.v68))

    def kappa(v32):
        return v32 if v32 in v28 else '~' + (v29(v32)[1] or '0')
    v30 = {'corpus': v23.v67, 'lines': v33(v4), 'func': v23.v28, 'endings': v23.v68, 'topm': v23.v15}
    for v69, v42 in (('exact', lambda v32: v32), ('shaped', v7)):
        v13, v8, v10 = v91(v4, v23.v5, v23.v6, v42)
        v70 = v105((v33(v47) for v47 in v13)) / v94(1, v10)
        v20 = v92(v13, v8, v23.v15, v23.v93, v87.v62(v23.v63))
        v71 = v94(1, v20['seen'])
        v30[v69] = {'places': v33(v13), 'coverage': v70, 'fillers': v105((v33(v47) for v47 in v13)) / v94(1, v33(v13)), 'seen': v20['seen'], 'unreach': v20['unreach'] / v71, 'reach': v20['reach'] / v71, 'hit1': v20['hit1'] / v94(1, v20['in_off'])}
        v72 = v30[v69]
        v76(f"{v69:7s} places {v72['places']:6d}  cov {v70:.4f}  fill/place {v72['fillers']:.2f}  seen {v20['seen']}  unreach {v72['unreach']:.4f}  hit1 {v72['hit1']:.4f}")
    v73, v21 = (v30['exact'], v30['shaped'])
    v30['unreach_rel'] = (v73['unreach'] - v21['unreach']) / v94(1e-09, v73['unreach'])
    v30['hit_delta'] = v21['hit1'] - v73['hit1']
    v30['cov_delta'] = v21['coverage'] - v73['coverage']
    v1.v95.v74(parents=True, exist_ok=True)
    v1.v75(v106.v96(v30, indent=1), encoding='utf-8')
    v76(f"\nGATES  unreach -{v30['unreach_rel'] * 100:.1f}% rel (need >=20)   hit1 {v30['hit_delta']:+.4f} (need > -0.02)   cov {v30['cov_delta']:+.4f}")
    if v97(v73['seen'], v21['seen']) < 400:
        v76('VOID - too few questions; widen --window-lines.')
    elif v30['unreach_rel'] >= 0.2 and v30['hit_delta'] > -0.02:
        v76('THE ADDRESS ATOM WAS THE WALL. Frames spelled in form-classes pool the places, the material arrives at the write path, and nothing was produced by hand.')
    else:
        v76('THE ADDRESS ATOM WAS NOT THE WALL at this width - pooling frames does not deliver the missing material, or it costs the hit. Read cov and fill/place before concluding: if both rose and unreach did not fall, the missing truths live outside ANY recurring frame, and the wall is the corpus, not the alphabet.')
    v76(f'\nwritten to {v1}')
    return 0
if v31 == '__main__':
    raise v77(v98())