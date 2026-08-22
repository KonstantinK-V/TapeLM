"""HOW MUCH OF THE TEXT DOES THE TAPE HAVE A HOLE FOR? The angle nobody has taken.

354 and 355 closed memory - content and reading both - and that closes the loop as a source of
capability. The rule says the VIEW MUST CHANGE, so here is the change, and it is not another
lever on the same picture.

WHAT WE HAVE MEASURED FOR 350 STEPS: accuracy, CONDITIONAL ON THE HOLE BEING ADDRESSABLE. Every
number in this project - reach, hit, pick, gate, the transplant, the invariant - is computed
over questions drawn from `keep`, and `keep` is the set of positions whose frame RECURS and
which have TWO OR MORE distinct fillers. Positions failing either test are dropped in the write
path and have never appeared in any measurement, as question or as answer.

NOBODY HAS EVER STATED WHAT FRACTION OF THE CORPUS THAT IS. If it is small, then every result
holds inside a sliver, and "knowledge separate, decision separate" is proven for the part of
language that happens to be substitutable - which may be exactly the part that carries no facts.

AND THERE IS A RULE IN THE WRITE PATH THAT PHI NEVER GOT A SAY IN. `frame_keep` reads each
position at its WIDEST RECURRING frame. Widest = fewest, most precise fillers. Narrowest =
most positions addressable, vaguest paradigm. That is a real trade with cardinality on both
sides, it was fixed by fiat at the very first commit, and it is the last decision in the
pipeline still made by the tape instead of by the mind. WIDTH IS COVERAGE. This audit measures
what that fiat costs.

WHAT IS COUNTED, no model, no torch:

    addressable    positions whose frame recurs at some width <= frame_max
    on_tape        addressable AND sitting in a place with >= min_fillers distinct fillers,
                   i.e. the positions that can actually be asked about        THE REAL DENOMINATOR
    by width       where the widest recurring frame lands. Width 1 is the cheap majority
    by rarity      P(on_tape | corpus frequency of the token). THE MONEY ROW: a fact is a rare
                   token. If rare tokens are never on the tape, the knowledge half is a lattice
                   of CATEGORIES by construction, and no amount of mind reaches a fact.

    python _audit356_coverage.py
    python _audit356_coverage.py --frame-max 1     # what the narrowest reading would address
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v4('data/_wikitext103_train.txt')
v1 = v4('results/_stage356_coverage.json')
v2 = ((1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 10 ** 9))

def main() -> v3:
    v5 = v67.v29()
    v5.v30('--bytes', type=v3, default=30000000)
    v5.v30('--frame-max', type=v3, default=3)
    v5.v30('--min-fillers', type=v3, default=2)
    v5.v30('--lines', type=v3, default=25000)
    v5.v30('--corpus', default=v79(v0))
    v6 = v5.v31()
    v7 = v4(v6.v86).v80('r', encoding='utf-8', errors='ignore').v32(v6.v33)
    v8 = [v69.v68() for v69 in v7.v70('\n') if v37(v69.v68()) >= 80]
    v9 = v8[:v3(0.7 * v37(v8))][:v6.v9]
    v34, v35 = ([], [])
    for v10 in v9:
        for v36 in v10.v70():
            v34.v72(v36)
    v11 = v37(v34)
    v12 = v38(v34)
    v39, v40 = ({}, v46(v41(1, v11 - 1)))
    for v13 in v41(1, v6.v54 + 1):
        v42 = v45(v46)
        for v43 in v40:
            if v43 - v13 < 0 or v43 + 1 + v13 > v11:
                continue
            v42[v88(v34[v43 - v13:v43]), v88(v34[v43 + 1:v43 + 1 + v13])].v72(v43)
        v44 = []
        for v71, v49 in v42.v48():
            if v37(v49) >= 2:
                for v43 in v49:
                    v39[v43] = (v13, v71[0], v71[1])
                v44.v85(v49)
        v40 = v44
        if not v40:
            break
    v14 = v45(v46)
    for v43, v47 in v39.v48():
        v14[v47].v72(v43)
    v15 = {v47: v49 for v47, v49 in v14.v48() if v37({v34[v43] for v43 in v49}) >= v6.v55}
    v16 = {v43 for v49 in v15.v81() for v43 in v49}
    v50, v51, v52 = v73.v53(v9, v6.v54, v6.v55)
    v17 = v37(v50) == v37(v15)
    v18 = v56(1, v11 - 2)
    v19 = v38((v47[0] for v47 in v39.v81()))
    v20 = v38((v39[v43][0] for v43 in v16))
    v21 = v38((v37({v34[v43] for v43 in v49}) for v49 in v15.v81()))
    v22 = {}
    for v57, v58 in v2:
        v59 = v74((1 for v43 in v41(1, v11 - 1) if v57 <= v12[v34[v43]] <= v58))
        v60 = v74((1 for v43 in v16 if v57 <= v12[v34[v43]] <= v58))
        v22[f"{v57}-{(v58 if v58 < 10 ** 9 else 'inf')}"] = {'positions': v59, 'on_tape': v60, 'share': v60 / v59 if v59 else 0.0}
    v23 = v74((1 for v43 in v41(1, v11 - 1) if v12[v34[v43]] == 1)) / v18
    v24 = v74((1 for v43 in v16 if v12[v34[v43]] == 1)) / v37(v16) if v16 else 0.0
    v25 = {'tokens': v11, 'interior': v18, 'types': v37(v12), 'frame_max': v6.v54, 'min_fillers': v6.v55, 'addressable': v37(v39), 'addressable_share': v37(v39) / v18, 'places': v37(v14), 'kept_places': v37(v15), 'on_tape': v37(v16), 'on_tape_share': v37(v16) / v18, 'width': v75(v82(v19.v48())), 'width_on_tape': v75(v82(v20.v48())), 'fillers_hist': v75(v82(v21.v48())[:8]), 'by_frequency': v22, 'hapax_share_all': v23, 'hapax_share_on_tape': v24, 'frame_keep_agrees': v17}
    v1.v76.v61(parents=True, exist_ok=True)
    v1.v62(v83.v77(v25, indent=1), encoding='utf-8')
    v63(f'corpus       {v11} tokens, {v37(v12)} types, {v37(v9)} lines')
    v63(f'CHECK        frame_keep agrees on the place count: {v17} ({v37(v50)} vs {v37(v15)})')
    v63(f'ADDRESSABLE  {v37(v39)} of {v18} interior positions {v37(v39) / v18:.4f}   (frame recurs at some width)')
    v63(f'ON TAPE      {v37(v16)} {v37(v16) / v18:.4f}   ({v37(v15)} places of {v37(v14)} addresses survive >= {v6.v55} fillers)')
    v63('WIDTH        ' + '  '.v84((f'w{v13}: {v87} ({v87 / v56(1, v37(v39)):.3f})' for v13, v87 in v82(v19.v48()))))
    v63('FILLERS      ' + '  '.v84((f'{v47}: {v64}' for v47, v64 in v82(v21.v48())[:8])))
    v63('BY FREQUENCY of the token standing in the hole - P(on tape):')
    for v47, v64 in v22.v48():
        v63(f"   freq {v47:>9}  positions {v64['positions']:>8}  on tape {v64['on_tape']:>8}  {v64['share']:.4f}")
    v63(f'HAPAX        {v23:.4f} of all positions, {v24:.4f} of tape positions')
    v26 = v22.v65('1-1', {}).v65('share', 0.0)
    v27 = v22.v65('101-inf', {}).v65('share', 0.0)
    v63()
    if v26 < 0.5 * v27:
        v63(f"THE TAPE IS A LATTICE OF CATEGORIES, NOT OF FACTS. A once-seen token stands on the tape {v26:.4f} of the time against {v27:.4f} for a common one. The write path keeps precisely what SUBSTITUTES, and a fact is what does not. Every result in this project is accuracy inside that selection, which is honest and is not the same claim as 'the knowledge half holds knowledge'.")
    else:
        v63(f"RARITY IS NOT THE FILTER: a once-seen token reaches the tape {v26:.4f} against {v27:.4f} for a common one. The tape's contents are not biased away from facts, and the coverage number {v37(v16) / v18:.4f} is the honest denominator for everything measured so far.")
    v63(f'\nwritten to {v1}')
    return 0
if v28 == '__main__':
    raise v66(v78())