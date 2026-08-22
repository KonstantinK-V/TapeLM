"""THE DUAL OF SHAPE: DOES THE LINE CLOSE WHAT THE REST OF THE SCOPE LACKS? Torch-free.

399 asked "is the line LIKE the rest of the def" and the answer was a coin on three seeds, below
the SIZE confound every time (section 39). This is the dual question, and it is the second and
last object drawn from that failure: not similarity but DEFICIT - the line brings node types the
remainder does NOT have, because a body missing exactly this line is missing exactly what it does.

THE PHRASE, DECLARED BEFORE THE RUN

    deficit(S)   = how many of the line's AST node TYPES are ABSENT from S's remaining body
    twin(S)      = the scope of this file whose remaining body has the NEAREST line count to S's,
                   S itself excluded; ties by tape order, so the twin is deterministic and no
                   number moves with a draw
    score(S)     = deficit(S) - deficit(twin(S))

The twin is what makes this not the size prior wearing a new hat: a small scope lacks more types
for the same reason it holds fewer of everything, and subtracting a size-matched scope's deficit
removes precisely that. How well it matches is REPORTED (`twin_gap`), so "size was removed" is a
number rather than a claim.

THE DIRECTION IS DECLARED AND WILL NOT BE FLIPPED. The true scope is the one that MISSES what the
line brings, so the argmax of `score` is the attachment. If the measurement comes back symmetric
below the coin - which would mean the argmin wins - that is a FINDING and not a licence to turn
the sign around after the fact. Reversing a declared direction on seeing the number is the
after-the-fact rescue this project has refused four times.

NO IDENTIFIER IS READ. The feature is `type(node).__name__` and nothing else, exactly as in 399,
and `_check400_deficit.py` re-asserts the equivariance: rename every symbol, every number
unchanged.

ONE BIT, REPORTED SEPARATELY because a diffuse deficit can be noisy while a single closure is not:
a `Return` line offered to a scope whose REMAINING body has no `Return` at all. Its own accuracy
and its own population mass, beside the general score and never blended into it.

    POPULATION   only `amb_live` - the lines where the name count ties at a non-zero score, which
                 is 23% of the corpus and the only place a decision exists (38.3). The full
                 population is the PAIRED CONTROL, not the target.
    RIVALS       the coin (1/|tied|), the UNDIVIDED name overlap (a coin there by construction),
                 SIZE (the incumbent carrier, +0.03..+0.08 at 39), and the RAW deficit without
                 the twin - the control that says whether the twin did the work.

  VOID CHECK, READ FIRST
      `twin_gap` - the mean difference in remaining lines between a candidate and its twin. If the
      twin cannot match size, the subtraction did not remove the size prior and nothing below is
      about deficit.

  GATE, as Kostya set it
      score - coin > 0.05 on 3 of 3 seeds, AND score >= the name count.
      DECLARED READING, not a gate clause: if the score beats the coin but sits BELOW SIZE, that
      is section 39's verdict repeating in another currency and it is NOT a pass.

  If it fails: the pair (two pooled lines into one def) is a SEPARATE step with its own phrase, or
  the honest end - on code there is no evidence above a counter of scopes and a size prior. No 1b,
  no third form.

    python _audit400_deficit.py
    python _audit400_deficit.py --seed 8642
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import _audit398_scope as S
import _audit399_shape as H
v0 = v2('results/_stage400_deficit.json')

def twin_of(v3, v4):
    """The scope with the nearest remaining-line count, itself excluded. Deterministic: nearest
    first, then the lowest index, so the twin is a property of the file and not of a draw."""
    v18, v19 = (None, None)
    for v5 in v20(v42(v3)):
        if v5 == v4:
            continue
        v21 = v43(v3[v5] - v3[v4])
        if v18 is None or v21 < v18:
            v18, v19 = (v21, v5)
    return (v19, 0 if v19 is None else v43(v3[v19] - v3[v4]))

def run(v6, v7, v8):
    v9 = v22()
    for v10 in v6:
        try:
            v44 = v2(v10).v74(encoding='utf-8', errors='ignore')
        except v45:
            continue
        v23 = v75.v46(v44)
        if not v23 or v42(v23['funcs']) < v7.v76:
            continue
        v47, v48 = v77.v49(v44, v23['owner'])
        if v47 is None:
            continue
        v24 = v42(v23['funcs'])
        v25 = v22(v23['owner'].v78())
        v26 = [v28 for v28 in v34(v23['owner']) if v28 not in v23['heads']]
        v8.v50(v26)
        v27 = 0
        for v28 in v26:
            if v27 >= v7.v79:
                break
            v51 = v23['used'].v80(v28, v86())
            if not v51:
                continue
            v27 += 1
            v52 = v23['owner'][v28]
            v53 = v75.v81(v23, v28)
            v54 = [v75.v87(v51, v75.v95(v23, v4, v28), v53, True) for v4 in v20(v24)]
            v55 = [v75.v87(v51, v75.v95(v23, v4, v28), v53, False) for v4 in v20(v24)]
            v56 = v66(v54)
            v57 = [v4 for v4 in v20(v24) if v54[v4] == v56]
            v9['full_n'] += 1
            if v42(v57) < 2 or v56 <= 0.0 or v52 not in v57:
                continue
            v9['n'] += 1
            v3 = [v25[v4] - (1 if v4 == v52 else 0) for v4 in v20(v24)]
            v58 = {v4: v77.v88(v48, v4, v28) for v4 in v20(v24)}
            v59 = v47.v80(v28, v86())
            v60 = [v42([v96 for v96 in v59 if v96 not in v58[v4]]) for v4 in v20(v24)]
            v61 = []
            for v4 in v20(v24):
                v89, v90 = v91(v3, v4)
                v9['twin_gap'] += v90 if v4 in v57 else 0
                v9['twin_n'] += 1 if v4 in v57 else 0
                v61.v92(v93(v60[v4] - (v60[v89] if v89 is not None else 0)))
            v62 = [v93(v3[v4]) for v4 in v20(v24)]
            v9['tied_size'] += v42(v57)
            v9['coin'] += 1.0 / v42(v57)
            v9['deficit'] += v77.v82(v61, v52, v57)
            v9['deficit_raw'] += v77.v82([v93(v97) for v97 in v60], v52, v57)
            v9['rawname'] += v77.v82(v55, v52, v57)
            v9['size'] += v77.v82(v62, v52, v57)
            if 'Return' in v59:
                v9['bit_n'] += 1
                v83 = [1.0 if 'Return' not in v58[v4] else 0.0 for v4 in v20(v24)]
                v9['bit'] += v77.v82(v83, v52, v57)
                v9['bit_coin'] += 1.0 / v42(v57)
                v84 = {v4: v77.v88(v48, v4, -1) for v4 in v57}
                v9['bit_rivals'] += v42(v57) - 1
                v9['bit_rivals_noret'] += v94((1 for v4 in v57 if v4 != v52 and 'Return' not in v84[v4]))
                v9['bit_true_noret_full'] += v1('Return' not in v84[v52])
    return v9

def main() -> v1:
    v11 = v63.v29()
    v11.v30('--files', default='_*.py')
    v11.v30('--per-file', type=v1, default=40)
    v11.v30('--min-scopes', type=v1, default=8)
    v11.v30('--seed', type=v1, default=1337)
    v11.v30('--out', default=v71(v0))
    v7 = v11.v31()
    v8 = v64.v32(v7.v33)
    v6 = v34(v2('.').v65(v7.v6))
    v9 = v35(v6, v7, v8)
    v36, v37 = (v66(1, v9['n']), v66(1, v9['bit_n']))
    v12 = {'seed': v7.v33, 'files': v42(v6), 'scored': v9['full_n'], 'decided_on': v9['n'], 'amb_live_with_truth': v9['n'] / v66(1, v9['full_n']), 'tied_size': v9['tied_size'] / v36, 'twin_gap': v9['twin_gap'] / v66(1, v9['twin_n']), 'coin': v9['coin'] / v36, 'deficit': v9['deficit'] / v36, 'deficit_raw': v9['deficit_raw'] / v36, 'rawname': v9['rawname'] / v36, 'size': v9['size'] / v36, 'bit_n': v9['bit_n'], 'bit': v9['bit'] / v37, 'bit_coin': v9['bit_coin'] / v37, 'bit_rivals_noret': v9['bit_rivals_noret'] / v66(1, v9['bit_rivals']), 'bit_true_noret_full': v9['bit_true_noret_full'] / v37}
    v12['deficit_minus_coin'] = v12['deficit'] - v12['coin']
    v12['deficit_minus_name'] = v12['deficit'] - v12['rawname']
    v12['deficit_minus_size'] = v12['deficit'] - v12['size']
    v12['bit_minus_coin'] = v12['bit'] - v12['bit_coin']
    v38(f"{v12['files']} files, {v12['scored']} lines scored, decided on {v12['decided_on']} ({v12['amb_live_with_truth']:.4f} of them), {v12['tied_size']:.2f} tied scopes each")
    v38(f"VOID CHECK  twin_gap {v12['twin_gap']:.3f} lines   <- read first: how far the size-twin misses. If it cannot match size, nothing below is about deficit")
    v38(f"TIE-BREAK   deficit {v12['deficit']:.4f}   coin {v12['coin']:.4f}   raw name {v12['rawname']:.4f}   size {v12['size']:.4f}   deficit w/o twin {v12['deficit_raw']:.4f}")
    v38(f"            deficit-coin {v12['deficit_minus_coin']:+.4f}   deficit-name {v12['deficit_minus_name']:+.4f}   deficit-size {v12['deficit_minus_size']:+.4f}")
    v38(f"ONE BIT     Return into a remainder with no Return: {v12['bit']:.4f} against {v12['bit_coin']:.4f} on {v12['bit_n']} lines  ({v12['bit_minus_coin']:+.4f})")
    v38(f"            IS IT THE CUT? rivals whose FULL body has no Return {v12['bit_rivals_noret']:.4f}; the true scope's full body has none {v12['bit_true_noret_full']:.4f}  <- if the first is ~0 the bit is detecting the removal, not the line")
    v13 = v12['deficit_minus_coin'] > 0.05 and v12['deficit'] >= v12['rawname']
    v14 = v12['deficit'] < v12['size']
    v12['gate'], v12['under_size'] = (v67(v13), v67(v14))
    if v13 and (not v14):
        v38('\nDEFICIT CARRIES THE TIE. The line closes what its own scope lacks, the size-twin removed the prior that carried 399, and the name count is a coin there. This is evidence about the LINE, and it is the first thing on this object that a counter does not already hold.')
    elif v13 and v14:
        v38("\nNOT A PASS, BY THE READING DECLARED BEFORE THE RUN: deficit beats the coin but sits BELOW SIZE, which is section 39's verdict in another currency. The twin did not remove the prior.")
    else:
        v38('\nDEFICIT DOES NOT CARRY THE TIE: ' + ('it does not beat the coin. ' if v12['deficit_minus_coin'] <= 0.05 else '') + ('the undivided name overlap does as well. ' if v12['deficit'] < v12['rawname'] else '') + 'The declared direction is not reversed on seeing the number. Next is the PAIR as its own step with its own phrase, or the honest end: on code there is no evidence above a counter of scopes and a size prior.')
    v15 = v2(v7.v15)
    v15.v68.v39(parents=True, exist_ok=True)
    v16 = v85.v70(v15.v74()) if v15.v69() else {}
    v16[v71(v7.v33)] = v12
    v15.v40(v85.v72(v16, indent=1))
    v38(f'wrote {v15}')
    return 0
if v17 == '__main__':
    raise v41(v73())