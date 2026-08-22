"""THE CEILING OF A CLOSED PATH. Does the trace of the walk say anything the offer does not?

WHAT THIS IS NOT, because three neighbouring operations are already closed:
  * NOT a wider offer. The offer stays eight - 347 measured four times what widening costs, and
    393 measured that committing to one place loses to the merge.
  * NOT a second read that adds candidates. 388's hop2@8 did not beat hop1@8.
  * NOT 387's rerank, which reordered the SAME evidence by a different rule. This brings evidence
    the offer never had: where the walk goes FROM each candidate.

WHAT THE TRACE IS. GPT's understanding does not live in its weights - it lives in the context: the
partial result becomes the next input. This project has no such state at all; every read starts
from zero (30.6). The state that can be carried here without breaking the separation contract is
the PATH, never the values: addresses are structure, values are facts. 351 made the same argument
for "where it stands" - policy state, not a fact.

THE MEASUREMENT. For each candidate c of the question's own eight, step to the place c actually
stands at and walk from THERE. Does that walk come back - is the question's place among the k it
reaches?

    close(c)   the walk from c's place reaches the question's place
    truth      close() on the true candidate
    decoy      close() on a FREQUENCY-MATCHED candidate from the same offer
    random     close() from a random place of the same size          the floor

A path that closes is a consequence of the chain: my place points at c AND c's place points back.
Nothing about the value is carried - only whether the two addresses see each other.

  VOID CHECK, READ FIRST
      `close_rate` overall. If almost every candidate closes, the relation is symmetric by
      construction and carries nothing; if almost none does, there is no signal to rank on.

  GATE
      truth - decoy > 0.05 AND truth - random > 0.05, on 3 of 3 seeds. The decoy is the one that
      matters: it says the truth closes because it is the truth, not because it is frequent.

  If it fails, the trace carries nothing on this tape and "consequence of a chain" is not
  available here - which closes the last untried shape rather than leaving it as a hope.

    python _audit402_trace.py
    python _audit402_trace.py --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
import _audit390_address as A
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage402_trace.json')

def closes(v4, v5, v6, v7, v8, v9=()):
    """Does the walk from `from_pid` reach `pid`? The arm's own step - filler overlap, cut at k -
    with ONE correction that decides the whole measurement.

    THE SECTION 27 LEAK, IN THE RETURN DIRECTION. `T["prof"][pid]` still contains the HIDDEN
    token: the question's place holds the truth, that is what makes it the answer. A plain walk
    from the candidate's place would therefore find the question's place THROUGH THE HIDDEN
    TOKEN - the path would close because we hid the answer there, not because the two addresses
    see each other. One mention is subtracted from the target's overlap and from its norm, which
    is the same subtraction `reach_places` already makes on the query side.
    """
    if v6 is None or v6 == v5:
        return None
    v10 = v4['prof'][v6]
    v11 = v28()
    for v29, v30 in v10.v31():
        for v32 in v4['at_value'].v64(v29, ()):
            if v32 == v6 or v32 in v9:
                continue
            v65 = v4['prof'][v32][v29] - (1 if v32 == v5 and v29 == v7 else 0)
            if v65 > 0:
                v11[v32] += v30 * v65
    v12 = v93.v66(v94((v16 * v16 for v16 in v10.v108()))) or 1.0

    def norm(v32):
        if v32 != v5:
            return v4['norm'][v32]
        v33 = v4['prof'][v32].v64(v7, 0)
        v34 = v94((v16 * v16 for v16 in v4['prof'][v32].v108())) - (2 * v33 - 1 if v33 > 0 else 0)
        return v93.v66(v34) if v34 > 0 else 1.0
    v13 = v67(v11, key=lambda v32: (-(v11[v32] / (v12 * v109(v32))), v32))[:v8]
    return v5 in v68(v13)

def run(v4, v14, v15):
    v16 = v28()
    v35, v36, v37 = (v4['toks'], v4['place_of'], v4['owner'])
    v17 = [v18 for v69 in v4['places'] for v18 in v69]
    v15.v38(v17)
    for v18 in v17:
        if v16['n'] >= v14.v70:
            break
        v5 = v36[v18]
        v39 = v35[v18]
        v40 = {v35[v95] for v95 in v4['places'][v5] if v95 != v18}
        if not v40 or v39 in v40:
            continue
        v41 = v28((v35[v95] for v95 in v4['places'][v5] if v95 != v18))
        v9 = v68(v4['on_line'][v37[v18]])
        v9.v71(v5)
        v42 = v84.v72(v4, v5, v41, v14.v73, v9)
        v74, v43, v75 = ({}, [], v68(v40))
        for v32 in v42:
            for v29, v96 in v4['prof'][v32].v97():
                if v29 not in v75:
                    v75.v105(v29)
                    v74[v29] = v32
                    v43.v106(v29)
        v43 = v43[:v14.v98]
        if v39 not in v43:
            continue
        v16['n'] += 1

        def close_of(v29):
            return v99(v4, v5, v74.v64(v29), v39, v14.v73, v9)
        v44 = v76(v39)
        if v44 is not None:
            v16['truth_n'] += 1
            v16['truth'] += v2(v44)
        v45 = v84.v77(v4, v39, v40 | {v39}, v15)
        if v45 is not None and v45 in v43:
            v78 = v76(v45)
            if v78 is not None:
                v16['decoy_n'] += 1
                v16['decoy'] += v2(v78)
        v32 = v15.v79(v85(v4['places']))
        if v32 != v5 and v32 not in v9:
            v78 = v99(v4, v5, v32, v39, v14.v73, v9)
            if v78 is not None:
                v16['rand_n'] += 1
                v16['rand'] += v2(v78)
        for v29 in v43:
            v78 = v76(v29)
            if v78 is not None:
                v16['any_n'] += 1
                v16['any'] += v2(v78)
    return v16

def main() -> v2:
    v19 = v80.v46()
    v19.v47('--bytes', type=v2, default=30000000)
    v19.v47('--frame-max', type=v2, default=3)
    v19.v47('--min-fillers', type=v2, default=1)
    v19.v47('--lines', type=v2, default=25000)
    v19.v47('--window-lines', type=v2, default=400)
    v19.v47('--places', type=v2, default=8)
    v19.v47('--topm', type=v2, default=8)
    v19.v47('--max-questions', type=v2, default=3000)
    v19.v47('--seed', type=v2, default=1337)
    v19.v47('--corpus', default=v90(v0))
    v19.v47('--out', default=v90(v1))
    v14 = v19.v48()
    v20 = v3(v14.v107).v100('r', encoding='utf-8', errors='ignore').v49(v14.v50)
    v21 = [v82.v81() for v82 in v20.v101('\n') if v85(v82.v81()) >= 80]
    v22 = v21[:v2(0.7 * v85(v21))][:v14.v22]
    v15 = v83.v51(v14.v52)
    if v14.v53 and v14.v53 < v85(v22):
        v54 = v15.v79(v85(v22) - v14.v53)
        v22 = v22[v54:v54 + v14.v53]
    v4 = v84.v55(v22, v14.v56, v14.v57)
    if not v4['places']:
        v59('no tape')
        return 1
    v16 = v58(v4, v14, v15)

    def rate(v8):
        return v16[v8] / v102(1, v16[v8 + '_n'])
    v23 = {'seed': v14.v52, 'n': v16['n'], 'places': v85(v4['places']), 'close_rate': v86('any'), 'truth': v86('truth'), 'decoy': v86('decoy'), 'random': v86('rand'), 'truth_n': v16['truth_n'], 'decoy_n': v16['decoy_n']}
    v23['truth_minus_decoy'] = v23['truth'] - v23['decoy']
    v23['truth_minus_random'] = v23['truth'] - v23['random']
    v59(f"{v23['places']} places, {v23['n']} questions where the offer holds the truth")
    v59(f"VOID CHECK  close_rate {v23['close_rate']:.4f}  <- read first: near 1 or near 0 and the relation carries nothing to rank on")
    v59(f"CLOSES      truth {v23['truth']:.4f} ({v23['truth_n']})   decoy {v23['decoy']:.4f} ({v23['decoy_n']})   random {v23['random']:.4f}")
    v59(f"            truth-decoy {v23['truth_minus_decoy']:+.4f}   truth-random {v23['truth_minus_random']:+.4f}")
    v24 = v23['truth_minus_decoy'] > 0.05 and v23['truth_minus_random'] > 0.05
    v23['gate'] = v60(v24)
    v59('\nTHE PATH CLOSES ON THE TRUTH. The trace carries evidence the offer does not, it is not frequency, and it is structure rather than content - so it can be state without breaking the separation contract.' if v24 else '\nTHE PATH SAYS NOTHING: ' + ('the truth closes no more often than a frequency-matched twin. ' if v23['truth_minus_decoy'] <= 0.05 else '') + ('it closes no more often than a random place. ' if v23['truth_minus_random'] <= 0.05 else '') + 'The trace is not evidence on this tape.')
    v25 = v3(v14.v25)
    v25.v87.v61(parents=True, exist_ok=True)
    v26 = v103.v89(v25.v104()) if v25.v88() else {}
    v26[v90(v14.v52)] = v23
    v25.v62(v103.v91(v26, indent=1))
    v59(f'wrote {v25}')
    return 0
if v27 == '__main__':
    raise v63(v92())