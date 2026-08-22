"""THE TAPE HAS NEVER BEEN ALLOWED TO HOLD A FACT. Two filters, and both were deliberate.

358 FAILED ITS GATE AND TOLD US WHY. Ingestion moved everything in the right direction and
nothing far enough:

    askable   base .2612   null .2627   ingest .2743      by thirds  .2684 .2694 .2849
    reach     base .0384   null .0384   ingest .0434      by thirds  .0417 .0421 .0461
    null and base are FLAT down the document; ingest RISES, on both rows, monotonically.
    gain over null +0.0049, late-early +0.0064.  Gate wanted +0.05.

The mechanism is alive and correctly shaped - a document's own past makes its later holes exist
and reachable, and only its own. It is small because the write path deletes almost all of what
ingestion creates. THE SUSPECT WAS DECLARED BEFORE THE RUN and it is confirmed, but it is
bigger than one filter:

  1. `min_fillers >= 2` DELETES CONSTANT FRAMES. A frame that always holds the same token is
     not substitutable, so it is dropped. Self-reference - an article repeating its subject -
     produces exactly constant frames.
  2. THE EXAM CANNOT ASK ABOUT ONE EVEN IF KEPT. The lens is the place's OTHER fillers and the
     offer excludes the lens value (`w != v`), so for a constant place the truth is excluded
     from its own offer by construction.

TOGETHER THEY DEFINE THE EXAM AS "GUESS THE ALTERNATIVE" AND NEVER "RECALL THE VALUE". Every
number for 350 steps is substitution. But a FACT IS A CONSTANT FRAME - "X was born in Y" holds
Y and only ever Y - and recall is what knowing one means. The project's own thesis says
knowledge is counts and the mind is the decision, so recall being count-trivial is not a
problem: IT IS THE KNOWLEDGE HALF FINALLY HOLDING SOMETHING. The mind's job becomes WHICH
CHANNEL THIS HOLE WANTS, which is a decision with cardinality on both sides - Phi's kind.

WHAT IS MEASURED, torch-free, on a tape built with min_fillers=1 so constant frames survive:

    RECALL       the place's other POSITIONS (this position excluded, the place kept) - what
                 stood here before. Position-level exclusion, not place-level.
    SUBST        today's offer: place-level ban, lens = other distinct fillers, `w != v`.
    ORACLE       the truth in either.                     THE CEILING OF A PERFECT CHANNEL PICK
    RULE         a count, no heuristic: recall if the place's others are all one token.
    NULL         recall from a RANDOM OTHER PLACE of the same size. If recall pays here the
                 tape is not holding anything, it is guessing frequent tokens.

  GATE  oracle - max(always recall, always subst) > 0.05 AND recall - null > 0.05. The first
        says a channel decision exists to be made; the second says recall is real.

    python _audit359_recall.py
    python _audit359_recall.py --min-fillers 2     # the old world, for the contrast
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage359_recall.json')

def main() -> v2:
    v4 = v64.v24()
    v4.v25('--bytes', type=v2, default=30000000)
    v4.v25('--frame-max', type=v2, default=3)
    v4.v25('--min-fillers', type=v2, default=1)
    v4.v25('--lines', type=v2, default=25000)
    v4.v25('--window-lines', type=v2, default=400)
    v4.v25('--topm', type=v2, default=8)
    v4.v25('--max-questions', type=v2, default=4000)
    v4.v25('--seed', type=v2, default=1337)
    v4.v25('--corpus', default=v95(v0))
    v5 = v4.v26()
    v6 = v3(v5.v104).v96('r', encoding='utf-8', errors='ignore').v27(v5.v28)
    v7 = [v66.v65() for v66 in v6.v97('\n') if v91(v66.v65()) >= 80]
    v8 = v7[:v2(0.7 * v91(v7))][:v5.v8]
    v9 = v67.v29(v5.v30)
    if v5.v31 and v5.v31 < v91(v8):
        v18 = v9.v68(v91(v8) - v5.v31)
        v8 = v8[v18:v18 + v5.v31]
    v32, v33, v34 = v69.v35(v8, v5.v36, v5.v37)
    if not v32:
        v62('no tape')
        return 1
    v10 = [v41(v13) for v98, v13 in v32]
    v11 = {}
    for v38, v13 in v39(v10):
        for v18 in v13:
            v11[v18] = v38
    v12 = v40(v41)
    for v13 in v10:
        for v18 in v13:
            v12[v33[v18]].v70(v18)
    v14 = v40(v41)
    for v38, v13 in v39(v10):
        v14[v91(v13)].v70(v38)
    v15 = {}

    def co(v42):
        v17 = v15.v71(v42)
        if v17 is None:
            v17 = v44()
            for v18 in v12[v42]:
                for v99 in v10[v11[v18]]:
                    if v33[v99] != v42:
                        v17[v33[v99]] += 1
            v15[v42] = v17
        return v17
    v16 = [v18 for v13 in v10 for v18 in v13]
    v9.v43(v16)
    v16 = v16[:v5.v72]
    v17 = v44()
    for v18 in v16:
        v38 = v11[v18]
        v45 = v33[v18]
        v46 = [v73 for v73 in v10[v38] if v73 != v18]
        v17['n'] += 1
        v17['const'] += v91({v33[v73] for v73 in v46}) == 1 if v46 else 0
        v47 = v44((v33[v73] for v73 in v46))
        v48 = v47.v74(2)
        v49 = v45 in {v100 for v100, v105 in v47.v74(v5.v59)}
        v50 = v14.v71(v91(v10[v38]), ())
        v51 = False
        for v52 in v75(4):
            if not v50:
                break
            v76 = v50[v9.v68(v91(v50))]
            if v76 != v38:
                v51 = v45 in {v33[v73] for v73 in v10[v76][:v5.v59]}
                break
        v53 = v44((v33[v73] for v73 in v10[v38]))
        v53[v45] -= 1
        if v53[v45] <= 0:
            del v53[v45]
        v54 = v41(v53)[:6]
        v55 = False
        v77, v78, v79 = ([], [], v44())
        if v54:
            v17['subst_askable'] += 1
            v80 = v44((v33[v73] for v73 in v10[v38]))
            v79 = v44()
            for v42 in v54:
                for v100, v19 in v106(v42).v103():
                    v19 -= v80.v71(v100, 0)
                    if v19 > 0 and v100 != v42:
                        v79[v100] += v19
            v78 = v79.v74(v5.v59)
            v77 = v78[:2]
            v55 = v45 in {v100 for v100, v105 in v78}
        v17['recall'] += v49
        v17['subst'] += v55
        v17['null'] += v51
        v17['oracle'] += v49 or v55
        v17['both'] += v49 and v55
        v17['only_recall'] += v49 and (not v55)
        v17['only_subst'] += v55 and (not v49)

        def gap(v81):
            if not v81:
                return (-1, -1)
            return (v81[0][1], v81[0][1] - v81[1][1] if v91(v81) > 1 else v81[0][1])
        v82, v83 = v84(v48)
        v85, v86 = v84(v77)
        v17['rule_agree'] += v49 if v46 and v91({v33[v73] for v73 in v46}) == 1 else v55
        v17['rule_top1'] += v49 if v82 >= v85 else v55
        v17['rule_margin'] += v49 if v83 >= v86 else v55
        v17['rule_rep'] += v49 if v82 >= 2 else v55
        v56 = v47 + (v79 if v54 else v44())
        v17['rival_merge'] += v45 in {v100 for v100, v105 in v56.v74(v5.v59)}
        v87, v88, v89 = ([], v41(v47.v74(v5.v59)), v41(v78 or ()))
        while v91(v87) < v5.v59 and (v88 or v89):
            for v90 in (v88, v89):
                if v90 and v91(v87) < v5.v59:
                    v100 = v90.v107(0)[0]
                    if v100 not in v87:
                        v87.v70(v100)
        v17['rival_inter'] += v45 in v101(v87)
    v19 = v57(1, v17['n'])
    v20 = {v58: v17[v58] / v19 for v58 in ('recall', 'subst', 'null', 'oracle', 'both', 'only_recall', 'only_subst', 'const', 'subst_askable', 'rule_agree', 'rule_top1', 'rule_margin', 'rule_rep', 'rival_merge', 'rival_inter')}
    v21 = {'lines': v91(v8), 'places': v91(v10), 'questions': v17['n'], 'topm': v5.v59, 'min_fillers': v5.v37, **v20}
    v21['best_fixed'] = v57(v20['recall'], v20['subst'])
    v21['best_rival'] = v57(v21['best_fixed'], v20['rule_agree'], v20['rule_top1'], v20['rule_margin'], v20['rule_rep'], v20['rival_merge'], v20['rival_inter'])
    v21['decision'] = v20['oracle'] - v21['best_fixed']
    v21['headroom'] = v20['oracle'] - v21['best_rival']
    v21['recall_over_null'] = v20['recall'] - v20['null']
    v1.v92.v60(parents=True, exist_ok=True)
    v1.v61(v102.v93(v21, indent=1), encoding='utf-8')
    v62(f"tape     {v91(v10)} places, {v17['n']} questions, min_fillers {v5.v37}, topm {v5.v59}")
    v62(f"         constant places {v20['const']:.4f}   substitution-askable {v20['subst_askable']:.4f}")
    v62(f"RECALL   {v20['recall']:.4f}   null {v20['null']:.4f}   over null {v21['recall_over_null']:+.4f}")
    v62(f"SUBST    {v20['subst']:.4f}   (today's only channel)")
    v62(f"ORACLE   {v20['oracle']:.4f}   both {v20['both']:.4f}   only recall {v20['only_recall']:.4f}   only subst {v20['only_subst']:.4f}")
    v62(f"RIVALS   agree {v20['rule_agree']:.4f}   top1 {v20['rule_top1']:.4f}   margin {v20['rule_margin']:.4f}   repeated {v20['rule_rep']:.4f}")
    v62(f"NO-PICK  merge {v20['rival_merge']:.4f}   interleave {v20['rival_inter']:.4f}   (both at the same top-m a chooser gets)")
    v62(f"DECISION oracle over the best fixed channel {v21['decision']:+.4f}   over the BEST RIVAL of any kind {v21['headroom']:+.4f}")
    v22 = v21['headroom'] > 0.05 and v21['recall_over_null'] > 0.05
    if v22:
        v62(f"\nTHERE ARE TWO CHANNELS AND CHOOSING BETWEEN THEM IS WORTH {v21['decision']:+.4f}. The tape can hold a FACT - a constant frame recalled, not an alternative guessed - and the write path has been deleting them since the first commit. Recall is count-trivial and that is correct: it is the knowledge half. The mind's new job is WHICH CHANNEL THIS HOLE WANTS, one decision, one objective, and NO COUNT MAKES IT - the best rival of any kind reads {v21['best_rival']:.4f} against an oracle of {v20['oracle']:.4f}.")
    elif v21['recall_over_null'] > 0.05:
        v62(f"\nRECALL IS REAL BUT NOT A CHOICE: {v20['recall']:.4f} against a null of {v20['null']:.4f}, yet the oracle beats the best fixed channel by only {v21['decision']:+.4f}. Add recall to the tape as a channel, give the mind nothing new to decide.")
    else:
        v62("\nRECALL IS NOT REAL: reading a place's own past beats a random place of the same size by nothing. Constant frames carry no more than token frequency, the two filters cost nothing, and substitution is the whole substrate after all.")
    v62(f'\nwritten to {v1}')
    return 0
if v23 == '__main__':
    raise v63(v94())