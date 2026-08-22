"""READING IS WRITING. The dependence wikitext actually has, and the project never used.

357 CLOSED RESOLUTION: gain over the permuted null +0.038 thin and +0.0325 thick - under the
gate, SHRINKING with thickness, and against the plain wide tape the thick setting is negative.
One position, one address costs nothing. That is the third audit in a row (354, 355, 357) where
the substrate under INDEPENDENT cloze questions had nothing more to give.

THE DIAGNOSIS, SAID PLAINLY. 354 and 355 did not close memory. They closed memory ON AN EXAM
WHOSE QUESTIONS ARE INDEPENDENT BY CONSTRUCTION - both drawn from a tape that already contained
everything either question needed. No memory can pay on an i.i.d. exam, and that is a property
of the exam, not of minds. The dependence wikitext DOES have, at exactly one scale, is
SELF-REFERENCE WITHIN A DOCUMENT: an article introduces its subject once and then repeats it,
paraphrases it, pronouns it, for forty lines. The first mention is unanswerable from outside;
every later mention is answerable FROM THE ARTICLE'S OWN EARLIER LINES and often from nowhere
else.

AND THE PROJECT HAS NEVER SEEN THIS, because the write path treats the corpus as one finished
bag of lines. The tape is built once; a document is never NEW; nothing is ever ingested. The
loop everyone kept reaching for is not "remember your answers" (354, lost) - it is READING A
DOCUMENT INTO THE WORLD, line by line, where the mind's work on line k is what makes line k+1
askable at all. The answer does not go into a lens set. IT BECOMES WORLD: the line joins the
corpus, frame_keep counts it, new places exist.

WHY THIS EVADES 354'S CLOSURE. There, tape and question shared one world; a remembered answer
was a lens pointing at content the tape already indexed, so a wrong answer of the same shape
bought the same offer. Here the document is NOT in the old world. Its own earlier lines are the
only place its later holes' fillers stand. The null is the same-shaped prefix of a DIFFERENT
document: same added mass, same added places, wrong content. If ingestion is just
offer-widening the null matches it and this closes like everything else.

WHAT IS MEASURED, torch-free, counts only. For each document (a contiguous run of doc-lines
lines from OUTSIDE the old window) and each line k:

    BASE     tape = old world + line k                       the mind never read the document
    INGEST   tape = old world + doc[:k] + line k             it read every line before this one
    NULL     tape = old world + other-doc[:k] + line k       it read the same AMOUNT of noise

    questions = every token position of line k; a position not on a tape counts 0 for that arm,
    so all three arms share one absolute denominator (the 342a lesson).

    reported pooled and BY THIRDS of k - the signature of real ingestion is a gain that GROWS
    down the document, because more of the article's own paradigm exists to stand on.

  GATE  pooled reach_ingest - reach_null > 0.05, with the late third strictly above the early.

    python _audit358_ingest.py
    python _audit358_ingest.py --docs 20 --doc-lines 16     # quick pass
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage358_ingest.json')

def reach_line(v4, v5, v6, v7, v8):
    """corpus is a list of lines whose LAST line is the question line. Every token of that last
    line is a question; the denominator is all of them, askable or not.

    THE SECOND CHANNEL, ADDED AFTER 358'S POST-MORTEM AND NOT INSTEAD OF THE FIRST. 8-RESULT-6
    named TWO reasons the write path deletes what ingestion creates, and `--min-fillers 1` lifts
    only the first:

      1. min_fillers >= 2 deletes CONSTANT frames, which is exactly what self-reference makes.
      2. EVEN KEPT, THE EXAM CANNOT ASK ABOUT ONE. The lens is the place's other fillers and the
         offer excludes the lens value (`w != v`), so a constant place's truth is excluded from
         its own offer BY CONSTRUCTION - verified there on a frame holding only XARWIN.

    So running this file at min_fillers 1 with the old columns alone would admit the places
    ingestion creates and then score every one of them a miss, and the gate would read "there is
    nowhere to step" when the truth is "the exam still cannot ask". That is 33-VOID's mistake run
    backwards, and it is the reason for the recall column.

      subst   the old number, computed by the OLD LINES, bit for bit: the lens is the place's
              other fillers, the offer excludes them, `hit` is unchanged and comparable to
              every number 358 has on record.
      recall  the truth among the place's OTHER POSITIONS, this position excluded - 359's
              channel. For a constant place this is the whole content of the place; for a mixed
              one it is true only where the value repeats.
      oracle  either. What ingestion made answerable AT ALL, by whichever channel.

    RECALL IS NOT FREE AND IS NOT TRIVIALLY ONE. Under `base` a place needs its repetitions in
    the OLD world; under `null` the same mass of foreign text gets the same opportunity. Only
    under `ingest` can the document's own earlier lines be what the place is made of. The null
    is what separates "recall is real" from "recall is trivial", and it is 358's own null.

    CONST/MIXED, for the two terminals. A position is `const` when the place, with this position
    taken out, holds ONE distinct value - the hub of a paradigm. Reported as the split of what
    ingestion ADDED (reached under ingest, not under base), because a design with two terminals
    and two rewards needs to know whether both of them carry any mass.
    """
    v41, v42, v43 = v84.v44(v4, v6, v7)
    v9 = v45(v4[-1].v85())
    if not v41:
        return (v87({'tok': v9}), {})
    v10 = [v49(v13) for v103, v13 in v41]
    v11 = {}
    for v46, v13 in v47(v10):
        for v18 in v13:
            v11[v18] = v46
    v12 = v48(v49)
    for v13 in v10:
        for v18 in v13:
            v12[v42[v18]].v104(v18)
    v14 = v45(v4) - 1
    v15 = [v18 for v18 in v11 if v43[v18] == v14]
    v16 = v50((v45(v90.v85()) for v90 in v4[:-1]))
    v17 = {}

    def co(v51):
        v52 = v17.v86(v51)
        if v52 is None:
            v52 = v87()
            for v18 in v12[v51]:
                for v105 in v10[v11[v18]]:
                    if v42[v105] != v51:
                        v52[v42[v105]] += 1
            v17[v51] = v52
        return v52
    v52, v53 = (v87(), {})
    v52['tok'] = v9
    for v18 in v15:
        v46 = v11[v18]
        v54 = v42[v18]
        v55 = v87((v42[v119] for v119 in v10[v46]))
        v55[v54] -= 1
        if v55[v54] <= 0:
            del v55[v54]
        v56 = v87((v42[v119] for v119 in v10[v46]))
        v56[v54] -= 1
        if v56[v54] <= 0:
            del v56[v54]
        v57 = v2(v54 in {v42[v119] for v119 in v10[v46] if v119 != v18})
        v58 = v2(v45(v56) == 0 or (v45(v56) == 1 and v57))
        v52['rec_ask'] += 1
        v52['rec'] += v57
        v52['const'] += v58
        v59 = v49(v55)[:6]
        if not v59:
            v53[v18 - v16] = (0, v57, v58)
            v52['orc'] += v57
            continue
        v60 = 1
        v52['ask'] += v60
        v61 = v87((v42[v119] for v119 in v10[v46]))
        v62 = v87()
        for v51 in v59:
            for v106, v107 in v125(v51).v108():
                v107 -= v61.v86(v106, 0)
                if v107 > 0 and v106 != v51:
                    v62[v106] += v107
        v63 = v2(v54 in {v106 for v106, v126 in v62.v127(v8)})
        v52['hit'] += v63
        v52['orc'] += v2(v63 or v57)
        v53[v18 - v16] = (v63, v57, v58)
    return (v52, v53)

def main() -> v2:
    v19 = v88.v64()
    v19.v65('--bytes', type=v2, default=30000000)
    v19.v65('--frame-max', type=v2, default=3)
    v19.v65('--min-fillers', type=v2, default=2)
    v19.v65('--lines', type=v2, default=25000)
    v19.v65('--window-lines', type=v2, default=400)
    v19.v65('--doc-lines', type=v2, default=24)
    v19.v65('--docs', type=v2, default=40)
    v19.v65('--topm', type=v2, default=8)
    v19.v65('--seed', type=v2, default=1337)
    v19.v65('--corpus', default=v109(v0))
    v20 = v19.v66()
    v21 = v3(v20.v4).v110('r', encoding='utf-8', errors='ignore').v67(v20.v68)
    v22 = [v90.v89() for v90 in v21.v85('\n') if v45(v90.v89()) >= 80]
    v23 = v22[:v2(0.7 * v45(v22))][:v20.v23]
    v24 = v91.v69(v20.v70)
    v25 = v24.v71(v92(1, v45(v23) - v20.v76))
    v26 = v23[v25:v25 + v20.v76]
    v27 = v23[:v25] + v23[v25 + v20.v76:]
    if v45(v27) < 3 * v20.v77:
        v80('not enough lines outside the window')
        return 1

    def draw_doc():
        v33 = v24.v71(v45(v27) - v20.v77)
        return (v27[v33:v33 + v20.v77], v33)
    v28 = ('base', 'ingest', 'null')
    v29 = {v33: v87() for v33 in v28}
    v30 = {v33: [v87(), v87(), v87()] for v33 in v28}
    for v31 in v72(v20.v73):
        v93, v94 = v95()
        while True:
            v111, v112 = v95()
            if v120(v112 - v94) > 2 * v20.v77:
                break
        for v74 in v72(1, v20.v77):
            v96 = v113(2, 3 * v74 // v20.v77)
            v97 = {}
            for v114, v115 in (('base', []), ('ingest', v93[:v74]), ('null', v111[:v74])):
                v4 = v26 + v115 + [v93[v74]]
                v121, v53 = v122(v4, v93[v74], v20.v6, v20.v7, v20.v8)
                v97[v114] = v53
                for v52 in (v29[v114], v30[v114][v96]):
                    for v123 in ('tok', 'ask', 'hit', 'rec_ask', 'rec', 'const', 'orc'):
                        v52[v123] += v121[v123]
            for v116, (v63, v57, v58) in v97['ingest'].v108():
                v117 = v97['base'].v86(v116, (0, 0, 0))
                if (v63 or v57) and (not (v117[0] or v117[1])):
                    for v52 in (v29['ingest'], v30['ingest'][v96]):
                        v52['new_const' if v58 else 'new_mixed'] += 1

    def row(v52):
        v75 = v92(1, v52['tok'])
        return {'positions': v52['tok'], 'askable': v52['ask'] / v75, 'reach': v52['hit'] / v75, 'recall': v52['rec'] / v75, 'oracle': v52['orc'] / v75, 'on_place': v52['rec_ask'] / v75, 'const': v52['const'] / v75, 'new_const': v52['new_const'], 'new_mixed': v52['new_mixed']}
    v32 = {'window_lines': v20.v76, 'doc_lines': v20.v77, 'docs': v20.v73, 'topm': v20.v8}
    for v33 in v28:
        v32[v33] = v98(v29[v33])
        v32[v33 + '_thirds'] = [v98(v52) for v52 in v30[v33]]
    v32['gain'] = v32['ingest']['reach'] - v32['base']['reach']
    v32['gain_over_null'] = v32['ingest']['reach'] - v32['null']['reach']
    v34 = v32['ingest_thirds'][2]['reach'] - v32['null_thirds'][2]['reach']
    v35 = v32['ingest_thirds'][0]['reach'] - v32['null_thirds'][0]['reach']
    v32['late_minus_early'] = v34 - v35
    v32['gain_over_null_oracle'] = v32['ingest']['oracle'] - v32['null']['oracle']
    v36 = v32['ingest_thirds'][2]['oracle'] - v32['null_thirds'][2]['oracle']
    v37 = v32['ingest_thirds'][0]['oracle'] - v32['null_thirds'][0]['oracle']
    v32['late_minus_early_oracle'] = v36 - v37
    v32['min_fillers'] = v20.v7
    v1.v99.v78(parents=True, exist_ok=True)
    v1.v79(v118.v100(v32, indent=1), encoding='utf-8')
    v80(f'old world {v20.v76} lines, {v20.v73} documents of {v20.v77} lines, topm {v20.v8}')
    for v33 in v28:
        v81 = v32[v33]
        v82 = v32[v33 + '_thirds']
        v80(f"{v33.v128():7s}  reach {v81['reach']:.4f}   askable {v81['askable']:.4f}   by thirds " + '  '.v124((f"{v119['reach']:.4f}" for v119 in v82)))
        v80(f"         recall {v81['recall']:.4f}  oracle {v81['oracle']:.4f}  on a place {v81['on_place']:.4f}  constant {v81['const']:.4f}   by thirds " + '  '.v124((f"{v119['oracle']:.4f}" for v119 in v82)))
    v80(f"GAIN     over base {v32['gain']:+.4f}   over null {v32['gain_over_null']:+.4f}   late-early {v32['late_minus_early']:+.4f}")
    v80(f"ORACLE   over null {v32['gain_over_null_oracle']:+.4f}   late-early {v32['late_minus_early_oracle']:+.4f}")
    v80(f"ADDED    by ingestion, at a constant place {v32['ingest']['new_const']}  at a mixed place {v32['ingest']['new_mixed']}")
    v38 = v32['gain_over_null'] > 0.05 and v34 > v35
    v39 = v32['gain_over_null_oracle'] > 0.05 and v36 > v37
    v32['gate_subst'], v32['gate_oracle'] = (v101(v38), v101(v39))
    if v38:
        v80("\nINGESTION IS REAL, AND THE OFFER CAN ASK ABOUT IT. A document's own earlier lines make its later holes reachable beyond the same mass of foreign text, and the gain grows down the document. This is the loop: not remembering answers - READING NEW TEXT INTO THE WORLD, where the reward for line k is that line k+1 becomes askable. Every closed single-shot result stands; the mind's next job is to choose what to ingest and when to trust it, and that is a decision.")
    elif v39:
        v80("\nINGESTION IS REAL AND THE OFFER CANNOT ASK ABOUT IT. The oracle column passes the same gate the substitution column fails, which is 8-RESULT-6's SECOND cause measured rather than argued: a constant place's truth is excluded from its own offer by construction, so what a document's own past creates is exactly what the exam is built not to ask. The lever is the QUESTION, not the reward - and no number here says anything about hopping on wiki-cloze.")
    else:
        v80("\nINGESTION BUYS NOTHING HERE, ON EITHER CHANNEL: a document's own past is worth no more than an equal mass of foreign text, by substitution and by recall alike. Self-reference either does not survive the frame filter or is already covered by the old world, and there is nowhere to step. Do not touch the reward.")
    if v32['ingest']['new_const'] and v32['ingest']['new_mixed']:
        v80(f"  Both terminals carry mass: {v32['ingest']['new_const']} constant and {v32['ingest']['new_mixed']} mixed positions were added by ingestion.")
    else:
        v80(f"  ONE TERMINAL ONLY: the split of what ingestion added is {v32['ingest']['new_const']} constant / {v32['ingest']['new_mixed']} mixed, so a design with two holes and two rewards has one of them empty.")
    v80(f'\nwritten to {v1}')
    return 0
if v40 == '__main__':
    raise v83(v102())