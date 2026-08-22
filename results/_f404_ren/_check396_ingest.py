"""Check of 358's recall column - the one that decides whether `--min-fillers 1` is readable.

8-RESULT-6 named TWO reasons the write path deletes what ingestion creates:

  1. `min_fillers >= 2` deletes CONSTANT frames, which is exactly what self-reference makes.
  2. EVEN KEPT, THE EXAM CANNOT ASK ABOUT ONE - the lens is the place's other fillers and the
     offer excludes them (`w != v`), so a constant place's truth is outside its own offer by
     construction.

`--min-fillers 1` lifts the first and NOT the second. Re-running 358 with the old column alone
would therefore admit every place ingestion creates and score all of them misses, and the gate
would read "there is nowhere to step" when the truth is "the exam still cannot ask". The recall
column exists so those two are separable, and these are its properties. Each is a number on a
designed corpus, and each has its own failure re-introduced below.

  1. THE OLD COLUMN IS UNCHANGED. `hit` still skips a place with no lens, still bans the place's
     own values and still excludes the lens value. Every 358 number on record stays comparable.
  2. THE DENOMINATOR IS ABSOLUTE. No tape is `tok` with zeros, never a smaller denominator
     (the 342a lesson).
  3. A CONSTANT PLACE: unreachable by substitution, reachable by recall. This is 8-RESULT-6's
     second cause as a measurement instead of an argument.
  4. AT min_fillers 2 THE SAME PLACE DOES NOT EXIST. That is the first cause, and the two must
     be visible apart.
  5. RECALL EXCLUDES THIS POSITION. A truth standing only where it is hidden must not be
     recalled - otherwise the channel reads the answer.
  6. CONSTANT IS DEFINED ON THE PLACE WITH THIS POSITION REMOVED.
  7. THE SPLIT COUNTS WHAT INGESTION ADDED - reached under ingest and not under base.

    python _check396_ingest.py
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
import _audit358_ingest as A
v0 = v8('_audit358_ingest.py')
v1 = ['alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi', 'one two three four five six seven eight nine ten eleven twelve thirteen fourteen']
v2 = 'the XARWIN team won the opening match of the season in a long padded line of text'
v3 = 'the XARWIN team lost the closing match of the season in a long padded line here'
v4 = 'a CAT sat on the mat while the weather outside was cold and grey for many days'
v5 = 'a DOG sat on the mat while the weather outside was warm and dry for many days'

def call(v9, v10, v11):
    return v44.v20(v1 + v9 + [v10], v10, 1, v11, 8)

def props(v12=None):
    """`src` is the source the STATIC halves are read from - the mutants patch the module in
    memory, so a check that re-read the file from disk could never see one and would be a
    comment. The behavioural halves run against whatever `A` currently holds."""
    v13 = []
    v12 = v0.v34(encoding='utf-8') if v12 is None else v12
    v14 = v45.v21('"""(?:.|\\n)*?"""', '', v45.v60('^def reach_line\\(.*?(?=\\ndef )', v12, v45.v61 | v45.v62).v46(0))
    for v15 in ('if not lens:', 'cnt -= ban.get(w, 0)', 'if cnt > 0 and w != v:', 'off.most_common(topm)'):
        if v15 not in v14:
            v13.v47(f'1. the substitution column changed: {v15!r} is gone, so no 358 number on record is comparable any more')
    v22, v23 = v44.v20(['aa bb cc'], 'aa bb cc', 1, 1, 8)
    if v22['tok'] != 3 or v22['hit'] or v22['rec']:
        v13.v47(f'2. a line with no tape reports {v49(v22)}, not 3 positions and zeros')
    v24, v25 = v26([], v3, 1)
    v27, v25 = v26([v2], v3, 1)
    v28, v25 = v26([v2], v3, 2)
    if (v24['rec_ask'], v24['orc']) != (0, 0):
        v13.v47(f"3. without the document the question line is already on a place ({v24['rec_ask']} positions) - the designed case is not designed")
    if (v27['ask'], v27['hit']) != (10, 0):
        v13.v47(f"3. with the document ingested the substitution column reads ask={v27['ask']} hit={v27['hit']}, expected 10 and 0 - a constant place's truth is excluded from its own offer by construction, and if `hit` is non-zero here the offer stopped excluding the lens value")
    if (v27['rec'], v27['const'], v27['orc']) != (8, 8, 8):
        v13.v47(f"3. recall reads rec={v27['rec']} const={v27['const']} orc={v27['orc']}, expected 8/8/8 - the document's own past is what those places are made of")
    if (v28['rec_ask'], v28['orc']) != (2, 0):
        v13.v47(f"4. at min_fillers 2 the same document gives rec_ask={v28['rec_ask']} orc={v28['orc']}, expected 2 and 0 - the constant frames must be DELETED there, which is the first of 8-RESULT-6's two causes")
    v29, v30 = v26([v4], v5, 1)
    v16 = v30.v31(1)
    if v16 is None:
        v13.v47('5. the mixed position is not on a place - the designed case is not designed')
    else:
        v48, v32, v33 = v16
        if v32:
            v13.v47('5. recall fired at a position whose truth stands nowhere else - the channel is reading the answer')
        if v33:
            v13.v47('6. a place holding CAT and DOG counts as constant - `const` is not read with this position removed')
    if 'if (hit or rec) and not (b[0] or b[1]):' not in v12:
        v13.v47('7. the const/mixed split does not condition on the base arm, so it counts places ingestion did not add')
    return v13
v6 = (('the offer stops excluding the lens value', '                if cnt > 0 and w != v:', '                if cnt > 0:', '1.'), ('no tape shrinks the denominator', '        return Counter({"tok": n_tok}), {}', '        return Counter(), {}', '2.'), ('recall reads the hidden position too', '        rec = int(truth in {toks[x] for x in places[pid] if x != s})', '        rec = int(truth in {toks[x] for x in places[pid]})', '5.'), ('constant forgets that the others must be the truth', '        const = int(len(others) == 0 or (len(others) == 1 and rec))', '        const = int(len(others) <= 1)', '6.'), ('the split ignores the base arm', '                if (hit or rec) and not (b[0] or b[1]):', '                if hit or rec:', '7.'))

def main() -> v7:
    v12 = v0.v34(encoding='utf-8')
    v17 = v35()
    for v36, v37, v38, v39 in v6:
        if v12.v55(v37) != 1:
            v17.v47(f'MUTATION {v39} ({v36}): its anchor occurs {v12.v55(v37)} times')
            continue
        v40 = v49(v44.v50)
        v41 = v12.v51(v37, v38, 1)
        v52(v56(v41, '<mutant>', 'exec'), v44.v50)
        try:
            v53 = v35(src=v41)
        finally:
            v44.v50.v57()
            v44.v50.v58(v40)
        if not v59((v64.v63(v39) for v64 in v53)):
            v17.v47(f'MUTATION {v39} ({v36}): the failure was re-introduced and check {v39} did not fire - it is a comment, not a check')
    for v18 in v17:
        v42('FAIL ' + v18)
    v42(f'{v65(v17)} failures' if v17 else f'all properties hold, and all {v65(v6)} re-introduced failures were caught')
    return 1 if v17 else 0
if v19 == '__main__':
    raise v43(v54())