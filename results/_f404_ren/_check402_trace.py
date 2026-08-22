"""Check of 402's closed path, on a designed tape. No torch, no corpus.

The two faults this file exists to catch both fired on the first draft, and both are faults this
project has already paid for once:

  1. THE SECTION 27 LEAK, IN THE RETURN DIRECTION. The question's place still holds the HIDDEN
     token, so a walk from the candidate's place finds it THROUGH the answer we hid there. The
     path would close because of the hiding, not because the addresses see each other. One
     mention is subtracted from the target's overlap AND from its norm.
  2. 380'S ROOT. A walked candidate keeps THE PLACE IT WAS OFFERED FROM. "The first place holding
     this value" is exactly the bug 380 found in reach_deep, where an unrelated root cost three
     seeds of hit_of_deep.

  3. The offer is unchanged - own values excluded, cut at topm. 4. Same-line places are dropped in
  both directions. 5. The decoy is frequency-matched AND taken from the same offer. 6. The floor
  is a random place asked the same question. 7. The population is only questions whose offer
  already holds the truth: the trace reranks, it does not reach.

    python _check402_trace.py
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
import _audit390_address as A
import _audit402_trace as T2
v0 = v4('_audit402_trace.py')
v1 = ['aa the cat sat bb', 'cc the zebra sat dd the fox saw ee2', 'ee the zebra ran ff', 'gg the moose ran hh', 'ii the dog sat jj', 'kk one pig sat ll', 'mm one cow sat nn', 'oo xx cat yy pp', 'qq xx bird yy rr', 's1 one elk ran s2', 's3 one ibex ran s4', 't1 the wolf saw t2']

def designed():
    v5 = v29.v15(v1, frame_max=1, min_fillers=1)
    v6 = v5['of_addr'][1, ('the',), ('sat',)]
    v7 = v5['of_addr'][1, ('the',), ('ran',)]
    return (v5, v6, v7)

def props(v8=None):
    v8 = v0.v19(encoding='utf-8') if v8 is None else v8
    v9 = []
    v5, v6, v7 = v16()
    v10 = v30(v5['prof'][v6]) & v30(v5['prof'][v7])
    if v10 != {'zebra'}:
        v9.v31(f'0. the designed tape shares {v10}, not just the hidden token')
    v11 = v32.v17(v5, v6, v7, 'zebra', 8, ())
    if v11:
        v9.v31("1. the path closes on a place whose only link is the HIDDEN token - the return walk is finding the question's place through the answer we hid there")
    if v32.v17(v5, v6, v7, '\x00nothing', 8, ()) is not True:
        v9.v31('1. without the subtraction the designed case does not close either, so the check cannot tell the leak from an empty tape')
    if '- (1 if (j == pid and v == hidden) else 0)' not in v8:
        v9.v31("1. the hidden mention is not subtracted from the target's overlap")
    if '2 * m - 1' not in v8:
        v9.v31("1. the hidden mention is not subtracted from the target's NORM, so the target is scored against a length it no longer has")
    if 'root[v] = j' not in v8 or 'root.get(v)' not in v8:
        v9.v31('2. a candidate does not keep the walked place it was offered from (380)')
    if v33.v18('js\\[0\\]', v8):
        v9.v31("2. the root is 'the first place holding this value' - 380's bug, restored")
    if 'cands = cands[:args.topm]' not in v8 or 'root, cands, seen = {}, [], set(own)' not in v8:
        v9.v31("3. the offer is not the arm's - own values excluded, cut at topm")
    if 'if truth not in cands:' not in v8:
        v9.v31("7. questions whose offer does not hold the truth are not skipped - the trace would be credited with reaching, which is 347's operation")
    if 'drop = set(T["on_line"][owner[s]])' not in v8 or 'args.places, drop)' not in v8:
        v9.v31('4. same-line places are not dropped in both directions')
    if 'A.band_draw(' not in v8 or 'and d in cands' not in v8:
        v9.v31('5. the decoy is not frequency-matched or is not taken from the same offer')
    if 'rng.randrange(len(T["places"]))' not in v8:
        v9.v31('6. the floor is not a random place of the same tape')
    return v9
v2 = (("the hidden mention stays in the target's overlap", '            pj = T["prof"][j][v] - (1 if (j == pid and v == hidden) else 0)', '            pj = T["prof"][j][v]', '1.'), ("the target's norm keeps the hidden mention", '        ss = sum(c * c for c in T["prof"][j].values()) - (2 * m - 1 if m > 0 else 0)', '        ss = sum(c * c for c in T["prof"][j].values())', '1.'), ("380's root is replaced by the first place holding the value", '                    root[v] = j', '                    root[v] = [x for x in T["at_value"].get(v, ()) if x != pid][0]', '2.'), ('the offer is not cut', '        cands = cands[:args.topm]', '        cands = cands[:]', '3.'), ('questions the offer misses are counted too', '        if truth not in cands:\n            continue', '        if False:\n            continue', '7.'))

def main() -> v3:
    v8 = v0.v19(encoding='utf-8')
    v12 = v20()
    for v21, v22, v23, v24 in v2:
        if v8.v39(v22) != 1:
            v12.v31(f'MUTATION {v24} ({v21}): its anchor occurs {v8.v39(v22)} times')
            continue
        v25 = v34(v32.v35)
        v26 = v8.v36(v22, v23, 1)
        try:
            v40(v44(v26, '<mutant>', 'exec'), v32.v35)
            v11 = v20(src=v26)
        except v37 as e:
            v11 = [f'{v24} the mutant raised {v48(v49).v14}']
        finally:
            v32.v35.v41()
            v32.v35.v42(v25)
        if not v43((v46.v45(v24) for v46 in v11)):
            v12.v31(f'MUTATION {v24} ({v21}): the failure was re-introduced and check {v24} did not fire - it is a comment, not a check')
    for v13 in v12:
        v27('FAIL ' + v13)
    v27(f'{v47(v12)} failures' if v12 else f'all properties hold, and all {v47(v2)} re-introduced failures were caught')
    return 1 if v12 else 0
if v14 == '__main__':
    raise v28(v38())