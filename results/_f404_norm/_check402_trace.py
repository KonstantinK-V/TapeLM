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
SRC = Path('_audit402_trace.py')
LINES = ['aa the cat sat bb', 'cc the zebra sat dd the fox saw ee2', 'ee the zebra ran ff', 'gg the moose ran hh', 'ii the dog sat jj', 'kk one pig sat ll', 'mm one cow sat nn', 'oo xx cat yy pp', 'qq xx bird yy rr', 's1 one elk ran s2', 's3 one ibex ran s4', 't1 the wolf saw t2']

def designed():
    T = A.build_tape(LINES, frame_max=1, min_fillers=1)
    q = T['of_addr'][1, ('the',), ('sat',)]
    other = T['of_addr'][1, ('the',), ('ran',)]
    return (T, q, other)

def props(src=None):
    src = SRC.read_text(encoding='utf-8') if src is None else src
    f = []
    T, q, other = designed()
    shared = set(T['prof'][q]) & set(T['prof'][other])
    if shared != {'zebra'}:
        f.append(f'0. the designed tape shares {shared}, not just the hidden token')
    got = T2.closes(T, q, other, 'zebra', 8, ())
    if got:
        f.append("1. the path closes on a place whose only link is the HIDDEN token - the return walk is finding the question's place through the answer we hid there")
    if T2.closes(T, q, other, '\x00nothing', 8, ()) is not True:
        f.append('1. without the subtraction the designed case does not close either, so the check cannot tell the leak from an empty tape')
    if '- (1 if (j == pid and v == hidden) else 0)' not in src:
        f.append("1. the hidden mention is not subtracted from the target's overlap")
    if '2 * m - 1' not in src:
        f.append("1. the hidden mention is not subtracted from the target's NORM, so the target is scored against a length it no longer has")
    if 'root[v] = j' not in src or 'root.get(v)' not in src:
        f.append('2. a candidate does not keep the walked place it was offered from (380)')
    if re.search('js\\[0\\]', src):
        f.append("2. the root is 'the first place holding this value' - 380's bug, restored")
    if 'cands = cands[:args.topm]' not in src or 'root, cands, seen = {}, [], set(own)' not in src:
        f.append("3. the offer is not the arm's - own values excluded, cut at topm")
    if 'if truth not in cands:' not in src:
        f.append("7. questions whose offer does not hold the truth are not skipped - the trace would be credited with reaching, which is 347's operation")
    if 'drop = set(T["on_line"][owner[s]])' not in src or 'args.places, drop)' not in src:
        f.append('4. same-line places are not dropped in both directions')
    if 'A.band_draw(' not in src or 'and d in cands' not in src:
        f.append('5. the decoy is not frequency-matched or is not taken from the same offer')
    if 'rng.randrange(len(T["places"]))' not in src:
        f.append('6. the floor is not a random place of the same tape')
    return f
MUTANTS = (("the hidden mention stays in the target's overlap", '            pj = T["prof"][j][v] - (1 if (j == pid and v == hidden) else 0)', '            pj = T["prof"][j][v]', '1.'), ("the target's norm keeps the hidden mention", '        ss = sum(c * c for c in T["prof"][j].values()) - (2 * m - 1 if m > 0 else 0)', '        ss = sum(c * c for c in T["prof"][j].values())', '1.'), ("380's root is replaced by the first place holding the value", '                    root[v] = j', '                    root[v] = [x for x in T["at_value"].get(v, ()) if x != pid][0]', '2.'), ('the offer is not cut', '        cands = cands[:args.topm]', '        cands = cands[:]', '3.'), ('questions the offer misses are counted too', '        if truth not in cands:\n            continue', '        if False:\n            continue', '7.'))

def main() -> int:
    src = SRC.read_text(encoding='utf-8')
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f'MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times')
            continue
        saved = dict(T2.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, '<mutant>', 'exec'), T2.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f'{tag} the mutant raised {type(e).__name__}']
        finally:
            T2.__dict__.clear()
            T2.__dict__.update(saved)
        if not any((g.startswith(tag) for g in got)):
            fails.append(f'MUTATION {tag} ({name}): the failure was re-introduced and check {tag} did not fire - it is a comment, not a check')
    for x in fails:
        print('FAIL ' + x)
    print(f'{len(fails)} failures' if fails else f'all properties hold, and all {len(MUTANTS)} re-introduced failures were caught')
    return 1 if fails else 0
if __name__ == '__main__':
    raise SystemExit(main())