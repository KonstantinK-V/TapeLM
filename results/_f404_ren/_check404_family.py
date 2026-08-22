"""Check of 404: the fitted family, the vocabulary channel, and the ablation that decides both.

404 is the first step here that FITS, so the ways it can lie are the ways fitting lies:

  1. HELD OUT BY FILE. The train and test file sets must be disjoint - a fit scored on a file it
     saw is a memory, not a result.
  2. THE VOID CHECK IS COUNTED WITHOUT THE CAP. Read off two capped collections it printed
     1.0000, which is a property of the cap and not of the arena - and it is the number that is
     supposed to be read first.
  3. CHANNEL A READS NO IDENTIFIER. `feats_A` never touches the ids field, and the corpus run
     confirms it: unparsed and unparsed-renamed give the same number to the digit.
  4. THE VOCABULARY ADMITS ONLY CROSS-FILE WORDS. A word must occur in at least k DIFFERENT
     files - that is what makes it language rather than one tape's content, and it is the whole
     justification for letting any identifier in at all.
  5. THE POPULATION IS THE DECISION ONE: two or more safe candidates, and the true next line must
     itself be legal, or the state has no answer in it.
  6. THE ABLATION USES THE SAME HELD-OUT STATES as the family. A best-single measured elsewhere
     could not settle whether this is a joint.
  7. EXPECTED ACCURACY under uniform tie-breaking, so a model that ties at the top is priced at
     1/|argmax| rather than by a coin.

    python _check404_family.py
"""
from __future__ import annotations
import re
from pathlib import Path
import _audit404_family as A
v0 = v4('_audit404_family.py')
v1 = '\ndef f(arg):\n    made = arg + 1\n    other = arg + 2\n    third = arg + 3\n    used = made + other + third\n    return used\n'

def props(v5=None):
    v5 = v0.v18(encoding='utf-8') if v5 is None else v5
    v6 = []
    v7 = v28.v15(v1)
    if v39(v7) != 1 or v39(v7[0]) != 5:
        return [f'0. the designed body is {v7 and v39(v7[0])} lines, not 5']
    v8 = v7[0]
    v9 = v16(v28.v29(v8, v40(), 8, False, 2))
    if not v9:
        v6.v30('5. the designed state is not in the population - two safe candidates with the true next among them is exactly what this arena is')
    else:
        v31, v32 = v9[0]
        if v32 != 0:
            v6.v30(f'5. the target index is {v32}, not 0 - the candidates are listed in true order, so the true next is the first of them')
        if v39(v31) < 2:
            v6.v30('5. a state with fewer than two safe candidates entered the population')
    if 'if len(safe_ix) < min_safe or 0 not in safe_ix:' not in v5:
        v6.v30('5. states whose true next line is itself illegal are not skipped')
    v10 = v47.v41('^def feats_A\\(.*?(?=\\ndef )', v5, v47.v48 | v47.v49).v17(0)
    if '[4]' in v10 or 'ids' in v10.v35('_ids', ''):
        v6.v30('3. channel A reads the identifiers of the line')
    if '_l, s, d, ty, _ids = cand' not in v10:
        v6.v30("3. channel A's unpacking changed - the ids field must stay unread")
    v11 = v47.v41('^def build_vocab\\(.*?(?=\\ndef )', v5, v47.v48 | v47.v49).v17(0)
    if 'seen[w] += 1' not in v11 or 'n >= k' not in v11:
        v6.v30("4. the vocabulary is not 'a word seen in at least k different files'")
    if 'ws = set()' not in v11:
        v6.v30("4. a word repeated inside ONE file counts more than once, so one tape's own content can enter the vocabulary")
    if 'train_f, test_f = files[:half], files[half:]' not in v5:
        v6.v30('1. the split is not by file')
    if 'te1 = collect(test_f,' not in v5:
        v6.v30('6. the ablation is not read on the held-out states')
    if 'for xs, _y in states_of(body, vocab, args.vocab_dim, False, 1):' not in v5:
        v6.v30('2. the void check is read off a capped collection again')
    if 'hit += (1.0 / len(best)) if y in best else 0.0' not in v5:
        v6.v30('7. accuracy is not 1/|argmax| under a tie')
    return v6
v2 = (('the fit is scored on the files it saw', '    train_f, test_f = files[:half], files[half:]', '    train_f, test_f = files[:half], files[:half]', '1.'), ('the void check goes back to the capped collections', '            for xs, _y in states_of(body, vocab, args.vocab_dim, False, 1):', '            for xs, _y in states_of(body, vocab, args.vocab_dim, False, 9):', '2.'), ('channel A is handed the identifiers', '    _l, s, d, ty, _ids = cand', '    _l, s, d, ty, ids4 = cand', '3.'), ('a word repeated in one file enters the vocabulary', '        ws = set()', '        ws = []', '4.'), ('states whose true next is illegal are kept', '        if len(safe_ix) < min_safe or 0 not in safe_ix:', '        if len(safe_ix) < min_safe:', '5.'), ('the ablation is read on the training states', '    te1 = collect(test_f,', '    te1 = collect(train_f,', '6.'))

def main() -> v3:
    v5 = v0.v18(encoding='utf-8')
    v12 = v19()
    for v20, v21, v22, v23 in v2:
        if v5.v42(v21) != 1:
            v12.v30(f'MUTATION {v23} ({v20}): its anchor occurs {v5.v42(v21)} times')
            continue
        v24 = v33(v28.v34)
        v25 = v5.v35(v21, v22, 1)
        try:
            v43(v50(v25, '<mutant>', 'exec'), v28.v34)
            v36 = v19(src=v25)
        except v37 as e:
            v36 = [f'{v23} the mutant raised {v53(v54).v14}']
        finally:
            v28.v34.v44()
            v28.v34.v45(v24)
        if not v46((v52.v51(v23) for v52 in v36)):
            v12.v30(f'MUTATION {v23} ({v20}): the failure was re-introduced and check {v23} did not fire - it is a comment, not a check')
    for v13 in v12:
        v26('FAIL ' + v13)
    v26(f'{v39(v12)} failures' if v12 else f'all properties hold, and all {v39(v2)} re-introduced failures were caught')
    return 1 if v12 else 0
if v14 == '__main__':
    raise v27(v38())