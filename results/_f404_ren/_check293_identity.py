"""Prove 293 before spending an hour on it. Seconds, no corpus and no model.

293 moves the mind to where fp_addresses' threshold currently stands: the verb is no longer
"what value goes in this slot" but "do these mentions name the same place". That is the one
decision in this project still made by a hand-set rule, and putting Phi there is worth nothing
if the question can be answered without understanding anything. Seven things have to hold, and
each is a way the run could print a plausible number that means nothing:

  1 THE LABEL IS NOT CIRCULAR AND NOT TRIVIAL. The truth is the (anchor, value) pair - two
    strings the encoder never touches - and every candidate, the truth included, has relation
    words DISJOINT from the core's. A label taken from fp_addresses' grouping would make the
    rival the labeller; a label taken from string-address identity would be free to the shared
    word channel, because the same string address means the same relation words.
  2 THE LABEL CANNOT BE SEEN. The value is what the label is made of and it is excluded from
    every channel: hidden as a sentinel, so the same-value edge is exactly zero in every world,
    and already excluded from the ink and the word sets by ctx_fp/context_words. --ident-values
    show is the sanity bolt in the other direction: with values visible the label IS an input.
  3 EVERY WORLD CARRIES THE SAME NUMBER OF ROWS, and they differ in exactly one row. Unequal
    counts are the bookkeeping tell that made 289's ladder unreadable and 291's refusal unfair.
  4 THE CANDIDATE ROW IS MARKED, IDENTICALLY IN ALL FOUR WORLDS. A proposed member may not be
    mistaken for an observed one, and the mark may not carry the label.
  5 POSITION SAYS NOTHING. 292's rungs were built by relatedness and Phi learned to read the
    construction rather than the fit - an inverted landscape on three seeds out of three. Here
    the four candidates are shuffled, so the label's position is uniform.
  6 THE QUESTION IS DETERMINISTIC given its rng, or every evaluation is a different question.
  7 THE HEURISTIC RIVAL IS THE WRITING RULE, not a description of it: at an impossible tau it
    declines to link and answers nothing, at a permissive one it links and answers.

    python _check293_identity.py
"""
from __future__ import annotations
import random
from collections import Counter
import torch
import _stage289_derivation as s289

class FakeTape:

    def __init__(v35, v36):
        v35.v36 = v36

def fake_pack():
    """One anchor. Kaluga is written twice in words that share nothing - one place said two ways,
    which is the fragmentation case - and four same-anchor mentions carry other values."""
    v1 = ['kostya was born in kaluga in the spring of that year', 'kostya reportedly hails from kaluga according to the parish register', 'kostya played for spartak during the winter season', 'kostya died in moscow after a long illness', 'kostya captained the reserve side for two seasons', 'kostya studied at gorky university before the war']
    v2 = ['Kaluga', 'Kaluga', 'Spartak', 'Moscow', 'Reserve', 'Gorky']
    v3 = ['kostya|born in', 'kostya|hails from', 'kostya|played for', 'kostya|died in', 'kostya|captained', 'kostya|studied at']
    v4 = [{'S': 'kostya', 'address': 'fp0:kostya|born in', 'slots': [0, 1], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp1:kostya|played for', 'slots': [2], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp2:kostya|died in', 'slots': [3], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp3:kostya|captained', 'slots': [4], 'kind': 'clean'}, {'S': 'kostya', 'address': 'fp4:kostya|studied at', 'slots': [5], 'kind': 'clean'}]
    v5 = {}
    for v37, v38 in v39(v1):
        for v40 in v38.v74():
            v5.v108(v40, []).v89(v37)
    v6 = v41(v1)
    v7 = v79.v90().v42(293)
    v8 = v79.v91.v75.v43(v79.v76(v6, 16, generator=v7), dim=-1)
    v9 = v79.v91.v75.v43(v79.v76(v6, 16, generator=v7), dim=-1)
    return {'tape': v77(v2), 'texts': v1, 'texts_lc': [v38.v92() for v38 in v1], 'items': v4, 'postings': v5, 'n_slots': v6, 'straddr': v3, 'slot_keys_slot': v78(v66(v6)), 'ctx_keys': v8, 'anc_keys': v9}

class FakeBank:

    def ctx_fp(v35, v44, v45=None):
        v7 = v79.v90().v42(v100(v109(v44)) % 2 ** 31)
        return v79.v91.v75.v43(v79.v76(16, generator=v7), dim=-1)

def main() -> v0:
    v10 = True
    v11 = v79.v46('cpu')
    v52.v47, v52.v48, v52.v49 = (True, 4, 3)
    v52.v21, v52.v50, v52.v51 = ('hide', 0.9, 2)
    v52.v12 = v53(v52.v54)
    v52.v55, v52.v56, v52.v57 = (0, False, 0)
    v58, v59 = (v80(), v81())
    v52.v82.v60()
    v13 = [v14 for v14 in v52.v93(v58, v99.v87(0)) if v14.v94('ident')]
    v61(f'0 questions built: {v41(v13)}  supply {v101(v52.v82)}')
    v10 &= v41(v13) == 2
    v14 = v13[0]
    v15 = v14['cand_slots'][v14['label']]
    v16 = v53()
    for v17 in v14['slots']:
        v16 |= v52.v95(v58['straddr'][v17])[1]
    v18 = [v17 for v17 in v14['cand_slots'] if v58['tape'].v36[v17] == v14['place'][1]]
    v19 = v18 == [v15] and v63((v52.v95(v58['straddr'][v17])[0] == v14['S'] for v17 in v14['cand_slots'])) and v63((not v52.v95(v58['straddr'][v17])[1] & v16 for v17 in v14['cand_slots'])) and (v58['straddr'][v15] not in {v58['straddr'][v17] for v17 in v14['slots']}) and (v15 not in v14['slots'])
    v10 &= v62(v19)
    v61(f"1 place {v14['place']}, core {v14['slots']} rel {v102(v16)}, cands {v14['cand_slots']} truth {v15}")
    v61(f'  value identifies the truth uniquely, no candidate shares a relation word, the truth is phrased differently: {v62(v19)}')
    v20 = [v52.v64(v58, v52.v83(v58, v14, v17), v59, v11, query_value=None, import_k=0) for v17 in v14['cand_slots']]
    v19 = v63((v103(v104.v100().v98()) == 0.0 for v97, v104, v97 in v20))
    v10 &= v19
    v61(f'2 same-value edge is exactly zero in every world (values hidden): {v19}')
    v52.v21 = 'show'
    v22 = v52.v64(v58, v52.v83(v58, v14, v14['cand_slots'][0]), v59, v11, query_value=None, import_k=0)
    v52.v21 = 'hide'
    v61(f'  --ident-values show restores them (nonzero same edges: {v0((v22[1] != 0).v98())}), so the shortcut stays measurable')
    v23 = {v84(v96.v85) for v97, v97, v96 in v20}
    v19 = v41(v23) == 1 and v41(v14['slots']) + 1 == v20[0][2].v85[0]
    v10 &= v19
    v61(f'3 every world has the same shape {v23}, core+1 rows: {v19}')
    v24 = [v96[:, 4].v86() for v97, v97, v96 in v20]
    v19 = v63((v105 == v24[0] for v105 in v24)) and v24[0][-1] == 1.0 and (v98(v24[0]) == 1.0)
    v10 &= v19
    v61(f'4 the candidate row is the query row in all four, identically {v24[0]}: {v19}')
    v25 = v65()
    for v26 in v66(400):
        v67 = v52.v68(v58, 'kostya', 'Kaluga', 0, v99.v87(v26))
        v25[v67['label']] += 1
    v19 = v41(v25) == 4 and v106(v25.v36()) - v107(v25.v36()) < 80
    v10 &= v19
    v61(f'5 label position over 400 draws {v101(v102(v25.v4()))}: {v19}')
    v27 = v52.v68(v58, 'kostya', 'Kaluga', 0, v99.v87(11))
    v28 = v52.v68(v58, 'kostya', 'Kaluga', 0, v99.v87(11))
    v19 = v27['cand_slots'] == v28['cand_slots'] and v27['label'] == v28['label'] and (v27['slots'] == v28['slots'])
    v10 &= v19
    v61(f'6 same rng, same question: {v19}')
    v52.v50, v52.v51 = (2.0, 2)
    v29 = v52.v69(v58, v14)
    v52.v50, v52.v51 = (-1.0, 0)
    v30 = v52.v69(v58, v14)
    v52.v50, v52.v51 = (0.9, 2)
    v19 = v29['heur'] is None and v30['heur'] is not None and (v30['_heur_accepted'] == 4)
    v10 &= v19
    v61(f"7 tau 2.0 -> declines to link ({v29['heur']}), tau -1 -> links all four ({v30['_heur_accepted']}): {v19}")
    v61(f"  cos1nn {v30['cos1nn']}  rare {v30['rare']}  truth s{v15}")
    v52.v31 = 2
    v14.v70('_ibudget', None)
    v28 = v52.v71(v58, v14)
    v32 = {v52.v64(v58, v52.v83(v58, v14, v17), v59, v11, query_value=None, import_k=0)[2].v85[0] for v17 in v14['cand_slots']}
    v52.v31 = 0
    v19 = v28 == 0 and v41(v32) == 1
    v10 &= v19
    v61(f'8 --ident-import 2 on a tape with no siblings: budget {v28}, world sizes {v32}: {v19}')
    v33 = v52.v72(v58, v99.v87(1))
    v61(f'\naudit on the toy tape: {v33}')
    v61('\nIDENT OK' if v10 else '\nIDENT FAILED')
    return 0 if v10 else 1
if v34 == '__main__':
    raise v73(v88())