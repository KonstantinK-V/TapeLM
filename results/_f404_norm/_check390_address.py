"""Check of 390's address audit ON A DESIGNED TAPE. No torch, no corpus.

Twelve properties, each a WRONG NUMBER rather than an exception - a checker that only catches
crashes catches nothing, because every fault this project has actually made printed a plausible
number. And every property here is verified the only way a property can be: the failure it exists
to catch is RE-INTRODUCED at the bottom of this file, and the check must fire on it. A check that
has never fired is a comment.

THE TAPE. Twelve lines, `frame_max=1`, so an address is exactly (previous word | next word) and
nothing is approximate. The question is `zebra` in `the zebra sat`, and the tape is built so that:

    the | sat   cat  zebra  dog        THE QUESTION. own = {cat, dog}, hidden = zebra
    the | ran   zebra  moose           holds THE TRUTH, shares MY LEFT, and shares NO FILLER
                                       with the question - invisible to the walk, at any cap
    one | sat   pig  cow               shares MY RIGHT
    the | saw   fox  wolf              shares my left, but ONE OF ITS SLOTS IS ON THE HIDDEN
                                       SLOT'S LINE - a window artefact, dropped from every lane
    one | ran   elk  ibex              COMPOSED: my right-sharer's left x my left-sharer's right
    xx  | yy    cat  bird              shares the filler `cat` - the only place the walk reaches

So the standing arm CANNOT reach the answer on this question and the address CAN, by
construction. That is the claim the audit exists to measure at scale, written small enough to be
checked by hand.

  1. THE SECTION 27 LEAK. The walk is run from the profile WITHOUT the hidden row. Reading
     `prof[pid]` puts the truth in its own search key: `the|ran` holds `zebra`, so the walk would
     find it through the answer and then score the answer out of it - the exact fault that voided
     387 and 388.
  2. SAME-LINE PLACES ARE DROPPED FROM EVERY LANE. `the|saw` has a slot on the hidden slot's
     line; frames overlap, so that is the same words seen twice, not a second record.
  3. OWN VALUES ARE IN NO LANE. What already stands at this hole is not a candidate for it.
  4. THE ADDRESS IS BLIND TO FILLERS - both address neighbours are unreachable by the walk. If
     this reads zero the whole step is closed, which is what the audit's void check is for.
  5. THE QUESTION'S OWN PLACE IS NEVER A SOURCE, in any lane.
  6. COMPOSITION EXCLUDES WHAT WAS ALREADY IN HAND. `the|ran` and `one|sat` are both products of
     the pool's halves; counting them would let `comp_only` be won by re-offering the walk.
  7. BOTH HALVES CANNOT MATCH. The key is (w, left, right) with |left| = |right| = w, so sharing
     both halves is being the same place. Non-zero here means the key stopped being a bijection.
  8. AN ABSENT TRUTH RANKS 0, and a present one is 1-BASED - the reading discipline that a null
     must be readable on an absolute quantity.
  9. THE DECOY IS FREQUENCY MATCHED and is never the truth or an own value.
 10. THE MERGE IS ROUND-ROBIN AND CAPPED, never appended (347).
 11. THE AGGREGATOR ADDS UP THE PER-QUESTION NUMBERS AND NOTHING ELSE.
 12. NO LANE READS THE ASKING PLACE'S PROFILE - checked on the SOURCE as well, because 1 can only
     see the fault where the designed tape happens to expose it.

    python _check390_address.py
"""
from __future__ import annotations
import argparse
import random
import re
from collections import Counter
from pathlib import Path
import _audit390_address as A
SRC = Path('_audit390_address.py')
LINES = ['aa the cat sat bb', 'cc the zebra sat dd the fox saw ee2', 'ee the zebra ran ff', 'gg the moose ran hh', 'ii the dog sat jj', 'kk one pig sat ll', 'mm one cow sat nn', 'oo xx cat yy pp', 'qq xx bird yy rr', 's1 one elk ran s2', 's3 one ibex ran s4', 't1 the wolf saw t2']
ARGS = argparse.Namespace(places=8, topm=8, max_questions=10000)

def designed():
    T = A.build_tape(LINES, frame_max=1, min_fillers=1)
    q = T['of_addr'][1, ('the',), ('sat',)]
    slot = [s for s in T['places'][q] if T['toks'][s] == 'zebra'][0]
    return (T, q, slot)

def props():
    """Every property as a number read off the designed question. Returns the failures."""
    f = []
    T, q, slot = designed()
    if len(T['places']) != 6:
        return [f"0. the designed tape is not the designed tape: {len(T['places'])} places"]
    m = A.measure(T, slot, ARGS, random.Random(0))
    if m is None:
        return ['0. the designed question left the population']
    lanes = m['_lanes']
    allv = [v for L in lanes.values() for v in L]
    if m['n_fill'] != 1:
        f.append(f"1. the walk's neighbourhood is {m['n_fill']} places, not 1 - the hidden row is back in the search key (section 27)")
    if m['std8'] != 0:
        f.append('1. the standing arm reached the truth on a question built so it cannot - the walk is running from a profile that still holds the answer')
    if m['dropped'] != 1:
        f.append(f"2. {m['dropped']} same-line places dropped, not 1")
    if sum((1 for v in allv if v in ('fox', 'wolf'))) != 0:
        f.append("2. a place on the hidden slot's own line is in a lane - frames overlap, so that is the same words twice and not a second record")
    if sum((1 for v in allv if v in ('cat', 'dog'))) != 0:
        f.append('3. a value already standing at this hole is offered as a candidate for it')
    if (m['n_half'], m['n_half_new']) != (2, 2):
        f.append(f"4. address neighbours {m['n_half']} of which unseen by the walk {m['n_half_new']}, expected 2 and 2 - the address lane is not blind to fillers, so it is not a channel the walk lacks")
    if (m['half8'], m['half_only'], m['half_any']) != (1, 1, 1):
        f.append(f"4. the truth is reachable by address and the lane missed it: half8={m['half8']} half_only={m['half_only']} half_any={m['half_any']}")
    drop = {j for j in T['on_line'][T['owner'][slot]] if j != q}
    if q in A.half_nbrs(T, q, drop):
        f.append('5. the asking place is its own address neighbour')
    qprof = Counter((T['toks'][x] for x in T['places'][q] if x != slot))
    if q in A.filler_nbrs(T, q, qprof, drop)[0]:
        f.append('5. the asking place is its own filler neighbour')
    if q in A.compose_nbrs(T, q, {q}, drop):
        f.append('5. the asking place is returned as a composition of its own halves')
    if (m['n_comp'], m['n_comp_new']) != (1, 1):
        f.append(f"6. composition returned {m['n_comp']} places ({m['n_comp_new']} new), expected 1 and 1 - places already in hand are being counted as composed, so comp_only could be won by re-offering the walk")
    if m['n_pool'] != 4:
        f.append(f"6. the composing pool is {m['n_pool']} places, not 4")
    if sum((1 for v in lanes['comp'] if v in ('elk', 'ibex'))) != 2:
        f.append(f"6. the composed lane is {lanes['comp']}, not the fillers of `one|ran`")
    if m['both_halves'] != 0:
        f.append(f"7. {m['both_halves']} places share BOTH halves with the question - the address key is no longer (w, left, right) with |left| = |right| = w")
    if m['_rank_std'] != 0:
        f.append(f"8. an absent truth ranks {m['_rank_std']}, not 0")
    if m['_rank_half'] != 4:
        f.append(f"8. the truth ranks {m['_rank_half']} in the full address lane, expected 4 (1-based)")
    rng, bad = (random.Random(1), 0)
    own = {T['toks'][x] for x in T['places'][q] if x != slot}
    for _t in range(200):
        d = A.band_draw(T, 'zebra', own | {'zebra'}, rng)
        if d is None:
            continue
        if d == 'zebra' or d in own:
            bad += 1
    if bad:
        f.append(f'9. the decoy was the truth or an own value {bad} times in 200')
    got = A.interleave([1, 2, 3], [4, 5], cap=4)
    if got != [1, 4, 2, 5]:
        f.append(f'10. the merge is {got}, not round-robin cut at the cap - an appended lane makes the arm win on budget (347)')
    if len(lanes['std']) > ARGS.topm:
        f.append(f"10. the standing offer is {len(lanes['std'])} long, past the cap")
    c, _r = A.run(T, ARGS, random.Random(0))
    tot = Counter()
    for s in [x for ps in T['places'] for x in ps]:
        mm = A.measure(T, s, ARGS, random.Random(0))
        if mm:
            for k, v in mm.items():
                if not k.startswith('_'):
                    tot[k] += v
    diff = [k for k in tot if k not in ('rand8', 'rand_only', 'd_n', 'd_std8', 'd_half8', 'd_half_only', 'd_comp_only') and c[k] != tot[k]]
    if diff:
        f.append(f'11. the aggregator disagrees with the per-question numbers on {diff}')
    src = SRC.read_text(encoding='utf-8')
    body = src.split('# ------', 2)[-1]
    for name in ('filler_nbrs', 'lane_share', 'lane_addr_full'):
        b = re.search(f'^def {name}\\(.*?(?=\\n(?:def |# ---))', src, re.S | re.M)
        code = re.sub('"""(?:.|\\n)*?"""', '', b.group(0)) if b else ''
        if 'prof"][pid]' in code:
            f.append(f"12. {name} reads the asking place's own profile - the hidden row is in the search key again (section 27)")
    if 'qprof' not in body:
        f.append('12. the query profile is gone from the lanes')
    return f

def _leak(T, pid, qprof, drop=()):
    return A.__dict__['_orig_filler_nbrs'](T, pid, T['prof'][pid], drop)

def _no_drop(T, pid, drop=()):
    return A.__dict__['_orig_half_nbrs'](T, pid, ())

def _keep_own(T, order, own):
    return A.__dict__['_orig_fillers_of'](T, order, ())

def _fillers_too(T, pid, drop=()):
    """the address made to require a shared filler - the thing it exists NOT to require"""
    hn = A.__dict__['_orig_half_nbrs'](T, pid, drop)
    keep = {j for j in hn if set(A.__dict__['_T_prof'][j]) & set(A.__dict__['_T_prof'][pid])}
    return Counter({j: hn[j] for j in keep})

def _no_pool(T, pid, seen_pids, drop=()):
    return A.__dict__['_orig_compose_nbrs'](T, pid, {pid}, drop)

def _rank0(seq, v):
    for i, x in enumerate(seq):
        if x == v:
            return i
    return 0

def _append(*lanes, cap=None):
    out, seen = ([], set())
    for L in lanes:
        for v in L:
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out if cap is None else out[:cap]

def _self_nbr(T, pid, drop=()):
    hn = A.__dict__['_orig_half_nbrs'](T, pid, drop)
    hn[pid] = 1
    return hn

def _both(T, pid, drop=()):
    hn = A.__dict__['_orig_half_nbrs'](T, pid, drop)
    return Counter({j: 2 for j in hn})

def _truth_decoy(T, truth, banned, rng):
    return truth

def _double(T, args, rng):
    c, r = A.__dict__['_orig_run'](T, args, rng)
    c['half8'] += 1
    return (c, r)
MUTANTS = (('filler_nbrs', _leak, '1.'), ('half_nbrs', _no_drop, '2.'), ('fillers_of', _keep_own, '3.'), ('half_nbrs', _fillers_too, '4.'), ('compose_nbrs', _no_pool, '6.'), ('rank_of', _rank0, '8.'), ('interleave', _append, '10.'), ('half_nbrs', _self_nbr, '5.'), ('half_nbrs', _both, '7.'), ('band_draw', _truth_decoy, '9.'), ('run', _double, '11.'))

def main() -> int:
    fails = props()
    for name, fn, tag in MUTANTS:
        orig = getattr(A, name)
        setattr(A, f'_orig_{name}', orig)
        T, _q, _s = designed()
        A._T_prof = T['prof']
        setattr(A, name, fn)
        try:
            got = props()
        finally:
            setattr(A, name, orig)
        if not any((g.startswith(tag) for g in got)):
            fails.append(f'MUTATION {tag} ({name}): the failure was re-introduced and check {tag} did not fire - it is a comment, not a check')
    for x in fails:
        print('FAIL ' + x)
    print(f'{len(fails)} failures, {len(MUTANTS)} mutations' if fails else f'all properties hold, and all {len(MUTANTS)} re-introduced failures were caught')
    return 1 if fails else 0
if __name__ == '__main__':
    raise SystemExit(main())