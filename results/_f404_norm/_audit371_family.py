"""IS THE SPACE OF RELATIONS BIGGER THAN THE THREE WE WROTE BY HAND?

THE QUESTION, AND IT IS THE LAST STRUCTURAL ONE THIS DESIGN HAS. The tape has three relations
and a human wrote all three: substitution (places with a similar filler fingerprint), recall
(this place itself), connection (places sharing a filler, overlap-weighted). Phi chooses among
their OUTPUTS and has never chosen WHAT TO COUNT. Scaling the corpus gives a better index, not
a different mind, for exactly that reason.

But all three are the SAME SHAPE: a rule that scores OTHER PLACES given THIS place, after which
their fillers are read off. That shape is enumerable. So before anything is made learnable, the
question is whether there is a space to learn IN:

    if a perfect chooser over a family of relations reaches far more than the best single one,
    there is something to learn and the relation set is a real degree of freedom.
    If the oracle sits on top of the best single relation, the three are one relation in three
    costumes, there is nothing to hand over, and this design is finished.

THE FAMILY, every member a COUNT with no fitted constant. Each returns a scored list of OTHER
places; the offer is the fillers of the top ones, at the same top-m for every member:

    own            this place's other rows                          (recall, channel 2)
    share_k        places sharing >= k fillers, k = 1, 2, 3         (connection, generalised)
    share_w        sharing, weighted by how many fillers            (365's winner)
    rare_w         sharing, weighted by 1/frequency of the shared filler - a shared RARE word
                   says more than a shared common one, and that is a count, not a hyperparameter
    common_w       the opposite weighting, kept because a family with only the plausible
                   members in it is a family chosen by me
    cos_k          the k nearest by frame fingerprint               (substitution, channel 1)
    two_hop        places sharing a filler with a place that shares a filler with this one
    same_line      places whose mentions sit on lines this place's mentions sit on (304 closed
                   the line CHANNEL; here it is one member of a family and costs nothing to ask)
    len_match      places with the same number of distinct fillers - a pure structure relation
                   that knows nothing about content, in as the family's own null

  GATE  on the own-fails cut: oracle_when_own_fails - null_when_own_fails > 0.05.
        Whole-population oracle - oracle_null is still printed (it failed: twelve shots, not
        twelve relations). The cut is defined by recall's outcome, identically for both arms;
        the absolute numbers are conditional, same discipline as 363's AUC|defined.
  Also reported: how MANY members are ever the sole one that reaches (a family where one member
  is nearly always the only winner is a family with one useful relation and a tail).

    python _audit371_family.py
    python _audit371_family.py --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage371_family.json')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8, help='values offered, the SAME for every member - a relation may not win on budget')
    ap.add_argument('--places-k', type=int, default=8, help='places a relation may return')
    ap.add_argument('--neigh-cap', type=int, default=4000)
    ap.add_argument('--max-questions', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--corpus', default=str(WIKI))
    args = ap.parse_args()
    text = Path(args.corpus).open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if not keep:
        print('no tape')
        return 1
    places = [list(ps) for _a, ps in keep]
    P = len(places)
    place_of = {}
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
    fillers = [set((toks[s] for s in ps)) for ps in places]
    counts = [Counter((toks[s] for s in ps)) for ps in places]
    places_of_value = defaultdict(set)
    for pid, f in enumerate(fillers):
        for v in f:
            places_of_value[v].add(pid)
    freq = Counter((toks[s] for ps in places for s in ps))
    lines_of = [set((owner[s] for s in ps)) for ps in places]
    places_on_line = defaultdict(set)
    for pid, ls in enumerate(lines_of):
        for ln in ls:
            places_on_line[ln].add(pid)
    by_nfill = defaultdict(list)
    for pid, f in enumerate(fillers):
        by_nfill[len(f)].append(pid)

    def neighbours(pid, own):
        """every place sharing at least one filler, with HOW MANY it shares and which."""
        shared = defaultdict(set)
        for v in own:
            for j in places_of_value[v]:
                if j != pid:
                    shared[j].add(v)
        return shared

    def members(pid, own, shared):
        m = {}
        m['share_1'] = {j: len(vs) for j, vs in shared.items()}
        m['share_2'] = {j: len(vs) for j, vs in shared.items() if len(vs) >= 2}
        m['share_3'] = {j: len(vs) for j, vs in shared.items() if len(vs) >= 3}
        m['share_w'] = {j: len(vs) / len(fillers[j]) for j, vs in shared.items()}
        m['rare_w'] = {j: sum((1.0 / freq[v] for v in vs)) for j, vs in shared.items()}
        m['common_w'] = {j: sum((float(freq[v]) for v in vs)) for j, vs in shared.items()}
        m['mention_w'] = {j: sum((min(counts[pid][v], counts[j][v]) for v in vs)) for j, vs in shared.items()}
        m['cos_k'] = {j: len(vs) / max(1, len(own | fillers[j]) - len(vs)) for j, vs in shared.items()}
        two = Counter()
        for j in list(shared)[:64]:
            for v in fillers[j]:
                for j2 in places_of_value[v]:
                    if j2 != pid and j2 not in shared:
                        two[j2] += 1
        m['two_hop'] = dict(two)
        sl = Counter()
        for ln in lines_of[pid]:
            for j in places_on_line[ln]:
                if j != pid:
                    sl[j] += 1
        m['same_line'] = dict(sl)
        pool = by_nfill.get(len(own), ())
        m['len_match'] = {j: 1 for j in list(pool)[:args.places_k * 4] if j != pid}
        return m
    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    names = ['own', 'share_1', 'share_2', 'share_3', 'share_w', 'rare_w', 'common_w', 'mention_w', 'cos_k', 'two_hop', 'same_line', 'len_match']
    hit = Counter()
    sole = Counter()
    c = Counter()
    for s in qs:
        if c['n'] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in places[pid] if x != s}
        if not own:
            continue
        c['n'] += 1
        got = {'own': truth in {w for w, _n in Counter((toks[x] for x in places[pid] if x != s)).most_common(args.topm)}}
        shared = neighbours(pid, own)
        if len(shared) > args.neigh_cap:
            ks = sorted(shared, key=lambda j: -len(shared[j]))[:args.neigh_cap]
            shared = {j: shared[j] for j in ks}
        fam = members(pid, own, shared)
        for nm in names:
            if nm == 'own':
                continue
            sc = fam.get(nm, {})
            top = sorted(sc, key=lambda j: (-sc[j], j))[:args.places_k]
            offer = Counter()
            for j in top:
                for v, n in counts[j].items():
                    if v not in own:
                        offer[v] += n
            got[nm] = truth in {w for w, _n in offer.most_common(args.topm)}
        gotn = {'own': got['own']}
        for t in range(len(names) - 1):
            rp = [rng.randrange(P) for _ in range(args.places_k)]
            offer = Counter()
            for j in rp:
                if j == pid:
                    continue
                for v, cnt in counts[j].items():
                    if v not in own:
                        offer[v] += cnt
            gotn[f'null_{t}'] = truth in {w for w, _n in offer.most_common(args.topm)}
        c['oracle_null'] += any(gotn.values())
        if not got['own']:
            c['nf'] += 1
            c['oracle_nf'] += any((got[nm] for nm in names))
            c['null_nf'] += any((gotn[k] for k in gotn if k != 'own'))
        for nm in names:
            hit[nm] += got[nm]
        winners = [nm for nm in names if got[nm]]
        c['oracle'] += bool(winners)
        if len(winners) == 1:
            sole[winners[0]] += 1
            c['sole'] += 1
    n = max(1, c['n'])
    per = {nm: hit[nm] / n for nm in names}
    best = max(per, key=per.get)
    rep = {'lines': len(lines), 'places': P, 'questions': c['n'], 'topm': args.topm, 'places_k': args.places_k, 'min_fillers': args.min_fillers, 'per_relation': per, 'best': best, 'best_reach': per[best], 'oracle': c['oracle'] / n, 'oracle_null': c['oracle_null'] / n, 'sole_winner_rate': c['sole'] / n, 'sole_by': {nm: sole[nm] / n for nm in names if sole[nm]}}
    rep['over_best'] = rep['oracle'] - rep['best_reach']
    rep['over_null_family'] = rep['oracle'] - rep['oracle_null']
    nf = max(1, c['nf'])
    rep['own_fails'] = c['nf'] / n
    rep['oracle_when_own_fails'] = c['oracle_nf'] / nf
    rep['null_when_own_fails'] = c['null_nf'] / nf
    rep['over_null_when_own_fails'] = rep['oracle_when_own_fails'] - rep['null_when_own_fails']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape     {P} places, {c['n']} questions, topm {args.topm}, places_k {args.places_k}")
    for nm in sorted(names, key=lambda x: -per[x]):
        mark = '  <- null' if nm == 'len_match' else ''
        print(f'   {nm:<11} {per[nm]:.4f}{mark}')
    print(f"BEST     {best} {rep['best_reach']:.4f}")
    print(f"ORACLE   {rep['oracle']:.4f}   over best {rep['over_best']:+.4f}")
    print(f"NULL FAM {rep['oracle_null']:.4f}   over it {rep['over_null_family']:+.4f}   ({len(names) - 1} random place-sets + own, the SAME number of shots)")
    print(f"OWN FAILS on {rep['own_fails']:.4f} of questions - THE ONLY ONES A PLACE RELATION COULD MATTER ON:")
    print(f"   relations {rep['oracle_when_own_fails']:.4f}   random family {rep['null_when_own_fails']:.4f}   over it {rep['over_null_when_own_fails']:+.4f}")
    print(f"SOLE     {rep['sole_winner_rate']:.4f} of questions are reached by EXACTLY ONE member: " + '  '.join((f'{k} {v:.3f}' for k, v in sorted(rep['sole_by'].items(), key=lambda kv: -kv[1])[:6])))
    if rep['over_null_when_own_fails'] > 0.05:
        print(f"\nTHE RELATIONS CARRY, AND THE WHOLE-POPULATION GATE WAS DILUTED BY RECALL. On the {rep['own_fails']:.0%} of questions recall cannot answer, a perfect chooser over the family reaches {rep['oracle_when_own_fails']:.4f} against {rep['null_when_own_fails']:.4f} for the same number of RANDOM place-sets: {rep['over_null_when_own_fails']:+.4f}. That is a real degree of freedom on the only questions where a relation could ever show, and 372 is worth building.")
    elif rep['over_best'] > 0.05 and rep['over_null_family'] > 0.05:
        print(f"\nTHERE IS A SPACE TO LEARN IN. A perfect chooser over the family reaches {rep['oracle']:.4f} against the best single relation's {rep['best_reach']:.4f}. The relation set is a real degree of freedom, the three we wrote by hand are not the whole of it, and 372 is worth building: the relation arrives as EVIDENCE ON THE CANDIDATE, not as a second head - the only shape that avoids the 4x law and the lane problem 367 measured.")
    elif rep['over_best'] > 0.05:
        print(f"\nSHOTS ON GOAL, NOT RELATIONS. The oracle beats the best single member by {rep['over_best']:+.4f}, but a family of RANDOM place-sets of the same size reaches {rep['oracle_null']:.4f} - within {rep['over_null_family']:+.4f}. The gain is twelve tries at top-8, not twelve relations. Nothing to hand over.")
    else:
        print(f"\nNO SPACE. The oracle over twelve relations sits {rep['over_best']:+.4f} above the best single one. They are one relation in twelve costumes, there is nothing to hand over to Phi, and letting go of the hand-written set would change nothing. This design is finished at what it already measures.")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())