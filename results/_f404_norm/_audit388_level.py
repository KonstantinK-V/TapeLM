"""IS THERE A LEVEL ABOVE A PLACE? Kostya's question, made into a count.

WHERE THIS COMES FROM. Asked for e=mc2 a person does not scan facts - something like
science -> physics -> relativity happens first, and only then the answer, with a logic linking
each narrowing to the question. THIS PROJECT HAS NO SUCH LEVEL. Every operation it has ever run
is place -> place: the fingerprint walk, connect, copy, the deep read, the moves. There is no
"physics" on the tape, only individual holes.

WHAT THE LAST THREE STEPS LEFT, so this is not another lap of the same track:
  387 depth   `step` is exhausted (lane of 24, truth at rank 5). `share` HOLDS THE TRUTH .78-.88
              of the time and puts it at rank 67 - 315 at w1600 - out of 444 - 2466 candidates.
              The material is there and the ORDER is hopeless.
  387 rerank  ranking that lane by MEAN overlap instead of the SUM made it far worse
              (share@8 .516 -> .166, and the frequency decoy fell too, so it was not a rarity
              artifact). That is a real finding in the opposite direction: connect's signal is
              ACCUMULATED WEAK EVIDENCE - many loosely related places agreeing - not one strong
              link. The 317/383 "divide, do not sum" precedent does not transfer here.
  387 gate    the move as an output space is closed: pooling beats choosing, headroom +0.02.

CLOSED, AND NOT TO BE PROPOSED AGAIN: strict two-filler connect (365, lost badly), intersecting
two lenses (346), value-lenses at all (384), a wider offer (347, four times).

WHAT HAS NEVER EXISTED IS TRANSITIVITY. `connect` is ONE hop of sharing a filler. Places A and C
can belong to one region through B without sharing anything directly, and a region is not a
place. This measures whether such regions exist on the tape at all, before anything is built to
use them.

    THE LEVEL, BY COUNTING AND NOTHING ELSE. The tape is a bipartite graph of places and values.
    Every place starts as its own label; then each value takes the commonest label among the
    places holding it, and each place takes the commonest label among the values standing in it,
    ties to the smaller label. R rounds, swept. Linear in the incidences - no pairwise blow-up,
    no threshold, no fitted constant. Values come out labelled too, so every candidate has a
    region and the question's place has one.

  same_label      the truth's region is the question's region ...
  decoy_label     ... against a FREQUENCY-MATCHED twin, because big regions match by size alone
  null_label      ... and against labels permuted between values, size distribution preserved
  in_label@8      the connect lane RESTRICTED to the question's region, same sum ranking
  narrow          how much of the lane the region keeps - the compression the level buys

  GATE  in_label@8 - share@8 > 0.05 AND same_label - decoy_label > 0.05.
        Then the tape HAS regions, narrowing works, and the "linking logic" of the metaphor
        becomes a count: a candidate is admissible if it is in the question's region. If not,
        there is no level above a place here, and one would have to be BUILT rather than found.

    python _audit388_level.py
    python _audit388_level.py --rounds 8 --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage388_level.json')
CUTS = (8, 16, 32)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--weight', choices=('plain', 'inv'), default='inv', help="how a place weighs a value's vote for a region. `plain` is one vote per standing; `inv` divides by how many places hold that value. HUB VALUES BRIDGE EVERYTHING - function words stand in holes too - and with `plain` the propagation collapses to one giant region, which would print as 'the tape has no level' when it means 'this construction collapsed'. Swept, not tuned: run both")
    ap.add_argument('--bridges', type=int, default=32, help='how many hop-1 places the two-hop block expands through, best overlap first. A cost bound, swept, not a threshold on the graph')
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--max-questions', type=int, default=3000)
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
    keep, toks, _owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if not keep:
        print('no tape')
        return 1
    places = [list(ps) for _a, ps in keep]
    place_of = {}
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
    rng0 = random.Random(args.seed)
    scan = [s for ps in places for s in ps]
    rng0.shuffle(scan)
    full_prof = [Counter((toks[s] for s in ps)) for ps in places]
    place_id = place_of
    qsel, hidden = ([], set())
    for s in scan:
        if len(qsel) >= args.max_questions:
            break
        pid = place_id[s]
        own_v = {toks[x] for x in places[pid] if x != s}
        if not own_v or toks[s] in own_v:
            continue
        qsel.append(s)
        hidden.add(s)
    prof = [Counter((toks[x] for x in ps if x not in hidden)) for ps in places]
    at_value = defaultdict(list)
    for pid, pr in enumerate(prof):
        for v in pr:
            at_value[v].append(pid)
    vocab = sorted(at_value)
    freq = Counter()
    for ps in places:
        for s in ps:
            freq[toks[s]] += 1
    lab_place = list(range(len(places)))
    lab_value = {}
    for _r in range(args.rounds):
        for v in vocab:
            cnt = Counter((lab_place[j] for j in at_value[v]))
            if cnt:
                lab_value[v] = min(cnt, key=lambda L: (-cnt[L], L))
        for pid, pr in enumerate(prof):
            cnt = Counter()
            for v, k in pr.items():
                cnt[lab_value[v]] += k / len(at_value[v]) if args.weight == 'inv' else k
            if cnt:
                lab_place[pid] = min(cnt, key=lambda L: (-cnt[L], L))
    sizes = Counter(lab_value.values())
    region_values = defaultdict(set)
    for v, L in lab_value.items():
        region_values[L].add(v)
    shuffled = list(lab_value.values())
    rng.shuffle(shuffled)
    lab_null = dict(zip(sorted(lab_value), shuffled))
    by_band = defaultdict(list)
    for v in vocab:
        by_band[freq[v].bit_length()].append(v)

    def band_draw(truth, banned):
        b = freq[truth].bit_length()
        src, w = (list(by_band[b]), 0)
        while len(src) < 16 and w < 20:
            w += 1
            src += by_band.get(b - w, []) + by_band.get(b + w, [])
        for _t in range(64):
            v = src[rng.randrange(len(src))]
            if v != truth and v not in banned:
                return v
        return None

    def lane(pid, own, qprof, only_label=None):
        """connect's lane, ranked by the SUM of overlaps - 387 measured that summing is right
        here - optionally restricted to one region."""
        ov = Counter()
        for v in qprof:
            for j in at_value[v]:
                if j != pid:
                    ov[j] += 1
        score = Counter()
        for j, o in ov.items():
            for v in prof[j]:
                if v in own:
                    continue
                if only_label is not None and lab_value[v] != only_label:
                    continue
                score[v] += o
        return [v for v, _n in score.most_common()]

    def lane_hop2(pid, own, hop1_places):
        """TRANSITIVITY, LOCALLY, WITH NOTHING GLOBAL TO COLLAPSE.

        The propagation above is one instrument and it can degenerate into a single region; this
        block does not depend on it. A place two hops away shares nothing with the question -
        it is related THROUGH a bridge - and that is the relation this project has never had.
        `connect` is one hop, `--reach-depth 2` is a second sequential READ rooted at a
        candidate, and neither is this.

        The score is a count: for a value v standing at a hop-2 place j, sum over the bridges b
        of min(overlap(q,b), overlap(b,j)) - the weakest link of the path, added over paths,
        because 387 measured that this channel's signal is ACCUMULATED weak evidence.
        """
        seen1 = set(hop1_places) | {pid}
        score = Counter()
        for b, ob in hop1_places.items():
            ov2 = Counter()
            for v in prof[b]:
                for j in at_value[v]:
                    if j not in seen1:
                        ov2[j] += 1
            for j, oj in ov2.items():
                wgt = min(ob, oj)
                for u in prof[j]:
                    if u not in own:
                        score[u] += wgt
        return [v for v, _n in score.most_common()]
    c, keptfrac = (Counter(), [])
    for s in qsel:
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in places[pid] if x != s}
        c['n'] += 1
        here = lab_place[pid]
        c['same'] += lab_value.get(truth) == here
        c['null'] += lab_null.get(truth) == here
        d = band_draw(truth, own)
        if d is not None:
            c['dn'] += 1
            c['decoy'] += lab_value.get(d) == here
        qprof = Counter((toks[x] for x in places[pid] if x != s))
        full = lane(pid, own, qprof)
        small = lane(pid, own, qprof, only_label=here)
        ov1 = Counter()
        for v in qprof:
            for j in at_value[v]:
                if j != pid:
                    ov1[j] += 1
        hop1 = dict(ov1.most_common(args.bridges))
        h2 = lane_hop2(pid, own, hop1)
        c['h2_len'] += len(h2)
        r2 = h2.index(truth) + 1 if truth in h2 else 0
        if r2:
            c['h2_in'] += 1
            for cut in CUTS:
                c[f'h2@{cut}'] += r2 <= cut
        r1 = full.index(truth) + 1 if truth in full else 0
        c['h2_only'] += bool(r2) and (not r1)
        c['h1_only'] += bool(r1) and (not r2)
        if full:
            keptfrac.append(len(small) / len(full))
        reg = region_values.get(here, ())
        c['reg_n'] += len(reg)
        in_reg = truth in reg
        c['reg_hit'] += in_reg
        c['reg_only'] += in_reg and truth not in set(full[:args.topm])
        c['reg_union'] += in_reg or truth in set(full[:args.topm])
        for nm, L in (('all', full), ('lab', small)):
            r = L.index(truth) + 1 if truth in L else 0
            if r:
                for cut in CUTS:
                    c[f'{nm}@{cut}'] += r <= cut
    n, nd = (max(1, c['n']), max(1, c['dn']))
    rep = {'corpus': args.corpus, 'lines': len(lines), 'places': len(places), 'values': len(vocab), 'questions': c['n'], 'rounds': args.rounds, 'weight': args.weight, 'regions': len(sizes), 'region_max': max(sizes.values()) if sizes else 0, 'region_max_frac': max(sizes.values()) / len(vocab) if sizes else 0.0, 'region_mean': sum(sizes.values()) / len(sizes) if sizes else 0.0, 'same_label': c['same'] / n, 'decoy_label': c['decoy'] / nd, 'null_label': c['null'] / n, 'narrow': sum(keptfrac) / max(1, len(keptfrac))}
    for cut in CUTS:
        rep[f'all_at_{cut}'] = c[f'all@{cut}'] / n
        rep[f'lab_at_{cut}'] = c[f'lab@{cut}'] / n
    for cut in CUTS:
        rep[f'hop2_at_{cut}'] = c[f'h2@{cut}'] / n
    rep['hop2_present'] = c['h2_in'] / n
    rep['hop2_only'] = c['h2_only'] / n
    rep['hop1_only'] = c['h1_only'] / n
    rep['hop2_len'] = c['h2_len'] / n
    rep['region_size'] = c['reg_n'] / n
    rep['region_reach'] = c['reg_hit'] / n
    rep['region_only'] = c['reg_only'] / n
    rep['region_union'] = c['reg_union'] / n
    rep['union_gain'] = rep['region_union'] - rep['all_at_8']
    rep['label_gain'] = rep['same_label'] - rep['decoy_label']
    rep['reach_gain'] = rep['lab_at_8'] - rep['all_at_8']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape     {len(places)} places, {len(vocab)} values, {c['n']} questions where the truth is not among the question's own values, {args.rounds} rounds")
    print(f"LEVEL    {rep['regions']} regions over the values ({args.weight})   mean {rep['region_mean']:.1f}   largest {rep['region_max']} ({rep['region_max_frac']:.1%} of the values)   the lane keeps {rep['narrow']:.4f} of itself")
    print(f"BELONGS  truth {rep['same_label']:.4f}   frequency twin {rep['decoy_label']:.4f}   permuted labels {rep['null_label']:.4f}   gain {rep['label_gain']:+.4f}")
    print('REACH    ' + '  '.join((f"@{cut} {rep[f'all_at_{cut}']:.4f}->{rep[f'lab_at_{cut}']:.4f}" for cut in CUTS)) + '   (whole lane -> its region)')
    print(f"SOURCE   region holds {rep['region_size']:.1f} values   reach {rep['region_reach']:.4f}   only region {rep['region_only']:.4f}   region+lane@8 {rep['region_union']:.4f}   over lane@8 {rep['union_gain']:+.4f}   cost {rep['region_size'] + args.topm:.0f}")
    print(f'HOP2     ' + '  '.join((f"@{cut} {rep[f'hop2_at_{cut}']:.4f}" for cut in CUTS)) + f"   present {rep['hop2_present']:.4f}   lane {rep['hop2_len']:.0f}   only hop2 {rep['hop2_only']:.4f}   only hop1 {rep['hop1_only']:.4f}")
    if c['n'] < 300:
        print(f"\nVOID, NOT A RESULT. Only {c['n']} questions; widen --window-lines.")
    elif rep['region_max_frac'] > 0.5:
        print(f"\nTHE PROPAGATION COLLAPSED - NOT A RESULT ABOUT THE TAPE. One region holds {rep['region_max_frac']:.1%} of the values, so belonging to it says nothing: truth {rep['same_label']:.4f} and twin {rep['decoy_label']:.4f} are both near one by construction. HUB VALUES BRIDGE EVERYTHING - function words stand in holes too - and they merge every region into one. Try --weight {('plain' if args.weight == 'inv' else 'inv')} and fewer --rounds; if the largest region stays over half the values at every setting, label propagation is the wrong instrument here and the question is still open.")
    elif rep['reach_gain'] > 0.05 and rep['label_gain'] > 0.05:
        print(f"\nTHE TAPE HAS REGIONS AND NARROWING WORKS. The truth shares the question's region {rep['same_label']:.4f} of the time against {rep['decoy_label']:.4f} for a frequency twin, and keeping only that region lifts the offered eight {rep['reach_gain']:+.4f} while throwing away {1 - rep['narrow']:.0%} of the lane. The linking logic of the metaphor is now a count: a candidate is admissible if it is in the question's region. That is the first level above a place this project has had.")
    elif rep['label_gain'] > 0.05:
        print(f"\nTHE REGIONS ARE REAL AND DO NOT NARROW USEFULLY. The truth belongs to the question's region {rep['same_label']:.4f} against a twin's {rep['decoy_label']:.4f}, so the level EXISTS - but restricting the lane to it moves the offered eight {rep['reach_gain']:+.4f}. The region is not the bottleneck: what buries the truth at rank 67 is inside the region too.")
    else:
        print(f"\nTHIS CONSTRUCTION FINDS NO USABLE LEVEL. The truth shares the question's region {rep['same_label']:.4f} against {rep['decoy_label']:.4f} for a frequency twin and {rep['null_label']:.4f} for permuted labels - so what looks like a region is mostly its own size. Propagating labels over the tape finds communities of places, and they are not the communities a question needs. Sweep --weight and --rounds before concluding: the claim earned here is about LABEL PROPAGATION on this tape, not about every possible level, and a level may still have to be BUILT from something the tape does not already say.")
    if rep['hop2_only'] > 0.05:
        print(f"\nTRANSITIVITY REACHES WHERE ONE HOP CANNOT. {rep['hop2_only']:.4f} of these questions have the truth at a place two hops away and NOWHERE one hop away - related through a bridge, sharing nothing with the question directly. That is a relation this project has never had: connect is one hop and --reach-depth 2 is a second sequential read rooted at a candidate, not this. Whether it PAYS is the @8 column, and whether it is affordable is the lane size.")
    else:
        print(f"\nTWO HOPS ADD NOTHING THE ONE HOP DID NOT HAVE. Only {rep['hop2_only']:.4f} of questions are reachable at two hops and not at one, against {rep['hop1_only']:.4f} the other way. The sharing graph closes on itself: a place two hops out holds the same values as the places one hop out, so transitivity is not a new route here, and a level above a place cannot be built by walking further.")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())