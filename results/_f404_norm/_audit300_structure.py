"""How much structure is there to compose over - measured before anything is built for it.

WHY THIS COMES FIRST. 299 settled that the mind ties counting nearly everywhere and beats it in
one corner, and the corner turned out to be the boring one: ceiling 23-27%, of which own_hit is
18-22% and walk_only about 5%. So roughly four fifths of every answerable question is CONFIRM -
the hole repeats a filler its own frame already showed - and the walk, the whole point of 299,
can affect a twentieth of the outcome. Polishing a reader against that is polishing a lookup.

The reason is structural, not statistical. THE FRAME TAPE IS A FLAT BAG OF INDEPENDENT PLACES.
The parsed tape's address had two halves, so places were tied together by a shared anchor and
one could ask what a subject's OTHER relations say. That tie went out with the parser - rightly,
its relation half was a function word - and what replaced it, filler-bag similarity, is not a
tie between places at all. It is a measure of resemblance, and resemblance lets you find
something similar; it does not let you put two records together.

So this asks whether the corpus carries a tie we have not used, and it asks in counts only - no
ink, no training, no model. Three candidate ties, each exact:

  OWN      the truth already stands at this very frame (the CONFIRM case, today's 18-22%)
  SHARE    it stands at a frame that shares at least one filler with this one. This is the exact
           ceiling of the filler-bag walk - no fingerprint can reach a place with nothing in
           common, so whatever the cosine does, it cannot beat this.
  LINE     it stands at ANOTHER FRAME IN THE SAME LINE. Two frames written in one sentence are
           about one thing. Counted, no grammar, no stopwords - the same kind of fact the cutter
           itself is built from.
  NEAR     the same, one line either side.

What decides whether to build anything is not LINE's size but LINE MINUS (OWN or SHARE): the
questions only adjacency can answer. And beside it, the price - how many candidates a step along
the line offers, against the eight the walk offers now. A tie that answers a third of the
questions by handing over two hundred candidates is not an answer, it is a bigger haystack.

    python _audit300_structure.py                       600 KB, frame_max 3, as 299 ran
    python _audit300_structure.py --bytes 4000000 --frame-max 3 --addresses 3000
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage300_structure.json')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=4000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--addresses', type=int, default=3000)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--max-questions', type=int, default=20000)
    (ap.add_argument('--window-lines', type=int, default=0, help='grow the region to EXACTLY this many lines instead of stopping at an address budget. Without it, changing --min-fillers changes the window too: at 1 every position qualifies at once, the window closes after about fifty lines and mentions per address falls to 1.2, so mf=1 and mf=2 are never compared on the same text. Fix the window and only the rule moves'),)
    (ap.add_argument('--min-fillers', type=int, default=2, help='how many DIFFERENT values a hole must have taken for its frame to be kept. 2 since 297, never questioned, and it deletes exactly the deterministic records: a frame whose surroundings FIX the answer has one filler and is discarded as a fixed phrase. `capital of france is ___` can never be on the tape. That is why an address is made of recurring glue and almost never names an entity, and therefore why there are no pointers to compose over - we delete them when writing'),)
    ap.add_argument('--sample', choices=('uniform', 'region'), default='uniform', help="how the tape's addresses are chosen. `uniform` is what every run has done: 3000 drawn from the whole corpus's qualifying frames, 1.2% of 250195, so two frames of one sentence are on the tape together about one time in a hundred. `region` takes EVERY frame of a contiguous stretch of lines until the budget is full, which is the same tape size over a connected object")
    args = ap.parse_args()
    text = WIKI.open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    pool = len(keep)
    if args.sample == 'region':
        if args.window_lines:
            by_line = tframes._by_line(keep, owner)
            start = rng.randrange(max(1, len(lines)))
            acc = defaultdict(list)
            for d in range(args.window_lines):
                for k, i in by_line.get((start + d) % len(lines), ()):
                    acc[k].append(i)
            keep = [(k, sorted(v)) for k, v in acc.items() if len({toks[i] for i in v}) >= args.min_fillers]
        else:
            keep = tframes.frame_region(keep, toks, owner, len(lines), args.addresses, rng, args.min_fillers)
    elif args.addresses and pool > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print('no tape')
        return 1
    place, value, line, addrs = ([], [], [], [])
    for (w, left, right), pos in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        addrs.append(name)
        for i in pos:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
    n = len(place)
    slots_at = defaultdict(list)
    fillers_at = defaultdict(set)
    places_with = defaultdict(set)
    on_line = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
        fillers_at[place[s]].add(value[s])
        places_with[value[s]].add(place[s])
        if line[s] >= 0:
            on_line[line[s]].append(s)
    qs = list(range(n))
    if args.max_questions and n > args.max_questions:
        qs = rng.sample(qs, args.max_questions)
    addr_words = {}
    for pl in slots_at:
        left, _, right = pl.partition('|')
        addr_words[pl] = [w for w in left.split() + right.split() if w]
    word_places = defaultdict(set)
    for v, ps in places_with.items():
        word_places[v] = ps

    def ref_places_of(pl):
        ws = sorted(set(addr_words.get(pl, ())), key=lambda w: len(word_places.get(w, ())))
        out = []
        for w in ws:
            for q in sorted(word_places.get(w, ()))[:8]:
                if q != pl and q not in out:
                    out.append(q)
                if len(out) >= 8:
                    return out
        return out
    line_df = Counter()
    for li, ln in enumerate(lines):
        for w in set(ln.split()):
            line_df[w] += 1
    RARE = 3

    def rare_shared(li_a, li_b):
        if li_a < 0 or li_b < 0 or li_a == li_b:
            return 0
        A = {w for w in lines[li_a].split() if line_df[w] <= RARE}
        return sum((1 for w in set(lines[li_b].split()) if w in A))
    CAPS = (8, 16, 32, 64, 128, 256, 512, 10 ** 9)
    hit_at = Counter()
    near_memo = {}

    def near8_of(pl, seen_here):
        key = (pl, frozenset(seen_here)) if len(seen_here) < 6 else pl
        if key in near_memo:
            return near_memo[key]
        cand = {q for f in seen_here for q in places_with[f]} - {pl}
        out = sorted(cand, key=lambda q: -len(fillers_at[q] & seen_here))[:8]
        near_memo[key] = out
        return out
    c = Counter()
    line_cands, share_cands, line_places, ranks, hop2_cands = ([], [], [], [], [])
    ref_cands, conf_true, conf_wrong = ([], [], [])
    for s in qs:
        p, v, li = (place[s], value[s], line[s])
        own = any((value[t] == v for t in slots_at[p] if t != s))
        seen_here = {value[t] for t in slots_at[p] if t != s}
        share = any((fillers_at[q] & seen_here for q in places_with[v] if q != p))
        sib = [t for t in on_line.get(li, ()) if place[t] != p] if li >= 0 else []
        ln = any((value[t] == v for t in sib))
        near = ln
        for d in (-1, 1):
            if not near and li >= 0:
                near = any((value[t] == v and place[t] != p for t in on_line.get(li + d, ())))
        anywhere = len(places_with[v] - {p}) > 0
        c['n'] += 1
        c['own'] += own
        c['share'] += share
        c['line'] += ln
        c['near'] += near
        c['anywhere'] += anywhere
        c['own_or_share'] += own or share
        c['line_only'] += ln and (not (own or share))
        c['near_only'] += near and (not (own or share))
        c['nothing'] += not (own or share or ln or near)
        line_cands.append(len({value[t] for t in sib}))
        line_places.append(len({place[t] for t in sib}))
        near8 = near8_of(p, seen_here)
        share_cands.append(sum((len(fillers_at[q]) for q in near8)))
        seen_c, rank = (set(), None)
        for q in near8:
            for t in slots_at[q]:
                w = value[t]
                if w not in seen_c:
                    seen_c.add(w)
                    if w == v and rank is None:
                        rank = len(seen_c)
        for cap in CAPS:
            hit_at[cap] += int(rank is not None and rank <= cap)
        if rank is not None:
            ranks.append(rank)
        hop1 = rank is not None and rank <= 8
        ring1 = set(near8) | {p}
        seen2, hop2 = (set(), False)
        n2 = 0
        for q in near8:
            for q2 in near8_of(q, fillers_at[q]):
                if q2 in ring1:
                    continue
                ring1.add(q2)
                for t in slots_at[q2]:
                    w = value[t]
                    if w not in seen2:
                        seen2.add(w)
                        n2 += 1
                        if w == v and n2 <= 8:
                            hop2 = True
        refp = ref_places_of(p)
        seen_r, nref, ref = (set(), 0, False)
        for q in refp:
            for t in slots_at[q]:
                w = value[t]
                if w not in seen_r:
                    seen_r.add(w)
                    nref += 1
                    if w == v and nref <= 8:
                        ref = True
        c['ref'] += ref
        c['ref_only'] += ref and (not (own or share or ln))
        ref_cands.append(nref)
        wrong = next((w for w in (value[t] for q in near8 for t in slots_at[q]) if w != v), None)
        if li >= 0 and wrong is not None:
            conf_true.append(max((rare_shared(li, line[t]) for t in slots_at[p] if value[t] == v and t != s), default=0) or max((rare_shared(li, line[t]) for q in near8 for t in slots_at[q] if value[t] == v), default=0))
            conf_wrong.append(max((rare_shared(li, line[t]) for q in near8 for t in slots_at[q] if value[t] == wrong), default=0))
        c['hop2'] += hop2
        c['hop2_only'] += hop2 and (not (own or hop1))
        hop2_cands.append(n2)

    def pct(k):
        return c[k] / max(1, c['n'])
    rep = {'bytes': args.bytes, 'frame_max': args.frame_max, 'sample': args.sample, 'min_fillers': args.min_fillers, 'window_lines': args.window_lines, 'addresses': len(addrs), 'frame_pool': pool, 'slots': n, 'questions': c['n'], 'lines': len(lines), 'mentions_per_address': n / max(1, len(addrs)), 'own': pct('own'), 'share': pct('share'), 'line': pct('line'), 'near': pct('near'), 'anywhere': pct('anywhere'), 'own_or_share': pct('own_or_share'), 'line_only': pct('line_only'), 'near_only': pct('near_only'), 'nothing': pct('nothing'), 'line_candidates_mean': sum(line_cands) / max(1, len(line_cands)), 'line_places_mean': sum(line_places) / max(1, len(line_places)), 'share_candidates_mean_at_8': sum(share_cands) / max(1, len(share_cands)), 'reachable_at_cap': {'all' if k > 10 ** 8 else k: hit_at[k] / max(1, c['n']) for k in CAPS}, 'truth_rank_median': sorted(ranks)[len(ranks) // 2] if ranks else None, 'truth_found_share': len(ranks) / max(1, c['n']), 'hop2': pct('hop2'), 'hop2_only': pct('hop2_only'), 'hop2_candidates_mean': sum(hop2_cands) / max(1, len(hop2_cands)), 'ref': pct('ref'), 'ref_only': pct('ref_only'), 'ref_candidates_mean': sum(ref_cands) / max(1, len(ref_cands)), 'confirm_rare_true': sum(conf_true) / max(1, len(conf_true)), 'confirm_rare_wrong': sum(conf_wrong) / max(1, len(conf_wrong)), 'confirm_separates': sum((1 for a, b in zip(conf_true, conf_wrong) if a > b)) / max(1, sum((1 for a, b in zip(conf_true, conf_wrong) if a != b)))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape   [{args.sample} min_fillers={args.min_fillers}] {rep['addresses']} addresses, {n} slots, {rep['mentions_per_address']:.2f} mentions/address, pool {pool}")
    print(f"reach  own {rep['own']:.3f}  share {rep['share']:.3f} (the walk's exact ceiling)  own_or_share {rep['own_or_share']:.3f}")
    print(f"ties   line {rep['line']:.3f}  near {rep['near']:.3f}  anywhere {rep['anywhere']:.3f}")
    print(f"NEW    line_only {rep['line_only']:.3f}   near_only {rep['near_only']:.3f}   unanswerable by anything {rep['nothing']:.3f}")
    print('APERTURE  reachable when the walk may name the top C fillers of its eight places:')
    print('          ' + '  '.join((f'C={k}:{v:.3f}' for k, v in rep['reachable_at_cap'].items())))
    print(f"          the truth sits at rank {rep['truth_rank_median']} of that list (median, over the {rep['truth_found_share']:.3f} where it is in the list at all)")
    print(f"REF    through a pointer {rep['ref']:.4f}   and ONLY so {rep['ref_only']:.4f}   (offers {rep['ref_candidates_mean']:.0f} fillers)")
    print(f"CONF   rare words shared with this line: truth {rep['confirm_rare_true']:.2f} vs wrong {rep['confirm_rare_wrong']:.2f}, truth ahead in {rep['confirm_separates']:.3f} of the pairs that differ")
    print(f"HOP2   reachable in two hops {rep['hop2']:.4f}   and ONLY in two {rep['hop2_only']:.4f}   (the second ring offers {rep['hop2_candidates_mean']:.0f} fillers)")
    print(f"price  a line step offers {rep['line_candidates_mean']:.1f} fillers from {rep['line_places_mean']:.1f} places; the filler walk offers {rep['share_candidates_mean_at_8']:.1f} from 8")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())