"""Is there ANY pair signal left once co-occurrence is subtracted - measured before building.

WHY. 309b was a clean negative: evidence-at-fill doubled per-hole accuracy (0.26 -> 0.53) and
pair hits sat exactly on the product of the mind's own marginals, twice, on two different world
constructions. Before that is read as "Phi cannot compose", the other reading has to be priced:
COMP_STRICT is DEFINED as "the pair co-occurs nowhere" - we subtracted the joint statistic from
the subset by construction. walk_only was winnable because the information existed (the truth
stood at a reachable place); for a strict pair the only information that can remain is SECOND
ORDER: do the HOMES of the two values agree - shared rare words, similar places - more for the
true pair than for a wrong one? If yes, composition has fuel and the verb needs that channel.
If no, P(a,b) = P(a)P(b) is the CORRECT posterior on this tape and no mind should beat it -
the negative is a property of the corpus, not of Phi, and the lever is a denser corpus.

This is 305's shape exactly: Kostya's confirmation channel was validated by measuring
separation (0.647) before one line of the verb was written. Same discipline, one level up.

    python _audit310_pairsignal.py --bytes 30000000 --frame-max 3 --sample region         --window-lines 400 --min-fillers 2
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage310_pairsignal.json')
HOMES = 8
RARE_DF = 3

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=2)
    ap.add_argument('--addresses', type=int, default=1500)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--sample', choices=('uniform', 'region'), default='region')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--pairs-per-line', type=int, default=4)
    ap.add_argument('--max-questions', type=int, default=4000)
    args = ap.parse_args()
    text = WIKI.open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
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
    elif args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print('no tape')
        return 1
    place, value, line, pos = ([], [], [], [])
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            pos.append(i)
    n = len(place)
    on_line = defaultdict(list)
    homes_of = defaultdict(list)
    fillers_at = defaultdict(set)
    for s in range(n):
        on_line[line[s]].append(s)
        if line[s] not in homes_of[value[s]]:
            homes_of[value[s]].append(line[s])
        fillers_at[place[s]].add(value[s])
    used = sorted({line[s] for s in range(n)})
    df = Counter()
    words_of = {}
    for li in used:
        ws = set(lines[li].split())
        words_of[li] = ws
        for w0 in ws:
            df[w0] += 1
    for li in used:
        words_of[li] = {w0 for w0 in words_of[li] if df[w0] <= RARE_DF}

    def pair_signal(x, y, qline):
        """Best rare-word overlap between a home line of x and a home line of y - the two
        lines DIFFERENT and both different from the question's, so co-occurrence (the bag)
        and the question's own record can contribute nothing."""
        best = 0
        hx = [l for l in homes_of[x] if l != qline][:HOMES]
        hy = [l for l in homes_of[y] if l != qline][:HOMES]
        for la in hx:
            wa = words_of.get(la, ())
            if not wa:
                continue
            for lb in hy:
                if lb == la:
                    continue
                best = max(best, len(wa & words_of.get(lb, set())))
        return best
    bag = defaultdict(set)
    for li, ss in on_line.items():
        vs = {value[s] for s in ss}
        for v in vs:
            bag[v] |= vs
    qs = []
    for li, ss in on_line.items():
        if len(ss) < 2:
            continue
        cand = [(a, b) for ai, a in enumerate(ss) for b in ss[ai + 1:] if place[a] != place[b] and abs(pos[a] - pos[b]) > args.frame_max]
        rng.shuffle(cand)
        qs.extend(cand[:args.pairs_per_line])
    rng.shuffle(qs)
    if args.max_questions:
        qs = qs[:args.max_questions]
    c = Counter()
    sep_t, sep_w = ([], [])
    for a, b in qs:
        va, vb, qline = (value[a], value[b], line[a])
        c['n'] += 1
        together = sum((1 for l1 in homes_of[va] if l1 in set(homes_of[vb])))
        if together > 1:
            c['cooccur'] += 1
            continue
        wrongs = [w0 for w0 in sorted(fillers_at[place[b]]) if w0 != vb]
        if not wrongs:
            c['no_distractor'] += 1
            continue
        wb = wrongs[rng.randrange(len(wrongs))]
        st = pair_signal(va, vb, qline)
        sw = pair_signal(va, wb, qline)
        c['strict'] += 1
        sep_t.append(st)
        sep_w.append(sw)
        if st != sw:
            c['differ'] += 1
            c['true_higher'] += st > sw
        c['nonzero'] += st > 0 or sw > 0
    strict = max(1, c['strict'])
    differ = max(1, c['differ'])
    rep = {'bytes': args.bytes, 'frame_max': args.frame_max, 'sample': args.sample, 'window_lines': args.window_lines, 'min_fillers': args.min_fillers, 'slots': n, 'questions': c['n'], 'strict': c['strict'], 'cooccur_dropped': c['cooccur'], 'no_distractor': c['no_distractor'], 'signal_nonzero': c['nonzero'] / strict, 'mean_true': sum(sep_t) / strict, 'mean_wrong': sum(sep_w) / strict, 'differ_rate': c['differ'] / strict, 'separates': c['true_higher'] / differ}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape    {n} slots, {c['n']} two-hole questions, {c['strict']} strict (pair co-occurs nowhere else, distractor exists)")
    print(f"signal  nonzero on {rep['signal_nonzero']:.4f} of strict   mean true {rep['mean_true']:.3f} vs wrong {rep['mean_wrong']:.3f}")
    print(f"DECIDE  differ {rep['differ_rate']:.4f}   separates {rep['separates']:.4f}   (0.5 = coin; 305's confirm read 0.647 and was built)")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())