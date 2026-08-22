"""Could memory buy anything on this tape - measured before any memory is built.

WHAT MEMORY WOULD MEAN HERE. Every question is answered from nothing but the tape; nothing
found at question k is available at question k+1 except the weights. The cheapest real memory
is a WRITE-BACK: once a hole is resolved, that (place, value) becomes an ordinary row, visible
to every later question. Successes then accumulate and mistakes poison - which is what memory
is, and why it must be priced before it is built.

THE CEILING IS WHAT THIS MEASURES, not the mechanism. Assume a PERFECT memory: every earlier
hole resolved correctly and written back. How many later questions would that answer, over and
above what the question's own rows already give? If the answer is ~0, memory cannot pay on this
tape whatever mechanism carries it, and the right move is to say so instead of building it.

Three ceilings, from tightest to loosest:
  same place   - a later hole at the SAME place whose truth a remembered row already holds.
                 This overlaps CONFIRM by construction, so the number that matters is the part
                 NOT already in the question's own rows.
  shared       - a later hole at a place that SHARES A FILLER with a remembered place, and whose
                 truth is among that place's remembered values. This is the walk with memory.
  any          - the truth was remembered anywhere at all. A loose upper bound, printed so the
                 tighter numbers can be read against something.

    python _audit324_memory.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage324_memory.json')

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
    ap.add_argument('--max-questions', type=int, default=4000)
    ap.add_argument('--recall', type=float, default=1.0, help='the share of earlier holes memory gets RIGHT. 1.0 is the ceiling; lower values show how fast the ceiling decays when memory is wrong, which is the risk a write-back actually carries')
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
            if args.addresses and len(keep) > args.addresses:
                keep = rng.sample(keep, args.addresses)
        else:
            keep = tframes.frame_region(keep, toks, owner, len(lines), args.addresses, rng, args.min_fillers)
    elif args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print('no tape')
        return 1
    vals_at = []
    for (_w, left, right), ps in keep:
        vals_at.append([toks[i] for i in ps])
    P = len(vals_at)
    bags = [Counter(v) for v in vals_at]
    where = defaultdict(set)
    for j, b in enumerate(bags):
        for v in b:
            where[v].add(j)
    qs = [(j, i) for j in range(P) for i in range(len(vals_at[j])) if len(vals_at[j]) >= 2]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]
    K, CANDS = (8, 8)

    def walk_offer(j, own):
        share = Counter()
        for v, cnt in own.items():
            for o in where[v]:
                if o != j:
                    share[o] += cnt
        places = [o for o, _n in share.most_common(K)]
        seen, out = (set(), [])
        for o in places:
            for v in vals_at[o]:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
        return (places, set(out[:CANDS]))
    mem_place = defaultdict(set)
    mem_any = set()
    c = Counter()
    for j, i in qs:
        truth = vals_at[j][i]
        own = Counter(vals_at[j])
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        c['n'] += 1
        in_own = truth in own
        c['in_own'] += in_own
        same = truth in mem_place.get(j, ())
        places, offered = walk_offer(j, own)
        c['walk_reach'] += truth in offered
        shared = False
        if not same:
            for o in places:
                if truth in mem_place.get(o, ()):
                    shared = True
                    break
        c['same_place'] += same and (not in_own)
        c['shared'] += (same or shared) and (not in_own) and (truth not in offered)
        c['shared_incl_walk'] += (same or shared) and (not in_own)
        c['any'] += truth in mem_any and (not in_own) and (truth not in offered)
        c['answerable_wo_own'] += not in_own
        c['unreached'] += not in_own and truth not in offered
        if args.recall >= 1.0 or rng.random() < args.recall:
            mem_place[j].add(truth)
            mem_any.add(truth)
    n = max(1, c['n'])
    ur = max(1, c['unreached'])
    rep = {'bytes': args.bytes, 'sample': args.sample, 'window_lines': args.window_lines, 'recall': args.recall, 'places': P, 'questions': c['n'], 'in_own': c['in_own'] / n, 'walk_reach': c['walk_reach'] / n, 'same_place_gain': c['same_place'] / n, 'shared_gain': c['shared'] / n, 'shared_incl_walk': c['shared_incl_walk'] / n, 'any_gain': c['any'] / n, 'shared_gain_of_unreached': c['shared'] / ur}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape    {P} places, {c['n']} questions in order, in_own {rep['in_own']:.4f}")
    print(f"CEILING beyond the question's own rows, with recall {args.recall}:")
    print(f"        walk already reaches {rep['walk_reach']:.4f}")
    print(f"        same place {rep['same_place_gain']:.4f}   shared-and-remembered, BEYOND the walk {rep['shared_gain']:.4f}   (including what the walk gives: {rep['shared_incl_walk']:.4f})")
    print(f"        anywhere, beyond the walk {rep['any_gain']:.4f}")
    print(f"        as a share of what neither own rows nor the walk reach: {rep['shared_gain_of_unreached']:.4f}")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())