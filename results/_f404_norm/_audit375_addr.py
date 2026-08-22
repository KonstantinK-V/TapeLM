"""THE ATOM OF THE ADDRESS. 373/374 tested the value's atom; the frame's was never tested.

374 on german said both things at once: the frame's function words carry the ending
(pred_func .52 vs null .27) and the material for production is absent (oracle .02). So the
form signal is real and its place is not the VALUE but the ADDRESS. Today an address is a
tuple of EXACT tokens - `singleton_bucket 1.0`, every frame unique - which is why places are
thin, `own` is thin, the offer is thin, and an unreachable population exists at all.

ONE LEVER: generalise the alphabet the frame is written in, leave the value whole.

    kappa(t) = t        if t is one of the corpus's F most frequent tokens (counted, no list)
             = e(t)     otherwise - its ending class, 374's counted inventory

    addr'(i) = (w, kappa(left), kappa(right))

`the walked dog of` and `the crossed cat of` become one address. Frames recur more, places
pool, own/offer thicken - the unreachable set is attacked at the WRITE PATH, where 356 said
coverage lives, instead of teaching production inside it.

  GATES, before the run, exact tape as baseline on the same window:
    G1  unreachable/seen drops by >= 20% relative
    G2  top-1 hit on the offer does not fall by more than 0.02 absolute
  Swept, not tuned: --func 16/32/64.

    python _audit375_addr.py
    python _audit375_addr.py --func 64 --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from _audit374_shape import ending_inventory, make_split
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage375_addr.json')

def build_tape(lines, frame_max, min_fillers, kappa):
    """374's frame cutter with the frame spelled in kappa; the hole's value stays the token."""
    toks = [t for line in lines for t in line.split()]
    ktoks = [kappa(t) for t in toks]
    n = len(toks)
    best_key, alive = ({}, list(range(1, n - 1)))
    for w in range(1, frame_max + 1):
        pos_of = defaultdict(list)
        for i in alive:
            if i - w < 0 or i + 1 + w > n:
                continue
            pos_of[tuple(ktoks[i - w:i]), tuple(ktoks[i + 1:i + 1 + w])].append(i)
        nxt = []
        for k, ps in pos_of.items():
            if len(ps) >= 2:
                for i in ps:
                    best_key[i] = (w,) + k
                nxt.extend(ps)
        alive = nxt
        if not alive:
            break
    by_addr = defaultdict(list)
    for i, k in best_key.items():
        by_addr[k].append(i)
    keep = [sorted(ps) for k, ps in by_addr.items() if len({toks[i] for i in ps}) >= min_fillers]
    return (keep, toks, n)

def exam(places, toks, topm, max_q, rng):
    place_of, where = ({}, defaultdict(list))
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
            where[toks[s]].append(s)
    cooc = {}
    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    c = Counter()
    for s in qs:
        if c['seen'] >= max_q:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in places[pid] if x != s}
        if not own:
            continue
        c['seen'] += 1
        ownc = Counter((toks[x] for x in places[pid]))
        ownc[truth] -= 1
        if ownc[truth] <= 0:
            del ownc[truth]
        lens = list(ownc)[:6]
        pairs = []
        if lens:
            ban = Counter((toks[x] for x in places[pid]))
            off = Counter()
            for v in lens:
                cv = cooc.get(v)
                if cv is None:
                    cv = Counter()
                    for s2 in where[v]:
                        for s3 in places[place_of[s2]]:
                            if toks[s3] != v:
                                cv[toks[s3]] += 1
                    cooc[v] = cv
                for w, k in cv.items():
                    k -= ban.get(w, 0)
                    if k > 0 and w != v:
                        off[w] += k
            pairs = off.most_common(topm)
        offs = {w for w, _n in pairs}
        if truth in own or truth in offs:
            c['reach'] += 1
            if truth in offs:
                c['in_off'] += 1
                c['hit1'] += pairs[0][0] == truth
        else:
            c['unreach'] += 1
    return c

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--endings', type=int, default=64)
    ap.add_argument('--suffix-max', type=int, default=4)
    ap.add_argument('--func', type=int, default=32)
    ap.add_argument('--max-questions', type=int, default=4000)
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
    toks_all = [t for l in lines for t in l.split()]
    freq = Counter(toks_all)
    func = {w for w, _n in freq.most_common(args.func)}
    split = make_split(ending_inventory(sorted(set(toks_all)), args.suffix_max, args.endings))

    def kappa(t):
        return t if t in func else '~' + (split(t)[1] or '0')
    rep = {'corpus': args.corpus, 'lines': len(lines), 'func': args.func, 'endings': args.endings, 'topm': args.topm}
    for name, k in (('exact', lambda t: t), ('shaped', kappa)):
        keep, toks, n = build_tape(lines, args.frame_max, args.min_fillers, k)
        cov = sum((len(ps) for ps in keep)) / max(1, n)
        c = exam(keep, toks, args.topm, args.max_questions, random.Random(args.seed))
        seen = max(1, c['seen'])
        rep[name] = {'places': len(keep), 'coverage': cov, 'fillers': sum((len(ps) for ps in keep)) / max(1, len(keep)), 'seen': c['seen'], 'unreach': c['unreach'] / seen, 'reach': c['reach'] / seen, 'hit1': c['hit1'] / max(1, c['in_off'])}
        r = rep[name]
        print(f"{name:7s} places {r['places']:6d}  cov {cov:.4f}  fill/place {r['fillers']:.2f}  seen {c['seen']}  unreach {r['unreach']:.4f}  hit1 {r['hit1']:.4f}")
    e, s = (rep['exact'], rep['shaped'])
    rep['unreach_rel'] = (e['unreach'] - s['unreach']) / max(1e-09, e['unreach'])
    rep['hit_delta'] = s['hit1'] - e['hit1']
    rep['cov_delta'] = s['coverage'] - e['coverage']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"\nGATES  unreach -{rep['unreach_rel'] * 100:.1f}% rel (need >=20)   hit1 {rep['hit_delta']:+.4f} (need > -0.02)   cov {rep['cov_delta']:+.4f}")
    if min(e['seen'], s['seen']) < 400:
        print('VOID - too few questions; widen --window-lines.')
    elif rep['unreach_rel'] >= 0.2 and rep['hit_delta'] > -0.02:
        print('THE ADDRESS ATOM WAS THE WALL. Frames spelled in form-classes pool the places, the material arrives at the write path, and nothing was produced by hand.')
    else:
        print('THE ADDRESS ATOM WAS NOT THE WALL at this width - pooling frames does not deliver the missing material, or it costs the hit. Read cov and fill/place before concluding: if both rose and unreach did not fall, the missing truths live outside ANY recurring frame, and the wall is the corpus, not the alphabet.')
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())