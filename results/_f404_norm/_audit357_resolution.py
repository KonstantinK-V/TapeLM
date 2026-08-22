"""ONE POSITION, ONE ADDRESS - the fiat 356 exposed, and the only part of it worth anything.

WHAT 356 SETTLED, INCLUDING AGAINST MY OWN PREDICTION:

    coverage      0.5594 of ALL interior positions are on the tape (0.5937 at frame_max 1).
                  Not a sliver. Everything measured for 350 steps sat inside a MAJORITY of the
                  text, and the honest end-to-end figure is coverage x hit = 0.559 x 0.445
                  = 0.25 of every token position in wikitext, recovered by counts plus 5633
                  parameters.
    rarity        A ONCE-SEEN token reaches the tape 0.6836 of the time; a token seen 101+
                  times reaches it 0.5260. I PREDICTED THE OPPOSITE. The tape is not a lattice
                  of categories - a frequent token fails because its hole has ONE filler and is
                  dropped, which is predictability, not rarity. Facts are present.
    the fiat      costs 3.4 points of coverage (0.5937 -> 0.5594) and touches only the 16.2% of
                  positions that reach width 2 or 3. AS A COVERAGE LEVER IT IS NOT WORTH A RUN,
                  and the plan of "give the width to Phi" dies here on its own numbers.

WHAT SURVIVES, AND IT IS A DIFFERENT MECHANISM. `frame_keep` writes `best_key[i] = the widest
recurring frame` - ONE address per position, the narrower paradigms thrown away. But a width-3
position is simultaneously a member of its width-2 and width-1 paradigms, and THOSE ARE THE
COARSE ONES: more fillers, more bridges, the very places a walk needs to leave a narrow
paradigm and land in another. The tape currently deletes, for 16% of its positions, the only
edges that connect them widely.

THIS IS NOT A SECOND RELATION. 349 (same line) and 350 (directed triples) added a DIFFERENT
kind of edge and both lost. This is the SAME substitution relation read at more than one
granularity - the one thing the write path has always thrown away.

    reach_wide    the truth in the offer, one address per position          (today)
    reach_multi   the same with every recurring width kept as an address
    multi_only    reached with the coarse layer and not without it          THE NUMBER
    reach_null    THE NULL: the coarse addresses PERMUTED among the positions that have one -
                  same layer, same sizes, wrong owners. A gain that survives this is the
                  position's own coarse paradigm; a gain that does not is offer-widening again,
                  which is how 354 died.

  GATE  reach_multi - reach_null > 0.05 at matched top-m.

    python _audit357_resolution.py
    python _audit357_resolution.py --window-lines 3200      # thicker, as 347 swept it
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage357_resolution.json')

def build(lines, frame_max, min_fillers):
    """every recurring width kept, not just the widest. Returns the token list and, per width,
    the address -> positions map. The widest-only tape is recovered from the same tables, so
    the two arms cannot differ by anything except which addresses are kept."""
    toks = [t for line in lines for t in line.split()]
    n = len(toks)
    at_width = {}
    alive = list(range(1, n - 1))
    for w in range(1, frame_max + 1):
        pos_of = defaultdict(list)
        for i in alive:
            if i - w < 0 or i + 1 + w > n:
                continue
            pos_of[tuple(toks[i - w:i]), tuple(toks[i + 1:i + 1 + w])].append(i)
        keep = {k: ps for k, ps in pos_of.items() if len(ps) >= 2}
        if not keep:
            break
        at_width[w] = keep
        alive = [i for ps in keep.values() for i in ps]
    return (toks, at_width)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=2)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--max-questions', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--corpus', default=str(WIKI))
    args = ap.parse_args()
    text = Path(args.corpus).open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s:s + args.window_lines]
    toks, at_width = build(lines, args.frame_max, args.min_fillers)
    if not at_width:
        print('no tape')
        return 1
    widths = sorted(at_width)
    best = {}
    for w in widths:
        for k, ps in at_width[w].items():
            for i in ps:
                best[i] = (w, k)
    wide_places, multi_places = ([], [])
    wide_of, multi_of = (defaultdict(list), defaultdict(list))

    def add(places, index, ps):
        if len({toks[i] for i in ps}) < args.min_fillers:
            return
        pid = len(places)
        places.append(list(ps))
        for i in ps:
            index[i].append(pid)
    by_best = defaultdict(list)
    for i, (w, k) in best.items():
        by_best[w, k].append(i)
    for _a, ps in by_best.items():
        add(wide_places, wide_of, sorted(ps))
    for w in widths:
        for _k, ps in at_width[w].items():
            add(multi_places, multi_of, sorted(ps))
    null_places, null_of = ([], defaultdict(list))
    for w in widths:
        addrs, extras = ([], [])
        for _k, ps in at_width[w].items():
            if len({toks[i] for i in ps}) < args.min_fillers:
                continue
            nat = [i for i in ps if best[i][0] == w]
            ext = [i for i in ps if best[i][0] != w]
            addrs.append((nat, len(ext)))
            extras.extend(ext)
        rng.shuffle(extras)
        cut = 0
        for nat, m in addrs:
            seat = nat + extras[cut:cut + m]
            cut += m
            pid = len(null_places)
            null_places.append(seat)
            for i in seat:
                null_of[i].append(pid)

    def offers(places, index):
        where = defaultdict(list)
        for pid, ps in enumerate(places):
            for i in ps:
                where[toks[i]].append(i)
        cache = {}

        def co(v, banned):
            c = cache.get(v)
            if c is None:
                c = Counter()
                for i in where[v]:
                    for pid in index[i]:
                        for i2 in places[pid]:
                            if toks[i2] != v:
                                c[toks[i2]] += 1
                cache[v] = c
            out = Counter()
            here = Counter((toks[i] for i in banned))
            for wd, cnt in c.items():
                cnt -= here.get(wd, 0)
                if cnt > 0:
                    out[wd] = cnt
            return out
        return co
    co_w, co_m, co_n = (offers(wide_places, wide_of), offers(multi_places, multi_of), offers(null_places, null_of))
    qs = [(pid, i) for pid, ps in enumerate(wide_places) for i in ps]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]
    c = Counter()
    sw, sm = ([], [])
    for pid, i in qs:
        truth = toks[i]
        own = Counter((toks[x] for x in wide_places[pid]))
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        lens = list(own)[:6]
        if not lens:
            continue
        c['n'] += 1
        ban_w = [x for p in wide_of[i] for x in wide_places[p]]
        ban_m = [x for p in multi_of[i] for x in multi_places[p]]
        ban_n = [x for p in null_of[i] for x in null_places[p]]

        def top(co, ban):
            s = Counter()
            for v in lens:
                s += co(v, ban)
            return ({w for w, _n in s.most_common(args.topm)}, len(s))
        tw, nw = top(co_w, ban_w)
        tm, nm = top(co_m, ban_m)
        tn, _nn = top(co_n, ban_n)
        c['reach_wide'] += truth in tw
        c['reach_multi'] += truth in tm
        c['reach_null'] += truth in tn
        c['multi_only'] += truth in tm and truth not in tw
        c['lost'] += truth in tw and truth not in tm
        sw.append(nw)
        sm.append(nm)
    n = max(1, c['n'])
    rep = {'lines': len(lines), 'questions': c['n'], 'topm': args.topm, 'wide_places': len(wide_places), 'multi_places': len(multi_places), 'reach_wide': c['reach_wide'] / n, 'reach_multi': c['reach_multi'] / n, 'reach_null': c['reach_null'] / n, 'multi_only': c['multi_only'] / n, 'lost': c['lost'] / n, 'offer_wide': sum(sw) / max(1, len(sw)), 'offer_multi': sum(sm) / max(1, len(sm))}
    rep['gain'] = rep['reach_multi'] - rep['reach_wide']
    rep['gain_over_null'] = rep['reach_multi'] - rep['reach_null']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape     {len(wide_places)} widest-only places, {len(multi_places)} multi-resolution, {c['n']} questions, topm {args.topm}")
    print(f"WIDE     reach {rep['reach_wide']:.4f}   offer {rep['offer_wide']:.0f}")
    print(f"MULTI    reach {rep['reach_multi']:.4f}   offer {rep['offer_multi']:.0f}   gain {rep['gain']:+.4f}")
    print(f"NULL     reach {rep['reach_null']:.4f}   (coarse addresses permuted)   gain over null {rep['gain_over_null']:+.4f}")
    print(f"APART    only with the coarse layer {rep['multi_only']:.4f}   lost to the crowd {rep['lost']:.4f}")
    if rep['gain_over_null'] > 0.05:
        print(f"\nRESOLUTION IS A LEVER. Keeping every recurring width as an address reaches {rep['gain_over_null']:+.4f} beyond a coarse layer of the same shape with the wrong owners. The write path has been deleting the edges that connect narrow paradigms, and it is the same substitution relation - nothing new is assumed.")
    else:
        print('\nRESOLUTION IS NOT A LEVER: the coarse layer buys no more than a permuted one of the same size. One position, one address costs nothing, the fiat is vindicated, and the substitution lattice is as connected as it is going to get.')
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())