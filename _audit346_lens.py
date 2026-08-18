"""Is there a PEAK to find? The ceiling of one lens and of two, measured before either is built.

WHY THIS EXISTS. Step 1 spent two levers and they failed in OPPOSITE directions:

    L1 raw count   answerable 0.0440   present@topm 0.1450   argmax goes to globally frequent
    L2 share       answerable 0.0040   present@topm 0.0365   argmax goes to SINGLETONS

Share is cooc(v,w)/total(w), so a value standing once in the whole tape and once beside the lens
scores a perfect 1.0. Count takes the frequent tail, share takes the singleton tail, and neither
is the truth. `chosen_share 0.13` in L1 already said it: THE CO-OCCURRENCE DISTRIBUTION AT A LENS
HAS NO PEAK. Not a wrong rule for picking the maximum - no maximum worth picking.

L3 (two lenses intersected) is the last named lever for step 1, and its mechanism is exactly
"sharpen a flat distribution". But if the tape is thin, two lenses intersect to NOTHING far more
often than they intersect to the truth, and L3 would be spent on an operation the tape cannot
support. 324 and 327 measured a ceiling before building; this does the same.

AND IT TESTS THE ONE THING THE CORPUS LEVER HAS NEVER BEEN ASKED. 335 swept the tape's WIDTH
(more places) and its DEPTH (more text per region) and found neither helps. It never asked for
THICKER PLACES: more mentions per place at a FIXED number of places. A flat co-occurrence
distribution is what thinness looks like from inside, so run this at two corpus sizes with
--addresses held fixed and read `support` - if the peak appears as places thicken, the corpus
lever is real for the first time in this project, and it is real for the constraint and not for
the walk.

    python _audit346_lens.py --bytes 30000000
    python _audit346_lens.py --bytes 120000000        # same --addresses: thicker, not wider
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage346_lens.json")
TOPM = 8            # CONS_TOPM: a lens's offer, matched to the walk's eight candidates
LENSES = 6          # CONS_LENSES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-questions", type=int, default=3000)
    ap.add_argument("--corpus", default=str(WIKI))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]

    rng = random.Random(args.seed)
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.window_lines:
        by_line = tframes._by_line(keep, owner)
        start = rng.randrange(max(1, len(lines)))
        acc = defaultdict(list)
        for d in range(args.window_lines):
            for k, i in by_line.get((start + d) % len(lines), ()):
                acc[k].append(i)
        keep = [(k, sorted(v)) for k, v in acc.items()
                if len({toks[i] for i in v}) >= args.min_fillers]
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print("no tape")
        return 1

    vals_at = [[toks[i] for i in ps] for _k, ps in keep]
    P = len(vals_at)
    bags = [Counter(v) for v in vals_at]
    where = defaultdict(list)                 # value -> [(place, count)]
    total = Counter()                         # value -> mentions anywhere
    for j, b in enumerate(bags):
        for v, c in b.items():
            where[v].append((j, c))
            total[v] += c

    # THE CO-OCCURRENCE TABLE, once. cooc[v][w] = how many mentions of w stand at places holding v
    cooc = {}

    def co(v):
        c = cooc.get(v)
        if c is None:
            c = Counter()
            for j, _n in where[v]:
                for w, cnt in bags[j].items():
                    c[w] += cnt
            cooc[v] = c
        return c

    qs = [(j, i) for j in range(P) for i in range(len(vals_at[j])) if len(vals_at[j]) >= 2]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]

    c = Counter()
    sizes, isizes = [], []
    for j, i in qs:
        truth = vals_at[j][i]
        own_c = Counter(vals_at[j])
        own_c[truth] -= 1
        if own_c[truth] <= 0:
            del own_c[truth]
        lens = list(own_c)[:LENSES]
        if not lens:
            continue
        c["n"] += 1
        c["in_own"] += truth in own_c
        here = bags[j]

        def resolved(v):
            """The lens's counter with THIS place taken out - the same subtraction the stage
            makes, and the reason a leak cannot flatter these numbers."""
            out = {}
            mine = here if any(pj == j for pj, _n in where[v]) else {}
            for w, n in co(v).items():
                if w == v:
                    continue
                n -= mine.get(w, 0)
                if n > 0:
                    out[w] = n
            return out

        rs = {v: resolved(v) for v in lens}
        sizes.append(sum(len(x) for x in rs.values()) / len(rs))
        # ---- ONE LENS: what L1 and L2 were choosing between
        best_count = best_share = None
        bc = bs = -1.0
        pres = False
        for v, r in rs.items():
            top_c = sorted(r.items(), key=lambda e: (-e[1], e[0]))[:TOPM]
            if truth in {w for w, _n in top_c}:
                pres = True
            for w, n in r.items():
                if n > bc:
                    bc, best_count = n, w
                sh = n / max(1, total[w])
                if sh > bs:
                    bs, best_share = sh, w
        c["one_present_topm"] += pres
        c["one_count_right"] += best_count == truth
        c["one_share_right"] += best_share == truth

        # ---- TWO LENSES INTERSECTED: L3's ceiling, before L3 is built
        hit_i = pres_i = nonempty = 0
        best_i, bi = None, -1.0
        for a in range(len(lens)):
            for b in range(a + 1, len(lens)):
                ra, rb = rs[lens[a]], rs[lens[b]]
                if not ra or not rb:
                    continue
                inter = {w: min(ra[w], rb[w]) for w in ra.keys() & rb.keys()}
                if not inter:
                    continue
                nonempty += 1
                isizes.append(len(inter))
                top_i = sorted(inter.items(), key=lambda e: (-e[1], e[0]))[:TOPM]
                if truth in {w for w, _n in top_i}:
                    pres_i = 1
                for w, n in inter.items():
                    if n > bi:
                        bi, best_i = n, w
        c["pair_nonempty"] += nonempty > 0
        c["pair_present_topm"] += pres_i
        c["pair_count_right"] += best_i == truth
        c["pair_pairs"] += nonempty

    # ---- THICKNESS: is there anything for a peak to be made of --------------------------
    sup = Counter()
    for v in list(cooc):
        for w, n in cooc[v].items():
            if w != v:
                sup[n] += 1
    tot_pairs = max(1, sum(sup.values()))
    n = max(1, c["n"])
    rep = {
        "bytes": args.bytes, "corpus": args.corpus, "places": P, "questions": c["n"],
        "mentions_per_place": sum(len(v) for v in vals_at) / P,
        "mean_lens_offer": sum(sizes) / max(1, len(sizes)),
        "in_own": c["in_own"] / n,
        # ONE LENS - what step 1 measured, reproduced here without a mind
        "one_present_topm": c["one_present_topm"] / n,
        "one_count_right": c["one_count_right"] / n,
        "one_share_right": c["one_share_right"] / n,
        # TWO LENSES - L3's ceiling
        "pair_nonempty": c["pair_nonempty"] / n,
        "pair_present_topm": c["pair_present_topm"] / n,
        "pair_count_right": c["pair_count_right"] / n,
        "pair_mean_size": (sum(isizes) / len(isizes)) if isizes else float("nan"),
        "pairs_per_question": c["pair_pairs"] / n,
        # THICKNESS - the share of co-occurring pairs seen more than once. A distribution made
        # of singletons cannot have a peak, and no rule for taking its maximum will invent one.
        "support_1": sup[1] / tot_pairs,
        "support_2plus": sum(v for k, v in sup.items() if k >= 2) / tot_pairs,
        "support_3plus": sum(v for k, v in sup.items() if k >= 3) / tot_pairs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"tape    {P} places, {rep['mentions_per_place']:.2f} mentions each, "
          f"{c['n']} questions, in_own {rep['in_own']:.4f}")
    print(f"THICK   co-occurring pairs seen once {rep['support_1']:.4f}   "
          f"twice or more {rep['support_2plus']:.4f}   three or more {rep['support_3plus']:.4f}")
    print(f"ONE     present@{TOPM} {rep['one_present_topm']:.4f}   "
          f"count-argmax {rep['one_count_right']:.4f}   "
          f"share-argmax {rep['one_share_right']:.4f}   offer {rep['mean_lens_offer']:.1f}")
    print(f"TWO     non-empty {rep['pair_nonempty']:.4f}   present@{TOPM} "
          f"{rep['pair_present_topm']:.4f}   count-argmax {rep['pair_count_right']:.4f}   "
          f"size {rep['pair_mean_size']:.1f}   pairs/q {rep['pairs_per_question']:.1f}")
    # THE READING, printed rather than left to the eye
    if rep["support_2plus"] < 0.10:
        print("\nNO PEAK TO FIND: over 90% of co-occurring pairs are seen exactly once, so the "
              "distribution a lens resolves is made of singletons. No rule for taking its "
              "maximum can work, and L3 would be spent on an operation the tape cannot "
              "support. Thicken the places before building it.")
    elif rep["pair_present_topm"] > rep["one_present_topm"]:
        print("\nTWO LENSES SHARPEN IT: the intersection reaches more of the truth than one "
              "lens does. L3 is worth a lever.")
    else:
        print("\nTWO LENSES DO NOT SHARPEN IT: the intersection reaches no more than one lens, "
              "so L3's mechanism does not hold on this tape.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
