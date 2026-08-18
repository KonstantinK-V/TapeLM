"""COPY AT READ TIME. The one channel reading has never had.

375 closed the address atom: pooling frames raised coverage by +0.5..0.77 and moved the
unreachable share not at all - the missing truths barely REPEAT inside the window, and a count
cannot reach what was counted once. But "counted once at a place" is not "absent": a wiki
article repeats its subject in the running text constantly. 358 measured writing the document
into the TAPE (alive, small). Nobody measured whether the unreachable truth simply STANDS IN
THE NEIGHBOURING LINES of the question.

    copy(q, D) = truth in tokens(lines of q +- D), the question's own position excluded
    null       = the same test for a word from the truth's own frequency band (373's bands)

  Population: exactly 373/374's B - truth in neither own nor offer.
  GATE  copy - null > 0.05 at some D. Then the offer gains a copy lane - candidates from the
  +-D lines ranked by closeness, which is a count, holds no fact in the mind, and scales.
  D is swept in one pass: 0 (own line), 1, 4, 16.

    python _audit376_copy.py
    python _audit376_copy.py --corpus data/_morph_de.txt --window-lines 8000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage376_copy.json")
DS = (0, 1, 4, 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]

    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if not keep:
        print("no tape")
        return 1
    places = [list(ps) for _a, ps in keep]
    place_of, where = {}, defaultdict(list)
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
            where[toks[s]].append(s)
    line_toks = defaultdict(Counter)          # line -> counts over ALL tokens, not only kept
    for i, t in enumerate(toks):
        line_toks[owner[i]][t] += 1
    vocab = sorted(where)
    freq = {v: len(where[v]) for v in vocab}
    band_of = {v: freq[v].bit_length() for v in vocab}
    by_band = defaultdict(list)
    for v in vocab:
        by_band[band_of[v]].append(v)

    cooc = {}

    def offer_of(pid, truth):
        own = Counter(toks[x] for x in places[pid])
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        lens = list(own)[:6]
        if not lens:
            return set()
        ban = Counter(toks[x] for x in places[pid])
        off = Counter()
        for v in lens:
            c = cooc.get(v)
            if c is None:
                c = Counter()
                for s2 in where[v]:
                    for s3 in places[place_of[s2]]:
                        if toks[s3] != v:
                            c[toks[s3]] += 1
                cooc[v] = c
            for w, n in c.items():
                n -= ban.get(w, 0)
                if n > 0 and w != v:
                    off[w] += n
        return {w for w, _n in off.most_common(args.topm)}

    def band_draw(truth, banned):
        src, w = list(by_band[band_of[truth]]), 0
        while len(src) < 16 and w < 20:
            w += 1
            src += by_band.get(band_of[truth] - w, []) + by_band.get(band_of[truth] + w, [])
        for _t in range(64):
            v = src[rng.randrange(len(src))]
            if v != truth and v not in banned:
                return v
        return None

    def present(v, li, d, self_discount):
        """is v in lines li-d..li+d - the question's own standing subtracted once."""
        for l2 in range(max(0, li - d), li + d + 1):
            n = line_toks[l2].get(v, 0)
            if l2 == li and self_discount:
                n -= 1
            if n > 0:
                return True
        return False

    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    c = Counter()
    for s in qs:
        if c["n"] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in places[pid] if x != s}
        if not own:
            continue
        c["seen"] += 1
        off = offer_of(pid, truth)
        if truth in own or truth in off:
            continue
        c["n"] += 1
        li = owner[s]
        for d in DS:
            c[f"copy{d}"] += present(truth, li, d, True)
        rv = band_draw(truth, own | off)
        if rv is None:
            continue
        c["nn"] += 1
        for d in DS:
            c[f"null{d}"] += present(rv, li, d, False)

    n, nn = max(1, c["n"]), max(1, c["nn"])
    rep = {"corpus": args.corpus, "lines": len(lines), "places": len(places),
           "topm": args.topm, "seen": c["seen"], "unreachable": c["n"], "nulls": c["nn"]}
    print(f"tape     {len(places)} places   B population {c['n']} of {c['seen']} seen")
    best = (-1.0, None)
    for d in DS:
        cp, nu = c[f"copy{d}"] / n, c[f"null{d}"] / nn
        rep[f"copy_{d}"], rep[f"null_{d}"], rep[f"gain_{d}"] = cp, nu, cp - nu
        best = max(best, (cp - nu, d))
        print(f"D={d:<3d} copy {cp:.4f}   null {nu:.4f}   gain {cp - nu:+.4f}")
    rep["best_gain"], rep["best_d"] = best
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    if c["n"] < 400 or c["nn"] < 400:
        print("\nVOID - too few questions; widen --window-lines.")
    elif best[0] > 0.05:
        print(f"\nTHE TRUTH IS STANDING NEXT DOOR. On the holes no channel reaches, it is in "
              f"the +-{best[1]} lines {rep[f'copy_{best[1]}']:.4f} of the time against "
              f"{rep[f'null_{best[1]}']:.4f} for its frequency twin. Reading has been walking "
              f"the whole corpus and never LOOKING AT THE PAGE. The offer gains a copy lane - "
              f"nearby tokens ranked by closeness - a count, no fact in the mind.")
    else:
        print("\nNOT EVEN NEXT DOOR. The unreachable truths do not stand near their own holes "
              "more than frequency predicts. Nothing in the window carries them: the ceiling "
              "is the corpus's, and 0.339 is the tape's honest bound on it.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
