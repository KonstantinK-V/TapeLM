"""How much of the truth is ON THE TAPE, against how much the walk reaches.

WHY THIS COULD OVERTURN A CONCLUSION WE HAVE ALREADY DRAWN. 310 read the composition failure as
the corpus's fault: strict pairs carry no joint signal, so the product of marginals is correct
and no mind should beat it. That reading rests on a ceiling of 0.17 - but 0.17 is the ceiling OF
OUR OFFER, eight candidates from eight places, not the ceiling of the tape. A hidden truth is a
filler, and a filler usually stands somewhere else too.

So: what share of hidden truths is present ANYWHERE on the tape, and what share does the walk
actually put in front of the mind? If presence is far above reach, the binding constraint is our
RETRIEVAL, not the corpus, and 310 has to be re-read - "there was no signal" becomes "we could
not get to it", which is a different and far more tractable problem.

The distances are nested, so the number to read is where the mass stops growing:
  own      - the question's own place, other rows. What a lookup gets for free.
  walk K   - the eight nearest places by fingerprint. What the mind is offered today.
  walk 2K  - the same compass, twice as far. Separates "our cap is tight" from "the direction
             is wrong": if 2K barely beats K, walking further is not the answer.
  shared   - places holding a filler in common, exactly. 323's second compass.
  any      - anywhere on the tape. The tape's own ceiling, and the number 310 assumed was 0.17.

    python _audit327_presence.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage327_presence.json")
K, CANDS, DIM = 8, 8, 64


def fp(s: str, dim: int = DIM):
    v = [0.0] * dim
    t = f"  {s}  "
    for i in range(len(t) - 2):
        h = hashlib.blake2b(t[i:i + 3].encode("utf-8"), digest_size=8).digest()
        v[int.from_bytes(h[:4], "big") % dim] += 1.0
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--sample", choices=("uniform", "region"), default="region")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-questions", type=int, default=3000)
    args = ap.parse_args()

    text = WIKI.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]

    rng = random.Random(args.seed)
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.sample == "region":
        if args.window_lines:
            by_line = tframes._by_line(keep, owner)
            start = rng.randrange(max(1, len(lines)))
            acc = defaultdict(list)
            for d in range(args.window_lines):
                for k, i in by_line.get((start + d) % len(lines), ()):
                    acc[k].append(i)
            keep = [(k, sorted(v)) for k, v in acc.items()
                    if len({toks[i] for i in v}) >= args.min_fillers]
            # THE ADDRESS CAP APPLIES HERE TOO, and 335 is why it is spelled out. With
            # --window-lines set this branch built `keep` from the window and never read
            # --addresses at all, so a width sweep from 375 to 6000 returned the same numbers to
            # four decimals - identical output over a 16x range, which is the tell that a flag
            # is dead rather than that a quantity is scale-invariant. SAMPLED, never truncated:
            # a deterministic prefix is 298's lesson and would make width mean "the widest
            # frames" instead of "this many frames".
            if args.addresses and len(keep) > args.addresses:
                keep = rng.sample(keep, args.addresses)
        else:
            keep = tframes.frame_region(keep, toks, owner, len(lines), args.addresses, rng,
                                        args.min_fillers)
    elif args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print("no tape")
        return 1

    vals_at = [[toks[i] for i in ps] for _k, ps in keep]
    P = len(vals_at)
    bags = [Counter(v) for v in vals_at]
    where = defaultdict(set)
    for j, b in enumerate(bags):
        for v in b:
            where[v].add(j)
    fps = []
    for b in bags:
        acc = [0.0] * DIM
        for v, c in b.items():
            f = fp(v)
            for i in range(DIM):
                acc[i] += c * f[i]
        n = sum(x * x for x in acc) ** 0.5
        fps.append([x / n for x in acc] if n else acc)

    qs = [(j, i) for j in range(P) for i in range(len(vals_at[j])) if len(vals_at[j]) >= 2]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]

    c = Counter()
    for j, i in qs:
        truth = vals_at[j][i]
        own = Counter(vals_at[j])
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        c["n"] += 1
        in_own = truth in own
        c["own"] += in_own

        acc = [0.0] * DIM
        for v, cnt in own.items():
            f = fp(v)
            for d in range(DIM):
                acc[d] += cnt * f[d]
        nn = sum(x * x for x in acc) ** 0.5
        qv = [x / nn for x in acc] if nn else acc
        sims = sorted(((sum(a * b for a, b in zip(qv, fps[o])), o)
                       for o in range(P) if o != j), reverse=True)

        def offered(places):
            seen, out = set(), []
            for o in places:
                for v in vals_at[o]:
                    if v not in seen:
                        seen.add(v)
                        out.append(v)
            return set(out[:CANDS])

        k1 = offered([o for _s, o in sims[:K]])
        k2 = offered([o for _s, o in sims[:2 * K]])
        share = Counter()
        for v, cnt in own.items():
            for o in where[v]:
                if o != j:
                    share[o] += cnt
        ks = offered([o for o, _n in share.most_common(K)])
        # PRESENT ANYWHERE: the value stands at some OTHER place, or again at this one.
        # `where` counts places, so a truth unique to this one mention has |where| == 1.
        anywhere = len(where.get(truth, ())) > 1 or in_own

        c["walk_k"] += truth in k1
        c["walk_2k"] += truth in k2
        c["shared"] += truth in ks
        c["union"] += truth in (k1 | ks)
        c["anywhere"] += anywhere
        # THE GAP THAT DECIDES: on the tape, but not in anything we put in front of the mind
        c["present_unreached"] += anywhere and truth not in (k2 | ks) and not in_own

    n = max(1, c["n"])
    rep = {"bytes": args.bytes, "sample": args.sample, "window_lines": args.window_lines,
           "places": P, "questions": c["n"],
           "own": c["own"] / n, "walk_k": c["walk_k"] / n, "walk_2k": c["walk_2k"] / n,
           "shared": c["shared"] / n, "union_k_shared": c["union"] / n,
           # THE TAPE'S OWN CEILING, against the 0.17 that 310's reading assumed
           "anywhere": c["anywhere"] / n,
           "present_unreached": c["present_unreached"] / n,
           "reach_over_presence": (c["walk_k"] / max(1, c["anywhere"]))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"tape    {P} places, {c['n']} questions")
    print(f"nested  own {rep['own']:.4f}  walk K {rep['walk_k']:.4f}  walk 2K "
          f"{rep['walk_2k']:.4f}  shared {rep['shared']:.4f}  union {rep['union_k_shared']:.4f}")
    print(f"TAPE    present anywhere {rep['anywhere']:.4f}   "
          f"the walk shows the mind {rep['reach_over_presence']:.4f} of it")
    print(f"GAP     on the tape and reached by nothing we offer: "
          f"{rep['present_unreached']:.4f}")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
