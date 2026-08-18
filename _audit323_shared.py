"""Is the ink a lossy copy of something the tape already counts exactly?

THE QUESTION. The walk goes to places whose FINGERPRINTS are near - a hash over the bag of
fillers, so an approximation. But the tape holds an exact relation the ink can only imitate:
two places are linked when the SAME FILLER was written in both holes, and the tape knows how
many times. That is a count, so by this project's own invariant it belongs in the write path,
and interpolation belongs only where the mind reads.

If an exact shared-filler walk reaches the truth as often as the cosine walk, the ink is a
lossy copy of a countable relation and should be replaced by it. If the cosine reaches MORE,
the ink is generalising past what is written - which is a finding of the opposite sign and
just as useful, because it says the approximation earns its place.

This is 310's discipline: measure the substrate before building anything that stands on it.

    python _audit323_shared.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
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
OUT = Path("results/_stage323_shared.json")
K = 8            # places a walk may reach - REACH_K, so the two walks are budget-matched
CANDS = 8        # fillers it may score - REACH_CANDS, same reason
DIM = 64         # width of the stand-in fingerprint


def fp(s: str, dim: int = DIM):
    """The ink, standing in for bank.fp: blake2b over character trigrams, nothing trained.

    Not the stage's exact encoder, and it does not need to be - what is being compared is
    A HASH OF THE FILLER BAG against EXACT FILLER SHARING. Any pure hash of the same bag has
    the same character: it collides, it cannot count, and it cannot be inspected. If the
    conclusion depended on which hash, it would not be a conclusion about approximation.
    """
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

    names, vals_at = [], []
    for (w, left, right), ps in keep:
        names.append(f"{' '.join(left)}|{' '.join(right)}")
        vals_at.append([toks[i] for i in ps])
    P = len(names)

    # THE TWO COMPASSES, built from the SAME bag of fillers so nothing but exactness differs.
    bags = [Counter(v) for v in vals_at]
    fps = []
    for b in bags:
        acc = [0.0] * DIM
        for v, c in b.items():
            f = fp(v)
            for i in range(DIM):
                acc[i] += c * f[i]
        n = sum(x * x for x in acc) ** 0.5
        fps.append([x / n for x in acc] if n else acc)
    # EXACT: value -> the places that ever held it, so sharing is a lookup and not a scan
    where = defaultdict(set)
    for j, b in enumerate(bags):
        for v in b:
            where[v].add(j)

    qs = [(j, i) for j in range(P) for i in range(len(vals_at[j])) if len(vals_at[j]) >= 2]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]

    c = Counter()
    rank_cos, rank_sh = [], []
    for j, i in qs:
        truth = vals_at[j][i]
        rest = Counter(vals_at[j])
        rest[truth] -= 1                      # THE HIDDEN MENTION COMES OUT OF THE QUERY BAG
        if rest[truth] <= 0:
            del rest[truth]
        c["n"] += 1
        c["in_own"] += truth in rest

        # the cosine walk, with the hidden mention subtracted from the query fingerprint
        acc = [0.0] * DIM
        for v, cnt in rest.items():
            f = fp(v)
            for d in range(DIM):
                acc[d] += cnt * f[d]
        nn = sum(x * x for x in acc) ** 0.5
        qv = [x / nn for x in acc] if nn else acc
        sims = [(sum(a * b for a, b in zip(qv, fps[o])), o) for o in range(P) if o != j]
        sims.sort(reverse=True)
        cos_places = [o for _s, o in sims[:K]]

        # the exact walk: places sharing the most fillers with this one, hidden mention removed.
        # Ties break by the shared COUNT, then by place order - all counts, nothing fitted.
        share = Counter()
        for v, cnt in rest.items():
            for o in where[v]:
                if o != j:
                    share[o] += cnt
        sh_places = [o for o, _n in share.most_common(K)]

        def offer(places):
            seen, out = set(), []
            for o in places:
                for v in vals_at[o]:
                    if v not in seen:
                        seen.add(v)
                        out.append(v)
            return out[:CANDS]

        oc, os_ = offer(cos_places), offer(sh_places)
        c["cos_reach"] += truth in set(oc)
        c["share_reach"] += truth in set(os_)
        c["both"] += (truth in set(oc)) and (truth in set(os_))
        c["cos_only"] += (truth in set(oc)) and (truth not in set(os_))
        c["share_only"] += (truth in set(os_)) and (truth not in set(oc))
        # WHERE THE TRUTH SITS IN EACH OFFER, uncapped, so a cap cannot hide a difference
        full_c, full_s = offer(cos_places + []), offer(sh_places + [])
        rank_cos.append(full_c.index(truth) + 1 if truth in full_c else 0)
        rank_sh.append(full_s.index(truth) + 1 if truth in full_s else 0)
        c["overlap"] += len(set(cos_places) & set(sh_places))

    n = max(1, c["n"])
    d = max(1, c["cos_only"] + c["share_only"])
    rep = {"bytes": args.bytes, "sample": args.sample, "window_lines": args.window_lines,
           "places": P, "questions": c["n"], "own_hit": c["in_own"] / n,
           "cos_reach": c["cos_reach"] / n, "share_reach": c["share_reach"] / n,
           "both": c["both"] / n, "cos_only": c["cos_only"], "share_only": c["share_only"],
           # THE NUMBER THAT DECIDES. Paired: only the questions where the two disagree carry
           # any contrast, exactly as McNemar everywhere else in this project.
           "paired_z": ((c["cos_only"] - c["share_only"]) / (d ** 0.5)),
           "places_overlap_mean": c["overlap"] / n / K,
           "rank_cos_mean": sum(rank_cos) / n, "rank_share_mean": sum(rank_sh) / n}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"tape    {P} places, {c['n']} questions, own_hit {rep['own_hit']:.4f}")
    print(f"reach   cosine {rep['cos_reach']:.4f}   exact-shared {rep['share_reach']:.4f}   "
          f"both {rep['both']:.4f}")
    print(f"PAIRED  cosine-only {c['cos_only']}  shared-only {c['share_only']}   "
          f"z {rep['paired_z']:+.2f}   (positive = the ink beats the count)")
    print(f"places  the two walks pick the same place {rep['places_overlap_mean']:.4f} of the time")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
