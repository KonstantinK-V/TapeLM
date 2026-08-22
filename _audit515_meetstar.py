"""515: MEET is a new star center, not next-read of v."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _tape_frames as tframes
from _audit511_ring import cheap_rec, mentions, walk

OUT = Path("results/_stage515_meetstar.json")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")


def pick_corpus(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"no corpus {p}")
        kind = "wiki" if "wiki" in p.name.lower() else "given"
        return p, kind, 80 if kind == "wiki" else 20
    if WIKI.exists():
        return WIKI, "wiki", 80
    if FALLBACK.exists():
        return FALLBACK, "tinystories-fallback", 20
    raise SystemExit("no corpus")


def graph(lines, frame_max, min_fillers):
    keep, toks, owner = tframes.frame_keep(lines, frame_max, min_fillers)
    if not keep:
        return None
    value, keys = [], []
    for (w, left, right), ps in keep:
        ks = tuple(x for x in list(left) + list(right) if x)
        for i in ps:
            value.append(toks[i])
            keys.append(ks)
    n = len(value)
    df = Counter()
    for s in range(n):
        df[value[s]] += 1
        for k in keys[s]:
            df[k] += 1
    return dict(n=n, value=value, keys=keys, df=df)


def meet_nodes(g, by, nodes, v, cache):
    if len(nodes) < 2:
        return set()
    sets = [set(cheap_rec(g, by, c, cache)) for c in nodes]
    found = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            meet = sets[i] & sets[j]
            meet.discard(v)
            meet -= set(nodes)
            found |= meet
    return found


def rings(g, by, v, cache):
    allow = max(1, int(200 / max(g["df"][v], 1)))
    r1, seen = [], {v}
    for c in cheap_rec(g, by, v, cache):
        if len(r1) >= allow:
            break
        if c in seen:
            continue
        seen.add(c)
        r1.append(c)
    remain = allow - len(r1)
    r2, frontier = [], list(r1)
    while remain > 0 and frontier:
        nxt = []
        for a in frontier:
            if remain <= 0:
                break
            for c in cheap_rec(g, by, a, cache):
                if remain <= 0:
                    break
                if c in seen:
                    continue
                seen.add(c)
                r2.append(c)
                nxt.append(c)
                remain -= 1
        frontier = nxt
        if not nxt:
            break
    marks = meet_nodes(g, by, r1 + r2, v, cache)
    return r1, r2, marks


def mean_walk(g, by, vs, cache):
    rows = [walk(g, by, v, cache) for v in vs]
    n = max(len(rows), 1)
    return dict(
        n=len(rows),
        d1=sum(r["d1"] for r in rows) / n,
        d2=sum(r["d2"] for r in rows) / n,
        m2=sum(r["m2"] for r in rows) / n,
    )


def eval_mid(g, by, vs, rng):
    cache = {}
    meets, ctrls = [], []
    n_src = 0
    for v in vs:
        r1, r2, marks = rings(g, by, v, cache)
        if not marks or not r2:
            continue
        n_src += 1
        m = rng.choice(sorted(marks))
        c = rng.choice(r2)
        meets.append(m)
        ctrls.append(c)
    return n_src, mean_walk(g, by, meets, cache), mean_walk(g, by, ctrls, cache)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    L = args.window_lines
    if L < len(pool):
        s0 = rng.randrange(len(pool) - L + 1)
        lines = pool[s0:s0 + L]
    else:
        lines = pool
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        print("no tape")
        return 1
    by = mentions(g)
    dfn = {v: len(sl) for v, sl in by.items()}
    mid = [v for v, d in dfn.items() if 8 <= d <= 30]
    n_src, meet_rep, r2_rep = eval_mid(g, by, mid, random.Random(args.seed + 1))
    void = meet_rep["n"] < 20
    gate = (not void) and (meet_rep["d2"] > r2_rep["d2"]) and (
        meet_rep["m2"] - r2_rep["m2"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines), n_src=n_src,
               meet=meet_rep, r2=r2_rep, void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  sources {n_src}")
    print(f"MEET-star n {meet_rep['n']}  d1 {meet_rep['d1']:.2f} d2 {meet_rep['d2']:.2f}  "
          f"m2 {meet_rep['m2']:.3f}")
    print(f"R2-star   n {r2_rep['n']}  d1 {r2_rep['d1']:.2f} d2 {r2_rep['d2']:.2f}  "
          f"m2 {r2_rep['m2']:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough MEET seeds.")
    elif gate:
        print("\nGO MEET-STAR: a meeting node is a richer star center than a random ring2 node.")
    else:
        print("\nSTOP: MEET is not a better new center than any ring2 leftover.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
