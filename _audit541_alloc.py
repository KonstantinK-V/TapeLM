"""541: the same total budget, spent somewhere else.

540 read the profile: the marginals are flat (r0/r1/r2 all ~0.35-0.43) and the
cumulative is steep (0.35 -> 0.67 -> 0.86 on the ~25% slice where anything hits).
So the gain is in HOW MANY slots a place gets, not in which candidate goes first,
and the whole 534->539 line was permuting equals inside a budget nobody questioned.

This arm does NOT sweep `allow`. A sweep would be arithmetic: cover is
|taken & held| / |held| and taken grows monotonically with allow, so allow -> inf
gives cover -> 1 and the gate returns GO without the tape saying anything. That is
the composition-dependent gate the project has hit six times.

Instead the TOTAL is frozen at what 511's law already spends, and only its
distribution changes:

    law       a_i = allow_of(v_i) = max(1, int(k*n/df))       the incumbent
    alloc     the same total, proportional to len(rec_i)      the arm
    uniform   the same total, evenly                          the null

Every arm sees the identical trials and spends the identical number of slots, so
cover is compared at equal budget and the arm can lose. `hops` is the number of
nodes actually offered - a place cannot spend slots it has no candidates for, and
that shows up as hops below its allocation rather than as hidden credit.

The arm must beat BOTH the law and the null, otherwise "reallocation helps" is
indistinguishable from "any reallocation helps".

Read `cov_alloc - cov_law` and `cov_alloc - cov_uni` together. And note before
reading anything: 540's 0.86 is conditional on the ~25% of places where any rank
hits at all, so the unconditional ceiling of this whole direction is about 0.21.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit511_ring import cheap_rec, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import allow_of
from _audit528_step import cover, trials
from _audit532_pool import slice_graph

OUT = Path("results/_stage541_alloc.json")
STORIES = "data/_tinystories_train.txt"


def nodes_with_allow(g, by, v, cache, allow, high_set):
    """v1_nodes of 527, with the budget passed in instead of computed."""
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    r1, seen = [], {v}
    for c in rec:
        if len(r1) >= allow:
            break
        if c in seen:
            continue
        seen.add(c)
        r1.append(c)
    if v in high_set:
        return r1
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
    return r1 + r2


def share_out(weights, total):
    """Largest-remainder split of `total` over `weights`, at least 1 each."""
    n = len(weights)
    if n == 0:
        return []
    base = [1] * n
    left = total - n
    if left <= 0:
        return base
    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * n
        wsum = float(n)
    exact = [left * w / wsum for w in weights]
    out = [b + int(e) for b, e in zip(base, exact)]
    rem = total - sum(out)
    order = sorted(range(n), key=lambda i: (exact[i] - int(exact[i]), weights[i]),
                   reverse=True)
    for i in range(rem):
        out[order[i % n]] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=250)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--lines", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=STORIES)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: args.lines]
    rng = random.Random(args.seed)
    graphs = []
    for _ in range(args.windows):
        g, _nL = slice_graph(pool, args.window_lines, rng, args.frame_max, args.min_fillers)
        if g is not None:
            graphs.append(g)
    if len(graphs) < 4:
        print("too few windows")
        return 1
    n_tr = max(2, int(0.7 * args.windows))
    test_g = graphs[n_tr:]

    rows = []
    rr = random.Random(args.seed + 17)
    for g in test_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        cache = {}
        for v, rest, held, maj in trials(g, by, mid, rr):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            n_rec = len([c for c in cheap_rec(g, by, v, cache) if c != v])
            by[v] = saved
            rows.append(dict(g=g, by=by, v=v, rest=rest, held=held,
                             high_set=high_set, cache=cache,
                             law=allow_of(g, v, k, high_set), n_rec=n_rec))

    n = len(rows)
    if n == 0:
        print("no trials")
        return 1
    total = sum(r["law"] for r in rows)
    a_law = [r["law"] for r in rows]
    a_alloc = share_out([float(r["n_rec"]) for r in rows], total)
    a_uni = share_out([1.0] * n, total)
    assert sum(a_alloc) == total and sum(a_uni) == total, "budgets not equal"

    arms = {}
    for name, alloc in (("law", a_law), ("alloc", a_alloc), ("uni", a_uni)):
        cov = hops = 0.0
        for r, a in zip(rows, alloc):
            g, by, v = r["g"], r["by"], r["v"]
            saved = by[v]
            by[v] = r["rest"]
            r["cache"].pop(v, None)
            nodes = nodes_with_allow(g, by, v, r["cache"], a, r["high_set"])
            by[v] = saved
            cov += cover(nodes, r["held"])
            hops += len(nodes)
        arms[name] = dict(cover=cov / n, hops=hops / n, slots=sum(alloc) / n)

    d_law = arms["alloc"]["cover"] - arms["law"]["cover"]
    d_uni = arms["alloc"]["cover"] - arms["uni"]["cover"]
    eff = {name: arms[name]["cover"] / max(arms[name]["hops"], 1e-9)
           for name in ("law", "alloc", "uni")}
    d_eff = eff["alloc"] - eff["law"]
    h_max = max(a["hops"] for a in arms.values())
    h_min = min(a["hops"] for a in arms.values())
    hop_gap = (h_max - h_min) / max(h_max, 1e-9)
    void = n < 40 or hop_gap > 0.05
    gate = (not void) and hop_gap <= 0.02 and d_law > 0.01 and d_uni > 0.01
    rec = dict(seed=args.seed, corpus=kind, windows=args.windows, n=n,
               total_slots=total, arms=arms, d_law=d_law, d_uni=d_uni,
               eff=eff, d_eff=d_eff,
               hop_gap=hop_gap, void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  trials {n}  total slots {total}")
    for name in ("law", "alloc", "uni"):
        a = arms[name]
        print(f"{name:6} cover {a['cover']:.4f}  hops {a['hops']:.3f}  "
              f"slots {a['slots']:.3f}")
    print(f"alloc - law {d_law:+.4f}   alloc - uni {d_uni:+.4f}   "
          f"hop gap {hop_gap:.4f}")
    print(f"eff/hop  law {eff['law']:.4f}  alloc {eff['alloc']:.4f}  "
          f"uni {eff['uni']:.4f}   alloc-law {d_eff:+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: too few trials, or the arms could not spend the same hops.")
    elif hop_gap > 0.02:
        print("\nHOPS NOT MATCHED. One arm cannot spend its slots; cover is not comparable.")
    elif d_uni <= 0.01:
        print("\nANY RESHUFFLE, NOT THIS ONE. len(rec) carries nothing over even "
              "spending.")
    elif d_law <= 0.01:
        print("\n511's 1/df spending is not beaten at equal budget. Allocation "
              "closed.")
    else:
        print("\nGO: the same budget spent by len(rec) beats both the law and the "
              "null.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
