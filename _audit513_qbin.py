"""513: Q on (df_bin, go/stop). Not Q[H]. Separate from 512."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, mentions
import _audit511_ring as R511

OUT = Path("results/_stage513_qbin.json")


def df_bin(d):
    if d <= 30:
        return "mid"
    if d <= 80:
        return "midhi"
    return "high"


def meets_delta(g, by, nodes, v, cache, seen_m):
    if len(nodes) < 2:
        return 0
    sets = [set(cheap_rec(g, by, c, cache)) for c in nodes]
    n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            meet = sets[i] & sets[j]
            meet.discard(v)
            meet -= set(nodes)
            for m in meet:
                if m not in seen_m:
                    seen_m.add(m)
                    n += 1
    return n


def run_policy(g, by, v, cache, choose, max_h=12):
    seen = {v}
    nodes = []
    seen_m = set()
    hops = 0
    m = 0
    cur = v
    while hops < max_h:
        b = df_bin(g["df"][cur])
        opts = [c for c in cheap_rec(g, by, cur, cache) if c not in seen]
        go = choose(b, bool(opts))
        if not go or not opts:
            break
        nxt = opts[0]
        seen.add(nxt)
        nodes.append(nxt)
        hops += 1
        m += meets_delta(g, by, nodes, v, cache, seen_m)
        cur = nxt
    return dict(hops=hops, d2=max(hops - 2, 0), meets=m)


def mean_run(g, by, vs, cache, choose):
    rows = [run_policy(g, by, v, cache, choose) for v in vs]
    n = max(len(rows), 1)
    return dict(
        n=len(rows),
        hops=sum(r["hops"] for r in rows) / n,
        d2=sum(r["d2"] for r in rows) / n,
        meets=sum(r["meets"] for r in rows) / n,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = R511.pick_corpus(args.corpus or None)
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
    g = R511.graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        print("no tape")
        return 1
    by = mentions(g)
    dfn = {v: len(sl) for v, sl in by.items()}
    mid = [v for v, d in dfn.items() if 8 <= d <= 30]
    high = [v for v, d in dfn.items() if d > 80]
    rng.shuffle(mid)
    rng.shuffle(high)
    cut_m, cut_h = int(0.6 * len(mid)), int(0.6 * len(high))
    train = mid[:cut_m] + high[:cut_h]
    test_m, test_h = mid[cut_m:], high[cut_h:]
    Q = defaultdict(float)
    cache = {}
    lr = 0.2
    for _ in range(args.epochs):
        rng.shuffle(train)
        for v in train:
            seen = {v}
            nodes = []
            seen_m = set()
            cur = v
            for _h in range(12):
                b = df_bin(g["df"][cur])
                opts = [c for c in cheap_rec(g, by, cur, cache) if c not in seen]
                qg, qs = Q[(b, "go")], Q[(b, "stop")]
                if rng.random() < args.eps:
                    go = rng.choice([True, False])
                else:
                    go = qg >= qs
                if not go or not opts:
                    Q[(b, "stop")] += lr * (0.0 - Q[(b, "stop")])
                    break
                nxt = opts[0]
                seen.add(nxt)
                nodes.append(nxt)
                dm = meets_delta(g, by, nodes, v, cache, seen_m)
                R = 0.3 * dm - 0.05
                Q[(b, "go")] += lr * (R - Q[(b, "go")])
                cur = nxt

    def learned(b, has):
        return has and Q[(b, "go")] >= Q[(b, "stop")]

    def always(b, has):
        return has

    lm = mean_run(g, by, test_m, cache, learned)
    lh = mean_run(g, by, test_h, cache, learned)
    ah = mean_run(g, by, test_h, cache, always)
    am = mean_run(g, by, test_m, cache, always)
    void = lm["n"] < 15 or lh["n"] < 5
    gate = (not void) and (lh["d2"] < 1.5) and (lm["d2"] > lh["d2"] + 1) and (
        lh["d2"] < ah["d2"] - 0.5)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               Q={f"{k[0]}_{k[1]}": v for k, v in Q.items()},
               learn_mid=lm, learn_high=lh, always_mid=am, always_high=ah,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}")
    print(f"LEARN mid n {lm['n']} hops {lm['hops']:.2f} d2 {lm['d2']:.2f} m {lm['meets']:.3f}")
    print(f"LEARN high n {lh['n']} hops {lh['hops']:.2f} d2 {lh['d2']:.2f} m {lh['meets']:.3f}")
    print(f"ALWAYS mid d2 {am['d2']:.2f}  high d2 {ah['d2']:.2f}")
    print("Q", {k: round(v, 3) for k, v in rec["Q"].items()})
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: tiny test split.")
    elif gate:
        print("\nGO QBIN: held-out mid keeps walking; high stops; not the flood policy.")
    else:
        print("\nSTOP: df_bin Q did not learn to keep физика and cut and.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
