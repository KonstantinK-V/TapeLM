"""532: 531 layers on MANY W=250 slices. Not a longer window."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import majority, v1_nodes
from _audit528_step import cover, mean_ep, run_ep, trials
from _audit531_layer import layer_of

OUT = Path("results/_stage532_pool.json")


def slice_graph(pool, L, rng, frame_max, min_fillers):
    if L >= len(pool):
        lines = pool
    else:
        s0 = rng.randrange(len(pool) - L + 1)
        lines = pool[s0:s0 + L]
    return graph(lines, frame_max, min_fillers), len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=250)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--lines", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: args.lines]
    rng = random.Random(args.seed)
    L = args.window_lines
    n_tr = max(2, int(0.7 * args.windows))
    graphs = []
    for _ in range(args.windows):
        g, nL = slice_graph(pool, L, rng, args.frame_max, args.min_fillers)
        if g is None:
            continue
        graphs.append((g, nL))
    if len(graphs) < 4:
        print("too few windows")
        return 1
    train_g, test_g = graphs[:n_tr], graphs[n_tr:]
    Q = defaultdict(float)
    lr, cache = 0.2, {}
    for g, _nL in train_g:
        by = mentions(g)
        mid, high, p25, p75 = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        train_v = list(mid) + list(high)
        for _ in range(args.epochs):
            rng.shuffle(train_v)
            for v, rest, held, maj in trials(g, by, train_v, rng):
                saved = by[v]
                by[v] = rest
                cache.pop(v, None)
                nodes = v1_nodes(g, by, v, cache, k, high_set)
                by[v] = saved
                if not nodes:
                    continue
                band = "high" if v in high_set else "mid"
                seen = set()
                if nodes[0] in held:
                    seen.add(nodes[0])
                # hop1 never updates Q2/Q3
                for i, c in enumerate(nodes[1:], start=1):
                    Lyr = layer_of(i)
                    residual = c in held and c not in seen and c != maj
                    r = 1.0 if residual else 0.0
                    Q[(band, Lyr)] += lr * (r - Q[(band, Lyr)])
                    go = rng.choice([True, False]) if rng.random() < args.eps else Q[(band, Lyr)] > 0.05
                    if not go:
                        break
                    if c in held:
                        seen.add(c)

    def run_learned(nodes, held, maj, band):
        if not nodes:
            return dict(hops=0, cover=0.0, go2=0, go3=0)
        take = [nodes[0]]
        g2 = g3 = 0
        for i, c in enumerate(nodes[1:], start=1):
            Lyr = layer_of(i)
            if Q[(band, Lyr)] <= 0.05:
                break
            take.append(c)
            if Lyr == 2:
                g2 = 1
            else:
                g3 = 1
        return dict(hops=len(take), cover=cover(take, held), go2=g2, go3=g3)

    def collect(gs, kind_):
        rows = []
        rr = random.Random(args.seed + 11)
        for g, _nL in gs:
            by = mentions(g)
            mid, high, _, _ = pct_band(g, by)
            k = 200.0 / max(g["n"], 1)
            high_set = set(high)
            vs = mid if kind_ != "high" else high
            choose_kind = "learn" if kind_ == "high" else kind_
            for v, rest, held, maj in trials(g, by, vs, rr):
                saved = by[v]
                by[v] = rest
                cache.pop(v, None)
                nodes = v1_nodes(g, by, v, cache, k, high_set)
                by[v] = saved
                band = "high" if v in high_set else "mid"
                if choose_kind == "learn":
                    rows.append(run_learned(nodes, held, maj, band))
                else:
                    choose = (lambda _b: True) if choose_kind == "all" else (lambda _b: False)
                    rows.append(run_ep(nodes, held, maj, choose, band))
        out = mean_ep(rows)
        if rows and kind_ in ("learn", "high"):
            n = max(len(rows), 1)
            out["go2"] = sum(r.get("go2", 0) for r in rows) / n
            out["go3"] = sum(r.get("go3", 0) for r in rows) / n
        return out

    lm = collect(test_g, "learn")
    lh = collect(test_g, "high")
    a1 = collect(test_g, "hop1")
    ag = collect(test_g, "all")
    void = lm["n"] < 20
    mid_between = a1["hops"] + 0.3 < lm["hops"] < ag["hops"] - 0.3
    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and mid_between and (
        lh["hops"] < 1.5)
    rec = dict(seed=args.seed, corpus=kind, windows=len(graphs),
               n_train=len(train_g), n_test=len(test_g), W=L,
               Q={f"{a}_{Lyr}": v for (a, Lyr), v in Q.items()},
               learn_mid=lm, learn_high=lh, hop1_mid=a1, allgo_mid=ag,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  windows {len(graphs)} ({len(train_g)}/{len(test_g)})  W {L}")
    print(f"LEARN mid n {lm['n']} hops {lm['hops']:.2f} cover {lm['cover']:.3f} "
          f"go2 {lm.get('go2', 0):.2f} go3 {lm.get('go3', 0):.2f}")
    print(f"LEARN high n {lh['n']} hops {lh['hops']:.2f} cover {lh['cover']:.3f}")
    print(f"HOP1  mid hops {a1['hops']:.2f} cover {a1['cover']:.3f}")
    print(f"ALLGO mid hops {ag['hops']:.2f} cover {ag['cover']:.3f}")
    print("Q", {k: round(v, 3) for k, v in rec["Q"].items()})
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: tiny pooled test.")
    elif gate:
        print("\nGO POOL: more slices made layer-2 residual usable. 529 kept.")
    else:
        print("\nSTOP: mass did not lift residual. Keep 529; data was not the knob.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
