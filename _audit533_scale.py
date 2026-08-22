"""533: hop1 on local W=250, hop2+ on pooled train tape.

hop1 never trains Q2/Q3. Residual = in LOCAL held, not in hop1, != maj.
Test hop2 graph does not contain the test window.
Gate = 531 (no miracle bar). 529 kept.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import majority, v1_nodes
from _audit528_step import cover, trials
from _audit531_layer import layer_of

OUT = Path("results/_stage533_scale.json")


def take_slice(pool, L, rng):
    if L >= len(pool):
        return list(pool)
    s0 = rng.randrange(len(pool) - L + 1)
    return pool[s0:s0 + L]


def pack(lines, frame_max, min_fillers):
    g = graph(lines, frame_max, min_fillers)
    if g is None:
        return None
    by = mentions(g)
    mid, high, p25, p75 = pct_band(g, by)
    k = 200.0 / max(g["n"], 1)
    return dict(g=g, by=by, mid=mid, high=high, high_set=set(high),
                k=k, p25=p25, p75=p75)


def path_of(pack_, v, cache):
    if v not in pack_["by"]:
        return []
    cache.pop(v, None)
    return v1_nodes(pack_["g"], pack_["by"], v, cache, pack_["k"], pack_["high_set"])


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
    slices = [take_slice(pool, L, rng) for _ in range(args.windows)]
    n_tr = max(2, int(0.7 * args.windows))
    tr_s, te_s = slices[:n_tr], slices[n_tr:]
    locals_tr = [p for p in (pack(s, args.frame_max, args.min_fillers) for s in tr_s) if p]
    locals_te = [p for p in (pack(s, args.frame_max, args.min_fillers) for s in te_s) if p]
    big_lines = [ln for s in tr_s for ln in s]
    big = pack(big_lines, args.frame_max, args.min_fillers)
    if not locals_tr or not locals_te or big is None:
        print("no tape")
        return 1
    Q = defaultdict(float)
    lr = 0.2
    cache_l, cache_b = {}, {}
    for loc in locals_tr:
        train_v = list(loc["mid"]) + list(loc["high"])
        for _ in range(args.epochs):
            rng.shuffle(train_v)
            for v, rest, held, maj in trials(loc["g"], loc["by"], train_v, rng):
                saved = loc["by"][v]
                loc["by"][v] = rest
                local_nodes = path_of(loc, v, cache_l)
                loc["by"][v] = saved
                if not local_nodes:
                    continue
                hop1 = local_nodes[0]
                pool_nodes = [c for c in path_of(big, v, cache_b) if c != hop1]
                nodes = [hop1] + pool_nodes
                band = "high" if v in loc["high_set"] else "mid"
                seen = set()
                if hop1 in held:
                    seen.add(hop1)
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

    def run_ep(local_nodes, pool_nodes, held, maj, band, take_all=None):
        if not local_nodes:
            return dict(hops=0, cover=0.0, go2=0, go3=0)
        hop1 = local_nodes[0]
        rest = [c for c in pool_nodes if c != hop1]
        nodes = [hop1] + rest
        if take_all is False:
            take = [hop1]
            return dict(hops=len(take), cover=cover(take, held), go2=0, go3=0)
        if take_all is True:
            take = nodes
            return dict(hops=len(take), cover=cover(take, held), go2=int(len(rest) > 0),
                        go3=int(len(rest) > 1))
        take = [hop1]
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

    def collect(locs, mode):
        rows = []
        rr = random.Random(args.seed + 13)
        for loc in locs:
            vs = loc["high"] if mode == "high" else loc["mid"]
            take_all = False if mode == "hop1" else True if mode == "all" else None
            for v, rest, held, maj in trials(loc["g"], loc["by"], vs, rr):
                saved = loc["by"][v]
                loc["by"][v] = rest
                local_nodes = path_of(loc, v, cache_l)
                loc["by"][v] = saved
                pool_nodes = path_of(big, v, cache_b)
                band = "high" if v in loc["high_set"] else "mid"
                rows.append(run_ep(local_nodes, pool_nodes, held, maj, band, take_all))
        n = max(len(rows), 1)
        return dict(n=len(rows),
                    hops=sum(r["hops"] for r in rows) / n,
                    cover=sum(r["cover"] for r in rows) / n,
                    go2=sum(r["go2"] for r in rows) / n,
                    go3=sum(r["go3"] for r in rows) / n)

    lm = collect(locals_te, "learn")
    lh = collect(locals_te, "high")
    a1 = collect(locals_te, "hop1")
    ag = collect(locals_te, "all")
    void = lm["n"] < 20
    mid_between = a1["hops"] + 0.3 < lm["hops"] < ag["hops"] - 0.3
    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and mid_between and (
        lh["hops"] < 1.5)
    rec = dict(seed=args.seed, corpus=kind, windows=args.windows,
               n_train=len(locals_tr), n_test=len(locals_te), W=L,
               big_n=big["g"]["n"],
               Q={f"{a}_{Lyr}": v for (a, Lyr), v in Q.items()},
               learn_mid=lm, learn_high=lh, hop1_mid=a1, allgo_mid=ag,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  windows {args.windows} ({len(locals_tr)}/{len(locals_te)})  "
          f"W {L}  pool n {big['g']['n']}")
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
        print("\nGO SCALE: local hop1 + pool hop2 lifted residual. 529 kept.")
    else:
        print("\nSTOP: pool did not lift residual. Keep 529; 534 = mark→offer.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
