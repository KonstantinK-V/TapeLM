"""528: per-step novelty teacher on frozen v1. Not wiki-400."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import graph, mentions, pick_corpus
from _audit517_window import comps
from _audit518_reldf import pct_band
from _audit527_learn import majority, v1_nodes

OUT = Path("results/_stage528_step.json")
C_STEP = 0.02  # soft: 0.05 killed go when novelty rare (stories-400 ceiling)


def cover(taken, held):
    if not held:
        return 0.0
    return len(set(taken) & held) / len(held)


def run_ep(nodes, held, maj, choose, band):
    if not nodes:
        return dict(hops=0, cover=0.0)
    take = [nodes[0]]
    for c in nodes[1:]:
        if not choose(band):
            break
        take.append(c)
    return dict(hops=len(take), cover=cover(take, held))


def mean_ep(rows):
    n = max(len(rows), 1)
    return dict(n=len(rows),
                hops=sum(r["hops"] for r in rows) / n,
                cover=sum(r["cover"] for r in rows) / n)


def trials(g, by, vs, rng, cap=12):
    out = []
    for v in vs:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        sl = sl[:cap]
        held = set(comps(g, sl[0], v))
        rest = sl[1:]
        if not held or not rest:
            continue
        maj = majority(g, rest, v)
        out.append((v, rest, held, maj))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=2400)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
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
    mid, high, p25, p75 = pct_band(g, by)
    k = 200.0 / max(g["n"], 1)
    high_set = set(high)
    rng.shuffle(mid)
    rng.shuffle(high)
    cut_m, cut_h = int(0.6 * len(mid)), int(0.6 * len(high))
    train_v = mid[:cut_m] + high[:cut_h]
    test_m, test_h = mid[cut_m:], high[cut_h:]
    Q = defaultdict(float)
    lr, cache = 0.2, {}
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
            for i, c in enumerate(nodes):
                new = c in held and c not in seen and c != maj
                if c in held:
                    seen.add(c)
                # hop1 free (always taken); −c only on chosen extra hops
                r = (1.0 if new else 0.0) - (0.0 if i == 0 else C_STEP)
                if i == 0:
                    Q[(band, "go")] += lr * (r - Q[(band, "go")])
                    continue
                qg, qs = Q[(band, "go")], Q[(band, "stop")]
                go = rng.choice([True, False]) if rng.random() < args.eps else qg >= qs
                if not go:
                    # stop is an action: 0, not a silence penalty
                    Q[(band, "stop")] += lr * (0.0 - Q[(band, "stop")])
                    break
                Q[(band, "go")] += lr * (r - Q[(band, "go")])

    def learned(band):
        return Q[(band, "go")] >= Q[(band, "stop")]

    def once(_b):
        return False

    def always(_b):
        return True

    def collect(vs, choose):
        rows = []
        rr = random.Random(args.seed + 9)
        for v, rest, held, maj in trials(g, by, vs, rr):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            nodes = v1_nodes(g, by, v, cache, k, high_set)
            by[v] = saved
            band = "high" if v in high_set else "mid"
            rows.append(run_ep(nodes, held, maj, choose, band))
        return mean_ep(rows)

    lm = collect(test_m, learned)
    lh = collect(test_h, learned)
    a1 = collect(test_m, once)
    ag = collect(test_m, always)
    ceiling = ag["cover"] - a1["cover"]
    thin_mid = lm["n"] < 20
    thin_ceil = ceiling <= 0.05
    void = thin_mid or thin_ceil
    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and (
        lm["hops"] < ag["hops"] - 0.5) and (lh["hops"] < 1.5)
    tag = "wiki" if kind == "wiki" else path.stem
    key = f"{args.seed}_{tag}_{len(lines)}"
    rec = dict(seed=args.seed, corpus=kind, tag=tag, n_lines=len(lines), k=k,
               p25=p25, p75=p75, ceiling=ceiling,
               Q={f"{a}_{b}": v for (a, b), v in Q.items()},
               learn_mid=lm, learn_high=lh, hop1_mid=a1, allgo_mid=ag,
               void=bool(void), thin_ceil=bool(thin_ceil), gate=bool(gate))
    print(f"corpus {tag}  window {len(lines)}  p25–75 {p25}-{p75}")
    print(f"LEARN mid n {lm['n']} hops {lm['hops']:.2f} cover {lm['cover']:.3f}")
    print(f"LEARN high n {lh['n']} hops {lh['hops']:.2f} cover {lh['cover']:.3f}")
    print(f"HOP1  mid hops {a1['hops']:.2f} cover {a1['cover']:.3f}")
    print(f"ALLGO mid hops {ag['hops']:.2f} cover {ag['cover']:.3f}  "
          f"ceiling {ceiling:+.3f}")
    print("Q", {k: round(v, 3) for k, v in rec["Q"].items()})
    print(f"VOID {void}  GATE {gate}")
    if thin_mid:
        print("\nVOID: tiny test mid.")
    elif thin_ceil:
        print("\nVOID: ALLGO−HOP1 ≤ 0.05 — exam ceiling starved, not a learner fail.")
    elif gate:
        print("\nGO STEP: covers more than hop1, spends less than always-go. Mind on frozen walk.")
    else:
        print("\nSTOP: ceiling alive, but novelty Q did not separate stop from v1-allow.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[key] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}  key={key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
