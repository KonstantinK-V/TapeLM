"""521: per-hop pay on 511 order. Teacher = tape, not CE."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, graph, mentions, pick_corpus
from _audit517_window import comps
from _audit518_reldf import pct_band

OUT = Path("results/_stage521_hoppay.json")


def allow_of(g, v, k, high_set):
    if v in high_set:
        return 1
    return max(1, int(k * g["n"] / max(g["df"][v], 1)))


def run_ep(g, by, v, held, cache, k, high_set, choose):
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    if not rec:
        return dict(hops=0, hit=0.0, n_hit=0)
    allow = allow_of(g, v, k, high_set)
    hops = [rec[0]]
    n_hit = int(rec[0] in held)
    band = "high" if v in high_set else "mid"
    for c in rec[1:]:
        if len(hops) >= allow:
            break
        if not choose(band):
            break
        hops.append(c)
        n_hit += int(c in held)
    return dict(hops=len(hops), hit=n_hit / len(hops), n_hit=n_hit)


def mean_ep(rows):
    n = max(len(rows), 1)
    return dict(
        n=len(rows),
        hops=sum(r["hops"] for r in rows) / n,
        hit=sum(r["hit"] for r in rows) / n,
    )


def trials(g, by, vs, rng, cap=12):
    out = []
    for v in vs:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        sl = sl[:cap]
        i = 0
        rest = sl[1:]
        held = set(comps(g, sl[i], v))
        if not held or not rest:
            continue
        out.append((v, rest, held, sl[i]))
    return out


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
    lr = 0.2
    cache = {}
    for _ in range(args.epochs):
        rng.shuffle(train_v)
        for v, rest_slots, held, _s in trials(g, by, train_v, rng):
            saved = by[v]
            by[v] = rest_slots
            cache.pop(v, None)
            rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
            by[v] = saved
            if not rec:
                continue
            allow = allow_of(g, v, k, high_set)
            band = "high" if v in high_set else "mid"
            r1 = (1.0 if rec[0] in held else 0.0) - 0.05  # tape teacher
            Q[(band, "go")] += lr * (r1 - Q[(band, "go")])
            n = 1
            for c in rec[1:]:
                if n >= allow:
                    break
                qg, qs = Q[(band, "go")], Q[(band, "stop")]
                if rng.random() < args.eps:
                    go = rng.choice([True, False])
                else:
                    go = qg >= qs
                if not go:
                    Q[(band, "stop")] += lr * (0.0 - Q[(band, "stop")])
                    break
                r = (1.0 if c in held else 0.0) - 0.05
                Q[(band, "go")] += lr * (r - Q[(band, "go")])
                n += 1

    def learned(band):
        return Q[(band, "go")] >= Q[(band, "stop")]

    def once(_band):
        return False

    def collect(vs, choose):
        rows = []
        rr = random.Random(args.seed + 9)
        for v, rest_slots, held, _s in trials(g, by, vs, rr):
            saved = by[v]
            by[v] = rest_slots
            cache.pop(v, None)
            rows.append(run_ep(g, by, v, held, cache, k, high_set, choose))
            by[v] = saved
        return mean_ep(rows)

    lm = collect(test_m, learned)
    lh = collect(test_h, learned)
    a1 = collect(test_m, once)
    void = lm["n"] < 20
    gate = (not void) and (lm["hops"] > lh["hops"] + 0.5) and (lh["hops"] < 1.5) and (
        lm["hit"] > a1["hit"] + 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines), k=k,
               p25=p25, p75=p75,
               Q={f"{a}_{b}": v for (a, b), v in Q.items()},
               learn_mid=lm, learn_high=lh, always1_mid=a1,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  p25–75 {p25}-{p75}")
    print(f"LEARN mid n {lm['n']} hops {lm['hops']:.2f} hit {lm['hit']:.3f}")
    print(f"LEARN high n {lh['n']} hops {lh['hops']:.2f} hit {lh['hit']:.3f}")
    print(f"STOP1 mid hops {a1['hops']:.2f} hit {a1['hit']:.3f}")
    print("Q", {k: round(v, 3) for k, v in rec["Q"].items()})
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: tiny test mid.")
    elif gate:
        print("\nGO HOPPAY: mid continues; high stays short; extra hops beat hop1-only on tape.")
    else:
        print("\nSTOP: per-hop + did not teach mid to walk and high to glue.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
