"""527: learner on frozen v1 walk. Teacher = tape majority, not 521 held-rec."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, graph, mentions, pick_corpus
from _audit517_window import comps
from _audit518_reldf import pct_band

OUT = Path("results/_stage527_learn.json")


def allow_of(g, v, k, high_set):
    if v in high_set:
        return 1
    return max(1, int(k * g["n"] / max(g["df"][v], 1)))


def v1_nodes(g, by, v, cache, k, high_set):
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    allow = allow_of(g, v, k, high_set)
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


def majority(g, slots, v):
    cnt = Counter()
    for s in slots:
        cnt.update(x for x in comps(g, s, v) if x != v)
    if not cnt:
        return None
    return cnt.most_common(1)[0][0]


def special(nodes, held, maj):
    return int(any(x in held and x != maj for x in nodes))


def run_ep(nodes, held, maj, choose, band):
    if not nodes:
        return dict(hops=0, spec=0)
    take = [nodes[0]]
    for c in nodes[1:]:
        if not choose(band):
            break
        take.append(c)
    return dict(hops=len(take), spec=special(take, held, maj))


def mean_ep(rows):
    n = max(len(rows), 1)
    return dict(n=len(rows),
                hops=sum(r["hops"] for r in rows) / n,
                spec=sum(r["spec"] for r in rows) / n)


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
        if maj is None:
            continue
        out.append((v, rest, held, maj))
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
            r1 = (1.0 if nodes[0] in held and nodes[0] != maj else 0.0) - 0.05  # not majority
            Q[(band, "go")] += lr * (r1 - Q[(band, "go")])
            n = 1
            for c in nodes[1:]:
                qg, qs = Q[(band, "go")], Q[(band, "stop")]
                go = rng.choice([True, False]) if rng.random() < args.eps else qg >= qs
                if not go:
                    Q[(band, "stop")] += lr * (0.0 - Q[(band, "stop")])
                    break
                r = (1.0 if c in held and c != maj else 0.0) - 0.05
                Q[(band, "go")] += lr * (r - Q[(band, "go")])
                n += 1

    def learned(band):
        return Q[(band, "go")] >= Q[(band, "stop")]

    def once(_band):
        return False

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
    void = lm["n"] < 20
    gate = (not void) and (lm["hops"] > lh["hops"] + 0.5) and (lh["hops"] < 1.5) and (
        lm["spec"] > a1["spec"] + 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines), k=k,
               p25=p25, p75=p75,
               Q={f"{a}_{b}": v for (a, b), v in Q.items()},
               learn_mid=lm, learn_high=lh, hop1_mid=a1,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  p25–75 {p25}-{p75}")
    print(f"LEARN mid n {lm['n']} hops {lm['hops']:.2f} spec {lm['spec']:.3f}")
    print(f"LEARN high n {lh['n']} hops {lh['hops']:.2f} spec {lh['spec']:.3f}")
    print(f"HOP1  mid hops {a1['hops']:.2f} spec {a1['spec']:.3f}")
    print("Q", {k: round(v, 3) for k, v in rec["Q"].items()})
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: tiny test mid.")
    elif gate:
        print("\nGO LEARN: mind spends extra hops that beat majority, high stays glue.")
    else:
        print("\nSTOP: majority teacher did not train a walker. v1 walk stays frozen.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
