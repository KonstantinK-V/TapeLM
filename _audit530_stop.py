"""530: oracle-stop on frozen v1. Policy does not see held."""
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

OUT = Path("results/_stage530_stop.json")


def rest_novel(nodes, i, seen, held, maj):
    for c in nodes[i:]:
        if c in held and c not in seen and c != maj:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=250)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=6)
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
    lr, cache = 0.3, {}
    ora_stop = ora_n = 0
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
            seen, zeros = set(), 0
            for i, c in enumerate(nodes):
                new = c in held and c not in seen and c != maj
                if i == 0:
                    if c in held:
                        seen.add(c)
                    zeros = 0 if new else 1
                    continue
                should_stop = not rest_novel(nodes, i, seen, held, maj)
                ora_stop += int(should_stop)
                ora_n += 1
                z = min(zeros, 2)
                if should_stop:
                    Q[(band, z, "stop")] += lr * (1.0 - Q[(band, z, "stop")])
                    Q[(band, z, "go")] += lr * (0.0 - Q[(band, z, "go")])
                    break
                Q[(band, z, "go")] += lr * (1.0 - Q[(band, z, "go")])
                Q[(band, z, "stop")] += lr * (0.0 - Q[(band, z, "stop")])
                qg = Q[(band, z, "go")]
                qs = Q[(band, z, "stop")]
                go = rng.choice([True, False]) if rng.random() < args.eps else qg >= qs
                if not go:
                    break
                if c in held:
                    seen.add(c)
                zeros = 0 if new else zeros + 1

    def run_learned(nodes, held, maj, band):
        if not nodes:
            return dict(hops=0, cover=0.0)
        take = [nodes[0]]
        seen, zeros = set(), 0
        c0 = nodes[0]
        new = c0 in held and c0 != maj
        if c0 in held:
            seen.add(c0)
        zeros = 0 if new else 1
        for c in nodes[1:]:
            z = min(zeros, 2)
            if Q[(band, z, "go")] < Q[(band, z, "stop")]:
                break
            take.append(c)
            new = c in held and c not in seen and c != maj
            if c in held:
                seen.add(c)
            zeros = 0 if new else zeros + 1
        return dict(hops=len(take), cover=cover(take, held))

    def collect(vs, kind_):
        rows = []
        rr = random.Random(args.seed + 9)
        for v, rest, held, maj in trials(g, by, vs, rr):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            nodes = v1_nodes(g, by, v, cache, k, high_set)
            by[v] = saved
            band = "high" if v in high_set else "mid"
            if kind_ == "learn":
                rows.append(run_learned(nodes, held, maj, band))
            else:
                choose = (lambda _b: True) if kind_ == "all" else (lambda _b: False)
                rows.append(run_ep(nodes, held, maj, choose, band))
        return mean_ep(rows)

    lm = collect(test_m, "learn")
    lh = collect(test_h, "learn")
    a1 = collect(test_m, "hop1")
    ag = collect(test_m, "all")
    ora = (ora_stop / ora_n) if ora_n else 0.0
    void = lm["n"] < 20 or not (0.1 < ora < 0.9)
    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and (
        a1["hops"] + 0.5 < lm["hops"] < ag["hops"] - 0.5) and (lh["hops"] < 1.5)
    tag = "wiki" if kind == "wiki" else path.stem
    key = f"{args.seed}_{tag}_{len(lines)}"
    rec = dict(seed=args.seed, corpus=kind, tag=tag, n_lines=len(lines), k=k,
               p25=p25, p75=p75, oracle_stop=ora, ora_n=ora_n,
               Q={f"{a}_{z}_{b}": v for (a, z, b), v in Q.items()},
               learn_mid=lm, learn_high=lh, hop1_mid=a1, allgo_mid=ag,
               void=bool(void), gate=bool(gate))
    print(f"corpus {tag}  window {len(lines)}  p25–75 {p25}-{p75}")
    print(f"ORACLE stop {ora:.3f}  (n {ora_n})")
    print(f"LEARN mid n {lm['n']} hops {lm['hops']:.2f} cover {lm['cover']:.3f}")
    print(f"LEARN high n {lh['n']} hops {lh['hops']:.2f} cover {lh['cover']:.3f}")
    print(f"HOP1  mid hops {a1['hops']:.2f} cover {a1['cover']:.3f}")
    print(f"ALLGO mid hops {ag['hops']:.2f} cover {ag['cover']:.3f}")
    print("Q", {k: round(v, 3) for k, v in rec["Q"].items()})
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: tiny mid or oracle never/always stops — nothing to teach.")
    elif gate:
        print("\nGO STOP: covers more than hop1, shorter than always-go. 529 walk kept.")
    else:
        print("\nSTOP: zeros-state did not learn the oracle cut. Keep 529 go; try another observable.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[key] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}  key={key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
