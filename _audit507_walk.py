"""507: walk the 506 star. Next read = high-lift recurrent companion."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage507_walk.json")
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


def mentions(g):
    by = defaultdict(list)
    for s in range(g["n"]):
        v = g["value"][s]
        by[v].append(s)
        for k in g["keys"][s]:
            if k != v:
                by[k].append(s)
    return {v: list(dict.fromkeys(sl)) for v, sl in by.items()}


def comps(g, s, v):
    out = [x for x in g["keys"][s] if x != v]
    fv = g["value"][s]
    if fv != v:
        out.append(fv)
    return out


def rec_lifts(g, bags, v):
    cnt = Counter()
    for b in bags:
        cnt.update(b)
    n_m = len(bags)
    N = max(g["n"], 1)
    rec = []
    for c, n in cnt.items():
        if n < 2 or c == v:
            continue
        pc = g["df"][c] / N
        if pc <= 0:
            continue
        rec.append((c, (n / n_m) / pc))
    rec.sort(key=lambda x: -x[1])
    return rec


def eval_bin(g, by, vs, rng, cap=30):
    n = n_l = n_r = 0
    for v in vs:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        sl = sl[:cap]
        for i, s in enumerate(sl):
            rest = sl[:i] + sl[i + 1:]
            bags = [set(comps(g, t, v)) for t in rest]
            held = set(comps(g, s, v))
            if not held:
                continue
            rec = rec_lifts(g, bags, v)
            if not rec:
                continue
            n += 1
            step = rec[0][0]
            n_l += int(step in held)
            pool = [c for c, _ in rec]
            n_r += int(rng.choice(pool) in held)
    return dict(n=n, hit_lift=n_l / max(n, 1), hit_rand=n_r / max(n, 1),
                delta=(n_l - n_r) / max(n, 1))


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
    high = [v for v, d in dfn.items() if d > 80]
    mid_rep = eval_bin(g, by, mid, random.Random(args.seed + 1))
    high_rep = eval_bin(g, by, high, random.Random(args.seed + 2))
    void = mid_rep["n"] < 50
    gate = (not void) and (mid_rep["delta"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               mid=mid_rep, high=high_rep, void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}")
    print(f"MID  n {mid_rep['n']}  lift {mid_rep['hit_lift']:.3f}  "
          f"rand {mid_rep['hit_rand']:.3f}  Δ {mid_rep['delta']:+.3f}")
    print(f"HIGH n {high_rep['n']}  lift {high_rep['hit_lift']:.3f}  "
          f"rand {high_rep['hit_rand']:.3f}  Δ {high_rep['delta']:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough mid walk trials.")
    elif gate:
        print("\nGO WALK: high-lift rec steps into the held-out frame better than a random rec.")
    else:
        print("\nSTOP: lift ranking does not walk better than a random recurrent companion.")
    if (not void) and high_rep["n"] and high_rep["delta"] >= mid_rep["delta"] - 0.01:
        print("DIAG: high-df gains as much — walk may not be mid-specific.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
