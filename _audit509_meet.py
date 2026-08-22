"""509: reward path meets. physics-school ∩ physics-science."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage509_meet.json")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")
HIGH_DF = 80


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


def rec_mid(g, by, v, cap=30):
    sl = list(by.get(v, ()))[:cap]
    if len(sl) < 8:
        return []
    cnt = Counter()
    for s in sl:
        cnt.update(set(comps(g, s, v)))
    return [c for c, n in cnt.items()
            if n >= 2 and c != v and g["df"][c] <= HIGH_DF]


def eval_bin(g, by, vs, rng, cap_v=40, cap_pair=8):
    mid_toks = [t for t, sl in by.items() if 8 <= len(sl) <= 30]
    n = n_m = n_r = n_sz = n_rsz = 0
    cache = {}
    def rec(x):
        if x not in cache:
            cache[x] = set(rec_mid(g, by, x))
        return cache[x]
    for v in vs:
        a_list = sorted(rec(v))
        if len(a_list) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(a_list) for b in a_list[i + 1:]]
        rng.shuffle(pairs)
        pairs = pairs[:cap_pair]
        for a, b in pairs:
            n += 1
            meet = rec(a) & rec(b)
            meet.discard(v)
            meet.discard(a)
            meet.discard(b)
            n_m += int(bool(meet))
            n_sz += len(meet)
            if len(mid_toks) >= 2:
                x, y = rng.sample(mid_toks, 2)
            else:
                x, y = a, b
            rm = rec(x) & rec(y)
            rm.discard(v)
            n_r += int(bool(rm))
            n_rsz += len(rm)
    d = max(n, 1)
    return dict(n=n, meet=n_m / d, rnd=n_r / d, delta=(n_m - n_r) / d,
                meet_sz=n_sz / d, rnd_sz=n_rsz / d)


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
    print(f"MID  n {mid_rep['n']}  meet {mid_rep['meet']:.3f}  rnd {mid_rep['rnd']:.3f}  "
          f"Δ {mid_rep['delta']:+.3f}  |meet| {mid_rep['meet_sz']:.2f}")
    print(f"HIGH n {high_rep['n']}  meet {high_rep['meet']:.3f}  rnd {high_rep['rnd']:.3f}  "
          f"Δ {high_rep['delta']:+.3f}  |meet| {high_rep['meet_sz']:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough rec-pairs.")
    elif gate:
        print("\nGO MEET: two hops of a mid word meet more than a random mid pair.")
    else:
        print("\nSTOP: path meets are not denser than random word pairs.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
