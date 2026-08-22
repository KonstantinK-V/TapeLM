"""503: coverage vs tape length. Reachability ≠ policy."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place

OUT = Path("results/_stage503_cover.json")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")
CAP = 8
LENGTHS = (100, 400, 1200, 2400)


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
    place, value, keys, line = [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        ks = {x for x in list(left) + list(right) if x}
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            keys.append(ks)
    slots_at = defaultdict(list)
    for s, P in enumerate(place):
        slots_at[P].append(s)
    by_key = defaultdict(set)
    for s, ks in enumerate(keys):
        for k in ks:
            by_key[k].add(place[s])
    return dict(n=len(place), place=place, value=value, line=line,
                slots_at=slots_at, by_key=by_key)


def nexts(g, P, v):
    return g["by_key"].get(v, set()) - {P}


def const_pins(g, rng, max_q):
    idx = list(range(g["n"]))
    rng.shuffle(idx)
    out = []
    for s in idx[:max_q]:
        v, P, li = g["value"][s], g["place"][s], g["line"][s]
        foreign = [t for t in g["slots_at"][P] if t != s and g["line"][t] != li]
        if not foreign:
            continue
        rng.shuffle(foreign)
        offer = foreign[:CAP]
        maj = Counter(g["value"][t] for t in offer).most_common(1)[0][0]
        if maj != v:
            continue
        out.append((P, v))
    return out


def measure(g, rng, max_q):
    pins = const_pins(g, rng, max_q)
    n1 = len(pins)
    F0 = Fp = FL = U = R2 = R3 = 0
    for P, v in pins:
        c = nexts(g, P, v)
        if len(c) == 0:
            F0 += 1
            continue
        if len(c) != 1:
            Fp += 1
            continue
        U += 1
        R = next(iter(c))
        pin2 = think_place(list(g["slots_at"][R]), g["value"], rng)
        if pin2 is None:
            FL += 1
            continue
        R2 += 1
        w = g["value"][pin2]
        c3 = nexts(g, R, w)
        if len(c3) != 1:
            continue
        S = next(iter(c3))
        pin3 = think_place(list(g["slots_at"][S]), g["value"], rng)
        R3 += int(pin3 is not None)
    d = max(n1, 1)
    return dict(n_const=n1, R1=1.0 if n1 else 0.0,
                U=U / d, R2=R2 / d, R3=R3 / d,
                F0=F0 / d, Famb=Fp / d, Fland=FL / d,
                n_U=U, n_R2=R2, n_R3=R3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--max-questions", type=int, default=8000)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    print(f"corpus {kind}  pool {len(pool)}")
    print(f"{'L':>6}  n_const   U     R2    R3    F0   Famb  Fland")
    by = {}
    for L in lengths:
        if L > len(pool):
            print(f"{L:>6}  skip")
            continue
        s0 = 0 if L == len(pool) else rng.randrange(len(pool) - L + 1)
        lines = pool[s0:s0 + L]
        g = graph(lines, args.frame_max, args.min_fillers)
        if g is None:
            print(f"{L:>6}  no tape")
            continue
        rep = measure(g, random.Random(args.seed + L), args.max_questions)
        by[str(L)] = rep
        print(f"{L:>6}  {rep['n_const']:7}  {rep['U']:.3f} {rep['R2']:.3f} {rep['R3']:.3f}  "
              f"{rep['F0']:.3f} {rep['Famb']:.3f} {rep['Fland']:.3f}")
    r2s = [(L, by[str(L)]["R2"]) for L in lengths if str(L) in by]
    n400 = by.get("400", {}).get("n_const", 0)
    void = n400 < 30 if "400" in by else True
    grow = bool(r2s) and r2s[-1][1] > r2s[0][1] + 0.02
    gate = (not void) and grow
    rec = dict(seed=args.seed, corpus=kind, pool=len(pool),
               by_len=by, void=bool(void), grow=bool(grow), gate=bool(gate))
    print(f"VOID {void}  GROW {grow}  GATE {gate}")
    if void:
        print("\nVOID: not enough 436 pins at L=400.")
    elif grow:
        print("\nGO COVER: R2 rose with N. Frontier is not counted as fail.")
    else:
        print("\nSTOP COVER: longer tape did not open more hop2 pins (+0.02).")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
