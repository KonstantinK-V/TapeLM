"""506: star = recurrent companions with lift over df. Not majority hop."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage506_lift.json")
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


def star_pmi(g, sl, v, rng, cap=40):
    sl = list(sl)
    rng.shuffle(sl)
    sl = sl[:cap]
    if len(sl) < 8:
        return None
    bags = [set(comps(g, s, v)) for s in sl]
    cnt = Counter()
    for b in bags:
        cnt.update(b)
    rec = [c for c, n in cnt.items() if n >= 2 and c != v]
    if not rec:
        return dict(n_rec=0, lift=0.0, pmi=0.0)
    n_m = len(bags)
    N = g["n"]
    lifts, pmis = [], []
    for c in rec:
        pc = g["df"][c] / max(N, 1)
        if pc <= 0:
            continue
        lift = (cnt[c] / n_m) / pc
        lifts.append(lift)
        pmis.append(math.log(max(lift, 1e-9)))
    if not pmis:
        return dict(n_rec=0, lift=0.0, pmi=0.0)
    return dict(n_rec=len(pmis),
                lift=sum(lifts) / len(lifts),
                pmi=sum(pmis) / len(pmis))


def bin_mean(g, by, vs, rng):
    rows = []
    for v in vs:
        r = star_pmi(g, by[v], v, rng)
        if r:
            rows.append(r)
    if not rows:
        return dict(n=0, lift=0.0, pmi=0.0, n_rec=0)
    return dict(
        n=len(rows),
        lift=sum(r["lift"] for r in rows) / len(rows),
        pmi=sum(r["pmi"] for r in rows) / len(rows),
        n_rec=sum(r["n_rec"] for r in rows) / len(rows),
    )


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
    mid_rep = bin_mean(g, by, mid, random.Random(args.seed + 1))
    high_rep = bin_mean(g, by, high, random.Random(args.seed + 2))
    void = mid_rep["n"] < 15 or high_rep["n"] < 15
    delta = mid_rep["pmi"] - high_rep["pmi"]
    gate = (not void) and (delta > 0.10)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               n_mid=mid_rep["n"], n_high=high_rep["n"],
               mid=mid_rep, high=high_rep, delta=delta,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  mid {mid_rep['n']} high {high_rep['n']}")
    print(f"MID  lift {mid_rep['lift']:.3f}  pmi {mid_rep['pmi']:.3f}  rec {mid_rep['n_rec']:.1f}")
    print(f"HIGH lift {high_rep['lift']:.3f}  pmi {high_rep['pmi']:.3f}  rec {high_rep['n_rec']:.1f}")
    print(f"Δpmi {delta:+.3f}  VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough mid or high tokens.")
    elif gate:
        print("\nGO LIFT: mid-df star is more specific than high-df (and).")
    else:
        print("\nSTOP: recurrent companions of физика-like words are not more specific than and.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
