"""517: GPT analog under TapeLM rules. Working window W, not Φ."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _tape_frames as tframes
from _audit511_ring import cheap_rec, mentions

OUT = Path("results/_stage517_window.json")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")
CAP_W = 8


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


def comps(g, s, v):
    out = [x for x in g["keys"][s] if x != v]
    fv = g["value"][s]
    if fv != v:
        out.append(fv)
    return out


def score_w(g, by, cand, W, cache):
    if not W:
        return 0
    rec = set(cheap_rec(g, by, cand, cache))
    return sum(1 for w in W if w in rec)


def hop2(g, by, v, cache, use_w):
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    if len(rec) < 2:
        return None
    h1 = rec[0]
    W = [h1][:CAP_W]
    rest = rec[1:]
    if use_w:
        rest = sorted(rest, key=lambda c: (-score_w(g, by, c, W, cache), g["df"][c]))
    return h1, rest[0] if rest else None


def eval_bin(g, by, vs, rng, cap=20):
    n = n_w = n_5 = 0
    cache = {}
    for v in vs:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        sl = sl[:cap]
        for i, s in enumerate(sl[:3]):
            rest = sl[:i] + sl[i + 1:]
            held = set(comps(g, s, v))
            if not held:
                continue
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            a = hop2(g, by, v, cache, True)
            b = hop2(g, by, v, cache, False)
            by[v] = saved
            if not a or not a[1] or not b or not b[1]:
                continue
            n += 1
            n_w += int(a[1] in held)
            n_5 += int(b[1] in held)
    d = max(n, 1)
    return dict(n=n, hit_w=n_w / d, hit_511=n_5 / d, delta=(n_w - n_5) / d)


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
    print(f"MID  n {mid_rep['n']}  W {mid_rep['hit_w']:.3f}  511 {mid_rep['hit_511']:.3f}  "
          f"Δ {mid_rep['delta']:+.3f}")
    print(f"HIGH n {high_rep['n']}  W {high_rep['hit_w']:.3f}  511 {high_rep['hit_511']:.3f}  "
          f"Δ {high_rep['delta']:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough hop2 trials.")
    elif gate:
        print("\nGO WINDOW: hop2 that reads W hits the held-out frame more than rare-first.")
    else:
        print("\nSTOP: working window is wired, but hop2 does not beat 511 on this exam.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
