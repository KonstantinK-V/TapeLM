"""505: walk every mention of a word. Unique hop not required."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage505_star.json")
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
    return dict(n=len(value), value=value, keys=keys)


def mentions(g):
    by = defaultdict(list)
    for s in range(g["n"]):
        v = g["value"][s]
        by[v].append(s)
        for k in g["keys"][s]:
            if k != v:
                by[k].append(s)
    return by


def comps(g, s, v):
    out = [x for x in g["keys"][s] if x != v]
    fv = g["value"][s]
    if fv != v:
        out.append(fv)
    return out


def maj_of(bags):
    c = Counter()
    for b in bags:
        c.update(b)
    if not c:
        return None
    return c.most_common(1)[0][0]


def eval_bin(g, slots_of, vs, rng):
    n = n_s = n_o = 0
    for v in vs:
        sl = list(dict.fromkeys(slots_of[v]))
        if len(sl) < 3:
            continue
        rng.shuffle(sl)
        sl = sl[:40]
        for i, s in enumerate(sl):
            rest = sl[:i] + sl[i + 1:]
            bags = [comps(g, t, v) for t in rest]
            held = set(comps(g, s, v))
            if not held:
                continue
            n += 1
            pred_s = maj_of(bags)
            n_s += int(pred_s is not None and pred_s in held)
            one = bags[rng.randrange(len(bags))]
            pred_o = maj_of([one])
            n_o += int(pred_o is not None and pred_o in held)
    return dict(n=n, acc_survey=n_s / max(n, 1), acc_one=n_o / max(n, 1),
                delta=(n_s - n_o) / max(n, 1))


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
    dfs = {v: len(set(sl)) for v, sl in by.items()}
    xs = sorted(dfs.values())
    p50, p90 = xs[len(xs) // 2], xs[int(len(xs) * 0.90)]
    mid = [v for v, d in dfs.items() if 3 <= d <= max(p50, 4) and d < p90]
    high = [v for v, d in dfs.items() if d >= max(p90, p50 + 1)]
    mid_rep = eval_bin(g, by, mid, random.Random(args.seed + 1))
    high_rep = eval_bin(g, by, high, random.Random(args.seed + 2))
    void = mid_rep["n"] < 20
    gate = (not void) and (mid_rep["delta"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               p50=p50, p90=p90, n_mid=len(mid), n_high=len(high),
               mid=mid_rep, high=high_rep, void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  p50 {p50} p90 {p90}  "
          f"mid_tok {len(mid)} high_tok {len(high)}")
    print(f"MID  n {mid_rep['n']}  survey {mid_rep['acc_survey']:.3f}  "
          f"one {mid_rep['acc_one']:.3f}  Δ {mid_rep['delta']:+.3f}")
    print(f"HIGH n {high_rep['n']}  survey {high_rep['acc_survey']:.3f}  "
          f"one {high_rep['acc_one']:.3f}  Δ {high_rep['delta']:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough mid-df mentions.")
    elif gate:
        print("\nGO STAR: walking all mentions of a mid-df word beats one mention.")
    else:
        print("\nSTOP: all-mentions survey does not organize better than one hop.")
    if (not void) and high_rep["n"] and high_rep["delta"] >= mid_rep["delta"] - 0.01:
        print("DIAG: high-df (and) gains as much — survey may just be more counts.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
