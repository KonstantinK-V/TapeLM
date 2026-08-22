"""510: and must not walk far. Step pay ∝ 1/df; meet bonus ∝ 1/df(meet).

  Expand rec companions of v, cheapest (rarest) first.
  allow = 200/df(v)  → and ≈1 hop, физика many
  skip companion df>80
  Meet: two walked hops share a rec_mid node.

VOID  n_mid < 20 or n_high < 5
GATE  meet_mid - meet_high > 0.05  AND  depth_mid > depth_high
No Q. Unique not required.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage510_cost.json")
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


def rec_all(g, by, v, cap=30):
    sl = list(by.get(v, ()))[:cap]
    if len(sl) < 8:
        return []
    cnt = Counter()
    for s in sl:
        cnt.update(set(comps(g, s, v)))
    return [(c, n) for c, n in cnt.items() if n >= 2 and c != v]


def walk(g, by, v, cache):
    rec = rec_all(g, by, v)
    rec.sort(key=lambda cn: g["df"][cn[0]])
    allow = max(1, int(200 / max(g["df"][v], 1)))
    walked = []
    pay = 0.0
    src = 1.0 / max(g["df"][v], 1)
    for c, _n in rec:
        if len(walked) >= allow:
            break
        dfc = g["df"][c]
        if dfc > HIGH_DF:
            continue
        walked.append(c)
        pay += src / dfc
    meets = 0
    bonus = 0.0
    if len(walked) >= 2:
        sets = []
        for c in walked:
            if c not in cache:
                cache[c] = {x for x, _ in rec_all(g, by, c) if g["df"][x] <= HIGH_DF}
            sets.append(cache[c])
        seen = set()
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                meet = sets[i] & sets[j]
                meet.discard(v)
                meet.discard(walked[i])
                meet.discard(walked[j])
                for m in meet:
                    if m not in seen:
                        seen.add(m)
                        meets += 1
                        bonus += src / max(g["df"][m], 1)
    return dict(depth=len(walked), pay=pay, meets=meets, bonus=bonus,
                score=pay + bonus, allow=allow)


def eval_bin(g, by, vs):
    cache = {}
    rows = [walk(g, by, v, cache) for v in vs]
    n = len(rows)
    if not n:
        return dict(n=0, depth=0.0, meets=0.0, score=0.0, pay=0.0)
    return dict(
        n=n,
        depth=sum(r["depth"] for r in rows) / n,
        meets=sum(r["meets"] for r in rows) / n,
        pay=sum(r["pay"] for r in rows) / n,
        score=sum(r["score"] for r in rows) / n,
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
    mid_rep = eval_bin(g, by, mid)
    high_rep = eval_bin(g, by, high)
    void = mid_rep["n"] < 20 or high_rep["n"] < 5
    d_meet = mid_rep["meets"] - high_rep["meets"]
    d_depth = mid_rep["depth"] - high_rep["depth"]
    gate = (not void) and (d_meet > 0.05) and (d_depth > 0)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               mid=mid_rep, high=high_rep,
               d_meet=d_meet, d_depth=d_depth,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}")
    print(f"MID  n {mid_rep['n']}  depth {mid_rep['depth']:.2f}  "
          f"meets {mid_rep['meets']:.3f}  score {mid_rep['score']:.3f}")
    print(f"HIGH n {high_rep['n']}  depth {high_rep['depth']:.2f}  "
          f"meets {high_rep['meets']:.3f}  score {high_rep['score']:.3f}")
    print(f"Δmeet {d_meet:+.3f}  Δdepth {d_depth:+.2f}  VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough words.")
    elif gate:
        print("\nGO COST: mid words walk farther and meet more than and, under 1/df pay.")
    else:
        print("\nSTOP: inverse-df cost does not keep физика walking and and short.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
