"""514: contract + MEET on working tape. Unique is not the walk."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _tape_frames as tframes
from _audit511_ring import cheap_rec, mentions

OUT = Path("results/_stage514_meetmark.json")
CONTRACT = Path("results/_CONTRACT_STAR.txt")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")
LAW = """TapeLM star contract (after 512 STOP, 513 STOP)
PIN  ≠ STAR. Do not pipeline 436 into 511.
STAR: all rec hops, 1/df budget, ring2 only if leftover allow.
MEET: write working tape[m]=MEET; not Q[H], not unique.
and: ~1 hop glue; must not flood.
Unique hop is a special case (toy 440), not the definition of walk.
"""


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


def meet_nodes(g, by, nodes, v, cache):
    if len(nodes) < 2:
        return set()
    sets = [set(cheap_rec(g, by, c, cache)) for c in nodes]
    found = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            meet = sets[i] & sets[j]
            meet.discard(v)
            meet -= set(nodes)
            found |= meet
    return found


def walk_mark(g, by, v, cache):
    allow = max(1, int(200 / max(g["df"][v], 1)))
    r1, seen = [], {v}
    for c in cheap_rec(g, by, v, cache):
        if len(r1) >= allow:
            break
        if c in seen:
            continue
        seen.add(c)
        r1.append(c)
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
    walked = r1 + r2
    tape = {}
    for m in meet_nodes(g, by, walked, v, cache):
        tape[m] = "MEET"
    return walked, tape


def eval_bin(g, by, vs, rng, cap=20):
    n = n_m = n_r = n_marks = 0
    cache = {}
    for v in vs:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        sl = sl[:cap]
        for i, s in enumerate(sl[:3]):
            rest_slots = sl[:i] + sl[i + 1:]
            held = set(comps(g, s, v))
            if not held:
                continue
            saved = by[v]
            by[v] = rest_slots
            cache.pop(v, None)
            walked, tape = walk_mark(g, by, v, cache)
            by[v] = saved
            marks = [m for m in tape if tape[m] == "MEET"]
            if not walked:
                continue
            n += 1
            n_marks += len(marks)
            n_m += int(any(m in held for m in marks)) if marks else 0
            if marks:
                bag = rng.sample(walked, min(len(marks), len(walked)))
            else:
                bag = []
            n_r += int(any(x in held for x in bag))
    d = max(n, 1)
    return dict(n=n, hit_meet=n_m / d, hit_rnd=n_r / d,
                delta=(n_m - n_r) / d, marks=n_marks / d)


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
               mid=mid_rep, high=high_rep, void=bool(void), gate=bool(gate),
               contract=LAW.strip())
    print(LAW)
    print(f"corpus {kind}  window {len(lines)}")
    print(f"MID  n {mid_rep['n']}  MEET {mid_rep['hit_meet']:.3f}  "
          f"rnd {mid_rep['hit_rnd']:.3f}  Δ {mid_rep['delta']:+.3f}  "
          f"marks {mid_rep['marks']:.2f}")
    print(f"HIGH n {high_rep['n']}  MEET {high_rep['hit_meet']:.3f}  "
          f"rnd {high_rep['hit_rnd']:.3f}  Δ {high_rep['delta']:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough mid trials.")
    elif gate:
        print("\nGO MARK: MEET on working tape predicts the held-out frame better than a random walked node.")
    else:
        print("\nSTOP: MEET marks are not a better next-read than the walk itself.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    CONTRACT.write_text(LAW, encoding="utf-8")
    print(f"wrote {out} and {CONTRACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
