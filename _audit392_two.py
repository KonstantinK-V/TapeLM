"""B2: TWO PLACES IN, ONE PLACE OUT. Understanding as a count. The cloze eight cannot sit here.

A and B are places from the SAME wiki document. Gold is a place C in the ADDRESS neighbourhood
of both (N_addr(A) ∩ N_addr(B)) that has a unique filler. Filler-neighbourhood cannot be N:
a unique-filler C in filler-N(A) means A already holds that filler, which is cloze.

Rivals, all cap 8, all PLACES (346's concat form, named new reason: places not values):
  walk A, walk B, concat N_walk(A) ∪ N_walk(B) cut at 8.
`both` is the address intersection, ranked the way 390 ranks address neighbours, cut at 8.

Null: B drawn from another document. Must be flat.

VOID, read first: if gold is already in walk(A)'s eight on more than half the questions, this
is cloze in a costume — do not read the gate.

GATE, declared before the run: hit(both) > max(hit A, hit B) + 0.05 on 3/4 seeds, AND the
null is below half of that gain. Four seeds 1337, 8642, 2890, 4711.

Not appointed-director. Not U. Not a vocab softmax.

    python _check392_two.py
    python _audit392_two.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _audit390_address as A

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage392_two.json")


def documents(lines):
    """Wiki articles: a `= title =` line starts a new doc. Fallback: one doc."""
    docs, cur = [], []
    for i, L in enumerate(lines):
        if L.startswith("=") and L.endswith("=") and cur:
            docs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        docs.append(cur)
    return [d for d in docs if len(d) >= 2]


def pids_on_lines(T, line_set):
    out = []
    for pid, ps in enumerate(T["places"]):
        if any(T["owner"][s] in line_set for s in ps):
            out.append(pid)
    return out


def unique_filler(T, pid, banned=()):
    vs = [v for v in T["prof"][pid] if v not in banned]
    return vs[0] if len(vs) == 1 else None


def walk8(T, pid, drop, k):
    qprof = T["prof"][pid]
    return A.walk_order(T, pid, qprof, k, drop)


def addr_set(T, pid, drop):
    return set(A.half_nbrs(T, pid, drop))


def rank_addr(T, pid, nbrs, k):
    return A.addr_order(T, pid, nbrs, k)


def measure_pair(T, a, b, args):
    """One (A,B). Gold = unique-filler place in addr-N(A) ∩ addr-N(B). None if empty."""
    drop_a = set(T["on_line"][T["owner"][T["places"][a][0]]])
    drop_b = set(T["on_line"][T["owner"][T["places"][b][0]]])
    drop_a.discard(a)
    drop_b.discard(b)
    na = A.half_nbrs(T, a, drop_a)
    nb = A.half_nbrs(T, b, drop_b)
    inter = {j: na[j] + nb[j] for j in na if j in nb and j != a and j != b}
    golds = []
    for j, mm in inter.items():
        v = unique_filler(T, j)
        if v is not None:
            golds.append((j, v, mm))
    if not golds:
        return None
    golds.sort(key=lambda t: (-t[2], t[0]))
    c, _v, _mm = golds[0]
    wa = walk8(T, a, drop_a, args.topm)
    wb = walk8(T, b, drop_b, args.topm)
    concat = A.interleave(wa, wb, cap=args.topm)
    both = rank_addr(T, a, {j: inter[j] for j in inter}, args.topm)
    return {
        "n": 1,
        "hit_a": int(c in wa), "hit_b": int(c in wb),
        "hit_concat": int(c in concat), "hit_both": int(c in both),
        "in_a_uncap": int(c in na), "in_b_uncap": int(c in nb),
        "n_inter": len(inter),
    }


def run(T, docs, args, rng, null=False):
    line_of = {}
    for di, lines in enumerate(docs):
        for li in lines:
            line_of[li] = di
    by_doc = [pids_on_lines(T, set(d)) for d in docs]
    c, n_void_a = Counter(), 0
    tries, seen = 0, 0
    while seen < args.max_questions and tries < args.max_questions * 40:
        tries += 1
        di = rng.randrange(len(docs))
        pool = by_doc[di]
        if len(pool) < 2:
            continue
        a, b = rng.sample(pool, 2)
        if T["owner"][T["places"][a][0]] == T["owner"][T["places"][b][0]]:
            continue
        if null:
            others = [j for j, p in enumerate(by_doc) if j != di and len(p) >= 1]
            if not others:
                continue
            b = rng.choice(by_doc[rng.choice(others)])
        m = measure_pair(T, a, b, args)
        if m is None:
            continue
        seen += 1
        for k, v in m.items():
            c[k] += v
        n_void_a += m["hit_a"]
    c["seen"] = seen
    c["void_a"] = n_void_a
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0: s0 + args.window_lines]

    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    docs = documents(lines)
    if len(docs) < 2:
        n = len(lines)
        docs = [list(range(i, min(i + 8, n))) for i in range(0, n, 8)]
        docs = [d for d in docs if len(d) >= 2]
        print(f"no article headers — {len(docs)} windows of 8 lines")
    c = run(T, docs, args, rng, null=False)
    cn = run(T, docs, args, random.Random(args.seed + 1), null=True)
    n = max(1, c["seen"])
    nn = max(1, cn["seen"])
    rep = {
        "seed": args.seed, "places": len(T["places"]), "docs": len(docs),
        "n": c["seen"], "n_null": cn["seen"],
        "hit_a": c["hit_a"] / n, "hit_b": c["hit_b"] / n,
        "hit_concat": c["hit_concat"] / n, "hit_both": c["hit_both"] / n,
        "null_both": cn["hit_both"] / nn,
        "void_frac": c["void_a"] / n, "n_inter": c["n_inter"] / n,
    }
    mx = max(rep["hit_a"], rep["hit_b"])
    gain = rep["hit_both"] - mx
    void = rep["void_frac"] > 0.5
    print(f"places {rep['places']}  docs {rep['docs']}  pairs {rep['n']}  null {rep['n_null']}")
    print(f"VOID CHECK  gold already in walk(A)@8  {rep['void_frac']:.3f}"
          + ("  CLOZE — do not read" if void else "  ok"))
    print(f"HIT        A {rep['hit_a']:.4f}  B {rep['hit_b']:.4f}  concat {rep['hit_concat']:.4f}  "
          f"both {rep['hit_both']:.4f}  vs max(A,B) {gain:+.4f}")
    print(f"NULL       both {rep['null_both']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
