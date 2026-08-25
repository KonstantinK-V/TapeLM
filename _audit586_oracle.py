"""586: true unique-extra chain, oracle at EVERY hop.

585 picked an on-path child once, then PMI-walked — A=B≈fill1.
584 r_inf is BFS over all children. Training needs the in-between:
one true edge per step (unique extra), no cap, whole path true,
choice at every depth.

Hands:
    A  PMI-top every hop
    B  among unique extras, one that still reaches held, every hop
    C  random unique extra every hop

True hop = unique extra of current == held. Else one true edge
and continue. Refuse: no unique extra, or (B) none on-path.

GATE  B−A > 0.05  AND  r_inf − B < 0.05
VOID  n < 40  OR  cover < 0.15
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats

OUT = Path("results/_stage586_oracle.json")
SAFETY = 16


def prefix_windows(pool, length, n_win):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    return blocks[: min(n_win, len(blocks))]


def env_mid(env, mid_set, high_set):
    return (env & mid_set) - high_set or (env - high_set)


def add_pairs(co, toks, delta):
    ts = list(toks)
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            co[(a, b)] += delta
            co[(b, a)] += delta


def co_table(g, by):
    co = Counter()
    frames = {}
    n_fr = 0
    seen = set()
    for v, slots in by.items():
        for s in slots:
            if s in seen:
                continue
            seen.add(s)
            toks = set(comps(g, s, v))
            toks.add(v)
            frames[s] = toks
            add_pairs(co, toks, 1)
            n_fr += 1
    return co, frames, n_fr


def mean_pmi(extra, env_m, co, df, n_fr):
    if not env_m or n_fr <= 0:
        return 0.0
    de = max(df.get(extra, 1), 1)
    acc = 0.0
    for w in env_m:
        c = co.get((extra, w), 0)
        if c <= 0:
            continue
        dw = max(df.get(w, 1), 1)
        lift = (c * n_fr) / (de * dw)
        acc += math.log(max(lift, 1e-9))
    return acc / len(env_m)


def unique_extras(g, by, node, s_q, env_m, mid_set, co, df, n_fr):
    scored = []
    seen = set()
    bag = []
    for t in by.get(node, ()):
        if t == s_q:
            continue
        fr = set(comps(g, t, node))
        extra = [
            tok for tok in fr
            if tok not in env_m and tok != node and tok in mid_set
        ]
        for tok in fr:
            if tok in mid_set and tok != node:
                bag.append(tok)
        if len(extra) != 1:
            continue
        tok = extra[0]
        if tok in seen:
            continue
        seen.add(tok)
        scored.append((mean_pmi(tok, env_m, co, df, n_fr), tok))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored, bag


def reachable(g, by, start, s_q, env_m, mid_set, held, co, df, n_fr, memo):
    if start == held:
        return 1

    def neigh(node):
        if node not in memo:
            scored, _bag = unique_extras(
                g, by, node, s_q, env_m, mid_set, co, df, n_fr,
            )
            memo[node] = [tok for _p, tok in scored]
        return memo[node]

    seen = {start}
    frontier = list(neigh(start))
    depth = 1
    if held in frontier:
        return 1
    while frontier and depth < SAFETY:
        depth += 1
        nxt = []
        for e in frontier:
            if e in seen:
                continue
            seen.add(e)
            ut = neigh(e)
            if held in ut:
                return depth
            nxt.extend(tok for tok in ut if tok not in seen)
        frontier = nxt
    return 0


def r_inf_of(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, memo):
    scored, _bag = unique_extras(g, by, v, s_q, env_m, mid_set, co, df, n_fr)
    n1 = [tok for _p, tok in scored]
    if held in n1:
        return 1
    return int(any(
        reachable(g, by, tok, s_q, env_m, mid_set, held, co, df, n_fr, memo)
        for tok in n1
    ))


def chain(kind, g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, rng, memo):
    cur = v
    seen = {v}
    fracs = []
    for depth in range(SAFETY):
        scored, _bag = unique_extras(
            g, by, cur, s_q, env_m, mid_set, co, df, n_fr,
        )
        n1 = [tok for _p, tok in scored]
        if not n1:
            return 0, depth, fracs
        if held in n1:
            return 1, depth + 1, fracs
        onpath = [
            tok for tok in n1
            if reachable(
                g, by, tok, s_q, env_m, mid_set, held, co, df, n_fr, memo,
            )
        ]
        fracs.append((depth + 1, len(onpath), len(n1)))
        if kind == "pmi":
            nxt = n1[0]
        elif kind == "rnd":
            nxt = n1[rng.randrange(len(n1))]
        elif kind == "oracle":
            if not onpath:
                return 0, depth + 1, fracs
            nxt = onpath[0]
        else:
            nxt = n1[0]
        if nxt in seen:
            return 0, depth + 1, fracs
        seen.add(nxt)
        cur = nxt
    return 0, SAFETY, fracs


def play(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, rng):
    memo = {}
    scored, bag = unique_extras(g, by, v, s_q, env_m, mid_set, co, df, n_fr)
    n1 = [tok for _p, tok in scored]
    if not n1:
        return None
    fill1 = int(n1[0] == held)
    r1 = int(held in n1)
    rinf = r_inf_of(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, memo)
    a, da, _ = chain(
        "pmi", g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, rng, memo,
    )
    b, db, fracs = chain(
        "oracle", g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, rng, memo,
    )
    c, dc, _ = chain(
        "rnd", g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, rng, memo,
    )
    op = {1: (0, 0), 2: (0, 0), 3: (0, 0)}
    for d, n_on, n_all in fracs:
        if d in op:
            op[d] = (n_on, n_all)
    return dict(
        fill1=fill1, r1=r1, rinf=rinf,
        a=a, b=b, c=c, da=da, db=db, dc=dc,
        on1=op[1][0], n1=op[1][1],
        on2=op[2][0], n2=op[2][1],
        on3=op[3][0], n3=op[3][1],
        cover=int(held in set(bag)),
        copy=int(v == held),
        has_bag=int(bool(bag)),
    )


def one_episode(g, by, v, s_pin, s_q, mid_set, high_set, rng, co, frames, n_fr, df):
    if s_pin == s_q:
        return None
    frame = list(comps(g, s_q, v))
    if len(frame) < 2:
        return None
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    if held not in mid_set or held == v:
        return None
    env_m = env_mid(env, mid_set, high_set)
    if not env_m:
        return None
    qtoks = frames.get(s_q)
    if qtoks:
        adjust_frame_stats(co, df, qtoks, -1)
        n_use = max(n_fr - 1, 1)
    else:
        n_use = n_fr
    try:
        row = play(
            g, by, v, s_q, env_m, mid_set, held, co, df, n_use, rng,
        )
    finally:
        if qtoks:
            adjust_frame_stats(co, df, qtoks, +1)
    if row is None or not row["has_bag"]:
        return None
    return row


def collect(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return []
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    co, frames, n_fr = co_table(g, by)
    df = g.get("df") or {tok: len(slots) for tok, slots in by.items()}
    rows = []
    keys = list(mid)
    rng.shuffle(keys)
    for v in keys:
        slots = list(by.get(v, ()))
        if len(slots) < 2:
            continue
        rng.shuffle(slots)
        s_pin = slots[0]
        for s_q in slots[1: args.cap_probe + 1]:
            row = one_episode(
                g, by, v, s_pin, s_q, mid_set, high_set, rng, co, frames, n_fr, df,
            )
            if row is not None:
                rows.append(row)
    return rows


def mean(rows, key):
    if not rows:
        return 0.0
    return sum(row[key] for row in rows) / len(rows)


def frac(rows, on, n):
    num = den = 0
    for row in rows:
        den += row[n]
        num += row[on]
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=80)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=120000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(
        f"586 oracle-chain  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    fill1 = mean(rows, "fill1")
    r1 = mean(rows, "r1")
    rinf = mean(rows, "rinf")
    a = mean(rows, "a")
    b = mean(rows, "b")
    c = mean(rows, "c")
    f1 = frac(rows, "on1", "n1")
    f2 = frac(rows, "on2", "n2")
    f3 = frac(rows, "on3", "n3")
    void = n < 40 or cover < 0.15
    d_a = b - a
    gap = rinf - b
    gate = (not void) and d_a > 0.05 and gap < 0.05

    print(
        f"n {n}  cover {cover:.3f}  copy {mean(rows, 'copy'):.3f}  "
        f"fill1 {fill1:.3f}  r1 {r1:.3f}  r_inf {rinf:.3f}"
    )
    print(
        f"A pmi {a:.3f} (d {mean(rows, 'da'):.2f})  "
        f"B oracle {b:.3f} (d {mean(rows, 'db'):.2f})  "
        f"C rnd {c:.3f} (d {mean(rows, 'dc'):.2f})"
    )
    print(f"B-A {d_a:+.3f}  r_inf-B {gap:+.3f}  onpath d1/d2/d3 {f1:.3f}/{f2:.3f}/{f3:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO TRAIN: oracle chain harvests r_inf; PMI does not.")
    else:
        print("STOP: no chain to imitate, or oracle still misses r_inf.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover,
        fill1=fill1, r1=r1, rinf=rinf, a=a, b=b, c=c,
        d_a=d_a, gap=gap, on1=f1, on2=f2, on3=f3,
        da=mean(rows, "da"), db=mean(rows, "db"), dc=mean(rows, "dc"),
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
