"""583: hop2+ until apples. Oracle shortest unique-extra path, not pmi-walk.
582 GO: bridge ~0.19. follow ~0.06 — pmi on the bridge is weak.
580 STOP: walking from the pmi-top extra (pear) does not reach held.
Object:
    graph  node → unique extras (same filter as 579)
    start  pin v
    goal   held
    path   shortest unique-extra path, depth cap 3
    hop2 only along that graph, never from a dead child
    r1  held is a unique extra of v
    r2  held reachable in ≤2 unique hops
    r3  ≤3
    d2  r2 − r1   does hop2 add fills?
VOID  n < 40  OR  cover < 0.15
GATE  d2 > 0.05
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

OUT = Path("results/_stage583_until.json")
CAP = 3


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
        extra = [tok for tok in fr if tok not in env_m and tok != node and tok in mid_set]
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


def shortest(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr, cap=CAP):
    memo = {}

    def neigh(node):
        if node not in memo:
            scored, _bag = unique_extras(
                g, by, node, s_q, env_m, mid_set, co, df, n_fr,
            )
            memo[node] = [tok for _p, tok in scored]
        return memo[node]

    n1 = neigh(v)
    fill1 = int(bool(n1) and n1[0] == held)
    if held in n1:
        return 1, fill1, n1
    seen = {v}
    frontier = list(n1)
    for depth in range(2, cap + 1):
        nxt = []
        for e in frontier:
            if e in seen:
                continue
            seen.add(e)
            ut = neigh(e)
            if held in ut:
                return depth, fill1, n1
            nxt.extend(ut)
        frontier = nxt
        if not frontier:
            break
    return 0, fill1, n1


def walk(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr):
    depth, fill1, n1 = shortest(
        g, by, v, s_q, env_m, mid_set, held, co, df, n_fr,
    )
    scored, bag = unique_extras(g, by, v, s_q, env_m, mid_set, co, df, n_fr)
    maj = Counter(bag).most_common(1)[0][0] if bag else None
    return dict(
        fill1=fill1,
        r1=int(depth == 1),
        r2=int(depth in (1, 2)),
        r3=int(depth in (1, 2, 3)),
        d_exact=depth,
        u_any=int(held in n1),
        maj=int(maj == held),
        cover=int(held in set(bag)),
        copy=int(v == held),
        has_bag=int(bool(bag)),
        n_ord=len(n1),
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
        row = walk(g, by, v, s_q, env_m, mid_set, held, co, df, n_use)
    finally:
        if qtoks:
            adjust_frame_stats(co, df, qtoks, +1)
    if not row["has_bag"]:
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
        f"583 until  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    fill1 = mean(rows, "fill1")
    r1 = mean(rows, "r1")
    r2 = mean(rows, "r2")
    r3 = mean(rows, "r3")
    u_any = mean(rows, "u_any")
    maj = mean(rows, "maj")
    copy = mean(rows, "copy")
    d2 = r2 - r1
    d3 = r3 - r2
    void = n < 40 or cover < 0.15
    gate = (not void) and d2 > 0.05

    print(
        f"n {n}  cover {cover:.3f}  copy {copy:.3f}  n_ord {mean(rows, 'n_ord'):.2f}"
    )
    print(
        f"fill1 {fill1:.3f}  r1 {r1:.3f}  r2 {r2:.3f}  r3 {r3:.3f}  "
        f"u_any {u_any:.3f}  maj {maj:.3f}"
    )
    print(f"d2 {d2:+.3f}  d3 {d3:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO UNTIL: hop2 adds fills. Length reward is legal.")
    else:
        print("STOP: hop2+ does not fill this hole. 579 stays. No length bonus.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover, copy=copy,
        fill1=fill1, r1=r1, r2=r2, r3=r3, u_any=u_any, maj=maj,
        d2=d2, d3=d3,
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
