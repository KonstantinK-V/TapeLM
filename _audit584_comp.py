"""584: until-refuse unique walk + static corpus connectivity.
583 capped at 3. User: do not cap; stop only when there is no unique extra
(refuse). A 1-hop chain is legal. Connectivity of the window may later
be the context — measure it, do not dismiss it as fan-out.
Two graphs, not one:
    EPISODE  query-out unique extras, BFS until frontier empty
             (safety cap 16). r1..r8, r_inf, mean hit depth, mean
             frontier. This is 'hop until apples or refuse'.
    STATIC   unique-extra skeleton of the window, no hole, no env.
             edge v—w iff some frame of v has unique mid extra w.
             giant / n_mid = how much of the window is one world.
Nested prefixes 100,400,1200,2400 — the scale hypothesis in one run.
VOID  n < 40  OR  cover < 0.15  (per N)
No STOP on fan-out. Print BLOB if giant ≥ 0.5, CHAIN if r_inf − r1 > 0.05.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats

OUT = Path("results/_stage584_comp.json")
SAFETY = 16
SIZES = (100, 400, 1200, 2400)
DEPTHS = (1, 2, 3, 4, 5, 8)


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


def static_skeleton(g, by, mid_set):
    adj = defaultdict(set)
    for v, slots in by.items():
        if v not in mid_set:
            continue
        for s in slots:
            extra = [
                tok for tok in comps(g, s, v)
                if tok != v and tok in mid_set
            ]
            if len(extra) != 1:
                continue
            w = extra[0]
            adj[v].add(w)
            adj[w].add(v)
    nodes = set(mid_set)
    seen = set()
    sizes = []
    for start in nodes:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        n = 0
        while q:
            u = q.popleft()
            n += 1
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        sizes.append(n)
    n_mid = max(len(nodes), 1)
    giant = max(sizes) / n_mid if sizes else 0.0
    n_edge = sum(len(v) for v in adj.values()) // 2
    return dict(
        n_mid=len(nodes), n_edge=n_edge, giant=giant,
        n_comp=len(sizes), mean_deg=n_edge * 2 / n_mid,
    )


def episode_bfs(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr):
    memo = {}

    def neigh(node):
        if node not in memo:
            scored, bag = unique_extras(
                g, by, node, s_q, env_m, mid_set, co, df, n_fr,
            )
            memo[node] = ([tok for _p, tok in scored], bag)
        return memo[node]

    n1, bag = neigh(v)
    fill1 = int(bool(n1) and n1[0] == held)
    hit_at = 0
    if held in n1:
        hit_at = 1
    seen = {v}
    frontier = list(n1)
    fronts = {1: len(frontier)}
    depth = 1
    while frontier and depth < SAFETY:
        if hit_at:
            break
        depth += 1
        nxt = []
        for e in frontier:
            if e in seen:
                continue
            seen.add(e)
            ut, _b = neigh(e)
            if (not hit_at) and held in ut:
                hit_at = depth
            nxt.extend(tok for tok in ut if tok not in seen)
        frontier = nxt
        fronts[depth] = len(frontier)
        if hit_at:
            break
    r_inf = int(hit_at > 0)
    visited = len(seen)
    return dict(
        fill1=fill1,
        r1=int(hit_at == 1),
        r2=int(hit_at in (1, 2)),
        r3=int(0 < hit_at <= 3),
        r4=int(0 < hit_at <= 4),
        r5=int(0 < hit_at <= 5),
        r8=int(0 < hit_at <= 8),
        r_inf=r_inf,
        hit_at=hit_at,
        front1=fronts.get(1, 0),
        front3=fronts.get(3, 0),
        front8=fronts.get(8, 0),
        visited=visited,
        cover=int(held in set(bag)),
        copy=int(v == held),
        has_bag=int(bool(bag)),
        n_ord=len(n1),
        capped=int(depth >= SAFETY and not hit_at and bool(frontier)),
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
        row = episode_bfs(
            g, by, v, s_q, env_m, mid_set, held, co, df, n_use,
        )
    finally:
        if qtoks:
            adjust_frame_stats(co, df, qtoks, +1)
    if not row["has_bag"]:
        return None
    return row


def collect(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], {}
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    skel = static_skeleton(g, by, mid_set)
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
    return rows, skel


def mean(rows, key):
    if not rows:
        return 0.0
    return sum(row[key] for row in rows) / len(rows)


def summarize(rows, skel, n_win):
    n = len(rows)
    cover = mean(rows, "cover")
    r1 = mean(rows, "r1")
    r_inf = mean(rows, "r_inf")
    void = n < 40 or cover < 0.15
    chain = (not void) and (r_inf - r1 > 0.05)
    blob = (not void) and skel.get("giant", 0) >= 0.5
    hits = [row["hit_at"] for row in rows if row["hit_at"] > 0]
    return dict(
        n_windows=n_win, n=n, cover=cover, copy=mean(rows, "copy"),
        fill1=mean(rows, "fill1"),
        r1=r1, r2=mean(rows, "r2"), r3=mean(rows, "r3"),
        r4=mean(rows, "r4"), r5=mean(rows, "r5"), r8=mean(rows, "r8"),
        r_inf=r_inf,
        d2=mean(rows, "r2") - r1,
        d_inf=r_inf - r1,
        hit_depth=sum(hits) / len(hits) if hits else 0.0,
        front1=mean(rows, "front1"),
        front3=mean(rows, "front3"),
        front8=mean(rows, "front8"),
        visited=mean(rows, "visited"),
        capped=mean(rows, "capped"),
        n_ord=mean(rows, "n_ord"),
        giant=skel.get("giant", 0.0),
        n_mid=skel.get("n_mid", 0),
        n_edge=skel.get("n_edge", 0),
        mean_deg=skel.get("mean_deg", 0.0),
        n_comp=skel.get("n_comp", 0),
        void=bool(void), chain=bool(chain), blob=bool(blob),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--n-win", type=int, default=40)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=200000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(
        f"584 comp  {path}  {kind}  sizes={list(SIZES)}  prefix, no shuffle",
        flush=True,
    )

    by_n = {}
    for n_lines in SIZES:
        windows = prefix_windows(pool, n_lines, args.n_win)
        rows = []
        skels = []
        for lines in windows:
            part, skel = collect(lines, args, rng)
            rows.extend(part)
            skels.append(skel)
        if skels:
            skel = dict(
                n_mid=sum(s["n_mid"] for s in skels) / len(skels),
                n_edge=sum(s["n_edge"] for s in skels) / len(skels),
                giant=sum(s["giant"] for s in skels) / len(skels),
                n_comp=sum(s["n_comp"] for s in skels) / len(skels),
                mean_deg=sum(s["mean_deg"] for s in skels) / len(skels),
            )
        else:
            skel = {}
        rec = summarize(rows, skel, len(windows))
        rec.update(seed=args.seed, corpus=kind, path=str(path), N=n_lines)
        by_n[str(n_lines)] = rec
        print(
            f"N={n_lines} n={rec['n']} cover={rec['cover']:.3f} "
            f"fill1={rec['fill1']:.3f} r1={rec['r1']:.3f} r3={rec['r3']:.3f} "
            f"r8={rec['r8']:.3f} r_inf={rec['r_inf']:.3f} "
            f"d_inf={rec['d_inf']:+.3f} hit_d={rec['hit_depth']:.2f}"
        )
        print(
            f"     front1={rec['front1']:.1f} front3={rec['front3']:.1f} "
            f"front8={rec['front8']:.1f} vis={rec['visited']:.1f} "
            f"giant={rec['giant']:.3f} deg={rec['mean_deg']:.2f} "
            f"VOID={rec['void']} CHAIN={rec['chain']} BLOB={rec['blob']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = dict(
        seed=args.seed, corpus=kind, path=str(path),
        elapsed_s=round(time.time() - t0, 1),
        by_N=by_n,
    )
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
