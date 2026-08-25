"""580: 579 step, twice. Torch-free. Same hole, same env.
579 GO: unique-extra + max pmi(extra, env) ~0.48 beats jac ~0.38 and maj.
Remainder U_any − lift ~0.23: a unique-held frame exists, pmi did not
pick it. The next algebraic object is not another scorer — it is the
same step from the token we DID pick.
    hop1  unique extra of pin v, max pmi with env     (579)
    if extra == held: stop
    hop2  that extra is the new pin; same rule, same env
    refuse if no unique extra
Query frame out of co (same as 579). Held never in the score.
Hands: hop1 (579) / chain (hop1 or hop2) / maj / u_any (oracle hop1)
Also hop2_only = miss then hit — the residual of the second step.
VOID  n < 40  OR  cover < 0.15
GATE  chain − hop1 > 0.05  AND  chain − maj > 0.05
If chain ≈ hop1, compose does not buy on this tape; 579 stays the walk.
    python _check580_chain.py
    python _audit580_chain.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage580_chain.json")


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


def unique_frames(g, by, node, s_q, env_m, mid_set, held, co, df, n_fr):
    bag = []
    uniq = []
    n_u = 0
    for t in by.get(node, ()):
        if t == s_q:
            continue
        fr = set(comps(g, t, node))
        extra = [tok for tok in fr if tok not in env_m and tok != node and tok in mid_set]
        for tok in fr:
            if tok in mid_set and tok != node:
                bag.append(tok)
        if extra == [held]:
            n_u += 1
        if len(extra) == 1:
            tok = extra[0]
            pmi = mean_pmi(tok, env_m, co, df, n_fr)
            uniq.append((pmi, tok))
    return uniq, bag, n_u


def walk(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr):
    uniq1, bag, n_u = unique_frames(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr)
    if not uniq1:
        maj = Counter(bag).most_common(1)[0][0] if bag else None
        return dict(
            hop1=0, hop2=0, chain=0, hop2_only=0,
            u_any=int(n_u > 0), maj=int(maj == held),
            cover=int(held in set(bag)), copy=int(v == held),
            has_bag=int(bool(bag)), n_uniq=0,
        )
    e1 = max(uniq1, key=lambda item: item[0])[1]
    hop1 = int(e1 == held)
    hop2 = 0
    if not hop1:
        uniq2, _b2, _n2 = unique_frames(
            g, by, e1, s_q, env_m, mid_set, held, co, df, n_fr,
        )
        if uniq2:
            e2 = max(uniq2, key=lambda item: item[0])[1]
            hop2 = int(e2 == held)
    chain = int(hop1 or hop2)
    maj = Counter(bag).most_common(1)[0][0] if bag else None
    return dict(
        hop1=hop1, hop2=hop2, chain=chain,
        hop2_only=int((not hop1) and hop2),
        u_any=int(n_u > 0), maj=int(maj == held),
        cover=int(held in set(bag)), copy=int(v == held),
        has_bag=int(bool(bag)), n_uniq=len(uniq1),
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
        f"580 chain  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    hop1 = mean(rows, "hop1")
    hop2 = mean(rows, "hop2")
    hop2_only = mean(rows, "hop2_only")
    chain = mean(rows, "chain")
    u_any = mean(rows, "u_any")
    maj = mean(rows, "maj")
    copy = mean(rows, "copy")
    d_h1 = chain - hop1
    d_maj = chain - maj
    void = n < 40 or cover < 0.15
    gate = (not void) and d_h1 > 0.05 and d_maj > 0.05

    print(
        f"n {n}  cover {cover:.3f}  copy {copy:.3f}  "
        f"n_uniq {mean(rows, 'n_uniq'):.2f}"
    )
    print(
        f"hop1 {hop1:.3f}  hop2_only {hop2_only:.3f}  chain {chain:.3f}  "
        f"u_any {u_any:.3f}  maj {maj:.3f}"
    )
    print(f"chain-hop1 {d_h1:+.3f}  chain-maj {d_maj:+.3f}  hop2 {hop2:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO CHAIN: second 579-step buys cover. Compose lives.")
    else:
        print("STOP: second step does not buy. 579 stays the walk.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover, copy=copy,
        hop1=hop1, hop2=hop2, hop2_only=hop2_only, chain=chain,
        u_any=u_any, maj=maj, d_h1=d_h1, d_maj=d_maj,
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
