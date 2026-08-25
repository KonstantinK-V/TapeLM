"""579: 506-lift among unique-extra frames. Torch-free.
578 STOP: Φ on jac/ov/width/df copies unique-then-jac (B−D ≈ +0.02).
Remainder U_any − u_jac1 ≈ +0.33 is not in those five counts.
This is the one leftover descriptor that is not the jac family:
jointness of the extra token with the query env (506 lift/pmi), still
a count, no letter in a net. Same offer as 577: |extra|==1.
    score(e) = mean_w∈env  pmi(e, w)
    pmi(e,w) = log( co(e,w) · n_fr / (df(e) df(w)) )
co is built on the window with the QUERY FRAME subtracted — otherwise
held and env co-occur by construction (the 403 leak).
Hands: u_best (574) / u_jac1 (577) / u_lift1 / u_rnd1 / u_any / maj
VOID  n < 40  OR  cover < 0.15
GATE  u_lift1 − u_jac1 > 0.05  AND  u_lift1 − maj > 0.05
Print U_any − u_lift1. If u_lift1 ≈ u_jac1 the remainder is not 506
jointness either — close frame-ranking with cause, no more Φ on this offer.
    python _check579_lift.py
    python _audit579_lift.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage579_lift.json")


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


def adjust_frame_stats(co, df, toks, delta):
    """Remove/restore one query frame from joint and marginal counts."""
    row = set(toks)
    add_pairs(co, row, delta)
    for tok in row:
        df[tok] += delta


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


def walk(g, by, v, s_q, env_m, mid_set, held, rng, co, frames, n_fr, df):
    qtoks = frames.get(s_q)
    if qtoks:
        adjust_frame_stats(co, df, qtoks, -1)
        n_fr = max(n_fr - 1, 1)
    try:
        bag = []
        best_all = None
        uniq_frames = []
        n_u = 0
        for t in by.get(v, ()):
            if t == s_q:
                continue
            fr = set(comps(g, t, v))
            ov = len(fr & env_m)
            jac = ov / max(len(fr | env_m), 1)
            extra = [tok for tok in fr if tok not in env_m and tok != v and tok in mid_set]
            for tok in fr:
                if tok in mid_set and tok != v:
                    bag.append(tok)
            key = (jac, ov, -len(extra))
            if best_all is None or key > best_all[0]:
                best_all = (key, extra)
            if extra == [held]:
                n_u += 1
            if len(extra) == 1:
                tok = extra[0]
                pmi = mean_pmi(tok, env_m, co, df, n_fr)
                uniq_frames.append((key, pmi, tok))
        extras = best_all[1] if best_all is not None else []
        u_best = int(len(extras) == 1 and extras[0] == held)
        if uniq_frames:
            u_jac1 = int(max(uniq_frames, key=lambda item: item[0])[2] == held)
            u_lift1 = int(max(uniq_frames, key=lambda item: item[1])[2] == held)
            u_rnd1 = int(rng.choice(uniq_frames)[2] == held)
        else:
            u_jac1 = u_lift1 = u_rnd1 = 0
        maj = Counter(bag).most_common(1)[0][0] if bag else None
        return dict(
            u_best=u_best,
            u_jac1=u_jac1,
            u_lift1=u_lift1,
            u_rnd1=u_rnd1,
            u_any=int(n_u > 0),
            n_u=n_u,
            n_uniq=len(uniq_frames),
            maj=int(maj == held),
            cover=int(held in set(bag)),
            copy=int(v == held),
            has_bag=int(bool(bag)),
        )
    finally:
        if qtoks:
            adjust_frame_stats(co, df, qtoks, +1)


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
    row = walk(g, by, v, s_q, env_m, mid_set, held, rng, co, frames, n_fr, df)
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
        f"579 lift  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    u_best = mean(rows, "u_best")
    u_jac1 = mean(rows, "u_jac1")
    u_lift1 = mean(rows, "u_lift1")
    u_rnd1 = mean(rows, "u_rnd1")
    u_any = mean(rows, "u_any")
    maj = mean(rows, "maj")
    copy = mean(rows, "copy")
    d_jac = u_lift1 - u_jac1
    d_maj = u_lift1 - maj
    d_or = u_any - u_lift1
    void = n < 40 or cover < 0.15
    gate = (not void) and d_jac > 0.05 and d_maj > 0.05

    print(
        f"n {n}  cover {cover:.3f}  copy {copy:.3f}  "
        f"n_uniq {mean(rows, 'n_uniq'):.2f}"
    )
    print(
        f"u_best {u_best:.3f}  u_jac1 {u_jac1:.3f}  u_lift1 {u_lift1:.3f}  "
        f"u_rnd1 {u_rnd1:.3f}  u_any {u_any:.3f}  maj {maj:.3f}"
    )
    print(f"lift-jac {d_jac:+.3f}  lift-maj {d_maj:+.3f}  u_any-lift {d_or:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO LIFT: extra-env pmi beats unique-then-jac. Algebra, not Phi.")
    else:
        print("STOP: 506 jointness does not beat jac on this offer. Close ranking.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover, copy=copy,
        u_best=u_best, u_jac1=u_jac1, u_lift1=u_lift1, u_rnd1=u_rnd1,
        u_any=u_any, maj=maj, d_jac=d_jac, d_maj=d_maj, d_or=d_or,
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
