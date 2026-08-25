"""577: held-free rule that walks the 576 path. Torch-free. Φ waits.
576 GO: some foreign frame uniquely names held (~0.70) and beats maj.
574 only looked at the best-jac frame, then demanded unique extra (~0.14).
Those are different walks. The first held-free path that 576 actually
opens is:
    keep foreign frames with |extra| == 1
    pick the one with largest jac(F, env)
    refuse if none
Not «best jac, then maybe unique» (574). Unique first, jac second.
Weights are not needed if this already sits on U_any. If it beats maj
but still leaves U_any − u_jac1 > 0.05, THAT remainder is Φ ranking
frames — next, not now.
Hands on the same episode (query out, jac without held):
    u_best   574
    u_jac1   unique-extra frames, max jac
    u_rnd1   random unique-extra frame
    u_any    oracle (576)
    maj      511
VOID  n < 40  OR  cover < 0.15
GATE  u_jac1 − maj > 0.05  AND  u_jac1 − u_rnd1 > 0.05
Print U_any − u_jac1: < 0.05 → algebra closed the path; > 0.05 → Φ next.
    python _check577_ujac.py
    python _audit577_ujac.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

OUT = Path("results/_stage577_ujac.json")


def prefix_windows(pool, length, n_win):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    return blocks[: min(n_win, len(blocks))]


def env_mid(env, mid_set, high_set):
    return (env & mid_set) - high_set or (env - high_set)


def walk(g, by, v, s_q, env_m, mid_set, held, rng):
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
        if len(extra) == 1:
            uniq_frames.append((key, extra[0]))
        if extra == [held]:
            n_u += 1
    extras = best_all[1] if best_all is not None else []
    u_best = int(len(extras) == 1 and extras[0] == held)
    if uniq_frames:
        uniq_frames.sort(key=lambda item: item[0], reverse=True)
        u_jac1 = int(uniq_frames[0][1] == held)
        u_rnd1 = int(rng.choice(uniq_frames)[1] == held)
    else:
        u_jac1 = 0
        u_rnd1 = 0
    maj = Counter(bag).most_common(1)[0][0] if bag else None
    return dict(
        u_best=u_best,
        u_jac1=u_jac1,
        u_rnd1=u_rnd1,
        u_any=int(n_u > 0),
        n_u=n_u,
        n_uniq=len(uniq_frames),
        maj=int(maj == held),
        cover=int(held in set(bag)),
        copy=int(v == held),
        has_bag=int(bool(bag)),
    )


def one_episode(g, by, v, s_pin, s_q, mid_set, high_set, rng):
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
    row = walk(g, by, v, s_q, env_m, mid_set, held, rng)
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
            row = one_episode(g, by, v, s_pin, s_q, mid_set, high_set, rng)
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
        f"577 ujac  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    u_best = mean(rows, "u_best")
    u_jac1 = mean(rows, "u_jac1")
    u_rnd1 = mean(rows, "u_rnd1")
    u_any = mean(rows, "u_any")
    maj = mean(rows, "maj")
    copy = mean(rows, "copy")
    n_uniq = mean(rows, "n_uniq")
    n_u = mean(rows, "n_u")
    d_maj = u_jac1 - maj
    d_rnd = u_jac1 - u_rnd1
    d_or = u_any - u_jac1
    void = n < 40 or cover < 0.15
    gate = (not void) and d_maj > 0.05 and d_rnd > 0.05

    print(
        f"n {n}  cover {cover:.3f}  copy {copy:.3f}  "
        f"n_uniq {n_uniq:.2f}  n_u {n_u:.2f}"
    )
    print(
        f"u_best {u_best:.3f}  u_jac1 {u_jac1:.3f}  u_rnd1 {u_rnd1:.3f}  "
        f"u_any {u_any:.3f}  maj {maj:.3f}"
    )
    print(f"d_maj {d_maj:+.3f}  d_rnd {d_rnd:+.3f}  u_any-u_jac1 {d_or:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate and d_or <= 0.05:
        print("GO ALGEBRA: unique-then-jac closes the 576 path. Phi not needed.")
    elif gate:
        print("GO RULE: unique-then-jac beats maj/rand; remainder for Phi.")
    else:
        print("STOP: unique-then-jac does not beat maj. Remainder is still U_any.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover, copy=copy,
        u_best=u_best, u_jac1=u_jac1, u_rnd1=u_rnd1, u_any=u_any, maj=maj,
        n_uniq=n_uniq, n_u=n_u, d_maj=d_maj, d_rnd=d_rnd, d_or=d_or,
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
