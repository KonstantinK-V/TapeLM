"""576: leftover unique-frame path on the 574 episode. Torch-free.

575 showed jac-weighted vote ≤ maj. That is a FEATURE ceiling, not
«untrained GPT». A linear Φ on jac reproduces wvote. Weights can only
help if some OTHER frame of the pin uniquely names held — a path 574
did not take because it only looked at the best-jac frame.

On the same episode as 574/575 (query out, held not in jac):

    U_best  unique extra of the single best-jac frame          (574)
    U_any   some foreign frame has extra == {held}
            (a path among frames exists that uniquely hits)
    n_u     how many such frames
    maj     unweighted companion majority                      (511)

The object to learn, if U_any − U_best > 0.05, is not a token and not
jac-weights. It is the next FRAME:

    path_0 = pin v
    a_t    = Φ scores remaining foreign frames of the current node
    tape   = unique extra of the chosen frame, or refuse
    r_t    = match the original

VOID  n < 40  OR  cover < 0.15
GATE  U_any − U_best > 0.05  AND  U_any − maj > 0.05

If U_any ≈ U_best, 574 already took the only unique frame — there is no
path for weights to move. If U_any > U_best but still < maj, ranking
frames buys precision, not enough recall to beat 511; say so, do not
train. Φ only if GATE.

    python _check576_path.py
    python _audit576_path.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage576_path.json")


def prefix_windows(pool, length, n_win):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    return blocks[: min(n_win, len(blocks))]


def env_mid(env, mid_set, high_set):
    return (env & mid_set) - high_set or (env - high_set)


def frame_paths(g, by, v, s_q, env_m, mid_set, held):
    bag = []
    best = None
    n_u = 0
    u_any = 0
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
        if best is None or key > best[0]:
            best = (key, extra)
        if extra == [held]:
            n_u += 1
            u_any = 1
    extras = best[1] if best is not None else []
    u_best = int(len(extras) == 1 and extras[0] == held)
    maj = Counter(bag).most_common(1)[0][0] if bag else None
    return dict(
        u_best=u_best,
        u_any=u_any,
        n_u=n_u,
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
    row = frame_paths(g, by, v, s_q, env_m, mid_set, held)
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
        f"576 path  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    u_best = mean(rows, "u_best")
    u_any = mean(rows, "u_any")
    n_u = mean(rows, "n_u")
    maj = mean(rows, "maj")
    copy = mean(rows, "copy")
    d_best = u_any - u_best
    d_maj = u_any - maj
    void = n < 40 or cover < 0.15
    gate = (not void) and d_best > 0.05 and d_maj > 0.05
    print(f"n {n}  cover {cover:.3f}  copy {copy:.3f}  mean n_u {n_u:.2f}")
    print(f"U_best {u_best:.3f}  U_any {u_any:.3f}  maj {maj:.3f}")
    print(f"U_any-best {d_best:+.3f}  U_any-maj {d_maj:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO PATH: some other frame uniquely names held. Phi ranks frames.")
    elif d_best > 0.05:
        print("PATH EXISTS but still under maj. Do not train — recall ceiling is 511.")
    else:
        print("STOP: 574 already took the only unique frame. No path for weights.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover, copy=copy,
        u_best=u_best, u_any=u_any, n_u=n_u, maj=maj,
        d_best=d_best, d_maj=d_maj,
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
