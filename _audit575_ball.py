"""575: Jaccard-weighted vote on the pin star. Torch-free.
574 STOP was not «pin cannot reach». cover≈0.80, copy=0, refuse≈0.77,
hit|commit≈0.67, overall pin 0.15 < maj 0.25. Unique extra of the best
foreign frame is precise and rare; unweighted companion majority always
commits and wins on recall.
The leftover mass is geometric, not XOR:
    score(tok) = Σ_F jac(F, env) · 1[tok ∈ F]
    F runs over foreign frames of the pin v (query slot out)
Hands on the SAME episode
    uniq     574: unique extra of the single best F, else refuse
    wvote    argmax score  (always commit; env-weighted)
    peaked   wvote only if top − second > 0.05, else refuse   (524)
    maj      unweighted count, ignores env                   (511)
    rand     random companion
VOID  n < 40  OR  cover < 0.15
GATE  wvote − maj > 0.05  AND  wvote − rand > 0.05
If wvote ≈ maj, environment does not move the star — 511 is the ceiling
and there is no ball to learn. peaked is diagnostic (honesty vs recall),
not the gate. Φ stays out.
    python _check575_ball.py
    python _audit575_ball.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage575_ball.json")
MARGIN = 0.05


def prefix_windows(pool, length, n_win):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    return blocks[: min(n_win, len(blocks))]


def env_mid(env, mid_set, high_set):
    return (env & mid_set) - high_set or (env - high_set)


def star_scores(g, by, v, s_q, env_m, mid_set):
    """Return (jac-weighted scores, unweighted bag, best-frame extras)."""
    weighted = Counter()
    bag = []
    best = None
    for t in by.get(v, ()):
        if t == s_q:
            continue
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        jac = ov / max(len(fr | env_m), 1)
        extra = [tok for tok in fr if tok not in env_m and tok != v and tok in mid_set]
        key = (jac, ov, -len(extra))
        if best is None or key > best[0]:
            best = (key, extra)
        for tok in fr:
            if tok in mid_set and tok != v:
                bag.append(tok)
                weighted[tok] += jac
    extras = best[1] if best is not None else []
    return weighted, bag, extras


def pick_peaked(weighted):
    if not weighted:
        return None
    ranked = weighted.most_common(2)
    top, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if s1 - s2 > MARGIN:
        return top
    return None


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
    weighted, bag, extras = star_scores(g, by, v, s_q, env_m, mid_set)
    if not bag:
        return None
    uniq = extras[0] if len(extras) == 1 else None
    wvote = weighted.most_common(1)[0][0] if weighted else None
    peaked = pick_peaked(weighted)
    maj = Counter(bag).most_common(1)[0][0]
    rnd = rng.choice(bag)
    return dict(
        uniq=int(uniq == held),
        uniq_c=int(uniq is not None),
        wvote=int(wvote == held),
        peaked=int(peaked == held),
        peaked_c=int(peaked is not None),
        maj=int(maj == held),
        rand=int(rnd == held),
        copy=int(v == held),
        cover=int(held in set(bag)),
    )


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
        f"575 ball  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    uniq = mean(rows, "uniq")
    uniq_c = mean(rows, "uniq_c")
    wvote = mean(rows, "wvote")
    peaked = mean(rows, "peaked")
    peaked_c = mean(rows, "peaked_c")
    maj = mean(rows, "maj")
    rnd = mean(rows, "rand")
    copy = mean(rows, "copy")
    committed_u = [row for row in rows if row["uniq_c"]]
    committed_p = [row for row in rows if row["peaked_c"]]
    hit_u = mean(committed_u, "uniq")
    hit_p = mean(committed_p, "peaked")
    d_maj = wvote - maj
    d_rnd = wvote - rnd
    void = n < 40 or cover < 0.15
    gate = (not void) and d_maj > 0.05 and d_rnd > 0.05

    print(f"n {n}  cover {cover:.3f}  copy {copy:.3f}")
    print(
        f"uniq {uniq:.3f} (c {uniq_c:.3f} hitc {hit_u:.3f})  "
        f"peaked {peaked:.3f} (c {peaked_c:.3f} hitc {hit_p:.3f})"
    )
    print(f"wvote {wvote:.3f}  maj {maj:.3f}  rand {rnd:.3f}")
    print(f"d_maj {d_maj:+.3f}  d_rand {d_rnd:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO BALL: env-weighted vote beats unweighted majority.")
    else:
        print("STOP: env does not move the pin star. 511 remains the ceiling.")
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover, copy=copy,
        uniq=uniq, uniq_c=uniq_c, hit_uniq=hit_u,
        wvote=wvote, peaked=peaked, peaked_c=peaked_c, hit_peaked=hit_p,
        maj=maj, rand=rnd, d_maj=d_maj, d_rand=d_rnd,
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
