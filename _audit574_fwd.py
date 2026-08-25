"""574: pin → forward → close with a tape place → match the original.
Not XOR. Not CE. Not a hole walking back to the last hop.
Last successful hop pins v (a line we actually read). The next line also
contains v; hide a DIFFERENT mid token there. Walk FORWARD from the pin:
other mentions of v, query slot out, never the same occurrence. Arrive at
the foreign frame that overlaps the query environment. Close the hole with
the unique extra token sitting on T in that frame. Score against the
original tape.
Rivals
    maj   most common companion of v (511 on the pin star; ignores the new line)
    rand  random companion of v
    copy  fill with v itself — should be ~0; the hole is not the pin
VOID  n < 40  OR  cover (held among pin companions) < 0.15
GATE  pin − maj > 0.05  AND  pin − rand > 0.05
Refuse when the arrived frame has 0 or 2+ extras — that is honesty, not a
second scorer. Overall hit counts refuse as miss. hit|commit is printed so
a silent-but-accurate pin is not read as «no mechanism».
Same-line sequential walk is not a hand: query slot is out. Φ stays out.
    python _check574_fwd.py
    python _audit574_fwd.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage574_fwd.json")


def prefix_windows(pool, length, n_win):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    return blocks[: min(n_win, len(blocks))]


def env_mid(env, mid_set, high_set):
    return (env & mid_set) - high_set or (env - high_set)


def pin_arrive(g, by, v, s_q, env_m, mid_set):
    """Best foreign frame of v. Query slot out. Extra = dest \\ env \\ {v}."""
    best = None
    bag = []
    for t in by.get(v, ()):
        if t == s_q:
            continue
        fr = set(comps(g, t, v))
        bag.extend(tok for tok in fr if tok in mid_set)
        ov = len(fr & env_m)
        jac = ov / max(len(fr | env_m), 1)
        extra = [tok for tok in fr if tok not in env_m and tok != v and tok in mid_set]
        key = (jac, ov, -len(extra))
        if best is None or key > best[0]:
            best = (key, extra, t)
    return best, bag


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
    best, bag = pin_arrive(g, by, v, s_q, env_m, mid_set)
    if not bag:
        return None
    extra = best[1] if best is not None else []
    pin_fill = extra[0] if len(extra) == 1 else None
    maj = Counter(bag).most_common(1)[0][0]
    rnd = rng.choice(bag)
    return dict(
        pin=int(pin_fill == held),
        commit=int(pin_fill is not None),
        maj=int(maj == held),
        rand=int(rnd == held),
        copy=int(v == held),
        cover=int(held in set(bag)),
        refuse=int(pin_fill is None),
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
        for s_q in slots[1:args.cap_probe + 1]:
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
        f"574 pin->forward  {path}  {kind}  windows={len(windows)}  "
        f"prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    pin = mean(rows, "pin")
    maj = mean(rows, "maj")
    rnd = mean(rows, "rand")
    copy = mean(rows, "copy")
    commit = mean(rows, "commit")
    refuse = mean(rows, "refuse")
    committed = [row for row in rows if row["commit"]]
    hit_c = mean(committed, "pin")
    d_maj = pin - maj
    d_rnd = pin - rnd
    void = n < 40 or cover < 0.15
    gate = (not void) and d_maj > 0.05 and d_rnd > 0.05

    print(f"n {n}  cover {cover:.3f}  commit {commit:.3f}  refuse {refuse:.3f}")
    print(
        f"pin {pin:.3f}  (hit|commit {hit_c:.3f})  maj {maj:.3f}  "
        f"rand {rnd:.3f}  copy {copy:.3f}"
    )
    print(f"d_maj {d_maj:+.3f}  d_rand {d_rnd:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO FWD: pin->foreign frame closes the next hole better than maj/rand.")
    else:
        print("STOP: forward-from-pin does not beat the pin-star majority.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover,
        pin=pin, hit_commit=hit_c, maj=maj, rand=rnd, copy=copy,
        commit=commit, refuse=refuse, d_maj=d_maj, d_rand=d_rnd,
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
