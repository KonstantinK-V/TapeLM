"""587: hop-1 train. Miss → refuse. Next hop only after success (not in this run).

Offer     unique extras of pin v (579 legs, frozen)
Teacher   extra == held          held not in features
Hands     A PMI-top (579) | B Q[counts] | C random extra
GATE      B−A > 0.05  AND  B−C > 0.05
VOID      n_eval < 40  OR  cover < 0.15  OR  r1 − fill_pmi < 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats

OUT = Path("results/_stage587_train.json")


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
    for _v, slots in by.items():
        for s in slots:
            if s in seen:
                continue
            seen.add(s)
            toks = set(comps(g, s, _v))
            toks.add(_v)
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


def feat(pmi, n_ord, df_extra, n_slots):
    pb = 0 if pmi < 0 else (1 if pmi < 1.0 else 2)
    ob = 1 if n_ord == 1 else (2 if n_ord == 2 else 3)
    if n_slots <= 0:
        bb = 1
    elif df_extra * 4 < n_slots:
        bb = 0
    elif df_extra * 2 > n_slots:
        bb = 2
    else:
        bb = 1
    return (pb, ob, bb)


class QTab:
    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def touch(self, key, r):
        self.n[key] += 1
        self.s[key] += (r - self.s[key]) / self.n[key]

    def get(self, key):
        return self.s[key] if self.n[key] else 0.0


def offer_of(g, by, v, s_q, env_m, mid_set, co, df, n_fr, n_slots):
    scored, bag = unique_extras(g, by, v, s_q, env_m, mid_set, co, df, n_fr)
    n_ord = len(scored)
    rows = []
    for pmi, tok in scored:
        rows.append((feat(pmi, n_ord, df.get(tok, 1), n_slots), pmi, tok))
    return rows, bag


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
        rows, bag = offer_of(
            g, by, v, s_q, env_m, mid_set, co, df, n_use, g["n"],
        )
    finally:
        if qtoks:
            adjust_frame_stats(co, df, qtoks, +1)
    if not bag:
        return None
    return dict(
        rows=rows, held=held, bag=bag, copy=int(v == held),
        has=int(bool(rows)),
    )


def collect(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return []
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    co, frames, n_fr = co_table(g, by)
    df = g.get("df") or {tok: len(slots) for tok, slots in by.items()}
    out = []
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
                out.append(row)
    return out


def pick_pmi(rows):
    return rows[0][2] if rows else None


def pick_rnd(rows, rng):
    return rows[rng.randrange(len(rows))][2] if rows else None


def pick_q(rows, tab):
    if not rows:
        return None
    best = None
    best_s = None
    for key, pmi, tok in rows:
        s = tab.get(key)
        if best is None or s > best_s or (s == best_s and pmi > best[1]):
            best = (tok, pmi)
            best_s = s
    return best[0]


def train(eps, tab):
    for ep in eps:
        held = ep["held"]
        rows = ep["rows"]
        if not rows:
            continue
        for key, _pmi, tok in rows:
            tab.touch(key, 1.0 if tok == held else 0.0)


def score_eps(eps, tab, rng):
    n = fill_a = fill_b = fill_c = r1 = cover = copy = refuse = 0
    for ep in eps:
        n += 1
        held = ep["held"]
        rows = ep["rows"]
        bag = set(ep["bag"])
        cover += int(held in bag)
        copy += ep["copy"]
        if not rows:
            refuse += 1
            continue
        toks = {tok for _k, _p, tok in rows}
        r1 += int(held in toks)
        fill_a += int(pick_pmi(rows) == held)
        fill_b += int(pick_q(rows, tab) == held)
        fill_c += int(pick_rnd(rows, rng) == held)
    if n <= 0:
        return None
    return dict(
        n=n, cover=cover / n, copy=copy / n, refuse=refuse / n,
        fill_pmi=fill_a / n, fill_q=fill_b / n, fill_rnd=fill_c / n,
        r1=r1 / n,
    )


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
    cut = max(1, int(0.7 * len(windows)))
    train_w, eval_w = windows[:cut], windows[cut:]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(
        f"587 hop1-train  {path}  {kind}  train_w={len(train_w)} eval_w={len(eval_w)}",
        flush=True,
    )

    train_eps, eval_eps = [], []
    for lines in train_w:
        train_eps.extend(collect(lines, args, rng))
    for lines in eval_w:
        eval_eps.extend(collect(lines, args, rng))

    tab = QTab()
    train(train_eps, tab)
    ev = score_eps(eval_eps, tab, random.Random(args.seed + 17))
    if ev is None:
        print("VOID: no eval episodes")
        return 0

    room = ev["r1"] - ev["fill_pmi"]
    void = ev["n"] < 40 or ev["cover"] < 0.15 or room < 0.05
    d_pmi = ev["fill_q"] - ev["fill_pmi"]
    d_rnd = ev["fill_q"] - ev["fill_rnd"]
    gate = (not void) and d_pmi > 0.05 and d_rnd > 0.05

    print(
        f"n {ev['n']}  cover {ev['cover']:.3f}  copy {ev['copy']:.3f}  "
        f"refuse {ev['refuse']:.3f}  keys {len(tab.s)}"
    )
    print(
        f"PMI {ev['fill_pmi']:.3f}  Q {ev['fill_q']:.3f}  rnd {ev['fill_rnd']:.3f}  "
        f"r1 {ev['r1']:.3f}  room {room:+.3f}"
    )
    print(f"Q-PMI {d_pmi:+.3f}  Q-rnd {d_rnd:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few eval, or PMI already at r1.")
    elif gate:
        print("GO TRAIN: counts beat PMI on unique extras. Hop2 only after this hit.")
    else:
        print("STOP: PMI remains the hop1 policy (38.3). Compose is 588, after success.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_train_w=len(train_w), n_eval_w=len(eval_w),
        n_train=len(train_eps), n_keys=len(tab.s),
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        d_pmi=d_pmi, d_rnd=d_rnd, room=room,
        **ev,
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
