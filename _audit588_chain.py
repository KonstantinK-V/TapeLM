"""588: hop2 only after THIS hand's hop1 hit. Same episode, not a new idea.

hop1  unique extra of v  == held1  -> pin held1
hop2  unique extra of pin == held2 on a NEW frame of that word
miss  -> refuse, no further hop

Hands share the hop1 exam; hop2 is scored only if that hand filled hop1.
Train: hop1 extras + teacher-forced hop2 after true held1 (n_follow).
Q has depth in the key: hop2 does not train hop1.

GATE  hop2: B-A > 0.05 AND B-C > 0.05
VOID  n_h2 < 40  OR  cover2 < 0.15
Print hop1 next to hop2. 587 not required to GO.

    python _check588_chain.py
    python _audit588_chain.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage588_chain.json")


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


def feat(pmi, n_ord, df_extra, n_slots, depth):
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
    return (depth, pb, ob, bb)


class QTab:
    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def touch(self, key, r):
        self.n[key] += 1
        self.s[key] += (r - self.s[key]) / self.n[key]

    def get(self, key):
        return self.s[key] if self.n[key] else 0.0


def rows_of(g, by, node, s_q, env_m, mid_set, co, df, n_fr, n_slots, depth):
    scored, bag = unique_extras(g, by, node, s_q, env_m, mid_set, co, df, n_fr)
    n_ord = len(scored)
    rows = [
        (feat(pmi, n_ord, df.get(tok, 1), n_slots, depth), pmi, tok)
        for pmi, tok in scored
    ]
    return rows, bag


def hole_at(g, by, node, s_q, mid_set, high_set, rng, co, frames, n_fr, df, depth):
    frame = list(comps(g, s_q, node))
    if len(frame) < 2:
        return None
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    if held not in mid_set or held == node:
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
        rows, bag = rows_of(
            g, by, node, s_q, env_m, mid_set, co, df, n_use, g["n"], depth,
        )
    finally:
        if qtoks:
            adjust_frame_stats(co, df, qtoks, +1)
    if not bag:
        return None
    return dict(rows=rows, held=held, bag=bag, s_q=s_q, node=node)


def hop2_of(g, by, pin, s_used, mid_set, high_set, rng, co, frames, n_fr, df):
    slots = [s for s in by.get(pin, ()) if s != s_used]
    rng.shuffle(slots)
    for s_q in slots[:4]:
        h = hole_at(
            g, by, pin, s_q, mid_set, high_set, rng, co, frames, n_fr, df, 2,
        )
        if h is not None:
            return h
    return None


def one_episode(g, by, v, s_pin, s_q, mid_set, high_set, rng, co, frames, n_fr, df):
    if s_pin == s_q:
        return None
    h1 = hole_at(
        g, by, v, s_q, mid_set, high_set, rng, co, frames, n_fr, df, 1,
    )
    if h1 is None:
        return None
    h2 = hop2_of(
        g, by, h1["held"], s_q, mid_set, high_set, rng, co, frames, n_fr, df,
    )
    return dict(h1=h1, h2=h2, copy=int(v == h1["held"]))


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
        h1 = ep["h1"]
        held1 = h1["held"]
        for key, _p, tok in h1["rows"]:
            tab.touch(key, 1.0 if tok == held1 else 0.0)
        h2 = ep["h2"]
        if h2 is None:
            continue
        held2 = h2["held"]
        for key, _p, tok in h2["rows"]:
            tab.touch(key, 1.0 if tok == held2 else 0.0)


def fill_of(rows, held, how, tab, rng):
    if not rows:
        return 0
    if how == "pmi":
        return int(pick_pmi(rows) == held)
    if how == "q":
        return int(pick_q(rows, tab) == held)
    return int(pick_rnd(rows, rng) == held)


def score_eps(eps, tab, rng):
    n = c1 = f1a = f1b = f1c = r1 = 0
    n2 = c2 = r2 = 0
    n2a = n2b = n2c = 0
    f2a = f2b = f2c = 0
    for ep in eps:
        h1 = ep["h1"]
        n += 1
        c1 += int(h1["held"] in set(h1["bag"]))
        toks1 = {tok for _k, _p, tok in h1["rows"]}
        r1 += int(h1["held"] in toks1)
        a = fill_of(h1["rows"], h1["held"], "pmi", tab, rng)
        b = fill_of(h1["rows"], h1["held"], "q", tab, rng)
        c = fill_of(h1["rows"], h1["held"], "rnd", tab, rng)
        f1a += a
        f1b += b
        f1c += c
        h2 = ep["h2"]
        if h2 is None:
            continue
        n2 += 1
        c2 += int(h2["held"] in set(h2["bag"]))
        toks2 = {tok for _k, _p, tok in h2["rows"]}
        r2 += int(h2["held"] in toks2)
        if a:
            n2a += 1
            f2a += fill_of(h2["rows"], h2["held"], "pmi", tab, rng)
        if b:
            n2b += 1
            f2b += fill_of(h2["rows"], h2["held"], "q", tab, rng)
        if c:
            n2c += 1
            f2c += fill_of(h2["rows"], h2["held"], "rnd", tab, rng)
    if n <= 0:
        return None
    return dict(
        n=n, cover1=c1 / n, fill1_pmi=f1a / n, fill1_q=f1b / n,
        fill1_rnd=f1c / n, r1=r1 / n,
        n2=n2, cover2=(c2 / n2 if n2 else 0.0),
        n2_pmi=n2a, n2_q=n2b, n2_rnd=n2c,
        fill2_pmi=(f2a / n2a if n2a else 0.0),
        fill2_q=(f2b / n2b if n2b else 0.0),
        fill2_rnd=(f2c / n2c if n2c else 0.0),
        r2=(r2 / n2 if n2 else 0.0),
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
        f"588 hop2-after-hit  {path}  {kind}  train_w={len(train_w)} eval_w={len(eval_w)}",
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

    void = ev["n2"] < 40 or ev["cover2"] < 0.15
    d_pmi = ev["fill2_q"] - ev["fill2_pmi"]
    d_rnd = ev["fill2_q"] - ev["fill2_rnd"]
    gate = (not void) and d_pmi > 0.05 and d_rnd > 0.05
    print(
        f"hop1 n {ev['n']}  PMI {ev['fill1_pmi']:.3f}  Q {ev['fill1_q']:.3f}  "
        f"rnd {ev['fill1_rnd']:.3f}  r1 {ev['r1']:.3f}"
    )
    print(
        f"hop2 n {ev['n2']}  cover {ev['cover2']:.3f}  "
        f"PMI {ev['fill2_pmi']:.3f}  Q {ev['fill2_q']:.3f}  "
        f"rnd {ev['fill2_rnd']:.3f}  r2 {ev['r2']:.3f}"
    )
    print(f"hop2 Q-PMI {d_pmi:+.3f}  Q-rnd {d_rnd:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: after hop1 hit there is no hop2 exam.")
    elif gate:
        print("GO CHAIN: hop2 after this-hand success, counts beat PMI.")
    else:
        print("STOP: hop2 exam exists; PMI still the policy after pin.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_train_w=len(train_w), n_eval_w=len(eval_w),
        n_train=len(train_eps), n_keys=len(tab.s),
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        d_pmi=d_pmi, d_rnd=d_rnd,
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
