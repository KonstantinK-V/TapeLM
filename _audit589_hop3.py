"""589: hop3 after this-hand hop2 hit. PMI frozen. No Q.

Machine (v1 walk, 579/588):
  pin v → unique extras vs query env → PMI-top
  extra == held → pin that word, NEW hole on T
  else REFUSE

Hands  A PMI | C random unique extra
GATE   fill3_pmi − fill3_rnd > 0.05
VOID   n3_pmi < 40 OR cover3 < 0.15

    python _check589_hop3.py
    python _audit589_hop3.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage589_hop3.json")


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


def hole_at(g, by, node, s_q, mid_set, high_set, rng, co, frames, n_fr, df):
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
        scored, bag = unique_extras(
            g, by, node, s_q, env_m, mid_set, co, df, n_use,
        )
    finally:
        if qtoks:
            adjust_frame_stats(co, df, qtoks, +1)
    if not bag:
        return None
    toks = [tok for _p, tok in scored]
    return dict(toks=toks, held=held, bag=bag, s_q=s_q, node=node)


def next_hole(g, by, pin, s_used, mid_set, high_set, rng, co, frames, n_fr, df):
    slots = [s for s in by.get(pin, ()) if s != s_used]
    rng.shuffle(slots)
    for s_q in slots[:4]:
        h = hole_at(
            g, by, pin, s_q, mid_set, high_set, rng, co, frames, n_fr, df,
        )
        if h is not None:
            return h
    return None


def one_episode(g, by, v, s_pin, s_q, mid_set, high_set, rng, co, frames, n_fr, df):
    if s_pin == s_q:
        return None
    h1 = hole_at(
        g, by, v, s_q, mid_set, high_set, rng, co, frames, n_fr, df,
    )
    if h1 is None:
        return None
    h2 = next_hole(
        g, by, h1["held"], s_q, mid_set, high_set, rng, co, frames, n_fr, df,
    )
    h3 = None
    if h2 is not None:
        h3 = next_hole(
            g, by, h2["held"], h2["s_q"], mid_set, high_set, rng, co, frames, n_fr, df,
        )
    return dict(h1=h1, h2=h2, h3=h3)


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


def pick_pmi(toks):
    return toks[0] if toks else None


def pick_rnd(toks, rng):
    return toks[rng.randrange(len(toks))] if toks else None


def hit(toks, held, how, rng):
    if not toks:
        return 0
    if how == "pmi":
        return int(pick_pmi(toks) == held)
    return int(pick_rnd(toks, rng) == held)


def score_eps(eps, rng):
    n = c1 = f1a = f1c = r1 = 0
    n2 = c2 = r2 = n2a = n2c = f2a = f2c = 0
    n3 = c3 = r3 = n3a = n3c = f3a = f3c = 0
    for ep in eps:
        h1 = ep["h1"]
        n += 1
        c1 += int(h1["held"] in set(h1["bag"]))
        r1 += int(h1["held"] in h1["toks"])
        a1 = hit(h1["toks"], h1["held"], "pmi", rng)
        c1h = hit(h1["toks"], h1["held"], "rnd", rng)
        f1a += a1
        f1c += c1h
        h2 = ep["h2"]
        if h2 is None:
            continue
        n2 += 1
        c2 += int(h2["held"] in set(h2["bag"]))
        r2 += int(h2["held"] in h2["toks"])
        a2 = c2h = 0
        if a1:
            n2a += 1
            a2 = hit(h2["toks"], h2["held"], "pmi", rng)
            f2a += a2
        if c1h:
            n2c += 1
            c2h = hit(h2["toks"], h2["held"], "rnd", rng)
            f2c += c2h
        h3 = ep["h3"]
        if h3 is None:
            continue
        n3 += 1
        c3 += int(h3["held"] in set(h3["bag"]))
        r3 += int(h3["held"] in h3["toks"])
        if a1 and a2:
            n3a += 1
            f3a += hit(h3["toks"], h3["held"], "pmi", rng)
        if c1h and c2h:
            n3c += 1
            f3c += hit(h3["toks"], h3["held"], "rnd", rng)
    if n <= 0:
        return None
    return dict(
        n=n, cover1=c1 / n, fill1_pmi=f1a / n, fill1_rnd=f1c / n, r1=r1 / n,
        n2=n2, cover2=(c2 / n2 if n2 else 0.0),
        n2_pmi=n2a, n2_rnd=n2c,
        fill2_pmi=(f2a / n2a if n2a else 0.0),
        fill2_rnd=(f2c / n2c if n2c else 0.0),
        r2=(r2 / n2 if n2 else 0.0),
        n3=n3, cover3=(c3 / n3 if n3 else 0.0),
        n3_pmi=n3a, n3_rnd=n3c,
        fill3_pmi=(f3a / n3a if n3a else 0.0),
        fill3_rnd=(f3c / n3c if n3c else 0.0),
        r3=(r3 / n3 if n3 else 0.0),
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
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"589 hop3  {path}  {kind}  windows={len(windows)}", flush=True)

    eps = []
    for lines in windows:
        eps.extend(collect(lines, args, rng))
    ev = score_eps(eps, random.Random(args.seed + 17))
    if ev is None:
        print("VOID: no episodes")
        return 0

    void = ev["n3_pmi"] < 40 or ev["cover3"] < 0.15
    d_rnd = ev["fill3_pmi"] - ev["fill3_rnd"]
    gate = (not void) and d_rnd > 0.05
    print(
        f"hop1 n {ev['n']}  PMI {ev['fill1_pmi']:.3f}  rnd {ev['fill1_rnd']:.3f}  r1 {ev['r1']:.3f}"
    )
    print(
        f"hop2 n_pmi {ev['n2_pmi']}  PMI {ev['fill2_pmi']:.3f}  rnd {ev['fill2_rnd']:.3f}  r2 {ev['r2']:.3f}"
    )
    print(
        f"hop3 n_pmi {ev['n3_pmi']}  PMI {ev['fill3_pmi']:.3f}  rnd {ev['fill3_rnd']:.3f}  "
        f"r3 {ev['r3']:.3f}  cover3 {ev['cover3']:.3f}"
    )
    print(f"hop3 PMI-rnd {d_rnd:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: after two hits there is no hop3 exam.")
    elif gate:
        print("GO HOP3: chain lives at depth 3. PMI frozen.")
    else:
        print("STOP: hop3 exam exists; PMI does not beat random.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), n_eps=len(eps),
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), d_rnd=d_rnd,
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
