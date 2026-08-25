"""582: ceiling for 'hop2 only after a correct hop1 that is not the hole'.

User object (not 580, not 581):
    hop1  unique extra e of pin v     e may be 'tree', not 'apples'
    hop2  only if that e is the kind of extra whose star uniquely
          names held
    check hop2 against the original (held), not against e

580 STOP: pmi-top extra is usually a dead child (pear → jam), ~0.01.
581 sibling extras of v is a different question.

This is the missing existence test:
    among unique extras e of v with e != held,
    does any e have held as a unique extra of e?

If oracle_bridge ≈ 0, there is no intermediate: the only 'correct hop1'
on this tape is e == held, and hop2-from-correct is 440 (add a fact),
not 'lead to apples'. Gradient on hop2-to-held is empty.

If oracle_bridge > 0.05, the path exists; then hop1 should pick a
bridge extra, hop2 fills held. That ranking is the next step, not Φ.

Hands (same episode, query out of co):
    fill1     579  hop1 extra == held
    pmi_h2    580  hop2 from pmi-top extra (even if e == held, skip)
    bridge    oracle: some unique e ≠ held of v uniquely names held
    follow    among those e, pick max pmi, then max-pmi unique extra of
              e == held  (algebraic hop2 after a real bridge)

VOID  n < 40  OR  cover < 0.15
GATE  bridge > 0.05
Print follow − pmi_h2: does choosing a bridge beat 580's dead child.

    python _check582_bridge.py
    python _audit582_bridge.py --seed 1337 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage582_bridge.json")


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
        extra = [tok for tok in fr if tok not in env_m and tok != node and tok in mid_set]
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


def walk(g, by, v, s_q, env_m, mid_set, held, co, df, n_fr):
    uniq_v, bag = unique_extras(g, by, v, s_q, env_m, mid_set, co, df, n_fr)
    toks = [tok for _p, tok in uniq_v]
    fill1 = int(bool(toks) and toks[0] == held)
    u_any = int(held in toks)

    pmi_h2 = 0
    if toks:
        e1 = toks[0]
        if e1 != held:
            uniq_e, _ = unique_extras(g, by, e1, s_q, env_m, mid_set, co, df, n_fr)
            if uniq_e:
                pmi_h2 = int(uniq_e[0][1] == held)

    bridges = []
    for _p, e in uniq_v:
        if e == held:
            continue
        uniq_e, _ = unique_extras(g, by, e, s_q, env_m, mid_set, co, df, n_fr)
        if any(tok == held for _q, tok in uniq_e):
            bridges.append((_p, e, uniq_e))
    bridge = int(bool(bridges))
    follow = 0
    if bridges:
        _p, e, uniq_e = max(bridges, key=lambda item: item[0])
        if uniq_e:
            follow = int(uniq_e[0][1] == held)

    maj = Counter(bag).most_common(1)[0][0] if bag else None
    return dict(
        fill1=fill1, pmi_h2=pmi_h2, bridge=bridge, follow=follow,
        u_any=u_any, maj=int(maj == held),
        cover=int(held in set(bag)), copy=int(v == held),
        has_bag=int(bool(bag)), n_ord=len(toks), n_br=len(bridges),
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
        f"582 bridge  {path}  {kind}  windows={len(windows)}  prefix, no shuffle",
        flush=True,
    )

    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))

    n = len(rows)
    cover = mean(rows, "cover")
    fill1 = mean(rows, "fill1")
    pmi_h2 = mean(rows, "pmi_h2")
    bridge = mean(rows, "bridge")
    follow = mean(rows, "follow")
    u_any = mean(rows, "u_any")
    maj = mean(rows, "maj")
    copy = mean(rows, "copy")
    void = n < 40 or cover < 0.15
    gate = (not void) and bridge > 0.05

    print(
        f"n {n}  cover {cover:.3f}  copy {copy:.3f}  "
        f"n_ord {mean(rows, 'n_ord'):.2f}  n_br {mean(rows, 'n_br'):.2f}"
    )
    print(
        f"fill1 {fill1:.3f}  pmi_h2 {pmi_h2:.3f}  bridge {bridge:.3f}  "
        f"follow {follow:.3f}  u_any {u_any:.3f}  maj {maj:.3f}"
    )
    print(f"follow-pmi_h2 {follow - pmi_h2:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: too few episodes or pin cannot reach the hole.")
    elif gate:
        print("GO BRIDGE: a unique extra that is not held still uniquely names held.")
    else:
        print("STOP: no intermediate. Only hop1==held is a correct first step.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n=n, cover=cover, copy=copy,
        fill1=fill1, pmi_h2=pmi_h2, bridge=bridge, follow=follow,
        u_any=u_any, maj=maj, d_follow=follow - pmi_h2,
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
