"""578: Φ ranks unique-extra frames. Remainder of 577.

577 GO: unique-then-jac ~0.38 beats maj ~0.25; U_any ~0.72. About 7
unique-extra frames, ~1.7 name held. Jac already picks among them
better than chance. The leftover +0.33 is a ranking of FRAMES, not
letters in Φ.

    offer   foreign frames of pin v with |extra|==1  (577's set)
    feat    jac, ov, width, df(extra), log mentions   — counts, no token
    teacher extra == held
    loss    softplus(-(s_pos − s_neg)) on a true unique-held frame vs
            another unique-extra frame of the same episode
    fill    extra of argmax Φ, from T

Arms: B Φ / C labels shuffled inside the episode / D 577 u_jac1.
Same prefix windows, first 70% train, last 30% test. No shuffle.

VOID  test n < 40  OR  cover < 0.15
GATE  B − D > 0.05  AND  B − maj > 0.05  on test
      (must beat unique-then-jac, not only majority)

    python _check578_frrank.py
    python _audit578_frrank.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

OUT = Path("results/_stage578_frrank.json")
FDIM = 5


def prefix_windows(pool, length, n_win):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    return blocks[: min(n_win, len(blocks))]


def env_mid(env, mid_set, high_set):
    return (env & mid_set) - high_set or (env - high_set)


def feat_of(jac, ov, width, df, mentions_n):
    return [
        jac,
        min(ov, 4) / 4.0,
        min(width, 8) / 8.0,
        min(df, 64) / 64.0,
        math.log1p(mentions_n) / math.log(33.0),
    ]


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FDIM, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def episode_cands(g, by, v, s_q, env_m, mid_set, held):
    bag = []
    best_all = None
    cands = []
    n_u = 0
    for t in by.get(v, ()):
        if t == s_q:
            continue
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        width = max(len(fr), 1)
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
        if len(extra) != 1:
            continue
        tok = extra[0]
        cands.append(dict(
            feat=feat_of(
                jac, ov, width,
                g["df"].get(tok, len(by.get(tok, ()))),
                len(by.get(tok, ())),
            ),
            jac=jac,
            y=int(tok == held),
            tok=tok,
        ))
    extras = best_all[1] if best_all is not None else []
    maj = Counter(bag).most_common(1)[0][0] if bag else None
    return dict(
        cands=cands,
        u_best=int(len(extras) == 1 and extras[0] == held),
        u_any=int(n_u > 0),
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
    row = episode_cands(g, by, v, s_q, env_m, mid_set, held)
    if not row["has_bag"] or not row["cands"]:
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


def collect_windows(windows, args, rng):
    rows = []
    for lines in windows:
        rows.extend(collect(lines, args, rng))
    return rows


def pairs_of(rows, rng, shuffle_y=False):
    out = []
    for row in rows:
        cands = row["cands"]
        ys = [c["y"] for c in cands]
        if shuffle_y:
            rng.shuffle(ys)
        pos = [c for c, y in zip(cands, ys) if y]
        neg = [c for c, y in zip(cands, ys) if not y]
        if not pos or not neg:
            continue
        for _ in range(min(4, len(pos) * len(neg))):
            out.append((rng.choice(pos)["feat"], rng.choice(neg)["feat"]))
    return out


def train_ranker(pairs, seed, steps=800):
    torch.manual_seed(seed)
    net = Ranker()
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    if not pairs:
        return net
    for step in range(steps):
        a, b = pairs[step % len(pairs)]
        s_pos = net(torch.tensor(a, dtype=torch.float32))
        s_neg = net(torch.tensor(b, dtype=torch.float32))
        loss = F.softplus(-(s_pos - s_neg))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def pick(row, net=None):
    cands = row["cands"]
    if net is None:
        return max(cands, key=lambda c: c["jac"])
    with torch.no_grad():
        scores = [
            float(net(torch.tensor(c["feat"], dtype=torch.float32)))
            for c in cands
        ]
    return cands[max(range(len(cands)), key=lambda i: scores[i])]


def eval_rows(rows, net=None):
    if not rows:
        return dict(n=0, hit=0.0, u_best=0.0, u_any=0.0, maj=0.0, cover=0.0, copy=0.0)
    hit = 0
    for row in rows:
        hit += int(pick(row, net)["y"] == 1)
    n = len(rows)
    return dict(
        n=n,
        hit=hit / n,
        u_best=sum(r["u_best"] for r in rows) / n,
        u_any=sum(r["u_any"] for r in rows) / n,
        maj=sum(r["maj"] for r in rows) / n,
        cover=sum(r["cover"] for r in rows) / n,
        copy=sum(r["copy"] for r in rows) / n,
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
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    cut = max(1, int(0.7 * len(windows)))
    train_w, test_w = windows[:cut], windows[cut:]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(
        f"578 frrank  {path}  {kind}  train_w={len(train_w)} test_w={len(test_w)}",
        flush=True,
    )

    train_rows = collect_windows(train_w, args, rng)
    test_rows = collect_windows(test_w, args, rng)
    pairs_b = pairs_of(train_rows, random.Random(args.seed + 1), shuffle_y=False)
    pairs_c = pairs_of(train_rows, random.Random(args.seed + 2), shuffle_y=True)
    net_b = train_ranker(pairs_b, args.seed, args.steps)
    net_c = train_ranker(pairs_c, args.seed + 3, args.steps)
    B = eval_rows(test_rows, net_b)
    C = eval_rows(test_rows, net_c)
    D = eval_rows(test_rows, None)
    n = B["n"]
    cover = B["cover"]
    d_d = B["hit"] - D["hit"]
    d_maj = B["hit"] - B["maj"]
    d_c = B["hit"] - C["hit"]
    d_or = B["u_any"] - B["hit"]
    void = n < 40 or cover < 0.15
    gate = (not void) and d_d > 0.05 and d_maj > 0.05

    print(
        f"test n {n}  cover {cover:.3f}  copy {B['copy']:.3f}  "
        f"pairs {len(pairs_b)}"
    )
    print(
        f"B {B['hit']:.3f}  C {C['hit']:.3f}  D(ujac1) {D['hit']:.3f}  "
        f"maj {B['maj']:.3f}  u_any {B['u_any']:.3f}  u_best {B['u_best']:.3f}"
    )
    print(
        f"B-D {d_d:+.3f}  B-maj {d_maj:+.3f}  B-C {d_c:+.3f}  "
        f"u_any-B {d_or:+.3f}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: test empty.")
    elif gate:
        print("GO PHI: frame ranker beats unique-then-jac and maj.")
    else:
        print("STOP: Phi does not beat 577's rule on test.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_train_w=len(train_w), n_test_w=len(test_w),
        n_train=len(train_rows), n_test=n, n_pairs=len(pairs_b),
        cover=cover, copy=B["copy"],
        B=B["hit"], C=C["hit"], D=D["hit"],
        maj=B["maj"], u_any=B["u_any"], u_best=B["u_best"],
        d_d=d_d, d_maj=d_maj, d_c=d_c, d_or=d_or,
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
