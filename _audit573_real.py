"""573: 572 with a real-training cut and the lookahead leak closed.

572 GO 1/3 was PRE ~0.54 vs an unlucky coin (0.42 / 0.34). Same PRE lost to
majority on 2890. Lookahead 0.80–0.87 is not a scorer: n_mid / n_next walk the
destination frame and count `held` (ha = held in fra). Same instrument as the
label. 400's Return-bit.

This run:

* windows in FILE ORDER, train = prefix, test = suffix (522). No shuffle.
* coin_fair = 0.5 on XOR (analytic). rng-coin still printed, not gated.
* PRE = 572 gated features (df / mentions / env size). No dest frame.
* TAPE = dest ∩ env only (env already excludes held). No scan of fr\\env.
* RULE = higher |fra ∩ env| — 572 rank, the honest rival for TAPE.
* leaky LOOK (n_mid on full fr) printed, not gated, so the drop is visible.

VOID  n_xor_test < 40
GATE  PRE − {0.5, null, rank, maj} all > 0.05   (same bar as 572)
TAPE is diagnostic: if TAPE−RULE ≤ 0.05, Φ on dest-frame is 38.3.

567 frozen. No synthetic.

    python _check573_real.py
    python _audit573_real.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit571_xorcensus import stand_frame

OUT = Path("results/_stage573_real.json")


def prefix_windows(pool, length, n_win):
    """Contiguous blocks in file order. No shuffle."""
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    return blocks[: min(n_win, len(blocks))]


def pre_features(g, by, addr, env_m):
    mentions_n = len(by.get(addr, []))
    df = g["df"].get(addr, mentions_n)
    return [
        min(mentions_n, 32) / 32.0,
        min(df, 64) / 64.0,
        min(len(env_m), 8) / 8.0,
        min(df / max(mentions_n, 1), 8.0) / 8.0,
        math.log1p(mentions_n) / math.log(33.0),
    ]


def tape_features(g, by, addr, env_m):
    """Legal dest evidence: only dest ∩ env. query not in env."""
    slots = list(by.get(addr, []))
    best_ov = best_jac = 0.0
    second = 0.0
    for t in slots[:8]:
        fr = set(comps(g, t, addr))
        inter = fr & env_m
        ov = len(inter)
        jac = ov / max(len(fr | env_m), 1)
        if jac > best_jac or (jac == best_jac and ov > best_ov):
            second = best_jac
            best_jac, best_ov = jac, ov
        elif jac > second:
            second = jac
    return [
        min(best_ov, 4) / 4.0,
        best_jac,
        best_jac - second,
        min(len(env_m), 8) / 8.0,
    ]


def leaky_features(g, by, addr, env_m, mid_set, v):
    """572 LOOK: counts mid tokens in full dest frame — includes held."""
    slots = list(by.get(addr, []))
    if not slots:
        return [0.0, 0.0]
    fr = set(comps(g, slots[0], addr))
    n_mid = sum(tok in mid_set for tok in fr)
    n_next = sum(tok in mid_set and tok not in {v, addr} for tok in fr)
    return [n_mid / max(len(fr), 1), min(n_next, 4) / 4.0]


def one_probe(g, by, v, s, mid_set, high_set, rng):
    frame = list(comps(g, s, v))
    if len(frame) < 3:
        return None
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set or (env - high_set)
    if not env_m:
        return None
    stood, place = stand_frame(g, by, v, env_m, exclude=s)
    if stood is None:
        return None
    if held in place:
        return None
    cand = [
        tok for tok in place
        if tok in mid_set and tok != v
    ]
    if len(cand) != 2:
        return None
    rng.shuffle(cand)
    a, b = cand
    _ta, fra = stand_frame(g, by, a, env_m)
    _tb, frb = stand_frame(g, by, b, env_m)
    ha = int(_ta is not None and held in fra)
    hb = int(_tb is not None and held in frb)
    return dict(
        fa=pre_features(g, by, a, env_m),
        fb=pre_features(g, by, b, env_m),
        ta=tape_features(g, by, a, env_m),
        tb=tape_features(g, by, b, env_m),
        ka=leaky_features(g, by, a, env_m, mid_set, v),
        kb=leaky_features(g, by, b, env_m, mid_set, v),
        ha=ha,
        hb=hb,
        rank_a=int(len(fra & env_m) >= len(frb & env_m)),
        maj_a=int(len(by.get(a, [])) >= len(by.get(b, []))),
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
        slots = list(by[v])
        if len(slots) < 8:
            continue
        rng.shuffle(slots)
        for s in slots[: args.cap_probe]:
            row = one_probe(g, by, v, s, mid_set, high_set, rng)
            if row is not None:
                rows.append(row)
    return rows


def xor_rows(rows):
    return [row for row in rows if row["ha"] != row["hb"]]


class PlaceScorer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(rows, seed, epochs, null=False, pair=("fa", "fb")):
    torch.manual_seed(seed)
    ka, kb = pair
    model = PlaceScorer(len(rows[0][ka]))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    fa = torch.tensor([row[ka] for row in rows], dtype=torch.float32)
    fb = torch.tensor([row[kb] for row in rows], dtype=torch.float32)
    labels = [float(row["ha"] > row["hb"]) for row in rows]
    if null:
        random.Random(seed + 991).shuffle(labels)
    y = torch.tensor(labels, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        logits = model(fa) - model(fb)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def model_pick_a(model, row, pair=("fa", "fb")):
    ka, kb = pair
    with torch.no_grad():
        fa = torch.tensor(row[ka], dtype=torch.float32).unsqueeze(0)
        fb = torch.tensor(row[kb], dtype=torch.float32).unsqueeze(0)
        return bool(model(fa).item() >= model(fb).item())


def accuracy(rows, picker):
    if not rows:
        return 0.0
    return sum(int(row["ha"] if picker(row) else row["hb"]) for row in rows) / len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=160)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=120000)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    cut = max(1, int(0.7 * len(windows)))
    rng = random.Random(args.seed)
    t0 = time.time()
    print(
        f"573 real-cut  {path}  {kind}  windows={len(windows)} "
        f"prefix {cut} / suffix {len(windows) - cut}",
        flush=True,
    )

    train_rows, test_rows = [], []
    for i, lines in enumerate(windows):
        rows = collect(lines, args, rng)
        (train_rows if i < cut else test_rows).extend(rows)
    train_xor, test_xor = xor_rows(train_rows), xor_rows(test_rows)
    n_train, n_test = len(train_xor), len(test_xor)
    if not train_xor:
        print("VOID: no train XOR")
        return 0

    pre = train_model(train_xor, args.seed + 1, args.epochs, pair=("fa", "fb"))
    nul = train_model(train_xor, args.seed + 2, args.epochs, null=True, pair=("fa", "fb"))
    tape = train_model(train_xor, args.seed + 4, args.epochs, pair=("ta", "tb"))
    leak = train_model(train_xor, args.seed + 5, args.epochs, pair=("ka", "kb"))

    learned = accuracy(test_xor, lambda r: model_pick_a(pre, r, ("fa", "fb")))
    null = accuracy(test_xor, lambda r: model_pick_a(nul, r, ("fa", "fb")))
    tape_acc = accuracy(test_xor, lambda r: model_pick_a(tape, r, ("ta", "tb")))
    leak_acc = accuracy(test_xor, lambda r: model_pick_a(leak, r, ("ka", "kb")))
    rank = accuracy(test_xor, lambda r: r["rank_a"])
    majority = accuracy(test_xor, lambda r: r["maj_a"])
    coin_fair = 0.5
    cr = random.Random(args.seed + 3)
    coin_rng = accuracy(test_xor, lambda _r: cr.random() < 0.5)

    void = n_test < 40
    margins = dict(
        fair=learned - coin_fair,
        null=learned - null,
        rank=learned - rank,
        majority=learned - majority,
    )
    gate = (not void) and all(m > 0.05 for m in margins.values())
    tape_vs_rule = tape_acc - rank
    print(f"XOR train/test {n_train}/{n_test}  two {len(train_rows)}/{len(test_rows)}")
    print(
        f"PRE {learned:.3f}  TAPE(clean) {tape_acc:.3f}  LOOK(leaky) {leak_acc:.3f}"
    )
    print(
        f"fair0.5 {coin_fair:.3f}  rng-coin {coin_rng:.3f}  null {null:.3f}  "
        f"rank {rank:.3f}  maj {majority:.3f}"
    )
    print("PRE margins " + " ".join(f"{k}={v:+.3f}" for k, v in margins.items()))
    print(f"TAPE-rank {tape_vs_rule:+.3f}  (38.3 if <=0.05)")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: test XOR < 40")
    elif gate:
        print("GO PRE: counts beat fair coin/null/rank/maj")
    else:
        print("STOP PRE: no one-shot from df/mentions. Do not drop rivals.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_windows=len(windows), n_train_windows=cut,
        n_test_windows=len(windows) - cut,
        n_xor_train=n_train, n_xor_test=n_test,
        learned=learned, tape=tape_acc, leaky=leak_acc,
        coin_fair=coin_fair, coin_rng=coin_rng,
        null=null, rank=rank, majority=majority,
        margins=margins, tape_vs_rule=tape_vs_rule,
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
