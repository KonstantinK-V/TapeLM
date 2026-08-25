"""572: one-shot place learner on natural hop2-XOR. Runtime 567 stays frozen.

Exactly two candidates. The exam permits one shot only. The gated shared
scorer sees only pre-hop count/df/env summaries; it never sees token identity
or held. A destination-frame lookahead scorer is printed as a diagnostic and
is not gated. Tape outcome supplies the pairwise teacher.

Train/test are disjoint contiguous windows.

VOID  test XOR < 40
GATE  learner beats coin, shuffled-label null, overlap rank, and
      frequency-majority by > 0.05 on test XOR.

    python _check572_xorlearn.py
    python _audit572_xorlearn.py --seed 1337 \
        --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage572_xorlearn.json")


def disjoint_windows(pool, length, n_win, rng):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    rng.shuffle(blocks)
    return blocks[: min(n_win, len(blocks))]


def best_frames(g, by, addr, env_m, cap=8):
    slots = list(by.get(addr, []))
    scored = []
    for t in slots[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        jac = ov / max(len(fr | env_m), 1)
        scored.append((jac, ov, fr))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return scored


def pre_features(g, by, addr, env_m):
    """Counts available before taking the candidate hop."""
    mentions_n = len(by.get(addr, []))
    df = g["df"].get(addr, mentions_n)
    return [
        min(mentions_n, 32) / 32.0,
        min(df, 64) / 64.0,
        min(len(env_m), 8) / 8.0,
        min(df / max(mentions_n, 1), 8.0) / 8.0,
        math.log1p(mentions_n) / math.log(33.0),
    ]


def lookahead_features(g, by, addr, v, env_m, mid_set, high_set):
    """Diagnostic only: reads destination-frame summaries before the choice."""
    frames = best_frames(g, by, addr, env_m)
    if not frames:
        return [0.0] * 11
    jac, ov, fr = frames[0]
    jac2 = frames[1][0] if len(frames) > 1 else 0.0
    width = max(len(fr), 1)
    n_mid = sum(token in mid_set for token in fr)
    n_high = sum(token in high_set for token in fr)
    n_next = sum(
        token in mid_set and token not in {v, addr}
        for token in fr
    )
    mentions_n = len(by.get(addr, []))
    df = g["df"].get(addr, mentions_n)
    return [
        min(mentions_n, 32) / 32.0,
        min(df, 64) / 64.0,
        min(len(env_m), 8) / 8.0,
        min(width, 8) / 8.0,
        min(ov, 4) / 4.0,
        jac,
        jac - jac2,
        n_mid / width,
        n_high / width,
        min(n_next, 4) / 4.0,
        math.log1p(mentions_n) / math.log(33.0),
    ]


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
        token for token in place
        if token in mid_set and token != v
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
        la=lookahead_features(g, by, a, v, env_m, mid_set, high_set),
        lb=lookahead_features(g, by, b, v, env_m, mid_set, high_set),
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


def train_model(rows, seed, epochs, null=False, feature_pair=("fa", "fb")):
    torch.manual_seed(seed)
    key_a, key_b = feature_pair
    model = PlaceScorer(len(rows[0][key_a]))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    fa = torch.tensor([row[key_a] for row in rows], dtype=torch.float32)
    fb = torch.tensor([row[key_b] for row in rows], dtype=torch.float32)
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


def model_pick_a(model, row, feature_pair=("fa", "fb")):
    key_a, key_b = feature_pair
    with torch.no_grad():
        fa = torch.tensor(row[key_a], dtype=torch.float32).unsqueeze(0)
        fb = torch.tensor(row[key_b], dtype=torch.float32).unsqueeze(0)
        return bool(model(fa).item() >= model(fb).item())


def accuracy(rows, picker):
    if not rows:
        return 0.0
    wins = 0
    for row in rows:
        pick_a = bool(picker(row))
        wins += int(row["ha"] if pick_a else row["hb"])
    return wins / len(rows)


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
    all_lines = [
        line.strip() for line in text.split("\n")
        if len(line.strip()) >= min_line
    ]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    windows = disjoint_windows(pool, args.window_lines, args.n_win, rng)
    cut = max(1, int(0.7 * len(windows)))
    t0 = time.time()
    print(
        f"572 one-shot XOR learner  corpus={path}  {kind}  "
        f"windows={len(windows)} ({cut}/{len(windows) - cut})",
        flush=True,
    )

    train_rows, test_rows = [], []
    for index, lines in enumerate(windows):
        rows = collect(lines, args, rng)
        (train_rows if index < cut else test_rows).extend(rows)
    train_xor = xor_rows(train_rows)
    test_xor = xor_rows(test_rows)
    n_train, n_test = len(train_xor), len(test_xor)

    if not train_xor:
        print("VOID: no train XOR")
        return 0
    model = train_model(train_xor, args.seed + 1, args.epochs, null=False)
    null_model = train_model(train_xor, args.seed + 2, args.epochs, null=True)
    look_model = train_model(
        train_xor,
        args.seed + 4,
        args.epochs,
        null=False,
        feature_pair=("la", "lb"),
    )

    learned = accuracy(test_xor, lambda row: model_pick_a(model, row))
    null = accuracy(test_xor, lambda row: model_pick_a(null_model, row))
    lookahead = accuracy(
        test_xor,
        lambda row: model_pick_a(look_model, row, feature_pair=("la", "lb")),
    )
    rank = accuracy(test_xor, lambda row: row["rank_a"])
    majority = accuracy(test_xor, lambda row: row["maj_a"])
    coin_rng = random.Random(args.seed + 3)
    coin = accuracy(test_xor, lambda _row: coin_rng.random() < 0.5)

    void = n_test < 40
    margins = dict(
        coin=learned - coin,
        null=learned - null,
        rank=learned - rank,
        majority=learned - majority,
    )
    gate = (not void) and all(margin > 0.05 for margin in margins.values())
    print(
        f"two train/test {len(train_rows)}/{len(test_rows)}  "
        f"XOR {n_train}/{n_test}"
    )
    print(
        f"PRE-learn {learned:.3f}  LOOK diagnostic {lookahead:.3f}  "
        f"coin {coin:.3f}  null {null:.3f}  rank {rank:.3f}  "
        f"majority {majority:.3f}"
    )
    print(
        "margins " +
        " ".join(f"{name}={value:+.3f}" for name, value in margins.items())
    )
    print(f"VOID {void}  GATE {gate}  (one-shot exam; runtime 567 unchanged)")
    if void:
        print("\nVOID: fewer than 40 natural test-XOR pairs.")
    elif gate:
        print("\nGO LEARN: pre-step place scorer beats all structural rivals.")
    else:
        print("\nSTOP: scorer did not beat coin/null/rank/majority.")

    rec = dict(
        seed=args.seed,
        corpus=kind,
        path=str(path),
        n_windows=len(windows),
        n_train_windows=cut,
        n_test_windows=len(windows) - cut,
        n_two_train=len(train_rows),
        n_two_test=len(test_rows),
        n_xor_train=n_train,
        n_xor_test=n_test,
        learned=learned,
        lookahead_diagnostic=lookahead,
        coin=coin,
        null=null,
        rank=rank,
        majority=majority,
        margins=margins,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void),
        gate=bool(gate),
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
