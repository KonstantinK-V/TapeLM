"""573: online one-shot place learning on rotating train tapes.

Closer to a real learner than 572:
  - split corpus first: first 70% train, untouched tail held-out;
  - windows are disjoint tapes; train/test never overlap;
  - policy chooses one of two places;
  - update sees only the chosen place's tape reward, not both outcomes;
  - test is the natural XOR slice, one shot.

The scorer uses only pre-hop count/df/env features. Runtime 567 is unchanged.

VOID  held-out XOR < 40
GATE  learner beats strongest of coin/null/rank/majority by > 0.02 and
      paired bootstrap 95% lower bound > 0.

    python _check573_streamlearn.py
    python _audit573_streamlearn.py --seed 1337 \
        --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit571_xorcensus import stand_frame
from _audit572_xorlearn import PlaceScorer, pre_features

OUT = Path("results/_stage573_streamlearn.json")


def block_windows(pool, length, limit, rng):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    rng.shuffle(blocks)
    return blocks[: min(limit, len(blocks))]


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
    cand = sorted([
        token for token in place
        if token in mid_set and token != v
    ])
    if len(cand) != 2:
        return None
    rng.shuffle(cand)
    a, b = cand
    _ta, fra = stand_frame(g, by, a, env_m)
    _tb, frb = stand_frame(g, by, b, env_m)
    return dict(
        fa=pre_features(g, by, a, env_m),
        fb=pre_features(g, by, b, env_m),
        ha=int(_ta is not None and held in fra),
        hb=int(_tb is not None and held in frb),
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
    keys = sorted(mid)
    rng.shuffle(keys)
    for v in keys:
        slots = sorted(by[v])
        if len(slots) < 8:
            continue
        rng.shuffle(slots)
        for s in slots[: args.cap_probe]:
            row = one_probe(g, by, v, s, mid_set, high_set, rng)
            if row is not None:
                rows.append(row)
    return rows


def collect_tapes(tapes, args, rng):
    return [collect(lines, args, rng) for lines in tapes]


def train_bandit(tapes, seed, epochs, null=False):
    width = len(next(row for tape in tapes for row in tape)["fa"])
    torch.manual_seed(seed)
    model = PlaceScorer(width)
    opt = torch.optim.Adam(model.parameters(), lr=0.006)
    policy_rng = random.Random(seed + 17)
    outcomes = [
        (row["ha"], row["hb"])
        for tape in tapes
        for row in tape
    ]
    if null:
        random.Random(seed + 991).shuffle(outcomes)
    baseline = 0.45
    step = 0
    total_steps = max(sum(len(tape) for tape in tapes) * epochs, 1)
    for _epoch in range(epochs):
        order = list(range(len(tapes)))
        policy_rng.shuffle(order)
        outcome_index = 0
        for tape_index in order:
            rows = list(tapes[tape_index])
            policy_rng.shuffle(rows)
            for row in rows:
                fa = torch.tensor(row["fa"], dtype=torch.float32).unsqueeze(0)
                fb = torch.tensor(row["fb"], dtype=torch.float32).unsqueeze(0)
                logit = model(fa) - model(fb)
                epsilon = max(0.08, 0.50 * (1.0 - step / total_steps))
                greedy_a = bool(logit.detach().item() >= 0.0)
                choose_a = (
                    policy_rng.random() < 0.5
                    if policy_rng.random() < epsilon
                    else greedy_a
                )
                if null:
                    ha, hb = outcomes[outcome_index % len(outcomes)]
                    outcome_index += 1
                else:
                    ha, hb = row["ha"], row["hb"]
                hit = ha if choose_a else hb
                reward = 1.0 if hit else -0.08
                advantage = reward - baseline
                log_prob = F.logsigmoid(logit if choose_a else -logit)
                loss = -advantage * log_prob.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                baseline = 0.99 * baseline + 0.01 * reward
                step += 1
    return model, step


def xor_rows(tapes):
    return [
        row for tape in tapes for row in tape
        if row["ha"] != row["hb"]
    ]


def model_pick(model, row):
    with torch.no_grad():
        fa = torch.tensor(row["fa"], dtype=torch.float32).unsqueeze(0)
        fb = torch.tensor(row["fb"], dtype=torch.float32).unsqueeze(0)
        return bool((model(fa) - model(fb)).item() >= 0.0)


def wins(rows, picker):
    return [
        int(row["ha"] if picker(row) else row["hb"])
        for row in rows
    ]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def bootstrap_delta(left, right, seed, samples=3000):
    if not left:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(left)
    deltas = []
    paired = [a - b for a, b in zip(left, right)]
    for _ in range(samples):
        deltas.append(sum(paired[rng.randrange(n)] for _j in range(n)) / n)
    deltas.sort()
    return deltas[int(0.025 * samples)], deltas[int(0.975 * samples)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--train-windows", type=int, default=180)
    ap.add_argument("--test-windows", type=int, default=100)
    ap.add_argument("--train-epochs", type=int, default=5)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=160000)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [
        line.strip() for line in text.split("\n")
        if len(line.strip()) >= min_line
    ][: args.lines]
    cut = int(0.7 * len(all_lines))
    train_pool, test_pool = all_lines[:cut], all_lines[cut:]
    rng = random.Random(args.seed)
    train_windows = block_windows(
        train_pool, args.window_lines, args.train_windows, rng
    )
    test_windows = block_windows(
        test_pool, args.window_lines, args.test_windows, rng
    )
    t0 = time.time()
    print(
        f"573 stream learner  corpus={path}  {kind}  "
        f"windows={len(train_windows)}/{len(test_windows)}",
        flush=True,
    )
    train_tapes = collect_tapes(train_windows, args, rng)
    test_tapes = collect_tapes(test_windows, args, rng)
    model, steps = train_bandit(
        train_tapes, args.seed + 1, args.train_epochs, null=False
    )
    null_model, _ = train_bandit(
        train_tapes, args.seed + 2, args.train_epochs, null=True
    )

    test_xor = xor_rows(test_tapes)
    learned_w = wins(test_xor, lambda row: model_pick(model, row))
    null_w = wins(test_xor, lambda row: model_pick(null_model, row))
    rank_w = wins(test_xor, lambda row: bool(row["rank_a"]))
    majority_w = wins(test_xor, lambda row: bool(row["maj_a"]))
    # Exact expected coin, not one noisy realization on a small test slice.
    coin_w = [0.5] * len(test_xor)
    scores = dict(
        learned=mean(learned_w),
        coin=mean(coin_w),
        null=mean(null_w),
        rank=mean(rank_w),
        majority=mean(majority_w),
    )
    rival_name = max(
        ("coin", "null", "rank", "majority"),
        key=lambda name: scores[name],
    )
    rival_w = dict(
        coin=coin_w,
        null=null_w,
        rank=rank_w,
        majority=majority_w,
    )[rival_name]
    delta = scores["learned"] - scores[rival_name]
    ci_low, ci_high = bootstrap_delta(
        learned_w, rival_w, args.seed + 4
    )
    n_test = len(test_xor)
    void = n_test < 40
    gate = (not void) and delta > 0.02 and ci_low > 0.0

    n_train_rows = sum(len(tape) for tape in train_tapes)
    n_test_rows = sum(len(tape) for tape in test_tapes)
    n_train_xor = len(xor_rows(train_tapes))
    print(
        f"two train/test {n_train_rows}/{n_test_rows}  "
        f"XOR {n_train_xor}/{n_test}  updates={steps}"
    )
    print(
        "  ".join(f"{name} {value:.3f}" for name, value in scores.items())
    )
    print(
        f"strongest={rival_name}  delta={delta:+.3f}  "
        f"paired95=[{ci_low:+.3f},{ci_high:+.3f}]"
    )
    print(f"VOID {void}  GATE {gate}  (held-out tail; chosen reward only)")
    if void:
        print("\nVOID: held-out tail has fewer than 40 XOR pairs.")
    elif gate:
        print("\nGO STREAM: online place policy beats the strongest rival.")
    else:
        print("\nSTOP: online policy does not beat the strongest rival pairwise.")

    rec = dict(
        seed=args.seed,
        corpus=kind,
        path=str(path),
        n_train_windows=len(train_windows),
        n_test_windows=len(test_windows),
        n_train_rows=n_train_rows,
        n_test_rows=n_test_rows,
        n_train_xor=n_train_xor,
        n_test_xor=n_test,
        updates=steps,
        scores=scores,
        strongest=rival_name,
        delta=delta,
        ci95=[ci_low, ci_high],
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
