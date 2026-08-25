"""629: train the frozen context-v1 constraint policy.

The network never ranks candidate addresses or words.  It emits one fixed
constraint from _CONTEXT_CONTRACT_V1.txt; the tape applies that constraint to
all legal cards of CURRENT and resolves one exact address or REFUSE.

Train/test corpora are split before independent tapes are built.  Every action
gets full tape feedback.  Test is the 628 same-QUERY/CURRENT context swap.

    python _check629_ctxtrain.py
    python _audit629_ctxtrain.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit629_ctxtrain.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit629_ctxtrain.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from _audit511_ring import pick_corpus
from _audit606_bridge import bands
from _audit628_ctxswap import collect_window, peak_with_support
from _audit624_pick import hide_two
from _context_contract_v1 import (
    ACTIONS,
    C_READ,
    LAW,
    full_feedback,
    state_features,
)

OUT = Path("results/_stage629_ctxtrain.json")


def block_windows(pool, length, limit, rng):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    rng.shuffle(blocks)
    return blocks[: min(limit, len(blocks))]


def priced(hit, read):
    return (1.0 if hit else -C_READ) if read else 0.0


def make_examples(windows, args, seed):
    """Build held-blind features and full-feedback labels on each local tape."""
    examples = []
    diag = Counter()
    for wi, lines in enumerate(windows):
        pg, paired, one_diag = collect_window(
            lines, args, seed + 1009 * wi,
        )
        diag.update(one_diag)
        if pg is None:
            continue
        co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
        _mid_set, high_set = bands(pg)
        rng = random.Random(seed + 2003 * wi)
        for ep, swap_support in paired:
            hide_two(
                co, df, ep["hide_keys"], ep["held_ctx"], ep["held_ask"], -1,
            )
            n_use = max(n_fr - 2, 1)
            try:
                x_true, true_resolved = state_features(
                    pg, ep["rows"], ep["qi"], ep["current"], ep["support"],
                    high_set, co, df, n_use,
                )
                x_swap, swap_resolved = state_features(
                    pg, ep["rows"], ep["qi"], ep["current"], swap_support,
                    high_set, co, df, n_use,
                )
            finally:
                hide_two(
                    co, df, ep["hide_keys"], ep["held_ctx"], ep["held_ask"], +1,
                )

            held = ep["held_ask"]
            hits, reads, rewards = full_feedback(true_resolved, held)
            shits, sreads, srewards = full_feedback(swap_resolved, held)
            pis = [
                None if true_resolved[action] is None
                else true_resolved[action]["pi"]
                for action in ACTIONS
            ]
            spis = [
                None if swap_resolved[action] is None
                else swap_resolved[action]["pi"]
                for action in ACTIONS
            ]
            freq_hits = [
                int(
                    true_resolved[action] is not None
                    and true_resolved[action]["majority"] == held
                )
                for action in ACTIONS
            ]
            freq_reads = [
                int(true_resolved[action] is not None)
                for action in ACTIONS
            ]
            count_row = max(ep["rows"], key=lambda row: row["count_key"])
            peak_tok, _support = peak_with_support(ep["rows"])
            rand_row = rng.choice(ep["rows"])
            examples.append(dict(
                x=x_true,
                x_swap=x_swap,
                hits=hits,
                reads=reads,
                rewards=rewards,
                swap_hits=shits,
                swap_reads=sreads,
                swap_rewards=srewards,
                pis=pis,
                swap_pis=spis,
                freq_hits=freq_hits,
                freq_reads=freq_reads,
                count_hit=int(count_row["tok"] == held),
                peak_hit=int(peak_tok == held),
                peak_read=int(peak_tok is not None),
                rand_hit=int(rand_row["tok"] == held),
                address_ora=int(any(row["tok"] == held for row in ep["rows"])),
            ))
    diag["examples"] = len(examples)
    return examples, dict(diag)


class ConstraintPolicy(torch.nn.Module):
    """Small policy over constraint names, never over tape literals."""

    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, len(ACTIONS)),
        )

    def forward(self, x):
        return self.net(x)


def normalize_train(examples):
    x = torch.tensor([ep["x"] for ep in examples], dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-4)
    return mean, std


def train_policy(examples, seed, epochs, null=False):
    if not examples:
        return None, None, None
    mean, std = normalize_train(examples)
    x = torch.tensor([ep["x"] for ep in examples], dtype=torch.float32)
    x = (x - mean) / std
    y_rows = [list(ep["rewards"]) for ep in examples]
    if null:
        rng = random.Random(seed + 991)
        rng.shuffle(y_rows)
    y = torch.tensor(y_rows, dtype=torch.float32)

    torch.manual_seed(seed)
    model = ConstraintPolicy(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 17)
    for _epoch in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        for start in range(0, len(x), 32):
            idx = order[start:start + 32]
            q = model(x[idx])
            target = y[idx]
            target_prob = torch.softmax(target * 4.0, dim=1)
            listwise = -(
                target_prob * torch.log_softmax(q, dim=1)
            ).sum(dim=1).mean()
            value = torch.nn.functional.smooth_l1_loss(q, target)
            loss = listwise + 0.25 * value
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    return model, mean, std


def choose(model, mean, std, x):
    z = (torch.tensor(x, dtype=torch.float32) - mean) / std
    with torch.no_grad():
        return int(torch.argmax(model(z)).item())


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_delta(left, right, seed, samples=3000):
    if not left:
        return 0.0, 0.0
    paired = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(paired)
    draws = [
        sum(paired[rng.randrange(n)] for _j in range(n)) / n
        for _i in range(samples)
    ]
    draws.sort()
    return draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def evaluate(examples, model, norm, null_model, null_norm, seed):
    m_mean, m_std = norm
    n_mean, n_std = null_norm
    vec = defaultdict(list)
    actions = Counter()
    swap_actions = Counter()
    for ep in examples:
        action = choose(model, m_mean, m_std, ep["x"])
        swap_action = choose(model, m_mean, m_std, ep["x_swap"])
        null_action = choose(null_model, n_mean, n_std, ep["x"])
        actions[ACTIONS[action]] += 1
        swap_actions[ACTIONS[swap_action]] += 1

        vec["learned"].append(ep["hits"][action])
        vec["learned_read"].append(ep["reads"][action])
        vec["learned_net"].append(ep["rewards"][action])
        vec["swap"].append(ep["swap_hits"][swap_action])
        vec["swap_read"].append(ep["swap_reads"][swap_action])
        vec["swap_net"].append(ep["swap_rewards"][swap_action])
        vec["null"].append(ep["hits"][null_action])
        vec["null_read"].append(ep["reads"][null_action])
        vec["null_net"].append(ep["rewards"][null_action])
        vec["freq"].append(ep["freq_hits"][action])
        vec["freq_read"].append(ep["freq_reads"][action])
        vec["freq_net"].append(priced(
            ep["freq_hits"][action], ep["freq_reads"][action],
        ))
        vec["count"].append(ep["count_hit"])
        vec["count_net"].append(priced(ep["count_hit"], 1))
        vec["peak"].append(ep["peak_hit"])
        vec["peak_net"].append(priced(ep["peak_hit"], ep["peak_read"]))
        vec["rand"].append(ep["rand_hit"])
        vec["rand_net"].append(priced(ep["rand_hit"], 1))
        vec["changed"].append(
            int(ep["pis"][action] != ep["swap_pis"][swap_action])
        )
        vec["action_ora"].append(max(ep["hits"]))
        vec["address_ora"].append(ep["address_ora"])
        for ai, name in enumerate(ACTIONS):
            vec[f"fixed_{name}"].append(ep["hits"][ai])
            vec[f"fixed_{name}_net"].append(ep["rewards"][ai])

    rival_names = ["swap", "null", "count", "peak", "freq", "rand"] + [
        f"fixed_{name}" for name in ACTIONS
    ]
    rival = max(rival_names, key=lambda name: mean(vec[name]))
    rival_net = max(
        rival_names,
        key=lambda name: mean(vec[name + "_net"]),
    )
    low95, high95 = bootstrap_delta(
        vec["learned"], vec[rival], seed,
    )
    rates = {
        name: mean(values)
        for name, values in vec.items()
    }
    rates.update(
        strongest_name=rival,
        strongest=rates[rival],
        strongest_net_name=rival_net,
        strongest_net=rates[rival_net + "_net"],
        low95=low95,
        high95=high95,
        actions=dict(actions),
        swap_actions=dict(swap_actions),
    )
    return rates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--train-windows", type=int, default=160)
    ap.add_argument("--test-windows", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=160000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--checkpoint", default="")
    args = ap.parse_args()

    torch.set_num_threads(1)
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
        train_pool, args.window_lines, args.train_windows, rng,
    )
    test_windows = block_windows(
        test_pool, args.window_lines, args.test_windows, rng,
    )
    t0 = time.time()
    print(
        f"629 ctxtrain  {path}  {kind}  "
        f"windows {len(train_windows)}/{len(test_windows)}",
        flush=True,
    )

    train, train_diag = make_examples(
        train_windows, args, args.seed + 101,
    )
    test, test_diag = make_examples(
        test_windows, args, args.seed + 202,
    )
    print(
        f"examples train {len(train)}  test {len(test)}  "
        f"source {train_diag.get('source', 0)}/{test_diag.get('source', 0)}",
        flush=True,
    )
    if len(train) < 40 or len(test) < 40:
        print("VOID: context train/test pairs hungry.")
        rec = dict(
            seed=args.seed,
            corpus=kind,
            path=str(path),
            elapsed_s=round(time.time() - t0, 1),
            n_train=len(train),
            n_test=len(test),
            train_windows=len(train_windows),
            test_windows=len(test_windows),
            void=True,
            gate=False,
            reason="hungry_pairs",
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        prev[f"{args.seed}_{path.stem}"] = rec
        out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print(f"wrote {out}")
        return 0

    model, m_mean, m_std = train_policy(
        train, args.seed + 1, args.epochs, null=False,
    )
    null_model, n_mean, n_std = train_policy(
        train, args.seed + 2, args.epochs, null=True,
    )
    rates = evaluate(
        test, model, (m_mean, m_std),
        null_model, (n_mean, n_std), args.seed + 303,
    )
    delta = rates["learned"] - rates["strongest"]
    delta_swap = rates["learned"] - rates["swap"]
    action_room = rates["action_ora"] - rates["strongest"]
    void = len(test) < 40 or action_room <= 0.05
    gate = (
        (not void)
        and delta > 0.05
        and delta_swap > 0.05
        and rates["changed"] >= 0.10
        and rates["low95"] > 0.0
        and rates["learned_net"] > rates["strongest_net"]
    )

    print(
        f"POLICY true {rates['learned']:.3f}  swap {rates['swap']:.3f}  "
        f"null {rates['null']:.3f}  read {rates['learned_read']:.3f}  "
        f"net {rates['learned_net']:+.3f}"
    )
    print(
        f"RIVAL {rates['strongest_name']} {rates['strongest']:.3f}  "
        f"net {rates['strongest_net_name']} {rates['strongest_net']:+.3f}  "
        f"delta {delta:+.3f}  low95 {rates['low95']:+.3f}"
    )
    print(
        f"CONTEXT delta_swap {delta_swap:+.3f}  changed {rates['changed']:.3f}  "
        f"action_ora {rates['action_ora']:.3f}  "
        f"address_ora {rates['address_ora']:.3f}"
    )
    print(f"ACTIONS {rates['actions']}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID CTXTRAIN: fixed constraints have no held-out room.")
    elif gate:
        print(
            "GO CTXTRAIN: a name-free policy learned which search constraint "
            "to emit on fresh tapes."
        )
    else:
        print(
            "STOP CTXTRAIN: contract stays available, but this corpus did not "
            "teach a transferable context policy."
        )

    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else
        Path("checkpoints") /
        f"_stage629_ctxpolicy_{path.stem}_{args.seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state_dict=model.state_dict(),
        feature_mean=m_mean,
        feature_std=m_std,
        actions=ACTIONS,
        contract=LAW,
        seed=args.seed,
        corpus=str(path),
    ), checkpoint)

    rec = dict(
        seed=args.seed,
        corpus=kind,
        path=str(path),
        elapsed_s=round(time.time() - t0, 1),
        n_train=len(train),
        n_test=len(test),
        train_windows=len(train_windows),
        test_windows=len(test_windows),
        checkpoint=str(checkpoint),
        delta=delta,
        delta_swap=delta_swap,
        action_room=action_room,
        void=bool(void),
        gate=bool(gate),
        **rates,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out} and {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
