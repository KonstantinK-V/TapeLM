"""630: frozen 618 plus learned SEARCH/COMMIT/REFUSE on its REFUSE branch.

Peak is not learned.  SEARCH_ONE opens the next held-blind door in frozen tape
order.  COMMIT_RESOLVED is a frozen 618-style peak over accumulated hop1
observations.  The model emits only one of three operations.

    python _check630_integrated.py
    python _audit630_integrated.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit630_integrated.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit630_integrated.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from _audit511_ring import pick_corpus
from _audit606_bridge import bands, build_places
from _audit615_depth import extracts
from _audit618_peakpin import peak_pin
from _audit624_pick import hide_two
from _integrated_contract_v1 import (
    ACTIONS,
    CAP,
    C_COMMIT,
    C_SEARCH,
    LAW,
    commit_resolved,
    full_feedback_q,
    leftover_records,
    open_record,
    state_features,
    valid_actions,
)

OUT = Path("results/_stage630_integrated.json")


def block_windows(pool, length, limit, rng):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    rng.shuffle(blocks)
    return blocks[: min(limit, len(blocks))]


def collect_episodes(windows, args, seed):
    """Independent local tapes; train rows only after frozen 618 REFUSE."""
    episodes = []
    diag = Counter()
    for wi, lines in enumerate(windows):
        rng = random.Random(seed + 1009 * wi)
        pg = build_places(lines, args.frame_max, args.min_fillers)
        if pg is None:
            continue
        mid_set, high_set = bands(pg)
        if not mid_set:
            continue
        co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
        places = pg["places"]
        pins = sorted(mid_set)
        rng.shuffle(pins)
        for pin in pins:
            qids = list(pg["by_place"].get(pin, ()))
            if len(qids) < 3:
                continue
            rng.shuffle(qids)
            for qi in qids[: args.cap_probe]:
                query = places[qi]
                if pin not in set(query["keys"]):
                    continue
                env = set(query["keys"])
                env_m = (env & mid_set) - high_set or (env - high_set)
                if not env_m:
                    continue
                extras = [
                    tok for tok in dict.fromkeys(query["vals"])
                    if tok in mid_set and tok != pin and tok not in env
                    and tok not in high_set
                ]
                if len(extras) < 2:
                    continue
                rng.shuffle(extras)
                held_ctx, held_ask = extras[0], extras[1]
                hide_two(co, df, query["keys"], held_ctx, held_ask, -1)
                n_use = max(n_fr - 2, 1)
                try:
                    diag["n"] += 1
                    rows_p = extracts(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    if any(tok == held_ask for tok, _bag, _uniq in rows_p[:3]):
                        diag["direct_hit"] += 1
                        continue

                    all_records = leftover_records(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )
                    hat = peak_pin(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )
                    if hat is not None:
                        diag["peak_trials"] += 1
                        support = next(
                            (
                                rec["door_support_pi"]
                                for rec in all_records
                                if rec["door"] == hat
                            ),
                            qi,
                        )
                        opened_hat = open_record(
                            pg,
                            dict(door=hat, door_support_pi=support),
                            pin, qi, env_m, mid_set, high_set,
                            co, df, n_use,
                        )
                        hit = (
                            hat == held_ask
                            or any(
                                obs["tok"] == held_ask
                                for obs in opened_hat["observations"]
                            )
                        )
                        diag["peak_hit"] += int(hit)
                        continue

                    diag["refuse_branch"] += 1
                    records = all_records[:CAP]
                    if not records:
                        diag["empty"] += 1
                        continue
                    diag["door_direct"] += int(
                        any(rec["door"] == held_ask for rec in records)
                    )
                    opened = []
                    features = []
                    commits = []
                    x0, c0 = state_features(
                        pg, qi, pin, opened, len(records),
                        high_set, co, df, n_use,
                    )
                    features.append(x0)
                    commits.append(c0)
                    for record in records:
                        opened.append(open_record(
                            pg, record, pin, qi, env_m, mid_set, high_set,
                            co, df, n_use,
                        ))
                        x, commit = state_features(
                            pg, qi, pin, opened, len(records),
                            high_set, co, df, n_use,
                        )
                        features.append(x)
                        commits.append(commit)
                    targets = full_feedback_q(commits, held_ask)
                    episodes.append(dict(
                        features=features,
                        commits=commits,
                        targets=targets,
                        held=held_ask,
                        oracle_net=max(targets[0]),
                    ))
                finally:
                    hide_two(
                        co, df, query["keys"], held_ctx, held_ask, +1,
                    )
    diag["episodes"] = len(episodes)
    return episodes, dict(diag)


class RefusePolicy(torch.nn.Module):
    """Three operation scores; no tape-sized output layer."""

    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, len(ACTIONS)),
        )

    def forward(self, x):
        return self.net(x)


def flatten_samples(episodes):
    return [
        (features, target)
        for ep in episodes
        for features, target in zip(ep["features"], ep["targets"])
    ]


def train_policy(episodes, seed, epochs, null=False):
    samples = flatten_samples(episodes)
    if not samples:
        return None, None, None
    x = torch.tensor([row[0] for row in samples], dtype=torch.float32)
    y_rows = [list(row[1]) for row in samples]
    if null:
        rng = random.Random(seed + 991)
        rng.shuffle(y_rows)
    y = torch.tensor(y_rows, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-4)
    x = (x - mean) / std

    torch.manual_seed(seed)
    model = RefusePolicy(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 17)
    for _epoch in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        for start in range(0, len(x), 64):
            idx = order[start:start + 64]
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


def choose_action(model, mean, std, features, valid):
    x = (torch.tensor(features, dtype=torch.float32) - mean) / std
    with torch.no_grad():
        scores = model(x)
    masked = scores.clone()
    for ai, allowed in enumerate(valid):
        if not allowed:
            masked[ai] = -1e9
    return int(torch.argmax(masked).item())


def finish(ep, state_index, searches, action):
    net = -C_SEARCH * searches
    commit = ep["commits"][state_index]
    if action == "COMMIT_RESOLVED":
        hit = int(commit is not None and commit["tok"] == ep["held"])
        net += 1.0 if hit else -C_COMMIT
        return dict(
            hit=hit, net=net, searches=searches,
            commit=commit, action=action,
        )
    return dict(
        hit=0, net=net, searches=searches,
        commit=None, action=action,
    )


def rollout_model(ep, model, norm):
    mean, std = norm
    last = len(ep["features"]) - 1
    state_index = searches = 0
    while True:
        valid = valid_actions(
            state_index, last, ep["commits"][state_index],
        )
        ai = choose_action(
            model, mean, std, ep["features"][state_index], valid,
        )
        action = ACTIONS[ai]
        if action == "SEARCH_ONE":
            searches += 1
            state_index += 1
            continue
        return finish(ep, state_index, searches, action)


def rollout_fixed(ep, rule, rng=None):
    last = len(ep["features"]) - 1
    state_index = searches = 0
    if rule == "always_refuse":
        return finish(ep, 0, 0, "REFUSE")
    if rule == "one_then_commit":
        if last > 0:
            state_index = searches = 1
        action = (
            "COMMIT_RESOLVED"
            if ep["commits"][state_index] is not None else "REFUSE"
        )
        return finish(ep, state_index, searches, action)
    if rule == "search_all":
        state_index = searches = last
        action = (
            "COMMIT_RESOLVED"
            if ep["commits"][state_index] is not None else "REFUSE"
        )
        return finish(ep, state_index, searches, action)
    if rule == "first_peak":
        while (
            state_index < last
            and ep["commits"][state_index] is None
        ):
            state_index += 1
            searches += 1
        action = (
            "COMMIT_RESOLVED"
            if ep["commits"][state_index] is not None else "REFUSE"
        )
        return finish(ep, state_index, searches, action)
    if rule == "random":
        assert rng is not None
        while True:
            valid = valid_actions(
                state_index, last, ep["commits"][state_index],
            )
            choices = [i for i, allowed in enumerate(valid) if allowed]
            action = ACTIONS[rng.choice(choices)]
            if action == "SEARCH_ONE":
                searches += 1
                state_index += 1
                continue
            return finish(ep, state_index, searches, action)
    raise ValueError(rule)


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


def evaluate(episodes, model, norm, null_model, null_norm, seed):
    vec = defaultdict(list)
    actions = Counter()
    rng = random.Random(seed)
    rules = (
        "always_refuse", "one_then_commit", "first_peak",
        "search_all", "random",
    )
    for ep in episodes:
        learned = rollout_model(ep, model, norm)
        null = rollout_model(ep, null_model, null_norm)
        actions[learned["action"]] += 1
        vec["learned_hit"].append(learned["hit"])
        vec["learned_net"].append(learned["net"])
        vec["learned_search"].append(learned["searches"])
        vec["null_hit"].append(null["hit"])
        vec["null_net"].append(null["net"])
        for rule in rules:
            got = rollout_fixed(ep, rule, rng)
            vec[f"{rule}_hit"].append(got["hit"])
            vec[f"{rule}_net"].append(got["net"])

        commit = learned["commit"]
        freq_hit = int(
            commit is not None and commit["majority"] == ep["held"]
        )
        freq_net = -C_SEARCH * learned["searches"]
        if commit is not None:
            freq_net += 1.0 if freq_hit else -C_COMMIT
        vec["freq_hit"].append(freq_hit)
        vec["freq_net"].append(freq_net)
        vec["oracle_net"].append(ep["oracle_net"])

    fixed_names = (
        "always_refuse", "one_then_commit", "first_peak",
        "search_all", "random", "null", "freq",
    )
    strongest = max(
        fixed_names, key=lambda name: mean(vec[f"{name}_net"]),
    )
    zero = [0.0] * len(episodes)
    low_refuse, high_refuse = bootstrap_delta(
        vec["learned_net"], zero, seed + 1,
    )
    low_fixed, high_fixed = bootstrap_delta(
        vec["learned_net"], vec[f"{strongest}_net"], seed + 2,
    )
    rates = {
        name: mean(values)
        for name, values in vec.items()
    }
    rates.update(
        strongest_name=strongest,
        strongest_net=rates[f"{strongest}_net"],
        delta_fixed=rates["learned_net"] - rates[f"{strongest}_net"],
        low95_refuse=low_refuse,
        high95_refuse=high_refuse,
        low95_fixed=low_fixed,
        high95_fixed=high_fixed,
        actions=dict(actions),
    )
    return rates


def write_hungry(args, path, kind, t0, train, test, train_w, test_w):
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        elapsed_s=round(time.time() - t0, 1),
        n_train=len(train), n_test=len(test),
        train_windows=len(train_w), test_windows=len(test_w),
        void=True, enabled=False, kill_switch=True,
        mind_gate=False, reason="hungry_refuse_branch",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--train-windows", type=int, default=160)
    ap.add_argument("--test-windows", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=120)
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
        f"630 integrated  {path}  {kind}  "
        f"windows {len(train_windows)}/{len(test_windows)}",
        flush=True,
    )
    train, train_diag = collect_episodes(
        train_windows, args, args.seed + 101,
    )
    test, test_diag = collect_episodes(
        test_windows, args, args.seed + 202,
    )
    print(
        f"REFUSE episodes train {len(train)}  test {len(test)}  "
        f"total test {test_diag.get('n', 0)}",
        flush=True,
    )
    if len(train) < 40 or len(test) < 40:
        print("VOID: REFUSE branch hungry; kill switch ON.")
        write_hungry(
            args, path, kind, t0, train, test, train_windows, test_windows,
        )
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
    oracle_room = rates["oracle_net"]
    void = oracle_room <= 0.05
    enabled = (
        (not void)
        and rates["learned_net"] > 0.0
        and rates["low95_refuse"] > 0.0
    )
    kill_switch = not enabled
    mind_gate = (
        enabled
        and rates["delta_fixed"] > 0.05
        and rates["low95_fixed"] > 0.0
        and rates["learned_net"] > rates["freq_net"]
    )

    n_total = max(test_diag.get("n", 0), 1)
    base_hits = (
        test_diag.get("direct_hit", 0) + test_diag.get("peak_hit", 0)
    )
    baseline_cover = base_hits / n_total
    branch_gain = (
        sum(
            rollout_model(ep, model, (m_mean, m_std))["net"]
            for ep in test
        ) / n_total
        if enabled else 0.0
    )
    integrated_net = baseline_cover + branch_gain

    print(
        f"BASE 618 hit {base_hits}/{n_total} {baseline_cover:.3f}  "
        f"REFUSE share {test_diag.get('refuse_branch', 0) / n_total:.3f}"
    )
    print(
        f"POLICY hit {rates['learned_hit']:.3f}  "
        f"search {rates['learned_search']:.2f}  "
        f"priced {rates['learned_net']:+.3f}  "
        f"low95-vs-refuse {rates['low95_refuse']:+.3f}"
    )
    print(
        f"RIVAL {rates['strongest_name']} "
        f"{rates['strongest_net']:+.3f}  "
        f"delta {rates['delta_fixed']:+.3f}  "
        f"low95 {rates['low95_fixed']:+.3f}  "
        f"freq {rates['freq_net']:+.3f}"
    )
    print(
        f"oracle_net {oracle_room:+.3f}  actions {rates['actions']}  "
        f"VOID {void}  ENABLED {enabled}  MIND_GATE {mind_gate}"
    )
    if kill_switch:
        print("KILL SWITCH ON: deployed system = frozen 618 -> REFUSE.")
    elif mind_gate:
        print(
            "GO 630: priced REFUSE policy beats refuse and fixed rivals; "
            "integrated branch enabled."
        )
    else:
        print(
            "USEFUL ONLY: beats REFUSE after price, but not strongest fixed "
            "rival. Enable is safe; do not call it a learned mind."
        )

    checkpoint = (
        Path(args.checkpoint) if args.checkpoint else
        Path("checkpoints") /
        f"_stage630_integrated_{path.stem}_{args.seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        state_dict=model.state_dict(),
        feature_mean=m_mean,
        feature_std=m_std,
        actions=ACTIONS,
        contract=LAW,
        enabled=bool(enabled),
        kill_switch=bool(kill_switch),
        mind_gate=bool(mind_gate),
        seed=args.seed,
        corpus=str(path),
    ), checkpoint)

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        elapsed_s=round(time.time() - t0, 1),
        n_train=len(train), n_test=len(test),
        train_windows=len(train_windows), test_windows=len(test_windows),
        baseline_cover=baseline_cover,
        branch_gain=branch_gain,
        integrated_net=integrated_net,
        oracle_room=oracle_room,
        void=bool(void), enabled=bool(enabled),
        kill_switch=bool(kill_switch), mind_gate=bool(mind_gate),
        checkpoint=str(checkpoint),
        train_diag=train_diag, test_diag=test_diag,
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
