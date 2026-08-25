"""624: learned exact-place policy on the honest 618-REFUSE remainder.

This is not a vocabulary classifier.  Each action is an exact counted frame
address which the tape resolves to one unique door filler.  The chosen door is
READ first; if it is not the answer, the frozen unique+PMI reader takes one
hop.  REFUSE is an explicit zero-cost action.

Important repairs relative to the 614--623 exploratory ladder:
* held_ask never filters candidates or features;
* held_ctx and held_ask are two fillers from two rows of one address, so their
  rows are removed from co+df separately (there is no invented ctx--ask pair);
* train/test corpus tails are split before disjoint tapes are built.

The learner receives only the reward of its chosen address (or REFUSE).
Features are name-free counts, address structure, agreement and reciprocal
tape evidence.  Token identities and extracted literals are teacher/tape only.

GATE
  held-out forced address rank beats strongest first/vote/reciprocal/count/
  random/majority/null rival by > .05 with paired bootstrap low95 > 0;
  chosen policy net (hit - .05*READ) beats the strongest one-READ rival net.
VOID
  held-out n < 100 or oracle reach has <= .05 room over strongest rival.

    python _check624_placepolicy.py
    python _audit624_placepolicy.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit624_placepolicy.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit624_placepolicy.py --seed 2890 --corpus data/_tinystories_train.txt
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

from _audit511_ring import pick_corpus
from _audit589_hop3 import adjust_frame_stats
from _audit606_bridge import (
    bands,
    build_places,
    chooser_features,
    place_offer,
)
from _audit615_depth import extracts, walk
from _audit618_peakpin import peak_pin


OUT = Path("results/_stage624_placepolicy.json")
K = 3
CAP = 6
C_STEP = 0.05


def block_windows(pool, length, limit, rng):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    rng.shuffle(blocks)
    return blocks[: min(limit, len(blocks))]


class AddressPolicy(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def legal_unique(place, pin, env_m, mid_set, high_set):
    """Tape-resolved unique extras; target-independent by construction."""
    _bag, uniq = place_offer(place, pin, env_m, mid_set)
    return [tok for tok in uniq if tok in mid_set and tok not in high_set
            and tok != pin]


def door_candidates(pg, pin, skip, env_m, mid_set, high_set,
                    co, df, n_use, frame_max):
    """Group votes but expose one exact supporting address per door action."""
    groups = {}
    scan = list(pg["by_place"].get(pin, ()))
    for order, pi in enumerate(scan):
        if pi == skip:
            continue
        uniq = legal_unique(pg["places"][pi], pin, env_m, mid_set, high_set)
        if len(uniq) != 1:
            continue
        door = uniq[0]
        groups.setdefault(door, []).append((pi, order))
    if not groups:
        return []

    items = list(groups.items())[:CAP]
    total_votes = sum(len(support) for _door, support in items)
    vote_counts = sorted((len(support) for _door, support in items), reverse=True)
    top = vote_counts[0]
    second = vote_counts[1] if len(vote_counts) > 1 else 0
    row_tail = [
        len(items) / CAP,
        min(total_votes, 24) / 24.0,
        min(top, 8) / 8.0,
        min(second, 8) / 8.0,
        (top - second) / max(top, 1),
    ]

    candidates = []
    specific_width = None
    for ci, (door, support) in enumerate(items):
        rep_pi = support[0][0]
        rep = pg["places"][rep_pi]
        # Reciprocal evidence reads only visible env literals.
        back = extracts(pg, door, skip, env_m, mid_set, co, df, n_use)
        back = back[:K]
        recip_n = sum(tok in env_m for tok, _bag, _uniq in back if tok is not None)
        door_places = len(pg["by_place"].get(door, ()))
        ddf = max(df.get(door, 0), 1)
        assoc = [
            co.get((door, tok), 0)
            / math.sqrt(ddf * max(df.get(tok, 0), 1))
            for tok in env_m
        ]
        vote_n = len(support)
        orders = [order for _pi, order in support]
        spread = (max(orders) - min(orders)) / max(len(scan) - 1, 1)
        base = chooser_features(
            rep, pin, env_m, df, n_use, frame_max,
        )
        specific = base + [
            min(vote_n, 8) / 8.0,
            vote_n / max(total_votes, 1),
            (vote_n - max((len(s) for d, s in items if d != door), default=0))
            / max(vote_n, 1),
            recip_n / K,
            float(recip_n > 0),
            min(len(back), K) / K,
            math.log1p(door_places) / 4.0,
            math.log1p(ddf) / max(math.log1p(n_use), 1.0),
            sum(assoc) / max(len(assoc), 1),
            max(assoc, default=0.0),
            ci / max(len(items) - 1, 1),
            spread,
        ]
        specific_width = len(specific)
        candidates.append(dict(
            addr=rep["addr"],
            support_pi=rep_pi,
            door=door,              # tape/teacher only; never enters features
            votes=vote_n,
            recip=recip_n,
            count_key=rep["count_key"],
            majority=rep["majority"],
            feat=specific + row_tail + [0.0],
        ))
    refuse_feat = [0.0] * specific_width + row_tail + [1.0]
    return candidates, refuse_feat


def action_hit(pg, candidate, pin, held, skip, env_m, mid_set, high_set,
               co, df, n_use):
    """Resolve the selected exact address, then READ or take one frozen hop."""
    place = pg["places"][candidate["support_pi"]]
    resolved = legal_unique(place, pin, env_m, mid_set, high_set)
    assert len(resolved) == 1
    door = resolved[0]
    assert door == candidate["door"]
    if door == held:
        return 1
    _first, cumulative = walk(
        pg, door, held, skip, env_m, mid_set, high_set,
        co, df, n_use, {pin, door},
    )
    return int(cumulative[1])


def collect_window(lines, args, seed):
    rng = random.Random(seed)
    pg = build_places(lines, args.frame_max, args.min_fillers)
    if pg is None:
        return []
    mid_set, high_set = bands(pg)
    if not mid_set:
        return []
    co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
    places = pg["places"]
    rows = []
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

            # Two fillers of one address are two records, not one joint row.
            ctx_row = set(query["keys"]) | {held_ctx}
            ask_row = set(query["keys"]) | {held_ask}
            adjust_frame_stats(co, df, ctx_row, -1)
            adjust_frame_stats(co, df, ask_row, -1)
            n_use = max(n_fr - 2, 1)
            try:
                initial = extracts(
                    pg, pin, qi, env_m, mid_set, co, df, n_use,
                )
                if any(tok == held_ask for tok, _b, _u in initial[:K]):
                    continue
                # Candidate construction is held-blind.  A peak answer is READ,
                # not subtracted to manufacture a REFUSE row.
                hat = peak_pin(
                    pg, pin, qi, env_m, mid_set, high_set, {pin},
                )
                if hat is not None:
                    continue
                built = door_candidates(
                    pg, pin, qi, env_m, mid_set, high_set,
                    co, df, n_use, args.frame_max,
                )
                if not built:
                    continue
                candidates, refuse_feat = built
                labels = [
                    action_hit(
                        pg, cand, pin, held_ask, qi, env_m, mid_set, high_set,
                        co, df, n_use,
                    )
                    for cand in candidates
                ]
                if len(candidates) < 2:
                    continue
                rows.append(dict(
                    candidates=candidates,
                    refuse_feat=refuse_feat,
                    labels=labels,
                    held=held_ask,       # evaluator only
                    query_sig=(pin, tuple(sorted(env_m))),
                ))
            finally:
                adjust_frame_stats(co, df, ask_row, +1)
                adjust_frame_stats(co, df, ctx_row, +1)
    return rows


def collect_tapes(windows, args, seed):
    return [
        collect_window(lines, args, seed + 1009 * i)
        for i, lines in enumerate(windows)
    ]


def features_of(row, include_refuse=True):
    feats = [cand["feat"] for cand in row["candidates"]]
    if include_refuse:
        feats.append(row["refuse_feat"])
    return feats


def train_bandit(tapes, seed, epochs, null=False):
    first = next((row for tape in tapes for row in tape), None)
    if first is None:
        return None, 0
    width = len(first["candidates"][0]["feat"])
    torch.manual_seed(seed)
    model = AddressPolicy(width)
    opt = torch.optim.Adam(model.parameters(), lr=0.004)
    policy_rng = random.Random(seed + 17)
    null_rng = random.Random(seed + 991)
    baseline = 0.0
    step = 0
    total = max(sum(len(tape) for tape in tapes) * epochs, 1)
    for _epoch in range(epochs):
        tape_order = list(range(len(tapes)))
        policy_rng.shuffle(tape_order)
        for ti in tape_order:
            rows = list(tapes[ti])
            policy_rng.shuffle(rows)
            for row in rows:
                x = torch.tensor(features_of(row), dtype=torch.float32)
                logits = model(x)
                epsilon = max(0.10, 0.60 * (1.0 - step / total))
                if policy_rng.random() < epsilon:
                    action = policy_rng.randrange(len(logits))
                else:
                    action = int(torch.argmax(logits.detach()).item())
                ys = list(row["labels"])
                if null:
                    null_rng.shuffle(ys)
                is_read = action < len(ys)
                hit = ys[action] if is_read else 0
                reward = (1.0 if hit else -C_STEP) if is_read else 0.0
                advantage = reward - baseline
                logp = torch.log_softmax(logits, dim=0)[action]
                probs = torch.softmax(logits, dim=0)
                entropy = -(probs * torch.log_softmax(logits, dim=0)).sum()
                loss = -advantage * logp - 0.003 * entropy
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                baseline = 0.995 * baseline + 0.005 * reward
                step += 1
    return model, step


def picks(model, row):
    with torch.no_grad():
        all_logits = model(torch.tensor(features_of(row), dtype=torch.float32))
        rank_logits = all_logits[:len(row["candidates"])]
    return (
        int(torch.argmax(rank_logits).item()),
        int(torch.argmax(all_logits).item()),
    )


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_delta(left, right, seed, samples=3000):
    if not left:
        return 0.0, 0.0
    rng = random.Random(seed)
    paired = [a - b for a, b in zip(left, right)]
    n = len(paired)
    draws = [
        sum(paired[rng.randrange(n)] for _j in range(n)) / n
        for _i in range(samples)
    ]
    draws.sort()
    return draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def eval_tapes(tapes, model, null_model):
    names = (
        "learned", "null", "first", "vote", "reciprocal", "count",
        "random", "majority_same", "majority_route", "reach",
    )
    vectors = {name: [] for name in names}
    policy_hits, policy_reads = [], []
    null_policy_hits, null_policy_reads = [], []
    selected_addrs = 0
    for tape in tapes:
        for row in tape:
            candidates, ys = row["candidates"], row["labels"]
            if len(candidates) < 2:
                continue
            li, pa = picks(model, row)
            ni, na = picks(null_model, row)
            assert "addr" in candidates[li]
            selected_addrs += 1

            first_i = 0
            vote_i = max(range(len(candidates)),
                         key=lambda i: (candidates[i]["votes"], -i))
            recip_i = max(
                range(len(candidates)),
                key=lambda i: (
                    candidates[i]["votes"] + 2 * candidates[i]["recip"], -i,
                ),
            )
            count_i = max(range(len(candidates)),
                          key=lambda i: candidates[i]["count_key"])
            maj_i = max(
                range(len(candidates)),
                key=lambda i: candidates[i]["count_key"][0],
            )
            vectors["learned"].append(ys[li])
            vectors["null"].append(ys[ni])
            vectors["first"].append(ys[first_i])
            vectors["vote"].append(ys[vote_i])
            vectors["reciprocal"].append(ys[recip_i])
            vectors["count"].append(ys[count_i])
            vectors["random"].append(sum(ys) / len(ys))
            vectors["majority_same"].append(
                int(candidates[li]["majority"] == row["held"])
            )
            vectors["majority_route"].append(
                int(candidates[maj_i]["majority"] == row["held"])
            )
            vectors["reach"].append(max(ys))

            policy_read = int(pa < len(candidates))
            null_read = int(na < len(candidates))
            policy_reads.append(policy_read)
            null_policy_reads.append(null_read)
            policy_hits.append(ys[pa] if policy_read else 0)
            null_policy_hits.append(ys[na] if null_read else 0)
    assert selected_addrs == len(vectors["learned"])
    return (
        vectors, policy_hits, policy_reads,
        null_policy_hits, null_policy_reads,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--train-windows", type=int, default=80)
    ap.add_argument("--test-windows", type=int, default=40)
    ap.add_argument("--train-epochs", type=int, default=16)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=160000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
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
        f"624 placepolicy  {path}  {kind}  "
        f"windows={len(train_windows)}/{len(test_windows)}",
        flush=True,
    )
    train_tapes = collect_tapes(train_windows, args, args.seed + 101)
    test_tapes = collect_tapes(test_windows, args, args.seed + 202)
    model, updates = train_bandit(
        train_tapes, args.seed + 1, args.train_epochs, null=False,
    )
    null_model, null_updates = train_bandit(
        train_tapes, args.seed + 2, args.train_epochs, null=True,
    )
    if model is None or null_model is None:
        print("VOID: no train rows.")
        return 0

    (
        vectors, policy_hits, policy_reads,
        null_policy_hits, null_policy_reads,
    ) = eval_tapes(test_tapes, model, null_model)
    rates = {name: mean(values) for name, values in vectors.items()}
    rival_names = (
        "null", "first", "vote", "reciprocal", "count", "random",
        "majority_same", "majority_route",
    )
    strongest = max(rival_names, key=lambda name: rates[name])
    delta = rates["learned"] - rates[strongest]
    ci_low, ci_high = bootstrap_delta(
        vectors["learned"], vectors[strongest], args.seed + 404,
    )
    n_test = len(vectors["learned"])
    room = rates["reach"] - rates[strongest]

    policy_hit = mean(policy_hits)
    policy_read = mean(policy_reads)
    policy_net = policy_hit - C_STEP * policy_read
    null_policy_net = (
        mean(null_policy_hits) - C_STEP * mean(null_policy_reads)
    )
    strongest_net = max(
        0.0,
        null_policy_net,
        max(rates[name] - C_STEP for name in rival_names if name != "null"),
    )
    net_delta = policy_net - strongest_net
    void = n_test < 100 or room <= 0.05
    gate = (
        (not void)
        and delta > 0.05
        and ci_low > 0.0
        and net_delta > 0.0
    )

    n_train = sum(len(tape) for tape in train_tapes)
    mean_places = mean([
        len(row["candidates"]) for tape in test_tapes for row in tape
    ])
    print(
        f"rows train/test {n_train}/{n_test}  updates {updates}/{null_updates}  "
        f"places {mean_places:.1f}"
    )
    print("  ".join(
        f"{name} {rates[name]:.3f}"
        for name in (
            "learned", "null", "first", "vote", "reciprocal", "count",
            "random", "majority_same", "majority_route", "reach",
        )
    ))
    print(
        f"strongest {strongest}  delta {delta:+.3f}  "
        f"paired95 [{ci_low:+.3f},{ci_high:+.3f}]  room {room:+.3f}"
    )
    print(
        f"policy hit {policy_hit:.3f}  read {policy_read:.3f}  "
        f"net {policy_net:+.3f}  strongest_net {strongest_net:+.3f}  "
        f"net_delta {net_delta:+.3f}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: honest held-out address ceiling is absent or thin.")
    elif gate:
        print("GO PLACE POLICY: chosen-reward exact-address policy beats all rivals.")
    else:
        print("STOP: exact-address policy does not beat the strongest honest rival.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_train_windows=len(train_windows), n_test_windows=len(test_windows),
        n_train=n_train, n_test=n_test, mean_places=mean_places,
        updates=updates, null_updates=null_updates,
        rates=rates, strongest=strongest, delta=delta,
        ci95=[ci_low, ci_high], room=room,
        policy_hit=policy_hit, policy_read=policy_read,
        policy_net=policy_net, strongest_net=strongest_net,
        net_delta=net_delta,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        actual_place_ids=True,
        chooser_sees_names=False,
        chosen_reward_only=True,
        held_filters_candidates=False,
        two_record_holdout=True,
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
