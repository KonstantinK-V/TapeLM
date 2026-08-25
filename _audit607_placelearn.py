"""607: learn which real tape place to READ; tape resolves the value.

606 opened a large ceiling on exact frame addresses. This stage trains a
one-READ contextual bandit on rotating tapes:

  action   index of an exact (w,left,right) address
  input    pre-READ counts and equality-to-query only
  hidden   address words, extras, PMI, held, extracted value
  teacher  +1 iff the selected place's frozen extract equals held, else -0.08
  output   tape literal from the selected place, or REFUSE

The corpus is split before disjoint windows. The null policy receives the same
features and reward mass with labels shuffled inside each offer. 542 is neither
changed nor retrained.

Rivals: first | count-only place rule | expected random | same-place majority
| fixed majority-place route | shuffled null. Pooled bag majority is report-only.
GATE: learned - strongest > 0.05 and paired bootstrap lower 95% > 0.
VOID: held-out n < 100 or REACH - strongest <= 0.05.

    python _check607_placelearn.py
    python _audit607_placelearn.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit607_placelearn.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit607_placelearn.py --seed 2890 --corpus data/_tinystories_train.txt
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
from _audit606_bridge import collect

OUT = Path("results/_stage607_placelearn.json")


def block_windows(pool, length, limit, rng):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    rng.shuffle(blocks)
    return blocks[: min(limit, len(blocks))]


class PlacePolicy(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 24),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rows_of(windows, args, seed):
    rng = random.Random(seed)
    return [collect(lines, args, rng) for lines in windows]


def labels_of(row):
    held = row["held"]
    return [int(place["extract"] == held) for place in row["places"]]


def features_of(row):
    return [place["feat"] for place in row["places"]]


def train_policy(tapes, seed, epochs, null=False):
    first = next((row for tape in tapes for row in tape), None)
    if first is None:
        return None, 0
    width = len(first["places"][0]["feat"])
    torch.manual_seed(seed)
    model = PlacePolicy(width)
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
                feats = features_of(row)
                if len(feats) < 2:
                    continue
                x = torch.tensor(feats, dtype=torch.float32)
                logits = model(x)
                epsilon = max(0.08, 0.50 * (1.0 - step / total))
                if policy_rng.random() < epsilon:
                    action = policy_rng.randrange(len(feats))
                else:
                    action = int(torch.argmax(logits.detach()).item())
                ys = labels_of(row)
                if null:
                    null_rng.shuffle(ys)
                hit = ys[action]
                reward = 1.0 if hit else -0.08
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


def model_pick(model, row):
    feats = torch.tensor(features_of(row), dtype=torch.float32)
    with torch.no_grad():
        return int(torch.argmax(model(feats)).item())


def eval_rows(tapes, model, null_model):
    scores = {
        "learned": [], "null": [], "first": [], "count": [],
        "random": [], "majority_same": [], "majority_route": [],
        "bag_majority": [], "reach": [],
    }
    refuse = 0
    n_places = 0
    conflicts = 0
    for tape in tapes:
        tape_sig = defaultdict(Counter)
        for row in tape:
            tape_sig[row["query_sig"]][row["held"]] += 1
        conflicts += sum(
            sum(cnt.values()) - max(cnt.values()) for cnt in tape_sig.values()
        )
        for row in tape:
            places = row["places"]
            if len(places) < 2:
                continue
            ys = labels_of(row)
            pi = model_pick(model, row)
            ni = model_pick(null_model, row)
            assert "addr" in places[pi]
            ci = max(range(len(places)), key=lambda i: places[i]["count_key"])
            majority = Counter(row["bag0"]).most_common(1)[0][0]
            scores["learned"].append(ys[pi])
            scores["null"].append(ys[ni])
            scores["first"].append(ys[0])
            scores["count"].append(ys[ci])
            scores["random"].append(sum(ys) / len(ys))
            scores["majority_same"].append(
                int(places[pi]["majority"] == row["held"])
            )
            scores["majority_route"].append(
                int(places[ci]["majority"] == row["held"])
            )
            scores["bag_majority"].append(int(majority == row["held"]))
            scores["reach"].append(max(ys))
            refuse += int(places[pi]["extract"] is None)
            n_places += len(places)
    n = len(scores["learned"])
    return scores, refuse, n_places, conflicts / max(n, 1)


def mean(values):
    return sum(values) / len(values) if values else 0.0


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--train-windows", type=int, default=80)
    ap.add_argument("--test-windows", type=int, default=40)
    ap.add_argument("--train-epochs", type=int, default=6)
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
        f"607 placelearn  {path}  {kind}  "
        f"windows={len(train_windows)}/{len(test_windows)}",
        flush=True,
    )
    train_tapes = rows_of(train_windows, args, args.seed + 101)
    test_tapes = rows_of(test_windows, args, args.seed + 202)
    model, updates = train_policy(
        train_tapes, args.seed + 1, args.train_epochs, null=False,
    )
    null_model, null_updates = train_policy(
        train_tapes, args.seed + 2, args.train_epochs, null=True,
    )
    if model is None or null_model is None:
        print("VOID: no train rows.")
        return 0

    vectors, n_refuse, n_places, collision = eval_rows(
        test_tapes, model, null_model,
    )
    rates = {name: mean(values) for name, values in vectors.items()}
    rivals = (
        "null", "first", "count", "random",
        "majority_same", "majority_route",
    )
    strongest = max(rivals, key=lambda name: rates[name])
    delta = rates["learned"] - rates[strongest]
    ci_low, ci_high = bootstrap_delta(
        vectors["learned"], vectors[strongest], args.seed + 404,
    )
    n_test = len(vectors["learned"])
    room = rates["reach"] - rates[strongest]
    void = n_test < 100 or room <= 0.05 or collision > 0.02
    gate = (not void) and delta > 0.05 and ci_low > 0.0
    capture = delta / room if room > 0 else 0.0

    n_train = sum(len(tape) for tape in train_tapes)
    print(
        f"rows train/test {n_train}/{n_test}  updates {updates}/{null_updates}  "
        f"places {n_places / max(n_test, 1):.1f}"
    )
    print(
        "  ".join(f"{name} {rates[name]:.3f}" for name in
                 ("learned", "null", "first", "count", "random",
                  "majority_same", "majority_route", "bag_majority", "reach"))
    )
    print(
        f"strongest {strongest}  delta {delta:+.3f}  "
        f"paired95 [{ci_low:+.3f},{ci_high:+.3f}]  "
        f"room {room:+.3f}  capture {capture:.3f}  "
        f"refuse {n_refuse / max(n_test, 1):.3f}  collision {collision:.3f}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: held-out place ceiling is absent or test is thin.")
    elif gate:
        print("GO PLACE POLICY: learned address choice beats every rival.")
    else:
        print("STOP: learned address choice does not beat the strongest rival.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_train_windows=len(train_windows), n_test_windows=len(test_windows),
        n_train=n_train, n_test=n_test,
        updates=updates, null_updates=null_updates,
        mean_places=n_places / max(n_test, 1),
        rates=rates, strongest=strongest, delta=delta,
        ci95=[ci_low, ci_high], room=room, capture=capture,
        refuse=n_refuse / max(n_test, 1),
        collision=collision,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        actual_place_ids=True, chooser_sees_extras=False,
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
