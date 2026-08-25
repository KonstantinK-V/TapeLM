"""542 repaired: curriculum for hunting real places on novel tapes.

The original 542 paired arm windows but trained and evaluated on the same line
pool. Its online null also changed its own pick trajectory, so it did not receive
the claimed same reward stream.

This repair:
  * splits corpus first, then builds disjoint train/test tape windows;
  * builds each tape once and shares it across arms;
  * records A's exact (count-key,reward) stream and makes C by shuffling rewards
    across those same keys: same touches, keys and reward mass, association gone;
  * evaluates a max-majority place route in addition to random;
  * keeps actions as exact frame addresses; Q contains only the old count key.

    A  narrow2 -> eval unique                 curriculum
    B  unique  -> eval unique                 hard from scratch
    C  shuffled rewards on A's exact trace   fair credit null
    D  max-majority place route              frequency rival

GATE: A beats B, C and D by >0.05 on untouched-tail tapes.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit485_hunt import (
    build_window, narrow_next, pick_by_q, pick_corpus, pre, touch,
    unique_next,
)

OUT = Path("results/_stage542_curric.json")


def hunt(g, rng, q, tot, win, eps, budget, mode, learn, trace=None):
    """One place-hunt episode; optional trace records exact learner credit."""
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return 0, 0
    seen = set()
    n_touch = 0
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q, lambda x: pre(x, g), rng, eps if learn else 0.0)
        if P is None:
            break
        seen.add(P)
        k = pre(P, g)
        if mode == "unique":
            hit = unique_next(P, g) is not None
        else:
            hit = narrow_next(P, g, kmax=2) is not None
        if learn:
            reward = 1.0 if hit else -0.08
            if trace is not None:
                trace.append((k, reward))
            touch(q, tot, win, k, reward)
            n_touch += 1
        if hit:
            return 1, n_touch
    return 0, n_touch


def rand_unique(g, rng, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if not places:
        return 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = rng.choice(pool)
        seen.add(P)
        if unique_next(P, g) is not None:
            return 1
    return 0


def majority_unique(g, budget):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    scored = []
    for P in places:
        vals = [g["value"][i] for i in g["slots_at"][P]]
        frac = Counter(vals).most_common(1)[0][1] / max(len(vals), 1)
        scored.append((frac, P))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return int(any(unique_next(P, g) is not None for _frac, P in scored[:budget]))


def block_windows(pool, length, limit, rng):
    blocks = [
        pool[start:start + length]
        for start in range(0, len(pool) - length + 1, length)
    ]
    rng.shuffle(blocks)
    return blocks[: min(limit, len(blocks))]


def build_tapes(windows, args):
    tapes = []
    for lines in windows:
        g = build_window(
            lines, random.Random(args.seed), args.window, args.frame_max,
        )
        if g is not None:
            tapes.append(g)
    return tapes


def train(tapes, args, mode, record_trace=False):
    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng_tape = random.Random(args.seed)
    rng_pol = random.Random(args.seed + 5)
    trace = [] if record_trace else None
    n = hits = touches = 0
    for i in range(args.train_steps):
        g = tapes[rng_tape.randrange(len(tapes))]
        n += 1
        eps = max(0.05, 0.5 * (1 - i / max(args.train_steps, 1)))
        h, t = hunt(
            g, rng_pol, q, tot, win, eps, args.budget, mode, True, trace,
        )
        hits += h
        touches += t
    stats = dict(
        train_hit=hits / max(n, 1), n_train=n,
        n_touch=touches, n_keys=len(q),
    )
    return q, tot, win, stats, trace


def shuffled_credit(trace, seed, n_train, train_hit):
    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rewards = [reward for _key, reward in trace]
    random.Random(seed).shuffle(rewards)
    for (key, _reward), shuffled_reward in zip(trace, rewards):
        touch(q, tot, win, key, shuffled_reward)
    stats = dict(
        train_hit=train_hit, n_train=n_train,
        n_touch=len(trace), n_keys=len(q),
    )
    return q, tot, win, stats


def evaluate(tapes, args, q, tot, win):
    rng_tape = random.Random(args.seed + 77)
    rng_pol = random.Random(args.seed + 991)
    rng_rnd = random.Random(args.seed + 2024)
    n = hq = hr = hm = 0
    for _i in range(args.eval_steps):
        g = tapes[rng_tape.randrange(len(tapes))]
        n += 1
        h, _ = hunt(g, rng_pol, q, tot, win, 0.0, args.budget, "unique", False)
        hq += h
        hr += rand_unique(g, rng_rnd, args.budget)
        hm += majority_unique(g, args.budget)
    return dict(
        n_eval=n,
        eval_unique=hq / max(n, 1),
        eval_rand=hr / max(n, 1),
        eval_maj=hm / max(n, 1),
        lift=(hq - hr) / max(n, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-steps", type=int, default=2500)
    ap.add_argument("--eval-steps", type=int, default=1500)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--train-windows", type=int, default=80)
    ap.add_argument("--test-windows", type=int, default=40)
    ap.add_argument("--lines", type=int, default=160000)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path = pick_corpus(args.corpus)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    lines = [
        line.strip() for line in text.split("\n") if len(line.strip()) >= 20
    ][: args.lines]
    cut = int(0.7 * len(lines))
    train_pool, test_pool = lines[:cut], lines[cut:]
    rng_windows = random.Random(args.seed + 313)
    train_windows = block_windows(
        train_pool, args.window, args.train_windows, rng_windows,
    )
    test_windows = block_windows(
        test_pool, args.window, args.test_windows, rng_windows,
    )
    tframes._KEEP_MEMO.clear()
    train_tapes = build_tapes(train_windows, args)
    test_tapes = build_tapes(test_windows, args)
    if not train_tapes or not test_tapes:
        print("VOID: no disjoint train/test tapes.")
        return 0
    t0 = time.time()
    print(
        f"542 curriculum repaired  corpus={path}  "
        f"tapes={len(train_tapes)}/{len(test_tapes)}",
        flush=True,
    )

    q_a, tot_a, win_a, tr_a, trace = train(
        train_tapes, args, "narrow2", record_trace=True,
    )
    q_b, tot_b, win_b, tr_b, _ = train(
        train_tapes, args, "unique", record_trace=False,
    )
    q_c, tot_c, win_c, tr_c = shuffled_credit(
        trace, args.seed + 733, tr_a["n_train"], tr_a["train_hit"],
    )
    arms = {}
    for name, q, tot, win, tr in (
        ("A_curric", q_a, tot_a, win_a, tr_a),
        ("B_scratch", q_b, tot_b, win_b, tr_b),
        ("C_null", q_c, tot_c, win_c, tr_c),
    ):
        ev = evaluate(test_tapes, args, q, tot, win)
        arms[name] = dict(**tr, **ev)
        print(
            f"{name:10} train_hit {tr['train_hit']:.3f} "
            f"touches {tr['n_touch']} keys {tr['n_keys']}  "
            f"eval {ev['eval_unique']:.4f} rand {ev['eval_rand']:.4f} "
            f"maj {ev['eval_maj']:.4f} lift {ev['lift']:+.4f}",
            flush=True,
        )

    rands = {a["eval_rand"] for a in arms.values()}
    majs = {a["eval_maj"] for a in arms.values()}
    paired = len(rands) == 1
    d_scratch = arms["A_curric"]["lift"] - arms["B_scratch"]["lift"]
    d_null = arms["A_curric"]["lift"] - arms["C_null"]["lift"]
    d_maj = (
        arms["A_curric"]["eval_unique"] - arms["A_curric"]["eval_maj"]
    )
    null_touches_equal = tr_a["n_touch"] == tr_c["n_touch"]
    train_addrs = {P for g in train_tapes for P in g["slots_at"]}
    test_addrs = {P for g in test_tapes for P in g["slots_at"]}
    addr_overlap = len(train_addrs & test_addrs) / max(len(test_addrs), 1)
    void = (
        (not paired) or len(majs) != 1 or not null_touches_equal
        or arms["A_curric"]["n_eval"] < 200
        or arms["A_curric"]["n_keys"] < 3
    )
    gate = (
        (not void) and d_scratch > 0.05
        and d_null > 0.05 and d_maj > 0.05
    )
    rec = dict(
        seed=args.seed, corpus=str(path), arms=arms,
        n_train_tapes=len(train_tapes), n_test_tapes=len(test_tapes),
        paired_rand=bool(paired), null_touches_equal=bool(null_touches_equal),
        address_overlap=addr_overlap,
        d_scratch=d_scratch, d_null=d_null, d_maj=d_maj,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
    )
    print(
        f"A-B {d_scratch:+.4f}  A-C {d_null:+.4f}  A-MAJ {d_maj:+.4f}  "
        f"address_overlap {addr_overlap:.4f}"
    )
    print(
        f"paired_rand {paired}  null_touches_equal {null_touches_equal}"
    )
    print(f"VOID {void}   GATE {gate}")
    if not paired:
        print("\nVOID: the arms did not see the same baseline - windows are not paired.")
    elif void:
        print("\nVOID: too few eval windows, or fewer than 3 keys learned.")
    elif d_null <= 0.05:
        print("\nCREDIT NULL: curriculum association does not survive exact shuffle.")
    elif d_scratch <= 0.05:
        print("\nEASY-FIRST does not beat hard-first on untouched-tail tapes.")
    elif d_maj <= 0.05:
        print("\nCOUNT Q does not beat the majority place route.")
    else:
        print("\nGO: curriculum count-Q transfers to novel tapes and beats scratch, "
              "exact credit null, and majority route.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
