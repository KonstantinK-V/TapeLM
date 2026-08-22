"""497 NIGHT: learn refuse when not unique (teacher = unique_next).

At each try: action GO vs REFUSE. Reward: GO+unique +1, GO+not -1,
REFUSE+not +0.3, REFUSE+unique -0.5. Compare GO-rate and accuracy vs
always-GO / random.

    python _audit497_refuse.py --seed 1337 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit485_hunt import build_window, load_lines, pick_corpus, pre, unique_next

OUT = Path("results/_stage497_refuse.json")


def q_get(table, key, action):
    return table.get((key, action), 0.0)


def touch(table, tot, win, key, action, r):
    k = (key, action)
    tot[k] += 1
    win[k] += r
    table[k] = win[k] / tot[k]


def pick_action(table, key, rng, eps):
    if rng.random() < eps:
        return rng.choice(("GO", "REFUSE"))
    go = q_get(table, key, "GO")
    rf = q_get(table, key, "REFUSE")
    if go == rf:
        return rng.choice(("GO", "REFUSE"))
    return "GO" if go > rf else "REFUSE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    path = pick_corpus(args.corpus)
    lines = load_lines(path, args.bytes, 20, random.Random(args.seed))
    print(f"497 refuse  corpus={path}", flush=True)

    q, tot, win = {}, defaultdict(int), defaultdict(float)
    rng = random.Random(args.seed)
    n_ok = 0
    s = defaultdict(float)
    t0 = time.time()

    for i in range(args.steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, args.window, args.frame_max)
        if g is None:
            continue
        n_ok += 1
        places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
        if len(places) < 4:
            continue
        eps = max(0.05, 0.45 * (1 - i / max(args.steps, 1)))
        for _ in range(args.tries):
            P = rng.choice(places)
            k = pre(P, g)
            uniq = unique_next(P, g) is not None
            act = pick_action(q, k, rng, eps)
            s["n"] += 1
            s["uniq"] += int(uniq)
            if act == "GO":
                s["go"] += 1
                if uniq:
                    s["go_hit"] += 1
                    touch(q, tot, win, k, "GO", 1.0)
                else:
                    s["go_miss"] += 1
                    touch(q, tot, win, k, "GO", -1.0)
            else:
                s["refuse"] += 1
                if uniq:
                    s["refuse_miss"] += 1
                    touch(q, tot, win, k, "REFUSE", -0.5)
                else:
                    s["refuse_ok"] += 1
                    touch(q, tot, win, k, "REFUSE", 0.3)
        if (i + 1) % args.log_every == 0:
            n = max(s["n"], 1)
            print(
                f"  step {i+1}/{args.steps}  go {s['go']/n:.3f}  "
                f"go_acc {s['go_hit']/max(s['go'],1):.3f}  "
                f"ref_ok {s['refuse_ok']/max(s['refuse'],1):.3f}  "
                f"uniq_base {s['uniq']/n:.3f}  q {len(q)}  "
                f"{time.time()-t0:.0f}s",
                flush=True,
            )

    n = max(s["n"], 1)
    # baseline always-GO accuracy = uniq rate
    rec = dict(
        seed=args.seed,
        corpus=str(path),
        n_windows=n_ok,
        go_rate=s["go"] / n,
        go_acc=s["go_hit"] / max(s["go"], 1),
        refuse_rate=s["refuse"] / n,
        refuse_ok=s["refuse_ok"] / max(s["refuse"], 1),
        uniq_base=s["uniq"] / n,
        lift_go_acc=(s["go_hit"] / max(s["go"], 1)) - (s["uniq"] / n),
        n_q=len(q),
        elapsed_s=round(time.time() - t0, 1),
        note="GO/REFUSE vs unique teacher; lift_go_acc vs always-GO",
    )
    print("---- done ----", flush=True)
    print(f"go_acc {rec['go_acc']:.4f} uniq_base {rec['uniq_base']:.4f} "
          f"lift {rec['lift_go_acc']:.4f} refuse_ok {rec['refuse_ok']:.4f}",
          flush=True)
    out = Path(args.out)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
