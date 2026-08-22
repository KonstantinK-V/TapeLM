"""351'S DOOR TWO, ITS CEILING: does it matter WHERE the mind stands?

The walk is the life - no drawn questions, the mind stands somewhere, may speak or step, and the
reward is truths spoken under a budget over a trajectory. Before any of that is written, one
number decides whether there is a decision in it at all: if every place is equally productive,
choosing where to speak is worth nothing and the loop would be a policy over an indifferent world.

    value(place)   the share of that place's holes whose truth the walk from it already reaches
    oracle         speak at the B best places
    random         speak at B random places
    spread         the gap between the top decile and the median

  VOID   spread ~ 0: nowhere is better than anywhere.
  GATE   oracle - random > 0.05 per utterance, on 3 seeds.

The leak discipline is 390's: the asking hole is out of its own place's profile, and places sharing
a line with it are dropped.

    python _audit407_where.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _audit390_address as A


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--max-places", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default="results/_stage407_where.json")
    args = ap.parse_args()
    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    if not T["places"]:
        print("no tape")
        return 1
    toks, owner = T["toks"], T["owner"]
    ids = list(range(len(T["places"])))
    rng.shuffle(ids)
    vals = []
    for pid in ids[:args.max_places]:
        slots = T["places"][pid]
        if len(slots) < 2:
            continue
        ok = 0
        for s in slots:
            truth = toks[s]
            own = {toks[x] for x in slots if x != s}
            qprof = Counter(toks[x] for x in slots if x != s)
            drop = set(T["on_line"][owner[s]])
            drop.discard(pid)
            walked = A.walk_order(T, pid, qprof, args.places, drop)
            ok += int(truth in A.fillers_of(T, walked, own)[:args.topm])
        vals.append(ok / len(slots))
    vals.sort(reverse=True)
    n = len(vals)
    B = min(args.budget, n)
    oracle = sum(vals[:B]) / B
    rnd = sum(vals) / n
    med = vals[n // 2]
    top10 = sum(vals[:max(1, n // 10)]) / max(1, n // 10)
    rep = {"seed": args.seed, "places": n, "budget": B, "oracle": oracle, "random": rnd,
           "median": med, "top_decile": top10, "gain": oracle - rnd, "spread": top10 - med}
    print(f"{n} places, budget {B}")
    print(f"VALUE       median {med:.4f}   top decile {top10:.4f}   spread {rep['spread']:+.4f}")
    print(f"SPEAK       oracle {oracle:.4f}   random {rnd:.4f}   gain {rep['gain']:+.4f}")
    rep["gate"] = bool(rep["gain"] > 0.05)
    print("\n" + ("WHERE YOU STAND MATTERS: choosing the place is worth more than the bar, so "
                 "351's trajectory has a decision in it and the loop is worth writing."
                 if rep["gate"] else
                 "NOWHERE IS BETTER THAN ANYWHERE: choosing where to speak is under the bar, and "
                 "a trajectory policy would be learning over an indifferent world."))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
