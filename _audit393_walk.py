"""393: A WALKER, NOT A SMARTER MIND. One step to a bridge, then read from there.

392 closed bind: addr-N(A)∩addr-N(B) pays, and a B from another window pays the same.
391 closed requirement 4 on the eight. 388 measured hop-2 dump: the truth is often two hops
away (hop2-only ~0.18) and hop2@8 does not beat hop1@8. Dumping hop-2 into eight is 347.

This file asks the remaining question as a count: if the mind STEPS to one hop-1 place and
reads THAT place's walk@8, does the truth land more than staying put? The output of the step
is a place. The eight after the step are content of the arrived place's neighbourhood, not a
merged hop-2 dump.

  hop1          standing offer (step interleaved with connect, cap 8)     TODAY
  committed     walk@8 rooted at the single nearest hop-1 place           A WALKER
  oracle        truth in walk@8 of SOME hop-1 bridge (cap --bridges)      A PERFECT STEP
  dump2         388's hop-2 VALUE dump, cap 8                             347, the rival
  rand          walk@8 rooted at a random place                           the floor

qprof on the question. Bridges are walked from their full profile (they are handles).
Same-line of the question dropped. Section 27: the question place's prof[pid] is never a key.

VOID, read first. oracle - hop1 < 0.02: nowhere to walk, close without a chooser.
GATE, declared before the run, four seeds.
  1. oracle - hop1 > 0.05 on 3/4.
  2. dump2 does not beat oracle (a walker is not a wider offer).
  3. rand below half of oracle.
committed is reported, not a gate: if the top bridge loses, the step is real and the ranking
of bridges is the next lever (386's shape). Nothing here claims the mind got smarter.

    python _check393_walk.py
    python _audit393_walk.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _audit390_address as A

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage393_walk.json")


def dump_hop2(T, pid, qprof, own, bridges, drop):
    """388's hop-2 dump: values at places two hops out, scored by min(overlap) summed over paths."""
    seen1 = set(bridges) | {pid}
    score = Counter()
    ov1, _ = A.filler_nbrs(T, pid, qprof, drop)
    for b in bridges:
        ob = ov1[b]
        ov2, _ = A.filler_nbrs(T, b, T["prof"][b], drop | seen1)
        for j, oj in ov2.items():
            wgt = min(ob, oj)
            for u in T["prof"][j]:
                if u not in own:
                    score[u] += wgt
    return [v for v, _n in score.most_common()]


def from_bridge(T, b, own, drop, k, cap):
    """Arrive at b, then the standing walk from b, cap 8. b's full profile: it is a handle."""
    order = A.walk_order(T, b, T["prof"][b], k, drop)
    return A.fillers_of(T, order, own)[:cap]


def measure(T, s, args, rng):
    toks = T["toks"]
    pid = T["place_of"][s]
    truth = toks[s]
    own = {toks[x] for x in T["places"][pid] if x != s}
    if not own or truth in own:
        return None
    qprof = Counter(toks[x] for x in T["places"][pid] if x != s)
    drop = set(T["on_line"][T["owner"][s]])
    drop.discard(pid)

    st = A.lane_step(T, pid, own, qprof, args.places, drop)
    sh = A.lane_share(T, pid, own, qprof, drop, norm_by_places=False)
    hop1 = A.interleave(st, sh, cap=args.topm)
    bridges = A.walk_order(T, pid, qprof, args.bridges, drop)
    dump = dump_hop2(T, pid, qprof, own, bridges, drop)[:args.topm]

    offers = [from_bridge(T, b, own, drop | {pid}, args.places, args.topm) for b in bridges]
    committed = offers[0] if offers else []
    oracle = any(truth in of for of in offers)
    dump_hit = truth in dump
    hop1_hit = truth in hop1
    comm_hit = truth in committed
    others = [j for j in range(len(T["places"])) if j != pid and j not in drop]
    rnd = from_bridge(T, others[rng.randrange(len(others))], own, drop | {pid},
                      args.places, args.topm) if others else []
    return {
        "n": 1,
        "hop1": int(hop1_hit), "committed": int(comm_hit),
        "oracle": int(oracle), "dump2": int(dump_hit),
        "rand": int(truth in rnd),
        "walk_only": int(oracle and not hop1_hit),
        "n_bridges": len(bridges),
        "_lanes": {"hop1": hop1, "committed": committed, "dump2": dump},
    }


def run(T, args, rng):
    c = Counter()
    qs = [s for ps in T["places"] for s in ps]
    rng.shuffle(qs)
    for s in qs:
        if c["n"] >= args.max_questions:
            break
        m = measure(T, s, args, rng)
        if m is None:
            continue
        for k, v in m.items():
            if not k.startswith("_"):
                c[k] += v
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--bridges", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0: s0 + args.window_lines]

    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    c = run(T, args, rng)
    n = max(1, c["n"])

    def f(k):
        return c[k] / n

    rep = {
        "seed": args.seed, "places": len(T["places"]), "n": c["n"],
        "n_bridges": c["n_bridges"] / n,
        "hop1": f("hop1"), "committed": f("committed"), "oracle": f("oracle"),
        "dump2": f("dump2"), "rand": f("rand"), "walk_only": f("walk_only"),
        "oracle_minus_hop1": f("oracle") - f("hop1"),
        "dump_minus_oracle": f("dump2") - f("oracle"),
    }
    void = rep["oracle_minus_hop1"] < 0.02
    print(f"places {rep['places']}  questions {rep['n']}  bridges/q {rep['n_bridges']:.1f}")
    print(f"VOID CHECK  oracle-hop1 {rep['oracle_minus_hop1']:+.4f}"
          + ("  NOWHERE TO WALK" if void else "  ok"))
    print(f"HIT        hop1 {rep['hop1']:.4f}  committed {rep['committed']:.4f}  "
          f"oracle {rep['oracle']:.4f}  dump2 {rep['dump2']:.4f}  rand {rep['rand']:.4f}  "
          f"walk_only {rep['walk_only']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
