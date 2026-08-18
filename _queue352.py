"""~7 hours on THE MISSING CONSTRUCTION: the loop. Self-gating, cheapest decision first.

THE DIAGNOSIS THIS QUEUE IS BUILT ON. The pipeline is assembled and it is assembled FOR ONE
SHOT. Question k+1 knows nothing of question k; the answer is scored and discarded; the reward
is terminal; we choose the hole at random. Every closed result in this project - composition,
generation, revision - was closed AS A SINGLE-SHOT OPERATION, and all three are things a mind
does ACROSS steps.

OLD CONSTRAINTS THAT MAY NO LONGER APPLY, and this queue re-opens exactly two of them:

  324 CLOSED MEMORY. It measured a PERFECT write-back's marginal retrieval gain on INDEPENDENT
      questions and found ~0. It never measured a DEPENDENT CHAIN. Mis-scoped for the question
      we now care about, and the closure does not transfer.

  322 CLOSED DEPTH. It measured reachable 0.54 at depth two against 0.12 at depth one - THE
      LARGEST MOVEMENT OF REACH THIS PROJECT HAS EVER RECORDED - and was closed because CONFIRM
      collapsed and an honest rival read 2-3%. Both are single-shot artefacts: CONFIRM broke
      because ONE decision had to serve "answer at home" and "chase depth" at once, and a
      one-hop rival cannot follow a two-hop path by construction. Neither says the chain does
      not reach.

  342b was demoted as "about the architecture we are leaving". WRONG - we are not leaving it,
      we are extending it in TIME. If a chain needs a second decision type (when to stop), the
      4x price of a second objective is load-bearing again, and 342a showed capacity is not
      binding for ONE objective at d=64, which is not the same question.

WHAT IS NOT RE-OPENED: no heuristics, everything a count, rivals closed and declared, nulls and
transplant riding along. Those are why the results mean anything.

    TIER 1  ~1h, torch-free   the ceiling of a two-step chain (_audit351_chain)
    GATE    tiers 2-3 run only if a chain reaches materially more than one step
    TIER 2  ~3h              depth re-run at 4 seeds, read with the 337/339 machinery that did
                             not exist when 322 closed it
    TIER 3  ~2h              342b: re-price the second objective at d=64

    python _queue352.py
    python _queue352.py --go
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STAGE = "_stage289_derivation.py"
RES = (Path("results"), Path("out"))
ARMFILE = Path("results/_arm_ctrl_next.txt")
BASE = ("--reach --tape frames --frame-max 3 --min-mentions 1 --fp hash --write-fp hash "
        "--ink mean --write-ink mean --words ascii --reach-no-refuse --reach-lookahead "
        "--frame-fp fillers --tape-sample region --objective reward --addresses 1500 "
        "--reach-max-q 2000 --import-k 1 --gamma 1.0 --probe-period 1000 --probe-size 60 --cpu")
SEEDS = (1337, 2890, 4711, 8642)


def done(tag):
    return any((d / f"{p}stage289_decision_{tag}.json").exists()
               for d in RES for p in ("", "_"))


def run(cmd, go, label):
    print(f"  [{label}] " + " ".join(cmd[1:]), flush=True)
    if not go:
        return True
    t = time.time()
    r = subprocess.run(cmd)
    print(f"    -> exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    return r.returncode == 0


def swap(base, flag, value):
    out, i, seen = [], 0, False
    toks = list(base)
    while i < len(toks):
        if toks[i] == flag:
            out += [flag, str(value)]
            i += 2
            seen = True
        else:
            out.append(toks[i])
            i += 1
    if not seen:
        out += [flag, str(value)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--wiki", default="data/_wikitext103_train.txt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--skip-tier1", action="store_true")
    args = ap.parse_args()
    base = (args.base or (ARMFILE.read_text(encoding="utf-8").strip()
                          if ARMFILE.exists() else BASE)).split()
    print("arm:", " ".join(base))
    print("PLAN ONLY - add --go\n" if not args.go else "RUNNING\n")
    t0 = time.time()

    if not args.skip_tier1:
        print("TIER 1  the ceiling of a chain (torch-free, ~1h)")
        for tm, br in ((8, 8), (8, 32)):
            run([sys.executable, "_audit351_chain.py", "--topm", str(tm),
                 "--branch", str(br)], args.go, f"351 top{tm} branch{br}")

    # ---- THE GATE ---------------------------------------------------------------------------
    src = Path("results/_stage351_chain.json")
    if args.go and src.exists():
        d = json.loads(src.read_text(encoding="utf-8"))
        moved = d.get("oracle_or_1", 0) > d.get("reach1", 0) + 0.05
        print(f"\nGATE  one step {d.get('reach1', 0):.4f} -> a perfect chooser over "
              f"{d.get('paths', 0):.0f} paths {d.get('oracle_or_1', 0):.4f}  ->  "
              f"{'the chain reaches, tiers 2-3 run' if moved else 'THE CHAIN DOES NOT REACH'}")
        if not moved:
            print("  Tiers 2 and 3 SKIPPED. A second hop over the same relation reaches no more "
                  "than the first, so depth has nothing to route to and the second objective "
                  "has nothing to price. Substitution is closed one step or two, and the "
                  "project's result is the separation proof.")
            print(f"\n{(time.time() - t0) / 60:.0f} min")
            return 0

    # ---- TIER 2: depth, re-run and re-read. ~3h ---------------------------------------------
    #
    # The arm is 322's, unchanged. What is different is the READING: question_rank, the gate,
    # margin_by_stage and GATE-WO did not exist when depth was closed, and every one of them
    # speaks to whether the deep read is chosen well rather than merely taken.
    print("\nTIER 2  depth 2, 4 seeds, read with machinery 322 did not have (~3h)")
    for s in SEEDS:
        tag = f"352deep_s{s}"
        if done(tag):
            print(f"  [{tag}] already present")
            continue
        cmd = [sys.executable, STAGE] + swap(base, "--reach-depth", 2) + [
            "--wiki", args.wiki, "--seed", str(s), "--train-steps", str(args.steps),
            "--run-tag", tag]
        run(cmd, args.go, tag)

    # ---- TIER 3: 342b, un-demoted. ~2h ------------------------------------------------------
    #
    # Two unrelated second objectives cost the route 4.0x and 3.9x. If the loop needs a second
    # decision type - when to stop chaining - that price decides whether it can ever be added
    # or whether it must replace the objective. 342a settled that capacity is not binding for
    # ONE objective at d=64; this is the other question, and it was never asked.
    print("\nTIER 3  342b: is the 4x price capacity or law, at d=64 (~2h)")
    for s in SEEDS:
        tag = f"352price_s{s}"
        if done(tag):
            print(f"  [{tag}] already present")
            continue
        cmd = [sys.executable, STAGE] + swap(base, "--dim", 64) + [
            "--speak-batch", "8", "--speak-weight", "1.0",
            "--wiki", args.wiki, "--seed", str(s),
            "--train-steps", str(max(1, args.steps // 8)), "--run-tag", tag]
        run(cmd, args.go, tag)

    print(f"\n{(time.time() - t0) / 60:.0f} min")
    print("read with:")
    print("  python _read299.py <352deep reports> --held     # DEPTH, GATE-WO, MARGIN, RANK")
    print("  python _read299.py <352price reports> --held    # route first: the 4x is the veto")
    print("  compare 352price against the d=64 ctrl already in hand (342news_d64)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
