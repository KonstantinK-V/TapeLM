"""FULL BODY REORDER AS REPAIR. Not 'which line was next in the file' - can a policy finish at all?

THE QUESTION CHANGES, NOT THE MODEL. Teacher forcing (403/404) asked the next line given the
TRUE prefix. Here the body is shuffled into a pool and the policy must place every line; success
means EVERY prefix stayed def-use safe and the pool emptied. No score against the original order.

THE OPERATION-WALKER READING. Ballot = remaining safe lines. Trace = the prefix so far. Payment =
completed the body or stuck. No Phi - only three rival policies:

    random legal     uniform among safe lines now
    greedy Return    among safe, prefer a line whose node types include Return; else first safe
    greedy unblocks  among safe, prefer the line that unblocks the most remaining lines

  VOID CHECK, READ FIRST
      greedy unblocks already completes >0.90 of bodies -> the arena decides nothing.

  GATE, on a FOREIGN corpus, 3 of 3 seeds
      unblocks - random > 0.05 AND unblocks - Return > 0.05
      If not, the world is indistinguishable from a Return bit and Phi is not needed.

    python _audit405_repair.py --seed 1337
    python _audit405_repair.py --seed 8642
    python _audit405_repair.py --seed 2890
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import _audit404_family as F

OUT = Path("results/_stage405_repair.json")
DEFAULT_FOREIGN = os.path.join(sys.prefix, "Lib")


def keys_of(body):
    stores, loads = set(), set()
    for _l, s, d, _ty, _ids in body:
        stores |= s
        loads |= d
    return stores & loads


def safe_ix(pool, have, keys):
    return [i for i, r in enumerate(pool) if not ((r[2] & keys) - have)]


def unblocks(cand, pool, have, keys):
    n = 0
    for r in pool:
        if r is cand:
            continue
        need = (r[2] & keys) - have
        if need and not (need - cand[1]):
            n += 1
    return n


def pick_random(pool, safe, have, keys, placed, rng):
    return rng.choice(safe)


def pick_return(pool, safe, have, keys, placed, rng):
    ret = [i for i in safe if "Return" in pool[i][3]]
    return min(ret) if ret else min(safe)


def pick_unblocks(pool, safe, have, keys, placed, rng):
    return max(safe, key=lambda i: (unblocks(pool[i], pool, have, keys), -i))


def complete(body, pick_fn, rng):
    """Shuffle the full body, then walk until empty or stuck. None if the body has no keys."""
    keys = keys_of(body)
    if not keys:
        return None
    pool = list(body)
    rng.shuffle(pool)
    have = set()
    while pool:
        safe = safe_ix(pool, have, keys)
        if not safe:
            return False
        i = pick_fn(pool, safe, have, keys, [], rng)
        row = pool.pop(i)
        have |= row[1]
    return True


def run_files(files, rng, max_body=40):
    c = Counter()
    for p in files:
        for body in F.rows_cached(p, max_body):
            keys = keys_of(body)
            if not keys:
                c["skip"] += 1
                continue
            c["n"] += 1
            for tag, fn in (("random", pick_random), ("return", pick_return),
                            ("unblocks", pick_unblocks)):
                got = complete(body, fn, rng)
                if got:
                    c[tag] += 1
    return c


def rates(c):
    n = max(1, c["n"])
    return {k: c[k] / n for k in ("random", "return", "unblocks")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_FOREIGN,
                    help="foreign corpus (default: Python stdlib Lib/)")
    ap.add_argument("--glob", default="*.py")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--max-body", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    root = Path(args.corpus)
    files = sorted(root.glob(args.glob)) if root.is_dir() else []
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[: args.max_files]
    if not files:
        print(f"no files under {root}")
        return 1

    c = run_files(files, rng, args.max_body)
    r = rates(c)
    rep = {"seed": args.seed, "corpus": str(root), "files": len(files),
           "bodies": c["n"], "skipped_no_keys": c["skip"],
           "random": r["random"], "return": r["return"], "unblocks": r["unblocks"],
           "unblocks_minus_random": r["unblocks"] - r["random"],
           "unblocks_minus_return": r["unblocks"] - r["return"]}
    print(f"{rep['files']} foreign files, {rep['bodies']} bodies with def-use keys "
          f"({rep['skipped_no_keys']} skipped)")
    print(f"VOID CHECK  greedy unblocks completes {rep['unblocks']:.4f}  "
          f"<- read first: >0.90 and the arena decides nothing")
    print(f"COMPLETE    random {rep['random']:.4f}   Return {rep['return']:.4f}   "
          f"unblocks {rep['unblocks']:.4f}")
    print(f"            unblocks-random {rep['unblocks_minus_random']:+.4f}   "
          f"unblocks-Return {rep['unblocks_minus_return']:+.4f}")

    void = rep["unblocks"] > 0.90
    gate = (rep["unblocks_minus_random"] > 0.05 and rep["unblocks_minus_return"] > 0.05)
    rep.update({"void": bool(void), "gate": bool(gate and not void)})
    if void:
        print("\nVOID: greedy unblocks already finishes almost every body - nothing to decide.")
    elif gate:
        print("\nTHE REPAIR WORLD SEPARATES: unblocks beats random and Return by more than 0.05 - "
              "structure beyond a Return bit exists here without Phi.")
    else:
        print("\nTHE WORLD DOES NOT SEPARATE: " +
              ("unblocks does not beat random by 0.05. " if rep["unblocks_minus_random"] <= 0.05
               else "") +
              ("unblocks does not beat Return by 0.05 - indistinguishable from a Return bit. "
               if rep["unblocks_minus_return"] <= 0.05 else "") +
              "Phi is not needed on this arena.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
