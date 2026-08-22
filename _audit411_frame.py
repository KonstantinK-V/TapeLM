"""THE PIN CHOSEN BY THE FRAME, BEFORE THE WORD IS KNOWN. Torch-free.

410 pinned into the home of an ALREADY CHOSEN word and lost -0.086 on three seeds: moving there
costs the question its own context. This is the other thing entirely - the pin is chosen by the
CONSTRUCTION OF THE WINDOW, with the hole out of the key and no filler read:

    movement   dumb - places near on the tape, or random. NOT the filler walk.
    ranking    address similarity: the words of the left half and of the right half, each weighted
               by 1 / how many places carry that word in that half. A count, and nothing that
               stands IN a hole enters it.
    offer      the fillers of the chosen pin, top-M by count, own values excluded
    reward     the truth is in that offer

    STAY         the standing arm - the walk from the question's own place. What you get by not moving.
    RANDOM PIN   a pin drawn from the same dumb pool
    ADDRESS PIN  the pin the frame similarity picks

  GATE   address - random > 0.05 AND address - stay > 0.05.
         Beating only STAY would mean "understanding the frame" = sitting at home, which is 393.
         Losing to RANDOM would mean there is no construction in the window at all.

  WHAT THIS DOES NOT FIX, said before the run: the word still comes out of a BAG OF FILLERS. What
  changes is only WHICH bag, chosen before the word is known. 390 already priced the address as a
  SOURCE of candidates - half_only 0.041 against a 0.05 bar with new_share 0.87 - so the material
  the address reaches is thin; here it is used as a STEERING rule instead, which is a different
  use of the same relation and has to clear its own bar.

    python _audit411_frame.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _audit390_address as A


def half_index(T):
    """word -> how many places carry it in the left half, and the same for the right."""
    L, R = Counter(), Counter()
    for _w, left, right in T["addrs"]:
        for x in set(left):
            L[x] += 1
        for x in set(right):
            R[x] += 1
    return L, R


def frame_score(T, pid, j, L, R):
    """How alike are the two windows - counts only, the hole never consulted."""
    _w1, l1, r1 = T["addrs"][pid]
    _w2, l2, r2 = T["addrs"][j]
    s = 0.0
    for x in set(l1) & set(l2):
        s += 1.0 / max(1, L[x])
    for x in set(r1) & set(r2):
        s += 1.0 / max(1, R[x])
    return s


def offer_of(T, j, own, topm):
    return [v for v, _c in T["prof"][j].most_common() if v not in own][:topm]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--pool", type=int, default=32, help="dumb candidate pins: half random, half "
                                                        "adjacent by tape order")
    ap.add_argument("--max-questions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default="results/_stage411_frame.json")
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
    L, R = half_index(T)
    toks, owner, place_of = T["toks"], T["owner"], T["place_of"]
    npl = len(T["places"])
    qs = [s for ps in T["places"] for s in ps]
    rng.shuffle(qs)
    c = Counter()
    for s in qs:
        if c["n"] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in T["places"][pid] if x != s}
        if not own or truth in own:
            continue
        c["n"] += 1
        drop = set(T["on_line"][owner[s]])
        drop.discard(pid)
        # STAY: the standing arm, the walk from the question's own place
        qprof = Counter(toks[x] for x in T["places"][pid] if x != s)
        walked = A.walk_order(T, pid, qprof, args.places, drop)
        c["stay"] += int(truth in A.fillers_of(T, walked, own)[:args.topm])
        # THE DUMB POOL: half random, half adjacent by tape order. No filler is consulted.
        pool = set()
        for _k in range(args.pool // 2):
            pool.add(rng.randrange(npl))
        for d in range(1, args.pool // 4 + 1):
            pool.add((pid + d) % npl)
            pool.add((pid - d) % npl)
        pool = [j for j in pool if j != pid and j not in drop]
        if not pool:
            continue
        c["pooled"] += len(pool)
        j_rand = pool[rng.randrange(len(pool))]
        c["rand"] += int(truth in offer_of(T, j_rand, own, args.topm))
        j_addr = max(pool, key=lambda j: (frame_score(T, pid, j, L, R), -j))
        c["addr"] += int(truth in offer_of(T, j_addr, own, args.topm))
        c["addr_score"] += frame_score(T, pid, j_addr, L, R)
        c["addr_is_zero"] += int(frame_score(T, pid, j_addr, L, R) <= 0.0)
    n = max(1, c["n"])
    rep = {"seed": args.seed, "n": c["n"], "places": npl, "pool": c["pooled"] / n,
           "stay": c["stay"] / n, "random": c["rand"] / n, "addr": c["addr"] / n,
           "addr_score": c["addr_score"] / n, "addr_zero": c["addr_is_zero"] / n}
    rep["addr_minus_random"] = rep["addr"] - rep["random"]
    rep["addr_minus_stay"] = rep["addr"] - rep["stay"]
    print(f"{npl} places, {c['n']} questions, pool {rep['pool']:.1f} pins")
    print(f"VOID CHECK  the chosen pin scores ZERO on {rep['addr_zero']:.4f} of questions   "
          f"mean score {rep['addr_score']:.4f}  <- read first: at 1.0 the ranking never had "
          f"anything to rank")
    print(f"HIT@8       stay {rep['stay']:.4f}   random pin {rep['random']:.4f}   "
          f"address pin {rep['addr']:.4f}")
    print(f"            addr-random {rep['addr_minus_random']:+.4f}   "
          f"addr-stay {rep['addr_minus_stay']:+.4f}")
    void = rep["addr_zero"] > 0.95
    gate = rep["addr_minus_random"] > 0.05 and rep["addr_minus_stay"] > 0.05
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print("\n" + ("VOID: the frame ranking has nothing to rank - the chosen pin shares no window "
                 "word with the question on almost every question." if void else
                 ("THE FRAME CARRIES: a pin chosen by the construction of the window, before the "
                  "word is known, beats both a random pin and staying home." if gate else
                  "THE FRAME DOES NOT CARRY: " +
                  ("it does not beat a random pin - there is no construction in the window. "
                   if rep["addr_minus_random"] <= 0.05 else "") +
                  ("it does not beat STAY - understanding the frame would mean sitting at home, "
                   "which is 393 and 34.3's law again. " if rep["addr_minus_stay"] <= 0.05
                   else ""))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
