"""412: DUMB STEP + PIN BY ADDRESS. Not a search, not 410.

The walk is sequential neighbours on the tape. The mind picks a pin by WINDOW
construction (left/right handles), never by shared fillers. The tape utters the
word. Stay = this place's own other fillers (recall). Random = a neighbour from
the same dumb set.

    addr score   same left handle, same right handle, Jaccard of context tokens.
                 The hole's filler `w` is not in the key (390).
    offer        fillers of the pinned place, asking-place own excluded.

  VOID   share of holes with a neighbour whose addr score > 0  <= 0.05
  GATE   addr − random > 0.05  AND  addr − stay > 0.05, 3 seeds.

    python _check412_addrpin.py
    python _audit412_addrpin.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _audit390_address as A

OUT = Path("results/_stage412_addrpin.json")


def own_of(T, s):
    pid = T["place_of"].get(s)
    if pid is None:
        return set()
    return {T["toks"][x] for x in T["places"][pid] if x != s}


def fillers_place(T, pid, own, hide=None):
    out, seen = [], set(own)
    for x in T["places"][pid]:
        if x == hide:
            continue
        v = T["toks"][x]
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def addr_score(T, pid, j):
    """Construction overlap. `w` of either place is not used."""
    if pid == j:
        return -1.0
    _w1, L1, R1 = T["addrs"][pid]
    _w2, L2, R2 = T["addrs"][j]
    s = 0.0
    if L1 == L2:
        s += 1.0
    if R1 == R2:
        s += 1.0
    b1, b2 = set(L1) | set(R1), set(L2) | set(R2)
    if b1 and b2:
        s += len(b1 & b2) / len(b1 | b2)
    return s


def nearby(T, s, radius, cap):
    """Dumb candidates: other places whose a slot sits within `radius` tokens."""
    qpid = T["place_of"].get(s)
    if qpid is None:
        return []
    n = len(T["toks"])
    lo, hi = max(0, s - radius), min(n, s + radius + 1)
    seen, out = {qpid}, []
    for i in range(lo, hi):
        j = T["place_of"].get(i)
        if j is None or j in seen:
            continue
        seen.add(j)
        out.append(j)
        if len(out) >= cap:
            break
    return out


def pick_addr(T, pid, cands):
    if not cands:
        return None
    return max(cands, key=lambda j: (addr_score(T, pid, j), -j))


def hit_offer(offer, truth, topm):
    return int(truth in offer[:topm])


def measure(T, args, rng):
    hs = [s for s in T["place_of"]]
    rng.shuffle(hs)
    n = scored = stay_h = rnd_h = addr_h = 0
    for s in hs:
        if n >= args.max_q:
            break
        qpid = T["place_of"][s]
        cands = nearby(T, s, args.radius, args.cand)
        if not cands:
            continue
        n += 1
        truth = T["toks"][s]
        own = own_of(T, s)
        stay_h += hit_offer(fillers_place(T, qpid, own, hide=s), truth, args.topm)
        jr = cands[rng.randrange(len(cands))]
        rnd_h += hit_offer(fillers_place(T, jr, own), truth, args.topm)
        ja = pick_addr(T, qpid, cands)
        addr_h += hit_offer(fillers_place(T, ja, own), truth, args.topm)
        scored += int(max(addr_score(T, qpid, j) for j in cands) > 0)
    if n == 0:
        return None
    stay, rnd, addr = stay_h / n, rnd_h / n, addr_h / n
    return {
        "n": n, "stay": stay, "random": rnd, "addr": addr,
        "addr_minus_random": addr - rnd, "addr_minus_stay": addr - stay,
        "scored": scored / n, "working_cells": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--radius", type=int, default=80)
    ap.add_argument("--cand", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--max-q", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default=str(OUT))
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
    rep = measure(T, args, rng)
    if rep is None:
        print("no questions")
        return 1
    rep["seed"] = args.seed
    void = rep["scored"] <= 0.05
    gate = (not void) and rep["addr_minus_random"] > 0.05 and rep["addr_minus_stay"] > 0.05
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"{rep['n']} holes   neighbours with construction {rep['scored']:.4f}   "
          f"working cells {rep['working_cells']}")
    print(f"STAY     {rep['stay']:.4f}   (this place, recall)")
    print(f"RANDOM   {rep['random']:.4f}   (dumb neighbour)")
    print(f"ADDR     {rep['addr']:.4f}   vs random {rep['addr_minus_random']:+.4f}   "
          f"vs stay {rep['addr_minus_stay']:+.4f}")
    if void:
        print("\nVOID: almost no neighbour shares a window handle. Nothing to pin by address.")
    elif gate:
        print("\nADDRESS PAYS: a dumb step plus pin-by-window beats both sitting at home "
              "and a random neighbour.")
    elif rep["addr_minus_stay"] <= 0.05:
        print("\nSIT AT HOME: address does not beat stay. Understanding the frame is recall.")
    else:
        print("\nADDRESS DOES NOT PAY the double gate. Do not train Phi on this pin.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
