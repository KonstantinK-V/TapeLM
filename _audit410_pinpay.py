"""410: DOES THE PIN PAY? Hop 2 from the supplier place vs from the question place.

409 put a register on a PLACE of the original tape. This file asks whether standing
there helps the NEXT hole. Torch-free, same leak discipline as 390/407.

    pin            supplier of THIS hole's truth (409) — an ORACLE of the pin, declared
    TODAY hop 2    walk starts at the question place (register never moved)
    PIN hop 2      walk starts at `here`; that place is prepended so standing can be read
    score          next hole in the window: truth in fillers_of, own excluded

  VOID   share(pin != question) <= 0.05 — nowhere to move, 409 is a no-op on this tape.
  GATE   pin_hit - question_hit > 0.05 on 3 seeds, over ALL next-holes (when pin==question
         the two walks coincide; they must not be dropped, or the gain is cherry-picked).

Size is not a gate (408). No new cell (406). The mind still does not see the string.

    python _check410_pinpay.py
    python _audit410_pinpay.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _audit390_address as A

OUT = Path("results/_stage410_pinpay.json")


def supplier(T, hide, qpid, said):
    """First place on the original tape that holds `said`, not the asking slot. 409."""
    found = []
    for pid in T["at_value"].get(said, ()):
        slots = [s for s in T["places"][pid] if s != hide]
        if any(T["toks"][s] == said for s in slots):
            found.append(pid)
    if not found:
        return None
    other = [p for p in found if p != qpid]
    return other[0] if other else found[0]


def walk_from(T, here, qprof, k, drop):
    """Stand at `here`: read it, then its neighbours. `here` in drop (same line as the
    next hole) is the 390 window artefact — then it is not prepended."""
    order = A.walk_order(T, here, qprof, k, drop)
    if here not in drop:
        order = [here] + [j for j in order if j != here]
    return order[:k]


def hit_from(T, here, s, k, topm):
    pid = T["place_of"].get(s)
    if pid is None or here is None:
        return 0
    toks, owner = T["toks"], T["owner"]
    slots = T["places"][pid]
    truth = toks[s]
    own = {toks[x] for x in slots if x != s}
    qprof = Counter(toks[x] for x in slots if x != s)
    drop = set(T["on_line"][owner[s]])
    drop.discard(pid)
    walked = walk_from(T, here, qprof, k, drop)
    return int(truth in A.fillers_of(T, walked, own)[:topm])


def holes_of(T):
    """Slots that sit in a place, in tape order — the window's questions."""
    return sorted(T["place_of"])


def pairs(T, args):
    hs = holes_of(T)
    rng = getattr(args, "_rng", None)
    if rng is not None:
        hs = list(hs)
        rng.shuffle(hs)
    out = []
    by_pos = holes_of(T)
    nxt = {by_pos[i]: by_pos[i + 1] for i in range(len(by_pos) - 1)}
    for s in hs:
        if len(out) >= args.max_q:
            break
        t = nxt.get(s)
        if t is None:
            continue
        qpid = T["place_of"][s]
        said = T["toks"][s]
        pin = supplier(T, s, qpid, said)
        if pin is None:
            pin = qpid
        out.append((s, t, qpid, pin))
    return out


def measure(T, args):
    rows = pairs(T, args)
    n = len(rows)
    if n == 0:
        return None
    hq = hp = moved = 0
    for s, t, qpid, pin in rows:
        hq += hit_from(T, qpid, t, args.places, args.topm)
        hp += hit_from(T, pin, t, args.places, args.topm)
        moved += int(pin != qpid)
    return {
        "n": n, "question_hit": hq / n, "pin_hit": hp / n,
        "gain": (hp - hq) / n, "moved": moved / n,
        "working_cells": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)
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
    args._rng = rng
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    if not T["places"]:
        print("no tape")
        return 1
    rep = measure(T, args)
    if rep is None:
        print("no pairs")
        return 1
    rep["seed"] = args.seed
    void = rep["moved"] <= 0.05
    gate = (not void) and rep["gain"] > 0.05
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"{rep['n']} next-holes   pin!=question {rep['moved']:.4f}   working cells "
          f"{rep['working_cells']}")
    print(f"TODAY hop2   {rep['question_hit']:.4f}")
    print(f"PIN   hop2   {rep['pin_hit']:.4f}   gain {rep['gain']:+.4f}")
    if void:
        print("\nVOID: the pin almost never leaves the question place. 409 is a no-op here.")
    elif gate:
        print("\nTHE PIN PAYS: standing at the supplier helps the next hole by more than the bar.")
    else:
        print("\nTHE PIN DOES NOT PAY: the register moves, the next hole does not care. "
              "Do not wire --pin into reach.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = {k: v for k, v in rep.items()}
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
