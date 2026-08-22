"""409: THE PIN. The winner is a PLACE on the original tape, not a word and not a new cell.

Kostya: do not throw the winner away. The tape already is the dictionary. The mind holds WHERE
the answer lies and HOW it got there. Phi never sees the string.

    TODAY   after a pick, hop 2 still starts at the QUESTION place. The name was scored and
            discarded as a pointer. Knowledge is on the tape; the REGISTER was not set.
    PIN     here = the first walked place that holds `said`. Hop 2's walk starts there.
            No new slot, no copied value, corpus untouched.

Wrong pick pins a different place (OTHER's cluster). That is search without knowing the word.

    python _check409_pin.py
    python _audit409_pin.py
    python _audit409_pin.py --pin
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

OUT = Path("results/_stage409_pin.json")
CORPUS = [
    "the XARWIN team won the opening match of the season in a long padded line of text",
    "the XARWIN team lost the closing match of the season in a long padded line here",
    "also XARWIN here extra words for a second XARWIN place in a long padded line aa",
    "also XARWIN here extra words for a second XARWIN place in a long padded line bb",
    "the OTHER club played a different sport on a different field in a long padded line",
    "the OTHER club played another season on a different field in a long padded line zz",
]


def build():
    toks, owner = [], []
    for i, line in enumerate(CORPUS):
        ws = line.split()
        toks.extend(ws)
        owner.extend([i] * len(ws))
    n = len(toks)
    buckets = defaultdict(list)
    for s in range(n):
        left = toks[s - 1] if s > 0 and owner[s] == owner[s - 1] else None
        right = toks[s + 1] if s + 1 < n and owner[s] == owner[s + 1] else None
        buckets[(left, right)].append(s)
    places, place_of = [], {}
    for ps in buckets.values():
        if len(ps) < 2:
            continue
        pid = len(places)
        places.append(ps)
        for s in ps:
            place_of[s] = pid
    at_value = defaultdict(list)
    for pid, ps in enumerate(places):
        for s in ps:
            at_value[toks[s]].append(pid)
    return {"toks": toks, "owner": owner, "places": places, "place_of": place_of,
            "at_value": dict(at_value)}


def question(T):
    toks = T["toks"]
    hide = next(s for s, t in enumerate(toks)
                if t == "XARWIN" and T["owner"][s] == 1)
    qpid = T["place_of"][hide]
    return {"hide": hide, "qpid": qpid, "truth": toks[hide],
            "corpus_n": len(toks), "here": qpid, "said": None}


def supplier(T, q, said):
    """The place on the ORIGINAL tape that holds `said`, not the asking slot."""
    hide, qpid = q["hide"], q["qpid"]
    found = []
    for pid in T["at_value"].get(said, []):
        slots = [s for s in T["places"][pid] if s != hide]
        if any(T["toks"][s] == said for s in slots):
            found.append(pid)
    if not found:
        return None
    other = [p for p in found if p != qpid]
    return other[0] if other else found[0]


def apply_today(q, said):
    w = dict(q)
    w["said"] = said
    w["here"] = q["qpid"]          # register never moves
    return w


def apply_pin(T, q, said):
    w = dict(q)
    w["said"] = said
    pin = supplier(T, q, said)
    w["here"] = pin if pin is not None else q["qpid"]
    return w


def hop2_origin(q):
    return q["here"]


def measure(pin):
    T = build()
    q0 = question(T)
    rows = {}
    for said in (q0["truth"], "OTHER"):
        q = apply_pin(T, q0, said) if pin else apply_today(q0, said)
        rows[said] = {
            "corpus_untouched": len(T["toks"]) == q0["corpus_n"],
            "here": q["here"],
            "here_is_place": isinstance(q["here"], int),
            "here_is_string": isinstance(q["here"], str),
            "here_eq_question": q["here"] == q0["qpid"],
            "hop2_origin": hop2_origin(q),
            "working_cells": 0,
        }
    return {
        "pin": bool(pin),
        "truth": rows[q0["truth"]],
        "wrong": rows["OTHER"],
        "hop2_moved": rows[q0["truth"]]["here"] != q0["qpid"],
        "wrong_pin_differs": rows["OTHER"]["here"] != rows[q0["truth"]]["here"],
        "no_letter_in_register": (not rows[q0["truth"]]["here_is_string"]
                                  and not rows["OTHER"]["here_is_string"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rep = measure(args.pin)
    print(f"arm              {'PIN' if args.pin else 'TODAY (register stuck at question)'}")
    print(f"here is a place  {rep['truth']['here_is_place']}   letter in register "
          f"{rep['truth']['here_is_string']}")
    print(f"hop2 origin      {rep['truth']['hop2_origin']}   question place "
          f"{question(build())['qpid']}")
    print(f"hop2 moved       {rep['hop2_moved']}")
    print(f"wrong pin differs {rep['wrong_pin_differs']}")
    print(f"corpus untouched {rep['truth']['corpus_untouched']}   working cells "
          f"{rep['truth']['working_cells']}")
    if not args.pin:
        print("\nTODAY: the tape still holds the word. The mind does not hold the PLACE. "
              "Hop 2 starts at the question again — that is the missing context.")
    else:
        print("\nPIN: the mind holds a place id on the original tape. Hop 2 starts there. "
              "A wrong pick pins OTHER's cluster. No new cell, no letter in Phi.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev["pin" if args.pin else "today"] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
