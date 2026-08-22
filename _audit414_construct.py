"""414: HONEST CONSTRUCTION. Parallel scenes on the tape, GPT-local as control.

What GPT would do on OUR corpus (w400 / this designed tape), not on the internet:
  stay   exact frame (left, right)
  left   same left handle
The construction: peti/basket 'tasty APPLES' and a NEW hole 'trees grow ___ they tasty'.

    python _check414_construct.py
    python _audit414_construct.py --designed
    python _audit414_construct.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _audit390_address as A

OUT = Path("results/_stage414_construct.json")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


DESIGNED = [
    "peti has tasty APPLES at home now" + _pad(0),
    "peti has tasty APPLES at home two" + _pad(1),
    "basket holds tasty APPLES today yes" + _pad(2),
    "basket holds tasty APPLES today yes" + _pad(3),
    "trees grow APPLES they tasty" + _pad(4),
    "trees grow ORANGES they tasty" + _pad(5),
]


def by_ctx_of(T):
    ix = defaultdict(list)
    for pid, (_w, L, R) in enumerate(T["addrs"]):
        for tok in set(L) | set(R):
            ix[tok].append(pid)
    return ix


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


def offer_of(T, pids, own):
    out, seen = [], set(own)
    for pid in pids:
        for v in fillers_place(T, pid, seen):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def window_keys(T, s):
    li, hole = T["owner"][s], T["toks"][s]
    freq = T["freq"]
    seen, keys = set(), []
    for i, own in enumerate(T["owner"]):
        if own != li or i == s:
            continue
        v = T["toks"][i]
        if v == hole or v in seen:
            continue
        seen.add(v)
        keys.append(v)
    keys.sort(key=lambda v: (freq.get(v, 0), v))
    return keys


def retrieve(T, ix, keys, qpid, drop, cap):
    seen, out = {qpid} | set(drop), []
    for v in keys:
        for j in ix.get(v, ()):
            if j not in seen:
                seen.add(j)
                out.append(j)
                if len(out) >= cap:
                    return out
    return out


def joint_pids(T, ix, keys, qpid, drop, cap):
    n = Counter()
    for v in keys:
        for j in ix.get(v, ()):
            if j != qpid and j not in drop:
                n[j] += 1
    return [j for j, c in n.most_common() if c >= 2][:cap]


def left_pids(T, s, qpid, drop, cap):
    _w, L, _R = T["addrs"][qpid]
    out = []
    for j in T["by_left"].get(L, ()):
        if j != qpid and j not in drop:
            out.append(j)
            if len(out) >= cap:
                break
    return out


def arms_for(T, ix, s, cap, topm):
    qpid = T["place_of"][s]
    own = own_of(T, s)
    truth = T["toks"][s]
    drop = set(T["on_line"][T["owner"][s]])
    drop.discard(qpid)
    keys = window_keys(T, s)
    stay = fillers_place(T, qpid, own, hide=s)
    left = offer_of(T, left_pids(T, s, qpid, drop, cap), own)
    win = offer_of(T, retrieve(T, ix, keys, qpid, drop, cap), own)
    jnt = offer_of(T, joint_pids(T, ix, keys, qpid, drop, cap), own)
    return {
        "stay": int(truth in stay[:topm]),
        "left": int(truth in left[:topm]),
        "window": int(truth in win[:topm]),
        "joint": int(truth in jnt[:topm]),
        "keys": keys, "nonempty_win": int(len(win) > 0),
        "truth": truth, "qpid": qpid,
    }


def measure(T, args, rng):
    ix = by_ctx_of(T)
    hs = [s for s in T["place_of"]]
    rng.shuffle(hs)
    n = 0
    acc = {k: 0 for k in ("stay", "left", "window", "joint")}
    nonempty = 0
    for s in hs:
        if n >= args.max_q:
            break
        r = arms_for(T, ix, s, args.cap, args.topm)
        n += 1
        for k in acc:
            acc[k] += r[k]
        nonempty += r["nonempty_win"]
    if n == 0:
        return None
    out = {k: acc[k] / n for k in acc}
    out.update({"n": n, "nonempty": nonempty / n, "working_cells": 0})
    out["window_minus_stay"] = out["window"] - out["stay"]
    out["window_minus_left"] = out["window"] - out["left"]
    return out


def designed_probe():
    T = A.build_tape(DESIGNED, 3, 1)
    ix = by_ctx_of(T)
    hide = next(s for s, t in enumerate(T["toks"])
                if t == "APPLES" and T["owner"][s] == 4)
    return T, hide, arms_for(T, ix, hide, cap=8, topm=8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designed", action="store_true")
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--min-line", type=int, default=80,
                    help="drop shorter lines; TinyStories needs ~20, wiki 80")
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--max-q", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.designed:
        T, hide, r = designed_probe()
        print("DESIGNED hole: trees grow ___ they tasty   truth APPLES")
        print(f"  GPT-local  stay {r['stay']}  left {r['left']}   "
              "(exact frame / previous word on THIS tape)")
        print(f"  construct  window {r['window']}  joint {r['joint']}")
        print(f"  keys {r['keys'][:8]}   tasty_in_keys {('tasty' in r['keys'])}")
        ok = r["window"] == 1 and r["stay"] == 0 and r["left"] == 0
        print("\n" + ("CONSTRUCTION HOLDS: window finds APPLES from peti/basket; "
                      "GPT-local on this tape does not."
                      if ok else
                      "CONSTRUCTION BROKEN ON DESIGNED: do not go to wiki."))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(out.read_text()) if out.exists() else {}
        prev["designed"] = {k: r[k] for k in
                            ("stay", "left", "window", "joint", "nonempty_win")}
        out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
        return 0 if ok else 1
    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= args.min_line]
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
    rep["corpus"] = Path(args.corpus).name
    rep["window_lines"] = len(lines)
    rep["min_line"] = args.min_line
    void = rep["nonempty"] <= 0.05
    gate = (not void) and rep["window_minus_stay"] > 0.05 and rep["window_minus_left"] > 0.05
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"tape {rep['corpus']}   window {rep['window_lines']} lines   "
          f"min_line {rep['min_line']}")
    print(f"{rep['n']} holes   nonempty {rep['nonempty']:.4f}")
    print(f"GPT-local  stay {rep['stay']:.4f}   left {rep['left']:.4f}")
    print(f"construct  window {rep['window']:.4f}   joint {rep['joint']:.4f}")
    print(f"           vs stay {rep['window_minus_stay']:+.4f}   "
          f"vs left {rep['window_minus_left']:+.4f}")
    if void:
        print("\nVOID: window offers nothing.")
    elif gate:
        print("\nCONSTRUCTION PAYS ON THIS TAPE: window beats GPT-local stay and left.")
    else:
        print("\nCONSTRUCTION DOES NOT PAY on this tape. Parallel scenes are not on the tape; "
              "GPT would only have them if they were in THESE lines, or in its weights.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
