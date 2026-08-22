"""413: WINDOW WORDS RETRIEVE. Example-fit list from THIS sentence, words on the tape."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _audit390_address as A

OUT = Path("results/_stage413_winpin.json")


def by_ctx_of(T):
    """token → places that have it in LEFT or RIGHT, never as filler w."""
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


def window_keys(T, s):
    """Visible words of this line except the hole. Rarest first."""
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


def offer_of(T, pids, own):
    out, seen = [], set(own)
    for pid in pids:
        for v in fillers_place(T, pid, seen):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def measure(T, args, rng):
    ix = by_ctx_of(T)
    hs = [s for s in T["place_of"]]
    rng.shuffle(hs)
    n = stay_h = rnd_h = win_h = nonempty = 0
    for s in hs:
        if n >= args.max_q:
            break
        qpid = T["place_of"][s]
        keys = window_keys(T, s)
        if not keys:
            continue
        drop = set(T["on_line"][T["owner"][s]])
        drop.discard(qpid)
        own = own_of(T, s)
        truth = T["toks"][s]
        stay = fillers_place(T, qpid, own, hide=s)
        win_p = retrieve(T, ix, keys, qpid, drop, args.cap)
        win = offer_of(T, win_p, own)
        rk = keys[rng.randrange(len(keys))]
        rnd_p = retrieve(T, ix, [rk], qpid, drop, args.cap)
        rnd = offer_of(T, rnd_p, own)
        n += 1
        stay_h += int(truth in stay[:args.topm])
        rnd_h += int(truth in rnd[:args.topm])
        win_h += int(truth in win[:args.topm])
        nonempty += int(len(win) > 0)
    if n == 0:
        return None
    stay, rnd, win = stay_h / n, rnd_h / n, win_h / n
    return {
        "n": n, "stay": stay, "random": rnd, "window": win,
        "window_minus_stay": win - stay, "window_minus_random": win - rnd,
        "nonempty": nonempty / n, "working_cells": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--cap", type=int, default=8)
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
    void = rep["nonempty"] <= 0.05
    gate = (not void) and rep["window_minus_stay"] > 0.05 and rep["window_minus_random"] > 0.05
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"{rep['n']} holes   nonempty offer {rep['nonempty']:.4f}   working cells "
          f"{rep['working_cells']}")
    print(f"STAY     {rep['stay']:.4f}")
    print(f"RANDOM   {rep['random']:.4f}   (one window word)")
    print(f"WINDOW   {rep['window']:.4f}   vs stay {rep['window_minus_stay']:+.4f}   "
          f"vs random {rep['window_minus_random']:+.4f}")
    if void:
        print("\nVOID: window words almost never retrieve a filler. Nothing to try on.")
    elif gate:
        print("\nWINDOW PAYS: visible neighbours retrieve what stay and a single random "
              "window-word do not.")
    elif rep["window_minus_stay"] <= 0.05:
        print("\nSIT AT HOME: the window list does not beat recall. Not the GPT piece.")
    else:
        print("\nWINDOW DOES NOT PAY the double gate. Do not train Phi on this retrieval.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
