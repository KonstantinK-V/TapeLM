"""417: DENSE PIN ON THE STREAM. CE over places+REFUSE, not 0/1, not vocab.

Teacher ceiling only — Phi is not in this file. No 289 hole in the loss.

  VOID  live (teacher not refuse) <= 0.05
  GATE  oracle - random > 0.05
  REFUSE  df1 > df>=2
  dense_labels  length |cands|+1, sums to 1; target for future CE, not a vocab.

    python _check417_densepin.py
    python _audit417_densepin.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _audit390_address as A

OUT = Path("results/_stage417_densepin.json")


def by_ctx_of(T):
    ix = defaultdict(list)
    for pid, (_w, L, R) in enumerate(T["addrs"]):
        for tok in set(L) | set(R):
            ix[tok].append(pid)
    return ix


def fillers_place(T, pid, hide=None):
    out, seen = [], set()
    for x in T["places"][pid]:
        if x == hide:
            continue
        v = T["toks"][x]
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


def dense_labels(cands, hits):
    """Length |cands|+1 (last = REFUSE). Sums to 1."""
    n = len(cands) + 1
    y = [0.0] * n
    if hits:
        mass = 1.0 / len(hits)
        index = {p: i for i, p in enumerate(cands)}
        for p in hits:
            y[index[p]] = mass
    else:
        y[-1] = 1.0
    return y


def step_of(T, ix, s, cap):
    qpid = T["place_of"].get(s)
    if qpid is None:
        return None
    token = T["toks"][s]
    keys = window_keys(T, s)
    drop = set(T["on_line"][T["owner"][s]])
    drop.discard(qpid)
    cands = retrieve(T, ix, keys, qpid, drop, cap)
    hits = [j for j in cands if token in fillers_place(T, j)]
    y = dense_labels(cands, hits)
    return {
        "s": s, "token": token, "qpid": qpid, "keys": keys,
        "cands": cands, "hits": hits, "y": y,
        "refuse": int(not hits), "df": T["freq"].get(token, 1),
    }


def measure(T, args, rng):
    ix = by_ctx_of(T)
    slots = [s for s in T["place_of"]]
    rng.shuffle(slots)
    n = ora = rnd = ref1 = n1 = refm = nm = live = 0
    for s in slots:
        if n >= args.max_q:
            break
        st = step_of(T, ix, s, args.cap)
        if st is None:
            continue
        n += 1
        token, cands, hits = st["token"], st["cands"], st["hits"]
        live += int(not st["refuse"])
        k = len(cands) + 1
        if hits:
            ora += int(token in fillers_place(T, hits[0]))
        pick = rng.randrange(k)
        if pick < len(cands):
            rnd += int(token in fillers_place(T, cands[pick]))
        if st["df"] <= 1:
            n1 += 1
            ref1 += st["refuse"]
        else:
            nm += 1
            refm += st["refuse"]
    if n == 0:
        return None
    return {
        "n": n, "live": live / n,
        "oracle": ora / n, "random": rnd / n,
        "oracle_minus_random": (ora - rnd) / n,
        "refuse_df1": (ref1 / n1) if n1 else None,
        "refuse_df2": (refm / nm) if nm else None,
        "n_df1": n1, "n_df2": nm, "working_cells": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--min-line", type=int, default=80)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--max-q", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

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
        print("no steps")
        return 1
    rep["seed"] = args.seed
    void = rep["live"] <= 0.05
    gate = (not void) and rep["oracle_minus_random"] > 0.05
    r1, r2 = rep["refuse_df1"], rep["refuse_df2"]
    refuse_ok = (r1 is not None and r2 is not None and r1 > r2)
    rep["void"], rep["gate"], rep["refuse_ok"] = bool(void), bool(gate), bool(refuse_ok)
    print(f"{rep['n']} steps   live (non-refuse teacher) {rep['live']:.4f}")
    print(f"ORACLE   {rep['oracle']:.4f}   RANDOM {rep['random']:.4f}   "
          f"Δ {rep['oracle_minus_random']:+.4f}")
    print(f"REFUSE   df1 {r1}   df>=2 {r2}   "
          f"{'PASS' if refuse_ok else 'FAIL'}")
    if void:
        print("\nVOID: window almost never reaches the stream token. No pin to learn.")
    elif gate:
        print("\nDENSE PIN HAS A TEACHER: oracle over window-places beats random. "
              "CE over pins is not a coin. Phi is not in this file.")
    else:
        print("\nDENSE PIN IS A COIN on this tape. Do not train Phi on these labels.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
