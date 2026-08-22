"""417h: HONEST DENSE PIN. Joint window, not w=1 OR."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _audit390_address as A

OUT = Path("results/_stage417h_densepin.json")


def place_bags(T):
    bags = []
    n = len(T["toks"])
    for slots in T["places"]:
        lines = {T["owner"][s] for s in slots}
        b = {T["toks"][i] for i in range(n) if T["owner"][i] in lines}
        bags.append(b)
    return bags


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
    seen, keys = set(), []
    for i, own in enumerate(T["owner"]):
        if own != li or i == s:
            continue
        v = T["toks"][i]
        if v == hole or v in seen:
            continue
        seen.add(v)
        keys.append(v)
    return keys


def retrieve_joint(T, bags, keys, token, qpid, drop, cap, joint):
    keyset = set(keys)
    scored = []
    for j, b in enumerate(bags):
        if j == qpid or j in drop:
            continue
        ov = len((b - {token}) & keyset)
        if ov >= joint:
            scored.append((ov, -j, j))
    scored.sort(reverse=True)
    return [j for _ov, _nj, j in scored[:cap]]


def dense_labels(cands, hits):
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


def step_of(T, bags, s, cap, joint, min_keys):
    qpid = T["place_of"].get(s)
    if qpid is None:
        return None
    token = T["toks"][s]
    keys = window_keys(T, s)
    if len(keys) < min_keys:
        return {"thin": True}
    drop = set(T["on_line"][T["owner"][s]])
    drop.discard(qpid)
    cands = retrieve_joint(T, bags, keys, token, qpid, drop, cap, joint)
    hits = [j for j in cands if token in fillers_place(T, j)]
    y = dense_labels(cands, hits)
    return {
        "thin": False, "s": s, "token": token, "qpid": qpid, "keys": keys,
        "cands": cands, "hits": hits, "y": y,
        "refuse": int(not hits), "df": T["freq"].get(token, 1),
    }


def measure(T, args, rng):
    bags = place_bags(T)
    slots = [s for s in T["place_of"]]
    rng.shuffle(slots)
    n = thin = ora = rnd = ref1 = n1 = refm = nm = live = 0
    for s in slots:
        if n >= args.max_q:
            break
        st = step_of(T, bags, s, args.cap, args.joint, args.min_keys)
        if st is None:
            continue
        if st["thin"]:
            thin += 1
            continue
        n += 1
        token, cands, hits = st["token"], st["cands"], st["hits"]
        live += int(not st["refuse"])
        k = len(cands) + 1
        if hits:
            ora += int(token in fillers_place(T, hits[0]))
        pick = rng.randrange(max(k, 1))
        if cands and pick < len(cands):
            rnd += int(token in fillers_place(T, cands[pick]))
        if st["df"] <= 1:
            n1 += 1
            ref1 += st["refuse"]
        else:
            nm += 1
            refm += st["refuse"]
    if n == 0:
        return {"n": 0, "thin": thin, "live": 0.0, "oracle": 0.0, "random": 0.0,
                "oracle_minus_random": 0.0, "refuse_df1": None, "refuse_df2": None,
                "n_df1": 0, "n_df2": 0, "working_cells": 0}
    return {
        "n": n, "thin": thin, "live": live / n,
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
    ap.add_argument("--joint", type=int, default=2)
    ap.add_argument("--min-keys", type=int, default=4)
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
    rep["seed"] = args.seed
    void = rep["n"] == 0 or rep["live"] <= 0.05
    gate = (not void) and rep["oracle_minus_random"] > 0.05
    r1, r2 = rep["refuse_df1"], rep["refuse_df2"]
    refuse_ok = (r1 is not None and r2 is not None and r1 > r2)
    rep["void"], rep["gate"], rep["refuse_ok"] = bool(void), bool(gate), bool(refuse_ok)
    print(f"{rep['n']} steps   thin-skipped {rep['thin']}   live {rep['live']:.4f}   "
          f"joint {args.joint} min_keys {args.min_keys}")
    print(f"ORACLE   {rep['oracle']:.4f}   RANDOM {rep['random']:.4f}   "
          f"Δ {rep['oracle_minus_random']:+.4f}")
    print(f"REFUSE   df1 {r1}   df>=2 {r2}   {'PASS' if refuse_ok else 'FAIL'}")
    if void:
        print("\nVOID: joint window almost never reaches the stream token. "
              "417's one-word OR is not this. Do not train Phi.")
    elif gate:
        print("\nHONEST DENSE PIN HAS A TEACHER: overlap>=2 beats random. "
              "CE over pins is not a coin. Phi is not in this file.")
    else:
        print("\nHONEST DENSE PIN IS A COIN. Do not train Phi on these labels.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
