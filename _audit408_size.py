"""408: SIZE RIVAL for 407. Is the ceiling just 'speak at the biggest hubs'?

Same `value(place)` as 407. The rivals rank the SAME places by structural size instead of by
value, then ask whether speaking at the B biggest hubs reaches the same oracle:

    VALUE oracle   top-B by value           (must match 407)
    SIZE oracle    top-B by |slots|
    NFILL oracle   top-B by distinct fillers
    DEGREE oracle  top-B by filler-graph degree

  READ THIS, not another +0.55 headline:
      capture = VALUE gain - SIZE gain
      size_takes = (SIZE gain > 0.05) AND (capture < 0.05)
      -> the ceiling is a hub prior; do not write the loop.

    python _audit408_size.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _audit390_address as A

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage408_size.json")


def place_rows(T, args, rng):
    toks, owner = T["toks"], T["owner"]
    ids = list(range(len(T["places"])))
    rng.shuffle(ids)
    rows = []
    for pid in ids[: args.max_places]:
        slots = T["places"][pid]
        if len(slots) < 2:
            continue
        ok = 0
        for s in slots:
            truth = toks[s]
            own = {toks[x] for x in slots if x != s}
            qprof = Counter(toks[x] for x in slots if x != s)
            drop = set(T["on_line"][owner[s]])
            drop.discard(pid)
            walked = A.walk_order(T, pid, qprof, args.places, drop)
            ok += int(truth in A.fillers_of(T, walked, own)[: args.topm])
        prof = T["prof"][pid]
        nbrs = set()
        for v in prof:
            for j in T["at_value"][v]:
                if j != pid:
                    nbrs.add(j)
        rows.append({"pid": pid, "value": ok / len(slots), "size": len(slots),
                     "nfill": len(prof), "degree": len(nbrs)})
    return rows


def oracle_of(rows, key, B):
    ranked = sorted(rows, key=lambda r: (-r[key], r["pid"]))
    return sum(r["value"] for r in ranked[:B]) / B


def const_value(rows, T):
    """Mean value on places whose every hole has the truth in own - must be ~0."""
    vals = []
    toks = T["toks"]
    for r in rows:
        slots = T["places"][r["pid"]]
        if all(toks[s] in {toks[x] for x in slots if x != s} for s in slots):
            vals.append(r["value"])
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--max-places", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0 : s0 + args.window_lines]
    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    if not T["places"]:
        print("no tape")
        return 1

    rows = place_rows(T, args, rng)
    n = len(rows)
    B = min(args.budget, n)
    rnd = sum(r["value"] for r in rows) / n
    ov = oracle_of(rows, "value", B)
    osz = oracle_of(rows, "size", B)
    onf = oracle_of(rows, "nfill", B)
    odeg = oracle_of(rows, "degree", B)
    v_gain = ov - rnd
    sz_gain = osz - rnd
    capture = v_gain - sz_gain
    size_takes = (sz_gain > 0.05) and (capture < 0.05)
    cv = const_value(rows, T)

    rep = {"seed": args.seed, "places": n, "budget": B, "random": rnd,
           "oracle_value": ov, "oracle_size": osz, "oracle_nfill": onf,
           "oracle_degree": odeg, "value_gain": v_gain, "size_gain": sz_gain,
           "nfill_gain": onf - rnd, "degree_gain": odeg - rnd,
           "capture": capture, "const_value": cv, "size_takes": bool(size_takes)}
    print(f"{n} places, budget {B}")
    print(f"RANDOM      {rnd:.4f}")
    print(f"VALUE       oracle {ov:.4f}   gain {v_gain:+.4f}   <- must match 407")
    print(f"SIZE        oracle {osz:.4f}   gain {sz_gain:+.4f}   capture {capture:+.4f}")
    print(f"NFILL       oracle {onf:.4f}   gain {onf - rnd:+.4f}")
    print(f"DEGREE      oracle {odeg:.4f}   gain {odeg - rnd:+.4f}")
    print(f"CONST hubs  mean value {cv:.4f}   <- must be ~0")
    if size_takes:
        print("\nSIZE TAKES THE CEILING: speak at the biggest hubs; do not write the loop.")
    elif v_gain > 0.05 and capture >= 0.05:
        print("\nVALUE BEATS SIZE: there is something to learn in where to stand beyond |slots|.")
    else:
        print("\nINCONCLUSIVE on the size gate; read capture and const_value.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
