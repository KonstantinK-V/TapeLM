"""423: AND OF TWO WINDOW KEYS. Formation, not Phi. Not 390 composition."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _audit390_address as A
import _audit417h_densepin as H

OUT = Path("results/_stage423_keyand.json")


def eligible(T, qpid, drop):
    return {j for j in range(len(T["places"])) if j != qpid and j not in drop}


def places_for_key(bags, key, ok):
    return {j for j in ok if key in bags[j]}


def live_of(T, S, token):
    if not S:
        return None, 0
    hit = sum(1 for j in S if token in H.fillers_place(T, j))
    return hit / len(S), len(S)


def two_keys(T, keys):
    cand = [v for v in keys if T["freq"].get(v, 0) >= 2]
    cand.sort(key=lambda v: (T["freq"].get(v, 0), v))
    if len(cand) < 2:
        return None
    return cand[0], cand[1]


def bag_cands(T, bags, keys, token, ok, cap, joint):
    scored = []
    keyset = set(keys)
    for j in ok:
        ov = len((bags[j] - {token}) & keyset)
        if ov >= joint:
            scored.append((ov, -j, j))
    scored.sort(reverse=True)
    full = {j for _ov, _nj, j in scored}
    capped = {j for _ov, _nj, j in scored[:cap]}
    return full, capped


def measure(T, bags, args, rng):
    slots = [s for s in T["place_of"]]
    rng.shuffle(slots)
    n = used = 0
    sum_and = sum_a = sum_b = sum_sng = sum_bag = sum_rnd = 0.0
    n_and_sz = n_a_sz = n_b_sz = n_bag_sz = 0
    shrink_a = shrink_b = 0.0
    and_not_cap = n_and_places = 0
    n_empty_and = 0
    for s in slots:
        if n >= args.max_q:
            break
        st = H.step_of(T, bags, s, args.cap, args.joint, args.min_keys)
        if st is None or st.get("thin"):
            continue
        n += 1
        token, keys, qpid = st["token"], st["keys"], st["qpid"]
        pair = two_keys(T, keys)
        if pair is None:
            continue
        ka, kb = pair
        drop = set(T["on_line"][T["owner"][s]])
        ok = eligible(T, qpid, drop)
        if not ok:
            continue
        PA = places_for_key(bags, ka, ok)
        PB = places_for_key(bags, kb, ok)
        PAND = PA & PB
        bag_full, bag_cap = bag_cands(T, bags, keys, token, ok, args.cap, args.joint)
        used += 1
        if not PAND:
            n_empty_and += 1
        la, na = live_of(T, PA, token)
        lb, nb = live_of(T, PB, token)
        land, nand = live_of(T, PAND, token)
        lbag, nbag = live_of(T, bag_full, token)
        lrnd, _nr = live_of(T, ok, token)
        singles = [(la, na), (lb, nb)]
        lsng = max((x for x, sz in singles if x is not None), default=None)
        if land is not None:
            sum_and += land
        if la is not None:
            sum_a += la
            n_a_sz += na
        if lb is not None:
            sum_b += lb
            n_b_sz += nb
        if lsng is not None:
            sum_sng += lsng
        if lbag is not None:
            sum_bag += lbag
            n_bag_sz += nbag
        if lrnd is not None:
            sum_rnd += lrnd
        n_and_sz += nand
        if na:
            shrink_a += 1.0 - nand / na
        if nb:
            shrink_b += 1.0 - nand / nb
        n_and_places += nand
        and_not_cap += len(PAND - bag_cap)
    if used == 0:
        return None
    return {
        "n": n, "used": used, "empty_and": n_empty_and / used,
        "and_live": sum_and / used,
        "a_live": sum_a / used, "b_live": sum_b / used,
        "single_live": sum_sng / used,
        "bag_live": sum_bag / used, "random_live": sum_rnd / used,
        "and_minus_single": (sum_and - sum_sng) / used,
        "and_minus_bag": (sum_and - sum_bag) / used,
        "mean_and_size": n_and_sz / used, "mean_a_size": n_a_sz / used,
        "mean_b_size": n_b_sz / used, "mean_bag_size": n_bag_sz / used,
        "shrink_a": shrink_a / used, "shrink_b": shrink_b / used,
        "new_vs_A": 0.0,
        "and_not_in_bagcap": (and_not_cap / n_and_places) if n_and_places else 0.0,
        "working_cells": 0,
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
    bags = H.place_bags(T)
    rep = measure(T, bags, args, rng)
    if rep is None:
        print("no steps")
        return 1
    rep["seed"] = args.seed
    void = rep["used"] < 50 or (1.0 - rep["empty_and"]) <= 0.05
    gate = (not void) and rep["and_minus_single"] > 0.05
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"{rep['used']} paired steps  empty_AND {rep['empty_and']:.3f}  "
          f"mean |AND| {rep['mean_and_size']:.2f}  |A| {rep['mean_a_size']:.2f}  "
          f"|bag| {rep['mean_bag_size']:.2f}")
    print(f"LIVE   AND {rep['and_live']:.4f}  single {rep['single_live']:.4f}  "
          f"A {rep['a_live']:.4f}  B {rep['b_live']:.4f}")
    print(f"       bag≥2 {rep['bag_live']:.4f}  random {rep['random_live']:.4f}")
    print(f"AND−single {rep['and_minus_single']:+.4f}   AND−bag {rep['and_minus_bag']:+.4f}")
    print(f"DIAG   shrink A {rep['shrink_a']:.3f}  B {rep['shrink_b']:.3f}  "
          f"new_vs_A {rep['new_vs_A']:.3f} (0 by construction)  "
          f"AND not in bag-cap8 {rep['and_not_in_bagcap']:.3f}")
    if void:
        print("\nVOID: AND almost never fires. Two keys do not form a candidate-world.")
    elif gate:
        print("\nAND OF TWO KEY-TRACES IS PURER THAN THE BETTER SINGLE TRACE. "
              "Formation, not Phi. Not composition.")
    else:
        print("\nAND DOES NOT BEAT THE BETTER ONE-KEY TRACE. Intersection is not a new "
              "candidate-world — it is a thinner single. Stop before Phi.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
