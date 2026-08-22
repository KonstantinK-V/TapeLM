"""421: ORDERED-MATCH CEILING. No Phi.

Same 417h steps. Compare ordered L/R match vs bag overlap vs random on LIVE pins only.
If ordered does not beat bag, STOP — do not feed order to Phi.

  bag      |(bag - {token}) ∩ keys|   (= 417h retrieve_joint ov)
  ordered  side-matched mass + pairwise order agree on W_L↔L, W_R↔R
  random   uniform among cands

  VOID   n_live==0 or teacher_live<=0.05 among non-thin
  GO     ordered_minus_bag > 0.05 AND ordered_minus_random > 0.05
  STOP   ordered_minus_bag <= 0.05

    python _check421_ordceil.py
    python _audit421_ordceil.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _audit390_address as A
import _audit417h_densepin as D417h

OUT = Path("results/_stage421_ordceil.json")


def window_sides(T, s):
    """First-seen unique tokens left/right of hole; hole token never enters."""
    li, hole = T["owner"][s], T["toks"][s]
    seen_l, seen_r = set(), set()
    W_L, W_R = [], []
    for i, own in enumerate(T["owner"]):
        if own != li or i == s:
            continue
        v = T["toks"][i]
        if v == hole:
            continue
        if i < s:
            if v not in seen_l:
                seen_l.add(v)
                W_L.append(v)
        else:
            if v not in seen_r:
                seen_r.add(v)
                W_R.append(v)
    return W_L, W_R


def bag_score(bags, keys, token, j):
    return len((bags[j] - {token}) & set(keys))


def order_agree(win_seq, place_seq, shared):
    """Fraction of pairs in shared with matching relative order. Shared empty → 0."""
    shared = list(shared)
    if len(shared) < 2:
        return 0.0
    pos_w = {v: i for i, v in enumerate(win_seq)}
    pos_p = {v: i for i, v in enumerate(place_seq)}
    agree = total = 0
    for a in range(len(shared)):
        for b in range(a + 1, len(shared)):
            u, v = shared[a], shared[b]
            if u not in pos_w or v not in pos_w or u not in pos_p or v not in pos_p:
                continue
            total += 1
            same = (pos_w[u] < pos_w[v]) == (pos_p[u] < pos_p[v])
            agree += int(same)
    return agree / max(1, total)


def ordered_score(T, bags, keys, token, s, j):
    W_L, W_R = window_sides(T, s)
    _w, L, R = T["addrs"][j]
    L = tuple(x for x in L if x != token)
    R = tuple(x for x in R if x != token)
    set_L, set_R = set(L), set(R)
    sh_L = set(W_L) & set_L
    sh_R = set(W_R) & set_R
    n_side = len(sh_L) + len(sh_R)
    agree_L = order_agree(W_L, L, sh_L)
    agree_R = order_agree(W_R, R, sh_R)
    return float(n_side + agree_L + agree_R)


def pick_argmax(cands, score_fn):
    best_j, best_sc = None, None
    for j in cands:
        sc = score_fn(j)
        if best_sc is None or sc > best_sc or (sc == best_sc and j < best_j):
            best_sc, best_j = sc, j
    return best_j


def measure(T, bags, args, rng):
    slots = [s for s in T["place_of"]]
    rng.shuffle(slots)
    n = thin = live_teacher = 0
    n_live = bag_h = ord_h = rnd_h = 0
    n_refuse = 0
    margin_bag = margin_ord = n_margin = 0.0
    for s in slots:
        if n >= args.max_q:
            break
        st = D417h.step_of(T, bags, s, args.cap, args.joint, args.min_keys)
        if st is None:
            continue
        if st.get("thin"):
            thin += 1
            continue
        n += 1
        live_teacher += int(not st["refuse"])
        if st["refuse"] or not st["hits"] or not st["cands"]:
            n_refuse += 1
            continue
        n_live += 1
        token, cands, hits, keys = st["token"], st["cands"], st["hits"], st["keys"]
        hit_set = set(hits)

        def sc_bag(j):
            return bag_score(bags, keys, token, j)

        def sc_ord(j):
            return ordered_score(T, bags, keys, token, s, j)

        jb = pick_argmax(cands, sc_bag)
        jo = pick_argmax(cands, sc_ord)
        jr = cands[rng.randrange(len(cands))]
        bag_h += int(jb in hit_set)
        ord_h += int(jo in hit_set)
        rnd_h += int(jr in hit_set)

        non = [j for j in cands if j not in hit_set]
        if non:
            mb = sum(sc_bag(j) for j in hits) / len(hits) - sum(sc_bag(j) for j in non) / len(non)
            mo = sum(sc_ord(j) for j in hits) / len(hits) - sum(sc_ord(j) for j in non) / len(non)
            margin_bag += mb
            margin_ord += mo
            n_margin += 1

    teacher_live = live_teacher / max(1, n)
    if n_live == 0:
        return {
            "n": n, "thin": thin, "n_live": 0, "n_refuse_steps": n_refuse,
            "teacher_live": teacher_live,
            "bag_live": 0.0, "ordered_live": 0.0, "random_live": 0.0,
            "ordered_minus_bag": 0.0, "ordered_minus_random": 0.0,
            "margin_bag": None, "margin_ordered": None,
        }
    bag_live = bag_h / n_live
    ordered_live = ord_h / n_live
    random_live = rnd_h / n_live
    return {
        "n": n, "thin": thin, "n_live": n_live, "n_refuse_steps": n_refuse,
        "teacher_live": teacher_live,
        "bag_live": bag_live,
        "ordered_live": ordered_live,
        "random_live": random_live,
        "ordered_minus_bag": ordered_live - bag_live,
        "ordered_minus_random": ordered_live - random_live,
        "margin_bag": (margin_bag / n_margin) if n_margin else None,
        "margin_ordered": (margin_ord / n_margin) if n_margin else None,
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
    lines = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0 : s0 + args.window_lines]
    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    if not T["places"]:
        print("no tape")
        return 1
    bags = D417h.place_bags(T)
    rep = measure(T, bags, args, rng)
    rep["seed"] = args.seed
    rep["joint"] = args.joint
    rep["min_keys"] = args.min_keys
    void = rep["n_live"] == 0 or rep["teacher_live"] <= 0.05
    go = (not void) and rep["ordered_minus_bag"] > 0.05 and rep["ordered_minus_random"] > 0.05
    stop = (not void) and rep["ordered_minus_bag"] <= 0.05
    rep["void"] = bool(void)
    rep["go"] = bool(go)
    rep["stop"] = bool(stop)
    print(f"{rep['n']} steps   thin {rep['thin']}   n_live {rep['n_live']}   "
          f"teacher_live {rep['teacher_live']:.4f}")
    print(f"BAG      {rep['bag_live']:.4f}")
    print(f"ORDERED  {rep['ordered_live']:.4f}   vs bag {rep['ordered_minus_bag']:+.4f}   "
          f"vs random {rep['ordered_minus_random']:+.4f}")
    print(f"RANDOM   {rep['random_live']:.4f}")
    if rep["margin_bag"] is not None:
        print(f"MARGIN   bag {rep['margin_bag']:+.4f}   ordered {rep['margin_ordered']:+.4f}")
    if void:
        print("\nVOID: no live joint pins to score. Do not train Phi.")
    elif go:
        print("\nGO: ordered beats bag and random — structural signal for Phi feats.")
    elif stop:
        print("\nSTOP: ordered ≈ bag — order adds nothing. Do not feed order to Phi.")
    else:
        print("\nMIXED: ordered beats bag weakly or loses to random — read deltas.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
