"""422: ORDER TIE-BREAK. Does order break bag ties toward the teacher?

Same 417h live steps. No Phi. Bar 0.05 fixed (not moved for 1337).

  On each live step, bag scores over cands. Tie = ≥2 cands at max bag.
  Among the tied set: bag pick (argmin j), ordered argmax, random.
  Δ = ordered_tie_hit - bag_tie_hit  (teacher = hits)

  VOID   tie_rate <= 0.05 among live  → 421 was bag; stop order→Phi
  STOP   ties exist, Δ <= 0.05         → order does not break ties to teacher
  GO     ties exist, Δ > 0.05          → channel beyond bag; CE on sides later

  unique_agree  when bag unique: does ordered pick the same place? Record only, not gate.

    python _check422_ordertie.py
    python _audit422_ordertie.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _audit390_address as A
import _audit417h_densepin as D417h
import _audit421_ordceil as C421

OUT = Path("results/_stage422_ordertie.json")


def measure(T, bags, args, rng):
    slots = [s for s in T["place_of"]]
    rng.shuffle(slots)
    n = thin = live_teacher = 0
    n_live = n_tie = n_unique = 0
    bag_tie_h = ord_tie_h = rnd_tie_h = 0
    unique_agree = 0
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
            continue
        n_live += 1
        token, cands, hits, keys = st["token"], st["cands"], st["hits"], st["keys"]
        hit_set = set(hits)

        def sc_bag(j):
            return C421.bag_score(bags, keys, token, j)

        def sc_ord(j):
            return C421.ordered_score(T, bags, keys, token, s, j)

        scores = {j: sc_bag(j) for j in cands}
        mx = max(scores.values())
        tied = [j for j in cands if scores[j] == mx]
        if len(tied) == 1:
            n_unique += 1
            j_bag = tied[0]
            j_ord = C421.pick_argmax(cands, sc_ord)
            unique_agree += int(j_ord == j_bag)
            continue
        n_tie += 1
        j_bag = min(tied)  # bag cannot break; deterministic among tied
        j_ord = C421.pick_argmax(tied, sc_ord)
        j_rnd = tied[rng.randrange(len(tied))]
        bag_tie_h += int(j_bag in hit_set)
        ord_tie_h += int(j_ord in hit_set)
        rnd_tie_h += int(j_rnd in hit_set)

    teacher_live = live_teacher / max(1, n)
    tie_rate = n_tie / max(1, n_live)
    if n_tie == 0:
        bag_t = ord_t = rnd_t = dlt = 0.0
    else:
        bag_t = bag_tie_h / n_tie
        ord_t = ord_tie_h / n_tie
        rnd_t = rnd_tie_h / n_tie
        dlt = ord_t - bag_t
    return {
        "n": n, "thin": thin, "n_live": n_live, "n_tie": n_tie, "n_unique": n_unique,
        "teacher_live": teacher_live,
        "tie_rate": tie_rate,
        "bag_tie": bag_t, "ordered_tie": ord_t, "random_tie": rnd_t,
        "ordered_minus_bag_tie": dlt,
        "unique_agree": (unique_agree / n_unique) if n_unique else None,
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
    void = rep["n_live"] == 0 or rep["teacher_live"] <= 0.05 or rep["tie_rate"] <= 0.05
    go = (not void) and rep["n_tie"] > 0 and rep["ordered_minus_bag_tie"] > 0.05
    stop = (not void) and rep["n_tie"] > 0 and rep["ordered_minus_bag_tie"] <= 0.05
    # void from low tie_rate is the "421 was bag" reading
    void_ties = rep["n_live"] > 0 and rep["tie_rate"] <= 0.05
    rep["void"] = bool(void)
    rep["void_ties"] = bool(void_ties)
    rep["go"] = bool(go)
    rep["stop"] = bool(stop)
    print(f"{rep['n']} steps   n_live {rep['n_live']}   n_tie {rep['n_tie']}   "
          f"tie_rate {rep['tie_rate']:.4f}   teacher_live {rep['teacher_live']:.4f}")
    print(f"TIE bag {rep['bag_tie']:.4f}   ordered {rep['ordered_tie']:.4f}   "
          f"random {rep['random_tie']:.4f}   Δ {rep['ordered_minus_bag_tie']:+.4f}")
    ua = rep["unique_agree"]
    print(f"unique_agree {ua if ua is None else f'{ua:.4f}'}   (record, not gate)")
    if void_ties:
        print("\nVOID: almost no bag ties — 421 was the bag. Do not put order in Phi.")
    elif void:
        print("\nVOID: no live pins. Do not put order in Phi.")
    elif go:
        print("\nGO: order breaks bag ties toward teacher — CE on sides later, not 421 alone.")
    elif stop:
        print("\nSTOP: ties exist but order does not beat bag on them. Close order like 420.")
    else:
        print("\nMIXED — read Δ / tie_rate.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
