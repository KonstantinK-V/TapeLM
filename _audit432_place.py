"""432: place-teacher. One hole. No pair, no Φ, no second tape."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage432_place.json")
WIKI = Path("data/_wikitext103_train.txt")
K = 8


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def vote(cands, weights):
    sc = Counter()
    for c, w in zip(cands, weights):
        if w > 0:
            sc[c] += w
    if not sc:
        return None
    return sc.most_common(1)[0][0]


def measure(lines, args, rng):
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        return None
    place, value, line, pos = [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            pos.append(i)
    n = len(place)
    slots_at = defaultdict(list)
    on_line = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
    places_on_line = {li: {place[s] for s in ss} for li, ss in on_line.items()}
    line_toks = defaultdict(list)
    for i, tok in enumerate(toks):
        line_toks[owner[i]].append(tok)
    idx = list(range(n))
    rng.shuffle(idx)
    idx = idx[: args.max_questions]
    k = args.k
    c = Counter()
    hits = {"maj": [], "line": [], "frame": [], "sum": []}
    mix_hits = {"maj": [], "line": [], "frame": [], "sum": []}
    for s in idx:
        c["n"] += 1
        p, v, li = place[s], value[s], line[s]
        foreign = [t for t in slots_at[p] if t != s and line[t] != li][:k]
        if not foreign:
            continue
        c["foreign"] += 1
        maj = Counter(value[t] for t in foreign).most_common(1)[0][0]
        mixed = maj != v
        c["mixed"] += mixed
        nq = places_on_line[li] - {p}
        fq = set(line_toks[li]) - {v}
        w_line, w_frame, vals = [], [], []
        for t in foreign:
            nt = places_on_line[line[t]] - {p}
            ft = set(line_toks[line[t]]) - {value[t]}
            w_line.append(jaccard(nq, nt))
            w_frame.append(jaccard(fq, ft))
            vals.append(value[t])
        c["struct_line"] += any(w > 0 for w in w_line)
        c["struct_frame"] += any(w > 0 for w in w_frame)
        pick = {
            "maj": maj,
            "line": vote(vals, w_line) or maj,
            "frame": vote(vals, w_frame) or maj,
            "sum": vote(vals, [a + b for a, b in zip(w_line, w_frame)]) or maj,
        }
        for name, pv in pick.items():
            hits[name].append(int(pv == v))
            if mixed:
                mix_hits[name].append(int(pv == v))

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    return {
        "slots": n, "questions": c["n"],
        "foreign_nonempty": c["foreign"] / max(c["n"], 1),
        "n_foreign": c["foreign"],
        "mixed": c["mixed"] / max(c["foreign"], 1),
        "n_mixed": c["mixed"],
        "struct_line": c["struct_line"] / max(c["foreign"], 1),
        "struct_frame": c["struct_frame"] / max(c["foreign"], 1),
        "hit_maj": mean(hits["maj"]),
        "hit_line": mean(hits["line"]),
        "hit_frame": mean(hits["frame"]),
        "hit_sum": mean(hits["sum"]),
        "mix_maj": mean(mix_hits["maj"]),
        "mix_line": mean(mix_hits["line"]),
        "mix_frame": mean(mix_hits["frame"]),
        "mix_sum": mean(mix_hits["sum"]),
        "d_line": mean(mix_hits["line"]) - mean(mix_hits["maj"]),
        "d_frame": mean(mix_hits["frame"]) - mean(mix_hits["maj"]),
        "d_sum": mean(mix_hits["sum"]) - mean(mix_hits["maj"]),
        "k": k, "working_cells": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--pairs-per-line", type=int, default=2)
    ap.add_argument("--max-questions", type=int, default=20000)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    rep = measure(lines, args, rng)
    if rep is None:
        print("no tape")
        return 1
    rep["seed"] = args.seed
    best = max(rep["d_line"], rep["d_frame"], rep["d_sum"])
    void = (rep["foreign_nonempty"] <= 0.05) or (rep["mixed"] <= 0.05)
    gate = (not void) and (best > 0.05)
    rep["best_d"], rep["void"], rep["gate"] = best, bool(void), bool(gate)
    print(f"q {rep['questions']}  foreign {rep['foreign_nonempty']:.3f}  "
          f"mixed {rep['mixed']:.3f}  n_mixed {rep['n_mixed']}")
    print(f"struct  line {rep['struct_line']:.3f}  frame {rep['struct_frame']:.3f}")
    print(f"hit     maj {rep['hit_maj']:.3f}  line {rep['hit_line']:.3f}  "
          f"frame {rep['hit_frame']:.3f}  sum {rep['hit_sum']:.3f}")
    print(f"mixed   maj {rep['mix_maj']:.3f}  d_line {rep['d_line']:+.3f}  "
          f"d_frame {rep['d_frame']:+.3f}  d_sum {rep['d_sum']:+.3f}  best {best:+.3f}")
    print(f"VOID {rep['void']}  GATE {rep['gate']}")
    if void:
        if rep["foreign_nonempty"] <= 0.05:
            print("\nVOID: place does not repeat. Two tapes would be emptier.")
        else:
            print("\nVOID: majority already is the teacher. Catalog, not geometry.")
    elif gate:
        print("\nGO PLACE: structure beats majority on mixed. Then a second tape.")
    else:
        print("\nSTOP PLACE: mixed exists, LINE/FRAME do not beat majority. "
              "No pair-Phi, no second tape.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
