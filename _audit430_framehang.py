"""430: hang of FRAME fingerprints, not line bags. No Phi."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
import _audit425_hang as H425

OUT = Path("results/_stage430_framehang.json")
WIKI = Path("data/_wikitext103_train.txt")
K = 8


def cosine(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def frame_bag(s, frame, va, vb):
    return {w for w in frame[s] if w != va and w != vb}


def hang_of(va, vb, a, b, by_val, line, frame, k):
    eA = H425.evidence(va, by_val, line, line[a], {a, b}, k)
    eB = H425.evidence(vb, by_val, line, line[a], {a, b}, k)
    if not eA or not eB:
        return 0.0
    acc = n = 0
    for i in eA:
        ai = frame_bag(i, frame, va, vb)
        for j in eB:
            if line[i] == line[j]:
                continue
            acc += cosine(ai, frame_bag(j, frame, va, vb))
            n += 1
    return acc / n if n else 0.0


def measure(lines, args, rng):
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        return None
    place, value, line, pos, frame = [], [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        fr = list(left) + list(right)
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            pos.append(i)
            frame.append(fr)
    n = len(place)
    slots_at = defaultdict(list)
    on_line = defaultdict(list)
    lines_of_place = defaultdict(set)
    by_val = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
        lines_of_place[place[s]].add(line[s])
        by_val[value[s]].append(s)
    line_map = defaultdict(lambda: defaultdict(list))
    for s in range(n):
        line_map[line[s]][place[s]].append(value[s])
    mean_fr = sum(len(fr) for fr in frame) / max(n, 1)

    qs = []
    for li, ss in on_line.items():
        if len(ss) < 2:
            continue
        cand = [(a, b) for ai, a in enumerate(ss) for b in ss[ai + 1:]
                if place[a] != place[b] and abs(pos[a] - pos[b]) > args.frame_max]
        if not cand:
            continue
        rng.shuffle(cand)
        qs.extend(cand[:args.pairs_per_line])
    rng.shuffle(qs)
    qs = qs[:args.max_questions]

    c = Counter()
    d_all, d_co = [], []
    n_move = n_off = n_empty = 0
    k = args.k
    for a, b in qs:
        pa, pb, va, vb = place[a], place[b], value[a], value[b]
        ca = Counter(value[t] for t in slots_at[pa] if t != a)
        cb = Counter(value[t] for t in slots_at[pb] if t != b)
        c["n"] += 1
        both = (va in ca) and (vb in cb)
        c["both_offered"] += both
        if not both:
            continue
        n_off += 1
        marg = (ca.most_common(1)[0][0], cb.most_common(1)[0][0])
        right = marg == (va, vb)
        seen = False
        common = lines_of_place[pa] & lines_of_place[pb]
        for lj in common:
            if lj == line[a]:
                continue
            m = line_map[lj]
            if va in m.get(pa, ()) and vb in m.get(pb, ()):
                seen = True
                break
        c["joint_seen"] += seen
        comp = (not right) and (not seen)
        c["comp_only"] += comp
        ht = hang_of(va, vb, a, b, by_val, line, frame, k)
        hm = hang_of(marg[0], marg[1], a, b, by_val, line, frame, k)
        if ht == 0.0:
            n_empty += 1
        n_move += int(ht != hm)
        d_all.append(ht - hm)
        if comp:
            d_co.append(ht - hm)

    def pct(key, d="n"):
        return c[key] / max(1, c[d])

    d_hang = (sum(d_co) / len(d_co)) if d_co else 0.0
    return {
        "slots": n, "questions": c["n"], "n_offered": n_off,
        "n_comp_only": len(d_co),
        "both_offered": pct("both_offered"),
        "joint_seen": pct("joint_seen", "both_offered"),
        "comp_only_of_offered": pct("comp_only", "both_offered"),
        "d_hang": d_hang,
        "d_hang_all": (sum(d_all) / len(d_all)) if d_all else 0.0,
        "move_rate": n_move / max(n_off, 1),
        "empty_true": n_empty / max(n_off, 1),
        "mean_frame": mean_fr,
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
    void = (rep["n_comp_only"] < 30) or (rep["move_rate"] <= 0.05) or (rep["mean_frame"] <= 0)
    gate = (not void) and (rep["d_hang"] > 0.05)
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"mean_frame {rep['mean_frame']:.2f}  offered {rep['n_offered']}  "
          f"comp_only {rep['n_comp_only']}  move {rep['move_rate']:.3f}")
    print(f"d_hang  comp_only {rep['d_hang']:+.4f}  all_offered {rep['d_hang_all']:+.4f}")
    print(f"VOID {rep['void']}  GATE {rep['gate']}")
    if void:
        print("\nVOID: no frame mass, or hang does not move. Not a pair scorer.")
    elif gate:
        print("\nGO ALGEBRA: joint is in frame overlap. No net (38.3).")
    else:
        print("\nSTOP ALGEBRA: frames exist, hang is blind. Do not Phi on this ctx.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
