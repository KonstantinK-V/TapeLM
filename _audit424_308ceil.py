"""424: 308 CEILING, no net. Joint in the WORLD, not in a retrieval set.

Two holes on one line, further apart than frame_max (308's leak lock: neither
window contains the other token). Answer is a pair. Rival = independent
place-majority argmax (not a full product likelihood). Joint statistic = the
pair stood at these two places on ANOTHER line. `comp_only` is the ARENA:
offered, majorities wrong, pair unseen elsewhere — not a proof of composition.

  VOID    both_offered <= 0.05
  ROOM    joint_seen < 0.90 of offered
  ARENA   comp_only_of_offered > 0.05
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage424_308ceil.json")
WIKI = Path("data/_wikitext103_train.txt")


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
    lines_of_place = defaultdict(set)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
        lines_of_place[place[s]].add(line[s])
    line_map = defaultdict(lambda: defaultdict(list))
    n_lp_dup = 0
    for s in range(n):
        slot = line_map[line[s]][place[s]]
        if slot:
            n_lp_dup += 1
        slot.append(value[s])
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
    price = []
    for a, b in qs:
        pa, pb, va, vb = place[a], place[b], value[a], value[b]
        ca = Counter(value[t] for t in slots_at[pa] if t != a)
        cb = Counter(value[t] for t in slots_at[pb] if t != b)
        c["n"] += 1
        price.append(len(ca) * len(cb))
        both = (va in ca) and (vb in cb)
        c["both_offered"] += both
        if not both:
            continue
        marg = (ca.most_common(1)[0][0], cb.most_common(1)[0][0])
        right = marg == (va, vb)
        c["marginal_right"] += right
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
        c["comp_only"] += (not right) and (not seen)

    def pct(k, d="n"):
        return c[k] / max(1, c[d])

    return {
        "slots": n, "questions": c["n"],
        "both_offered": pct("both_offered"),
        "marginal_right": pct("marginal_right", "both_offered"),
        "joint_seen": pct("joint_seen", "both_offered"),
        "comp_only": pct("comp_only"),
        "comp_only_of_offered": pct("comp_only", "both_offered"),
        "pair_price_mean": sum(price) / max(1, len(price)),
        "n_line_place_dups": n_lp_dup,
        "working_cells": 0,
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
    void = rep["questions"] < 50 or rep["both_offered"] <= 0.05
    room = rep["joint_seen"] < 0.90
    arena = rep["comp_only_of_offered"] > 0.05
    go = (not void) and room and arena
    rep["void"], rep["room"], rep["arena"], rep["go"] = (
        bool(void), bool(room), bool(arena), bool(go))
    print(f"{rep['questions']} two-hole  both_offered {rep['both_offered']:.4f}  "
          f"price {rep['pair_price_mean']:.0f}  line-place dups {rep['n_line_place_dups']}")
    print(f"marginals {rep['marginal_right']:.4f}  joint_seen {rep['joint_seen']:.4f}  "
          f"comp_only {rep['comp_only']:.4f}  of_offered {rep['comp_only_of_offered']:.4f}")
    print(f"VOID {rep['void']}  ROOM {rep['room']}  ARENA {rep['arena']}")
    if void:
        print("\nVOID: two-hole is not askable here. Object does not form.")
    elif not room:
        print("\nNO ROOM: pair already stands together as a catalog. Phi hang has nothing to add (38.3).")
    elif not arena:
        print("\nNO ARENA: counting (marginals or pair-seen) already covers the offered pairs. "
              "Hang has no slice.")
    else:
        print("\nTWO-HOLE WORLD HAS A SLICE WHERE COUNTING IS BLIND. Phi is not in this file.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
