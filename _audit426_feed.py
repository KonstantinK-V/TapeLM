"""426: WHERE 425's feed dies. No hang gate. K unchanged. No Phi."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
import _audit425_hang as H425

OUT = Path("results/_stage426_feed.json")
WIKI = Path("data/_wikitext103_train.txt")


def text_occ(tok, line_toks, forbid_line):
    n = 0
    for li, ws in line_toks.items():
        if li == forbid_line:
            continue
        n += sum(1 for w in ws if w == tok)
    return n


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
    by_val = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
        lines_of_place[place[s]].add(line[s])
        by_val[value[s]].append(s)
    line_map = defaultdict(lambda: defaultdict(list))
    for s in range(n):
        line_map[line[s]][place[s]].append(value[s])
    line_toks = defaultdict(list)
    for i, tok in enumerate(toks):
        line_toks[owner[i]].append(tok)
    df = Counter()
    for ws in line_toks.values():
        for w in set(ws):
            df[w] += 1
    med = sorted(df.values())[len(df) // 2] if df else 1
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
    k = args.k
    acc = Counter()
    sums = Counter()

    def add(tag, row):
        acc[tag] += 1
        for key, v in row.items():
            sums[tag + "." + key] += v

    for a, b in qs:
        pa, pb, va, vb = place[a], place[b], value[a], value[b]
        ca = Counter(value[t] for t in slots_at[pa] if t != a)
        cb = Counter(value[t] for t in slots_at[pb] if t != b)
        acc["n"] += 1
        if not ((va in ca) and (vb in cb)):
            continue
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
        comp = (not right) and (not seen)
        eA = H425.evidence(va, by_val, line, line[a], {a, b}, k)
        eB = H425.evidence(vb, by_val, line, line[a], {a, b}, k)
        pairs = n_cross = n_rare = 0
        if eA and eB:
            cA = [H425.ctx_of(i, value, line, line_toks, vb) for i in eA]
            cB = [H425.ctx_of(j, value, line, line_toks, va) for j in eB]
            for i, ci in zip(eA, cA):
                for j, cj in zip(eB, cB):
                    pairs += 1
                    if line[i] == line[j]:
                        continue
                    n_cross += 1
                    if H425.rare_share(ci, cj, df, med) > 0:
                        n_rare += 1
        ta = text_occ(va, line_toks, line[a])
        tb = text_occ(vb, line_toks, line[a])
        ht = H425.hang_of(va, vb, a, b, by_val, line, value, line_toks, df, med, k)
        row = dict(
            slots_a=sum(1 for s in by_val.get(va, ()) if s not in (a, b)),
            slots_b=sum(1 for s in by_val.get(vb, ()) if s not in (a, b)),
            elig_a=len(eA), elig_b=len(eB),
            eA_ok=int(bool(eA)), eB_ok=int(bool(eB)),
            both_ok=int(bool(eA) and bool(eB)),
            pairs=pairs, cross=n_cross, rare=n_rare,
            cross_ok=int(n_cross > 0), rare_ok=int(n_rare > 0),
            hang=ht, hang_ok=int(ht > 0),
            text_a=ta, text_b=tb, text_both=int(ta > 0 and tb > 0),
        )
        add("off", row)
        if comp:
            add("co", row)

    def pack(tag):
        d = max(acc[tag], 1)

        def avg(key):
            return sums[tag + "." + key] / d

        return {
            "n": acc[tag],
            "mean_slots_a": avg("slots_a"),
            "mean_slots_b": avg("slots_b"),
            "eA_nonempty": avg("eA_ok"),
            "eB_nonempty": avg("eB_ok"),
            "both_nonempty": avg("both_ok"),
            "mean_elig_a": avg("elig_a"),
            "mean_elig_b": avg("elig_b"),
            "mean_pairs": avg("pairs"),
            "mean_cross": avg("cross"),
            "mean_rare": avg("rare"),
            "share_cross_gt0": avg("cross_ok"),
            "share_rare_gt0": avg("rare_ok"),
            "mean_hang": avg("hang"),
            "share_hang_gt0": avg("hang_ok"),
            "mean_text_a": avg("text_a"),
            "mean_text_b": avg("text_b"),
            "share_text_both": avg("text_both"),
        }

    n_rare_types = sum(1 for v in df.values() if v < med)
    return {
        "questions": acc["n"], "k": k,
        "df_median": med,
        "n_types": len(df),
        "n_rare_types": n_rare_types,
        "offered": pack("off"), "comp_only": pack("co"),
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
    ap.add_argument("--k", type=int, default=8)
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
    print(f"questions {rep['questions']}  k={rep['k']}  "
          f"df_median {rep['df_median']}  rare_types {rep['n_rare_types']}/{rep['n_types']}")
    for tag in ("offered", "comp_only"):
        p = rep[tag]
        print(f"\n{tag} n={p['n']}")
        print(f"  slots     a {p['mean_slots_a']:.2f}  b {p['mean_slots_b']:.2f}")
        print(f"  elig      a {p['mean_elig_a']:.2f}  b {p['mean_elig_b']:.2f}  "
              f"both_nonempty {p['both_nonempty']:.3f}")
        print(f"  pairs {p['mean_pairs']:.2f}  cross {p['mean_cross']:.2f}  "
              f"share_cross>0 {p['share_cross_gt0']:.3f}")
        print(f"  rare>0 {p['share_rare_gt0']:.3f}  hang {p['mean_hang']:.4f}  "
              f"hang>0 {p['share_hang_gt0']:.3f}")
        print(f"  TEXT occ  a {p['mean_text_a']:.2f}  b {p['mean_text_b']:.2f}  "
              f"both {p['share_text_both']:.3f}  (not evidence)")
    co = rep["comp_only"]
    if co["n"] and co["both_nonempty"] <= 0.05:
        print("\nDIE: answer-slots after question-line cut. Feed definition, not hang.")
        if co["share_text_both"] > 0.05:
            print("  text_occ of a and b lives — mentions exist, slots do not.")
    elif co["n"] and co["share_cross_gt0"] <= 0.05:
        print("\nDIE: evidence exists but no cross-line pairs (same-line lock ate them).")
    elif co["n"] and co["share_rare_gt0"] <= 0.05:
        print("\nDIE: cross-line pairs exist, rare_share is always 0. Algebra is flat.")
    else:
        print("\nFeed reaches hang on a real slice. Re-read 425 with this table; "
              "do not raise K.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
