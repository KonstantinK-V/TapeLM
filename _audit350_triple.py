"""The last cheap question: can a DIRECTED relation be counted out of the same raw text?

WHERE THIS SITS. 347 and 349 together closed the substrate:

    the tape records (frame) -> fillers. Two fillers relate when they can fill THE SAME HOLE.
    That is ALTERNATION, not association - a paradigm table, which is a lexicon.
    A lexicon ranks alternatives. It cannot produce content.

349 asked whether the same tape holds a second relation - "stood on the same line" instead of
"stands in the same hole" - and the answer was: distinguishable but strictly worse. 29% of what
it reaches is unique, and it reaches half as much (0.103 vs 0.193), resolves five times worse
(argmax 0.008 vs 0.035) and is less supported. Unioning the two is sixteen candidates instead of
eight, which is WIDER ENUMERATION - the operation 335 closed as asymptotically wrong. So it adds
nothing that is not already known to be the wrong direction.

WHAT HAS NEVER BEEN COUNTED. Both relations are UNDIRECTED and both are about co-membership: same
hole, or same line. Neither records WHAT STOOD BETWEEN. A directed triple does:

    (A, the text between, B)        A stood, then this, then B

That is still pure counting off raw text - no parsing, no model, no threshold - and it is the
first relation on the table that CAN GENERATE: given A and a pattern, it produces a B that A's
own paradigm never contained. Substitution can only ever hand back things that already stood
where A stands.

THIS AUDIT DOES NOT BUILD IT. It asks whether such triples RECUR at all in the same 30 MB, in the
same columns as 346 and 349 so all three relations are comparable on one page:

    present@8    the truth among the top eight the relation offers
    argmax       what it resolves to
    support2+    the share of (A, pattern) -> B triples seen more than once
    tri_only     truths reached by the triple and NOT by substitution - the number that decides

If triples do not recur, then at this corpus size there is nothing directed to count, and the
project's result is the separation proof, stated as such. If they do, the write path has a shape
worth rebuilding for, and that is a new phase rather than a lever.

THE QUESTION'S OWN LINES ARE EXCLUDED WHOLE, as in 349: a triple drawn from the very sentence the
hole sits in is the sentence, not evidence from elsewhere.

    python _audit350_triple.py
    python _audit350_triple.py --gap 2        # tighter patterns, fewer and sharper
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage350_triple.json")
TOPM = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--gap", type=int, default=4,
                    help="most tokens allowed BETWEEN the two ends of a triple. The pattern is "
                         "what constrains, so a wider gap means more patterns and each one "
                         "rarer - the same crowding 347 measured, in a different place")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-questions", type=int, default=3000)
    ap.add_argument("--corpus", default=str(WIKI))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]

    rng = random.Random(args.seed)
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.window_lines:
        by_line0 = tframes._by_line(keep, owner)
        start = rng.randrange(max(1, len(lines)))
        acc = defaultdict(list)
        for d in range(args.window_lines):
            for k, i in by_line0.get((start + d) % len(lines), ()):
                acc[k].append(i)
        win_lines = {(start + d) % len(lines) for d in range(args.window_lines)}
        keep = [(k, sorted(v)) for k, v in acc.items()
                if len({toks[i] for i in v}) >= args.min_fillers]
    else:
        win_lines = set(range(len(lines)))
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print("no tape")
        return 1

    vals_at = [[toks[i] for i in ps] for _k, ps in keep]
    slots_at = [list(ps) for _k, ps in keep]
    P = len(vals_at)
    place_of, line_slots = {}, defaultdict(list)
    for j, ps in enumerate(slots_at):
        for s in ps:
            place_of[s] = j
            line_slots[owner[s]].append(s)
    where = defaultdict(list)
    for j, ps in enumerate(slots_at):
        for s in ps:
            where[toks[s]].append(s)

    # ---- THE TRIPLES, counted once off the same text the tape was built from ----------------
    # Both ends must be TAPE VALUES, or the relation would be over a different vocabulary than
    # the questions are asked in and the comparison would be meaningless.
    on_tape = set(where)
    by_line_tok = defaultdict(list)          # line -> [(position in toks)]
    for s in range(len(toks)):
        if owner[s] in win_lines and toks[s] in on_tape:
            by_line_tok[owner[s]].append(s)
    tri = defaultdict(Counter)               # (A, pattern) -> Counter(B)
    tri_lines = defaultdict(set)             # (A, pattern, B) -> the lines it was seen on
    for li, poss in by_line_tok.items():
        for a in range(len(poss)):
            for b in range(a + 1, len(poss)):
                i, j2 = poss[a], poss[b]
                gap = j2 - i - 1
                if gap > args.gap:
                    break
                pat = tuple(toks[i + 1:j2])
                tri[(toks[i], pat)][toks[j2]] += 1
                tri_lines[(toks[i], pat, toks[j2])].add(li)

    qs = [(j, i) for j in range(P) for i in range(len(vals_at[j])) if len(vals_at[j]) >= 2]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]

    c = Counter()
    off_p, off_t = [], []
    sup_t = Counter()
    for j, i in qs:
        truth = vals_at[j][i]
        own_c = Counter(vals_at[j])
        own_c[truth] -= 1
        if own_c[truth] <= 0:
            del own_c[truth]
        lens = list(own_c)[:6]
        if not lens:
            continue
        c["n"] += 1
        mine = set(slots_at[j])
        my_lines = {owner[s] for s in mine}

        def paradigm(v):
            out = Counter()
            for s in where[v]:
                if s in mine:
                    continue
                for s2 in slots_at[place_of[s]]:
                    if s2 not in mine and toks[s2] != v:
                        out[toks[s2]] += 1
            return out

        def triple(v):
            """Everything the directed relation offers from this end, with the question's own
            LINES dropped whole - a triple drawn from the sentence the hole sits in is the
            sentence. Both directions, because "A then B" and "B then A" are different
            relations and a tape that recorded only one would be choosing a reading order."""
            out = Counter()
            for (a, pat), ends in tri.items():
                if a != v:
                    continue
                for b, n in ends.items():
                    seen = tri_lines[(a, pat, b)] - my_lines
                    if seen and b != v:
                        out[b] += len(seen)
            return out

        rp = Counter()
        rt = Counter()
        for v in lens:
            rp += paradigm(v)
            rt += triple(v)
        off_p.append(len(rp))
        off_t.append(len(rt))
        for _w, n in rt.items():
            sup_t[n] += 1
        pres_p = truth in {w for w, _n in rp.most_common(TOPM)}
        pres_t = truth in {w for w, _n in rt.most_common(TOPM)}
        c["par_present"] += pres_p
        c["tri_present"] += pres_t
        c["tri_right"] += bool(rt) and rt.most_common(1)[0][0] == truth
        c["par_right"] += bool(rp) and rp.most_common(1)[0][0] == truth
        c["tri_only"] += pres_t and not pres_p
        c["par_only"] += pres_p and not pres_t
        c["tri_empty"] += not rt

    n = max(1, c["n"])
    tt = max(1, sum(sup_t.values()))
    rep = {
        "bytes": args.bytes, "window_lines": args.window_lines, "gap": args.gap,
        "places": P, "questions": c["n"], "n_patterns": len(tri),
        "paradigm": {"present_topm": c["par_present"] / n, "argmax_right": c["par_right"] / n,
                     "offer": sum(off_p) / max(1, len(off_p))},
        "triple": {"present_topm": c["tri_present"] / n, "argmax_right": c["tri_right"] / n,
                   "offer": sum(off_t) / max(1, len(off_t)),
                   "support_2plus": sum(v for k, v in sup_t.items() if k >= 2) / tt,
                   "empty": c["tri_empty"] / n},
        "tri_only": c["tri_only"] / n, "par_only": c["par_only"] / n,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    p_, t_ = rep["paradigm"], rep["triple"]
    print(f"tape    {P} places, {c['n']} questions, {len(tri)} (A,pattern) keys, gap {args.gap}")
    print(f"SUBST   present@{TOPM} {p_['present_topm']:.4f}  argmax {p_['argmax_right']:.4f}  "
          f"offer {p_['offer']:.1f}")
    print(f"TRIPLE  present@{TOPM} {t_['present_topm']:.4f}  argmax {t_['argmax_right']:.4f}  "
          f"offer {t_['offer']:.1f}  support2+ {t_['support_2plus']:.4f}  "
          f"empty {t_['empty']:.4f}")
    print(f"APART   only by TRIPLE {rep['tri_only']:.4f}   only by SUBST {rep['par_only']:.4f}")
    if t_["empty"] > 0.5:
        print("\nNOTHING DIRECTED TO COUNT at this corpus size: over half the questions have no "
              "triple from any of their own rows. The relation is not in 30 MB.")
    elif rep["tri_only"] > 0.05 and t_["argmax_right"] >= p_["argmax_right"]:
        print("\nA GENERATIVE RELATION EXISTS: the triple reaches truths substitution cannot AND "
              "resolves at least as well. The write path has a shape worth rebuilding for - a "
              "new phase, not a lever.")
    else:
        print("\nNOT BETTER THAN SUBSTITUTION: the directed relation adds reach only by widening "
              "the offer, which 335 already closed. The substrate is the ceiling and the "
              "project's result is the separation proof.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
