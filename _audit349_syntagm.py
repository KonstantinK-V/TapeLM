"""Does the tape hold ANY relation other than substitutability? Measured before anything is built.

WHAT 347 SETTLED, AND IT IS NOT WHAT IT WAS BUILT TO SETTLE. Thickening a place works exactly as
intended and does not help:

    window   ment/pl   support2+    offer   present@8   argmax   in_own
       400      4.24      0.1600     37.8      0.1933   0.0373   0.2610
      6400      8.69      0.3512    202.9      0.1600   0.0397   0.5120

Thickness x2.05. Support x2.20 - the singleton problem GENUINELY IMPROVES, exactly as predicted.
And the lens's answer set grows x5.37, so the top eight of two hundred catches LESS than the top
eight of thirty-eight. The distribution gets more supported and more crowded at the same time,
and the ratio does not move. in_own nearly doubles: as a place thickens, the answer is more and
more often ALREADY THERE. Thickness helps the INDEX, not the constraint.

THE DIAGNOSIS THAT FOLLOWS, AND IT IS ABOUT THE TAPE AND NOT THE RULE. Two fillers "co-occur" on
this tape when they stand at THE SAME PLACE - the same frame, at different times. That does not
mean they are associated. It means they are ALTERNATIVES: things that can fill the same hole.
The tape's only relation is SUBSTITUTABILITY, which is a paradigmatic relation, and the walk's
fingerprint over filler bags is already an approximation of exactly it. So the constraint was
never a new relation - it was the walk's relation computed exactly instead of by hash, which is
why sharpening it, thickening it and re-resolving it all failed the same way.

A SUBSTITUTION RELATION CAN RANK ALTERNATIVES. IT CANNOT PRODUCE CONTENT. 343 said Phi is a
chooser and not a generator; this says the TAPE is a chooser's tape, and no operation over it
will generate.

WHAT THIS AUDIT ASKS. Is there a SYNTAGMATIC relation in the same tape - not "w can stand where v
stands" but "w stood ALONGSIDE v", in the same line, at a different place? That is a different
relation with a different shape: it links places rather than substituting within one, and it is
the only kind that can put a value in front of the mind that its own paradigm does not contain.

Everything is measured in 346's columns so the two relations are directly comparable, on the
same tape and the same questions:

    present@8   the truth in the top eight the relation offers
    argmax      what it resolves to
    support2+   the share of related pairs seen more than once
    offer       how many values it has to choose between

THE QUESTION'S OWN PLACE IS EXCLUDED, and so is any LINE it stands on - otherwise the value
sitting next to the hole in the very same sentence counts as evidence from elsewhere, which is
the same leak cons_resolve subtracts and 304's line channel was closed for.

    python _audit349_syntagm.py
    python _audit349_syntagm.py --window-lines 3200      # and on a thick tape
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage349_syntagm.json")
TOPM = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
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
        keep = [(k, sorted(v)) for k, v in acc.items()
                if len({toks[i] for i in v}) >= args.min_fillers]
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print("no tape")
        return 1

    vals_at = [[toks[i] for i in ps] for _k, ps in keep]
    slots_at = [list(ps) for _k, ps in keep]
    P = len(vals_at)
    place_of = {}                             # slot -> place index
    line_slots = defaultdict(list)            # line -> slots on it that are on the tape
    for j, ps in enumerate(slots_at):
        for s in ps:
            place_of[s] = j
            line_slots[owner[s]].append(s)
    where = defaultdict(list)                 # value -> slots holding it
    for j, ps in enumerate(slots_at):
        for s in ps:
            where[toks[s]].append(s)

    qs = [(j, i) for j in range(P) for i in range(len(vals_at[j])) if len(vals_at[j]) >= 2]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]

    c = Counter()
    offers_p, offers_s = [], []
    sup_p, sup_s = Counter(), Counter()
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
        c["in_own"] += truth in own_c
        mine = set(slots_at[j])

        def paradigm(v):
            """346's relation: values at the PLACES that hold v. Substitutability."""
            out = Counter()
            for s in where[v]:
                if s in mine:
                    continue
                for s2 in slots_at[place_of[s]]:
                    if s2 not in mine and toks[s2] != v:
                        out[toks[s2]] += 1
            return out

        def syntagm(v):
            """The other relation: values that stood ON THE SAME LINE as v, at other places.

            A line where v stands AT THIS QUESTION'S OWN PLACE is dropped whole - not just the
            hidden slot. The value beside the hole in the same sentence is the sentence, not
            evidence from elsewhere, and counting it is the leak 304 was closed for.
            """
            out = Counter()
            for s in where[v]:
                if s in mine:
                    continue
                ls = line_slots.get(owner[s], ())
                if any(s2 in mine for s2 in ls):
                    continue          # the question's own place stands on this line: drop it
                for s2 in ls:
                    if place_of[s2] != place_of[s] and toks[s2] != v:
                        out[toks[s2]] += 1
            return out

        bp, bs = None, None
        pb = sb = -1
        pres_p = pres_s = False
        for v in lens:
            rp, rs = paradigm(v), syntagm(v)
            offers_p.append(len(rp))
            offers_s.append(len(rs))
            for w, n in rp.items():
                sup_p[n] += 1
                if n > pb:
                    pb, bp = n, w
            for w, n in rs.items():
                sup_s[n] += 1
                if n > sb:
                    sb, bs = n, w
            if truth in {w for w, _n in rp.most_common(TOPM)}:
                pres_p = True
            if truth in {w for w, _n in rs.most_common(TOPM)}:
                pres_s = True
        c["par_present"] += pres_p
        c["par_right"] += bp == truth
        c["syn_present"] += pres_s
        c["syn_right"] += bs == truth
        # THE ONE THAT DECIDES: the truth reachable by the NEW relation and not by the old one.
        # A relation that only re-finds what substitutability already offers is not a second
        # relation, it is the same one in different clothes.
        c["syn_only"] += pres_s and not pres_p
        c["par_only"] += pres_p and not pres_s
        c["syn_empty"] += not any(syntagm(v) for v in lens)

    n = max(1, c["n"])
    tp, ts = max(1, sum(sup_p.values())), max(1, sum(sup_s.values()))
    rep = {
        "bytes": args.bytes, "window_lines": args.window_lines, "places": P,
        "questions": c["n"], "in_own": c["in_own"] / n,
        "paradigm": {"present_topm": c["par_present"] / n, "argmax_right": c["par_right"] / n,
                     "offer": sum(offers_p) / max(1, len(offers_p)),
                     "support_2plus": sum(v for k, v in sup_p.items() if k >= 2) / tp},
        "syntagm": {"present_topm": c["syn_present"] / n, "argmax_right": c["syn_right"] / n,
                    "offer": sum(offers_s) / max(1, len(offers_s)),
                    "support_2plus": sum(v for k, v in sup_s.items() if k >= 2) / ts,
                    "empty": c["syn_empty"] / n},
        "syn_only": c["syn_only"] / n, "par_only": c["par_only"] / n,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    p_, s_ = rep["paradigm"], rep["syntagm"]
    print(f"tape    {P} places, {c['n']} questions, in_own {rep['in_own']:.4f}, "
          f"window {args.window_lines}")
    print(f"SUBST   present@{TOPM} {p_['present_topm']:.4f}  argmax {p_['argmax_right']:.4f}  "
          f"offer {p_['offer']:.1f}  support2+ {p_['support_2plus']:.4f}")
    print(f"ALONG   present@{TOPM} {s_['present_topm']:.4f}  argmax {s_['argmax_right']:.4f}  "
          f"offer {s_['offer']:.1f}  support2+ {s_['support_2plus']:.4f}  "
          f"empty {s_['empty']:.4f}")
    print(f"APART   reached only by ALONG {rep['syn_only']:.4f}   "
          f"only by SUBST {rep['par_only']:.4f}")
    if s_["empty"] > 0.5:
        print("\nTHE RELATION IS NOT THERE: over half the questions have no same-line partner "
              "at all. A frame tape records holes, not neighbours, and this is what that costs.")
    elif rep["syn_only"] > 0.03:
        print("\nA SECOND RELATION EXISTS: it reaches truths substitutability does not. That is "
              "the first thing on this tape that could GENERATE rather than rank, and the write "
              "path is where it would have to be recorded properly.")
    else:
        print("\nSAME RELATION IN DIFFERENT CLOTHES: standing alongside reaches nothing that "
              "standing-in-place does not. The tape holds one relation and it is substitution.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
