"""Check of 409: is the tail findable without the answers. No torch, no corpus.

  1. NO FEATURE READS THE LABEL. `place_feats` never touches a hole, a truth or `value_of` - if it
     did, "findable" would be "computable from the answer".
  2. THE TWO WINDOWS ARE DISJOINT. The test window must start at least a full window after the
     train one, or the fit is read on places it was fitted on.
  3. FITTED ON A, READ ON B - both the family and the ablation.
  4. THE ORACLE IS THE SORTED TOP-B OF THE TEST WINDOW at the same budget, so `recovered` compares
     like with like.
  5. THE ABLATION IS FITTED-SINGLE IN BOTH SIGNS - a feature that predicts value NEGATIVELY is
     still one count.
  6. `value_of` keeps 390's discipline: the hole is out of its own place's profile, same-line
     places dropped.

    python _check409_findable.py
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("_audit409_findable.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    pf = re.search(r"^def place_feats\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    # `slots` and `owner[s]` are the place's OWN rows and are fine; what must never appear
    # is the label or a hidden hole.
    for bad in ("value_of", "truth", "hidden"):
        if bad in pf:
            f.append(f"1. place_feats reads {bad!r} - a feature must not see the answer")
    if "b0 = a0 + W + rng.randrange" not in src:
        f.append("2. the test window does not start a full window after the train one")
    if "Xa, ya = sample(TA" not in src or "Xb, yb = sample(TB" not in src:
        f.append("3. the two windows are not sampled separately")
    if "w = fit(Xa, ya," not in src or "learned = topB(Xb, yb, w," not in src:
        f.append("3. the fit is not on A and read on B")
    if "wj = fit([[r[j]] for r in Xa], ya," not in src:
        f.append("3/5. the ablation is not fitted-single on the training window")
    if "topB(Xb, yb, [-x for x in full]" not in src:
        f.append("5. the ablation does not try the negative sign")
    if "sorted(yb, reverse=True)[:min(args.budget, len(yb))]" not in src:
        f.append("4. the oracle is not the sorted top-B of the test window at the same budget")
    vo = re.search(r"^def value_of\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    if ("qprof = Counter(toks[x] for x in slots if x != s)" not in vo
            or "own = {toks[x] for x in slots if x != s}" not in vo
            or 'drop = set(T["on_line"][owner[s]])' not in vo):
        f.append("6. value_of drops 390's leak discipline")
    return f


MUTANTS = (
    ("a feature reads the answer",
     "    freq = [len(T[\"at_value\"].get(v, ())) for v in prof]",
     "    freq = [len(T[\"at_value\"].get(v, ())) for v in prof]\n    truth = 1", "1."),
    ("the windows overlap",
     "    b0 = a0 + W + rng.randrange(max(1, len(lines) - a0 - 2 * W))", "    b0 = a0", "2."),
    ("the fit is read on its own window",
     "    learned = topB(Xb, yb, w, args.budget)",
     "    learned = topB(Xa, ya, w, args.budget)", "3."),
    ("the ablation only tries one sign",
     "        s = max(topB(Xb, yb, full, args.budget),\n"
     "                topB(Xb, yb, [-x for x in full], args.budget))",
     "        s = topB(Xb, yb, full, args.budget)", "5."),
    ("the oracle is not the test window's own top-B",
     "        sum(sorted(yb, reverse=True)[:min(args.budget, len(yb))]) / min(args.budget, len(yb)))",
     "        sum(sorted(ya, reverse=True)[:min(args.budget, len(ya))]) / min(args.budget, len(ya)))",
     "4."),
    ("value_of keeps the hole in its own profile",
     "        qprof = Counter(toks[x] for x in slots if x != s)",
     "        qprof = Counter(toks[x] for x in slots)", "6."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        if not any(g.startswith(tag) for g in props(src.replace(old, new, 1))):
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
