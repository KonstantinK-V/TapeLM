"""Check of 358's recall column - the one that decides whether `--min-fillers 1` is readable.

8-RESULT-6 named TWO reasons the write path deletes what ingestion creates:

  1. `min_fillers >= 2` deletes CONSTANT frames, which is exactly what self-reference makes.
  2. EVEN KEPT, THE EXAM CANNOT ASK ABOUT ONE - the lens is the place's other fillers and the
     offer excludes them (`w != v`), so a constant place's truth is outside its own offer by
     construction.

`--min-fillers 1` lifts the first and NOT the second. Re-running 358 with the old column alone
would therefore admit every place ingestion creates and score all of them misses, and the gate
would read "there is nowhere to step" when the truth is "the exam still cannot ask". The recall
column exists so those two are separable, and these are its properties. Each is a number on a
designed corpus, and each has its own failure re-introduced below.

  1. THE OLD COLUMN IS UNCHANGED. `hit` still skips a place with no lens, still bans the place's
     own values and still excludes the lens value. Every 358 number on record stays comparable.
  2. THE DENOMINATOR IS ABSOLUTE. No tape is `tok` with zeros, never a smaller denominator
     (the 342a lesson).
  3. A CONSTANT PLACE: unreachable by substitution, reachable by recall. This is 8-RESULT-6's
     second cause as a measurement instead of an argument.
  4. AT min_fillers 2 THE SAME PLACE DOES NOT EXIST. That is the first cause, and the two must
     be visible apart.
  5. RECALL EXCLUDES THIS POSITION. A truth standing only where it is hidden must not be
     recalled - otherwise the channel reads the answer.
  6. CONSTANT IS DEFINED ON THE PLACE WITH THIS POSITION REMOVED.
  7. THE SPLIT COUNTS WHAT INGESTION ADDED - reached under ingest and not under base.

    python _check396_ingest.py
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import _audit358_ingest as A

SRC = Path("_audit358_ingest.py")

OLD = ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi",
       "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"]
# the article introduces its subject and repeats it: the frame `the|team` holds XARWIN twice
DOC0 = "the XARWIN team won the opening match of the season in a long padded line of text"
QLINE = "the XARWIN team lost the closing match of the season in a long padded line here"
# a MIXED place: `a|sat` holds CAT earlier and DOG in the question line
MIX0 = "a CAT sat on the mat while the weather outside was cold and grey for many days"
MIXQ = "a DOG sat on the mat while the weather outside was warm and dry for many days"


def call(prefix, qline, mf):
    return A.reach_line(OLD + prefix + [qline], qline, 1, mf, 8)


def props(src=None):
    """`src` is the source the STATIC halves are read from - the mutants patch the module in
    memory, so a check that re-read the file from disk could never see one and would be a
    comment. The behavioural halves run against whatever `A` currently holds."""
    f = []
    src = SRC.read_text(encoding="utf-8") if src is None else src
    body = re.sub(r'"""(?:.|\n)*?"""', "",
                  re.search(r"^def reach_line\(.*?(?=\ndef )", src, re.S | re.M).group(0))

    # 1: the old column, unchanged
    for want in ("if not lens:", "cnt -= ban.get(w, 0)", "if cnt > 0 and w != v:",
                 'off.most_common(topm)'):
        if want not in body:
            f.append(f"1. the substitution column changed: {want!r} is gone, so no 358 number "
                     f"on record is comparable any more")

    # 2: the absolute denominator
    c, _p = A.reach_line(["aa bb cc"], "aa bb cc", 1, 1, 8)
    if c["tok"] != 3 or c["hit"] or c["rec"]:
        f.append(f"2. a line with no tape reports {dict(c)}, not 3 positions and zeros")

    # 3 + 4: the constant place, and the filter that deletes it
    b1, _ = call([], QLINE, 1)
    i1, _ = call([DOC0], QLINE, 1)
    i2, _ = call([DOC0], QLINE, 2)
    if (b1["rec_ask"], b1["orc"]) != (0, 0):
        f.append(f"3. without the document the question line is already on a place "
                 f"({b1['rec_ask']} positions) - the designed case is not designed")
    if (i1["ask"], i1["hit"]) != (10, 0):
        f.append(f"3. with the document ingested the substitution column reads ask="
                 f"{i1['ask']} hit={i1['hit']}, expected 10 and 0 - a constant place's truth is "
                 f"excluded from its own offer by construction, and if `hit` is non-zero here "
                 f"the offer stopped excluding the lens value")
    if (i1["rec"], i1["const"], i1["orc"]) != (8, 8, 8):
        f.append(f"3. recall reads rec={i1['rec']} const={i1['const']} orc={i1['orc']}, "
                 f"expected 8/8/8 - the document's own past is what those places are made of")
    if (i2["rec_ask"], i2["orc"]) != (2, 0):
        f.append(f"4. at min_fillers 2 the same document gives rec_ask={i2['rec_ask']} "
                 f"orc={i2['orc']}, expected 2 and 0 - the constant frames must be DELETED "
                 f"there, which is the first of 8-RESULT-6's two causes")

    # 5 + 6: recall excludes this position, and `const` is read with it removed.
    # READ AT THE ONE POSITION THEY ARE ABOUT. The rest of the line repeats function words, so
    # its other positions ARE constant places and reading the line total here would test the
    # padding instead of the property.
    _m, per = call([MIX0], MIXQ, 1)
    dog = per.get(1)
    if dog is None:
        f.append("5. the mixed position is not on a place - the designed case is not designed")
    else:
        _hit, rec, const = dog
        if rec:
            f.append("5. recall fired at a position whose truth stands nowhere else - the "
                     "channel is reading the answer")
        if const:
            f.append("6. a place holding CAT and DOG counts as constant - `const` is not read "
                     "with this position removed")

    # 7: the split is what ingestion added
    if "if (hit or rec) and not (b[0] or b[1]):" not in src:
        f.append("7. the const/mixed split does not condition on the base arm, so it counts "
                 "places ingestion did not add")
    return f


MUTANTS = (
    ("the offer stops excluding the lens value",
     "                if cnt > 0 and w != v:", "                if cnt > 0:", "1."),
    ("no tape shrinks the denominator",
     '        return Counter({"tok": n_tok}), {}', "        return Counter(), {}", "2."),
    ("recall reads the hidden position too",
     "        rec = int(truth in {toks[x] for x in places[pid] if x != s})",
     "        rec = int(truth in {toks[x] for x in places[pid]})", "5."),
    # NOT "count the place with this position still in": that formulation is EQUIVALENT to the
    # real one (one distinct value at the place implies the others are the truth), so it would
    # be a no-op dressed as a mutation. The failure that is actually available is dropping the
    # `and rec` - which calls a place holding CAT and DOG a paradigm of CAT.
    ("constant forgets that the others must be the truth",
     "        const = int(len(others) == 0 or (len(others) == 1 and rec))",
     "        const = int(len(others) <= 1)", "6."),
    ("the split ignores the base arm",
     "                if (hit or rec) and not (b[0] or b[1]):", "                if hit or rec:",
     "7."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        saved = dict(A.__dict__)
        mutated = src.replace(old, new, 1)
        exec(compile(mutated, "<mutant>", "exec"), A.__dict__)
        try:
            got = props(src=mutated)
        finally:
            A.__dict__.clear()
            A.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): the failure was re-introduced and check "
                         f"{tag} did not fire - it is a comment, not a check")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
