"""Check of 400's deficit feature and of the artifact check that decided its one bit.

400 asks the dual of 399: does the line CLOSE what the rest of its scope lacks. The ways it can
print a number that means something else are specific, and one of them fired on the real run - the
`Return` bit, which turned out to be detecting the cut rather than the line.

  1. EQUIVARIANCE. Renaming every identifier must not move a single number. The feature is
     `type(node).__name__` and nothing else.
  2. THE LEAK, IN BOTH CURRENCIES. The pooled line is out of its own scope's TYPES and out of its
     LINE COUNT. Leaving it in either place lets the true scope be recognised by its own content.
  3. THE TWIN excludes the candidate itself, takes the nearest remaining-line count, and breaks
     ties by index - deterministic, so no number moves with a draw.
  4. THE DIRECTION IS THE ARGMAX OF THE DECLARED SCORE. Reversing a declared direction on seeing
     the number is the after-the-fact rescue this project has refused four times, so the code must
     contain no flip.
  5. THE POPULATION IS `amb_live` WITH THE TRUTH AMONG THE TIED - never the full corpus, where
     counting already answers.
  6. THE ARTIFACT CHECK READS THE FULL BODY. `bit_rivals_noret` must be computed with NOTHING
     dropped: if it read the remainder it would be the same quantity as the bit itself and could
     never fire. This is the check that voided a +0.34.
  7. THE BIT IS REPORTED BESIDE THE SCORE, NEVER BLENDED INTO IT.
  8. EXPECTED ACCURACY under uniform tie-breaking, exact and seed-free.

    python _check400_deficit.py
"""
from __future__ import annotations

import re
from pathlib import Path

import _audit398_scope as S
import _audit399_shape as H
import _audit400_deficit as A

SRC = Path("_audit400_deficit.py")

DESIGNED = '''
def alpha(one):
    two = one + 1
    for three in range(two):
        two = two + three
    return two


def beta(four):
    five = four
    return five


def gamma(six):
    seven = six
    eight = seven
    nine = eight
    return nine
'''
RENAMED = DESIGNED
for _a, _b in (("alpha", "zulu"), ("beta", "victor"), ("gamma", "whisky"), ("one", "kilo"),
               ("two", "lima"), ("three", "mike"), ("four", "november"), ("five", "oscar"),
               ("six", "papa"), ("seven", "quebec"), ("eight", "romeo"), ("nine", "sierra")):
    RENAMED = RENAMED.replace(_a, _b)


def deficit_vec(src):
    sc = S.scopes_of(src)
    line_t, body_t = H.types_of(src, sc["owner"])
    ln = min(l for l, ts in line_t.items() if "For" in ts)
    nf = len(sc["funcs"])
    true_i = sc["owner"][ln]
    from collections import Counter as C
    n_body = C(sc["owner"].values())
    rem = [n_body[i] - (1 if i == true_i else 0) for i in range(nf)]
    trem = {i: H.types_wo(body_t, i, ln) for i in range(nf)}
    lt = line_t.get(ln, set())
    d = [len([t for t in lt if t not in trem[i]]) for i in range(nf)]
    out = []
    for i in range(nf):
        tj, _g = A.twin_of(rem, i)
        out.append(float(d[i] - (d[tj] if tj is not None else 0)))
    return out, sorted(lt), rem


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []

    # 1: equivariance
    va, ta, ra = deficit_vec(DESIGNED)
    vb, tb, rb = deficit_vec(RENAMED)
    if (va, ta, ra) != (vb, tb, rb):
        f.append(f"1. renaming moved the feature: {va} vs {vb}, types {ta} vs {tb} - the deficit "
                 f"is reading identity")

    # 2: the leak, both currencies
    sc = S.scopes_of(DESIGNED)
    line_t, body_t = H.types_of(DESIGNED, sc["owner"])
    fl = min(l for l, ts in line_t.items() if "For" in ts)
    i = sc["owner"][fl]
    if "For" in H.types_wo(body_t, i, fl) or "For" not in H.types_wo(body_t, i, -1):
        f.append("2. the pooled line's types are not taken out of its own scope")
    if "rem = [n_body[i] - (1 if i == true_i else 0) for i in range(nf)]" not in src:
        f.append("2. the pooled line is not taken out of its own scope's LINE COUNT, so the "
                 "size-twin is matched against a size that still contains the answer")

    # 3: the twin
    if A.twin_of([5, 5, 9], 0)[0] != 1 or A.twin_of([5, 9, 6], 0)[0] != 2:
        f.append(f"3. the twin is not the nearest by size: {A.twin_of([5, 9, 6], 0)}")
    if A.twin_of([7, 7, 7], 1)[0] == 1:
        f.append("3. a scope is its own twin, so the subtraction is identically zero")
    if A.twin_of([5, 5, 9], 0)[1] != 0:
        f.append("3. the reported gap is not |size difference|")

    # 4: no flip of the declared direction
    if re.search(r"expected_acc\(\[-", src) or "s_def = [-" in src or "-x for x in s_def" in src:
        f.append("4. the declared direction is negated somewhere - the argmax of the score is "
                 "the attachment, and a sign flipped after the fact is a rescue")
    if 'c["deficit"] += H.expected_acc(s_def, true_i, tied)' not in src:
        f.append("4. the score entering the accuracy is not `s_def` as declared")

    # 5: the population
    if "if len(tied) < 2 or top <= 0.0 or true_i not in tied:" not in src:
        f.append("5. the population is not amb_live with the truth among the tied")

    # 6: the artifact check reads the FULL body
    if 'full = {i: H.types_wo(body_t, i, -1) for i in tied}' not in src:
        f.append("6. the artifact check does not read the FULL body - reading the remainder "
                 "would make it the same quantity as the bit, and it could never fire")
    if 'c["bit_true_noret_full"] += int("Return" not in full[true_i])' not in src:
        f.append("6. the true scope's full body is not checked, so 'the bit is the cut' cannot "
                 "be told from 'the true scope simply has no Return'")

    # 7: the bit is not blended in
    body = src[src.find('if "Return" in lt:'):]
    if 's_def' in body[:600]:
        f.append("7. the bit is mixed into the general score")

    # 8
    if H.expected_acc([1.0, 1.0, 0.0], 0, [0, 1, 2]) != 0.5:
        f.append("8. expected accuracy is not 1/|argmax|")
    return f


MUTANTS = (
    ("the twin may be the candidate itself",
     "        if j == i:\n            continue", "        if False:\n            continue", "3."),
    ("the twin ignores size",
     "        d = abs(rem[j] - rem[i])", "        d = float(j)", "3."),
    ("the pooled line stays in the size the twin matches",
     "            rem = [n_body[i] - (1 if i == true_i else 0) for i in range(nf)]",
     "            rem = [n_body[i] for i in range(nf)]", "2."),
    ("the declared direction is flipped",
     'c["deficit"] += H.expected_acc(s_def, true_i, tied)',
     'c["deficit"] += H.expected_acc([-x for x in s_def], true_i, tied)', "4."),
    ("the population widens to every scored line",
     "            if len(tied) < 2 or top <= 0.0 or true_i not in tied:",
     "            if False:", "5."),
    ("the artifact check reads the remainder, so it can never fire",
     "                full = {i: H.types_wo(body_t, i, -1) for i in tied}",
     "                full = {i: H.types_wo(body_t, i, ln) for i in tied}", "6."),
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
        try:
            exec(compile(mutated, "<mutant>", "exec"), A.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
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
