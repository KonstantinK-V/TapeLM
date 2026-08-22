"""Check of 404: the fitted family, the vocabulary channel, and the ablation that decides both.

404 is the first step here that FITS, so the ways it can lie are the ways fitting lies:

  1. HELD OUT BY FILE. The train and test file sets must be disjoint - a fit scored on a file it
     saw is a memory, not a result.
  2. THE VOID CHECK IS COUNTED WITHOUT THE CAP. Read off two capped collections it printed
     1.0000, which is a property of the cap and not of the arena - and it is the number that is
     supposed to be read first.
  3. CHANNEL A READS NO IDENTIFIER. `feats_A` never touches the ids field, and the corpus run
     confirms it: unparsed and unparsed-renamed give the same number to the digit.
  4. THE VOCABULARY ADMITS ONLY CROSS-FILE WORDS. A word must occur in at least k DIFFERENT
     files - that is what makes it language rather than one tape's content, and it is the whole
     justification for letting any identifier in at all.
  5. THE POPULATION IS THE DECISION ONE: two or more safe candidates, and the true next line must
     itself be legal, or the state has no answer in it.
  6. THE ABLATION USES THE SAME HELD-OUT STATES as the family. A best-single measured elsewhere
     could not settle whether this is a joint.
  7. EXPECTED ACCURACY under uniform tie-breaking, so a model that ties at the top is priced at
     1/|argmax| rather than by a coin.

    python _check404_family.py
"""
from __future__ import annotations

import re
from pathlib import Path

import _audit404_family as A

SRC = Path("_audit404_family.py")

# after `made` is placed, BOTH `other` and `third` are legal and only the order decides - which
# is exactly the population this arena is about
DESIGNED = '''
def f(arg):
    made = arg + 1
    other = arg + 2
    third = arg + 3
    used = made + other + third
    return used
'''


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    bodies = A.rows_of(DESIGNED)
    if len(bodies) != 1 or len(bodies[0]) != 5:
        return [f"0. the designed body is {bodies and len(bodies[0])} lines, not 5"]
    body = bodies[0]

    # 5: the population - at t=1 both `other` and `used`... `used` needs `other`, so only two of
    # the three are legal, and the true next (`other`) is among them
    st = list(A.states_of(body, set(), 8, False, 2))
    if not st:
        f.append("5. the designed state is not in the population - two safe candidates with the "
                 "true next among them is exactly what this arena is")
    else:
        xs, y = st[0]
        if y != 0:
            f.append(f"5. the target index is {y}, not 0 - the candidates are listed in true "
                     f"order, so the true next is the first of them")
        if len(xs) < 2:
            f.append("5. a state with fewer than two safe candidates entered the population")
    if "if len(safe_ix) < min_safe or 0 not in safe_ix:" not in src:
        f.append("5. states whose true next line is itself illegal are not skipped")

    # 3: channel A never reads the ids
    bodyA = re.search(r"^def feats_A\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    if "[4]" in bodyA or "ids" in bodyA.replace("_ids", ""):
        f.append("3. channel A reads the identifiers of the line")
    if "_l, s, d, ty, _ids = cand" not in bodyA:
        f.append("3. channel A's unpacking changed - the ids field must stay unread")

    # 4: the vocabulary is cross-file
    bodyV = re.search(r"^def build_vocab\(.*?(?=\ndef )", src, re.S | re.M).group(0)
    if "seen[w] += 1" not in bodyV or "n >= k" not in bodyV:
        f.append("4. the vocabulary is not 'a word seen in at least k different files'")
    if "ws = set()" not in bodyV:
        f.append("4. a word repeated inside ONE file counts more than once, so one tape's own "
                 "content can enter the vocabulary")

    # 1 + 6: held out by file, and the ablation on those same states
    if "train_f, test_f = files[:half], files[half:]" not in src:
        f.append("1. the split is not by file")
    if "te1 = collect(test_f," not in src:
        f.append("6. the ablation is not read on the held-out states")

    # 2: the void check is uncapped
    if "for xs, _y in states_of(body, vocab, args.vocab_dim, False, 1):" not in src:
        f.append("2. the void check is read off a capped collection again")

    # 7
    if "hit += (1.0 / len(best)) if y in best else 0.0" not in src:
        f.append("7. accuracy is not 1/|argmax| under a tie")
    return f


MUTANTS = (
    ("the fit is scored on the files it saw",
     "    train_f, test_f = files[:half], files[half:]",
     "    train_f, test_f = files[:half], files[:half]", "1."),
    ("the void check goes back to the capped collections",
     "            for xs, _y in states_of(body, vocab, args.vocab_dim, False, 1):",
     "            for xs, _y in states_of(body, vocab, args.vocab_dim, False, 9):", "2."),
    ("channel A is handed the identifiers",
     "    _l, s, d, ty, _ids = cand", "    _l, s, d, ty, ids4 = cand", "3."),
    ("a word repeated in one file enters the vocabulary",
     "        ws = set()", "        ws = []", "4."),
    ("states whose true next is illegal are kept",
     "        if len(safe_ix) < min_safe or 0 not in safe_ix:",
     "        if len(safe_ix) < min_safe:", "5."),
    ("the ablation is read on the training states",
     "    te1 = collect(test_f,", "    te1 = collect(train_f,", "6."),
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
