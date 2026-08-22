"""Check of 409: the register is a PLACE, not a word and not a new cell."""
from __future__ import annotations

from pathlib import Path

import _audit409_pin as A

SRC = Path("_audit409_pin.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    today, pin = A.measure(False), A.measure(True)
    if today["hop2_moved"] or not today["truth"]["here_eq_question"]:
        f.append("1. TODAY already moves the register — then the pin is not the missing piece")
    if today["truth"]["hop2_origin"] != today["truth"]["here"]:
        f.append("1. hop 2 origin is not the register")
    if not pin["hop2_moved"]:
        f.append("2. PIN did not move hop 2 off the question place")
    if pin["truth"]["here_is_string"] or not pin["truth"]["here_is_place"]:
        f.append("2. PIN put a letter in the register")
    if pin["truth"]["here"] is None:
        f.append("2. PIN found no supplier place for the true pick")
    if pin["truth"]["working_cells"] != 0 or not pin["truth"]["corpus_untouched"]:
        f.append("3. PIN copied a value or mutated the corpus — it must only set here")
    if not pin["wrong_pin_differs"]:
        f.append("4. OTHER and XARWIN pinned the same place")
    if not pin["no_letter_in_register"]:
        f.append("5. the register held a string")
    if 'w["here"] = q["qpid"]' not in src:
        f.append("1. TODAY does not leave here at the question")
    if "def supplier(" not in src:
        f.append("2. supplier() is missing")
    return f


MUTANTS = (
    ("today already pins",
     '    w["here"] = q["qpid"]          # register never moves',
     '    w["here"] = 99',
     "1."),
    ("pin copies the word into here",
     '    w["here"] = pin if pin is not None else q["qpid"]',
     '    w["here"] = said',
     "2."),
    ("pin appends a working cell",
     '            "working_cells": 0,',
     '            "working_cells": 1,',
     "3."),
    ("wrong pick pins the true place",
     "    other = [p for p in found if p != qpid]",
     "    other = [qpid]",
     "4."),
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
            fails.append(f"MUTATION {tag} ({name}): check did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
