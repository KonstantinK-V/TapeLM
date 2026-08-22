"""Check of 410: pin-pay. Hop 2 from the pin vs from the question."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import _audit390_address as A
import _audit410_pinpay as M

SRC = Path("_audit410_pinpay.py")


def _pad(k):
    return " " + " ".join(f"p{k}x{j}" for j in range(24))


DESIGNED = [
    "the XARWIN team won the opening match of the season" + _pad(0),
    "the XARWIN team lost the closing match of the season" + _pad(1),
    "also XARWIN here extra words for a second XARWIN place" + _pad(2),
    "also XARWIN here extra words for a second XARWIN place" + _pad(3),
    "the OTHER club played a different sport on a field" + _pad(4),
    "the OTHER club played another season on a field here" + _pad(5),
]


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T = A.build_tape(DESIGNED, 3, 1)
    if not T["places"]:
        return ["0. designed tape has no places"]
    toks = T["toks"]
    hide = next(s for s, t in enumerate(toks)
                if t == "XARWIN" and T["owner"][s] == 1)
    qpid = T["place_of"][hide]
    pin = M.supplier(T, hide, qpid, "XARWIN")
    if pin is None or pin == qpid:
        f.append("4. designed pin did not leave the question place — second XARWIN frame missing")
    other = M.supplier(T, hide, qpid, "OTHER")
    if other is None or other == pin:
        f.append("4. OTHER did not pin a different place")
    args = Namespace(places=8, topm=8, max_q=40, _rng=None)
    rep = M.measure(T, args)
    if rep is None:
        return ["0. designed tape has no next-hole pairs"]
    if rep["working_cells"] != 0:
        f.append("6. working cells are not 0")
    if not (0.0 <= rep["question_hit"] <= 1.0 and 0.0 <= rep["pin_hit"] <= 1.0):
        f.append("0. hits are not shares")
    if "qprof = Counter(toks[x] for x in slots if x != s)" not in src:
        f.append("1. qprof is not the next hole's place without the hole")
    if 'drop = set(T["on_line"][owner[s]])' not in src or "drop.discard(pid)" not in src:
        f.append("2. same-line places of the NEXT hole are not dropped")
    if "order = [here] + [j for j in order if j != here]" not in src:
        f.append("3. standing does not prepend here")
    if "if here not in drop:" not in src:
        f.append("3. a pin on the next hole's line is not withheld")
    if "other = [p for p in found if p != qpid]" not in src:
        f.append("4. supplier does not prefer a place other than the question")
    if 'void = rep["moved"] <= 0.05' not in src:
        f.append("4. VOID is not share(pin != question) <= 0.05")
    if 'gate = (not void) and rep["gain"] > 0.05' not in src:
        f.append("5. gate is not gain > 0.05 on all next-holes after VOID")
    if '"working_cells": 0' not in src:
        f.append("6. working_cells is not fixed at 0")
    return f


MUTANTS = (
    ("qprof includes the next hole",
     "    qprof = Counter(toks[x] for x in slots if x != s)",
     "    qprof = Counter(toks[x] for x in slots)",
     "1."),
    ("standing does not read here",
     "        order = [here] + [j for j in order if j != here]",
     "        order = [j for j in order if j != here]",
     "3."),
    ("pin never leaves the question",
     "    other = [p for p in found if p != qpid]",
     "    other = [qpid] if found else []",
     "4."),
    ("VOID reads the gain",
     '    void = rep["moved"] <= 0.05',
     '    void = rep["gain"] <= 0.05',
     "4."),
    ("gate ignores VOID and the bar",
     '    gate = (not void) and rep["gain"] > 0.05',
     "    gate = True",
     "5."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        saved = dict(M.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), M.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            M.__dict__.clear()
            M.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): the failure was re-introduced and check "
                         f"{tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
