"""Check of 472 standalone: env stop on dead hop."""
from __future__ import annotations

from pathlib import Path

import _audit472_stop as M

SRC = Path("_audit472_stop.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def opened(" not in src or "if not shrunk and op == 0" not in src:
        f.append("1. stop rule missing")
    if "from _audit471_arena import" in src:
        f.append("1. 471 import leaked")
    if 'hB["pin"] == 0.0' not in src:
        f.append("2. GATE missing")
    if "import torch" in src or "wikitext" in src:
        f.append("3. leak")
    return f


MUTANTS = (
    ("no stop",
     "        if not shrunk and op == 0:\n            return hops, False, True, tH, tT",
     "        if False:\n            return hops, False, True, tH, tT",
     "1."),
    ("no QH",
     '            and (hB["pin"] == 0.0)',
     "            and True",
     "2."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {n} times")
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
            fails.append(f"MUTATION {tag} ({name}): re-introduced and check {tag} did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
