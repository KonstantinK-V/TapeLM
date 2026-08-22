"""Check of 502: frozen 440 think_place on 501 unique-next arena. No Q."""
from __future__ import annotations

from pathlib import Path

import _audit502_compose as M

SRC = Path("_audit502_compose.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit440_compose import think_place" not in src:
        f.append("1. frozen 440 think_place missing")
    if "len(cands) != 1" not in src:
        f.append("1. 501 unique-next filter missing")
    if "pick_by_q" in src or "def train(" in src:
        f.append("1. Q leaked")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "hop_rate == 1.0" not in gate:
        f.append("2. GATE missing hop wiring")
    if "pin_rate > 0.05" not in gate:
        f.append("2. GATE missing pin floor")
    if "pin_rate > pin_rand + 0.05" not in gate:
        f.append("2. GATE missing pin vs random")
    if 'return WIKI, "wiki", 80' not in src:
        f.append("3. wiki auto-pick missing")
    if "import torch" in src or "PickNet" in src:
        f.append("4. Phi leaked")
    return f


MUTANTS = (
    ("no random beat",
     "            and (pin_rate > pin_rand + 0.05))",
     "            and (pin_rate > 0.05))",
     "2."),
    ("hop unwired",
     "    gate = ((not void) and (hop_rate == 1.0)",
     "    gate = ((not void) and True",
     "2."),
    ("Q leaked",
     "from _audit440_compose import think_place",
     "from _audit440_compose import think_place\nfrom _audit485_hunt import pick_by_q",
     "1."),
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
