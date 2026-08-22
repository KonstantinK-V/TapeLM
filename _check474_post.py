"""Check of 474: post-action S, no cls in pick."""
from __future__ import annotations

from pathlib import Path

import _audit474_post as M

SRC = Path("_audit474_post.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "def pick_post(" not in src or "START" not in src:
        f.append("1. post pick missing")
    body = src.split("def pick_post(")[1].split("def credit(")[0] if "def pick_post(" in src else ""
    if "observe(" in body or "opened(" in body or "filter_keys" in body:
        f.append("1. lookahead in pick_post")
    if 'hB["pin"] == 0.0' not in src or "pB[\"pin\"] < 0.90" not in src:
        f.append("2. GATE missing")
    if "import torch" in src or "wikitext" in src:
        f.append("3. leak")
    return f


MUTANTS = (
    ("lookahead",
     "    if table.get(S, 0.0) <= 0:\n        return None\n    return rng.choice(opts)",
     "    if table.get(S, 0.0) <= 0:\n        return None\n"
     "    return max(opts, key=lambda r: 1 if opened(r[1], g['value'][r[2]], g, r[3], cands) else 0)",
     "1."),
    ("allow 1.0",
     "            and (pB[\"pin\"] < 0.90)",
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
