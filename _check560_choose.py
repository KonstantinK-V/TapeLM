"""Check of 560: chooser Q[counts], no token in key.
    python _check560_choose.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit560_choose.py")
def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Φ")
    if 'key = (nbin, peaked, wbin)' not in src:
        f.append("1. key must be counts only")
    if "u_s <= 0.05" not in src:
        f.append("2. VOID when no STAR unique")
    if "d557 > 0.05 and drnd > 0.05" not in src:
        f.append("2. GATE vs 557 and rnd")
    if "policy_557" not in src:
        f.append("1. always-557 rival missing")
    if "rows_null" not in src:
        f.append("1. null shuffled gold missing")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src or "c != held" in src:
        f.append("1. direct READ/teacher subtraction broken")
    if "True if addr_s == held else" not in src or "addr_s == held or" in src:
        f.append("1. STAR direct answer dropped")
    return f
MUTANTS = (
    ("void without u_S",
     "    void = n_te < 40 or u_s <= 0.05",
     "    void = n_te < 40",
     "2."),
    ("gate only rnd",
     "    gate = (not void) and d557 > 0.05 and drnd > 0.05",
     "    gate = (not void) and drnd > 0.05",
     "2."),
    ("token in key",
     "    key = (nbin, peaked, wbin)",
     "    key = (nbin, peaked, wbin, v)",
     "1."),
)
def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        n = src.count(old)
        if n != 1:
            fails.append(f"MUTATION {tag} ({name}): anchor {n}")
            continue
        got = props(src=src.replace(old, new, 1))
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): not caught")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0
if __name__ == "__main__":
    raise SystemExit(main())
