"""Check of 562: write PIN on W[v]; no Q.
    python _check562_write.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit562_write.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src or "def pick(" in src or "Q[" in src or "q = {" in src:
        f.append("1. Q/Φ leaked")
    if "W[v] = addr_f" not in src:
        f.append("1. write W[v]=addr missing")
    if "if v in W:" not in src:
        f.append("1. reuse read missing")
    if "agree = int(addr_m == addr_f)" not in src:
        f.append("1. agree missing")
    if "n_r < 40" not in src:
        f.append("2. VOID on n_reuse")
    if "agree >= 0.80" not in src:
        f.append("2. GATE agree")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src or "c != held" in src:
        f.append("1. direct READ/teacher subtraction broken")
    return f


MUTANTS = (
    ("no write",
     "            W[v] = addr_f",
     "            _ = addr_f",
     "1."),
    ("void soft",
     "    void = n_r < 40",
     "    void = n_r < 0",
     "2."),
    ("gate soft",
     "    gate = (not void) and agree >= 0.80",
     "    gate = (not void) and agree >= 0.0",
     "2."),
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
