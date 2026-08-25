"""Check of 565: W[env] key; GATE agree_e; 562 untouched.
    python _check565_envkey.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit565_envkey.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Phi")
    if "W_e[ek] = addr" not in src:
        f.append("1. W_e write missing")
    if "W_v[v] = addr" not in src:
        f.append("1. W_v control missing")
    if 'rec["agree_e"] = int(W_e[ek] == addr)' not in src:
        f.append("1. agree_e missing")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src:
        f.append("1. direct READ not removed before PIN")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "void = n_e < 40" not in src:
        f.append("2. VOID on n_reuse_e")
    if "ae >= 0.80" not in src:
        f.append("2. GATE agree_e")
    for line in src.splitlines():
        if "gate =" in line and ("agree_v" in line or "av" in line or "hit" in line):
            f.append("2. agree_v/hit in GATE")
            break
    return f


MUTANTS = (
    ("no env write",
     "            W_e[ek] = addr",
     "            _ = ek",
     "1."),
    ("void soft",
     "    void = n_e < 40",
     "    void = n_e < 0",
     "2."),
    ("gate on v",
     "    gate = (not void) and ae >= 0.80",
     "    gate = (not void) and av >= 0.80",
     "2."),
    ("gate soft",
     "    gate = (not void) and ae >= 0.80",
     "    gate = (not void) and ae >= 0.0",
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
