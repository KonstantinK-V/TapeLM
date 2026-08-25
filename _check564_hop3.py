"""Check of 564: PIN hop3 from hop2 frame; VOID if rare; no Phi/hit-gate.
    python _check564_hop3.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit564_hop3.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src or "PickNet" in src:
        f.append("1. Phi leaked")
    if "W[stood] = addr" not in src:
        f.append("1. hop2 PIN write missing")
    if "W[stood2] = addr3" not in src:
        f.append("1. hop3 PIN write missing")
    if "stand_read(g, by, addr_w3" not in src:
        f.append("1. hop3 not from W mark")
    if "len(cand3) != 1" not in src:
        f.append("1. hop3 unique gate missing")
    if "void = n_h3 < 40" not in src:
        f.append("2. VOID not on n_hop3")
    if "from_w >= 1.0 and from_w3 >= 1.0" not in src:
        f.append("2. GATE wire missing")
    for line in src.splitlines():
        if "gate =" in line and ("hit2" in line or "hit3" in line or "hit " in line):
            f.append("2. hit in GATE")
            break
    if "agree >= 0.80" in src:
        f.append("1. 562 agree leaked")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src:
        f.append("1. direct READ not removed before PIN")
    if "if hit2:" not in src:
        f.append("1. hop3 continues after answer at hop2")
    if "{held" in src:
        f.append("1. teacher subtracted from candidate list")
    return f


MUTANTS = (
    ("no hop3 write",
     "    W[stood2] = addr3",
     "    _ = addr3",
     "1."),
    ("hop3 skips W",
     "    hit3, _, _ = stand_read(g, by, addr_w3, env_m, held)",
     "    hit3, _, _ = stand_read(g, by, addr3, env_m, held)",
     "1."),
    ("void on pin",
     "    void = n_h3 < 40",
     "    void = n_pin < 40",
     "2."),
    ("gate soft",
     "    gate = (not void) and from_w >= 1.0 and from_w3 >= 1.0",
     "    gate = (not void) and from_w >= 0.0",
     "2."),
    ("hit in gate",
     "    gate = (not void) and from_w >= 1.0 and from_w3 >= 1.0",
     "    gate = (not void) and from_w >= 1.0 and from_w3 >= 1.0 and hit3 >= 0.0",
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
