"""Check of 563: W[stood]=addr wire; hop2 from mark; refuse silent.
    python _check563_pinw.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit563_pinw.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src or "Q[" in src:
        f.append("1. Q/Φ leak")
    if "agree >= 0.80" in src:
        f.append("1. 562 agree gate leaked")
    if "W[stood] = addr" not in src:
        f.append("1. write W[stood]=addr missing")
    if "W[v] =" in src:
        f.append("1. still keys W by word v")
    if "stand_read(g, by, addr_w" not in src:
        f.append("1. hop2 not from W mark")
    if "len(cand) != 1:" not in src:
        f.append("1. refuse branch missing")
    if 'return dict(pin=0, refuse=1, wrote=0' not in src:
        f.append("1. refuse still writes")
    if 'if held in fr_p:' not in src or '"read_hit"' not in src:
        f.append("1. direct READ not removed before PIN")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "void = n_pin < 40" not in src:
        f.append("2. VOID on n_pin")
    if "from_w >= 1.0 and refuse_wrote <= 0.0" not in src:
        f.append("2. GATE wire missing")
    for line in src.splitlines():
        if "gate =" in line and "hit" in line:
            f.append("2. hit in GATE")
            break
    return f


MUTANTS = (
    ("key by v",
     "    W[stood] = addr",
     "    W[v] = addr",
     "1."),
    ("hop2 skips W",
     "    hit, _ = stand_read(g, by, addr_w, env_m, held)",
     "    hit, _ = stand_read(g, by, addr, env_m, held)",
     "1."),
    ("refuse writes",
     '        return dict(pin=0, refuse=1, wrote=0, from_w=0, hit=-1,\n'
     '                    stood=int(stood), n_cand=len(cand)), None',
     '        W[stood] = cand[0] if cand else None\n'
     '        return dict(pin=0, refuse=1, wrote=1, from_w=0, hit=-1,\n'
     '                    stood=int(stood), n_cand=len(cand)), None',
     "1."),
    ("void soft",
     "    void = n_pin < 40",
     "    void = n_pin < 0",
     "2."),
    ("gate soft",
     "    gate = (not void) and from_w >= 1.0 and refuse_wrote <= 0.0",
     "    gate = (not void) and from_w >= 0.0",
     "2."),
    ("teacher-subtracted candidates",
     "    cand = [c for c in fr_p if c in mid_set and c != v]",
     "    cand = [c for c in fr_p if c in mid_set and c != held and c != v]",
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
