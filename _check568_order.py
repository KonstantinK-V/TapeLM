"""Check of 568: order-Q beats random+null; no word/held; 567 stays.
    python _check568_order.py
"""
from __future__ import annotations
from pathlib import Path
SRC = Path("_audit568_order.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "torch" in src:
        f.append("1. Phi")
    if "return (band, n, int(ov > 0))" not in src:
        f.append("1. feat missing")
    if "held" in src and "feat(" in src:
        # feat must not take held / word identity
        for line in src.splitlines():
            if line.strip().startswith("def feat(") and ("held" in line or "v," in line):
                f.append("1. feat leaks held/word")
                break
    if "null=True" not in src and "null=True)" not in src:
        if "null=True" not in src:
            f.append("1. null arm missing")
    if 'if held in place:' not in src:
        f.append("1. direct READ not removed before order task")
    if "c != held" in src:
        f.append("1. teacher subtracted from candidate list")
    if "nte < 20" not in src:
        f.append("2. VOID on test pairs")
    if "h_q - h_u > 0.05" not in src or "h_q - h_n > 0.05" not in src:
        f.append("2. GATE Q vs rand+null")
    for line in src.splitlines():
        if "gate =" in line and "h_r" in line:
            f.append("2. rank required in GATE")
            break
    return f


MUTANTS = (
    ("no null",
     "    Qn = train(tr, random.Random(args.seed + 2), null=True)",
     "    Qn = train(tr, random.Random(args.seed + 2), null=False)",
     "1."),
    ("void soft",
     "    void = nte < 20",
     "    void = nte < 0",
     "2."),
    ("gate only rand",
     "    gate = (not void) and (h_q - h_u > 0.05) and (h_q - h_n > 0.05)",
     "    gate = (not void) and (h_q - h_u > 0.05)",
     "2."),
    ("gate needs rank",
     "    gate = (not void) and (h_q - h_u > 0.05) and (h_q - h_n > 0.05)",
     "    gate = (not void) and (h_q - h_u > 0.05) and (h_q - h_n > 0.05) and (h_q - h_r > 0.05)",
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
