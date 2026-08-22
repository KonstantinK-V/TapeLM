"""Check of 414: honest construction vs GPT-local on THIS tape.

  1. DESIGNED: stay=0, left=0, window=1 on trees grow ___ they tasty.
  2. THE HOLE IS NOT A KEY.
  3. INDEX IS LEFT/RIGHT, not filler w.
  4. WIKI GATE is vs stay AND left (GPT-local), not vs random window word.

    python _check414_construct.py
"""
from __future__ import annotations

from pathlib import Path

import _audit414_construct as M

SRC = Path("_audit414_construct.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    T, hide, r = M.designed_probe()
    if T["toks"][hide] != "APPLES" or T["owner"][hide] != 4:
        f.append("1. designed hole is not APPLES on the trees-grow line")
    if r["stay"] != 0 or r["left"] != 0:
        f.append(f"1. GPT-local not zero on designed: stay={r['stay']} left={r['left']}")
    if r["window"] != 1:
        f.append(f"1. window did not find APPLES on designed: window={r['window']}")
    if "tasty" not in r["keys"]:
        f.append("1. tasty not in window keys of the designed hole")
    if "APPLES" in r["keys"]:
        f.append("2. the hole APPLES entered the keys")

    if "if v == hole or v in seen:" not in src:
        f.append("2. window_keys does not exclude the hole")
    if "for tok in set(L) | set(R):" not in src:
        f.append("3. by_ctx_of is not left/right")
    if "_w, L, R" not in src:
        f.append("3. filler w is not unpacked aside")

    if 'rep["window_minus_stay"] > 0.05 and rep["window_minus_left"] > 0.05' not in src:
        f.append("4. wiki gate is not vs stay AND left")
    if "window_minus_random" in src and "window_minus_left" not in src:
        f.append("4. gate still reads random instead of left")
    return f


MUTANTS = (
    ("designed stays GPT-local",
     '    hide = next(s for s, t in enumerate(T["toks"])\n'
     '                if t == "APPLES" and T["owner"][s] == 4)',
     '    hide = next(s for s, t in enumerate(T["toks"])\n'
     '                if t == "APPLES" and T["owner"][s] == 0)',
     "1."),
    ("hole enters keys",
     "        if v == hole or v in seen:",
     "        if v in seen:",
     "2."),
    ("index includes filler w",
     "        for tok in set(L) | set(R):",
     "        for tok in set(L) | set(R) | {_w}:",
     "3."),
    ("gate drops left",
     '    gate = (not void) and rep["window_minus_stay"] > 0.05 and '
     'rep["window_minus_left"] > 0.05',
     '    gate = (not void) and rep["window_minus_stay"] > 0.05',
     "4."),
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
            fails.append(f"MUTATION {tag} ({name}): check did not fire")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
