"""Check repaired 542: novel tapes, exact credit null, majority rival."""
from __future__ import annotations

from pathlib import Path

import _audit542_curric as M

SRC = Path("_audit542_curric.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "train_pool, test_pool = lines[:cut], lines[cut:]" not in src:
        f.append("1. corpus not split before tapes")
    if "range(0, len(pool) - length + 1, length)" not in src:
        f.append("1. tape windows overlap")
    if "train_tapes = build_tapes(train_windows, args)" not in src:
        f.append("1. train tapes not built once/shared")
    if "test_tapes = build_tapes(test_windows, args)" not in src:
        f.append("1. held-out tapes missing")
    if "hr += rand_unique(g, rng_rnd, args.budget)" not in src:
        f.append("1. baseline must not consume the policy generator")
    if "random.Random(seed).shuffle(rewards)" not in src:
        f.append("1. exact reward-shuffle null missing")
    if "trace.append((k, reward))" not in src:
        f.append("1. A credit trace missing")
    if 'tr_a["n_touch"] == tr_c["n_touch"]' not in src:
        f.append("1. null touches not matched")
    if "n_touch += 1" not in src:
        f.append("1. touches not counted, so work is not compared")
    if "hm += majority_unique(g, args.budget)" not in src:
        f.append("1. majority place rival missing")
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    if "rands = {a[\"eval_rand\"] for a in arms.values()}" not in src:
        f.append("3. the pairing is not verified from the baseline itself")
    gate = src[src.find("    gate = "):src.find("    rec = dict")]
    if "d_null > 0.05 and d_maj > 0.05" not in gate:
        f.append("2. GATE must beat scratch, null and majority")
    if "(not paired) or len(majs) != 1 or not null_touches_equal" not in src:
        f.append("2. VOID start missing")
    if "n_keys\"] < 3" not in src:
        f.append("2. VOID on fewer than 3 keys missing")
    return f


MUTANTS = (
    ("null arm dropped from the gate",
     "        and d_null > 0.05 and d_maj > 0.05",
     "        and d_maj > 0.05",
     "2."),
    ("same corpus train/eval",
     "    train_pool, test_pool = lines[:cut], lines[cut:]",
     "    train_pool, test_pool = lines, lines",
     "1."),
    ("null rewards not shuffled",
     "    random.Random(seed).shuffle(rewards)",
     "    rewards = rewards",
     "1."),
    ("majority dropped",
     "        hm += majority_unique(g, args.budget)",
     "        hm += 0",
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
