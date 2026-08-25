"""Check of 545: every place learns, reach is global counts, order is the reading."""
from __future__ import annotations

from pathlib import Path

import _audit545_wave as M

SRC = Path("_audit545_wave.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit544_phi import RuleScorer, features" not in src:
        f.append("1. 544 local features and rule arm reuse missing")
    if "from _audit542_curric import rand_unique" not in src:
        f.append("1. 542 baseline reuse missing")
    if "y = [1.0 if narrow_next(P, g, kmax=2) is not None else -0.08 for P in ps]" not in src:
        f.append("1. every place of the window must get a target")
    if "X = [wide_features(P, g, gc, 0) for P in ps]" not in src:
        f.append("1. the batch must be built from all places")
    if "return loc + [log1p(gv) / 12.0, log1p(gk) / 12.0," not in src:
        f.append("1. the three global (out-of-window) counts missing")
    if "c.update(ln.split())" not in src:
        f.append("1. global counts must come from the whole line pool")
    if "hash(" in src or "ord(" in src:
        f.append("1. a token identity enters the features")
    if "y = [y[i] for i in idx]" not in src:
        f.append("1. the null must shuffle targets inside the batch")
    if src.count("build_window(lines, rng_win") != 2:
        f.append("1. train and eval must both draw windows with rng_win")
    if "mean_value" not in src or "frac_pos" not in src:
        f.append("3. the negative-by-base-rate readings must be printed")
    if 'rands = {a["eval_rand"] for a in arms.values()}' not in src:
        f.append("3. pairing not verified from the baseline itself")
    if "CrossEntropy" in src:
        f.append("4. CE leaked")
    gate = src[src.find("    gate = "):src.find("    rec = dict")]
    if "lb - ld > 0.05" not in gate:
        f.append("2. GATE must beat the single-count rule")
    if "lb - lc > 0.05" not in gate:
        f.append("2. GATE must beat the shuffled-target null")
    if 'arms["B_wave"]["batch"] < 50' not in src:
        f.append("2. VOID when the batch is not really the whole window")
    return f


MUTANTS = (
    ("only one place per window gets a target",
     "        y = [1.0 if narrow_next(P, g, kmax=2) is not None else -0.08 for P in ps]",
     "        y = [1.0 if narrow_next(ps[0], g, kmax=2) is not None else -0.08]",
     "1."),
    ("reach dropped: no global counts",
     "    return loc + [log1p(gv) / 12.0, log1p(gk) / 12.0,",
     "    return loc + [0.0, 0.0,",
     "1."),
    ("the null stops shuffling",
     "            y = [y[i] for i in idx]",
     "            y = list(y)",
     "1."),
    ("batch-width VOID removed",
     'arms["B_wave"]["batch"] < 50',
     'False',
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
