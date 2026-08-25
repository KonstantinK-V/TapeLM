"""Check of 544: Phi sees counts only, the null is real, the probe is unseen."""
from __future__ import annotations

from pathlib import Path

import _audit544_phi as M

SRC = Path("_audit544_phi.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit485_hunt import (" not in src:
        f.append("1. 485 hunt helpers reuse missing")
    if "from _audit542_curric import rand_unique" not in src:
        f.append("1. 542 baseline reuse missing")
    if "hash(" in src or "ord(" in src:
        f.append("1. a word identity enters the feature vector")
    if "n_k / max(n_s, 1) / 4.0, min(n_s, 12) / 12.0, min(n_k, 20) / 20.0]" not in src:
        f.append("1. the counts-only feature vector missing")
    if "torch.nn.Linear(FDIM, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)" not in src:
        f.append("1. Phi must stay tiny (FDIM->16->1)")
    if "r2 = self.buf[rng.randrange(len(self.buf))]" not in src:
        f.append("1. the null must train on a reward drawn from the past")
    if "scorer.update(P, g, 1.0 if hit else -0.08, rng)" not in src:
        f.append("1. the 542 reward stream (+1/-0.08) missing")
    if src.count("build_window(lines, rng_win") != 3:
        f.append("1. train/eval/probe must all draw windows with rng_win")
    if "sub = [P for P in places if pre(P, g) not in K]" not in src:
        f.append("1. the probe must be restricted to UNSEEN designed keys")
    if "CrossEntropy" in src:
        f.append("4. CE leaked")
    if 'rands = {a["eval_rand"] for a in arms.values()}' not in src:
        f.append("3. pairing not verified from the baseline itself")
    gate = src[src.find("    gate = "):src.find("    rec = dict")]
    if "lift_B - lift_C > 0.05" not in gate:
        f.append("2. GATE must beat the shuffled-reward null")
    if "lift_B > lift_A - 0.05" not in gate:
        f.append("2. GATE must not lose to the table on seen keys")
    if "d_probe > 0.05" not in gate:
        f.append("2. GATE must include the unseen-key probe")
    if 'void = (not paired) or arms["B_phi"]["n_eval"] < 200 or pr["probe_n"] < 100' not in src:
        f.append("2. VOID on unpaired / few windows / starved probe missing")
    return f


MUTANTS = (
    ("a word slips into the features",
     "min(n_s, 12) / 12.0, min(n_k, 20) / 20.0]",
     "min(n_s, 12) / 12.0, min(n_k, 20) / 20.0, float(hash(P) % 97)]",
     "1."),
    ("the null trains on the true reward",
     "        r2 = self.buf[rng.randrange(len(self.buf))]",
     "        r2 = r",
     "1."),
    ("the probe leaks seen keys",
     "        sub = [P for P in places if pre(P, g) not in K]",
     "        sub = list(places)",
     "1."),
    ("the probe drops out of the gate",
     "    gate = (not void) and lift_B - lift_C > 0.05 and lift_B > lift_A - 0.05 and d_probe > 0.05",
     "    gate = (not void) and lift_B - lift_C > 0.05 and lift_B > lift_A - 0.05",
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
