"""Check of 530: oracle-stop on zeros state. Default W=250. Policy does not see held."""
from __future__ import annotations

from pathlib import Path

import _audit530_stop as M

SRC = Path("_audit530_stop.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _audit527_learn import majority, v1_nodes" not in src:
        f.append("1. 527 v1_nodes reuse missing")
    if "from _audit528_step import cover, mean_ep, run_ep, trials" not in src:
        f.append("1. 528 step helpers missing")
    if "default=250" not in src:
        f.append("1. default window 250 missing")
    if "def rest_novel(" not in src:
        f.append("1. rest_novel oracle helper missing")
    if "should_stop = not rest_novel(nodes, i, seen, held, maj)" not in src:
        f.append("1. oracle uses rest_novel in training loop")
    if "Q[(band, z, \"go\")]" not in src or "Q[(band, z, \"stop\")]" not in src:
        f.append("1. Q[(band, zeros, go|stop)] missing")
    if "min(zeros, 2)" not in src:
        f.append("1. zeros capped at 2 missing")
    if "held" in src[src.find("def run_learned"):src.find("def collect")]:
        # run_learned uses held for cover only, not Q lookup — ok
        pass
    if "import torch" in src or "CrossEntropy" in src:
        f.append("4. CE leaked")
    gate = src[src.find("gate ="):src.find("tag =")]
    if 'lm["cover"] > a1["cover"] + 0.05' not in gate:
        f.append("2. GATE missing cover > hop1 + 0.05")
    if 'a1["hops"] + 0.5 < lm["hops"] < ag["hops"] - 0.5' not in gate:
        f.append("2. GATE missing hop1+0.5 < hops < allgo-0.5")
    if 'lh["hops"] < 1.5' not in gate:
        f.append("2. GATE missing high hops < 1.5")
    if "oracle_stop=ora" not in src:
        f.append("2. oracle_stop rate missing")
    if "not (0.1 < ora < 0.9)" not in src:
        f.append("2. VOID on oracle outside (0.1, 0.9) missing")
    if 'f"{args.seed}_{tag}_{len(lines)}"' not in src:
        f.append("3. JSON key must split seed/corpus/window")
    return f


MUTANTS = (
    ("gate cover only",
     '    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and (\n'
     '        a1["hops"] + 0.5 < lm["hops"] < ag["hops"] - 0.5) and (lh["hops"] < 1.5)',
     '    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05)',
     "2."),
    ("no rest_novel",
     "                should_stop = not rest_novel(nodes, i, seen, held, maj)",
     "                should_stop = False",
     "1."),
    ("no oracle void",
     "    void = lm[\"n\"] < 20 or not (0.1 < ora < 0.9)",
     "    void = lm[\"n\"] < 20",
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
