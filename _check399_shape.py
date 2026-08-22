"""Check of 399's shape feature. No torch, no corpus - the corpus is two designed sources.

399 exists to answer one question: on the 23% where names tie, is there evidence that is NOT the
name count? So the feature's defining property is not accuracy, it is BLINDNESS TO IDENTITY - and
that is checkable exactly, by renaming every symbol and requiring every number to be unchanged.

  1. EQUIVARIANCE. Two sources identical in structure and different in every identifier must give
     the SAME per-line type sets and the SAME shape scores. If one identifier reaches the
     feature, "shape" is names in disguise and the whole comparison is void.
  2. THE LEAK, ON SHAPE. A scope's types are read with the pooled line REMOVED: a line's own node
     types are otherwise evidence for its own scope, which is 398's `bound_wo` fault in the other
     currency.
  3. EXPECTED ACCURACY, NOT A COIN. A rival that ties at the top scores 1/|argmax|, exactly, so
     no number moves with the seed and a tying rival is priced honestly rather than by a draw.
  4. THE DECISION POPULATION IS ONLY `amb_live`. A line whose name argmax is unique must not
     enter the tie-break numbers at all, or the 23% would be diluted by the 61% counting already
     answers.
  5. THE SIZE RIVAL EXCLUDES THE POOLED LINE from the true scope's body count - conservative, and
     against the rival that turned out to carry the population.
  6. THE CONTROL IS ON THE FULL POPULATION, every scored line, or "the catalogue was not paid
     for" is a claim about the 23% alone.
  7. THE GATE NEEDS ALL THREE RIVALS BEATEN. Beating a coin while size does better is not
     evidence about the line.

    python _check399_shape.py
"""
from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path

import _audit398_scope as S
import _audit399_shape as A

SRC = Path("_audit399_shape.py")

DESIGNED = '''
def alpha(one):
    two = one + 1
    for three in range(two):
        two = two + three
    return two


def beta(four):
    five = four
    return five
'''
# the same file with every identifier replaced - structure identical, names disjoint
RENAMED = (DESIGNED.replace("alpha", "zulu").replace("one", "whisky").replace("two", "xray")
           .replace("three", "yankee").replace("beta", "victor").replace("four", "uniform")
           .replace("five", "tango"))


def shape_vec(src, ln_of):
    sc = S.scopes_of(src)
    line_t, body_t = A.types_of(src, sc["owner"])
    ln = ln_of(sc, line_t)
    nf = len(sc["funcs"])
    tsets = {i: A.types_wo(body_t, i, ln) for i in range(nf)}
    have = {}
    for i in range(nf):
        for t in tsets[i]:
            have[t] = have.get(t, 0) + 1
    lt = line_t.get(ln, set())
    return [sum(1.0 / max(1, have[t]) for t in lt if t in tsets[i]) for i in range(nf)], lt, ln


def first_for(sc, line_t):
    return min(l for l, ts in line_t.items() if "For" in ts)


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []

    # 1: equivariance - the centrepiece
    v_a, t_a, ln_a = shape_vec(DESIGNED, first_for)
    v_b, t_b, ln_b = shape_vec(RENAMED, first_for)
    if t_a != t_b or v_a != v_b or ln_a != ln_b:
        f.append(f"1. renaming every identifier moved the feature: line {ln_a} vs {ln_b}, types "
                 f"{sorted(t_a)} vs {sorted(t_b)}, scores {v_a} vs {v_b} - `shape` is reading "
                 f"identity, so it is the name count in disguise")
    if any(x in str(sorted(t_a)) for x in ("alpha", "one", "two", "three")):
        f.append(f"1. an identifier is inside the type set: {sorted(t_a)}")

    # 2: the leak on shape
    sc = S.scopes_of(DESIGNED)
    line_t, body_t = A.types_of(DESIGNED, sc["owner"])
    fl = first_for(sc, line_t)
    i = sc["owner"][fl]
    if "For" not in A.types_wo(body_t, i, -1):
        f.append("2. the designed case is not designed: the loop line is not in its scope")
    if "For" in A.types_wo(body_t, i, fl):
        f.append("2. a scope keeps the node types of the pooled line - the line is evidence for "
                 "its own scope (398's bound_wo fault, in the other currency)")

    # 3: expected accuracy
    got = (A.expected_acc([1.0, 1.0, 0.0], 0, [0, 1, 2]),
           A.expected_acc([1.0, 1.0, 0.0], 2, [0, 1, 2]),
           A.expected_acc([2.0, 1.0, 0.0], 0, [0, 1, 2]))
    if got != (0.5, 0.0, 1.0):
        f.append(f"3. expected accuracy reads {got}, expected (0.5, 0.0, 1.0) - a rival that "
                 f"ties must be priced at 1/|argmax|, not by a coin")

    # 4 + 5 + 6: the population, the size rival and the control, read off the source
    if 'if len(tied) < 2 or top <= 0.0:' not in src or 'c["amb"] += 1' not in src:
        f.append("4. the decision population is not restricted to a live tie")
    if 'c["n"] += 1' not in src.split('if true_i not in tied:')[-1][:200]:
        f.append("4. lines whose true scope is not among the tied still enter the numbers")
    if 's_size = [float(n_body[i] - (1 if i == true_i else 0))' not in src:
        f.append("5. the size rival does not take the pooled line out of the true scope, so the "
                 "confound is given a free point")
    if 'c["full_n"] += 1' not in src or 'c["full_name"] +=' not in src:
        f.append("6. the control is not counted on every scored line")

    # 7: all three rivals in the gate
    g = re.search(r"gate = \((?:.|\n)*?\)\n", src)
    body = g.group(0) if g else ""
    for k in ("shape_minus_random", "shape_minus_rawname", "shape_minus_size"):
        if k not in body:
            f.append(f"7. {k} is not in the gate - beating a coin while another rival does "
                     f"better is not evidence about the line")
    return f


MUTANTS = (
    ("the feature reads identifiers",
     '        if ln is not None:\n            line_t[ln].add(type(node).__name__)',
     '        if ln is not None:\n            line_t[ln].add(type(node).__name__)\n'
     '            line_t[ln].add(getattr(node, "id", ""))', "1."),
    ("a scope keeps the pooled line's types",
     '    return {t for t, lns in body_t.get(i, {}).items() if lns - {drop_line}}',
     '    return set(body_t.get(i, {}))', "2."),
    ("a tying rival is priced by a coin",
     "    return (1.0 / len(best)) if truth_i in best else 0.0",
     "    return 1.0 if truth_i in best else 0.0", "3."),
    ("unique-argmax lines enter the tie-break",
     "            if len(tied) < 2 or top <= 0.0:",
     "            if False:", "4."),
    ("the size rival keeps its free point",
     "            s_size = [float(n_body[i] - (1 if i == true_i else 0)) for i in range(nf)]",
     "            s_size = [float(n_body[i]) for i in range(nf)]", "5."),
    ("the control counts only the ambiguous lines",
     '            c["full_n"] += 1', '            c["full_n"] += 0', "6."),
    ("the gate drops the size rival",
     '    gate = (rep["shape_minus_random"] > 0.05 and rep["shape_minus_rawname"] > 0.05\n'
     '            and rep["shape_minus_size"] > 0.05)',
     '    gate = (rep["shape_minus_random"] > 0.05 and rep["shape_minus_rawname"] > 0.05)', "7."),
)


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    fails = props()
    for name, old, new, tag in MUTANTS:
        if src.count(old) != 1:
            fails.append(f"MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times")
            continue
        saved = dict(A.__dict__)
        mutated = src.replace(old, new, 1)
        try:
            exec(compile(mutated, "<mutant>", "exec"), A.__dict__)
            got = props(src=mutated)
        except Exception as e:
            got = [f"{tag} the mutant raised {type(e).__name__}"]
        finally:
            A.__dict__.clear()
            A.__dict__.update(saved)
        if not any(g.startswith(tag) for g in got):
            fails.append(f"MUTATION {tag} ({name}): the failure was re-introduced and check "
                         f"{tag} did not fire - it is a comment, not a check")
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else
          f"all properties hold, and all {len(MUTANTS)} re-introduced failures were caught")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
