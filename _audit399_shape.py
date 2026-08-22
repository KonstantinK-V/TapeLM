"""THE CEILING OF SHAPE, ON THE 23% WHERE NAMES TIE. Torch-free, before any Phi.

398 measured the object and gave the law for what comes next: on this corpus the counting rival
puts the true scope at the top 96% of the time and picks it uniquely 61% - so a mechanism trained
on the whole file would be that rival with parameters, which is 359's trap entered from the other
side. THE POPULATION IS ONLY `amb_live`: lines where two or more scopes TIE at a non-zero name
score. There, by construction, name overlap cannot decide.

SO THE EVIDENCE MAY NOT BE NAME OVERLAP, and one alternative is declared here BEFORE the run:
SHAPE. The multiset of AST NODE TYPES on the line, against the node types of the candidate's body.
No identifier is read - only `type(node).__name__` - so the feature is equivariant to renaming by
construction, not by an argument. `_check399_shape.py` asserts that no source identifier can reach
it.

    score_shape(scope) = sum over the node TYPES on the line of
                         [type occurs in that scope's body] / (how many scopes of this file have it)

which is the same form as 398's name count with types in place of names, so the two are comparable
and neither is given a shape the other lacks. The undivided version is the declared control
reading, exactly as `--raw` was there.

THE LEAK, INHERITED AND EXTENDED. 398's `bound_wo` already removes the pooled line from the scope
it came from; here the same removal is applied to the scope's TYPES, because a line's own node
types would otherwise be evidence for its own scope. Both are mutation-tested.

THE RIVALS, ALL THREE DECLARED BEFORE THE RUN. Every one is scored by EXPECTED accuracy under
uniform tie-breaking - `1/|argmax| if the truth is in the argmax else 0` - so no rival is helped
or hurt by a coin, and the numbers do not move with the seed.

    random     1/|tied|                     the floor: what a coin gets
    raw name   the UNDIVIDED name overlap   the nearest thing to a name-based tie-break, and the
                                            guard against "shape is names in disguise"
    size       the larger candidate body    THE CONFOUND: a bigger scope contains more node types
                                            for the same reason it contains more of everything

  VOID CHECK, READ FIRST
      `true_in_tie` - the share of ambiguous lines whose true scope is even among the tied. What
      is below that is not a decision anyone can win, and the ceiling would be measured on a
      population that has no answer in it.

  GATE
      shape - random > 0.05, AND shape - raw_name > 0.05, AND shape - size > 0.05.
      PAIRED CONTROL on the FULL population: name-argmax where it is unique, shape where it ties,
      must not fall below the pure name accuracy. The catalogue is not to be paid for the 23%.

  If the gate fails, Phi is not built here: there is no evidence on that population that is not
  the count, and 398's ceiling stands as "a counter of scopes", which is knowledge and not a mind.

    python _audit399_shape.py
    python _audit399_shape.py --raw          # the declared control reading
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _audit398_scope as S

OUT = Path("results/_stage399_shape.json")


def types_of(src, owner):
    """lineno -> the set of AST node TYPE NAMES on it, and scope -> the types of its body lines.

    Only `type(node).__name__` is read. No identifier, no attribute string, no constant value
    reaches this, so renaming every symbol in the file cannot move a single number.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, None
    line_t = defaultdict(set)
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", None)
        if ln is not None:
            line_t[ln].add(type(node).__name__)
    body_t = defaultdict(lambda: defaultdict(set))     # scope -> type -> the lines carrying it
    for ln, ts in line_t.items():
        i = owner.get(ln)
        if i is None:
            continue
        for t in ts:
            body_t[i][t].add(ln)
    return line_t, body_t


def types_wo(body_t, i, drop_line):
    """the types of scope i's body with the pooled line taken out - 398's discipline, on shape"""
    return {t for t, lns in body_t.get(i, {}).items() if lns - {drop_line}}


def expected_acc(scores, truth_i, cands):
    """EXPECTED accuracy under uniform tie-breaking: exact, seed-free, and it prices a rival that
    ties at the top honestly instead of letting a coin decide the number."""
    if not cands:
        return 0.0
    top = max(scores[c] for c in cands)
    best = [c for c in cands if scores[c] == top]
    return (1.0 / len(best)) if truth_i in best else 0.0


def run(files, args, rng):
    c = Counter()
    for p in files:
        try:
            src = Path(p).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        sc = S.scopes_of(src)
        if not sc or len(sc["funcs"]) < args.min_scopes:
            continue
        line_t, body_t = types_of(src, sc["owner"])
        if line_t is None:
            continue
        nf = len(sc["funcs"])
        n_body = Counter(sc["owner"].values())
        lines = [ln for ln in sorted(sc["owner"]) if ln not in sc["heads"]]
        rng.shuffle(lines)
        taken = 0
        for ln in lines:
            if taken >= args.per_file:
                break
            names = sc["used"].get(ln, set())
            if not names:
                continue
            taken += 1
            true_i = sc["owner"][ln]
            counts = S.bind_counts(sc, ln)
            bounds = [S.bound_wo(sc, i, ln) for i in range(nf)]
            s_name = [S.score(names, bounds[i], counts, True) for i in range(nf)]
            s_raw = [S.score(names, bounds[i], counts, False) for i in range(nf)]
            top = max(s_name)
            tied = [i for i in range(nf) if s_name[i] == top]
            # THE FULL-POPULATION CONTROL, taken on every scored line
            c["full_n"] += 1
            c["full_name"] += expected_acc(s_name, true_i, list(range(nf)))
            if len(tied) < 2 or top <= 0.0:
                c["full_combined"] += expected_acc(s_name, true_i, list(range(nf)))
                continue
            # ---- the decision population
            c["amb"] += 1
            c["true_in_tie"] += int(true_i in tied)
            tsets = {i: types_wo(body_t, i, ln) for i in range(nf)}
            have = Counter()
            for i in range(nf):
                for t in tsets[i]:
                    have[t] += 1
            lt = line_t.get(ln, set())
            s_shape = [sum((1.0 / max(1, have[t])) if args.normalise else 1.0
                           for t in lt if t in tsets[i]) for i in range(nf)]
            s_size = [float(n_body[i] - (1 if i == true_i else 0)) for i in range(nf)]
            c["full_combined"] += expected_acc(s_shape, true_i, tied)
            if true_i not in tied:
                continue                    # no answer in this population for this line
            c["n"] += 1
            c["tied_size"] += len(tied)
            c["random"] += 1.0 / len(tied)
            c["shape"] += expected_acc(s_shape, true_i, tied)
            c["rawname"] += expected_acc(s_raw, true_i, tied)
            c["size"] += expected_acc(s_size, true_i, tied)
            c["line_types"] += len(lt)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="_*.py")
    ap.add_argument("--per-file", type=int, default=40)
    ap.add_argument("--min-scopes", type=int, default=8)
    ap.add_argument("--raw", dest="normalise", action="store_false",
                    help="the declared control reading: the UNDIVIDED type count")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    files = sorted(Path(".").glob(args.files))
    c = run(files, args, rng)
    n, amb, fn = max(1, c["n"]), max(1, c["amb"]), max(1, c["full_n"])
    rep = {"seed": args.seed, "normalise": bool(args.normalise), "files": len(files),
           "scored": c["full_n"], "amb_live": c["amb"] / fn, "decided_on": c["n"],
           "true_in_tie": c["true_in_tie"] / amb, "tied_size": c["tied_size"] / n,
           "line_types": c["line_types"] / n,
           "random": c["random"] / n, "shape": c["shape"] / n,
           "rawname": c["rawname"] / n, "size": c["size"] / n,
           "full_name": c["full_name"] / fn, "full_combined": c["full_combined"] / fn}
    rep["shape_minus_random"] = rep["shape"] - rep["random"]
    rep["shape_minus_rawname"] = rep["shape"] - rep["rawname"]
    rep["shape_minus_size"] = rep["shape"] - rep["size"]
    rep["control_delta"] = rep["full_combined"] - rep["full_name"]

    print(f"{rep['files']} files, {rep['scored']} lines scored, amb_live {rep['amb_live']:.4f}, "
          f"{'normalised' if args.normalise else 'RAW'} type count")
    print(f"VOID CHECK  true_in_tie {rep['true_in_tie']:.4f}  <- read first: the share of "
          f"ambiguous lines whose true scope is even among the tied")
    print(f"POPULATION  {rep['decided_on']} lines, {rep['tied_size']:.2f} tied scopes each, "
          f"{rep['line_types']:.2f} node types per line")
    print(f"TIE-BREAK   shape {rep['shape']:.4f}   random {rep['random']:.4f}   "
          f"raw name {rep['rawname']:.4f}   size {rep['size']:.4f}")
    print(f"            shape-random {rep['shape_minus_random']:+.4f}   "
          f"shape-rawname {rep['shape_minus_rawname']:+.4f}   "
          f"shape-size {rep['shape_minus_size']:+.4f}")
    print(f"CONTROL     full population: name {rep['full_name']:.4f}  "
          f"name+shape {rep['full_combined']:.4f}  delta {rep['control_delta']:+.4f}")

    gate = (rep["shape_minus_random"] > 0.05 and rep["shape_minus_rawname"] > 0.05
            and rep["shape_minus_size"] > 0.05)
    ctrl = rep["control_delta"] >= 0.0
    rep["gate"], rep["control_ok"] = bool(gate), bool(ctrl)
    if gate and ctrl:
        print("\nSHAPE CARRIES THE TIE. There is evidence on the decision population that is NOT "
              "the name count, it beats the coin, the undivided name overlap and the size "
              "confound, and the catalogue is not paid for it. A mind built here would have "
              "something to learn that counting does not already hold.")
    elif not gate:
        print("\nSHAPE DOES NOT CARRY THE TIE: " +
              ("it does not beat a coin. " if rep["shape_minus_random"] <= 0.05 else "") +
              ("the undivided NAME overlap does as well - it is names in disguise. "
               if rep["shape_minus_rawname"] <= 0.05 else "") +
              ("SIZE does as well - a bigger scope holds more node types for the same reason it "
               "holds more of everything. " if rep["shape_minus_size"] <= 0.05 else "") +
              "Do not build Phi here on this feature.")
    else:
        print("\nTHE CATALOGUE WAS PAID FOR IT: the combined rule is below the name rule on the "
              "full population, so the 23% was bought with the 96%.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[f"{args.seed}{'' if args.normalise else '_raw'}"] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
