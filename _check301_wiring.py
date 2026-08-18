"""Does every key the reach path READS actually get WRITTEN, and does every verb get dispatched?

WHY THIS EXISTS. Two faults shipped in one day and neither was a wrong idea - both were wiring,
and both were invisible to every check I had.

  1 reach_questions_for was dead code. Its dispatch sat inside open_questions_for, which runs
    only when OPEN is set, and the flag block rejects --reach together with --open. Every check
    called the reach functions directly, so they tested the machinery and never the wiring.
  2 The exam read _rc["n_places"] and reach_candidates stopped producing it, so a run crashed
    after training - the most expensive possible moment - on a dictionary key.

Neither needs torch to catch, which is the point: this file reads the source, so it runs here
where the stage cannot.

    python _check301_wiring.py
"""
from __future__ import annotations

import ast
import re
import sys

SRC = "_stage289_derivation.py"


def dispatch(tree):
    """Every verb flag in questions_for, and every builder nothing routes to."""
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "questions_for"]
    if not fn:
        return {}, {"questions_for missing"}
    routed = {}
    for st in fn[0].body:
        if isinstance(st, ast.If) and isinstance(st.test, ast.Name):
            t = st.body[0]
            if isinstance(t, ast.Return) and isinstance(t.value, ast.Call):
                routed[st.test.id] = getattr(t.value.func, "id", "?")
    builders = {n.name for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.endswith("_questions_for")}
    return routed, builders - set(routed.values())


def walk_keys(src, tree):
    """Keys the walk's dict is read with, against the keys it is built with."""
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "reach_candidates"][0]
    written = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    written.add(k.value)
    read = set(re.findall(r'(?:_rc|rc)\[\s*"([a-z_0-9]+)"\s*\]', src))
    return written, read


def index_keys(src, tree):
    """Same for reach_index's table - but READ ONLY INSIDE THE WALK.

    `ix` is a common local name here: 293's identity index uses it too, and searching the whole
    file for ix["..."] reported by_place, swords and parts as missing keys of a table that never
    claimed them. A wiring check that cries wolf is a wiring check nobody runs, so the reads are
    taken from reach_* functions and from the exam's reach branch only.
    """
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reach_index"][0]
    written = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    written.add(k.value)
    scope = "\n".join(ast.get_source_segment(src, n) or "" for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name.startswith("reach_"))
    read = set(re.findall(r'ix\[\s*"([A-Za-z_0-9]+)"\s*\]', scope))
    return written, read


def main() -> int:
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    ok = True

    routed, orphan = dispatch(tree)
    print("dispatch in questions_for:")
    for flag, fn in sorted(routed.items()):
        print(f"  {flag:12s} -> {fn}")
    print(f"  builders nothing routes to: {sorted(orphan) or 'none'}")
    ok &= not orphan
    ok &= routed.get("REACH") == "reach_questions_for"
    print(f"  REACH is dispatched: {routed.get('REACH') == 'reach_questions_for'}")

    # EVERY GLOBAL main() DECLARES MUST HAVE A MODULE-LEVEL DEFAULT. Two of today's three faults
    # were a constant block that never landed while the code reading it did: the file imports,
    # the check passes, and the run dies on NameError after the tape is built.
    m = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    top = {t.id for n in tree.body if isinstance(n, ast.Assign)
           for t in n.targets if isinstance(t, ast.Name)}
    g = {nm for n in ast.walk(m) if isinstance(n, ast.Global) for nm in n.names}
    print(f"\nglobals in main with no module default: {sorted(g - top) or 'none'}")
    ok &= not (g - top)

    # THE REACH ROW MUST BE EXACTLY AS WIDE AS ITS NAME TUPLE. Three times a new column shifted
    # the ones after it and the report went on printing plausible wrong values - 304 reported a
    # rate of 6.4, which was the mean row count of a world. Nothing raises; it lies.
    for rowvar, colvar in (("reach_rows", "REACH_COLS"), ("pair_rows", "PAIR_COLS"),
                           ("cons_rows", "CONS_COLS")):
        ap = [x for x in ast.walk(tree) if isinstance(x, ast.Call)
              and isinstance(x.func, ast.Attribute) and x.func.attr == "append"
              and getattr(x.func.value, "id", "") == rowvar]
        cols = len(ap[0].args[0].elts) if ap else -1
        declared = len([e for n in tree.body if isinstance(n, ast.Assign)
                        and any(getattr(t, "id", "") == colvar for t in n.targets)
                        for e in n.value.elts])
        print(f"\n{rowvar}: {cols} columns appended, {declared} names declared -> "
              f"{'match' if cols == declared else 'MISMATCH'}")
        ok &= (cols == declared)

    for name, (written, read) in (("reach_candidates", walk_keys(src, tree)),
                                  ("reach_index", index_keys(src, tree))):
        missing = read - written
        unused = written - read
        print(f"\n{name}: writes {sorted(written)}")
        print(f"  read somewhere but never written: {sorted(missing) or 'none'}")
        print(f"  written but never read: {sorted(unused) or 'none'}")
        ok &= not missing

    # STAGE2_ALWAYS MUST NOT BECOME A STEP REWARD. The off-policy lesson is outside the policy
    # term; if the body of `if STAGE2_ALWAYS` mentions p1 or REACH_GAMMA, the lesson quietly
    # pays for movement and 314's claim is void. Names only - comments are free.
    rl = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reach_loss"]
    bad = set()
    if rl:
        for n in rl[0].body:
            if (isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                    and n.test.id == "STAGE2_ALWAYS"):
                names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
                bad = names & {"p1", "REACH_GAMMA"}
                break
        else:
            bad = {"missing STAGE2_ALWAYS block"}
    print(f"\nSTAGE2_ALWAYS guard (no p1/REACH_GAMMA in lesson): "
          f"{'GUARD OK' if not bad else 'BROKEN ' + str(sorted(bad))}")
    ok &= not bad

    # DEPTH MUST NOT INFLATE STAGE ONE. Concatenating the deep max onto l2 BEFORE lookahead
    # takes max(l2) for the step is the cardinality bias of 304 moved to "whether to leave".
    # Source order is the guard: deep cat onto l2 must come AFTER the stage-one tail.
    rl2 = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reach_logits"]
    depth_order = False
    if rl2:
        seg = ast.get_source_segment(src, rl2[0]) or ""
        i_tail = seg.find("l1 = torch.cat([l1] + tail)")
        i_deep = seg.find("l2 = torch.cat([l2, ld.max()")
        depth_order = i_tail >= 0 and i_deep >= 0 and i_deep > i_tail
    print(f"DEPTH-after-tail guard (deep max not in stage-one): "
          f"{'GUARD OK' if depth_order else 'BROKEN'}")
    ok &= depth_order

    # TWO_WAY: lookahead tail must not feed stage one, and both branch VALUES are expectations.
    tw_skip = "REACH_LOOKAHEAD and not TWO_WAY" in src
    tw_exp = False
    if rl:
        for n in rl[0].body:
            if (isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                    and n.test.id == "TWO_WAY"):
                body_src = ast.get_source_segment(src, n) or ""
                # stay value must be expectation over own logits, not a max
                tw_exp = ("softmax(lo" in body_src.replace(" ", "")
                          or "softmax(lo," in body_src.replace(" ", "")
                          or "torch.softmax(lo" in body_src)
                tw_exp = tw_exp and "v_stay" in body_src and "v2" in body_src
                # must not value a branch by max alone in the return
                if ".max()" in body_src and "v_stay" in body_src:
                    # max only allowed if not defining v_stay via max
                    tw_exp = tw_exp and "v_stay" in body_src and "softmax" in body_src
                break
        else:
            tw_exp = False
    print(f"TWO_WAY guard (no lookahead tail; both branches expectations): "
          f"{'GUARD OK' if (tw_skip and tw_exp) else 'BROKEN'}"
          + ("" if tw_skip else " missing and-not-TWO_WAY")
          + ("" if tw_exp else " stay/go not both expectations"))
    ok &= tw_skip and tw_exp

    print("\nWIRING OK" if ok else "\nWIRING BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
