"""Static and behavioural check of 385's move-valued output. No torch, no corpus.

385 is the first arm where Phi decides WHERE TO LOOK rather than WHICH NAME TO SAY, so the ways
it can silently become its own baseline are new. Ten properties (9-10 are 386, the ballot):

  1. OFF IS OFF. Without --moves, reach_candidates builds the merged offer exactly as before and
     no move is recorded.
  2. THE ORDER IS ENFORCED BY A CRASH. With moves on, asking for the offer before the move is
     chosen raises. If it merely defaulted, the arm would quietly rebuild the merged offer - the
     very thing it exists to replace - and every number downstream would look ordinary.
  3. A PROPOSAL IS NOT CACHED. Enumerating a lane while choosing must not become the question's
     offer, or the last move probed would win by bookkeeping.
  4. THE CHOSEN MOVE IS CACHED, so the rival, the deep read and the report all grade the same
     offer as the mind - the property reach_candidates has had since it was written.
  5. ONE MOVE, ONE LANE: `step` carries no connect and no copy candidate, `share` carries only
     connect's, `lines` only copy's - each at the UNCHANGED cap, so a move is a different offer
     and not a thinner one.
  6. EVERY PROBE IS ONE ROW. Two probes of different size would make the move choice a
     row-count comparison, which is the tell that undid 291 and 296.
  7. A MOVE THAT OFFERS NOTHING IS NOT ON THE BALLOT, and if none offers anything the question
     still gets an offer rather than silently leaving the population.
  8. THE CHOICE IS RECORDED as move_id, or "the mind decides" cannot be told apart from "the
     mind always takes the same move".
  9. (386) THE BALLOT IS CONFIGURABLE AND VALIDATED, and it is WRITTEN INTO THE REPORT: move_id
     is an index into it and means nothing without it. An unknown name must be refused rather
     than silently dropped, or an arm asked for two moves could quietly run with one.
 10. (386) HIT IS SPLIT BY MOVE. 385 could only be read by correlating the split against hit
     across four seeds; within a seed that correlation is a measurement, and without it a
     good move and a bad one average into one uninformative number.

    python _check385_moves.py
"""
from __future__ import annotations
import ast
import re
import sys
from pathlib import Path
SRC = Path('_stage289_derivation.py')

def main() -> int:
    src = SRC.read_text(encoding='utf-8')
    tree = ast.parse(src)
    fails = []

    def body(name):
        m = re.search(f'^def {name}\\(.*?(?=\\n(?:def |@|\\w))', src, re.S | re.M)
        if not m:
            fails.append(f'0. {name} not found')
            return ''
        return m.group(0)
    rc, mp = (body('reach_candidates'), body('reach_move_pick'))
    if 'which = q.get("_move", "all")' not in rc:
        fails.append('1. reach_candidates does not fall back to the merged offer when off')
    if 'if MOVES_ON and "_move" not in q:' not in rc or 'raise RuntimeError' not in rc:
        fails.append('2. an offer built before the move is chosen does not raise - the arm would silently rebuild the merged offer it exists to replace')
    if 'if which is None and "_reach_c" in q:' not in rc:
        fails.append("3. a proposal reads the cache and would return another move's offer")
    if 'if which == q.get("_move", "all"):\n        q["_reach_c"] = out' not in rc:
        fails.append("3/4. the cache is written for proposals too, so the last lane probed would become the question's offer")
    for want in ('if which == "step":\n            conn, cop = [], []', 'elif which == "share":', 'elif which == "lines":'):
        if want not in rc:
            fails.append(f'5. the lane branch is missing or reshaped: {want.splitlines()[0]!r}')
    if 'raise RuntimeError(f"385: unknown move' not in rc:
        fails.append('5. an unknown move is accepted instead of raising')
    if 'REACH_CANDS' not in rc.split('if which != "all":')[-1][:600]:
        fails.append("5. a move's lane is not built at the unchanged cap")
    if 'rows[:1]' not in mp:
        fails.append('6. probes are not one row each - the move choice becomes a size contest')
    if 'reach_world(p, q, bank, device, v, rw, 1)' not in mp:
        fails.append('6. the probe world is not built at budget 1')
    if 'if rows:' not in mp:
        fails.append('6. a candidate with no row is probed anyway, at a different size')
    if 'if not cs:\n            continue' not in mp:
        fails.append('7. a move offering nothing is still on the ballot')
    if 'if not props:' not in mp or 'q["_move"] = MOVES[0]' not in mp:
        fails.append('7. with no move offering anything the question loses its offer and silently leaves the population')
    cols = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.targets and (getattr(node.targets[0], 'id', None) == 'REACH_COLS'):
            cols = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    if cols is None:
        fails.append('8. REACH_COLS is not a plain tuple of names any more')
    elif 'move_id' not in cols:
        fails.append('8. move_id is not a reach column')
    elif cols[-1] != 'move_id':
        fails.append(f"8. move_id is not the LAST column (it is {cols.index('move_id')} of {len(cols)}) - earlier RIX indices shifted")
    if '"move_share"' not in src:
        fails.append('8. the move split is not reported, so one move on every question is indistinguishable from a decision')
    rl = body('reach_logits')
    if not re.search('if MOVES_ON and \\("_move" not in q or MOVE_TEACH\\):|if MOVES_ON and "_move" not in q:', rl):
        fails.append('2. reach_logits does not choose the move first')
    elif rl.index('reach_move_pick') > rl.index('rc = reach_candidates(p, q)'):
        fails.append('2. the move is chosen AFTER the offer is built')
    if 'MOVES_ON' not in body('reach_channel') and '_reach_c' in body('reach_channel'):
        pass
    if 'rc = q.get("_reach_c")' not in body('reach_channel'):
        fails.append("6. reach_channel no longer tolerates a missing offer, so probing would raise or read another move's provenance")
    if '"move_set": list(MOVES) if MOVES_ON else []' not in src:
        fails.append('9. the ballot is not written into the report - move_id is an index into it and is uninterpretable without it')
    if 'if not MOVES or any(m not in MOVE_ALL for m in MOVES):' not in src:
        fails.append('9. an unknown or empty move set is accepted instead of refused')
    if 'MOVE_ALL = ("step", "share", "lines")' not in src:
        fails.append('9. the set of moves that EXIST is not separate from the ballot')
    if '"move_hit"' not in src:
        fails.append('10. hit is not split by move, so a good move and a bad one average away')
    else:
        mh = src.split('"move_hit"', 1)[1][:400]
        if 'RIX["mind_right"]' not in mh or 'RIX["move_id"]' not in mh:
            fails.append("10. move_hit is not computed from mind_right on the move's own rows")
        if 'max(1,' not in mh:
            fails.append('10. move_hit divides by a count that can be zero - a move nobody took would raise or read as a rate')
    if fails:
        print('FAIL')
        for f in fails:
            print('  ' + f)
        return 1
    print('PASS  off is off; an offer before the choice raises; proposals are not cached;')
    print('  one move one lane at the unchanged cap; every probe is one row; a move offering')
    print('  nothing is off the ballot; the choice is the last reach column and is reported;')
    print('  the ballot is validated and recorded, and hit is split by move.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())