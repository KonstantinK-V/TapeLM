"""Check of 34.4: the walk_only route mask, and the reader that decides whether to run it.

The stage half is read off the source; the reader half is RUN, on dumps this file writes, because
the reader is pure python and a checker that can execute the thing it checks should.

  1. OFF IS OFF. `all` is the default and the mask lives inside its own branch, so every earlier
     run is bit for bit.
  2. REFUSED WITHOUT --two-way. Without it stage one's logits ARE the own worlds, so detaching
     them would freeze the home pick too and the arm would differ from its control in two
     decisions instead of one.
  3. THE MASK IS THE TAPE'S: `answerable and not truth_in_own`. A mask on the mind's own
     correctness would be a moving target - the fault calib_term's docstring names.
  4. ONLY THE ROUTER IS CUT. The detach is on `p1`, after the softmax. Detaching `l1` would cut
     the summaries' path to the worlds and take both picks with it.
  5. THE MASK IS IN THE LOSS ONLY. If ROUTE_ON reached reach_logits or reach_pick, the EXAM would
     see the truth - which is a leak, not a lever.
  6. THE DENOMINATOR IS EVERY QUESTION. `n` is counted before the branch, `live` inside it, or
     `route_on_live` reads 1.0 on an arm that trained the router on 4% of its questions.
  7. REPORTED, and 8. IN THE ARM SIGNATURE - a mind whose router was taught on a different
     population is not the same mind.
  9. THE READER'S FOUR VOID CHECKS EXIST WITH THEIR THRESHOLDS, and they are what decides whether
     the arm is run at all.
 10. THE READER POOLS AS COUNTS. A mean of four rates is a different number, and the population is
     recovered exactly as pick.n / arrive.
 11. A REPORTED nan READS AS MISSING, never as a zero that pools - the reading discipline that a
     null must be read on an absolute quantity.

Every property is a wrong number, and every one is verified by re-introducing its own failure.

    python _check394_route.py
"""
from __future__ import annotations
import io
import json
import re
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import _read394_walkonly as R
SRC = Path('_stage289_derivation.py')
RDR = Path('_read394_walkonly.py')

def code_of(text):
    return re.sub('"""(?:.|\\n)*?"""', '', text)

def body(src, name):
    m = re.search(f'^def {name}\\(.*?(?=\\n(?:def |@|# ---))', src, re.S | re.M)
    return m.group(0) if m else ''
DUMPS = ((1337, 0.3, 120, 44, 31, 0.061), (8642, 0.28, 110, 39, 30, 0.058))

def write_dumps(d, arrive_override=None):
    for seed, arrive, n_step, mind, cnt, deep in DUMPS:
        rep = {'seed': seed, 'wall_s': 1.0, 'reach': {'held_out': {'walk_only_arrive': arrive if arrive_override is None else arrive_override, 'walk_only_pick': {'n': n_step, 'mind': mind, 'rival': 20, 'count_rival': cnt}, 'step_rate': 0.21, 'deep_only_rate': deep, 'hit_of_deep_only': float('nan'), 'hit_of_own': 0.7, 'ceiling': 0.4, 'n': 9000}}}
        (d / f'stage289_decision_s{seed}.json').write_text(json.dumps(rep))

def read_out(d):
    buf = io.StringIO()
    with redirect_stdout(buf):
        R.main([str(p) for p in sorted(d.glob('*.json'))] + ['--held'])
    return buf.getvalue()

def props(src=None, rdr=None):
    src = SRC.read_text(encoding='utf-8') if src is None else src
    rdr = RDR.read_text(encoding='utf-8') if rdr is None else rdr
    f = []
    ls = code_of(body(src, 'reach_loss'))
    if not re.search('^ROUTE_ON = "all"', src, re.M):
        f.append('1. ROUTE_ON does not default to `all`, so earlier runs are not bit for bit')
    if 'if ROUTE_ON == "walk_only":' not in ls:
        f.append('1. the mask is not inside its own branch in reach_loss')
    if 'if ROUTE_ON != "all" and not args.two_way:' not in src:
        f.append('2. --route-on is accepted without --two-way, where cutting the route cuts the home pick with it')
    i0 = ls.find('if ROUTE_ON == "walk_only":')
    i1 = ls.find('p2 = torch.softmax(l2, 0)', max(0, i0))
    seg = ls[i0:i1] if 0 <= i0 < i1 else ''
    if not seg:
        f.append('0. the mask block was not found between its branch and the stage-two softmax')
    if 'if ans and q["truth_value"] not in set(own):' not in ls:
        f.append('3. the mask is not `answerable and not truth_in_own`')
    for bad in ('mind_right', '_said ==', 'argmax'):
        if bad in seg:
            f.append(f"3. the mask reads {bad!r} - the mind's own correctness is a moving target")
    if 'p1 = p1.detach()' not in ls:
        f.append("4. the route's probability is not the thing detached")
    for bad in ('l1 = l1.detach()', 'l2 = l2.detach()', 'v2 = v2.detach()', 'v_stay = v_stay.detach()', 'lo = lo.detach()'):
        if bad in ls:
            f.append(f'4. {bad} - that cuts a PICK, not the route, and the arm would differ from its control in two decisions')
    for name in ('reach_logits', 'reach_pick', 'reach_candidates', 'reach_move_pick'):
        if 'ROUTE_ON' in code_of(body(src, name)):
            f.append(f'5. ROUTE_ON reaches {name} - the mask would be visible to the exam')
    i_n, i_if, i_live = (seg.find('_ROUTE_LIVE["n"]'), seg.find('if ans and'), seg.find('_ROUTE_LIVE["live"]'))
    if not 0 <= i_n < i_if < i_live:
        f.append(f'6. the live count is not n-before-branch, live-inside (n={i_n} if={i_if} live={i_live}) - route_on_live would read 1.0 on an arm that trained the router on a few percent of its questions')
    for k in ('route_on_live', 'route_on_seen'):
        if f'"{k}"' not in src:
            f.append(f'7. {k} is not reported')
    sig = src[src.find('# 341 IS IN THE SIGNATURE'):][:900]
    if '"route_on": ROUTE_ON' not in sig:
        f.append('8. route_on is not in the arm signature - a mind whose router was taught on another population would transplant onto one that was not')
    for want in ('V1 FIRED', 'V2 FIRED', 'V3 FIRED', 'V4 FIRED', 'arrive >= 0.95', 'share < 0.02', 'deep_r <= 0.05', 'c_rate >= m_rate'):
        if want not in rdr:
            f.append(f'9. the reader is missing {want!r} - a void check that is not in the file is not a void check')
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_dumps(d)
        out = read_out(d)
        for want in ('walk_only 793 of 18000 rows (0.0440)', 'arrive 0.2901', 'mind 0.3609', 'count 0.2652'):
            if want not in out:
                f.append(f'10. the pooled reading is wrong: expected {want!r} in\n{out}')
        if 'no void check fired' not in out:
            f.append(f'10. a void check fired on a dump built not to fire one:\n{out}')
        if 'deep_only 0.0595' not in out:
            f.append('11. deep_only did not pool to 0.0595 - a nan or a missing key is being read as a number')
        write_dumps(d, arrive_override=0.97)
        out2 = read_out(d)
        if 'V1 FIRED' not in out2:
            f.append('9. V1 did not fire on arrive 0.97')
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_dumps(d)
        for i, (seed, *_r) in enumerate(DUMPS):
            fp = d / f'stage289_decision_s{seed}.json'
            rep = json.loads(fp.read_text())
            if i:
                rep['reach']['held_out']['deep_only_rate'] = float('nan')
            fp.write_text(json.dumps(rep))
        out3 = read_out(d)
        if 'deep_only 0.0610' not in out3:
            f.append(f'11. a nan deep_only_rate is being pooled instead of skipped:\n{out3}')
    return f
MUT_SRC = (('the mask runs by default', 'ROUTE_ON = "all"', 'ROUTE_ON = "walk_only"', '1.'), ('accepted without --two-way', 'if ROUTE_ON != "all" and not args.two_way:', 'if False:', '2.'), ("the mask reads the mind's own correctness", 'if ans and q["truth_value"] not in set(own):', 'if ans and int(l1.argmax()) == 1:', '3.'), ('the logits are detached, taking both picks with them', '            p1 = p1.detach()', '            l1 = l1.detach()', '4.'), ('the mask is visible to the offer', '    if which is None and "_reach_c" in q:', '    if ROUTE_ON == "walk_only" or (which is None and "_reach_c" in q):', '5.'), ('the denominator counts only the live questions', '        _ROUTE_LIVE["n"] += 1\n        if ans and q["truth_value"] not in set(own):\n            _ROUTE_LIVE["live"] += 1', '        if ans and q["truth_value"] not in set(own):\n            _ROUTE_LIVE["n"] += 1\n            _ROUTE_LIVE["live"] += 1', '6.'), ('not reported', '"route_on_live": (_ROUTE_LIVE["live"]', '"unused_live": (_ROUTE_LIVE["live"]', '7.'), ('not in the arm signature', '\n                "move_teach": MOVE_TEACH, "route_on": ROUTE_ON,', '\n                "move_teach": MOVE_TEACH,', '8.'))
MUT_RDR = (('a void check is gone', 'if not math.isnan(arrive) and arrive >= 0.95:', 'if False:', '9.'), ('the population is not recovered from the two reported numbers', '    pop = (stepped / arrive) if (arrive and stepped is not None and arrive > 0) else None', '    pop = stepped', '10.'), ('a nan pools as a zero', '    return None if math.isnan(v) else v', '    return v', '11.'))

def main() -> int:
    src, rdr = (SRC.read_text(encoding='utf-8'), RDR.read_text(encoding='utf-8'))
    fails = props()
    for name, old, new, tag in MUT_SRC:
        if src.count(old) != 1:
            fails.append(f'MUTATION {tag} ({name}): its anchor occurs {src.count(old)} times')
            continue
        if not any((g.startswith(tag) for g in props(src=src.replace(old, new, 1)))):
            fails.append(f'MUTATION {tag} ({name}): the failure was re-introduced and check {tag} did not fire - it is a comment, not a check')
    for name, old, new, tag in MUT_RDR:
        if rdr.count(old) != 1:
            fails.append(f'MUTATION {tag} ({name}): its anchor occurs {rdr.count(old)} times')
            continue
        mod = dict(R.__dict__)
        exec(compile(rdr.replace(old, new, 1), '<mutant>', 'exec'), R.__dict__)
        try:
            got = props(rdr=rdr.replace(old, new, 1))
        finally:
            R.__dict__.clear()
            R.__dict__.update(mod)
        if not any((g.startswith(tag) for g in got)):
            fails.append(f'MUTATION {tag} ({name}): the failure was re-introduced and check {tag} did not fire - it is a comment, not a check')
    for x in fails:
        print('FAIL ' + x)
    print(f'{len(fails)} failures' if fails else f'all properties hold, and all {len(MUT_SRC) + len(MUT_RDR)} re-introduced failures were caught')
    return 1 if fails else 0
if __name__ == '__main__':
    raise SystemExit(main())