"""Torch-free checks for 336, 337 and 338 - the three things added after 335.

WHAT THIS CATCHES, and each item is a fault this project has actually shipped:

  ONE ARGMAX. The staged decision (stay / walk / line, then the deeper read as the last option
  of stage two) now lives in reach_pick, because 337 needs the confidence per question and 338
  asks the same of a place. A second copy of that logic is how `step_line 6.4` happened - a
  plausible number for a rate that cannot exceed 1. Asserted: the branch `pick == len(n1)`
  appears in exactly one function.

  EVERY WALK HONOURS RETENTION. 338 drops places from the tape. If one entry point still sees
  the whole tape, the retained tape is not a tape - it is a mixture, and the comparison across
  rules would be measuring the leak. Asserted: reach_places, reach_places_from and the random
  control in reach_reachable all consult retain_keep.

  THE GUARDS ARE IN THE CODE, NOT IN THE COMMANDS. `--retain-by mind` must refuse a mind that
  is training (it would be fitting the tape to itself), and `--rival-mind` must refuse depth > 1
  (the deeper walk is rooted at the mind's own pick, so two minds would not be answering the
  same question). A guard that lives only in a runbook is a guard that is missed once.

  THE STATISTIC IS RIGHT. rank_auc and prec_at are pure Python, so they are RUN here against
  hand-computed answers - perfect, inverted, all-ties, and the half-credit tie case - rather
  than read. 337's whole claim is an AUC comparison; an off-by-one in the tie handling would
  make it a plausible wrong number of exactly the kind this file exists to stop.

    python _check337_rank.py
"""
from __future__ import annotations
import ast
import math
from pathlib import Path
SRC = Path('_stage289_derivation.py')

def fn(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None

def calls(node, name):
    return sum((1 for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == name))

def main() -> int:
    src = SRC.read_text(encoding='utf-8')
    tree = ast.parse(src)
    bad = []
    owners = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and 'pick == len(n1)' in ast.unparse(n):
            owners.append(n.name)
    print(f'staged argmax lives in: {owners}')
    if owners != ['reach_pick']:
        bad.append(f'the staged argmax should live only in reach_pick, found {owners}')
    cols = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'REACH_COLS':
            cols = [e.value for e in n.value.elts]
    tail = cols[-4:] if cols else []
    print(f'last four columns declared: {tail}')
    if tail != ['other_right', 'other_stepped', 'pick_score', 'pick_margin']:
        bad.append(f'columns 336/337 are not the last four declared: {tail}')
    app = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and ast.unparse(n).startswith('reach_rows.append'):
            app = [ast.unparse(e) for e in n.args[0].elts]
    print(f'last four appended: {(app[-4:] if app else None)}')
    if not app or app[-4:] != ['_o_right', '_o_step', '_pscore', '_pmarg']:
        bad.append('the exam does not append the 336/337 values last, in the declared order')
    same_q = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'reach_logits' and n.args and (getattr(n.args[0], 'id', None) == 'OTHER_NET') and ([getattr(a, 'id', None) for a in n.args[1:5]] == ['p', 'q', 'device', 'bank'])]
    print(f'rival mind reads the same p and q: {len(same_q)} call(s)')
    if not same_q:
        bad.append('the rival mind does not read the same p and q through reach_logits')
    for name in ('reach_places', 'reach_places_from', 'reach_reachable'):
        f = fn(tree, name)
        n = calls(f, 'retain_keep') if f else 0
        print(f'{name}: consults retain_keep {n}x')
        if n < 1:
            bad.append(f'{name} never consults retain_keep - a dropped place is still walkable')
    for what, needle in (('--retain-by mind requires a frozen loaded mind', 'RETAIN_BY == "mind" and RETAIN and (not args.load_mind or args.finetune)'), ('--rival-mind refuses depth > 1', 'args.rival_mind and args.reach_depth > 1'), ('--retain is NOT in the mind signature', '# --retain IS DELIBERATELY NOT IN THE SIGNATURE'), ('the mind judges places against the whole tape', '_RETAIN_BUSY = True')):
        ok = needle in src
        print(f"guard: {what} -> {('OK' if ok else 'MISSING')}")
        if not ok:
            bad.append(f'guard missing: {what}')
    sig = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'mind_sig':
            sig = ast.unparse(n)
    if sig and ('retain' in sig or 'rival' in sig):
        bad.append("retention is in the mind signature: 338's transplant becomes unrunnable")
    ns = {'math': math}
    for name in ('rank_auc', 'prec_at', 'gate_top'):
        f = fn(tree, name)
        if f is None:
            bad.append(f'{name} is gone')
            continue
        exec(compile(ast.Module(body=[f], type_ignores=[]), name, 'exec'), ns)
    auc, pat = (ns.get('rank_auc'), ns.get('prec_at'))
    if auc and pat:
        cases = [([3, 2, 1, 0], [1, 1, 0, 0], 1.0), ([0, 1, 2, 3], [1, 1, 0, 0], 0.0), ([1, 1, 1, 1], [1, 1, 0, 0], 0.5), ([2, 1, 1, 0], [1, 0, 1, 0], 0.875), ([1, 0], [1, 1], float('nan'))]
        for s, l, want in cases:
            g = auc(s, l)
            ok = math.isnan(g) and math.isnan(want) or abs(g - want) < 1e-12
            print(f"rank_auc({s}, {l}) = {g} want {want} -> {('OK' if ok else 'WRONG')}")
            if not ok:
                bad.append(f'rank_auc({s}, {l}) = {g}, expected {want}')
        for s, l, k, want in [([3, 2, 1, 0], [1, 1, 0, 0], 2, 1.0), ([3, 2, 1, 0], [0, 0, 1, 1], 2, 0.0), ([3, 2, 1, 0], [1, 0, 1, 0], 9, 0.5)]:
            g = pat(s, l, k)
            ok = abs(g - want) < 1e-12
            print(f"prec_at(k={k}) = {g} want {want} -> {('OK' if ok else 'WRONG')}")
            if not ok:
                bad.append(f'prec_at({s}, {l}, {k}) = {g}, expected {want}')
    gt = ns.get('gate_top')
    if gt is None:
        bad.append('gate_top is gone')
    else:
        for s, k, want in [([5, 4, 3, 2, 1], 2, {0, 1}), ([1, 2, 3], 0, set()), ([1, 2, 3], 9, {0, 1, 2}), ([7, 7, 7, 7], 2, {0, 1})]:
            g = gt(s, k)
            ok = g == want
            print(f"gate_top({s}, {k}) = {sorted(g)} want {sorted(want)} -> {('OK' if ok else 'WRONG')}")
            if not ok:
                bad.append(f'gate_top({s}, {k}) = {sorted(g)}, expected {sorted(want)}')
        import random as _r
        rr = _r.Random(11)
        for _t in range(200):
            m = rr.randrange(1, 40)
            k = rr.randrange(0, m + 3)
            sizes = {len(gt([rr.choice([0, 1, 2, 3]) for _ in range(m)], k)) for _ in range(3)}
            if len(sizes) != 1 or sizes != {min(k, m)}:
                bad.append(f'gate_top let through {sizes} of {m} at k={k}')
                break
        else:
            print('gate_top: matched coverage over 200 random ranker triples -> OK')
    fracs = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'GATE_FRACTIONS':
            fracs = ast.literal_eval(n.value)
    print(f'gate coverage grid: {fracs}')
    if not fracs or len(fracs) < 3:
        bad.append('GATE_FRACTIONS is missing or too short to be a grid rather than a choice')
    gb = fn(tree, 'gateblock')
    gbs = ast.unparse(gb) if gb else ''
    for need, why in (('yield', 'a gate can buy precision by answering less'), ('payoff', 'sharper is not the same as worth it'), ('always_silent', 'refusing everything is the floor, not the ungated run'), ('gain', 'payoff is only readable as a difference from that floor'), ('random', 'a matched-coverage random gate is the floor for WHICH k'), ('composition', 'where the kept answers came from decides what the gate is a claim about')):
        if need not in gbs:
            bad.append(f'the gate does not report `{need}`: {why}')
    if 'gate_walk_only' not in src:
        bad.append('the gate is not run on the walk-only subset, so it is scored mostly on questions an index already answers')
    rb = fn(tree, 'rankblock')
    if rb is None or "'right'" not in ast.unparse(rb):
        bad.append('rankblock does not score the ranking against `right`')
    rl_src = ast.unparse(fn(tree, 'reach_logits') or ast.parse(''))
    for needle, why in (('m = min(len(l1), len(l2))', 'the two branches must be summarised over equal option counts'), ('stay = summary(', 'both branches go through the same summary'), ('go = summary(', 'both branches go through the same summary'), ("TWO_WAY_BY == 'max'", 'max must remain the default path, reproducing every earlier run exactly')):
        if needle not in rl_src:
            bad.append(f'two-way: {why}')
    if 'l2 = torch.cat([l2, ld.max().reshape(1)])' not in src:
        bad.append('the deep option is no longer appended to l2 after stage one is priced')
    i_two = src.find('stay = summary(')
    i_deep = src.find('l2 = torch.cat([l2, ld.max().reshape(1)])')
    if 0 < i_deep < i_two:
        bad.append('the deep max is attached BEFORE the stay/go comparison: depth would be a reason to set out again, which 325 removed')
    st = fn(tree, 'speak_term')
    sts = ast.unparse(st) if st else ''
    if not st:
        bad.append('speak_term is gone')
    else:
        ok = 'torch.softmax(m, 0)' in sts
        print(f"speak_term: softmax over the batch -> {('OK' if ok else 'MISSING')}")
        if not ok:
            bad.append('speak_term does not softmax across the batch, so refusing everything is still expressible and the arm measures nothing new')
    rl = fn(tree, 'reach_loss')
    rls = ast.unparse(rl) if rl else ''
    if 'mixed_payoff(False, rt, ans) - mixed_payoff(True, rt, ans)' not in rls:
        bad.append('the speaking advantage is not derived from mixed_payoff')
    if 'keep_graph=True' not in rls:
        bad.append('reach_loss takes a detached margin: the speaking term would have no gradient')
    for needle, why in (('_SPEAK_ACC = []', "the batch's accumulator must be opened by the training loop"), ('_SPEAK_ACC = None', 'and closed, or the probe fills it with graph-holding tensors'), ('recorded', 'a short accumulator breaks the positional pairing and must raise'), ('SPEAK_BATCH < 2', 'a softmax over one margin is a constant')):
        if needle not in src:
            bad.append(f'341 guard missing ({why})')
    import math as _m

    def soft(xs):
        e = [_m.exp(x - max(xs)) for x in xs]
        return [v / sum(e) for v in e]
    for margins, advs, want in [([2.0, 0.0], [0.25, -1.75], 'positive'), ([0.0, 2.0], [0.25, -1.75], 'negative'), ([1.0, 1.0], [-2.0, -2.0], 'negative')]:
        p = soft(margins)
        v = sum((a * b for a, b in zip(p, advs)))
        got = 'positive' if v > 0 else 'negative'
        ok = got == want and abs(sum(p) - 1.0) < 1e-12
        print(f"speak_term({margins}, {advs}) = {v:+.4f} sum(p)={sum(p):.4f} want {want} -> {('OK' if ok else 'WRONG')}")
        if not ok:
            bad.append(f'the speaking term is {got} for {margins}/{advs}, expected {want}')
    print()
    if bad:
        for b in bad:
            print(f'BROKEN: {b}')
        return 1
    print('336/337/338 OK')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())