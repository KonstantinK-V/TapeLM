"""Does the two-hole verb build honest questions - checked here, where the stage cannot run.

WHY THIS EXISTS. Every fault this project has shipped was wiring or a leak, and both are
invisible until after training. The pair verb adds a new way to leak that none of the earlier
checks can see: two holes on ONE line, so one hole's frame can contain the other's hidden token
and hand over the answer. The rule that closes it is a distance in corpus positions, which is
exact and therefore checkable - so it is checked, on a corpus small enough to verify by hand.

The pure-python half of the verb is lifted out of the stage by AST and run against a synthetic
pack with the walk stubbed. Nothing here needs torch, which is the point.

    python _check309_pair.py
"""
from __future__ import annotations
import ast
import random
from collections import Counter, defaultdict
import _tape_frames as tframes
SRC = '_stage289_derivation.py'
WANT = ('reach_question', 'pair_offer', 'pair_offers', 'pair_question', 'pair_rivals', 'pair_joint_index', 'pair_questions_for', 'reach_line_index', 'outside_mentions')

def lift():
    """The verb's pure-python functions, executed with the walk stubbed out."""
    src = open(SRC, encoding='utf-8').read()
    tree = ast.parse(src)
    ns = {'Counter': Counter, 'defaultdict': defaultdict, 'object': object}
    for n in tree.body:
        if isinstance(n, ast.Assign) and all((isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)):
            try:
                exec(compile(ast.Module([n], []), '<c>', 'exec'), ns)
            except Exception:
                pass
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in WANT:
            exec(compile(ast.Module([n], []), '<f>', 'exec'), ns)
    missing = [w for w in WANT if w not in ns]
    if missing:
        raise SystemExit(f'could not lift {missing}')
    return ns

class Tape:

    def __init__(self, values):
        self.values = values

def build_pack(lines, frame_max, ns):
    """A pack in the stage's shape, with only what the pair verb reads."""
    asrt, _addrs, _pool = tframes.frame_assertions(lines, frame_max, 2, 0, random.Random(0))
    values = [a['value'] for a in asrt]
    straddr = [a['address'] for a in asrt]
    p = {'tape': Tape(values), 'straddr': straddr, 'line': [a['line'] for a in asrt], 'pos': [a['pos'] for a in asrt]}
    by_addr = defaultdict(list)
    for sl, ad in enumerate(straddr):
        by_addr[ad].append(sl)
    p['items'] = [{'address': ad, 'S': ad, 'slots': ss} for ad, ss in by_addr.items() if len(ss) >= 2]
    items = p['items']
    fills = [[(v, [s for s in it['slots'] if p['tape'].values[s] == v], c) for v, c in Counter((p['tape'].values[s] for s in it['slots'])).items()] for it in items]
    ix = {'items': items, 'fills': fills, 'of': {it['address']: i for i, it in enumerate(items)}}
    ns['reach_index'] = lambda _p: ix
    ns['reach_candidates'] = lambda _p, sub: {'cands': [v for j, it in enumerate(items) if it['address'] != sub['address'] for v, _r, _c in fills[j]][:8]}
    return p

def main() -> int:
    ns = lift()
    frame_max = 3
    ns['FRAME_MAX'] = frame_max
    lines = ['the capital of france is paris and the capital of spain is madrid today', 'the capital of spain is madrid and the capital of italy is rome today', 'the capital of italy is rome and the capital of france is paris today', 'a red car drove past the old grey house near the small river bank there', 'a blue car drove past the new grey house near the small river bank there', 'a green car drove past the old white house near the large river bank there']
    p = build_pack(lines, frame_max, ns)
    qs = ns['pair_questions_for'](p, random.Random(1337))
    for q in qs:
        ns['pair_offers'](p, q)
    ok = True
    print(f"tape   {len(p['straddr'])} slots, {len(p['items'])} places, {len(qs)} pair questions")
    if not qs:
        print('no questions built - the check cannot say anything')
        return 1
    bad = [q for q in qs if abs(p['pos'][q['holes'][0]['slot']] - p['pos'][q['holes'][1]['slot']]) <= frame_max]
    print(f"leak   holes closer than frame_max: {len(bad)} -> {('OK' if not bad else 'BROKEN')}")
    ok &= not bad
    leak = []
    for q in qs:
        a, b = q['holes']
        for x, y in ((a, b), (b, a)):
            if y['truth'] in x['address'].replace('|', ' ').split():
                leak.append((x['address'], y['truth']))
    print(f"leak   truth of one hole inside the other's address: {len(leak)} -> {('OK' if not leak else 'BROKEN')}")
    ok &= not leak
    self_row = [q for q in qs for h in q['holes'] if h['slot'] in h['rows']]
    print(f"hide   a hole carrying its own row as evidence: {len(self_row)} -> {('OK' if not self_row else 'BROKEN')}")
    ok &= not self_row
    shape = []
    for q in qs:
        for k, h in enumerate(q['holes']):
            if q['slots'][q['query_rows'][k]] != h['slot']:
                shape.append(q)
        if len(q['slots']) != len(q['vals']) or len(set(q['query_rows'])) != 2:
            shape.append(q)
    print(f"shape  worlds whose query rows do not point at the holes: {len(shape)} -> {('OK' if not shape else 'BROKEN')}")
    ok &= not shape
    mixed = sum((1 for q in qs for h in q['holes'] if any((v in h['own'] for v in h['offer'])) and any((v not in h['own'] for v in h['offer']))))
    holes = sum((len(q['holes']) for q in qs))
    print(f'offer  holes offering both own and walked values: {mixed}/{holes}')
    ok &= mixed > 0
    ns['pair_joint_index'](p)
    read = 0
    for q in qs:
        marg, joint, seen, _bb, bag_seen = ns['pair_rivals'](p, q)
        truth = tuple((h['truth'] for h in q['holes']))
        a, b = q['holes']
        elsewhere = sum((1 for li in set(p['line']) if li != q['line'] and {(p['straddr'][s], p['tape'].values[s]) for s in range(len(p['straddr'])) if p['line'][s] == li} >= {(a['address'], truth[0]), (b['address'], truth[1])}))
        if elsewhere == 0 and joint == truth:
            read += 1
        if (elsewhere > 0) != bool(seen):
            read += 1
    print(f"joint  rival reading the hidden line, or miscounting it: {read} -> {('OK' if not read else 'BROKEN')}")
    ok &= not read
    evbad = 0
    for q in qs:
        holes = {h['slot'] for h in q['holes']}
        for v, rows in q['_pair_ev'].items():
            if holes & set(rows) or set(rows) & set(q['slots']):
                evbad += 1
        if not isinstance(q['_pair_b'], int) or q['_pair_b'] < 0:
            evbad += 1
    print(f"evid   evidence rows leaking a hidden or evidence slot: {evbad} -> {('OK' if not evbad else 'BROKEN')}")
    ok &= not evbad
    ex = qs[0]
    print(f"\nexample  line {ex['line']}: [{ex['holes'][0]['address']}] = {ex['holes'][0]['truth']} + [{ex['holes'][1]['address']}] = {ex['holes'][1]['truth']}")
    print(f"         offers {ex['holes'][0]['offer']} x {ex['holes'][1]['offer']}, world {len(ex['slots'])} rows")
    print('\nPAIR OK' if ok else '\nPAIR BROKEN')
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())