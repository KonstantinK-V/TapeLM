"""A FAMILY OF FEATURES, FITTED, AND A VOCABULARY THAT MAY LEAK. Torch-free, held out by file.

WHY A FAMILY. Every step of 398-403 declared ONE count, measured it alone, got a coin and closed
a direction. A single dimension being a coin is what to EXPECT even when the joint carries, and
nothing in this project has ever measured a joint. The discipline that forbids fitting can only
ever reject; the honesty it protects is kept here by a HELD-OUT SPLIT BY FILE - fitting is allowed
because it is scored on files the fit never saw.

WHY A VOCABULARY CHANNEL, AND WHY IT IS NOT A LEAK. Kostya's point: a mind knowing the word
"physics" is not a mind holding a fact. A word that recurs ACROSS MANY TAPES is language; a word
that lives in one tape is that tape's content. That difference is a COUNT - in how many separate
files does this identifier occur - so the contract becomes measurable rather than declared:

    channel A   NAME-FREE. Counts of structure only; no identifier reaches it.
    channel B   VOCABULARY. Identifiers, admitted ONLY if they occur in at least --vocab-files
                different files. One tape's names cannot enter.

and the two are fitted separately and together, so what each is worth is visible.

THE ARENA is 403's - order inside one def, teacher forcing - and THE POPULATION is where the
tape's own rule is SILENT: states in which two or more pool lines are SAFE, so def-use permits
several and only the order decides. Everywhere else counting already answers, which is 38.3's law.

    target     which safe line is actually next
    floor      1 / |safe|, exact and closed form

  VOID CHECK, READ FIRST
      the share of states with two or more safe lines. Under 0.05 there is nothing to decide.

  GATE, DECLARED BEFORE THE RUN
      PRIMARY     A on held-out files - floor > 0.05
      VOCABULARY  (A+B) - A > 0.05, or the word channel bought nothing
      TRANSPLANT  A's gain on a FOREIGN corpus within 0.03 of its home gain -> it is structure.
                  If (A+B)'s extra gain vanishes there, the vocabulary was content and the
                  contract says drop it.
      EQUIVARIANCE  A identical under renaming; B must NOT be, or the two channels are one.

    python _audit404_family.py
    python _audit404_family.py --transplant /usr/lib/python3.11
"""
from __future__ import annotations
import argparse
import ast
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
OUT = Path('results/_stage404_family.json')
TYPES = ('Call', 'If', 'For', 'Return', 'Assign', 'Attribute', 'Compare', 'BinOp')
_ROWS = {}

def rows_cached(path, max_body):
    """Parsed once per file. `max_body` keeps the arena to bodies a repair could plausibly be
    about: `unblocks` is quadratic in the body, and one 1000-line function costs more than the
    rest of the corpus together while telling us nothing a 20-line one does not."""
    key = (str(path), max_body)
    got = _ROWS.get(key)
    if got is None:
        try:
            src = Path(path).read_text(encoding='utf-8', errors='ignore')
        except OSError:
            src = ''
        got = [b for b in rows_of(src) if 3 <= len(b) <= max_body]
        _ROWS[key] = got
    return got

def rows_of(src):
    """Every def's body: per line the names it stores and loads, its node types and its ids."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            last = max((getattr(n, 'lineno', node.lineno) for n in ast.walk(node)))
            funcs.append((node, node.lineno, last))
    funcs.sort(key=lambda f: f[2] - f[1])
    owner, heads = ({}, {a for _n, a, _b in funcs})
    for i, (_n, a, b) in enumerate(funcs):
        for ln in range(a, b + 1):
            owner.setdefault(ln, i)
    st, ld, ty, ids = (defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set))
    for node in ast.walk(tree):
        ln = getattr(node, 'lineno', None)
        if ln is None:
            continue
        ty[ln].add(type(node).__name__)
        if isinstance(node, ast.Name):
            (st if isinstance(node.ctx, ast.Store) else ld)[ln].add(node.id)
            ids[ln].add(node.id)
        elif isinstance(node, ast.Attribute):
            ids[ln].add(node.attr)
    out = []
    for i in range(len(funcs)):
        lns = sorted((l for l, o in owner.items() if o == i and l not in heads))
        body = [(l, st.get(l, set()), ld.get(l, set()), ty.get(l, set()), ids.get(l, set())) for l in lns]
        body = [r for r in body if r[1] or r[2]]
        if len(body) >= 3:
            out.append(body)
    return out

def feats_A(cand, pool, placed_stores, keys, t, n_rows, last_store, safe_ix):
    """CHANNEL A - no identifier reaches this. Every entry is a count of structure."""
    _l, s, d, ty, _ids = cand
    unblocks = 0
    for r in pool:
        if r is cand:
            continue
        need = (r[2] & keys) - placed_stores
        if need and (not need - s):
            unblocks += 1
    blockers = sum((1 for r in pool if r is not cand and cand[2] & keys & r[1]))
    f = [1.0 if s else 0.0, float(len(d)), float(len(s)), float(len(ty)), float(unblocks), float(blockers), float(len(pool)), float(len(safe_ix)), t / max(1, n_rows), 1.0 if last_store else 0.0]
    f += [1.0 if k in ty else 0.0 for k in TYPES]
    return f

def feats_B(cand, vocab, dim):
    """CHANNEL B - identifiers, but only those the VOCABULARY admits: a word must occur in at
    least `--vocab-files` different files to be language rather than one tape's content."""
    v = [0.0] * dim
    for w in cand[4]:
        if w in vocab:
            v[hash(w) % dim] = 1.0
    return v

def states_of(body, vocab, dim, want_b, min_safe):
    keys = set()
    stores, loads = (set(), set())
    for _l, s, d, _t, _i in body:
        stores |= s
        loads |= d
    keys = stores & loads
    if not keys:
        return
    for t in range(1, len(body)):
        placed, pool = (body[:t], body[t:])
        have = set()
        for _l, s, _d, _t2, _i in placed:
            have |= s
        safe_ix = [i for i, r in enumerate(pool) if not (r[2] & keys) - have]
        if len(safe_ix) < min_safe or 0 not in safe_ix:
            continue
        last_store = bool(placed[-1][1])
        xs = []
        for i in safe_ix:
            a = feats_A(pool[i], pool, have, keys, t, len(body), last_store, safe_ix)
            xs.append(a + (feats_B(pool[i], vocab, dim) if want_b else []))
        yield (xs, safe_ix.index(0))

def fit(data, dim, epochs, lr, seed):
    """Listwise logistic fit in plain python: one softmax over the safe candidates of a state."""
    if not data:
        return []
    w = [0.0] * len(data[0][0][0])
    rng = random.Random(seed)
    idx = list(range(len(data)))
    for _e in range(epochs):
        rng.shuffle(idx)
        for k in idx:
            xs, y = data[k]
            zs = [sum((wi * xi for wi, xi in zip(w, x))) for x in xs]
            m = max(zs)
            ex = [math.exp(z - m) for z in zs]
            ssum = sum(ex) or 1.0
            for j, x in enumerate(xs):
                g = ex[j] / ssum - (1.0 if j == y else 0.0)
                if g:
                    for c, xc in enumerate(x):
                        if xc:
                            w[c] -= lr * g * xc
    return w

def acc(data, w):
    if not data:
        return (float('nan'), float('nan'))
    hit = flo = 0.0
    for xs, y in data:
        zs = [sum((wi * xi for wi, xi in zip(w, x))) for x in xs]
        top = max(zs)
        best = [j for j in range(len(zs)) if zs[j] == top]
        hit += 1.0 / len(best) if y in best else 0.0
        flo += 1.0 / len(xs)
    return (hit / len(data), flo / len(data))

def collect(files, vocab, dim, want_b, min_safe, cap, max_body=40):
    data = []
    for p in files:
        for body in rows_cached(p, max_body):
            for st in states_of(body, vocab, dim, want_b, min_safe):
                data.append(st)
                if len(data) >= cap:
                    return data
    return data

def build_vocab(files, k):
    """A word is LANGUAGE when it occurs in at least k different files, and CONTENT otherwise."""
    seen = Counter()
    for p in files:
        try:
            src = Path(p).read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        ws = set()
        for body in rows_of(src):
            for r in body:
                ws |= r[4]
        for w in ws:
            seen[w] += 1
    return ({w for w, n in seen.items() if n >= k}, seen)

def norm_files(files, out_dir, rename):
    """Both equivariance arms are unparsed from the AST, so only the identifiers differ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in files:
        try:
            tree = ast.parse(Path(p).read_text(encoding='utf-8', errors='ignore'))
        except SyntaxError:
            continue
        if rename:
            names = {}
            for nd in ast.walk(tree):
                if isinstance(nd, ast.Name):
                    nd.id = names.setdefault(nd.id, f'v{len(names)}')
                elif isinstance(nd, ast.arg):
                    nd.arg = names.setdefault(nd.arg, f'v{len(names)}')
                elif isinstance(nd, ast.Attribute):
                    nd.attr = names.setdefault(nd.attr, f'v{len(names)}')
        try:
            txt = ast.unparse(tree)
            ast.parse(txt)
        except Exception:
            continue
        q = out_dir / Path(p).name
        q.write_text(txt, encoding='utf-8')
        out.append(q)
    return out

def arm(train_f, test_f, vocab, dim, want_b, args):
    tr = collect(train_f, vocab, dim, want_b, args.min_safe, args.cap, args.max_body)
    te = collect(test_f, vocab, dim, want_b, args.min_safe, args.cap, args.max_body)
    w = fit(tr, dim, args.epochs, args.lr, args.seed)
    a, floor = acc(te, w)
    return (a, floor, len(tr), len(te), w)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', default='_*.py')
    ap.add_argument('--transplant', default='/usr/lib/python3.11')
    ap.add_argument('--vocab-files', type=int, default=5)
    ap.add_argument('--vocab-dim', type=int, default=64)
    ap.add_argument('--min-safe', type=int, default=2)
    ap.add_argument('--max-body', type=int, default=40)
    ap.add_argument('--foreign', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--cap', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    files = sorted(Path('.').glob(args.files))
    rng.shuffle(files)
    half = len(files) // 2
    train_f, test_f = (files[:half], files[half:])
    vocab, seen = build_vocab(train_f, args.vocab_files)
    n_all = n_dec = 0
    for p_ in files:
        for body in rows_cached(p_, args.max_body):
            for xs, _y in states_of(body, vocab, args.vocab_dim, False, 1):
                n_all += 1
                n_dec += int(len(xs) >= 2)
    rep = {'seed': args.seed, 'files': len(files), 'vocab': len(vocab), 'vocab_files': args.vocab_files, 'states_all': n_all, 'states_decide': n_dec, 'decide_share': n_dec / max(1, n_all)}
    print(f"{rep['files']} files, vocabulary {rep['vocab']} words seen in >= {args.vocab_files} files")
    print(f"VOID CHECK  states with a decision {rep['decide_share']:.4f} ({n_dec} of {n_all})  <- read first")
    a_acc, floor, ntr, nte, _wa = arm(train_f, test_f, vocab, args.vocab_dim, False, args)
    ab_acc, _f2, _n1, _n2, _wb = arm(train_f, test_f, vocab, args.vocab_dim, True, args)
    rep.update({'A': a_acc, 'AB': ab_acc, 'floor': floor, 'n_train': ntr, 'n_test': nte})
    rep['A_gain'] = a_acc - floor
    rep['B_adds'] = ab_acc - a_acc
    print(f"HELD OUT    floor {floor:.4f}   A (name-free) {a_acc:.4f} ({rep['A_gain']:+.4f})   A+B (vocabulary) {ab_acc:.4f} ({rep['B_adds']:+.4f} over A)")
    print(f'            fitted on {ntr} states of {len(train_f)} files, read on {nte} states of {len(test_f)} files')
    te1 = collect(test_f, vocab, args.vocab_dim, False, args.min_safe, args.cap, args.max_body)
    nfeat = len(te1[0][0][0]) if te1 else 0
    singles = []
    for j in range(nfeat):
        best = max(acc(te1, [1.0 if c == j else 0.0 for c in range(nfeat)])[0], acc(te1, [-1.0 if c == j else 0.0 for c in range(nfeat)])[0])
        singles.append(best)
    rep['singles'] = singles
    rep['best_single'] = max(singles) if singles else float('nan')
    rep['family_over_single'] = a_acc - rep['best_single']
    print(f"ABLATION    best single feature {rep['best_single']:.4f} (#{singles.index(rep['best_single'])})   family over it {rep['family_over_single']:+.4f}")
    tp = sorted(Path(args.transplant).glob('*.py')) if Path(args.transplant).exists() else []
    rng.shuffle(tp)
    tp = tp[:args.foreign]
    if tp:
        ta, tfloor, _n1, nte2, _w = arm(train_f, tp, vocab, args.vocab_dim, False, args)
        tab, _tf2, _n3, _n4, _w2 = arm(train_f, tp, vocab, args.vocab_dim, True, args)
        rep.update({'T_A': ta, 'T_AB': tab, 'T_floor': tfloor, 'T_n': nte2, 'T_A_gain': ta - tfloor, 'T_B_adds': tab - ta})
        print(f'TRANSPLANT  floor {tfloor:.4f}   A {ta:.4f} ({ta - tfloor:+.4f})   A+B {tab:.4f} ({tab - ta:+.4f} over A)   on {nte2} foreign states')
    d = Path(args.out).parent
    nf = norm_files(files, d / '_f404_norm', False)
    rf = norm_files(files, d / '_f404_ren', True)
    for tag, fl in (('norm', nf), ('ren', rf)):
        h = len(fl) // 2
        v, _s = build_vocab(fl[:h], args.vocab_files)
        aa, _fl, _n1, _n2, _w = arm(fl[:h], fl[h:], v, args.vocab_dim, False, args)
        bb, _fl2, _n3, _n4, _w2 = arm(fl[:h], fl[h:], v, args.vocab_dim, True, args)
        rep[f'A_{tag}'], rep[f'AB_{tag}'] = (aa, bb)
    print(f"RENAMING    A {rep['A_norm']:.4f} -> {rep['A_ren']:.4f} (must be identical)   A+B {rep['AB_norm']:.4f} -> {rep['AB_ren']:.4f} (must move)")
    void = rep['decide_share'] <= 0.05
    prim = rep['A_gain'] > 0.05
    vocab_ok = rep['B_adds'] > 0.05
    trans = 'T_A_gain' in rep and rep['T_A_gain'] >= rep['A_gain'] - 0.03
    equi = abs(rep['A_norm'] - rep['A_ren']) < 1e-09
    rep.update({'void': bool(void), 'primary': bool(prim), 'vocab_pays': bool(vocab_ok), 'transplants': bool(trans), 'equivariant': bool(equi)})
    print('\n' + ('VOID: nothing to decide in this arena.' if void else ('THE FAMILY CARRIES' if prim else 'THE FAMILY IS A COIN') + f"; the vocabulary {('ADDS' if vocab_ok else 'adds nothing')}; A {('TRANSPLANTS' if trans else 'does NOT transplant')}; A is {('name-free' if equi else 'NOT name-free')}."))
    rep['is_family'] = bool(rep['family_over_single'] > 0.05)
    print('A JOINT, not another count.' if rep['is_family'] else 'NOT A JOINT: one feature alone reaches the family, so this is the eleventh count and the argument for fitting does not hold.')
    if prim and trans and equi:
        print('Structure fitted jointly and held out by file is worth more than any single count this project has declared, and it survives a foreign corpus - which is the separation contract measured rather than asserted.')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())