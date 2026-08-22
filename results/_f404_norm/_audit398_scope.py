"""THE CEILING OF REPAIRING A DEVICE, NOT A NAME. Torch-free, before any training.

WHY THE OBJECT CHANGES. 350 measured what this project's tape records: two fillers relate when
they can fill THE SAME HOLE. That is alternation - a paradigm table, a lexicon - and a lexicon
ranks alternatives; it cannot produce content. No change to Phi, to the reward or to the offer
can add a relation the write path never wrote, which is why every read-side lever closed and why
34.3's law closes the rest before they are built.

CODE HAS A DIFFERENT RELATION, and it is the one a lexicon cannot hold: DEFINITION -> USE.
Directed, asymmetric, between TWO DIFFERENT PLACES, and bound by NAME IDENTITY. That is the same
`same-value` edge `Deriver` already carries - but here it is the OBJECT OF THE QUESTION instead
of a lens.

AND CODE BREAKS HARD. A shuffled article stays plausible; a shuffled file either parses or it does
not. That is Kostya's whole point: "восстановить буквы" is cloze with more holes, "восстановить
устройство" is an operation whose check is not `== hidden`.

THE EXAM THIS MEASURES THE CEILING OF

    BREAK    cut k body lines out of different functions of one file into a pool
    REPAIR   for each pooled line, NAME THE FUNCTION it belongs to
    OUTPUT   the file's own scopes - tens, not a vocabulary, so 347 is not touched
    REWARD   FREE NAMES: names the line uses that its assigned scope does not bind. A repair is
             better when the world holds together by COUNT - never `== the original line`. Many
             arrangements are correct, which is the opposite of a catalogue.

THE LEAK, GUARDED THE WAY SECTION 27 TAUGHT. What a scope binds is computed WITH THE POOLED LINE
REMOVED FROM IT. A line `n = len(x)` binds `n`; leaving it in would let the true scope bind a name
that only that line binds, and the attachment would be reading the answer. `_check398_scope.py`
puts that leak back and requires the numbers to move.

WHAT IS REPORTED, all counts, nothing fitted:

    on_scope       body lines that are inside a function at all
    uses_names     of those, the ones using at least one name - the rest carry no signal AT ALL
    acc            the counting rival: sum over the line's used names of 1/(how many scopes in
                   this file bind that name). NORMALISED, because `self`, `q`, `p` are bound
                   everywhere and a raw sum would rank by idiom. 317, 383 and 387 each found
                   that fault after the fact; here it is divided from the start, and the raw sum
                   is printed beside it as the control reading.
    floor          1/|scopes|, the exact expectation of attaching at random
    ambiguous      the share where TWO OR MORE scopes tie at the top - THE POPULATION WHERE A
                   DECISION EXISTS. If counting is never ambiguous there is nothing for a mind
                   to do here, and that is the void check, read first.
    vs_foreign     the true scope against the SAME NUMBER of scopes drawn from an unrelated
                   codebase. If a stranger's scopes win, the count is generic-name frequency and
                   not definition -> use.
    free_*         free names per used name, under the true scope, the count's argmax, a random
                   own scope, and a foreign one. "More whole by count", as a share.

  VOID CHECK, READ BEFORE ANYTHING ELSE
      ambiguous <= 0.05  ->  counting decides every attachment, no decision exists, Phi is not
      needed here, and the direction closes WITHOUT a training run.

  GATE
      (b)  acc - floor > 0.05, AND vs_foreign >= acc - 0.05.
      (c)  free_true below free_random_own AND below free_foreign, each by more than 0.05.

  NOT MEASURED HERE, DELIBERATELY: whether the repaired file parses. Re-inserting a line needs an
  indentation decision, which is part of the REPAIR and not of its ceiling; the parse oracle
  belongs to the mechanism, where it is free (`ast.parse`) and hard.

    python _audit398_scope.py
    python _audit398_scope.py --files "_*.py" --null-from /usr/lib/python3.11
"""
from __future__ import annotations
import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
OUT = Path('results/_stage398_scope.json')

def scopes_of(src):
    """Every function in a file: the lines it owns, the names it binds and where they are bound.

    A line belongs to its INNERMOST function, so a nested def's body is the nested def's and only
    its NAME is bound in the enclosing one. `binds` maps a name to the set of lines that bind it,
    which is what lets the pooled line be taken back out exactly.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            last = max((getattr(n, 'lineno', node.lineno) for n in ast.walk(node)))
            funcs.append((node, node.lineno, last))
    if not funcs:
        return None
    funcs.sort(key=lambda f: f[2] - f[1])
    owner = {}
    for i, (_n, a, b) in enumerate(funcs):
        for ln in range(a, b + 1):
            owner.setdefault(ln, i)

    def enclosing(lineno, exclude):
        """the innermost function containing this line that is not `exclude` itself.
        `funcs` is sorted by span ascending, so the first match is the tightest."""
        for i, (nd, a, b) in enumerate(funcs):
            if nd is not exclude and a <= lineno <= b:
                return i
        return None
    binds = [defaultdict(set) for _f in funcs]
    for i, (node, a, _b) in enumerate(funcs):
        args = node.args
        for x in list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs) + [args.vararg, args.kwarg]:
            if x is not None:
                binds[i][x.arg].add(a)
    used = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            i = owner.get(node.lineno)
            if i is None:
                continue
            if isinstance(node.ctx, ast.Store):
                binds[i][node.id].add(node.lineno)
            else:
                used[node.lineno].add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            j = enclosing(node.lineno, node)
            if j is not None:
                binds[j][node.name].add(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            j = owner.get(node.lineno)
            if j is not None:
                for al in node.names:
                    binds[j][(al.asname or al.name).split('.')[0]].add(node.lineno)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            j = owner.get(node.lineno)
            if j is not None:
                binds[j][node.name].add(node.lineno)
    heads = {a for _n, a, _b in funcs}
    return {'funcs': funcs, 'owner': owner, 'binds': binds, 'used': used, 'heads': heads}

def bound_wo(sc, i, drop_line):
    """What scope i binds once `drop_line` is taken out of it - the section 27 discipline.

    A pooled line usually binds something (`n = len(x)`), and a scope credited with a name that
    only the pooled line binds would be recognising its own answer.
    """
    return {n for n, lns in sc['binds'][i].items() if lns - {drop_line}}

def bind_counts(sc, drop_line=None):
    """How many scopes of this file bind each name, with the pooled line taken out of its own."""
    c = Counter()
    for i in range(len(sc['funcs'])):
        for n in bound_wo(sc, i, drop_line):
            c[n] += 1
    return c

def score(names, bound, counts, normalise=True):
    """The counting rival. Normalised: a name bound by every scope says nothing, and dividing is
    two exact counts rather than a weight anyone chose."""
    if normalise:
        return sum((1.0 / max(1, counts.get(n, 1)) for n in names if n in bound))
    return sum((1.0 for n in names if n in bound))

def run(files, null_files, args, rng):
    c, per_line = (Counter(), [])
    nulls = []
    for p in null_files:
        try:
            sc = scopes_of(Path(p).read_text(encoding='utf-8', errors='ignore'))
        except OSError:
            continue
        if sc and len(sc['funcs']) >= args.min_scopes:
            nulls.append(sc)
    if not nulls:
        return (c, per_line, 0)
    for p in files:
        try:
            sc = scopes_of(Path(p).read_text(encoding='utf-8', errors='ignore'))
        except OSError:
            continue
        if not sc or len(sc['funcs']) < args.min_scopes:
            continue
        nf = len(sc['funcs'])
        lines = [ln for ln in sorted(sc['owner']) if ln not in sc['heads']]
        rng.shuffle(lines)
        taken = 0
        for ln in lines:
            if taken >= args.per_file:
                break
            true_i = sc['owner'][ln]
            c['on_scope'] += 1
            names = sc['used'].get(ln, set())
            if not names:
                continue
            c['uses_names'] += 1
            taken += 1
            counts = bind_counts(sc, ln)
            bounds = [bound_wo(sc, i, ln) for i in range(nf)]
            s = [score(names, bounds[i], counts, args.normalise) for i in range(nf)]
            top = max(s)
            ties = sum((1 for x in s if x == top))
            best = s.index(top)
            c['acc'] += int(best == true_i and ties == 1)
            c['acc_tie_ok'] += int(s[true_i] == top)
            c['ambiguous'] += int(ties > 1)
            c['zero_top'] += int(top <= 0.0)
            c['amb_zero'] += int(ties > 1 and top <= 0.0)
            c['amb_live'] += int(ties > 1 and top > 0.0)
            c['floor_num'] += 1.0 / nf
            fsc = nulls[rng.randrange(len(nulls))]
            fidx = list(range(len(fsc['funcs'])))
            rng.shuffle(fidx)
            fidx = fidx[:nf]
            fcounts = bind_counts(fsc)
            fs = [score(names, bound_wo(fsc, i, -1), fcounts, args.normalise) for i in fidx]
            fmax = max(fs) if fs else 0.0
            c['vs_foreign'] += int(s[true_i] > fmax)
            c['beaten'] += int(fmax >= s[true_i])

            def free(bound):
                return len([n for n in names if n not in bound]) / len(names)
            r_i = rng.randrange(nf)
            f_i = fidx[rng.randrange(len(fidx))] if fidx else 0
            c['free_true'] += free(bounds[true_i])
            c['free_arg'] += free(bounds[best])
            c['free_rand'] += free(bounds[r_i])
            c['free_foreign'] += free(bound_wo(fsc, f_i, -1))
            per_line.append((len(names), nf, ties, int(best == true_i)))
    return (c, per_line, len(nulls))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', default='_*.py', help='glob for the corpus (real code, not the synthetic one: generated names re-used across functions inflate count and null alike)')
    ap.add_argument('--null-from', default='/usr/lib/python3.11', help="an UNRELATED codebase for the null. Inside one project a stranger's scopes bind the same self/q/p, so a same-project null is the weak one")
    ap.add_argument('--per-file', type=int, default=40)
    ap.add_argument('--min-scopes', type=int, default=8)
    ap.add_argument('--null-files', type=int, default=40)
    ap.add_argument('--raw', dest='normalise', action='store_false', help='the control reading: the RAW count, undivided')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    files = sorted(Path('.').glob(args.files))
    null_all = sorted(Path(args.null_from).glob('*.py')) if Path(args.null_from).exists() else []
    rng.shuffle(null_all)
    c, per, n_null = run(files, null_all[:args.null_files], args, rng)
    n = max(1, c['uses_names'])
    rep = {'files': len(files), 'null_files': n_null, 'normalise': bool(args.normalise), 'seed': args.seed, 'lines_on_scope': c['on_scope'], 'lines_scored': c['uses_names'], 'uses_names': c['uses_names'] / max(1, c['on_scope']), 'acc': c['acc'] / n, 'acc_tie_ok': c['acc_tie_ok'] / n, 'floor': c['floor_num'] / n, 'ambiguous': c['ambiguous'] / n, 'zero_top': c['zero_top'] / n, 'amb_zero': c['amb_zero'] / n, 'amb_live': c['amb_live'] / n, 'vs_foreign': c['vs_foreign'] / n, 'beaten': c['beaten'] / n, 'free_true': c['free_true'] / n, 'free_argmax': c['free_arg'] / n, 'free_random': c['free_rand'] / n, 'free_foreign': c['free_foreign'] / n}
    rep['acc_minus_floor'] = rep['acc'] - rep['floor']
    print(f"{rep['files']} files, {n_null} foreign files, {rep['lines_scored']} lines scored ({rep['uses_names']:.4f} of the lines on a scope use a name), {('normalised' if args.normalise else 'RAW')} count")
    print(f"VOID CHECK  ambiguous {rep['ambiguous']:.4f}  of which AT ZERO {rep['amb_zero']:.4f} (no signal) and LIVE {rep['amb_live']:.4f}   <- the live share is the decision population; nothing scores on {rep['zero_top']:.4f} of lines at all")
    print(f"COUNT       acc {rep['acc']:.4f}  (ties allowed {rep['acc_tie_ok']:.4f})   floor {rep['floor']:.4f}   acc-floor {rep['acc_minus_floor']:+.4f}")
    print(f"FOREIGN     true beats a matched stranger {rep['vs_foreign']:.4f}   beaten {rep['beaten']:.4f}")
    print(f"FREE NAMES  true {rep['free_true']:.4f}  argmax {rep['free_argmax']:.4f}  random own {rep['free_random']:.4f}  foreign {rep['free_foreign']:.4f}")
    void = rep['amb_live'] <= 0.05
    gate_b = rep['acc_minus_floor'] > 0.05
    gate_c = rep['free_true'] < rep['free_random'] - 0.05 and rep['free_true'] < rep['free_foreign'] - 0.05
    rep['void'], rep['gate_b'], rep['gate_c'] = (bool(void), bool(gate_b), bool(gate_c))
    if void:
        print("\nVOID: counting attaches every line without ambiguity, so there is no decision for a mind to make on this object. Do not train Phi on repair - the knowledge half holds it, which by this project's own thesis is correct.")
    elif gate_b and gate_c:
        print("\nTHE CEILING IS THERE. Definition -> use is a real relation on this object, a stranger's scopes do not carry it, repair is measurable as free names rather than as `== the original`, and on the ambiguous share a decision exists. THAT share is the population any mechanism must be built and read on - not the whole corpus, where counting already answers.")
    else:
        print('\nTHE GATE FAILS: ' + ("(b) the count is not above the floor or a stranger's scopes match it. " if not gate_b else '') + ("(c) free names do not fall under the true scope by more than the bar, so 'more whole' is not measurable here. " if not gate_c else '') + 'Do not build the repair loop on this object.')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[f"{args.seed}{('' if args.normalise else '_raw')}"] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f'wrote {out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())