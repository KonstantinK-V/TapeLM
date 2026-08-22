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
v0 = v2('results/_stage398_scope.json')

def scopes_of(v3):
    """Every function in a file: the lines it owns, the names it binds and where they are bound.

    A line belongs to its INNERMOST function, so a nested def's body is the nested def's and only
    its NAME is bound in the enclosing one. `binds` maps a name to the set of lines that bind it,
    which is what lets the pooled line be taken back out exactly.
    """
    try:
        v34 = v69.v68(v3)
    except v35:
        return None
    v4 = []
    for v5 in v69.v36(v34):
        if v70(v5, (v69.v102, v69.v103)):
            v71 = v63((v134(v26, 'lineno', v5.v40) for v26 in v69.v36(v5)))
            v4.v104((v5, v5.v40, v71))
    if not v4:
        return None
    v4.v37(key=lambda v126: v126[2] - v126[1])
    v6 = {}
    for v11, (v72, v45, v73) in v38(v4):
        for v39 in v47(v45, v73 + 1):
            v6.v105(v39, v11)

    def enclosing(v40, v41):
        """the innermost function containing this line that is not `exclude` itself.
        `funcs` is sorted by span ascending, so the first match is the tightest."""
        for v11, (v106, v45, v73) in v38(v4):
            if v106 is not v41 and v45 <= v40 <= v73:
                return v11
        return None
    v7 = [v43(v44) for v74 in v4]
    for v11, (v5, v45, v75) in v38(v4):
        v20 = v5.v20
        for v42 in v120(v20.v20) + v120(v20.v135) + v120(v20.v127) + [v20.v107, v20.v108]:
            if v42 is not None:
                v7[v11][v42.v140].v128(v45)
    v8 = v43(v44)
    for v5 in v69.v36(v34):
        if v70(v5, v69.v76):
            v11 = v6.v109(v5.v40)
            if v11 is None:
                continue
            if v70(v5.v110, v69.v111):
                v7[v11][v5.v129].v128(v5.v40)
            else:
                v8[v5.v40].v128(v5.v129)
        elif v70(v5, (v69.v102, v69.v103, v69.v130)):
            v112 = v131(v5.v40, v5)
            if v112 is not None:
                v7[v112][v5.v139].v128(v5.v40)
        elif v70(v5, (v69.v136, v69.v137)):
            v112 = v6.v109(v5.v40)
            if v112 is not None:
                for v138 in v5.v14:
                    v7[v112][(v138.v143 or v138.v139).v142('.')[0]].v128(v5.v40)
        elif v70(v5, v69.v141) and v5.v139:
            v112 = v6.v109(v5.v40)
            if v112 is not None:
                v7[v112][v5.v139].v128(v5.v40)
    v9 = {v45 for v72, v45, v75 in v4}
    return {'funcs': v4, 'owner': v6, 'binds': v7, 'used': v8, 'heads': v9}

def bound_wo(v10, v11, v12):
    """What scope i binds once `drop_line` is taken out of it - the section 27 discipline.

    A pooled line usually binds something (`n = len(x)`), and a scope credited with a name that
    only the pooled line binds would be recognising its own answer.
    """
    return {v26 for v26, v113 in v10['binds'][v11].v114() if v113 - {v12}}

def bind_counts(v10, v12=None):
    """How many scopes of this file bind each name, with the pooled line taken out of its own."""
    v13 = v46()
    for v11 in v47(v77(v10['funcs'])):
        for v26 in v78(v10, v11, v12):
            v13[v26] += 1
    return v13

def score(v14, v15, v16, v17=True):
    """The counting rival. Normalised: a name bound by every scope says nothing, and dividing is
    two exact counts rather than a weight anyone chose."""
    if v17:
        return v48((1.0 / v63(1, v16.v109(v26, 1)) for v26 in v14 if v26 in v15))
    return v48((1.0 for v26 in v14 if v26 in v15))

def run(v18, v19, v20, v21):
    v13, v49 = (v46(), [])
    v22 = []
    for v23 in v19:
        try:
            v10 = v115(v2(v23).v125(encoding='utf-8', errors='ignore'))
        except v79:
            continue
        if v10 and v77(v10['funcs']) >= v20.v116:
            v22.v104(v10)
    if not v22:
        return (v13, v49, 0)
    for v23 in v18:
        try:
            v10 = v115(v2(v23).v125(encoding='utf-8', errors='ignore'))
        except v79:
            continue
        if not v10 or v77(v10['funcs']) < v20.v116:
            continue
        v50 = v77(v10['funcs'])
        v51 = [v39 for v39 in v58(v10['owner']) if v39 not in v10['heads']]
        v21.v59(v51)
        v52 = 0
        for v39 in v51:
            if v52 >= v20.v117:
                break
            v80 = v10['owner'][v39]
            v13['on_scope'] += 1
            v14 = v10['used'].v109(v39, v44())
            if not v14:
                continue
            v13['uses_names'] += 1
            v52 += 1
            v16 = v118(v10, v39)
            v81 = [v78(v10, v11, v39) for v11 in v47(v50)]
            v82 = [v132(v14, v81[v11], v16, v20.v17) for v11 in v47(v50)]
            v83 = v63(v82)
            v84 = v48((1 for v42 in v82 if v42 == v83))
            v85 = v82.v119(v83)
            v13['acc'] += v1(v85 == v80 and v84 == 1)
            v13['acc_tie_ok'] += v1(v82[v80] == v83)
            v13['ambiguous'] += v1(v84 > 1)
            v13['zero_top'] += v1(v83 <= 0.0)
            v13['amb_zero'] += v1(v84 > 1 and v83 <= 0.0)
            v13['amb_live'] += v1(v84 > 1 and v83 > 0.0)
            v13['floor_num'] += 1.0 / v50
            v86 = v22[v21.v121(v77(v22))]
            v87 = v120(v47(v77(v86['funcs'])))
            v21.v59(v87)
            v87 = v87[:v50]
            v88 = v118(v86)
            v89 = [v132(v14, v78(v86, v11, -1), v88, v20.v17) for v11 in v87]
            v90 = v63(v89) if v89 else 0.0
            v13['vs_foreign'] += v1(v82[v80] > v90)
            v13['beaten'] += v1(v90 >= v82[v80])

            def free(v15):
                return v77([v26 for v26 in v14 if v26 not in v15]) / v77(v14)
            v91 = v21.v121(v50)
            v92 = v87[v21.v121(v77(v87))] if v87 else 0
            v13['free_true'] += v122(v81[v80])
            v13['free_arg'] += v122(v81[v85])
            v13['free_rand'] += v122(v81[v91])
            v13['free_foreign'] += v122(v78(v86, v92, -1))
            v49.v104((v77(v14), v50, v84, v1(v85 == v80)))
    return (v13, v49, v77(v22))

def main() -> v1:
    v24 = v93.v53()
    v24.v54('--files', default='_*.py', help='glob for the corpus (real code, not the synthetic one: generated names re-used across functions inflate count and null alike)')
    v24.v54('--null-from', default='/usr/lib/python3.11', help="an UNRELATED codebase for the null. Inside one project a stranger's scopes bind the same self/q/p, so a same-project null is the weak one")
    v24.v54('--per-file', type=v1, default=40)
    v24.v54('--min-scopes', type=v1, default=8)
    v24.v54('--null-files', type=v1, default=40)
    v24.v54('--raw', dest='normalise', action='store_false', help='the control reading: the RAW count, undivided')
    v24.v54('--seed', type=v1, default=1337)
    v24.v54('--out', default=v123(v0))
    v20 = v24.v55()
    v21 = v94.v56(v20.v57)
    v18 = v58(v2('.').v95(v20.v18))
    v25 = v58(v2(v20.v133).v95('*.py')) if v2(v20.v133).v96() else []
    v21.v59(v25)
    v13, v60, v61 = v62(v18, v25[:v20.v19], v20, v21)
    v26 = v63(1, v13['uses_names'])
    v27 = {'files': v77(v18), 'null_files': v61, 'normalise': v97(v20.v17), 'seed': v20.v57, 'lines_on_scope': v13['on_scope'], 'lines_scored': v13['uses_names'], 'uses_names': v13['uses_names'] / v63(1, v13['on_scope']), 'acc': v13['acc'] / v26, 'acc_tie_ok': v13['acc_tie_ok'] / v26, 'floor': v13['floor_num'] / v26, 'ambiguous': v13['ambiguous'] / v26, 'zero_top': v13['zero_top'] / v26, 'amb_zero': v13['amb_zero'] / v26, 'amb_live': v13['amb_live'] / v26, 'vs_foreign': v13['vs_foreign'] / v26, 'beaten': v13['beaten'] / v26, 'free_true': v13['free_true'] / v26, 'free_argmax': v13['free_arg'] / v26, 'free_random': v13['free_rand'] / v26, 'free_foreign': v13['free_foreign'] / v26}
    v27['acc_minus_floor'] = v27['acc'] - v27['floor']
    v64(f"{v27['files']} files, {v61} foreign files, {v27['lines_scored']} lines scored ({v27['uses_names']:.4f} of the lines on a scope use a name), {('normalised' if v20.v17 else 'RAW')} count")
    v64(f"VOID CHECK  ambiguous {v27['ambiguous']:.4f}  of which AT ZERO {v27['amb_zero']:.4f} (no signal) and LIVE {v27['amb_live']:.4f}   <- the live share is the decision population; nothing scores on {v27['zero_top']:.4f} of lines at all")
    v64(f"COUNT       acc {v27['acc']:.4f}  (ties allowed {v27['acc_tie_ok']:.4f})   floor {v27['floor']:.4f}   acc-floor {v27['acc_minus_floor']:+.4f}")
    v64(f"FOREIGN     true beats a matched stranger {v27['vs_foreign']:.4f}   beaten {v27['beaten']:.4f}")
    v64(f"FREE NAMES  true {v27['free_true']:.4f}  argmax {v27['free_argmax']:.4f}  random own {v27['free_random']:.4f}  foreign {v27['free_foreign']:.4f}")
    v28 = v27['amb_live'] <= 0.05
    v29 = v27['acc_minus_floor'] > 0.05
    v30 = v27['free_true'] < v27['free_random'] - 0.05 and v27['free_true'] < v27['free_foreign'] - 0.05
    v27['void'], v27['gate_b'], v27['gate_c'] = (v97(v28), v97(v29), v97(v30))
    if v28:
        v64("\nVOID: counting attaches every line without ambiguity, so there is no decision for a mind to make on this object. Do not train Phi on repair - the knowledge half holds it, which by this project's own thesis is correct.")
    elif v29 and v30:
        v64("\nTHE CEILING IS THERE. Definition -> use is a real relation on this object, a stranger's scopes do not carry it, repair is measurable as free names rather than as `== the original`, and on the ambiguous share a decision exists. THAT share is the population any mechanism must be built and read on - not the whole corpus, where counting already answers.")
    else:
        v64('\nTHE GATE FAILS: ' + ("(b) the count is not above the floor or a stranger's scopes match it. " if not v29 else '') + ("(c) free names do not fall under the true scope by more than the bar, so 'more whole' is not measurable here. " if not v30 else '') + 'Do not build the repair loop on this object.')
    v31 = v2(v20.v31)
    v31.v98.v65(parents=True, exist_ok=True)
    v32 = v124.v99(v31.v125()) if v31.v96() else {}
    v32[f"{v20.v57}{('' if v20.v17 else '_raw')}"] = v27
    v31.v66(v124.v100(v32, indent=1))
    v64(f'wrote {v31}')
    return 0
if v33 == '__main__':
    raise v67(v101())