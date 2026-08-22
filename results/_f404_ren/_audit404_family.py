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
v0 = v4('results/_stage404_family.json')
v1 = ('Call', 'If', 'For', 'Return', 'Assign', 'Attribute', 'Compare', 'BinOp')
v2 = {}

def rows_cached(v5, v6):
    """Parsed once per file. `max_body` keeps the arena to bodies a repair could plausibly be
    about: `unblocks` is quadratic in the body, and one 1000-line function costs more than the
    rest of the corpus together while telling us nothing a 20-line one does not."""
    v7 = (v146(v5), v6)
    v8 = v2.v76(v7)
    if v8 is None:
        try:
            v9 = v4(v5).v196(encoding='utf-8', errors='ignore')
        except v147:
            v9 = ''
        v8 = [v148 for v148 in v168(v9) if 3 <= v158(v148) <= v6]
        v2[v7] = v8
    return v8

def rows_of(v9):
    """Every def's body: per line the names it stores and loads, its node types and its ids."""
    try:
        v77 = v150.v149(v9)
    except v78:
        return []
    v10 = []
    for v11 in v150.v79(v77):
        if v151(v11, (v150.v197, v150.v198)):
            v152 = v166((v155(v217, 'lineno', v11.v210) for v217 in v150.v79(v11)))
            v10.v174((v11, v11.v210, v152))
    v10.v80(key=lambda v25: v25[2] - v25[1])
    v81, v82 = ({}, {v116 for v153, v116, v211 in v10})
    for v13, (v153, v116, v148) in v83(v10):
        for v84 in v89(v116, v148 + 1):
            v81.v199(v84, v13)
    v85, v86, v87, v88 = (v154(v96), v154(v96), v154(v96), v154(v96))
    for v11 in v150.v79(v77):
        v84 = v155(v11, 'lineno', None)
        if v84 is None:
            continue
        v87[v84].v156(v212(v11).v75)
        if v151(v11, v150.v157):
            (v85 if v151(v11.v230, v150.v231) else v86)[v84].v156(v11.v200)
            v88[v84].v156(v11.v200)
        elif v151(v11, v150.v201):
            v88[v84].v156(v11.v213)
    v12 = []
    for v13 in v89(v158(v10)):
        v90 = v122((v202 for v202, v220 in v81.v218() if v220 == v13 and v202 not in v82))
        v30 = [(v202, v85.v76(v202, v96()), v86.v76(v202, v96()), v87.v76(v202, v96()), v88.v76(v202, v96())) for v202 in v90]
        v30 = [v23 for v23 in v30 if v23[1] or v23[2]]
        if v158(v30) >= 3:
            v12.v174(v30)
    return v12

def feats_A(v14, v15, v16, v17, v18, v19, v20, v21):
    """CHANNEL A - no identifier reaches this. Every entry is a count of structure."""
    v91, v92, v65, v87, v93 = v14
    v22 = 0
    for v23 in v15:
        if v23 is v14:
            continue
        v94 = (v23[2] & v17) - v16
        if v94 and (not v94 - v92):
            v22 += 1
    v24 = v95((1 for v23 in v15 if v23 is not v14 and v14[2] & v17 & v23[1]))
    v25 = [1.0 if v92 else 0.0, v159(v158(v65)), v159(v158(v92)), v159(v158(v87)), v159(v22), v159(v24), v159(v158(v15)), v159(v158(v21)), v18 / v166(1, v19), 1.0 if v20 else 0.0]
    v25 += [1.0 if v45 in v87 else 0.0 for v45 in v1]
    return v25

def feats_B(v14, v26, v27):
    """CHANNEL B - identifiers, but only those the VOCABULARY admits: a word must occur in at
    least `--vocab-files` different files to be language rather than one tape's content."""
    v28 = [0.0] * v27
    for v29 in v14[4]:
        if v29 in v26:
            v28[v221(v29) % v27] = 1.0
    return v28

def states_of(v30, v26, v27, v31, v32):
    v17 = v96()
    v97, v98 = (v96(), v96())
    for v91, v92, v65, v99, v100 in v30:
        v97 |= v92
        v98 |= v65
    v17 = v97 & v98
    if not v17:
        return
    for v18 in v89(1, v158(v30)):
        v102, v15 = (v30[:v18], v30[v18:])
        v101 = v96()
        for v91, v92, v160, v161, v100 in v102:
            v101 |= v92
        v21 = [v13 for v13, v23 in v83(v15) if not (v23[2] & v17) - v101]
        if v158(v21) < v32 or 0 not in v21:
            continue
        v20 = v143(v102[-1][1])
        v103 = []
        for v13 in v21:
            v116 = v203(v15[v13], v15, v101, v17, v18, v158(v30), v20, v21)
            v103.v174(v116 + (v228(v15[v13], v26, v27) if v31 else []))
        yield (v103, v21.v214(0))

def fit(v33, v27, v34, v35, v36):
    """Listwise logistic fit in plain python: one softmax over the safe candidates of a state."""
    if not v33:
        return []
    v29 = [0.0] * v158(v33[0][0][0])
    v37 = v162.v104(v36)
    v38 = v105(v89(v158(v33)))
    for v39 in v89(v34):
        v37.v123(v38)
        for v45 in v38:
            v103, v106 = v33[v45]
            v107 = [v95((v225 * v226 for v225, v226 in v229(v29, v204))) for v204 in v103]
            v163 = v166(v107)
            v164 = [v222.v215(v216 - v163) for v216 in v107]
            v165 = v95(v164) or 1.0
            for v63, v204 in v83(v103):
                v205 = v164[v63] / v165 - (1.0 if v63 == v106 else 0.0)
                if v205:
                    for v223, v224 in v83(v204):
                        if v224:
                            v29[v223] -= v35 * v205 * v224
    return v29

def acc(v33, v29):
    if not v33:
        return (v159('nan'), v159('nan'))
    v40 = v41 = 0.0
    for v103, v106 in v33:
        v107 = [v95((v225 * v226 for v225, v226 in v229(v29, v204))) for v204 in v103]
        v108 = v166(v107)
        v109 = [v63 for v63 in v89(v158(v107)) if v107[v63] == v108]
        v40 += 1.0 / v158(v109) if v106 in v109 else 0.0
        v41 += 1.0 / v158(v103)
    return (v40 / v158(v33), v41 / v158(v33))

def collect(v42, v26, v27, v31, v32, v43, v6=40):
    v33 = []
    for v44 in v42:
        for v30 in v167(v44, v6):
            for v85 in v206(v30, v26, v27, v31, v32):
                v33.v174(v85)
                if v158(v33) >= v43:
                    return v33
    return v33

def build_vocab(v42, v45):
    """A word is LANGUAGE when it occurs in at least k different files, and CONTENT otherwise."""
    v46 = v110()
    for v44 in v42:
        try:
            v9 = v4(v44).v196(encoding='utf-8', errors='ignore')
        except v147:
            continue
        v111 = v96()
        for v30 in v168(v9):
            for v23 in v30:
                v111 |= v23[4]
        for v29 in v111:
            v46[v29] += 1
    return ({v29 for v29, v217 in v46.v218() if v217 >= v45}, v46)

def norm_files(v42, v47, v48):
    """Both equivariance arms are unparsed from the AST, so only the identifiers differ."""
    v47.v112(parents=True, exist_ok=True)
    v12 = []
    for v44 in v42:
        try:
            v77 = v150.v149(v4(v44).v196(encoding='utf-8', errors='ignore'))
        except v78:
            continue
        if v48:
            v169 = {}
            for v170 in v150.v79(v77):
                if v151(v170, v150.v157):
                    v170.v200 = v169.v199(v170.v200, f'v{v158(v169)}')
                elif v151(v170, v150.v227):
                    v170.v227 = v169.v199(v170.v227, f'v{v158(v169)}')
                elif v151(v170, v150.v201):
                    v170.v213 = v169.v199(v170.v213, f'v{v158(v169)}')
        try:
            v171 = v150.v207(v77)
            v150.v149(v171)
        except v172:
            continue
        v113 = v47 / v4(v44).v173
        v113.v144(v171, encoding='utf-8')
        v12.v174(v113)
    return v12

def arm(v49, v50, v26, v27, v31, v51):
    v52 = v114(v49, v26, v27, v31, v51.v32, v51.v43, v51.v6)
    v53 = v114(v50, v26, v27, v31, v51.v32, v51.v43, v51.v6)
    v29 = v115(v52, v27, v51.v34, v51.v35, v51.v36)
    v116, v117 = v118(v53, v29)
    return (v116, v117, v158(v52), v158(v53), v29)

def main() -> v3:
    v54 = v175.v119()
    v54.v120('--files', default='_*.py')
    v54.v120('--transplant', default='/usr/lib/python3.11')
    v54.v120('--vocab-files', type=v3, default=5)
    v54.v120('--vocab-dim', type=v3, default=64)
    v54.v120('--min-safe', type=v3, default=2)
    v54.v120('--max-body', type=v3, default=40)
    v54.v120('--foreign', type=v3, default=60)
    v54.v120('--epochs', type=v3, default=6)
    v54.v120('--lr', type=v159, default=0.05)
    v54.v120('--cap', type=v3, default=20000)
    v54.v120('--seed', type=v3, default=1337)
    v54.v120('--out', default=v146(v0))
    v51 = v54.v121()
    v37 = v162.v104(v51.v36)
    v42 = v122(v4('.').v176(v51.v42))
    v37.v123(v42)
    v55 = v158(v42) // 2
    v49, v50 = (v42[:v55], v42[v55:])
    v26, v46 = v124(v49, v51.v125)
    v56 = v57 = 0
    for v58 in v42:
        for v30 in v167(v58, v51.v6):
            for v103, v208 in v206(v30, v26, v51.v132, False, 1):
                v56 += 1
                v57 += v3(v158(v103) >= 2)
    v59 = {'seed': v51.v36, 'files': v158(v42), 'vocab': v158(v26), 'vocab_files': v51.v125, 'states_all': v56, 'states_decide': v57, 'decide_share': v57 / v166(1, v56)}
    v126(f"{v59['files']} files, vocabulary {v59['vocab']} words seen in >= {v51.v125} files")
    v126(f"VOID CHECK  states with a decision {v59['decide_share']:.4f} ({v57} of {v56})  <- read first")
    v127, v117, v128, v129, v130 = v131(v49, v50, v26, v51.v132, False, v51)
    v133, v134, v135, v136, v137 = v131(v49, v50, v26, v51.v132, True, v51)
    v59.v138({'A': v127, 'AB': v133, 'floor': v117, 'n_train': v128, 'n_test': v129})
    v59['A_gain'] = v127 - v117
    v59['B_adds'] = v133 - v127
    v126(f"HELD OUT    floor {v117:.4f}   A (name-free) {v127:.4f} ({v59['A_gain']:+.4f})   A+B (vocabulary) {v133:.4f} ({v59['B_adds']:+.4f} over A)")
    v126(f'            fitted on {v128} states of {v158(v49)} files, read on {v129} states of {v158(v50)} files')
    v60 = v114(v50, v26, v51.v132, False, v51.v32, v51.v43, v51.v6)
    v61 = v158(v60[0][0][0]) if v60 else 0
    v62 = []
    for v63 in v89(v61):
        v109 = v166(v118(v60, [1.0 if v223 == v63 else 0.0 for v223 in v89(v61)])[0], v118(v60, [-1.0 if v223 == v63 else 0.0 for v223 in v89(v61)])[0])
        v62.v174(v109)
    v59['singles'] = v62
    v59['best_single'] = v166(v62) if v62 else v159('nan')
    v59['family_over_single'] = v127 - v59['best_single']
    v126(f"ABLATION    best single feature {v59['best_single']:.4f} (#{v62.v214(v59['best_single'])})   family over it {v59['family_over_single']:+.4f}")
    v64 = v122(v4(v51.v219).v176('*.py')) if v4(v51.v219).v177() else []
    v37.v123(v64)
    v64 = v64[:v51.v178]
    if v64:
        v179, v180, v135, v181, v182 = v131(v49, v64, v26, v51.v132, False, v51)
        v183, v184, v185, v186, v187 = v131(v49, v64, v26, v51.v132, True, v51)
        v59.v138({'T_A': v179, 'T_AB': v183, 'T_floor': v180, 'T_n': v181, 'T_A_gain': v179 - v180, 'T_B_adds': v183 - v179})
        v126(f'TRANSPLANT  floor {v180:.4f}   A {v179:.4f} ({v179 - v180:+.4f})   A+B {v183:.4f} ({v183 - v179:+.4f} over A)   on {v181} foreign states')
    v65 = v4(v51.v12).v66
    v67 = v139(v42, v65 / '_f404_norm', False)
    v68 = v139(v42, v65 / '_f404_ren', True)
    for v140, v141 in (('norm', v67), ('ren', v68)):
        v142 = v158(v141) // 2
        v28, v188 = v124(v141[:v142], v51.v125)
        v189, v190, v135, v136, v182 = v131(v141[:v142], v141[v142:], v28, v51.v132, False, v51)
        v191, v192, v185, v186, v187 = v131(v141[:v142], v141[v142:], v28, v51.v132, True, v51)
        v59[f'A_{v140}'], v59[f'AB_{v140}'] = (v189, v191)
    v126(f"RENAMING    A {v59['A_norm']:.4f} -> {v59['A_ren']:.4f} (must be identical)   A+B {v59['AB_norm']:.4f} -> {v59['AB_ren']:.4f} (must move)")
    v69 = v59['decide_share'] <= 0.05
    v70 = v59['A_gain'] > 0.05
    v71 = v59['B_adds'] > 0.05
    v72 = 'T_A_gain' in v59 and v59['T_A_gain'] >= v59['A_gain'] - 0.03
    v73 = v193(v59['A_norm'] - v59['A_ren']) < 1e-09
    v59.v138({'void': v143(v69), 'primary': v143(v70), 'vocab_pays': v143(v71), 'transplants': v143(v72), 'equivariant': v143(v73)})
    v126('\n' + ('VOID: nothing to decide in this arena.' if v69 else ('THE FAMILY CARRIES' if v70 else 'THE FAMILY IS A COIN') + f"; the vocabulary {('ADDS' if v71 else 'adds nothing')}; A {('TRANSPLANTS' if v72 else 'does NOT transplant')}; A is {('name-free' if v73 else 'NOT name-free')}."))
    v59['is_family'] = v143(v59['family_over_single'] > 0.05)
    v126('A JOINT, not another count.' if v59['is_family'] else 'NOT A JOINT: one feature alone reaches the family, so this is the eleventh count and the argument for fitting does not hold.')
    if v70 and v72 and v73:
        v126('Structure fitted jointly and held out by file is worth more than any single count this project has declared, and it survives a foreign corpus - which is the separation contract measured rather than asserted.')
    v12 = v4(v51.v12)
    v12.v66.v112(parents=True, exist_ok=True)
    v74 = v209.v98(v12.v196()) if v12.v177() else {}
    v74[v146(v51.v36)] = v59
    v12.v144(v209.v194(v74, indent=1))
    v126(f'wrote {v12}')
    return 0
if v75 == '__main__':
    raise v145(v195())