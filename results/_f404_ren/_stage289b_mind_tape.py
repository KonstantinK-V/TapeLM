"""
Stage 289b — Is there a mind tape? Is the space of situations smooth enough to interpolate in?

A second tape was proposed: not facts, but PATTERNS. Not "Alan Kay was born in Springfield" but
"a coalition of three agreeing mentions with high context rank, plus one outlier nobody
corroborates". Today that knowledge sits smeared across 4,417 weights. On a mind tape it would
sit as POINTS, and inference would be neighbourhood and interpolation - the continuous space a
discrete tape does not have, and the reason a transformer exceeds its dataset while a lookup
table cannot.

Three things it would buy, and they are not small:
  - the repertoire grows WITHOUT retraining, the way facts already do, so the invariant "the
    mind does not grow with knowledge" holds where it should - the READER stays constant while
    experience accumulates;
  - judgment gets provenance: "why did you flag this?" - "because it resembles these three
    stored cases". The project bought that for knowledge; this buys it for reasoning;
  - between two stored patterns there IS a point, so a novel situation can be answered by
    blending. That is the machinery a conjecture needs.

And the condition without which it degenerates immediately: the key must be STRUCTURE ONLY.
Four arms measured the alternative - frozen trunk, rank-8, anonymised text, delta channel - and
every continuous store keyed on identity became a lookup table. So the pattern vector here is
built from ranks, ratios and indicators, is fixed-length regardless of how many mentions the
address has, and cannot name anyone.

This probe trains NOTHING. It fills a memory with patterns from one tape and asks four
questions of the space itself:

  1. SIGNAL. Does 1-nearest-neighbour beat the counting detector and the majority-class floor
     on a held-out tape? k=1 so there is no k to choose.
  2. SMOOTHNESS. Is being right related to being CLOSE? If correctness is independent of
     distance, the space is not a space, it is a bag, and interpolation in it means nothing.
     Measured as the AUC of distance separating wrong neighbours from right ones.
  3. INTERPOLATION - the one that matters for conjecture. Take two stored patterns with the
     SAME label and look up their midpoint: does it land near that label too? Against the
     control of midpoints between DIFFERENT labels, which should not. A space where midpoints
     are meaningless can store patterns but cannot blend them, and blending is the whole
     reason to want continuity.
  4. GROWTH. Does accuracy rise as the memory fills? That is "repertoire grows without
     retraining", measured rather than assumed.

What this cannot answer: whether the mind tape beats the TRAINED head. That needs 288's
checkpoint and is the follow-up. Note though that a tie is already a win - 4,417 weights over
about a thousand examples is close to a memory already, and an explicit memory is auditable
and extensible where weights are neither.

  python _stage289b_mind_tape.py --smoke
  python _stage289b_mind_tape.py
  python _stage289b_mind_tape.py --holdout address
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage286_evidence as s286
import _stage288_repair as s288
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('data/_wikitext103_train.txt')
v3 = 289
v4 = (0.0, 0.25, 0.5, 0.75, 1.0)
v5 = v0 / '_stage289b_log.txt'

def log(v9: v76) -> None:
    v10 = v9 if v9.v157('\n') else v9 + '\n'
    try:
        v158(v10, end='', flush=True)
    except v77:
        v158(v10.v248('ascii', 'replace').v227('ascii'), end='', flush=True)
    v5.v159.v78(parents=True, exist_ok=True)
    with v5.v160('a', encoding='utf-8') as v79:
        v79.v161(v10)

def rank_pairs(v11: v14.v6) -> v14.v6:
    """Within-example ranks in [0,1], ties sharing a rank. The only currency allowed here."""
    if v11.v80 == 0:
        return v11
    v12 = v11.v81()
    v13 = v14.v82(v90(v11), dtype=v143)
    v13[v12] = v14.v83(v90(v11), dtype=v143)
    v84, v85 = v14.v86(v11, return_inverse=True)
    if v90(v84) > 1:
        v87 = v14.v89(v90(v84))
        v88 = v14.v89(v90(v84))
        v14.v205.v162(v87, v85, v13)
        v14.v205.v162(v88, v85, 1.0)
        return (v87 / v88)[v85] / v198(1, v90(v11) - 1)
    return v14.v89(v90(v11))

def pattern_of(v15: v14.v6, v16: v14.v6, v17: v14.v6, v18: v14.v6, v19: v14.v6) -> v14.v6:
    """One example as a fixed-length point, whatever its number of mentions.

    Every entry is a rank, a ratio or a count normalised by n, so two addresses about different
    subjects with the same shape land in the same place - which is the entire premise. Fixed
    length is what makes it a POINT rather than a set, and points are what a continuous space
    is made of.
    """
    v20 = v90(v18)
    v21 = v14.v91(v20, 1)
    v22 = [1.0 / v20, v143(v14.v136(v15[v21])) if v21[0].v80 else 0.0]
    for v47, v92 in ((v15, False), (v16, True), (v17, True)):
        v11 = v47[v21] if v21[0].v80 else v14.v89(1)
        v11 = v163(v11) if v92 else v11
        v22 += [v143(v14.v136(v11))] + [v143(v14.v241(v11, v26)) for v26 in v4]
    for v23 in (v18, v163(v19)):
        v22 += [v143(v14.v136(v23)), v143(v14.v228(v23)), v143(v14.v198(v23)), v143(v14.v135(v23))]
    return v14.v93(v22, dtype=v143)

def knn_predict(v24: v14.v6, v25: v94, v26: v14.v6, v27: v182[v7] | None=None):
    """1-NN. k=1 because any other k is a constant somebody chose."""
    v28 = v14.v164.v95(v24 - v26, axis=1)
    if v27:
        v28 = v28.v165()
        v28[v94(v27)] = v14.v96
    v29 = v7(v28.v166())
    return (v25[v29], v143(v28[v29]), v29)

def main() -> v7:
    v30 = v167.v97()
    v30.v98('--smoke', action='store_true')
    v30.v98('--addresses', type=v7, default=0)
    v30.v98('--min-mentions', type=v7, default=2)
    v30.v98('--tapes', type=v7, default=8, help="how many resampled training tapes fill the memory. The memory is the point of the stage, so it gets more than one tape's worth.")
    v30.v98('--address-tau', type=v143, default=0.9)
    v30.v98('--address-overlap', type=v7, default=2)
    v30.v98('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v30.v98('--holdout', choices=('corpus', 'address'), default='corpus')
    v30.v98('--run-tag', type=v76, default='')
    v31 = v30.v99()
    global LOG_PATH
    v32 = v31.v155 and f'_{v31.v155}' or ''
    v32 += '_addrholdout' if v31.v115 == 'address' else ''
    v5 = v0 / f'_stage289b_log{v32}.txt'
    v5.v159.v78(parents=True, exist_ok=True)
    v5.v100('', encoding='utf-8')
    v33 = v168.v33('cuda' if v168.v229.v206() else 'cpu')
    v34 = v169.v101(v3)
    v35 = v102.v102()
    v36 = v31.v103 or (300 if v31.v154 else 400)
    v104(f'Stage289b mind tape start {v246.v239(v247.v240).v202()} device={v33} holdout={v31.v115} tapes={v31.v131}')
    v64, v64, v105, v106 = v107()
    v37 = v170.v108(v76(v207.v171))
    v38 = v37.v172(v173) or 0
    v39 = v208(v106, v37.v230()).v109(v33)
    v39.v110(v168.v209(v1, map_location=v33, weights_only=False)['model'])
    v39.v111()
    for v40 in v39.v112():
        v40.v174(False)
    v41 = v113(v39, v105, v33)
    with v2.v160('r', encoding='utf-8', errors='ignore') as v79:
        v114 = v79.v175(4000000 if v31.v154 else 30000000)
    v42 = [v147.v176() for v147 in v114.v210('\n') if 80 <= v90(v147.v176()) <= 400]
    v43 = v7(0.7 * v90(v42))
    v44 = v42[:v43][:3000 if v31.v154 else 25000]
    v45 = v42[v43:][:1500 if v31.v154 else 12000]
    if v31.v115 == 'address':
        v45 = v44

    def side(v116: v76) -> v7:
        v117 = v116.v210(':', 1)[-1].v210('|')[0]
        return v7(v255.v249(v117.v248('utf-8')).v231(), 16) & 1

    def new_pack(v13, v118, v119):
        v40 = v211.v177(v118, bank=v41, tok=v37, pad_id=v38, device=v33, rng=v13, n_addr=v36, min_mentions=v31.v212, tau=v31.v213, overlap=v31.v214, soft_match=0.0, min_per_family=8, addr_key=v31.v215)
        if v31.v115 == 'address':
            v40 = v216(v40)
            v40['items'] = [v127 for v127 in v40['items'] if v250(v127['address']) == v119]
        return v40

    def graph(v40, v120, v121):
        """The three channels and the two node columns, as plain arrays."""
        v178, v179 = (v120['slots'], v120['vals'])
        v20 = v90(v178)
        v180, v181 = (v40.v217('_ctx', {}), v40.v217('_words', {}))
        for v122 in v182(v178):
            if v122 not in v180:
                v88 = v41.v232(v40['texts'][v122], exclude=v40['tape'].v251[v122])
                v180[v122] = v252.v242(v88, dim=-1) if v88 is not None else None
                v181[v122] = v182(v243(v40['texts'][v122], exclude=v40['tape'].v251[v122]))
        v123 = v40.v183('_median')
        if v123 is None:
            v184 = v218((v90(v11) for v11 in v40['postings'].v251()))
            v123 = v184[v90(v184) // 2] if v184 else 1
            v40['_median'] = v123
        v15 = v14.v89((v20, v20))
        v16 = v14.v89((v20, v20))
        v17 = v14.v89((v20, v20))
        for v29 in v130(v20):
            for v185 in v130(v29 + 1, v20):
                v233, v234 = (v178[v29], v178[v185])
                v15[v29, v185] = v15[v185, v29] = v143(v179[v29] == v179[v185])
                if v180[v233] is not None and v180[v234] is not None:
                    v16[v29, v185] = v16[v185, v29] = v143(v180[v233] @ v180[v234])
                v219 = v235((1 for v253 in v181[v233] & v181[v234] if v90(v40['postings'].v183(v253, ())) < v123))
                v17[v29, v185] = v17[v185, v29] = v219 / v198(1, v228(v90(v181[v233]), v90(v181[v234])))
        v124 = v186(v179)
        v125 = v182(v121['slots'])
        v126 = {v11: v193.v220(v40, v121['S'], v11, v125) for v11 in v124}
        v18 = v14.v93([v124[v11] / v20 for v11 in v179])
        v19 = v14.v93([v143(v126[v11]) for v11 in v179])
        return (v15, v16, v17, v18, v19)

    def collect(v40, v13):
        """Every (address, corruption) pair as a point plus its label."""
        v187, v188 = ([], [])
        for v127 in v40['items']:
            if v90(v127['slots']) < 2:
                continue
            for v122 in v127['slots']:
                for v221 in v236.v222:
                    v120 = v236.v244(v40, v127, v13, v221, v122)
                    if v120 is None:
                        continue
                    v187.v191(v254(*v256(v40, v120, v127)))
                    v188.v191(v221)
        return (v187, v188)
    v128, v129 = ([], [])
    for v46 in v130(v31.v131):
        v40 = v137(v169.v101(v3 + v46), v44, 0)
        v132, v133 = v140(v40, v169.v101(v3 + 100 + v46))
        v128 += v132
        v129 += v133
        v104(f'  tape {v46 + 1}/{v31.v131}: +{v90(v132)} patterns (memory {v90(v128)})')
    if v90(v128) < 8 * v193.v189:
        v104(f'  memory too small: {v90(v128)}')
        return 1
    v47 = v14.v134(v128)
    v48 = v47.v135(0)
    v48[v48 < 1e-09] = 1.0
    v49 = v47.v136(0)
    v47 = (v47 - v49) / v48
    v104(f'  memory {v47.v237[0]} patterns x {v47.v237[1]} dims, labels {v226.v203(v216(v186(v129)))}')
    v50 = v137(v169.v101(v3 + 99), v45, 1)
    v138, v139 = v140(v50, v169.v101(v3 + 999))
    if v90(v138) < 4 * v193.v189:
        v104(f'  held-out too small: {v90(v138)}')
        return 1
    v51 = (v14.v134(v138) - v49) / v48
    v104(f'  held out {v51.v237[0]} patterns, labels {v226.v203(v216(v186(v139)))}')
    v141, v142 = ([], [])
    for v26 in v51:
        v25, v28, v64 = v190(v47, v129, v26)
        v141.v191(v25)
        v142.v191(v28)
    v52 = [v7(v117 == v195) for v117, v195 in v223(v141, v139)]
    v53 = v143(v14.v136(v52))
    v54 = v186(v129).v224(1)[0][0]
    v55 = v143(v14.v136([v7(v54 == v195) for v195 in v139]))
    v56 = []
    for v29, v25 in v144(v139):
        v56.v191(v7(v25 == 'none'))
    v57 = v143(v14.v136(v56))
    v58 = [v28 for v28, v192 in v223(v142, v52) if not v192]
    v59 = [v28 for v28, v192 in v223(v142, v52) if v192]
    v60 = v193.v145(v58, v59)
    v61 = v193.v146(v60, v90(v58), v90(v59))
    v62 = v169.v101(v3 + 5)
    v63 = {}
    for v29, v147 in v144(v129):
        v63.v217(v147, []).v191(v29)
    v148, v149 = ([], [])
    for v64 in v130(400):
        v147 = v62.v194(v94(v63))
        if v90(v63[v147]) < 2:
            continue
        v117, v195 = v62.v196(v63[v147], 2)
        v25, v64, v64 = v190(v47, v129, (v47[v117] + v47[v195]) / 2.0, exclude={v117, v195})
        v148.v191(v7(v25 == v147))
        v150 = v62.v194([v151 for v151 in v63 if v151 != v147] or [v147])
        if not v63[v150]:
            continue
        v88 = v62.v194(v63[v150])
        v197, v64, v64 = v190(v47, v129, (v47[v117] + v47[v88]) / 2.0, exclude={v117, v88})
        v149.v191(v7(v197 == v147))
    v65 = v143(v14.v136(v148)) if v148 else v143('nan')
    v66 = v143(v14.v136(v149)) if v149 else v143('nan')
    v67 = {}
    for v68 in (0.125, 0.25, 0.5, 1.0):
        v151 = v198(1, v7(v68 * v90(v47)))
        v199, v200 = (v47[:v151], v129[:v151])
        v152 = v143(v14.v136([v7(v190(v199, v200, v26)[0] == v46) for v26, v46 in v223(v51, v139)]))
        v67[f'{v68:.3f}'] = v152
    v69 = v153(v53 > v55 and v53 > v57)
    v70 = v153(not v245.v238(v61) and v61 > 1.645)
    v71 = v153(not v245.v238(v65) and (not v245.v238(v66)) and (v65 > v66))
    v72 = v153(v67['1.000'] > v67['0.125'])
    v73 = 'MIND_TAPE_REAL' if v69 and v70 and v71 and v72 else 'MIND_TAPE_STORES_BUT_CANNOT_BLEND' if v69 and (not v71) else 'MIND_TAPE_PARTIAL' if v69 else 'MIND_TAPE_NO'
    v74 = {'stage': '289b', 'overall': v73, 'seed': v3, 'smoke': v31.v154, 'holdout': v31.v115, 'run_tag': v31.v155, 'trained_parameters': 0, 'memory': {'n': v7(v47.v237[0]), 'dims': v7(v47.v237[1]), 'labels': v216(v186(v129)), 'tapes': v31.v131}, 'held_out': {'n': v7(v51.v237[0]), 'labels': v216(v186(v139))}, 'signal': {'knn_1_accuracy': v53, 'majority_class_floor': v55, 'counting_floor': v57}, 'smoothness': {'auc_distance_separates_wrong_from_right': v60, 'auc_z': v61, 'mean_distance_when_right': v143(v14.v136(v59)) if v59 else v143('nan'), 'mean_distance_when_wrong': v143(v14.v136(v58)) if v58 else v143('nan')}, 'interpolation': {'midpoint_same_label': v65, 'midpoint_different_label_control': v66, 'n_same': v90(v148), 'n_diff': v90(v149)}, 'growth': v67, 'gates': {'G_space_has_signal': v69, 'G_space_is_smooth': v70, 'G_interpolation_holds': v71, 'G_grows_without_training': v72}, 'fp_version': v225.v201(), 'note': "Whether a mind tape can exist: not facts but PATTERNS, stored as points in a continuous space where a novel situation is answered by neighbourhood and blending. Nothing is trained. The pattern is built from ranks, ratios and indicators and is fixed-length whatever the address size, so it cannot name anyone - four measured arms showed that any continuous store keyed on identity becomes a lookup table, so structure-only is the condition, not a preference. Four questions of the space itself: does 1-NN beat the majority-class and counting floors held out; is being right related to being CLOSE, without which the space is a bag and interpolation is meaningless; does the midpoint between two same-label patterns land near that label, against the control of midpoints between different labels - the property a conjecture would rest on; and does accuracy rise as the memory fills, which is 'the repertoire grows without retraining' measured rather than assumed. What it cannot say is whether the mind tape beats 288's trained head; that needs the checkpoint. A tie there is already a win, because 4,417 weights over a thousand examples is close to a memory already and an explicit memory is auditable and extensible where weights are neither.", 'timestamp': v246.v239(v247.v240).v202(), 'wall_s': v102.v102() - v35}
    v0.v78(parents=True, exist_ok=True)
    (v0 / f'stage289b_decision{v32}.json').v100(v226.v203(v74, indent=2), encoding='utf-8')
    v104(v226.v203({'overall': v73, 'gates': v74['gates'], 'signal': v74['signal'], 'interpolation': v74['interpolation']}, indent=2))
    return 0
if v75 == '__main__':
    raise v156(v204())