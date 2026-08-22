"""
Stage 288 — Repair: the mind learns to fix the tape, scored against ground truth it cannot see.

286 taught reading agreement; this teaches RESTORING it. The loop is: break a clean address,
show the mind the broken evidence, require (1) WHERE the break is and (2) WHAT stood there -
or the admission that the remaining evidence cannot say. The proposal's original reward -
"did the world become more consistent by the judges" - is refused as a training signal,
because its fixed point is an empty tape: erase every dispute and the judges purr. It is kept
as an OBSERVER (verdict_restored_rate), which is where signals that cannot be trusted with
gradients live in this project.

What replaces it is better than a judge: the corruption is synthetic, so the truth is known
for free. We broke it, we know what it was. That gives three things 286 could not have:
  - unlimited examples: corruption is generated, not harvested, so the diagnosed bottleneck
    (about a thousand distinct examples, 22 usable addresses per tape) disappears;
  - honestly-trained honesty: we KNOW which breaks are unrecoverable, because we made them;
  - the write side becomes learnable: WRITE/CONFIRM/DISPUTE is code today; a mind that can
    point at the forged mention is the mind that can one day dispute it.

Cloze was a special case all along: hiding a mention is the DELETE corruption, the lying tape
is REPLACE, --lie-dup is REPLACE with copies. 286's exams were points in this family; 288
trains on the family. DELETE itself stays in 286 (it is measured there); here the ops are
NONE / REPLACE / DUP, because they are the ones with a visible culprit.

The mind is the relational one - the only arm that ever passed anything - with two heads on
one shared graph embedding: a DIAGNOSIS head scoring every mention plus a CLEAN row (the same
shape as candidates plus UNKNOWN), and a REPAIR head scoring the surviving values plus
UNKNOWN. Identity stays unrepresentable: ranks and indicators only. The answer stays an index.

Falsifiers, all relative, none tunable:
  G_detects_forgery       beat the counting detector (flag the minority value) AND the
                          per-example random floor, held out
  G_detects_dup           same, on the subset where the forged value is the MAJORITY -
                          the counting detector is wrong there by construction
  G_flags_clean           the CLEAN margin separates untouched addresses from forged ones
                          (AUC above its own noise) - false alarms are the failure mode of
                          every repair system
  G_repairs               with the true flag given, restore the original value better than
                          majority-of-the-rest does
  G_honest_unrecoverable  UNKNOWN's margin ranks unrecoverable breaks above recoverable ones

  python _stage288_repair.py --smoke
  python _stage288_repair.py --train-steps 6000
  python _stage288_repair.py --train-steps 6000 --holdout address
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage286_evidence as s286
import _stage279_write_decision as s279
from _tape_speed import CachedBank, install_all
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('data/_wikitext103_train.txt')
v3 = 288
v4 = ('none', 'replace', 'dup')
v5 = v0 / '_stage288_log.txt'

def log(v9: v65) -> None:
    v10 = v9 if v9.v164('\n') else v9 + '\n'
    try:
        v165(v10, end='', flush=True)
    except v66:
        v165(v10.v324('ascii', 'replace').v274('ascii'), end='', flush=True)
    v5.v166.v67(parents=True, exist_ok=True)
    with v5.v167('a', encoding='utf-8') as v68:
        v68.v168(v10)

def corrupt(v11, v12, v13, v14: v65, v15: v7 | None):
    """One broken address, with the truth attached - because we are the ones who broke it.

    none      the address as written. The target diagnosis is CLEAN, and its being a real
              class is what keeps the detector honest: a mind that flags everything scores
              zero here, no judge needed to say so.
    replace   one genuine mention swapped for a foreign one (another subject's value and
              text). The culprit is visible and the original value is recorded.
    dup       the same swap, but the forged mention arrives in 2-3 copies - the regime where
              counting is wrong BY CONSTRUCTION, since the forgery is now the majority.
    """
    v69, v70 = (v11['tape'].v71, v11['texts'])
    v16 = v72(v12['slots'])
    if v14 == 'none':
        v77, v78, v79 = (v16, v170(), None)
    else:
        v73 = v16.v83(v15)
        v74 = 1 if v14 == 'replace' else v13.v237((2, 3))
        v75 = None
        for v76 in v146(64):
            v169 = v13.v238(v11['n_slots'])
            if v169 not in v170(v16) and v69[v169] != v69[v15]:
                v75 = v169
                break
        if v75 is None:
            return None
        v77 = v16[:v73] + [v75] * v74 + v16[v73 + 1:]
        v78 = v170(v146(v73, v73 + v74))
        v79 = v69[v15]
    v17 = [v69[v171] for v171 in v77]
    if v14 != 'none' and v172(v170(v17)) < 2:
        return None
    return {'op': v14, 'slots': v77, 'vals': v17, 'texts': [v70[v171] for v171 in v77], 'forged': v78, 'orig': v79}

def repair_candidates(v18, v19: v170[v7]):
    """What survives once the flagged mentions are set aside, plus UNKNOWN."""
    v20 = [v80 for v134, v80 in v239(v18['vals']) if v134 not in v19]
    v21 = [v80 for v80, v76 in v81(v20).v84(8)]
    v22 = v21.v83(v18['orig']) if v18['orig'] in v21 else v172(v21)
    return (v21, v22, v20)

def votes_detector(v18):
    """Flag a mention of the minority value; CLEAN when unanimous. The counter's best."""
    v23 = v81(v18['vals'])
    if v172(v23) == 1:
        return None
    v24 = v82(v23, key=lambda v80: v23[v80])
    return v18['vals'].v83(v24)

def votes_repair(v18, v19: v170[v7]):
    v20 = [v80 for v134, v80 in v239(v18['vals']) if v134 not in v19]
    v23 = v81(v20)
    v25 = v23.v84(2)
    if not v25 or (v172(v25) > 1 and v25[0][1] == v25[1][1]):
        return None
    return v25[0][0]

class RepairMind(v26.v6):
    """One relational embedding, two heads. Identity cannot enter: ranks and indicators only.

    The diagnosis head scores each MENTION (plus a CLEAN row from the global pool) - the same
    shape as 286's candidates-plus-UNKNOWN, pointed at slots instead of values. The repair
    head is 286's candidate head. Sharing the embedding is the claim that the same relations
    answer both questions: a forged mention sits in the graph as the value nobody's context
    agrees with, and the repaired value is the one whose coalition is tightest without it.
    """

    def __init__(v85, v30, v86: v7=32):
        v275().v173()
        v85.v87 = v26.v276(v26.v313(3, v86), v26.v314()).v116(v30)
        v85.v88 = v26.v276(v26.v313(2 + 2 * v86, v86), v26.v314()).v116(v30)
        v85.v89 = v26.v276(v26.v313(2 * v86 + 1, v86), v26.v314(), v26.v313(v86, 1)).v116(v30)
        v85.v90 = v26.v276(v26.v313(2 * v86 + 1, v86), v26.v314(), v26.v313(v86, 1)).v116(v30)
        for v91 in (v85.v89, v85.v90):
            v26.v277.v240(v91[-1].v241)
            v26.v277.v240(v91[-1].v242)

    def embed(v85, v92, v93, v94):
        v95 = v85.v87(v92)
        v96 = (v95 * v93).v211(1) / v93.v211(1).v243(min=1.0)
        v97 = v85.v88(v180.v177([v94, v96, v95.v244(1)], -1))
        return (v97, v97.v244(0))

    def diagnose(v85, v97, v98):
        v174, v175 = (v180.v202(1, device=v98.v30), v180.v245(1, device=v98.v30))
        v99 = [v85.v89(v180.v177([v97[v134], v98, v174])) for v134 in v146(v97.v315[0])]
        v99.v176(v85.v89(v180.v177([v180.v261(v98), v98, v175])))
        return v180.v177(v99)

    def repair(v85, v97, v98, v100):
        v174, v175 = (v180.v202(1, device=v98.v30), v180.v245(1, device=v98.v30))
        v99 = [v85.v90(v180.v177([v97[v9].v244(0), v98, v174])) for v9 in v100]
        v99.v176(v85.v90(v180.v177([v180.v261(v98), v98, v175])))
        return v180.v177(v99)

def main() -> v7:
    v27 = v178.v101()
    v27.v102('--smoke', action='store_true')
    v27.v102('--train-steps', type=v7, default=0)
    v27.v102('--tape-period', type=v7, default=0)
    v27.v102('--addresses', type=v7, default=0)
    v27.v102('--min-mentions', type=v7, default=2)
    v27.v102('--address-tau', type=v179, default=0.9)
    v27.v102('--address-overlap', type=v7, default=2)
    v27.v102('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v27.v102('--abstain-reward', type=v179, default=0.75)
    v27.v102('--wrong-cost', type=v179, default=1.0)
    v27.v102('--lr', type=v179, default=0.001)
    v27.v102('--holdout', choices=('corpus', 'address'), default='corpus', help='corpus: eval tape from the disjoint 30% of lines. address: one corpus, subjects split by a stable hash of the anchor - 286 measured honesty crossing that split (0.754) while failing the corpus one, so both views matter and neither is redundant.')
    v27.v102('--run-tag', type=v65, default='')
    v27.v102('--no-speedups', action='store_true', help='run the original unmemoised paths - they are byte-identical, and this is how that stays checkable')
    v28 = v27.v103()
    global LOG_PATH
    v29 = v28.v162 and f'_{v28.v162}' or ''
    v29 += '_addrholdout' if v28.v123 == 'address' else ''
    v29 = '_smoke' + v29 if v28.v104 else v29
    v5 = v0 / f'_stage288_log{v29}.txt'
    v5.v166.v67(parents=True, exist_ok=True)
    v5.v105('', encoding='utf-8')
    v30 = v180.v30('cuda' if v180.v278.v246() else 'cpu')
    v13 = v181.v106(v3)
    v180.v107(v3)
    v31 = v108.v108()
    v32 = v28.v109 or (600 if v28.v104 else 6000)
    v33 = v28.v33 or (50 if v28.v104 else 50)
    v34 = v28.v110 or (300 if v28.v104 else 400)
    v111(f'Stage288 repair start {v322.v311(v323.v312).v234()} device={v30} steps={v32} holdout={v28.v123}')
    v76, v76, v112, v113 = v114()
    v35 = v182.v115(v65(v247.v183))
    v36 = v35.v184(v185) or 0
    v37 = v248(v113, v35.v279()).v116(v30)
    v37.v117(v180.v249(v1, map_location=v30, weights_only=False)['model'])
    v37.v118()
    for v38 in v37.v119():
        v38.v186(False)
    if not v28.v120:
        v187(v188)
    v39 = v189(v37, v112, v30) if v28.v120 else v190(v189(v37, v112, v30))
    v40 = v191.v121(v37)
    with v2.v167('r', encoding='utf-8', errors='ignore') as v68:
        v122 = v68.v192(4000000 if v28.v104 else 30000000)
    v41 = [v194.v193() for v194 in v122.v250('\n') if 80 <= v172(v194.v193()) <= 400]
    v42 = v7(0.7 * v172(v41))
    v43 = v41[:v42][:3000 if v28.v104 else 25000]
    v44 = v41[v42:][:1500 if v28.v104 else 12000]
    if v28.v123 == 'address':
        v44 = v43

    def anchor_side(v124: v65) -> v7:
        v125 = v124.v250(':', 1)[-1].v250('|')[0]
        return v7(v331.v325(v125.v324('utf-8')).v280(), 16) & 1

    def split_items(v38, v126):
        if v28.v123 == 'corpus':
            return v38
        v38 = v195(v38)
        v38['items'] = [v152 for v152 in v38['items'] if v316(v152['address']) == v126]
        return v38

    def new_pack(v127, v128, v126):
        return v196(v281.v251(v128, bank=v39, tok=v35, pad_id=v36, device=v30, rng=v127, n_addr=v34, min_mentions=v28.v282, tau=v28.v283, overlap=v28.v284, soft_match=0.0, min_per_family=8, addr_key=v28.v285), v126)

    def graph_inputs(v38, v18, v12):
        v16, v197 = (v18['slots'], v18['vals'])
        v129 = v172(v16)
        v198, v199 = (v38.v252('_ctx', {}), v38.v252('_words', {}))
        for v130 in v170(v16):
            if v130 not in v198:
                v169 = v39.v286(v38['texts'][v130], exclude=v38['tape'].v71[v130])
                v198[v130] = v293.v317(v169, dim=-1) if v169 is not None else None
                v199[v130] = v170(v318(v38['texts'][v130], exclude=v38['tape'].v71[v130]))
        v131 = v38.v200('_median')
        if v131 is None:
            v201 = v207((v172(v80) for v80 in v38['postings'].v71()))
            v131 = v201[v172(v201) // 2] if v201 else 1
            v38['_median'] = v131
        v93 = v180.v202(v129, v129)
        v132 = v180.v202(v129, v129)
        v133 = v180.v202(v129, v129)
        for v134 in v146(v129):
            for v73 in v146(v134 + 1, v129):
                v287, v288 = (v16[v134], v16[v73])
                v93[v134, v73] = v93[v73, v134] = v179(v197[v134] == v197[v73])
                if v198[v287] is not None and v198[v288] is not None:
                    v132[v134, v73] = v132[v73, v134] = v179(v198[v287] @ v198[v288])
                v253 = v211((1 for v326 in v199[v287] & v199[v288] if v172(v38['postings'].v200(v326, ())) < v131))
                v133[v134, v73] = v133[v73, v134] = v253 / v297(1, v82(v172(v199[v287]), v172(v199[v288])))
        v135 = v180.v203(v129, v129, offset=1)

        def rank_norm(v204):
            if v135.v289() == 0:
                return v204
            v80 = v204[v135[0], v135[1]]
            v205 = v80.v254()
            v127 = v180.v255(v205, dtype=v180.v263)
            v127[v205] = v180.v256(v172(v80), dtype=v180.v263)
            v257, v258 = v80.v259(return_inverse=True)
            if v172(v257) > 1:
                v260 = v180.v202(v172(v257)).v290(0, v258, v127, 'mean', include_self=False)
                v127 = v260[v258] / (v172(v80) - 1 if v172(v80) > 1 else 1)
            else:
                v127 = v180.v261(v127)
            v206 = v180.v261(v204)
            v206[v135[0], v135[1]] = v127
            v206[v135[1], v135[0]] = v127
            return v206
        v92 = v180.v291([v93, v327(v132), v327(v133)], -1).v116(v30)
        v23 = v81(v197)
        v96 = v170(v12['slots'])
        v136 = {v80: v264.v262(v38, v12['S'], v80, v96) for v80 in v23}
        v137 = v207(v170(v136.v71()))
        v94 = v180.v208([[v23[v197[v134]] / v129, v137.v83(v136[v197[v134]]) / v297(1, v172(v137) - 1)] for v134 in v146(v129)], dtype=v180.v263, device=v30)
        return (v92, v93.v295(-1).v116(v30), v94)
    v45 = v138(v30)
    v46 = v180.v209.v139(v45.v119(), lr=v28.v210, weight_decay=0.01)
    v47 = v7(v211((v292.v289() for v292 in v45.v119())))

    def usable(v38):
        return [(v152, v171) for v152 in v38['items'] if v172(v152['slots']) >= 2 for v171 in v152['slots']]
    v11 = v140(v13, v43, 0)
    v48 = v141(v11)
    v111(f"  tape: {v11['n_addresses']} addresses, {v11['n_slots']} slots, {v172(v48)} corruption sites, params={v47}")
    if v172(v48) < 4 * v264.v212:
        v142 = v81((v172(v152['slots']) for v152 in v11['items']))
        v111(f"  too few corruption sites: {v172(v48)} < {4 * v264.v212}. items {v172(v11['items'])}, mentions-per-address {v272.v235(v195(v142))}. Raise --addresses or lower --min-mentions.")
        return 1
    v143, v144, v145 = ([], [], v81())
    for v49 in v146(1, v32 + 1):
        if (v49 - 1) % v33 == 0 and v49 > 1:
            v11 = v140(v13, v43, 0)
            v48 = v141(v11)
            if not v48:
                v111('  empty tape after resample')
                return 1
        v152, v130 = v48[v13.v238(v172(v48))]
        v14 = v4[v13.v238(v172(v4))]
        v18 = v213(v11, v152, v13, v14, v130)
        if v18 is None:
            continue
        v145[v14] += 1
        v92, v93, v94 = v214(v11, v18, v152)
        v97, v98 = v45.v215(v92, v93, v94)
        v147 = v45.v216(v97, v98)
        v129 = v172(v18['vals'])
        if v18['forged']:
            v217 = v293.v265(v147, dim=-1)
            v149 = -v180.v294(v217[v72(v18['forged'])], dim=0)
        else:
            v149 = v293.v266(v147.v295(0), v180.v208([v129], device=v30))
        v148 = v149
        if v18['forged']:
            v21, v22, v20 = v267(v18, v18['forged'])
            if v172(v21) >= 2:
                v100 = [v180.v208([v134 not in v18['forged'] and v18['vals'][v134] == v169 for v134 in v146(v129)], device=v30) for v169 in v21]
                v268 = v45.v296(v97, v98, v100)
                v148 = v148 + v293.v266(v268.v295(0), v180.v208([v22], device=v30))
        v46.v218(set_to_none=True)
        v148.v219()
        v180.v26.v269.v220(v45.v119(), 1.0)
        v46.v49()
        v143.v176(v179(v148))
        if v49 % v297(1, v32 // 8) == 0:
            v144.v176({'step': v49, 'loss': v179(v329.v244(v143[-200:])), 'op': v18['op']})
            v111(f"  step {v49}/{v32} loss={v329.v244(v143[-200:]):.4f} [{v18['op']}]")
    v45.v118()
    v50 = v191.v121(v37)

    @v180.v157()
    def examine(v38):
        v127 = v181.v106(v3 + 7)
        v150 = {v14: v223(v72) for v14 in v4}
        v221, v222 = ([], [])
        v90 = v223(v72)
        v224, v225 = ([], [])
        v226, v227 = ([], [])
        v228, v229 = ([], [])
        v151 = 0
        for v152 in v38['items']:
            if v172(v152['slots']) < 2:
                continue
            for v130 in v152['slots']:
                for v14 in v4:
                    v18 = v213(v38, v152, v127, v14, v130)
                    if v18 is None:
                        continue
                    v92, v93, v94 = v214(v38, v18, v152)
                    v97, v98 = v45.v215(v92, v93, v94)
                    v147 = v45.v216(v97, v98)
                    v129 = v172(v18['vals'])
                    v298 = v7(v147.v328())
                    v299 = v179(v147[v129] - v147[:v129].v297())
                    v300 = v319(v18)
                    if v14 == 'none':
                        v150[v14]['model'].v176(v7(v298 == v129))
                        v150[v14]['votes'].v176(v7(v300 is None))
                        v221.v176(v299)
                        continue
                    v222.v176(v299)
                    v150[v14]['model'].v176(v7(v298 in v18['forged']))
                    v150[v14]['votes'].v176(v7(v300 in v18['forged']))
                    v228.v176(v172(v18['forged']) / (v129 + 1))
                    v21, v22, v20 = v267(v18, v18['forged'])
                    if v172(v21) < 2:
                        v151 += 1
                        continue
                    v100 = [v180.v208([v134 not in v18['forged'] and v18['vals'][v134] == v169 for v134 in v146(v129)], device=v30) for v169 in v21]
                    v268 = v45.v296(v97, v98, v100)
                    v301 = v7(v268.v328())
                    v302 = v21[v301] if v301 < v172(v21) else None
                    v303 = v22 < v172(v21)
                    v304 = v179(v268[-1] - v268[:v172(v21)].v297()) if v21 else 0.0
                    (v224 if v303 else v225).v176(v304)
                    v229.v176(1.0 / v172(v21) if v303 else 0.0)
                    if v302 is None:
                        v90['model_r'].v176(v28.v232)
                    else:
                        v90['model_r'].v176(1.0 if v302 == v18['orig'] else -v28.v271)
                        v90['model_acc'].v176(v7(v302 == v18['orig']))
                    v90['model_ans'].v176(v7(v302 is not None))
                    v305 = v320(v18, v18['forged'])
                    if v305 is None:
                        v90['votes_r'].v176(v28.v232)
                    else:
                        v90['votes_r'].v176(1.0 if v305 == v18['orig'] else -v28.v271)
                        v90['votes_acc'].v176(v7(v305 == v18['orig']))
                    v90['votes_ans'].v176(v7(v305 is not None))
                    v306 = [v38['tape'].v71[v171] for v171 in v152['slots']]
                    v307 = v81(v306).v84(1)[0][0]
                    v308 = [v80 for v134, v80 in v239(v18['vals']) if v134 not in v18['forged']]
                    v226.v176(v7(v161(v302) and v81(v308 + [v302]).v84(1)[0][0] == v307))
                    v227.v176(v7(v161(v305) and v81(v308 + [v305]).v84(1)[0][0] == v307))
        v9 = lambda v270: v179(v329.v244(v270)) if v172(v270) else v179('nan')
        v153 = [v230 for v14 in ('replace', 'dup') for v230 in v150[v14]['model']]
        v154 = [v230 for v14 in ('replace', 'dup') for v230 in v150[v14]['votes']]
        v155 = v264.v231(v221, v222)
        v156 = v264.v231(v225, v224)
        return {'n_by_op': {v14: v172(v150[v14]['model']) for v14 in v4}, 'detection': {'model_forged': v9(v153), 'votes_forged': v9(v154), 'random_floor': v9(v228), 'model_dup': v9(v150['dup']['model']), 'votes_dup': v9(v150['dup']['votes']), 'model_clean_pass': v9(v150['none']['model']), 'votes_clean_pass': v9(v150['none']['votes']), 'clean_margin_auc': v155, 'clean_margin_auc_z': v264.v309(v155, v172(v221), v172(v222))}, 'repair_true_flag': {'model_reward': v9(v90['model_r']), 'votes_reward': v9(v90['votes_r']), 'model_accuracy': v9(v90['model_acc']), 'votes_accuracy': v9(v90['votes_acc']), 'model_coverage': v9(v90['model_ans']), 'votes_coverage': v9(v90['votes_ans']), 'random_floor': v9(v229), 'unknown_margin_auc': v156, 'unknown_margin_auc_z': v264.v309(v156, v172(v225), v172(v224)), 'n_recoverable': v172(v224), 'n_unrecoverable': v172(v225), 'single_candidate_skipped': v151}, 'observer_verdict_restored': {'model': v9(v226), 'votes': v9(v227), 'n': v172(v226), 'note': "the proposal's original reward, demoted to an observer: gradient on this has an empty tape as its fixed point - and measured to rank two real arms backwards (min3 repair 0.433 vs votes 0.049 while the observer said 0.507 vs 0.775; min2 repair 0.085 vs 0.184 while the observer said 0.842 vs 0.803)"}}
    v51 = v158(v11)
    v52 = v140(v181.v106(v3 + 99), v44, 1)
    v111(f"  held tape: {v52['n_addresses']} addresses, {v52['n_slots']} slots")
    v53 = v158(v52)
    v111(f'  CONTROL {v272.v235(v51)}')
    v111(f'  HELD {v272.v235(v53)}')
    v86, v159 = (v53['detection'], v53['repair_true_flag'])
    v54 = v40 == v50
    v55 = v160((v53['n_by_op'][v14] >= 2 * v264.v212 for v14 in v4))
    v56 = v161(v86['model_forged'] > v86['votes_forged'] and v86['model_forged'] > v86['random_floor'])
    v57 = v161(not v321.v310(v86['model_dup']) and v86['model_dup'] > v86['votes_dup'])
    v58 = v161(not v321.v310(v86['clean_margin_auc_z']) and v86['clean_margin_auc_z'] > 1.645)
    v59 = v161(v159['model_reward'] > v159['votes_reward'] and v159['model_coverage'] * v53['n_by_op']['replace'] >= v264.v212 and (not v321.v310(v159['model_accuracy'])))
    v60 = v159['n_recoverable'] >= v264.v212 and v159['n_unrecoverable'] >= v264.v212
    v61 = None if not v60 else v161(v159['unknown_margin_auc_z'] > 1.645)
    v62 = 'NO_TASK' if not v55 else 'REPAIR_OK' if v54 and v56 and v57 and v58 and v59 and v61 else 'REPAIR_PARTIAL' if v54 and (v56 or v59) else 'REPAIR_NO'
    v63 = {'stage': 288, 'overall': v62, 'seed': v3, 'smoke': v28.v104, 'run_tag': v28.v162, 'holdout': v28.v123, 'train_steps': v32, 'tape_period': v33, 'params': v47, 'reward': {'correct': 1.0, 'wrong': -v28.v271, 'abstain': v28.v232}, 'train_ops_seen': v195(v145), 'gates': {'G_arc_enc_frozen': v54, 'G_task_exists': v55, 'G_detects_forgery': v56, 'G_detects_dup': v57, 'G_flags_clean': v58, 'G_repairs': v59, 'G_honest_unrecoverable': v61}, 'held_out': v53, 'train_control': v51, 'curve': v144, 'arc_enc_hash_before': v40, 'arc_enc_hash_after': v50, 'fp_version': v191.v233(), 'note': "The direction 286 opened, taken to its end: the tape is not only the label, it is the exercise machine. Break a clean address - swap a mention for a foreign one, or for two or three copies of it - and require where and what, or the admission that the surviving evidence cannot say. The truth is free because the break is ours, so examples are unlimited, honesty is trained on breaks KNOWN to be unrecoverable, and the counting rival is wrong by construction exactly where the forgery is the majority. The judges' consistency - the proposal's original reward - is an observer only: its fixed point as a gradient is an empty tape. Diagnosis and repair share one relational embedding of ranks and indicators, identity unrepresentable, answers are indices, CLEAN and UNKNOWN are rows. Cloze, the lying tape and lie-dup were all points in this corruption family; 288 trains on the family and is examined at points it did not train on: a held-out corpus or held-out subjects, chosen by --holdout.", 'timestamp': v322.v311(v323.v312).v234(), 'wall_s': v108.v108() - v31}
    v0.v67(parents=True, exist_ok=True)
    (v0 / f'stage288_decision{v29}.json').v105(v272.v235(v63, indent=2), encoding='utf-8')
    (v0 / f'stage288_mini{v29}.md').v105(f"# Stage 288 repair ({v28.v123} holdout)\n\n**{v62}**{(' · SMOKE' if v28.v104 else '')} · params **{v47}**\n\n| held out | model | votes | random |\n|---|---:|---:|---:|\n| detect forged | {v86['model_forged']:.3f} | {v86['votes_forged']:.3f} | {v86['random_floor']:.3f} |\n| detect dup | {v86['model_dup']:.3f} | {v86['votes_dup']:.3f} | |\n| clean pass | {v86['model_clean_pass']:.3f} | {v86['votes_clean_pass']:.3f} | |\n| repair reward | {v159['model_reward']:.3f} | {v159['votes_reward']:.3f} | {v159['random_floor']:.3f} |\n\n- clean-margin AUC {v86['clean_margin_auc']:.3f} ({v86['clean_margin_auc_z']:+.2f} sigma)\n- UNKNOWN margin AUC {v159['unknown_margin_auc']:.3f} ({v159['unknown_margin_auc_z']:+.2f} sigma) on {v159['n_unrecoverable']} unrecoverable vs {v159['n_recoverable']} recoverable\n- observer: verdict restored {v53['observer_verdict_restored']['model']:.3f} vs votes {v53['observer_verdict_restored']['votes']:.3f}\n\n## Gates\n\n" + ''.v273((f'- {v74}: **{v80}**\n' for v74, v80 in v63['gates'].v330())), encoding='utf-8')
    v111(v272.v235({'overall': v62, 'gates': v63['gates']}, indent=2))
    return 0
if v64 == '__main__':
    raise v163(v236())