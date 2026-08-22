"""
Stage 289a — Should this question be answered at all?

The first verb that is not a lookup. Everything until now was
question → read → answer, which is what a reference book does. A mind notices when the
question should not be answered as asked, and that judgment is about the QUERY rather than
about the evidence.

Four verdicts, all constructible from the tape so the labels are free:

  OK              the evidence answers the relation that was asked
  WRONG_RELATION  the evidence is about the subject and answers a DIFFERENT relation
  CONTESTED       the address exists and the corpus never settled it
  FALSE_PREMISE   the question asserts a value the evidence contradicts

WRONG_RELATION is the reason this stage exists, and the first draft of it did not have it.
The obvious construction — "absent" as evidence pulled from other addresses, "false premise"
as a value borrowed from elsewhere, "contested" as the tie family — is a TAUTOLOGY: each class
is defined by exactly the property a two-line hand rule reads off, so the rule scores about
1.0 and nothing is measured. That is 286's exam mistake in a new costume, and it was caught
before the run rather than after.

Wrong-relation is not readable that way, but the second draft nearly was, and the offline test
caught it: if the question ASSERTS the sibling relation's value, the counter sees a value it
cannot find in the evidence and cries false premise. The label was still a property of the
construction rather than of the situation.

So ok and wrong_relation carry NO asserted value. They are queries, not claims - subject plus
the relation being asked - and they differ only in the TEXT of the question. Every count is
then identical between them: the subject appears in every mention, the values cohere, the
majority is clear, and the tie test is negative both times. A counter cannot separate them
even in principle, because the thing that separates them is not a count. That blindness is by
construction, as it is on 288's duplicated forgery, and it is what makes the comparison worth
running.

What CAN separate them is whether the question's own context agrees with the mentions' -
the rank channel the relational mind already has, now measured between the query row and the
evidence instead of between two mentions.

The query enters the graph as one more row - a phantom mention carrying the question's own
context, and an asserted value only where the question actually claims one - so the same
relational machinery compares question to evidence with no new representation. Ranks and indicators only; identity stays unrepresentable. The
output is a verdict from a closed set of four, not a value, so nothing is generated.

The first run measured nothing, and the fault was mine and structural. The claim reached the
graph only as same-value edges on the query row, and that row is all zeros both when the
question claims nothing and when its claim is missing from the evidence - so ok,
wrong_relation and false_premise were the same input. Three classes out of four could not be
separated in principle; the mind called them all false_premise and the counting rival, which
reads the claim directly, looked better on a comparison it could not lose. One indicator on
the query row - does this question assert anything - fixes it without touching identity, and
it leaves exactly ONE pair no count can separate. That pair is the claim, so it is now scored
on its own (blind_pair) rather than inferred from two one-sided recalls.

  python _stage289a_presupposition.py --smoke
  python _stage289a_presupposition.py --train-steps 6000
  python _stage289a_presupposition.py --train-steps 6000 --holdout address
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
import _stage279_write_decision as s279
import _stage280_raw_exam as s280
import _stage286_evidence as s286
from _tape_speed import CachedBank, install_assertion_cache, install_fast_fp_addresses
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
v0 = v9('results')
v1 = v9('checkpoints/stage191_p1_curve.pt')
v2 = v9('data/_wikitext103_train.txt')
v3 = 2891
v4 = ('ok', 'wrong_relation', 'contested', 'false_premise')
v5 = v0 / '_stage289a_log.txt'

def log(v10: v6) -> None:
    v11 = v10 if v10.v182('\n') else v10 + '\n'
    try:
        v183(v11, end='', flush=True)
    except v82:
        v183(v11.v325('ascii', 'replace').v297('ascii'), end='', flush=True)
    v5.v184.v83(parents=True, exist_ok=True)
    with v5.v185('a', encoding='utf-8') as v84:
        v84.v186(v11)

def anchor_of(v12: v6) -> v6:
    return v12.v187(':', 1)[-1].v187('|')[0]

def relation_of(v12: v6) -> v6:
    return (v12.v187(':', 1)[-1].v187('|', 1) + [''])[1]

def make_question(v13, v14, v15, v16, v17=None, v18=None):
    """One question with a known verdict, built out of the tape.

    The asked relation and the evidence come apart only in wrong_relation, which is the whole
    point: there the subject is right, the mentions cohere, and the question is still the wrong
    one. `sibling` is another address of the SAME anchor with a different relation - that is
    what supplies the asked-but-unanswered relation.
    """
    v19 = v13['tape'].v20
    v21 = v85(v14['slots'])
    v22 = [v19[v188] for v188 in v21]
    v23 = v86(v22)
    v24 = v23.v90(1)[0][0]
    v25 = v87(v14['address'])
    if v15 == 'ok':
        v88 = None
    elif v15 == 'false_premise':
        if v18 is None:
            return None
        v88 = v19[v18]
        if v88 in v23:
            return None
    elif v15 == 'contested':
        v28 = v23.v90(2)
        if v216(v28) < 2 or v28[0][1] != v28[1][1]:
            return None
        v88 = v28[0][0]
    elif v15 == 'wrong_relation':
        if v17 is None:
            return None
        v25 = v87(v17['address'])
        if not v25 or v25 == v87(v14['address']):
            return None
        v88 = None
        if v86((v19[v188] for v188 in v17['slots'])).v90(1)[0][0] in v23:
            return None
    else:
        return None
    v26 = (v14['S'] + ' ' + v25).v89()
    return {'verdict': v15, 'slots': v21, 'vals': v22, 'asserted': v88, 'query': v204.v254.v189(S=v26), 'S': v14['S'], 'asked_rel': v25}

def counting_rival(v13, v27):
    """The best a counter can do, and it is blind in exactly one place.

    Subject missing from the mentions is not a class here, so the rule has three moves: a tie
    at the top is contested, an asserted value absent from the evidence is a false premise,
    otherwise the question looks answerable. Wrong-relation presents as answerable to it -
    every count is healthy - which is the blindness this stage measures.
    """
    v23 = v86(v27['vals'])
    v28 = v23.v90(2)
    if v216(v28) > 1 and v28[0][1] == v28[1][1]:
        return 'contested'
    if v27['asserted'] is not None and v27['asserted'] not in v23:
        return 'false_premise'
    return 'ok'

class Judge(v29.v7):
    """The graph of mentions plus the question as one more row, pooled to four verdicts.

    Example-level output, unlike 288's per-row heads, because "what kind of question is this"
    is a property of the whole situation. It is still a closed set of four, so no value is
    produced and nothing is generated - the mind judges the query and never invents an answer.
    """

    def __init__(v91, v33, v92: v8=32, v93: v8=3, v94: v8=4):
        v298().v190()
        v91.v95 = v29.v299(v29.v315(v93, v92), v29.v316()).v120(v33)
        v91.v96 = v29.v299(v29.v315(v94 + 2 * v92, v92), v29.v316()).v120(v33)
        v91.v80 = v29.v299(v29.v315(3 * v92, v92), v29.v316(), v29.v315(v92, v216(v4))).v120(v33)
        v29.v255.v191(v91.v80[-1].v192)
        v29.v255.v191(v91.v80[-1].v193)

    def forward(v91, v97, v98, v99, v100):
        v101 = v91.v95(v97)
        v102 = (v101 * v98).v231(1) / v98.v231(1).v256(min=1.0)
        v103 = v91.v96(v196.v257([v99, v102, v101.v317(1)], -1))
        v104 = v196.v257([v103[:v100], v103[v100 + 1:]], 0) if v103.v300[0] > 1 else v103
        return v91.v80(v196.v257([v104.v317(0), v103[v100], v101[v100].v317(0)]))

def main() -> v8:
    v30 = v194.v105()
    v30.v106('--smoke', action='store_true')
    v30.v106('--train-steps', type=v8, default=0)
    v30.v106('--tape-period', type=v8, default=50)
    v30.v106('--addresses', type=v8, default=0)
    v30.v106('--min-mentions', type=v8, default=2)
    v30.v106('--no-scan-cache', action='store_true', help='disable the exact corpus-scan memo (use to verify it changes nothing)')
    v30.v106('--eval-period', type=v8, default=10)
    v30.v106('--eval-probe', type=v8, default=200)
    v30.v106('--no-fast-grouping', action='store_true', help='disable the batched single-link grouping (use to verify it changes nothing)')
    v30.v106('--wiki-bytes', type=v8, default=0)
    v30.v106('--train-lines', type=v8, default=0)
    v30.v106('--eval-lines', type=v8, default=0)
    v30.v106('--address-tau', type=v195, default=0.9)
    v30.v106('--address-overlap', type=v8, default=2)
    v30.v106('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v30.v106('--lr', type=v195, default=0.001)
    v30.v106('--holdout', choices=('corpus', 'address'), default='corpus')
    v30.v106('--run-tag', type=v6, default='')
    v31 = v30.v107()
    global LOG_PATH
    v32 = v31.v180 and f'_{v31.v180}' or ''
    v32 += '_addrholdout' if v31.v134 == 'address' else ''
    v5 = v0 / f'_stage289a_log{v32}.txt'
    v5.v184.v83(parents=True, exist_ok=True)
    v5.v108('', encoding='utf-8')
    v33 = v196.v33('cuda' if v196.v301.v258() else 'cpu')
    v16 = v197.v109(v3)
    v196.v110(v3)
    v34 = v111.v111()
    v35 = v31.v112 or (600 if v31.v179 else 6000)
    v36 = v31.v113 or (300 if v31.v179 else 400)
    v114(f'Stage289a presupposition start {v323.v313(v324.v314).v251()} device={v33} holdout={v31.v134}')
    v115, v115, v116, v117 = v118()
    v37 = v198.v119(v6(v259.v199))
    v38 = v37.v200(v201) or 0
    v39 = v260(v117, v37.v302()).v120(v33)
    v39.v121(v196.v261(v1, map_location=v33, weights_only=False)['model'])
    v39.v122()
    for v40 in v39.v123():
        v40.v202(False)
    v41 = v124(v203(v39, v116, v33))
    v42 = v204.v125(v39)
    v43: v126[v8, v205] = {}
    v44 = v127.v45

    def _cached_common(v128, v129: v8=3):
        v130 = (v262(v128), v216(v128), v129)
        if v130 not in v43:
            v43[v130] = v44(v128, v129)
        return v43[v130]
    v127.v45 = v46
    if not v31.v131:
        v206(v127)
    if not v31.v132:
        v207(v127)
    with v2.v185('r', encoding='utf-8', errors='ignore') as v84:
        v133 = v84.v208(v31.v263 or (4000000 if v31.v179 else 30000000))
    v47 = [v209.v89() for v209 in v133.v187('\n') if 80 <= v216(v209.v89()) <= 400]
    v48 = v8(0.7 * v216(v47))
    v49 = v47[:v48][:v31.v49 or (3000 if v31.v179 else 25000)]
    v50 = v47[v48:][:v31.v50 or (1500 if v31.v179 else 12000)]
    if v31.v134 == 'address':
        v50 = v49

    def side(v12: v6) -> v8:
        return v8(v332.v326(v333(v12).v325('utf-8')).v303(), 16) & 1

    def new_pack(v135, v128, v136):
        v40 = v264.v210(v128, bank=v41, tok=v37, pad_id=v38, device=v33, rng=v135, n_addr=v36, min_mentions=v31.v265, tau=v31.v266, overlap=v31.v267, soft_match=0.0, min_per_family=8, addr_key=v31.v268)
        if v31.v134 == 'address':
            v40 = v126(v40)
            v40['items'] = [v138 for v138 in v40['items'] if v327(v138['address']) == v136]
        return v40

    def questions(v40, v135):
        """Every verdict the tape can supply, balanced by construction where it can be."""
        v137 = v211(v85)
        for v138 in v40['items']:
            if v216(v138['slots']) >= 2:
                v137[v333(v138['address'])].v243(v138)
        v80 = []
        for v212, v213 in v137.v214():
            for v138 in v213:
                v269 = [v304 for v304 in v213 if v87(v304['address']) != v87(v138['address'])]
                for v226 in v4:
                    if v226 == 'wrong_relation':
                        for v318 in v269:
                            v27 = v319(v40, v138, v226, v135, sibling=v318)
                            if v27 is not None:
                                v80.v243(v27)
                        continue
                    v27 = v319(v40, v138, v226, v135, other=v135.v288(v40['n_slots']) if v226 == 'false_premise' else None)
                    if v27 is not None:
                        v80.v243(v27)
        return v80

    def by_verdict(v139):
        """Class-uniform training, the way 286 samples pair-uniformly: the four verdicts arrive
        in wildly different numbers off the tape, and a mind trained on that imbalance buys
        accuracy by never naming the rare class - which is exactly the failure the first run
        showed. The examiner still sees the natural mix."""
        v92 = v211(v85)
        for v27 in v139:
            v92[v27['verdict']].v243(v27)
        return [v92[v226] for v226 in v4 if v92[v226]]

    def graph(v40, v27):
        v21, v215 = (v27['slots'], v27['vals'])
        v140 = v216(v21)
        v217, v218 = (v40.v270('_ctx', {}), v40.v270('_words', {}))
        for v141 in v205(v21):
            if v141 not in v217:
                v271 = v41.v221(v40['texts'][v141], exclude=v40['tape'].v20[v141])
                v217[v141] = v289.v273(v271, dim=-1) if v271 is not None else None
                v218[v141] = v205(v274(v40['texts'][v141], exclude=v40['tape'].v20[v141]))
        v142 = v40.v219('_median')
        if v142 is None:
            v220 = v272((v216(v226) for v226 in v40['postings'].v20()))
            v142 = v220[v216(v220) // 2] if v220 else 1
            v40['_median'] = v142
        v143 = v41.v221(v27['query'], exclude=v27['asserted'])
        v143 = v289.v273(v143, dim=-1) if v143 is not None else None
        v144 = v205(v274(v27['query'], exclude=v27['asserted']))
        v145 = [v217[v188] for v188 in v21] + [v143]
        v146 = [v218[v188] for v188 in v21] + [v144]
        v147 = v85(v215) + [v27['asserted'] if v27['asserted'] is not None else v320()]
        v10 = v140 + 1
        v98 = v196.v222(v10, v10)
        v148 = v196.v222(v10, v10)
        v149 = v196.v222(v10, v10)
        for v150 in v166(v10):
            for v223 in v166(v150 + 1, v10):
                v98[v150, v223] = v98[v223, v150] = v195(v147[v150] == v147[v223])
                if v145[v150] is not None and v145[v223] is not None:
                    v148[v150, v223] = v148[v223, v150] = v195(v145[v150] @ v145[v223])
                v275 = v146[v150] & v146[v223]
                v276 = v231((1 for v328 in v275 if v216(v40['postings'].v219(v328, ())) < v142))
                v149[v150, v223] = v149[v223, v150] = v276 / v234(1, v329(v216(v146[v150]), v216(v146[v223])))
        v151 = v196.v224(v10, v10, offset=1)

        def rank_norm(v225):
            if v151.v305() == 0:
                return v225
            v226 = v225[v151[0], v151[1]]
            v227 = v226.v277()
            v135 = v196.v278(v227, dtype=v196.v244)
            v135[v227] = v196.v279(v216(v226), dtype=v196.v244)
            v280, v281 = v226.v282(return_inverse=True)
            if v216(v280) > 1:
                v283 = v196.v222(v216(v280)).v306(0, v281, v135, 'mean', include_self=False)
                v135 = v283[v281] / (v216(v226) - 1 if v216(v226) > 1 else 1)
            else:
                v135 = v196.v284(v135)
            v228 = v196.v284(v225)
            v228[v151[0], v151[1]] = v135
            v228[v151[1], v151[0]] = v135
            return v228
        v97 = v196.v307([v98, v330(v148), v330(v149)], -1).v120(v33)
        v23 = v86(v147[:v140])
        v152 = v27['S']
        v153 = v195(v27['asserted'] is not None)
        v99 = v196.v170([[v23.v219(v147[v150], 0) / v140, v195(v152 in v40['texts_lc'][v21[v150]]) if v150 < v140 else 0.0, v195(v150 == v140), v153 if v150 == v140 else 0.0] for v150 in v166(v10)], dtype=v196.v244, device=v33)
        return (v97, v98.v290(-1).v120(v33), v99, v140)
    v51 = v154(v33)
    v52 = v196.v229.v155(v51.v123(), lr=v31.v230, weight_decay=0.01)
    v53 = v8(v231((v308.v305() for v308 in v51.v123())))
    v54 = v111.v111()
    v13 = v156(v16, v49, 0)
    v55 = v157(v13, v16)
    v56 = v158(v55)
    v114(f'  first pack: {v111.v111() - v54:.1f}s (cold - the memo pays back from the second)')
    v114(f"  tape: {v13['n_addresses']} addresses, {v13['n_slots']} slots | questions {v296.v252(v126(v86((v27['verdict'] for v27 in v55))))} | params {v53}")
    if v216(v55) < 4 * v285.v232 or v86((v27['verdict'] for v27 in v55))['wrong_relation'] < v285.v232:
        v114('  too few questions, or no wrong_relation pairs: raise --addresses. wrong_relation needs one anchor to carry two different relations.')
        return 1
    v57 = v156(v197.v109(v3 + 99), v50, 1)
    v58 = v157(v57, v197.v109(v3 + 7))
    v59 = v197.v109(v3 + 11)
    v60 = [v27 for v27 in v58 if v27['verdict'] in ('ok', 'wrong_relation')]
    v59.v159(v60)
    v60 = v60[:v31.v233]
    v61 = v234(v86((v27['verdict'] for v27 in v60)).v20(), default=0) / v234(1, v216(v60))
    v114(f'  saturation probe: {v216(v60)} blind-pair questions, floor {v61:.4f}')

    @v196.v161()
    def probe():
        v51.v122()
        v160 = v231((v8((v4[v8(v51(*v238(v57, v27)).v322())] == 'wrong_relation') == (v27['verdict'] == 'wrong_relation')) for v27 in v60))
        v51.v235()
        return v160 / v234(1, v216(v60))
    v162, v163, v164, v165 = ([], [], [], 1)
    for v62 in v166(1, v35 + 1):
        if (v62 - 1) % v31.v309 == 0 and v62 > 1:
            v236 = v111.v111()
            v13 = v156(v16, v49, 0)
            v55 = v157(v13, v16)
            v56 = v158(v55)
            v114(f'  resample {v165 + 1} at step {v62}: {v111.v111() - v236:.1f}s')
            if not v55:
                v114('  empty tape after resample')
                return 1
            v165 += 1
            if v31.v286 and v165 % v31.v286 == 0:
                v287 = v310()
                v164.v243({'tapes': v165, 'step': v62, 'blind_pair_accuracy': v287})
                v114(f'  [tape {v165}] step {v62} blind_pair={v287:.4f} floor={v61:.4f}')
        v167 = v56[v16.v288(v216(v56))]
        v27 = v167[v16.v288(v216(v167))]
        v97, v98, v99, v237 = v238(v13, v27)
        v168 = v51(v97, v98, v99, v237)
        v169 = v289.v239(v168.v290(0), v196.v170([v4.v321(v27['verdict'])], device=v33))
        v52.v240(set_to_none=True)
        v169.v241()
        v196.v29.v291.v242(v51.v123(), 1.0)
        v52.v62()
        v162.v243(v195(v169))
        if v62 % v234(1, v35 // 8) == 0:
            v163.v243({'step': v62, 'loss': v195(v334.v317(v162[-200:]))})
            v114(f'  step {v62}/{v35} loss={v334.v317(v162[-200:]):.4f}')
    v51.v122()
    v63 = v204.v125(v39)
    v64 = v86((v27['verdict'] for v27 in v55))
    v65 = v196.v170([v311.v114(v234(v64[v226], 1) / v234(1, v231(v64.v20()))) for v226 in v4], dtype=v196.v244, device=v33)
    v114(f'  train prior {v296.v252({v226: v64[v226] for v226 in v4})}')

    @v196.v161()
    def examine(v40, v135, v139=None):
        v139 = v157(v40, v135) if v139 is None else v139
        v171 = {v226: {'n': 0, 'model': 0, 'rival': 0} for v226 in v4}
        v172 = v86()
        v173 = {'n': 0, 'model': 0, 'rival': 0, 'by_true': v86(), 'pos': [], 'neg': [], 'prior': [], 'prior_hit': 0}
        for v27 in v139:
            v97, v98, v99, v237 = v238(v40, v27)
            v245 = v51(v97, v98, v99, v237)
            v246 = v4[v8(v245.v322())]
            v247 = v292(v40, v27)
            v171[v27['verdict']]['n'] += 1
            v171[v27['verdict']]['model'] += v8(v246 == v27['verdict'])
            v171[v27['verdict']]['rival'] += v8(v247 == v27['verdict'])
            v172[v27['verdict'], v246] += 1
            if v27['verdict'] in ('ok', 'wrong_relation'):
                v173['n'] += 1
                v173['by_true'][v27['verdict']] += 1
                v293 = v27['verdict'] == 'wrong_relation'
                v173['model'] += v8((v246 == 'wrong_relation') == v293)
                v294 = v195(v196.v331(v245, -1)[v4.v321('wrong_relation')])
                (v173['pos'] if v293 else v173['neg']).v243(v294)
                v295 = v245 + v65
                v173['prior_hit'] += v8((v4[v8(v295.v322())] == 'wrong_relation') == v293)
                v173['rival'] += v8((v247 == 'wrong_relation') == v293)
        v174 = v285.v248(v173['pos'], v173['neg'])
        v175 = v285.v249(v174, v216(v173['pos']), v216(v173['neg']))
        v140 = v231((v171[v226]['n'] for v226 in v4))
        v80 = {'n': v140, 'model_accuracy': v231((v171[v226]['model'] for v226 in v4)) / v234(1, v140), 'rival_accuracy': v231((v171[v226]['rival'] for v226 in v4)) / v234(1, v140), 'majority_floor': v234((v171[v226]['n'] for v226 in v4)) / v234(1, v140), 'per_verdict': {v226: {'n': v171[v226]['n'], 'model_recall': v171[v226]['model'] / v234(1, v171[v226]['n']), 'rival_recall': v171[v226]['rival'] / v234(1, v171[v226]['n'])} for v226 in v4}, 'blind_pair': {'n': v173['n'], 'model_accuracy': v173['model'] / v234(1, v173['n']), 'prior_corrected_accuracy': v173['prior_hit'] / v234(1, v173['n']), 'auc': v174, 'auc_z': v175, 'n_wrong_relation': v216(v173['pos']), 'n_ok': v216(v173['neg']), 'rival_accuracy': v173['rival'] / v234(1, v173['n']), 'majority_floor': v234(v173['by_true'].v20(), default=0) / v234(1, v173['n'])}, 'confusion': {f'{v287}->{v167}': v271 for (v287, v167), v271 in v272(v172.v214())}}
        return v80
    v66 = v176(v13, v197.v109(v3 + 5))
    v67 = v176(v57, v197.v109(v3 + 7), qq=v58)
    v114(f'  CONTROL {v296.v252(v66)}')
    v114(f'  HELD {v296.v252(v67)}')
    v68 = v67['per_verdict']['wrong_relation']
    v69 = v42 == v63
    v70 = v177((v67['per_verdict'][v226]['n'] >= v285.v232 for v226 in v4))
    v71 = v178(v67['model_accuracy'] > v67['rival_accuracy'])
    v72 = v178(v67['model_accuracy'] > v67['majority_floor'])
    v73 = v178(v68['n'] >= v285.v232 and v68['model_recall'] > v68['rival_recall'])
    v74 = v67['per_verdict']['ok']
    v75 = v178(v74['n'] >= v285.v232 and v74['model_recall'] >= v74['rival_recall'])
    v76 = v67['blind_pair']
    v77 = v178(v76['n'] >= 2 * v285.v232 and (not v311.v312(v76['auc_z'])) and (v76['auc_z'] > 1.645))
    v78 = v178(v76['prior_corrected_accuracy'] > v76['majority_floor'])
    v79 = 'NO_TASK' if not (v70 and v69) else 'PRESUPPOSITION_OK' if v71 and v72 and v77 and v73 and v75 else 'PRESUPPOSITION_PARTIAL' if v73 or v71 else 'PRESUPPOSITION_NO'
    v80 = {'stage': '289a', 'overall': v79, 'seed': v3, 'smoke': v31.v179, 'holdout': v31.v134, 'run_tag': v31.v180, 'train_steps': v35, 'params': v53, 'verdicts': v85(v4), 'gates': {'G_arc_enc_frozen': v69, 'G_task_exists': v70, 'G_beats_counting_rival': v71, 'G_beats_majority_floor': v72, 'G_catches_wrong_relation': v73, 'G_does_not_cry_wolf': v75, 'G_separates_blind_pair': v77, 'G_blind_pair_usable_at_argmax': v78}, 'rival_ceiling_note': 'the counting rival solves contested and false_premise and is blind to the ok / wrong_relation split by construction, so its ceiling on a balanced set is about 0.50; overall accuracy above it is necessary, and G_catches_wrong_relation is the claim', 'blind_pair_note': "G_separates_blind_pair reads the AUC because the argmax cannot settle it: training draws the four verdicts uniformly, by design, and the examiner sees the tape's natural mix, so comparing an argmax against the majority floor charges the mind for a prior shift the examiner introduced. The AUC is prior-free and its null point is 286's usual 1.645. prior_corrected_accuracy is the argmax after the exact Bayes correction using the TRAIN tape's frequencies - never the held-out ones - and it is what someone using this mind would actually get", 'held_out': v67, 'train_control': v66, 'curve': v163, 'n_tapes': v165, 'tape_curve': v164, 'tape_curve_note': f'blind-pair accuracy on ONE fixed held-out tape, measured every {v31.v286} distinct training tapes against floor {v61:.4f}. Where it flattens is how many tapes the mind actually needs; resampling is not a simulation of use - real use is one growing tape - it is the proof that no single tape was memorised, which is the whole separation claim', 'tape_curve_floor': v61, 'tape_curve_n': v216(v60), 'arc_enc_hash_before': v42, 'arc_enc_hash_after': v63, 'fp_version': v204.v250(), 'note': "The first verb that is not a lookup: should this question be answered at all. Four verdicts built from the tape so the labels are free - ok, wrong_relation, contested, false_premise. The first draft was a tautology and was caught before the run: building 'absent' as evidence from other addresses and 'false premise' as a borrowed value makes each class exactly the property a two-line hand rule reads off, so the rule scores 1.0 and nothing is measured, which is 286's exam mistake again. wrong_relation replaces that: ask about one relation of a subject while the evidence answers another, and every counting signal stays HEALTHY - the subject is in every mention, the values cohere, the majority is clear - so a counter answers confidently and answers a different question. Its blindness there is by construction, as it is on 288's duplicated forgery. The query joins the graph as one more row carrying the asserted value and the question's own context, so the same relational machinery compares question to evidence with no new representation. Ranks and indicators only; the output is one of four verdicts, so no value is produced and nothing is generated. THE FIRST RUN WAS UNMEASURABLE and the defect was in the graph, not in the mind: the asserted value reached the graph only through the query row's same-value edges, and that row is all-zero both when the question claims nothing and when its claim is absent from the evidence, while ctx_fp excluded the value from the text as well - so ok, wrong_relation and false_premise were literally the same input and three of four classes were indistinguishable in principle. The mind put all of them in one class (ok recall 0.0, wrong_relation recall 0.0, on the held-out set AND on the train control) while the rival read the claim directly in python, so it was being compared against a strictly better-informed opponent. Fixed by one identity-free indicator - does this query assert anything - which leaves exactly one pair, ok / wrong_relation, blind to every count, and that pair is now scored on its own as blind_pair. Second defect, same run: one random sibling per item gave wrong_relation n=6 held out, under MIN_ANSWERED, so the class the stage exists for had no denominator; every eligible sibling is now its own question, and training samples the four verdicts uniformly so accuracy cannot be bought by never naming the rare one.", 'timestamp': v323.v313(v324.v314).v251(), 'wall_s': v111.v111() - v34}
    v0.v83(parents=True, exist_ok=True)
    (v0 / f'stage289a_decision{v32}.json').v108(v296.v252(v80, indent=2), encoding='utf-8')
    v114(v296.v252({'overall': v79, 'gates': v80['gates'], 'model': v67['model_accuracy'], 'rival': v67['rival_accuracy'], 'wrong_relation': v68}, indent=2))
    return 0
if v81 == '__main__':
    raise v181(v253())