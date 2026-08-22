"""
Stage 275 — Knowing that you do not know.

Aggregation is not the interesting claim. A tally over retrieved witnesses is arithmetic, and 270
showed plain majority already does it. The claim worth making is the one neither lookup nor
majority can express: recognising that the tape does not settle the question, and declining.

So every bank carries three families and the policy is never told which it is looking at:

    clean       one witness                     answer it
    decidable   3 witnesses agree, 2 share a lie   read, tally, answer the majority
    tie         2 and 2, no majority exists      there is no right answer — abstain

Reward makes silence a real option rather than a way to dodge work:

    +1.0  correct answer      -0.3  wrong answer      0.0  abstain      -read_cost per read

A policy that always answers loses on ties. A policy that always abstains scores zero and is
caught by G_answers_when_decidable, which is a validity gate for exactly that reason: the
degenerate solution is cheap and has to be excluded before any abstention number means anything.

The teacher stays executable — it consults no gold value and no family label. A *repeat* is the
dispute signal:

    lead, second = top two frequencies among opened values
    if lead >= 2 and lead > second:   ANSWER the leader     # decidable (and keep reading while
                                                            # second==0 and unread remain, else a
                                                            # 2-0 prefix would fake a majority on ties)
    if lead >= 2 and lead == second:  STOP / abstain        # tie
    else:                             READ if unread left, else ANSWER top retrieve score  # clean

No repeats means no contradiction — answer by retrieval. A repeat with a lead means answer the
majority. A repeat without a lead is the only place to stay silent. 274's strict majority and the
first 275 margin-≥2 rule both failed on clean: unique opens never produce a lead of 2, so the
teacher abstained everywhere and the policy copied silence (ABSTAIN_INVALID).

topk defaults to 7 so five witnesses are not crowded out of the retrieve set by distractors.

The headline is not accuracy. It is risk-coverage: how much of the exam the policy chose to answer,
and how right it was on that part. A mind that answers 60% at 0.95 is better than one that answers
100% at 0.7, and only this table can say so.

  python _stage275_abstain.py --smoke
  python _stage275_abstain.py --rl-episodes 3000
"""
from __future__ import annotations
import argparse
import json
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
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
import _stage274_truthfree_oracle as s274
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, TapeView
from _tape_index import context_words
v0 = v9('results')
v1 = v9('checkpoints/stage191_p1_curve.pt')
v2 = v9('checkpoints/stage253_joint_l02.pt')
v3 = v9('checkpoints/stage275_abstain.pt')
v4 = v9('data/_wikitext103_train.txt')
v5 = 275
v6 = ('clean', 'decidable', 'tie')

def paths(v10: v102):
    v11 = '_frozen' if v10 else ''
    return (v0 / f'stage275_decision{v11}.json', v0 / f'stage275_mini{v11}.md', v0 / f'_stage275_log{v11}.txt')
v7 = v0 / '_stage275_log.txt'

def log(v12: v103) -> None:
    v13 = v12 if v12.v201('\n') else v12 + '\n'
    try:
        v202(v13, end='', flush=True)
    except v104:
        v202(v13.v326('ascii', 'replace').v299('ascii'), end='', flush=True)
    v7.v203.v105(parents=True, exist_ok=True)
    with v7.v204('a', encoding='utf-8') as v106:
        v106.v205(v13)

def build_tape(*, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27):
    """clean / decidable (n_wit-n_liars vs n_liars) / tie (half and half, no majority)."""
    v28 = [v107 for v107 in v19 if v107 not in v21 and v212(v107) >= 5]
    v18.v108(v28)
    v29 = v22 + v23 + v24
    v30 = [v107 for v107 in v266(v131(v21) | v131(v28), v18, v29 + 80) if v212(v107) >= 5 and v107 not in v21]
    v30 = v109(v185.v206(v30))
    v31 = v22 + 2 * (v23 + v24)
    if v212(v30) < v29 or v212(v28) < v31:
        raise v207(f'pool exhausted: subs={v212(v30)} vals={v212(v28)} need={v31}')
    v110, v111, v112, v113 = ([], [], [], [])
    v32 = 0

    def add(v114, v115, v116):
        v117 = v218.v300[v116 % v212(v218.v300)].v143(S=v114, V=v115)
        v118 = v14.v208(v117, exclude=v115)
        v119 = v14.v267([v114])[0]
        v110.v209(v316.v301(v119 + v118, dim=-1) if v118 is not None else v119)
        v111.v209(v115)
        v112.v209(v117)
        return v212(v111) - 1
    v33 = 0

    def witness_block(v120, v121, v122):
        nonlocal si, vi
        v114 = v30[v33]
        v33 += 1
        v123 = v28[v32]
        v32 += 1
        v124 = v28[v32]
        v32 += 1
        v125 = [v123] * v121 + [v124] * v122
        v18.v108(v125)
        v126 = [v210(v114, v128, v268) for v268, v128 in v134(v125)]
        v113.v209({'S': v114, 'truth': None if v120 == 'tie' else v123, 'values': v125, 'slots': v126, 'kind': v120})
        v21.v210(v123)
        v21.v210(v124)
        v21.v210(v114)
    for v34 in v127(v22):
        v114 = v30[v33]
        v33 += 1
        v128 = v28[v32]
        v32 += 1
        v129 = v210(v114, v128, 0)
        v113.v209({'S': v114, 'truth': v128, 'values': [v128], 'slots': [v129], 'kind': 'clean'})
        v21.v210(v128)
        v21.v210(v114)
    for v34 in v127(v23):
        v211('decidable', v25 - v26, v26)
    for v34 in v127(v24):
        v130 = v25 // 2
        v211('tie', v130, v130)
    v35 = v131(v111)
    v36 = v212(v111) + v27
    for v37 in v20:
        if v212(v111) >= v36:
            break
        for v12 in v269.v213(v37):
            v214 = v12.v270(1)
            if v212(v214) < 5 or v214 in v35:
                continue
            v271, v272 = (v164(0, v12.v327() - 120), v302(v212(v37), v12.v328() + 120))
            v118 = v14.v208(v37[v271:v272], exclude=v214)
            if v118 is None:
                continue
            v215 = [v107 for v107 in v329.v317(v37[v271:v12.v327()]) if v107 != v214]
            if not v215:
                continue
            v110.v209(v316.v301(v14.v267([v215[-1]])[0] + v118, dim=-1))
            v111.v209(v214)
            v112.v209(v37[v271:v272])
            v35.v210(v214)
            if v212(v111) >= v36:
                break
    from collections import defaultdict as _dd
    v38 = v132(v109)
    for v133, (v128, v11) in v134(v216(v111, v112)):
        for v107 in v144(v11, exclude=v128):
            v38[v107].v209(v133)
    import math as _m
    v39 = {v107: 1.0 / v303.v167(2.0 + v212(v38[v107])) for v107 in v38}
    return {'tape': v217(v230.v318(v110, 0).v173(v17), v111, v15, v16), 'texts': v112, 'items': v113, 'postings': v38, 'idf': v39}

def teacher(*, v40, v41, v42, v43, v44, v45, v46, v47=None):
    """Executable. Abstains only on a repeated tie. No gold, no family label.

    Repeat is the dispute signal. If the retrieve list itself has no duplicated values, no
    amount of reading can produce a repeat — answer by score immediately (clean). Otherwise
    open slots until: lead>=2 and lead>second with margin ≥2 (or unread exhausted) → ANSWER;
    lead>=2 and lead==second → STOP; exhausted without that → ANSWER by score / first open.
    """
    if not v40:
        return v218.v135
    v48 = v47 or {}

    def answer_value(v136: v103) -> v8:
        for v139, v118 in v134(v40):
            if v142.v273(v118) == v136:
                return 2 + v46 + v139
        return 2 + v46

    def answer_top_score() -> v8:
        v219, v220 = (0, v229('-inf'))
        for v139, v118 in v134(v40):
            v221 = v229(v48.v304(v118, 0.0))
            if v221 > v220:
                v220, v219 = (v221, v139)
        return 2 + v46 + v219
    v49 = v137((v142.v273(v118) for v118 in v40))
    if v49 and v164(v49.v274()) <= 1:
        return v140()
    v50 = v137(v42)
    v51 = v50.v138(2)
    v52 = v51[0][1] if v51 else 0
    v53 = v51[1][1] if v212(v51) > 1 else 0
    v54 = [v139 for v139, v118 in v134(v40) if v118 not in v41]
    if v52 >= 2 and v52 == v53:
        if not v54 or v43 >= v45 or v43 + 2 > v44:
            return 2 + 2 * v46
        return 2 + v54[0]
    if v52 >= 2 and v52 > v53 and (v52 - v53 >= 2 or not v54):
        return v222(v51[0][0])
    if v54 and v43 < v45 and (v43 + 2 <= v44):
        return 2 + v54[0]
    if v42:
        return v222(v42[0])
    return v140()

def rollout(v55, v56, v57, v15, v58, v59, v16, v17, *, v46, v44, v45, v60, v61, v62=False, v63=True, v64=False):
    v141, v38, v39 = (v58['tape'], v58['postings'], v58['idf'])
    v142.v65 = {v139: v128 for v139, v128 in v134(v141.v274)}
    v66 = v218.v223.v143(S=v59['S'])
    v67 = v144(v66)
    v68 = v66
    v40: v109[v8] = []
    v69: v109[v103] = []
    v41: v131[v8] = v131()
    v70: v109[v103] = []
    v145, v146, v147, v148 = ([], [], [], [])
    v43, v149, v150 = (0, None, False)
    for v34 in v127(v44):
        if v64:
            v224 = v275(cands=v40, seen_reads=v41, opened_values=v70, n_reads=v43, max_steps=v44, max_reads=v45, k=v46, cand_scores=v58.v304('_sc'))
        else:
            v225 = v142.v276(v55, v56, v57, v15, v58, v68, v40, v41, v70, v69, v43, v16, v17, v46, v44)
            if v225 is None:
                break
            v277, v34 = v225
            if v62:
                v224 = v275(cands=v40, seen_reads=v41, opened_values=v70, n_reads=v43, max_steps=v44, max_reads=v45, k=v46, cand_scores=v58.v304('_sc'))
                if not v230.v330(v277[v224]) or v277[v224] < -100000000.0:
                    break
                v145.v209(v316.v319(v277.v331(0), v230.v332([v224], device=v17)))
            else:
                v278 = v230.v320.v305(logits=v277)
                v224 = v8(v277.v333()) if v63 else v8(v278.v334())
                v146.v209(v278.v321(v230.v332(v224, device=v17)))
                v147.v209(v278.v322())
        v148.v209(v218.v306(v46)[v224])
        if v224 in (v218.v135, v218.v279):
            v226 = v67 if v224 == v218.v135 else v69
            v40, v48 = v218.v280(v226, v38, v39, v46)
            v227 = [v118 for v118 in v40 if v59['S'] in v58['texts'][v118]]
            v40 = v227 if v227 else v40
            v58['_sc'] = {v118: v48.v304(v118, 0.0) for v118 in v40}
        elif v224 == 2 + 2 * v46:
            v150 = True
            break
        elif v224 < 2 + v46:
            v139 = v224 - 2
            if v139 >= v212(v40):
                break
            v307 = v40[v139]
            v68 = (v68 + ' | ' + v58['texts'][v307])[-2000:]
            v69 = v144(v58['texts'][v307], exclude=v141.v274[v307])
            v41.v210(v307)
            v70.v209(v141.v274[v307])
            v43 += 1
        else:
            v139 = v224 - 2 - v46
            if v139 >= v212(v40):
                break
            v149 = v141.v274[v40[v139]]
            break
    if v150 or v149 is None:
        v151, v71 = (0, 0.0)
        v150 = True
    else:
        v151 = v8(v59['truth'] is not None and v149 == v59['truth'])
        v71 = 1.0 if v151 else -v61
    v71 -= v60 * v43
    return {'loss': v230.v318(v145).v281() if v145 else v230.v282((), device=v17), 'logps': v146, 'entropy': v147, 'reward': v71, 'correct': v151, 'abstained': v150, 'n_reads': v43, 'trace': v148, 'kind': v59['kind'], 'answer_is_slot': v149 is None or v149 in v131(v141.v274)}

def main() -> v8:
    v72 = v228.v152()
    v72.v153('--smoke', action='store_true')
    v72.v153('--bc-episodes', type=v8, default=0)
    v72.v153('--rl-episodes', type=v8, default=0)
    v72.v153('--tape-period', type=v8, default=0)
    v72.v153('--clean', type=v8, default=4)
    v72.v153('--decidable', type=v8, default=4)
    v72.v153('--tie', type=v8, default=4)
    v72.v153('--witnesses', type=v8, default=5)
    v72.v153('--liars', type=v8, default=2)
    v72.v153('--distractor-slots', type=v8, default=0)
    v72.v153('--topk', type=v8, default=7)
    v72.v153('--max-steps', type=v8, default=10)
    v72.v153('--max-reads', type=v8, default=7)
    v72.v153('--read-cost', type=v229, default=0.02)
    v72.v153('--wrong-cost', type=v229, default=0.3)
    v72.v153('--entropy-bonus', type=v229, default=0.01)
    v72.v153('--lr-policy', type=v229, default=0.001)
    v72.v153('--lr-upper', type=v229, default=3e-05)
    v72.v153('--frozen-trunk', action='store_true')
    v73 = v72.v154()
    global LOG_PATH
    v155, v156, v7 = v157(v73.v158)
    v7.v203.v105(parents=True, exist_ok=True)
    v7.v159('', encoding='utf-8')
    v17 = v230.v17('cuda' if v230.v308.v283() else 'cpu')
    v18 = v231.v160(v5)
    v230.v161(v5)
    v74 = v162.v162()
    v75 = v73.v163 or (400 if v73.v197 else 4000)
    v76 = v164(0, v73.v165)
    v77 = v73.v77 or (50 if v73.v197 else 200)
    v27 = v73.v166 or (150 if v73.v197 else 1000)
    v46 = v73.v78
    v79 = 'none' if v73.v158 else 'upper'
    v167(f'Stage275 abstain start {v324.v314(v325.v315).v263()} device={v17} bc={v75} rl={v76} clean={v73.v257} dec={v73.v258} tie={v73.v259} wit={v73.v198} liars={v73.v199} k={v46} wrong_cost={v73.v61} mode={v79}')
    v34, v34, v168, v169 = v170()
    v15 = v232.v171(v103(v284.v233))
    v80 = v15.v172()
    v16 = v15.v234(v235) or 0
    v57 = v309.v285(v15, v168, v16, v80).v173(v17)
    v81 = v2 if v2.v236() else v1
    v56 = v286(v169, v80).v173(v17)
    v56.v174(v230.v287(v81, map_location=v17, weights_only=False)['model'])
    v237.v175(v56, v79)
    v82 = v218.v176(v56)
    v83 = v286(v169, v80).v173(v17)
    v83.v174(v230.v287(v1, map_location=v17, weights_only=False)['model'])
    v83.v177()
    for v84 in v83.v178():
        v84.v238(False)
    v14 = v179(v83, v168, v17)
    with v4.v204('r', encoding='utf-8', errors='ignore') as v106:
        v180 = v106.v239(1500000 if v73.v197 else 8000000)
    v19 = v109(v185.v206((v12.v270(1) for v12 in v269.v213(v180) if v212(v12.v270(1)) >= 5)))
    v18.v108(v19)
    v20 = [v289.v288() for v289 in v180.v310('\n') if v212(v289.v288()) >= 60][:400 if v73.v197 else 6000]
    v55 = v142.v181(2 * (v56.v311.v290 // 2), v46, v17)
    v85 = [v84 for v84 in v56.v178() if v84.v240]
    v86 = v230.v241.v182([{'params': v55.v178(), 'lr': v73.v312}] + ([{'params': v85, 'lr': v73.v323}] if v85 else []), weight_decay=0.01)
    v21: v131[v103] = v131()
    v58, v183, v184 = (None, 0.0, [])
    v87 = v185(k=v46, max_steps=v73.v44, max_reads=v73.v45, read_cost=v73.v60, wrong_cost=v73.v61)

    def new_tape(v186):
        return v242(bank=v14, tok=v15, pad_id=v16, device=v17, rng=v186, pool=v19, lines=v20, used=v21, n_clean=v73.v257, n_dec=v73.v258, n_tie=v73.v259, n_wit=v73.v198, n_liars=v73.v199, n_dist=v27)
    v55.v187()
    v56.v187(v79 != 'none')
    for v88 in v127(1, v75 + 1):
        if v58 is None or (v88 - 1) % v77 == 0:
            v58 = v256(v18)
        v59 = v58['items'][v18.v291(v212(v58['items']))]
        v100 = v243(v55, v56, v57, v15, v58, v59, v16, v17, bc=True, **v87)
        v86.v244(set_to_none=True)
        v100['loss'].v245()
        v230.v313.v292.v246(v109(v55.v178()) + v85, 1.0)
        v86.v247()
        if v88 % v164(1, v75 // 8) == 0:
            v184.v209({'phase': 'bc', 'episode': v88, 'loss': v229(v100['loss']), 'kind': v100['kind'], 'trace': v100['trace']})
            v167(f"  bc {v88}/{v75} loss={v229(v100['loss']):.4f} [{v100['kind']}] {v100['trace']}")
    for v88 in v127(1, v76 + 1):
        if (v88 - 1) % v77 == 0:
            v58 = v256(v18)
        v59 = v58['items'][v18.v291(v212(v58['items']))]
        v100 = v243(v55, v56, v57, v15, v58, v59, v16, v17, greedy=False, **v87)
        if not v100['logps']:
            continue
        v183 = 0.99 * v183 + 0.01 * v100['reward']
        v188 = v230.v318(v100['entropy']).v293() if v100['entropy'] else v230.v282((), device=v17)
        v189 = -(v100['reward'] - v183) * v230.v318(v100['logps']).v293() - v73.v294 * v188
        v86.v244(set_to_none=True)
        v189.v245()
        v230.v313.v292.v246(v109(v55.v178()) + v85, 1.0)
        v86.v247()
        if v88 % v164(1, v76 // 8) == 0:
            v184.v209({'phase': 'rl', 'episode': v88, 'baseline': v183, 'kind': v100['kind'], 'trace': v100['trace']})
            v167(f"  rl {v88}/{v76} baseline={v183:.3f} [{v100['kind']}] {v100['trace']}")
    v55.v177()
    v56.v177()
    v89 = v218.v176(v56)

    @v230.v193()
    def evaluate(v84):
        v190 = {v106: {'correct': [], 'abstain': [], 'reads': [], 'reward': []} for v106 in v6}
        v191 = {v106: {'correct': [], 'abstain': [], 'reward': []} for v106 in v6}
        v248, v249 = ([], [])
        for v192 in v84['items']:
            v250 = v243(v55, v56, v57, v15, v84, v192, v16, v17, **v87)
            v11 = v243(v55, v56, v57, v15, v84, v192, v16, v17, teacher_only=True, **v87)
            v106 = v192['kind']
            v190[v106]['correct'].v209(v250['correct'])
            v190[v106]['abstain'].v209(v8(v250['abstained']))
            v190[v106]['reads'].v209(v250['n_reads'])
            v190[v106]['reward'].v209(v250['reward'])
            v191[v106]['correct'].v209(v11['correct'])
            v191[v106]['abstain'].v209(v8(v11['abstained']))
            v191[v106]['reward'].v209(v11['reward'])
            v248.v209(v8(v250['answer_is_slot']))
            v249.v209({'kind': v106, 'trace': v250['trace'], 'correct': v250['correct'], 'abstained': v250['abstained']})
        v12 = lambda v295: v229(v335.v281(v295)) if v295 else v229('nan')
        v100 = {'answer_is_slot': v12(v248), 'traces': v249, 'reward_total': v12([v186 for v106 in v6 for v186 in v190[v106]['reward']]), 'teacher_reward_total': v12([v186 for v106 in v6 for v186 in v191[v106]['reward']])}
        v251, v252 = (0, 0)
        for v106 in v6:
            v253 = 1.0 - v12(v190[v106]['abstain'])
            v254 = v293((1 for v224 in v190[v106]['abstain'] if not v224))
            v255 = v293(v190[v106]['correct']) / v254 if v254 else v229('nan')
            v251 += v293(v190[v106]['correct'])
            v252 += v254
            v100[v106] = {'coverage': v253, 'acc_answered': v255, 'acc_all': v12(v190[v106]['correct']), 'abstain': v12(v190[v106]['abstain']), 'mean_reads': v12(v190[v106]['reads']), 'reward': v12(v190[v106]['reward']), 'teacher_abstain': v12(v191[v106]['abstain']), 'teacher_acc_all': v12(v191[v106]['correct'])}
        v100['coverage_all'] = v252 / v164(1, v212(v84['items']))
        v100['acc_answered_all'] = v251 / v164(1, v252)
        return v100
    v90 = v194(v58)
    v91 = v194(v256(v231.v160(v5 + 99)))
    v167(f"  NOVEL {v297.v264({v261: v262 for v261, v262 in v91.v113() if v261 != 'traces'})}")
    v92 = v82 == v89
    v93 = v91['answer_is_slot'] >= 0.99
    v94 = v91['clean']['abstain'] <= 0.15 and v91['decidable']['abstain'] <= 0.25
    v95 = v91['tie']['abstain'] >= 0.7
    v96 = v91['tie']['teacher_abstain'] >= 0.7
    v97 = v91['reward_total'] > 0.0
    v98 = v91['reward_total'] >= v90['reward_total'] - 0.15
    v99 = v91['acc_answered_all'] >= 0.75
    if not (v92 and v93 and v94):
        v195 = 'ABSTAIN_INVALID'
    elif not v96:
        v195 = 'TEACHER_CANNOT_ABSTAIN'
    elif v95 and v99 and v98:
        v195 = 'KNOWS_WHAT_IT_DOES_NOT_KNOW'
    elif v95 or v99:
        v195 = 'ABSTAIN_PARTIAL'
    else:
        v195 = 'ABSTAIN_NO'
    v230.v196({'policy': v55.v296(), 'model': v56.v296(), 'stage': 275, 'arc_enc_hash': v89}, v3)
    v100 = {'stage': 275, 'overall': v195, 'frozen_trunk': v73.v158, 'trunk_mode': v79, 'smoke': v73.v197, 'seed': v5, 'bc_episodes': v75, 'rl_episodes': v76, 'families': {'clean': v73.v257, 'decidable': v73.v258, 'tie': v73.v259}, 'witnesses': v73.v198, 'liars': v73.v199, 'topk': v46, 'reward': {'correct': 1.0, 'wrong': -v73.v61, 'abstain': 0.0, 'read': -v73.v60}, 'teacher': 'ASK_Q; read while hunting repeats; ANSWER leader if lead>=2 and lead>second (after opposition seen or unread exhausted); STOP if lead>=2 and lead==second; else ANSWER top retrieve score. No gold, no family.', 'fp_version': v218.v260(), 'used_pool_final': v212(v21), 'gates': {'G_arc_enc_frozen': v92, 'G_answer_is_slot': v93, 'G_answers_when_decidable': v94, 'G_teacher_abstains_on_tie': v96, 'G_abstain_on_tie': v95, 'G_acc_when_answering': v99, 'G_beats_always_answer': v97, 'G_novel_tape': v98}, 'train_tape': {v261: v262 for v261, v262 in v90.v113() if v261 != 'traces'}, 'novel_tape': v91, 'arc_enc_hash_before': v82, 'arc_enc_hash_after': v89, 'curve': v184, 'note': 'Three families, never labelled to the policy: clean, decidable (3 vs 2), tie (2 vs 2, no truth exists). Reward +1 / -0.3 / 0 makes silence a real option, and G_answers_when_decidable is a validity gate because always abstaining scores zero and would otherwise look like wisdom. The teacher reads until the leader is ahead by two — 274 used a strict majority, which one read satisfies, so it answered after a single read and never aggregated. Headline is risk-coverage, not accuracy: answering 60% at 0.95 beats answering everything at 0.7, and only the table shows it.', 'timestamp': v324.v314(v325.v315).v263(), 'wall_s': v162.v162() - v74}
    v0.v105(parents=True, exist_ok=True)
    v155.v159(v297.v264(v100, indent=2), encoding='utf-8')
    v156.v159(f"# Stage 275 abstain\n\n**{v195}**{(' · SMOKE' if v73.v197 else '')}\n\n| family (novel) | coverage | acc answered | abstain | teacher abstain | reads |\n|---|---:|---:|---:|---:|---:|\n" + ''.v298((f"| {v106} | {v91[v106]['coverage']:.2f} | {v91[v106]['acc_answered']:.2f} | {v91[v106]['abstain']:.2f} | {v91[v106]['teacher_abstain']:.2f} | {v91[v106]['mean_reads']:.1f} |\n" for v106 in v6)) + f"\n- overall coverage {v91['coverage_all']:.2f} at accuracy {v91['acc_answered_all']:.2f}\n- reward: policy {v91['reward_total']:.3f} vs teacher {v91['teacher_reward_total']:.3f}\n\n## Gates\n\n" + ''.v298((f'- {v261}: **{v262}**\n' for v261, v262 in v100['gates'].v113())), encoding='utf-8')
    v167(v297.v264({'overall': v195, 'gates': v100['gates']}, indent=2))
    return 0
if v101 == '__main__':
    raise v200(v265())