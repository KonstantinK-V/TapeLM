"""
Stage 267 — Read, refine, read again: the mind's job is the loop, not the projection.

266 closed the single-shot question. A learned query vector loses to plain words on every trunk
(top1 0.062 vs 0.199); LLM keywords do not beat the question's own words; blind paraphrase wakes
4 silent queries out of 176. The cause was named in 266's own data: the model invents words that
exist somewhere on the tape (novel_on_tape 0.756) but not beside the fact being asked for. It
cannot bridge because it does not know what is written.

But after one retrieval it does know. That is the whole idea here:

    hop 0   question words          -> votes -> top-k slots
    read    the write-contexts of those slots, verbatim tape text
    hop 1   words chosen while LOOKING at that text -> votes again

The new words are taken from the tape rather than guessed at it. This is 257's mechanism — the
second hop anchored on what the first hop returned — carried over from the composition exam to
the open-bank query. Nothing in memory changes: same bank, same postings, same idf, zero trained
parameters at read time.

Arms, so that "the loop helps" cannot be confused with "prompting helps":

    A  hop0            surface words only                    (= 264's votes arm, validity anchor)
    B  refine_grounded refine while reading the RETRIEVED passages
    C  refine_random   refine while reading RANDOM passages   (causal control for grounding)
    D  refine_selective grounded, but only where hop0 is uncertain
    E  refine_blind    refine with no passages at all         (= 266's paraphrase, reference)

C is the gate that matters. If B ≈ C the loop is not reading the tape, it is just being prompted,
and 267 is a NO however good B looks. D exists because 266 taught that adding words to an already
healthy query hurts it — union lost 0.199 -> 0.165 — so refinement has to be spent only where hop0
is in trouble, and "in trouble" must be computable without the gold slot.

Silence is scored on the gold vote mass, and rank counts a silent gold as last, not first: 266's
paraphrase arm read top1 0.477 purely because 71 empty answers left an empty score table and the
old formula called that rank 1.

  python _stage267_read_refine.py [--smoke]
  python _stage267_read_refine.py --k-read 5 --hops 2
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage24x_lib as L
from _stage191_night import SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE
from _stage261_nl_query import WORD_RE, collect, ctx_words, jaccard
from _stage262_trunk_swap import ExternalTrunk
from _tape_index import context_words
v0 = v15('results')
v1 = v0 / 'stage267_decision.json'
v2 = v0 / 'stage267_mini.md'
v3 = v0 / '_stage267_log.txt'
v4 = v0 / 'stage264_decision.json'
v5 = v15('checkpoints/stage191_p1_curve.pt')
v6 = v15('data/_wikitext103_train.txt')
v7 = 261
v8 = 'Qwen/Qwen2.5-0.5B-Instruct'
v9 = 'Text fragment:\n{q}\n\nPassages retrieved from an index (some may be irrelevant):\n{passages}\n\nList 3-8 English content words that would most likely appear in the passage that actually continues the fragment. Prefer words you can see in the passages above when they look relevant to the fragment. Comma-separated words only — no sentences.'
v10 = 'Text fragment:\n{q}\n\nList 3-8 English content words that would most likely appear in the passage that actually continues the fragment. Comma-separated words only — no sentences.'

def log(v16: v11) -> None:
    v17 = v16 if v16.v211('\n') else v16 + '\n'
    try:
        v212(v17, end='', flush=True)
    except v110:
        v212(v17.v292('ascii', 'replace').v123('ascii'), end='', flush=True)
    v3.v213.v111(parents=True, exist_ok=True)
    with v3.v214('a', encoding='utf-8') as v112:
        v112.v215(v17)

def ensure_short_hf_home() -> v11 | None:
    """Windows + long HF hub paths → OSError Errno 22. Prefer a short HF_HOME."""
    if v216.v113 != 'nt':
        return None
    v18 = v216.v115.v130('HF_HOME') or v216.v115.v130('HUGGINGFACE_HUB_CACHE')
    if v18 and v156(v15(v18).v293().v273()) < 40:
        return v18
    v19 = v15(v216.v115.v130('SOTE_HF_HOME', 'C:\\hf'))
    try:
        v19.v111(parents=True, exist_ok=True)
    except v114:
        v19 = v15.v274() / 'hf'
        v19.v111(parents=True, exist_ok=True)
    v216.v115['HF_HOME'] = v11(v19)
    v216.v115.v116('HUGGINGFACE_HUB_CACHE', v11(v19 / 'hub'))
    v216.v115.v116('TRANSFORMERS_CACHE', v11(v19 / 'transformers'))
    return v11(v19)

def free_cuda() -> None:
    v217.v117()
    if v126.v218.v118():
        v126.v218.v219()

def chat_wrap(v20, v21: v11) -> v11:
    return v20.v119([{'role': 'user', 'content': v21}], tokenize=False, add_generation_prompt=True)

@v126.v32()
def generate_word_list(v22: v120, v23: v11, *, v24: v12=48) -> v33[v11]:
    v25 = v121(v22.v20, v23)
    v26 = v22.v20(v25, return_tensors='pt', truncation=True, max_length=1024)
    v26 = {v71: v252.v178(v22.v75) for v71, v252 in v26.v46()}
    v27 = v12(v26['input_ids'].v220[1])
    v28 = v22.v207.v122(**v26, max_new_tokens=v24, do_sample=False, pad_token_id=v22.v20.v221)
    v29 = v22.v20.v123(v28[0][v27:], skip_special_tokens=True)
    v31, v124 = ([], v157())
    for v30 in v222.v125('[,;\\n|/]+', v29):
        for v37 in v253.v223(v30):
            v224 = v37.v248()
            if v224 not in v124:
                v124.v262(v224)
                v31.v229(v224)
    return v31

def vote(v31, v34, v35) -> v14[v12, v128]:
    v36: v14[v12, v128] = v127(v128)
    for v37 in v31:
        for v129 in v34.v130(v37, ()):
            v36[v129] += v35.v130(v37, 0.0)
    return v36

def rank_of(v36: v14[v12, v128], v38: v12, v39: v12) -> v12:
    """Silence is last, not first.

    266's paraphrase arm scored top1 0.477 because 71 queries produced no words, left `sc` empty,
    and `1 + len({v > 0})` called that rank 1. A gold slot with no votes is tied with every other
    unvoted slot; the pessimistic reading is the only honest one.
    """
    v40 = v36.v130(v38, 0.0)
    if v40 <= 0.0:
        return v39
    return 1 + v187((1 for v252 in v36.v225() if v252 > v40))

def nway_strict(v41: v128, v42) -> v13:
    return v131((v41 > v254 for v254 in v42))

def hop0_uncertainty(v36: v14[v12, v128]) -> v45[v13, v128]:
    """Is this query in trouble? Computable without the gold slot.

    Silent (nothing voted) or a flat top — those are the queries worth spending a refinement on.
    266's union arm lost top1 precisely by spending words on queries that were already fine.
    """
    if not v36:
        return (True, 0.0)
    v43 = v132(v36.v225(), reverse=True)
    if v156(v43) == 1:
        return (False, 1.0)
    v44 = (v43[0] - v43[1]) / v147(v43[0], 1e-09)
    return (v44 < 0.15, v44)

def score_arm(v46, v47, v34, v35, v39, v48, *, v49=20) -> v14:
    v50 = v226.v133(v7 + 5)
    v134, v135, v136, v137 = ([], [], [], [])
    for v51 in v46:
        v36 = v227(v47(v51), v34, v35)
        v40 = v36.v130(v51['slot'], 0.0)
        v52 = v228(v36, v51['slot'], v39)
        v134.v229(v52)
        v136.v229(v40)
        v138 = [v255 for v255 in v50.v286(v279(v39), v276(v49 * 3, v39)) if v255 != v51['slot']][:v49 - 1]
        v135.v229(v12(v275(v40, (v36.v130(v255, 0.0) for v255 in v138))))
        v137.v229(v51['overlap'] <= v48)
    v52 = v230.v139(v134, dtype=v230.v231)
    v53 = v230.v139([v40 <= 0.0 for v40 in v136])
    v54 = v230.v139(v137)
    v55 = v52 == 1

    def _m(v140, v141):
        return v128(v141[v140].v257()) if v140.v256() else v128('nan')
    return {'top1': v128(v55.v257()), 'mrr': v128(v230.v257(1.0 / v52)), 'median_rank': v128(v230.v241(v52)), 'acc_20way': v128(v230.v257(v135)), 'chance_20way': 1.0 / v49, 'tie_at_zero_frac': v128(v53.v257()), 'tie_at_zero_frac_low_overlap': v232(v54, v53), 'tie_at_zero_frac_high_overlap': v232(~v54, v53), 'top1_low_overlap': v232(v54, v55), 'top1_high_overlap': v232(~v54, v55), 'top1_low_overlap_given_vote': v232(v54 & ~v53, v55), 'n': v156(v134), 'n_tie_at_zero': v12(v53.v187()), 'n_low_overlap': v12(v54.v187())}

def woken_frac(v46, v56, v57, v34, v35) -> v14:
    """Queries the tape was silent on that the refinement actually gave vote mass to.

    This is the metric the mind has to move. It is not tie_at_zero on its own: waking a query with
    wrong words converts "no answer" into "wrong answer", so the rank of the woken ones is
    reported beside the count.
    """
    v142, v143, v144 = (0, 0, [])
    v39 = None
    for v51 in v46:
        v145 = v227(v56(v51), v34, v35)
        if v145.v130(v51['slot'], 0.0) > 0.0:
            continue
        v146 = v227(v57(v51), v34, v35)
        if v146.v130(v51['slot'], 0.0) > 0.0:
            v142 += 1
            v52 = 1 + v187((1 for v252 in v146.v225() if v252 > v146[v51['slot']]))
            v144.v229(v52)
            v143 += v12(v52 == 1)
    v58 = v147(1, v156(v46))
    return {'woken_n': v142, 'woken_frac': v142 / v58, 'woken_top1': v143 / v142 if v142 else v128('nan'), 'woken_median_rank': v128(v230.v241(v144)) if v144 else v128('nan')}

def build_exam(v59, v60, v61, v62, v63):
    v64 = v117(v60, v59)
    v65 = v132(v64)[:v61]
    v63.v148(v65)
    v149, v150, v46, v70 = ([], [], [], [])
    for v66 in v65:
        v151 = v64[v66]
        v146, v145 = (v151[0], v151[1])
        v152 = v146['line'][v147(0, v146['start'] - 140):v276(v156(v146['line']), v146['end'] + 140)]
        v153 = v145['line'][v147(0, v145['start'] - 200):v145['start']].v233()
        v154 = v234(v152, exclude=v66)
        v155 = v234(v153, exclude=v66)
        if v156(v154) < 4 or v156(v155) < 4:
            continue
        v46.v229({'ent': v66, 'slot': v156(v149), 'qtext': v153, 'wctx': v152, 'qwords': v155, 'overlap': v277(v287(v152, v66), v287(v153, v66))})
        v149.v229(v66)
        v150.v229(v154)
        v70.v229(v152)
    v67 = v156(v149)
    v68 = v157(v149)
    for v69 in v60:
        if v156(v149) >= v67 + v62:
            break
        for v16 in v258.v235(v69):
            v66 = v16.v259(1)
            if v156(v66) < 5 or v66 in v68:
                continue
            v260, v261 = (v147(0, v16.v294() - 140), v276(v156(v69), v16.v295() + 140))
            v154 = v234(v69[v260:v261], exclude=v66)
            if v156(v154) < 4:
                continue
            v149.v229(v66)
            v150.v229(v154)
            v70.v229(v69[v260:v261])
            v68.v262(v66)
            if v156(v149) >= v67 + v62:
                break
    v34: v14[v11, v33[v12]] = v127(v33)
    for v129, v154 in v158(v150):
        for v37 in v154:
            v34[v37].v229(v129)
    v35 = {v37: 1.0 / v278.v172(2.0 + v156(v34[v37])) for v37 in v34}
    return (v149, v70, v46, v67, v34, v35)

def passages_for(v36: v14[v12, v128], v70, v71: v12, v63=None, v39=0) -> v33[v11]:
    """Top-k retrieved write-contexts, or k random ones for the grounding control."""
    if v63 is not None:
        v159 = [v63.v263(v39) for v173 in v279(v71)]
    else:
        v159 = [v129 for v129, v173 in v132(v36.v46(), key=lambda v283: -v283[1])[:v71]]
    return [v70[v188][:300] for v188 in v159]

def published_264_surface() -> v14 | None:
    if not v4.v236():
        return None
    try:
        v160 = v264.v237(v4.v265(encoding='utf-8'))
        return (v160.v130('summary') or {}).v130('retrieval', {}).v130('votes')
    except v161:
        return None

def main() -> v12:
    v72 = v238.v162()
    v72.v163('--smoke', action='store_true')
    v72.v163('--entities', type=v12, default=0)
    v72.v163('--distractor-slots', type=v12, default=0)
    v72.v163('--model', type=v11, default=v8)
    v72.v163('--k-read', type=v12, default=5, help='passages shown to the model at hop 1')
    v72.v163('--n-way', type=v12, default=20)
    v72.v163('--margin-thresh', type=v128, default=0.15, help='hop0 top1/top2 margin below which the selective arm refines')
    v72.v163('--no-blind', action='store_true', help='skip the E reference arm')
    v73 = v72.v164()
    v3.v165('', encoding='utf-8')
    v74 = v166()
    v75 = v126.v75('cuda' if v126.v218.v118() else 'cpu')
    v63 = v226.v133(v7)
    v126.v167(v7)
    v76 = v168.v168()
    v61 = v73.v169 or (60 if v73.v171 else 400)
    v62 = v73.v170 or (400 if v73.v171 else 4000)
    v77 = 3000 if v73.v171 else 25000
    v172(f'Stage267 read-refine start {v289.v284(v290.v285).v249()} device={v75} model={v73.v207} k_read={v73.v208}' + (f' HF_HOME={v74}' if v74 else ''))
    v173, v173, v174, v175 = v176()
    v78 = v280.v266(v11(v288.v281)).v177()
    v79 = v267(v175, v78).v178(v75)
    v79.v179(v126.v268(v5, map_location=v75, weights_only=False)['model'])
    v79.v180()
    for v80 in v79.v181():
        v80.v239(False)
    v59 = v182(v79, v174, v75)
    with v6.v214('r', encoding='utf-8', errors='ignore') as v112:
        v183 = v112.v240(3000000 if v73.v171 else 20000000)
    v60 = [v269.v233() for v269 in v183.v125('\n') if 80 <= v156(v269.v233()) <= 400][:v77]
    v149, v70, v46, v67, v34, v35 = v184(v59, v60, v61, v62, v63)
    if v156(v46) < 16:
        v172('  not enough exam pairs')
        return 1
    v39 = v156(v149)
    v48 = v128(v230.v241([v51['overlap'] for v51 in v46]))
    v172(f'  exam={v67} bank={v39} vocab={v156(v34)} postings={v187((v156(v252) for v252 in v34.v225()))} overlap_med={v48:.3f}')
    del v79
    v185()
    v81 = v186(v46, lambda v51: v51['qwords'], v34, v35, v39, v48, n_way=v73.v49)
    v172(f'  A hop0 surface: {v264.v250(v81)}')
    v82 = {}
    for v51 in v46:
        v36 = v227(v51['qwords'], v34, v35)
        v242, v44 = v243(v36)
        v82[v51['slot']] = {'sc': v36, 'uncertain': v242, 'margin': v44}
    v83 = v187((1 for v252 in v82.v225() if v252['uncertain']))
    v172(f'  hop0 uncertain (silent or margin<{v73.v209}): {v83}/{v156(v46)}')
    v172(f'\n== loading {v73.v207} ==')
    try:
        v22 = v120(v73.v207, v75)
    except v161 as e:
        v172(f'  LOAD FAIL: {v296(v66).v109}: {v66}')
        v1.v165(v264.v250({'stage': 267, 'overall': 'READ_REFINE_INVALID', 'error': f'{v296(v66).v109}: {v66}', 'model': v73.v207, 'timestamp': v289.v284(v290.v285).v249()}, indent=2), encoding='utf-8')
        return 1
    if not v244(v22.v20, 'chat_template', None):
        v172('  FATAL: model has no chat_template — an Instruct run without it is a base run')
        v1.v165(v264.v250({'stage': 267, 'overall': 'READ_REFINE_INVALID', 'error': 'missing_chat_template', 'model': v73.v207, 'timestamp': v289.v284(v290.v285).v249()}, indent=2), encoding='utf-8')
        return 1
    v84 = v226.v133(v7 + 77)
    v85: v14[v12, v33[v11]] = {}
    v86: v14[v12, v33[v11]] = {}
    v87: v14[v12, v33[v11]] = {}
    v88 = {'grounded': 0, 'random': 0, 'blind': 0}
    v89 = []
    for v188, v51 in v158(v46):
        v36 = v82[v51['slot']]['sc']
        v189 = v245(v36, v70, v73.v208)
        v23 = v9.v246(q=v51['qtext'], passages='\n---\n'.v272(v189))
        v40 = v247(v22, v23)
        v85[v51['slot']] = v40
        v88['grounded'] += v12(not v40)
        v190 = v245(None, v70, v73.v208, rng=v84, n_slots=v39)
        v191 = v9.v246(q=v51['qtext'], passages='\n---\n'.v272(v190))
        v192 = v247(v22, v191)
        v86[v51['slot']] = v192
        v88['random'] += v12(not v192)
        if not v73.v197:
            v145 = v247(v22, v10.v246(q=v51['qtext']))
            v87[v51['slot']] = v145
            v88['blind'] += v12(not v145)
        if v156(v89) < 5:
            v89.v229({'qtext': v51['qtext'][:120], 'hop0_uncertain': v82[v51['slot']]['uncertain'], 'passages_head': [v80[:90] for v80 in v189[:2]], 'grounded': v40[:8], 'random_ctrl': v192[:8], 'from_passages': [v37 for v37 in v40 if v256((v37 in v80.v248() for v80 in v189))][:8]})
        if (v188 + 1) % 25 == 0:
            v172(f'    refined {v188 + 1}/{v156(v46)} ({v168.v168() - v76:.0f}s)')
    del v22
    v185()

    def _union(v193, v194):
        return lambda v51: v33(v14.v282(v33(v51['qwords']) + v33(v194.v130(v51['slot'], []))))

    def _selective(v194):

        def f(v51):
            if v82[v51['slot']]['uncertain']:
                return v33(v14.v282(v33(v51['qwords']) + v33(v194.v130(v51['slot'], []))))
            return v33(v51['qwords'])
        return v112
    v90 = v195(None, v85)
    v91 = v195(None, v86)
    v92 = v196(v85)
    v93 = v186(v46, v90, v34, v35, v39, v48, n_way=v73.v49)
    v94 = v186(v46, v91, v34, v35, v39, v48, n_way=v73.v49)
    v95 = v186(v46, v92, v34, v35, v39, v48, n_way=v73.v49)
    v96 = None
    if not v73.v197:
        v96 = v186(v46, v195(None, v87), v34, v35, v39, v48, n_way=v73.v49)
    v97 = v198(v46, lambda v51: v51['qwords'], v90, v34, v35)
    v98 = v198(v46, lambda v51: v51['qwords'], v91, v34, v35)
    v99 = v198(v46, lambda v51: v51['qwords'], v92, v34, v35)
    v199, v200 = (0, 0)
    for v51 in v46:
        v189 = ' '.v272(v245(v82[v51['slot']]['sc'], v70, v73.v208)).v248()
        for v37 in v85.v130(v51['slot'], []):
            v200 += 1
            v199 += v12(v37 in v189)
    v100 = v199 / v147(1, v200)
    v172(f'  B grounded : {v264.v250(v93)}\n  woken {v264.v250(v97)}')
    v172(f'  C random   : {v264.v250(v94)}\n  woken {v264.v250(v98)}')
    v172(f'  D selective: {v264.v250(v95)}\n  woken {v264.v250(v99)}')
    if v96:
        v172(f'  E blind    : {v264.v250(v96)}')
    v172(f'  copy_rate (refined words seen in the shown passages): {v100:.3f}')

    def headline_beats(v201, v193):
        return v13(v201['top1'] >= v193['top1'] + 0.03 or v201['acc_20way'] >= v193['acc_20way'] + 0.05)
    v101 = v202()
    v102 = v101 is None or v270(v81['top1'] - v128(v101.v130('top1', v81['top1']))) <= 0.05
    v203, v204 = v147([('B_grounded', v93), ('D_selective', v95)], key=lambda v283: v283[1]['top1'])
    v103 = v205(v204, v81)
    v104 = v13(v204['top1'] >= v94['top1'] + 0.05 or v204['acc_20way'] >= v94['acc_20way'] + 0.05)
    v105 = v13(v81['tie_at_zero_frac'] - v204['tie_at_zero_frac'] >= 0.05)
    v106 = v13((v97 if v203 == 'B_grounded' else v99)['woken_frac'] >= 0.05 and v204['top1'] >= v81['top1'] - 0.02)
    v107 = v13(v95['top1'] >= v93['top1'] + 0.03)
    v108 = v13(v100 >= 0.3)
    if not v102:
        v206 = 'READ_REFINE_INVALID'
    elif v103 and v104:
        v206 = 'READ_REFINE_OK'
    elif v104 and (v105 or v106):
        v206 = 'READ_REFINE_PARTIAL'
    elif v103 and (not v104):
        v206 = 'PROMPTING_NOT_READING'
    else:
        v206 = 'READ_REFINE_NO'
    v28 = {'stage': 267, 'overall': v206, 'model': v73.v207, 'seed': v7, 'smoke': v73.v171, 'bank_slots': v39, 'exam_slots': v67, 'n_eval': v156(v46), 'k_read': v73.v208, 'margin_thresh': v73.v209, 'overlap_median': v48, 'trained_parameters': 0, 'fp_version': v244(v271, 'canonical_fp_version', lambda: v5.v113)(), 'hop0_uncertain_n': v83, 'empty_generations': v88, 'copy_rate_from_passages': v100, 'gates': {'G_hop0_reproduces_264': v102, 'G_refine_beats_hop0': v103, 'G_grounding_causal': v104, 'G_silence_reduced': v105, 'G_woken_useful': v106, 'G_selective_beats_always': v107, 'G_reads_passages': v108, 'best_arm': v203}, 'arms': {'A_hop0_surface': v81, 'B_refine_grounded': v93, 'C_refine_random_passages': v94, 'D_refine_selective': v95, 'E_refine_blind': v96}, 'woken': {'B_grounded': v97, 'C_random': v98, 'D_selective': v99}, 'reference_264_votes': v101, 'samples': v89, 'note': "The loop, not the projection: hop0 votes retrieve passages, the model picks words while LOOKING at them, hop1 votes again. C shows the same model the same number of RANDOM passages — if B does not beat C the loop is prompting, not reading, and the verdict is PROMPTING_NOT_READING however good B looks. D spends refinement only where hop0 is uncertain, because 266 showed extra words hurt an already-healthy query. Rank counts a silent gold as last: 266's paraphrase arm read top1 0.477 only because empty answers left an empty score table. Memory is untouched — same bank, same postings, zero trained parameters.", 'timestamp': v289.v284(v290.v285).v249(), 'wall_s': v168.v168() - v76}
    v0.v111(parents=True, exist_ok=True)
    v1.v165(v264.v250(v28, indent=2), encoding='utf-8')

    def row(v113, v52):
        if not v52:
            return f'| {v113} | n/a | n/a | n/a | n/a |\n'
        return f"| {v113} | {v52['top1']:.3f} | {v52['median_rank']:.1f} | {v52['acc_20way']:.3f} | {v52['tie_at_zero_frac']:.3f} |\n"
    v2.v165(f'# Stage 267 read-refine\n\n**{v206}** · model={v73.v207} · bank={v39} · eval={v156(v46)} · trained params **0**\n\n| arm | top1 | median | 20-way | silence |\n|---|---:|---:|---:|---:|\n' + v291('A hop0 surface', v81) + v291('B refine grounded', v93) + v291('C refine RANDOM passages', v94) + v291('D refine selective', v95) + v291('E refine blind', v96) + f"\n- woken (grounded): **{v97['woken_n']}** queries, top1 among them {v97['woken_top1']:.3f}; random control woke {v98['woken_n']}\n- copy rate from shown passages: **{v100:.3f}**\n- hop0 uncertain: {v83}/{v156(v46)}\n\n## Gates\n\n" + ''.v272((f'- {v71}: **{v252}**\n' for v71, v252 in v28['gates'].v46())), encoding='utf-8')
    v172(v264.v250({'overall': v206, 'gates': v28['gates']}, indent=2))
    return 0
if v109 == '__main__':
    raise v210(v251())