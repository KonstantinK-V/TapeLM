"""
Stage 266 — Is 261's NO a trunk-capacity wall or an interface wall?

Same open-bank NL exam as 261. Only the frozen trunk changes, via ExternalTrunk (262).
W_q stays in fp space (256); SemQuery.in_dim is read off each trunk. Keys never see the
external tokenizer.

Matched-size control is the point: Qwen2.5-0.5B base vs Qwen2.5-0.5B-Instruct — same
parameter count, only the tuning differs. Instruct without chat template is an invalid
run (bare text ≈ base). Ladder then steps size (1.5B / 3B Instruct) to separate
"ability appears" from "size pulls".

Fourth arm (zero train, strongest thesis check): Instruct emits keywords in words;
those feed 261f-style word votes. "Mind formulates the query, tape answers."

Gates:
  G_instruct_beats_base_matched — 0.5B-Instruct vs 0.5B base on a live semantic channel
                                  (if Instruct alpha≈0, compare fp_only — not fp+sem≡fp)
  G_ladder_monotone             — 20-way nondecreasing along the size ladder; null if
                                  Instruct alphas collapsed (trunk out of the loop)
  G_prompted_query              — keyword→votes beat trained W_sem (matched Instruct)
  G_prompted_beats_surface      — Instruct keywords beat raw question words on headlines
                                  (top1 / 20-way only; median is G_prompted_median_better)
  G_union_beats_surface         — surface ∪ keywords beats surface on headlines
  G_mind_refines                — union wins headlines + preserves coverage vs surface

Verdict remaps 261:
  NO_AT_TRUNK_SCALE       — matched Instruct does not beat base → capacity at this scale
  NL_QUERY_NO            — honest interface NO (Instruct helps or prompted wins, open still fails)
  INSTRUCT_TRUNK_OK      — Instruct beats base and open-domain signal moves
  PROMPTED_QUERY_SIGNAL  — keywords→votes beat W_sem; interface still broken
  WORDS_FORMULATE_QUERY  — words crush learned query vector; remap QUERY_MUST_BE_WORDS
  MIND_REFINES_QUERY     — union (surface ∪ keywords) beats surface on headlines; mind helps

  python _stage266_instruct_trunk.py [--smoke]
  python _stage266_instruct_trunk.py --include-3b
  python _stage266_instruct_trunk.py --prompted-only   # cheap; merges into prior decision
  python _stage266_instruct_trunk.py --smoke --only qwen05_base --verify-seed
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
from _stage261_nl_query import SemQuery, WORD_RE, collect, ctx_words, fp_raw, jaccard
from _stage262_trunk_swap import ExternalTrunk
from _tape_index import context_words, nway_strict, vote_arm_fields, vote_rank
v0 = v15('results')
v1 = v0 / 'stage266_decision.json'
v2 = v0 / 'stage266_mini.md'
v3 = v0 / '_stage266_log.txt'
v4 = v15('checkpoints/stage191_p1_curve.pt')
v5 = v15('data/_wikitext103_train.txt')
v6 = 261
v7 = ({'id': 'qwen05_base', 'model': 'Qwen/Qwen2.5-0.5B', 'instruct': False, 'rung': 0}, {'id': 'qwen05_instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'instruct': True, 'rung': 1}, {'id': 'qwen15_instruct', 'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'instruct': True, 'rung': 2}, {'id': 'qwen3_instruct', 'model': 'Qwen/Qwen2.5-3B-Instruct', 'instruct': True, 'rung': 3})
v8 = 'Extract 3-8 content keywords (nouns and proper names) from the text below. Reply with comma-separated words only — no sentences.\n\n{text}'
v9 = 'The text below is a question fragment. List 3-8 English content words that do NOT appear in the text, but that a related Wikipedia article on the same topic would likely contain (synonyms, related people, places, technical terms). Comma-separated words only — no sentences.\n\n{text}'

def log(v16: v10) -> None:
    v17 = v16 if v16.v290('\n') else v16 + '\n'
    try:
        v291(v17, end='', flush=True)
    except v161:
        v291(v17.v429('ascii', 'replace').v177('ascii'), end='', flush=True)
    v3.v292.v162(parents=True, exist_ok=True)
    with v3.v293('a', encoding='utf-8') as v163:
        v163.v294(v17)

def ensure_short_hf_home() -> v10 | None:
    """Windows + long HF hub paths → OSError Errno 22. Prefer short HF_HOME (e.g. C:\\hf)."""
    if v295.v164 != 'nt':
        return None
    v18 = v295.v166.v199('HF_HOME') or v295.v166.v199('HUGGINGFACE_HUB_CACHE')
    if v18 and v193(v15(v18).v430().v402()) < 40:
        return v18
    v19 = v15(v295.v166.v199('SOTE_HF_HOME', 'C:\\hf'))
    try:
        v19.v162(parents=True, exist_ok=True)
    except v165:
        v19 = v15.v403() / 'hf'
        v19.v162(parents=True, exist_ok=True)
    v295.v166['HF_HOME'] = v10(v19)
    v295.v166.v167('HUGGINGFACE_HUB_CACHE', v10(v19 / 'hub'))
    v295.v166.v167('TRANSFORMERS_CACHE', v10(v19 / 'transformers'))
    return v10(v19)

def free_cuda() -> None:
    v296.v168()
    if v173.v297.v169():
        v173.v297.v298()

def chat_wrap(v20, v21: v10) -> v10:
    """Instruct models without this are base models wearing an Instruct name."""
    v22 = [{'role': 'user', 'content': v21}]
    return v20.v170(v22, tokenize=False, add_generation_prompt=True)

@v173.v26()
def trunk_state(v23: v171, v21: v10, *, v24: v11) -> v173.v27 | None:
    v25 = v174(v23.v20, v21) if v24 else v21
    return v23.v172(v25)

@v173.v26()
def generate_word_list(v23: v171, v28: v10, *, v29: v14=48) -> v38[v10]:
    v30 = v174(v23.v20, v28)
    v31 = v23.v20(v30, return_tensors='pt', truncation=True, max_length=512)
    v31 = {v175: v371.v253(v23.v86) for v175, v371 in v31.v78()}
    v32 = v14(v31['input_ids'].v299[1])
    v33 = v23.v300.v176(**v31, max_new_tokens=v29, do_sample=False, pad_token_id=v23.v20.v301)
    v34 = v23.v20.v177(v33[0][v32:], skip_special_tokens=True)
    v35 = []
    v36 = v178()
    for v37 in v302.v179('[,;\\n|/]+', v34):
        for v180 in v372.v303(v37):
            v304 = v180.v306()
            if v304 in v36:
                continue
            v36.v315(v304)
            v35.v313(v304)
    return v35

@v173.v26()
def extract_keywords(v23: v171, v21: v10, *, v29: v14=48) -> v38[v10]:
    return v181(v23, v8.v305(text=v21), max_new=v29)

@v173.v26()
def extract_paraphrase(v23: v171, v21: v10, *, v29: v14=48) -> v38[v10]:
    """Words absent from the question — bridge candidates for tape postings."""
    v39 = {v180.v306() for v180 in v372.v303(v21)}
    v34 = v181(v23, v9.v305(text=v21), max_new=v29)
    return [v180 for v180 in v34 if v180 not in v39]

def silence_beats(v40: v12, v41: v12, *, v42: v182=0.05) -> v11:
    """Primary mind-chance metric: reduce tie_at_zero (overall or low-overlap)."""
    v43 = v40.v199('silence') or {}
    v44 = v41.v199('silence') or {}
    return v11(v182(v40.v199('tie_at_zero_frac', 1.0)) <= v182(v41.v199('tie_at_zero_frac', 0.0)) - v42 or v182(v43.v199('tie_at_zero_frac_low_overlap', 1.0)) <= v182(v44.v199('tie_at_zero_frac_low_overlap', 0.0)) - v42)

def paraphrase_bridge_diag(v45, v46: v12, v47: v12) -> v12:
    """How often novel words actually exist on the tape index."""
    v48 = v49 = v50 = 0
    for v51 in v45:
        v183 = v46.v199(v51['slot'], [])
        v184 = [v180 for v180 in v183 if v180 in v47]
        v48 += v193(v183)
        v49 += v193(v184)
        if v184:
            v50 += 1
    v52 = v185(1, v193(v45))
    return {'n_queries': v193(v45), 'n_novel_words': v48, 'n_novel_on_tape': v49, 'novel_on_tape_frac': v182(v49 / v185(1, v48)), 'queries_with_bridge_word_frac': v182(v50 / v52)}

def build_exam(v53, v54, v55, v56, v57):
    """Keys + natural (write, ask) pairs — same seed/logic as 261.

    Harvest = remaining multi-mention entities (≥2 sentences), same recipe as the exam:
    write from mention A, ask from mention B. NOT same-sentence prefixes of noise slots
    (that would teach a different invariance than the open NL exam).
    """
    v58 = v168(v54, v53)
    v59 = v307(v58)[:v55]
    v57.v186(v59)
    v187, v188, v78 = ([], [], [])
    v60: v38[v10] = []

    def try_pair(v61, v189):
        v213, v308 = (v189[0], v189[1])
        v190 = v213['line'][v185(0, v213['start'] - 140):v404(v193(v213['line']), v213['end'] + 140)]
        v175 = v53.v309(v190, exclude=v61)
        if v175 is None:
            return None
        v191 = v308['line'][v185(0, v308['start'] - 200):v308['start']].v310()
        if v193(v372.v303(v191)) < 4:
            return None
        v34 = v311(v53, v191, use_anchor=True)
        if v34 is None:
            return None
        return {'key': v377.v329(v53.v431([v213['anchor']])[0] + v175, dim=-1), 'wctx': v190, 'qtext': v191, 'raw': v34, 'qwords': v320(v191, exclude=v61), 'overlap': v373(v405(v190, v61), v405(v191, v61))}
    for v61 in v59:
        v192 = v312(v61, v58[v61])
        if v192 is None:
            continue
        v187.v313(v192['key'])
        v60.v313(v192['wctx'])
        v78.v313({'ent': v61, 'slot': v193(v188), 'qtext': v192['qtext'], 'raw': v192['raw'], 'wctx': v192['wctx'], 'qwords': v192['qwords'], 'overlap': v192['overlap']})
        v188.v313(v61)
    v62 = v193(v187)
    v63 = {v51['ent'] for v51 in v78}
    v64: v38[v12] = []
    v65 = [v61 for v61 in v307(v58) if v61 not in v63]
    v66 = v314.v194(v6 + 11)
    v66.v186(v65)
    for v61 in v65:
        if v193(v64) >= v56:
            break
        v192 = v312(v61, v58[v61])
        if v192 is None:
            continue
        v195 = v193(v188)
        v187.v313(v192['key'])
        v60.v313(v192['wctx'])
        v188.v313(v61)
        v63.v315(v61)
        v64.v313({'ent': v61, 'slot': v195, 'qtext': v192['qtext'], 'raw': v192['raw'], 'wctx': v192['wctx'], 'qwords': v192['qwords'], 'overlap': v192['overlap'], 'harvest': True})
    v67 = v62 + v56
    for v68 in v54:
        if v193(v187) >= v67:
            break
        for v16 in v374.v316(v68):
            v61 = v16.v375(1)
            if v193(v61) < 5 or v61 in v63:
                continue
            v336, v337 = (v185(0, v16.v432() - 140), v404(v193(v68), v16.v433() + 140))
            v317 = v53.v309(v68[v336:v337], exclude=v61)
            if v317 is None:
                continue
            v318 = [v180 for v180 in v434.v303(v68[v336:v16.v432()]) if v180 != v61]
            if not v318:
                continue
            v187.v313(v377.v329(v53.v431([v318[-1]])[0] + v317, dim=-1))
            v60.v313(v68[v336:v337])
            v188.v313(v61)
            v63.v315(v61)
            if v193(v187) >= v67:
                break
    v47: v12[v10, v38[v14]] = v196(v38)
    for v197, (v61, v190) in v198(v319(v188, v60)):
        for v180 in v320(v190, exclude=v61):
            v47[v180].v313(v197)
    v69 = {v180: 1.0 / v406.v208(2.0 + v193(v47[v180])) for v180 in v47}
    return (v187, v188, v78, v62, v47, v69, v64)
v13 = {'qwen05_base': {'eval_top1_sem': 0.0, 'eval_20way_sem': 0.074, 'eval_20way_fp': 0.153, 'fit_20way_fp': 0.301, 'fit_top1_sem': 1.0}, 'qwen05_instruct': {'eval_top1_sem': 0.0, 'eval_20way_sem': 0.119, 'eval_20way_fp': 0.21, 'fit_20way_fp': 0.443, 'fit_top1_sem': 1.0}, 'qwen15_instruct': {'eval_top1_sem': 0.0, 'eval_20way_sem': 0.085, 'eval_20way_fp': 0.188, 'fit_20way_fp': 0.528, 'fit_top1_sem': 1.0}}

def harvest_improved(v70: v12, v71: v10) -> v11:
    """Relative to PRE_HARVEST. Requires eval-side lift — fit drop alone is undertraining."""
    v72 = v13.v199(v71)
    if not v72 or 'top1_sem' not in v70:
        return False
    v73 = v182(v70.v199('top1_sem', 0.0))
    v74 = v182(v70.v199('acc_20way_sem', 0.0))
    v75 = v182(v70.v199('acc_20way_fp', 0.0))
    v76 = v182(v70.v199('fit_acc_20way_fp', v72['fit_20way_fp']))
    v77 = v73 >= 0.02 or v74 >= 1.5 * v72['eval_20way_sem'] or v75 >= v72['eval_20way_fp'] + 0.05
    if not v77:
        return False
    return True

def still_collapsed(v70: v12) -> v11:
    """Absolute memorize signature: perfect fit, dead open top1, weak 20-way."""
    return v11(v182(v70.v199('fit_top1_sem', 0)) >= 0.9 and v182(v70.v199('top1_sem', 1)) <= 0.05 and (v182(v70.v199('acc_20way_sem', 1)) < 0.2))

def fill_h(v78, v23: v171, *, v24: v11) -> v14:
    v79 = 0
    for v51 in v78:
        v200 = v51.v321('h', None)
        del v200
        v201 = v322(v23, v51['qtext'], instruct=v24)
        v51['h'] = None if v201 is None else v201.v420().v376()
        if v51['h'] is not None:
            v79 += 1
    return v79

def clear_h(v78) -> None:
    for v51 in v78:
        v51.v321('h', None)

def train_and_score(v80, v81, v82, v83, v84, v85, v86, v87: v182, v88=20):
    """Train W_q+SemQuery on bank-wide harvest (+ fit exam); score fit vs eval.

    256 lesson: fitting adapters on a handful of exam pairs memorizes slots. Train pool
    must be harvested (prefix→slot) pairs from the bank — noise + fit, never eval.
    """
    v173.v202(v6)
    if v86.v203 == 'cuda':
        v173.v297.v323(v6)
    v80 = [v51 for v51 in v80 if v51.v199('h') is not None]
    v81 = [v51 for v51 in v81 if v51.v199('h') is not None]
    v82 = [v51 for v51 in v82 if v51.v199('h') is not None]
    if v193(v80) < 16 or v193(v81) < 8 or v193(v82) < 8:
        return None
    v89 = v14(v80[0]['h'].v324())
    v90 = v204(v89, v86)
    v91 = v325.v205(v86)
    v92 = v173.v421([v51['raw'] for v51 in v80]).v253(v86).v182()
    v93 = v173.v421([v51['h'].v253(v86) for v51 in v80]).v253(v86).v182()
    v94 = v173.v206([v51['slot'] for v51 in v80], device=v86, dtype=v173.v326)
    v95 = v92.v207(0)
    v208(f'    train_pool={v95} (harvest+fit) fit_diag={v193(v81)} eval={v193(v82)}')
    v96 = v173.v327.v209(v38(v90.v255()) + v38(v91.v255()), lr=0.002, weight_decay=0.01)
    for v97 in v210(1, v84 + 1):
        v211 = v173.v328(0, v95, (v404(32, v95),), device=v86)
        v212 = v377.v329(v91(v92[v211]), dim=-1)
        v213 = v90.v213(v93[v211], v212, v83, v85).v330(-1, 1)
        v214 = v377.v329((1 - v213) * v212 + v213 * v90.v214(v93[v211]), dim=-1)
        v215 = v377.v331(v214 @ v83.v422() / v85, v94[v211])
        v96.v332(set_to_none=True)
        v215.v333()
        v173.v407.v378.v334(v38(v90.v255()) + v38(v91.v255()), 1.0)
        v96.v97()
        if v97 == 40 or v97 % v185(1, v84 // 5) == 0:
            v208(f'    step {v97}/{v84} loss={v182(v215):.3f} a={v182(v213.v383()):.3f}')
    v90.v216()

    @v173.v26()
    def score(v217, v218, v219=v83):
        v103 = v314.v194(v6 + 5)
        v227, v335, v336, v337, v228 = ([], [], [], [], [])
        for v51 in v217:
            v212 = v377.v329(v91(v51['raw'].v435(0)), dim=-1)[0]
            v201 = v51['h'].v253(v86)
            if v218:
                v379 = v182(v90.v213(v201, v212, v219, v85).v330(-1)[0])
                v335.v313(v379)
                v380 = v377.v329(v90.v214(v201.v435(0)), dim=-1).v330(-1)
                v214 = v377.v329((1 - v379) * v212 + v379 * v380, dim=-1)
                v381 = v219 @ v214
            else:
                v381 = v219 @ v212
            v70 = 1 + v14((v381 > v381[v51['slot']]).v411())
            v227.v313(v70)
            (v337 if v51['overlap'] > v87 else v336).v313(v14(v70 == 1))
            v232 = [v382 for v382 in v103.v424(v210(v219.v207(0)), v404(v88 * 3, v219.v207(0))) if v382 != v51['slot']][:v88 - 1]
            v228.v313(v14(v361((v182(v381[v51['slot']]) > v182(v381[v382]) for v382 in v232))))
        v70 = v341.v234(v227, dtype=v341.v342)
        return {'top1': v182(v341.v383(v70 == 1)), 'mrr': v182(v341.v383(1.0 / v70)), 'median_rank': v182(v341.v346(v70)), 'top1_low_overlap': v182(v341.v383(v336)) if v336 else v182('nan'), 'top1_high_overlap': v182(v341.v383(v337)) if v337 else v182('nan'), 'alpha': v182(v341.v383(v335)) if v335 else 0.0, 'n': v193(v227), f'acc_{v88}way': v182(v341.v383(v228)), f'chance_{v88}way': 1.0 / v88}
    v220, v221 = (v225(v82, False), v225(v82, True))
    v222, v223 = (v225(v81, False), v225(v81, True))
    v98 = v173.v224(v83.v207(0), generator=v173.v423().v202(v6 + 1))
    v99 = v225(v82, True, Kmat=v83[v98.v253(v83.v86)])
    v100 = v226({'fit_top1_sem': v223['top1'], 'top1_sem': v221['top1'], 'acc_20way_sem': v221['acc_20way']})
    v208(f"    FIT  fp 20-way={v222['acc_20way']:.3f} sem 20-way={v223['acc_20way']:.3f} top1_sem={v223['top1']:.3f} a={v223['alpha']:.3f}")
    v208(f"    EVAL fp 20-way={v220['acc_20way']:.3f} sem 20-way={v221['acc_20way']:.3f} top1_sem={v221['top1']:.3f} a={v221['alpha']:.3f}  mixer_overfit={v100}")
    return {'h_dim': v89, 'n_fit': v193(v81), 'n_eval': v193(v82), 'n_train_pool': v95, 'overlap_median': v87, 'fp_only': v220, 'fp_plus_sem': v221, 'shuffled_keys': v99, 'fit_fp_only': v222, 'fit_fp_plus_sem': v223, 'mixer_overfit': v100, 'acc_20way_fp': v220['acc_20way'], 'acc_20way_sem': v221['acc_20way'], 'top1_sem': v221['top1'], 'alpha': v221['alpha'], 'fit_top1_sem': v223['top1'], 'fit_acc_20way_sem': v223['acc_20way'], 'fit_acc_20way_fp': v222['acc_20way']}

def score_votes(v45, v47, v69, v101, v102, v88=20, v87=0.0):
    v103 = v314.v194(v6 + 5)
    v227, v228, v229 = ([], [], [])
    for v51 in v45:
        v230 = v101(v51)
        v231: v12[v14, v182] = v196(v182)
        for v180 in v230:
            for v197 in v47.v199(v180, ()):
                v231[v197] += v69.v199(v180, 0.0)
        v338, v339 = v340(v231, v51['slot'], v102)
        v227.v313(v339)
        v232 = [v382 for v382 in v103.v424(v210(v102), v404(v88 * 3, v102)) if v382 != v51['slot']][:v88 - 1]
        v228.v313(v14(v408(v338, (v231.v199(v382, 0.0) for v382 in v232))))
        v229.v313({'gold_score': v182(v338), 'rank': v339, 'low_overlap': v51['overlap'] <= v87})
    v104 = v233(v229)
    v70 = v341.v234(v227, dtype=v341.v342)
    return {'top1': v182(v341.v383(v70 == 1)), 'mrr': v182(v341.v383(1.0 / v70)), 'median_rank': v182(v341.v346(v70)), 'top1_low_overlap': v104['top1_low_overlap'], 'top1_high_overlap': v104['top1_high_overlap'], 'tie_at_zero_frac': v104['tie_at_zero_frac'], 'silence': v104, 'n': v193(v227), f'acc_{v88}way': v182(v341.v383(v228)), f'chance_{v88}way': 1.0 / v88}

def headline_beats(v40: v12, v41: v12) -> v11:
    """Headline only: top1 or 20-way. Median rank is a separate diagnostic."""
    return v11(v40['top1'] >= v41['top1'] + 0.03 or v40['acc_20way'] >= v41['acc_20way'] + 0.05)

def median_rank_better(v40: v12, v41: v12, *, v42: v182=10.0) -> v11:
    return v11(v40['median_rank'] <= v41['median_rank'] - v42)

def main() -> v14:
    v105 = v343.v235()
    v105.v236('--smoke', action='store_true')
    v105.v236('--steps', type=v14, default=0)
    v105.v236('--entities', type=v14, default=0)
    v105.v236('--distractor-slots', type=v14, default=0)
    v105.v236('--tau', type=v182, default=0.05)
    v105.v236('--skip-3b', dest='skip_3b', action='store_true', help='skip Qwen2.5-3B-Instruct (default)')
    v105.v236('--include-3b', dest='skip_3b', action='store_false', help='try 3B Instruct; OOM → skip that rung')
    v105.v237(skip_3b=True)
    v105.v236('--prompt-model', type=v10, default='', help='HF id for keyword arm; default = matched 0.5B-Instruct')
    v105.v236('--verify-seed', action='store_true', help='train base twice on same h; confirm identical 20-way')
    v105.v236('--no-prompted', action='store_true', help='skip keyword→votes / paraphrase arms')
    v105.v236('--prompted-only', action='store_true', help='skip trunk ladder; run keyword→votes (+ union + paraphrase) and merge prior decision')
    v105.v236('--paraphrase-only', action='store_true', help='skip trunk ladder + keywords; run paraphrase silence arm and merge prior')
    v105.v236('--only', type=v10, default='', help='comma rung ids to run (e.g. qwen05_base); empty = full ladder')
    v106 = v105.v238()
    v107 = v11(v106.v107)
    if v106.v239 and v106.v240:
        v208('  --prompted-only and --no-prompted are mutually exclusive')
        return 1
    if v106.v241 and v106.v240:
        v208('  --paraphrase-only and --no-prompted are mutually exclusive')
        return 1
    v108 = v11(v106.v239 or v106.v241)
    v3.v242('', encoding='utf-8')
    v109 = v243()
    v86 = v173.v86('cuda' if v173.v297.v169() else 'cpu')
    v57 = v314.v194(v6)
    v173.v202(v6)
    v110 = v244.v244()
    v84 = v106.v84 or (150 if v106.v247 else 800)
    v55 = v106.v245 or (60 if v106.v247 else 400)
    v56 = v106.v246 or (400 if v106.v247 else 4000)
    v111 = 3000 if v106.v247 else 25000
    v208(f'Stage266 instruct trunk start {v427.v418(v428.v419).v368()} device={v86} steps={v84} smoke={v106.v247} skip_3b={v107}' + (f' HF_HOME={v109}' if v109 else ''))
    v248, v248, v249, v250 = v251()
    v112 = v409.v384(v10(v425.v410)).v252()
    v113 = v385(v250, v112).v253(v86)
    v113.v254(v173.v386(v4, map_location=v86, weights_only=False)['model'])
    v113.v216()
    for v114 in v113.v255():
        v114.v344(False)
    v53 = v256(v113, v249, v86)
    with v5.v293('r', encoding='utf-8', errors='ignore') as v163:
        v257 = v163.v345(3000000 if v106.v247 else 20000000)
    v54 = [v387.v310() for v387 in v257.v179('\n') if 80 <= v193(v387.v310()) <= 400][:v111]
    v187, v188, v78, v62, v47, v69, v64 = v258(v53, v54, v55, v56, v57)
    if v193(v78) < 16:
        v208('  not enough exam pairs')
        return 1
    v83 = v173.v421(v187).v253(v86).v182()
    v87 = v182(v341.v346([v51['overlap'] for v51 in v78]))
    v115 = v193(v64)
    v116 = v193(v188) - v62 - v115
    v208(f'  exam={v62} bank={v193(v188)} harvest_cross={v115} pure_noise={v116} postings={v411((v193(v371) for v371 in v47.v437()))} overlap_med={v87:.3f}')
    v117 = v38(v210(v193(v78)))
    v314.v194(v6).v186(v117)
    v118 = v193(v117) // 2
    v119 = [v78[v347] for v347 in v117[:v118]]
    v120 = [v78[v347] for v347 in v117[v118:]]
    v80 = v38(v119) + v38(v64)
    v208(f'  train_pool={v193(v80)} (fit_exam={v193(v119)} + harvest_cross={v193(v64)}) eval_exam={v193(v120)}')
    v121 = v259(v120, v47, v69, lambda v51: v51['qwords'], v193(v188), med=v87)
    v208(f'  surface votes (eval n={v193(v120)}): {v401.v369(v121)}')
    v122 = [v70 for v70 in v7 if not (v107 and v70['id'] == 'qwen3_instruct')]
    if v106.v239 or v106.v241:
        v122 = []
        v208(f"  --{('paraphrase' if v106.v241 else 'prompted')}-only: skipping trunk ladder")
    elif v106.v260:
        v348 = {v412.v310() for v412 in v106.v260.v179(',') if v412.v310()}
        v122 = [v70 for v70 in v122 if v70['id'] in v348]
        if not v122:
            v208(f'  --only {v348} matched no rungs')
            return 1
    v123: v12[v10, v12] = {}
    v124: v12 | None = None
    if v108 and v1.v286():
        try:
            v124 = v401.v388(v1.v413(encoding='utf-8'))
            for v71, v389 in (v124.v199('ladder') or {}).v78():
                if v362(v389, v12):
                    v123[v71] = v389
            v208(f"  merged prior ladder from {v1} ({v411((1 for v70 in v123.v437() if 'acc_20way_sem' in v70))} scored trunks)")
        except v349 as e:
            v208(f'  prior decision load fail: {v61}')
            v124 = None
    for v125 in v122:
        v208(f"\n== trunk {v125['id']} model={v125['model']} instruct={v125['instruct']} ==")
        v350()
        try:
            v23 = v171(v125['model'], v86)
        except v349 as e:
            v208(f'  LOAD FAIL: {v203(v61).v160}: {v61}')
            v123[v125['id']] = {'error': f'{v203(v61).v160}: {v61}', **v125}
            continue
        if v125['instruct'] and (not v414(v23.v20, 'chat_template', None)):
            v208('  FATAL: Instruct model has no chat_template — refuse to run bare text')
            v123[v125['id']] = {'error': 'missing_chat_template', **v125}
            del v23
            v350()
            continue
        v208(f"  hidden={v23.v391} chat_template={('yes' if v125['instruct'] else 'n/a (base)')}")
        v351(v78)
        v351(v64)
        v350()
        v79 = v352(v78, v23, instruct=v125['instruct'])
        v261 = v352(v64, v23, instruct=v125['instruct'])
        v208(f'  h filled exam {v79}/{v193(v78)} harvest {v261}/{v193(v64)}')
        try:
            v353 = v390(v80, v119, v120, v83, v84, v106.v85, v86, v87)
        except v173.v297.v354 as e:
            v208(f'  OOM during train: {v61}')
            v123[v125['id']] = {'error': 'OOM_train', **v125, 'hidden': v23.v391}
            del v23
            v351(v78)
            v351(v64)
            v350()
            continue
        if v353 is None:
            v123[v125['id']] = {'error': 'too_few_h', **v125}
        else:
            v123[v125['id']] = {**{v175: v125[v175] for v175 in ('id', 'model', 'instruct', 'rung')}, 'hidden': v23.v391, 'chat_template_used': v11(v125['instruct']), **v353}
            v208(f"  fp 20-way={v353['acc_20way_fp']:.3f} sem 20-way={v353['acc_20way_sem']:.3f} top1_sem={v353['top1_sem']:.3f} a={v353['alpha']:.3f} FIT top1_sem={v353['fit_top1_sem']:.3f} pool={v353['n_train_pool']}")
            if v106.v355 and v125['id'] == 'qwen05_base':
                v392 = v390(v80, v119, v120, v83, v84, v106.v85, v86, v87)
                v393 = v392 is not None and v400(v392['acc_20way_sem'] - v353['acc_20way_sem']) < 1e-12 and (v400(v392['acc_20way_fp'] - v353['acc_20way_fp']) < 1e-12) and (v400(v392['top1_sem'] - v353['top1_sem']) < 1e-12)
                v123[v125['id']]['verify_seed'] = {'second_acc_20way_sem': None if v392 is None else v392['acc_20way_sem'], 'second_acc_20way_fp': None if v392 is None else v392['acc_20way_fp'], 'identical': v393}
                v394 = 'None' if v392 is None else f"{v392['acc_20way_sem']:.6f}"
                v208(f"  verify_seed identical={v393} sem {v353['acc_20way_sem']:.6f} vs {v394}")
        del v23
        v351(v78)
        v351(v64)
        v350()
    v126 = None
    v127 = None
    v128 = None
    v129 = None
    v130 = None
    v131: v171 | None = None
    v132: v12 | None = None
    v133 = v11(v106.v355 or v106.v240 or (v106.v260 and (not v106.v239) and (not v106.v241) and ('qwen05_instruct' not in v106.v260) and ('prompt' not in v106.v260)))
    v134 = not v133 and (not v106.v241)
    v135 = not v133
    if v133:
        v208('\n== word arms: skipped (verify-seed / --only / --no-prompted) ==')
    else:
        v262 = v106.v356 or 'Qwen/Qwen2.5-0.5B-Instruct'
        v208(f'\n== word arms: loading {v262} ==')
        try:
            v350()
            v131 = v171(v262, v86)
            v132 = {'id': 'prompt', 'model': v262, 'instruct': True}
            if not v414(v131.v20, 'chat_template', None):
                v208('  FATAL: prompt model missing chat_template')
                v126 = {'error': 'missing_chat_template'}
                del v131
                v131 = None
        except v349 as e:
            v208(f'  prompt model load fail: {v61}')
            v131 = None
    if v106.v241 and v124:
        if v362(v124.v199('prompted_query'), v12) and 'acc_20way' in v124['prompted_query']:
            v126 = v124['prompted_query']
        if v362(v124.v199('union_votes'), v12) and 'acc_20way' in v124['union_votes']:
            v127 = v124['union_votes']
    v136 = False
    if v134 and v131 is not None and (v132 is not None) and (v126 is None or 'acc_20way' not in (v126 or {})):
        v208(f"\n== prompted keywords → votes via {v132.v199('model')} ==")
        v263: v38[v38[v10]] = []
        v264 = 0
        for v51 in v120:
            try:
                v395 = v415(v131, v51['qtext'])
            except v173.v297.v354:
                v208('  OOM on keyword generate — abort keyword arm')
                v126 = {'error': 'OOM_generate'}
                v136 = True
                break
            if not v395:
                v264 += 1
                v395 = v38(v51['qwords'])
            v263.v313(v395)
        if not v136:
            v357 = {v120[v347]['slot']: v263[v347] for v347 in v210(v193(v120))}
            v126 = v259(v120, v47, v69, lambda v51: v357[v51['slot']], v193(v188), med=v87)
            v126.v396({'model': v132.v199('model'), 'trained_parameters': 0, 'empty_keyword_fallback_n': v264, 'examples': [{'qtext': v120[v347]['qtext'][:120], 'keywords': v263[v347][:8]} for v347 in v210(v404(5, v193(v120)))]})
            v208(f"  prompted votes: {v401.v369({v175: v371 for v175, v371 in v126.v78() if v175 != 'examples'})}")
            v358 = {v120[v347]['slot']: v38(v12.v426(v38(v120[v347]['qwords']) + v38(v263[v347]))) for v347 in v210(v193(v120))}
            v127 = v259(v120, v47, v69, lambda v51: v358[v51['slot']], v193(v188), med=v87)
            v127['trained_parameters'] = 0
            v127['kind'] = 'surface_union_keywords'
            v208(f'  union votes: {v401.v369(v127)}')
    if v135 and v131 is not None and (v132 is not None) and (not v136):
        v208(f"\n== paraphrase (novel words) → votes via {v132.v199('model')} ==")
        v265: v38[v38[v10]] = []
        v266 = 0
        for v51 in v120:
            try:
                v397 = v416(v131, v51['qtext'])
            except v173.v297.v354:
                v208('  OOM on paraphrase generate — abort paraphrase arm')
                v128 = {'error': 'OOM_generate'}
                v136 = True
                break
            if not v397:
                v266 += 1
            v265.v313(v397)
        if not v136:
            v46 = {v120[v347]['slot']: v265[v347] for v347 in v210(v193(v120))}
            v130 = v398(v120, v46, v47)
            v128 = v259(v120, v47, v69, lambda v51: v46[v51['slot']], v193(v188), med=v87)
            v128.v396({'model': v132.v199('model'), 'trained_parameters': 0, 'kind': 'paraphrase_novel', 'empty_paraphrase_n': v266, 'bridge': v130, 'examples': [{'qtext': v120[v347]['qtext'][:120], 'paraphrase': v265[v347][:8], 'on_tape': [v180 for v180 in v265[v347][:8] if v180 in v47]} for v347 in v210(v404(5, v193(v120)))]})
            v208(f"  paraphrase votes: {v401.v369({v175: v371 for v175, v371 in v128.v78() if v175 != 'examples'})}")
            v359 = {v120[v347]['slot']: v38(v12.v426(v38(v120[v347]['qwords']) + v38(v265[v347]))) for v347 in v210(v193(v120))}
            v129 = v259(v120, v47, v69, lambda v51: v359[v51['slot']], v193(v188), med=v87)
            v129['trained_parameters'] = 0
            v129['kind'] = 'surface_union_paraphrase'
            v129['bridge'] = v130
            v360 = 0
            for v51 in v120:

                def _gold(v35, v417=v51['slot']):
                    v231 = 0.0
                    for v180 in v35:
                        if v417 in v47.v199(v180, ()):
                            v231 += v69.v199(v180, 0.0)
                    return v231
                if v436(v51['qwords']) <= 0.0 and v436(v359[v51['slot']]) > 0.0:
                    v360 += 1
            v129['surface_silent_woken_frac'] = v182(v360 / v185(1, v193(v120)))
            v129['surface_silent_woken_n'] = v360
            v208('  paraphrase∪surface: ' + v401.v369({v175: v371 for v175, v371 in v129.v78() if v175 != 'examples'}))
    if v131 is not None:
        del v131
        v131 = None
        v350()
    v137 = v133
    v138 = v123.v199('qwen05_base')
    v139 = v123.v199('qwen05_instruct')
    v140 = v138 and 'acc_20way_sem' in v138
    v141 = v139 and 'acc_20way_sem' in v139

    def _alpha_off(v70: v12 | None, v267: v182=1e-05) -> v11:
        return v11(v70 and 'alpha' in v70 and (v400(v182(v70['alpha'])) < v267))
    if v268(v139):
        v269 = v11(v140 and v141 and (v139['acc_20way_fp'] >= v138['acc_20way_fp'] + 0.05))
        v270 = 'alpha_collapsed_compare_fp_only'
    else:
        v269 = v11(v140 and v141 and (v139['acc_20way_sem'] >= v138['acc_20way_sem'] + 0.05))
        v270 = 'compare_fp_plus_sem'
    v142 = []
    v143 = []
    for v71 in ('qwen05_instruct', 'qwen15_instruct', 'qwen3_instruct'):
        v70 = v123.v199(v71)
        if v70 and 'acc_20way_sem' in v70:
            v142.v313(v70['acc_20way_sem'])
            v143.v313(v182(v70.v199('alpha', 1.0)))
    if v193(v142) >= 2 and v361((v400(v213) < 1e-05 for v213 in v143)):
        v271 = None
        v272 = 'alphas_collapsed_trunk_out'
    else:
        v271 = v11(v193(v142) >= 2 and v361((v142[v347] + 1e-09 >= v142[v347 - 1] - 0.02 for v347 in v210(1, v193(v142)))) and (v142[-1] >= v142[0] + 0.03))
        v272 = 'scored'
    v144 = v139['acc_20way_sem'] if v141 else v182('nan')
    v145 = v362(v126, v12) and 'acc_20way' in v126
    v146 = v362(v127, v12) and 'acc_20way' in v127
    v147 = v362(v128, v12) and 'acc_20way' in v128
    v148 = v362(v129, v12) and 'acc_20way' in v129
    if v133 or not v145:
        v273 = None
        v274 = None
        v275 = None
    else:
        v276 = v182(v139.v199('top1_sem', 0.0)) if v141 else 0.0
        v273 = v11(v126['top1'] >= v276 + 0.05 or (v141 and v126['acc_20way'] >= v144 + 0.05) or v126['top1'] >= 0.12)
        v274 = v363(v126, v121)
        v275 = v364(v126, v121)
    if not v146:
        v277 = None
        v278 = None
        v279 = None
    else:
        v277 = v363(v127, v121)
        v278 = v364(v127, v121)
        v280 = v182(v127.v199('tie_at_zero_frac', 1.0))
        v281 = v182(v121.v199('tie_at_zero_frac', 0.0))
        v279 = v11(v277 and v280 <= v281 + 0.03 and (v127['top1'] >= v126['top1'] - 1e-09 if v145 else True))
    if not v148:
        v282 = None
        v283 = None
    else:
        v282 = v365(v129, v121)
        v283 = v11(v130 is not None and v182(v130.v199('novel_on_tape_frac', 0.0)) >= 0.25)
    v149 = [v71 for v71 in ('qwen05_base', 'qwen05_instruct', 'qwen15_instruct', 'qwen3_instruct') if v399(v123.v199(v71) or {}, v71)]
    v150 = [v71 for v71 in ('qwen05_base', 'qwen05_instruct', 'qwen15_instruct', 'qwen3_instruct') if v226(v123.v199(v71) or {})]
    v151 = v11(v149)
    v152 = v11(v150) and (not v151)
    v153 = v11(v141 and (v121['top1'] >= v182(v139.v199('top1_sem', 0.0)) + 0.08 or (v145 and v126['top1'] >= v182(v139.v199('top1_sem', 0.0)) + 0.08)))
    if not (v140 and v141) and (not (v108 and v124 and v124.v199('ladder'))):
        v284 = 'INSTRUCT_TRUNK_INVALID'
        v285 = None
    elif v282:
        v284 = 'PARAPHRASE_BREAKS_SILENCE'
        v285 = 'QUERY_MUST_BE_WORDS'
    elif v279:
        v284 = 'MIND_REFINES_QUERY'
        v285 = 'QUERY_MUST_BE_WORDS'
    elif v153 or (v145 and v273):
        v284 = 'WORDS_FORMULATE_QUERY'
        v285 = 'QUERY_MUST_BE_WORDS'
    elif v151:
        v284 = 'HARVEST_FIXES_MIXER'
        v285 = 'MIXER_WAS_DATA'
    elif v152 or v11(v150):
        v284 = 'MIXER_OVERFIT'
        v285 = 'MIXER_DEFECT'
    elif v269 and (v271 or v273) and (v141 and v139['top1_sem'] >= 0.15 or (v145 and v126.v199('top1', 0) >= 0.15)):
        v284 = 'INSTRUCT_TRUNK_OK'
        v285 = 'NO_WAS_TRUNK_SCALE'
    elif not v269 and v140 and v141:
        v284 = 'NO_AT_TRUNK_SCALE'
        v285 = 'NO_AT_TRUNK_SCALE'
    elif v273 and (not v269):
        v284 = 'PROMPTED_QUERY_SIGNAL'
        v285 = 'NL_QUERY_NO'
    else:
        v284 = 'NL_QUERY_NO'
        v285 = 'NL_QUERY_NO'
    v154 = v11(v146 and v153 and (not v277))
    v155 = v11(v148 and (not v282))
    v156 = v0 / 'stage261_decision.json'
    v157 = {'ref': v10(v156), 'matched': None}
    if v156.v286():
        try:
            v366 = v401.v388(v156.v413(encoding='utf-8'))
            v157 = {'ref_slots': v366.v199('slots'), 'ref_exam_slots': v366.v199('exam_slots'), 'ref_n_eval': v366.v199('n_eval'), 'this_slots': v193(v188), 'this_exam_slots': v62, 'this_n_eval': v193(v120), 'slots_match': v366.v199('slots') == v193(v188), 'exam_slots_match': v366.v199('exam_slots') == v62, 'note': 'Same construction seed=261; count mismatch → trunk ladder still internally valid, but remap_261 needs a caveat.' if v366.v199('slots') != v193(v188) or v366.v199('exam_slots') != v62 else 'counts match published 261 decision'}
            v208(f'  exam_parity vs 261: {v401.v369(v157)}')
        except v349 as e:
            v157 = {'error': v10(v61)}
    v33 = {'stage': 266, 'overall': v284, 'remap_261': v285, 'smoke': v106.v247, 'seed': v6, 'train_reseed_per_trunk': True, 'steps': v84, 'slots': v193(v188), 'exam_slots': v62, 'noise_slots': v193(v188) - v62, 'harvest_noise_pairs': v193(v64), 'harvest_kind': 'cross_mention', 'harvest_capped_at_n_dist': True, 'train_pool_n': v193(v80), 'bank_matches_pre_harvest': v400(v193(v188) - 4352) <= 64, 'pre_harvest_baseline': v13, 'overlap_median': v87, 'exam_parity_261': v157, 'fp_version': v325.v367(), 'gates': {'G_instruct_beats_base_matched': v269, 'G_instruct_compare': v270, 'G_ladder_monotone': v271, 'G_ladder_note': v272, 'G_prompted_query': v273, 'G_prompted_beats_surface': v274, 'G_prompted_median_better': v275, 'G_union_beats_surface': v277, 'G_union_median_better': v278, 'G_mind_refines': v279, 'G_paraphrase_breaks_silence': v282, 'G_paraphrase_novel_on_tape': v283, 'G_paraphrase_useless': v155, 'G_words_crush_learned': v153, 'G_any_words_suffice': v154, 'G_mixer_overfit': v11(v150) and (not v151), 'G_harvest_helped': v151, 'harvest_helped_ids': v149, 'collapsed_ids': v150, 'instruct_alpha_collapsed': v268(v139), 'headline_beats_means': 'top1+0.03 or 20-way+0.05; median rank is separate', 'silence_beats_means': 'tie_at_zero −0.05 overall or low-overlap'}, 'fit_diag': {v71: {'fit_top1_sem': (v123.v199(v71) or {}).v199('fit_top1_sem'), 'eval_top1_sem': (v123.v199(v71) or {}).v199('top1_sem'), 'fit_acc_20way_sem': (v123.v199(v71) or {}).v199('fit_acc_20way_sem'), 'eval_acc_20way_sem': (v123.v199(v71) or {}).v199('acc_20way_sem'), 'eval_acc_20way_fp': (v123.v199(v71) or {}).v199('acc_20way_fp'), 'mixer_overfit': (v123.v199(v71) or {}).v199('mixer_overfit'), 'alpha': (v123.v199(v71) or {}).v199('alpha')} for v71 in ('qwen05_base', 'qwen05_instruct', 'qwen15_instruct', 'qwen3_instruct') if v123.v199(v71) and 'top1_sem' in (v123.v199(v71) or {})}, 'matched_pair': {'base': {v175: v138.v199(v175) for v175 in ('model', 'hidden', 'acc_20way_fp', 'acc_20way_sem', 'top1_sem', 'alpha', 'fit_top1_sem', 'mixer_overfit')} | {'verify_seed': v138.v199('verify_seed')} if v140 else v138, 'instruct': {v175: v139.v199(v175) for v175 in ('model', 'hidden', 'acc_20way_fp', 'acc_20way_sem', 'top1_sem', 'alpha', 'chat_template_used', 'fit_top1_sem', 'mixer_overfit')} if v141 else v139, 'delta_20way_sem': v139['acc_20way_sem'] - v138['acc_20way_sem'] if v140 and v141 else None, 'delta_20way_fp_only': v139['acc_20way_fp'] - v138['acc_20way_fp'] if v140 and v141 else None}, 'ladder': {v71: v123.v199(v71) for v71 in ('qwen05_base', 'qwen05_instruct', 'qwen15_instruct', 'qwen3_instruct')}, 'surface_votes': v121, 'prompted_query': v126, 'union_votes': v127, 'paraphrase': v128, 'paraphrase_union': v129, 'paraphrase_bridge': v130, 'secondary': {'harvest_fixes_mixer': v151, 'remap_if_harvest_only': 'MIXER_WAS_DATA', 'any_words_suffice': v154, 'paraphrase_useless': v155}, 'note': "Mind's only remaining chance is silence: novel paraphrase words that exist on tape (bridge), measured by tie_at_zero_frac — not top1. Keywords subtract noise (median better, coverage worse). Union∪keywords failed headlines → any words suffice. G_prompted_beats_surface is headline-only. Instruct alpha≈0 → fp+sem≡fp_only. QUERY_MUST_BE_WORDS when words crush trained W.", 'prompted_only_run': v11(v106.v239), 'paraphrase_only_run': v11(v106.v241), 'timestamp': v427.v418(v428.v419).v368(), 'wall_s': v244.v244() - v110}
    v1.v242(v401.v369(v33, indent=2), encoding='utf-8')

    def fmt_arm(v70):
        if not v70 or 'acc_20way_sem' not in v70:
            return v70.v199('error', 'n/a') if v362(v70, v12) else 'n/a'
        v287 = v70.v199('fit_top1_sem')
        v288 = f' / FIT top1_sem **{v287:.3f}**' if v287 is not None else ''
        return f"eval sem **{v70['acc_20way_sem']:.3f}** / fp {v70['acc_20way_fp']:.3f} / top1 {v70['top1_sem']:.3f} / a={v70['alpha']:.2f}{v288}"
    v158 = ''
    for v71 in ('qwen05_base', 'qwen05_instruct', 'qwen15_instruct', 'qwen3_instruct'):
        v70 = v123.v199(v71)
        if not v70 or 'fit_top1_sem' not in v70:
            continue
        v158 += f"| {v71} | **{v70['fit_top1_sem']:.3f}** | {v70['top1_sem']:.3f} | {v70['fit_acc_20way_sem']:.3f} | {v70['acc_20way_sem']:.3f} | {v70['acc_20way_fp']:.3f} | {v70['mixer_overfit']} |\n"
    v159 = f"# Stage 266 instruct trunk ladder\n\n**{v284}** · remap_261=`{v285}` · bank={v193(v188)} exam={v62}{(' · SMOKE' if v106.v247 else '')}\n\n## Fit vs eval (mixer diagnostic)\n\n| trunk | FIT top1_sem | EVAL top1_sem | FIT 20way_sem | EVAL 20way_sem | EVAL fp | overfit |\n|-------|-------------:|--------------:|--------------:|---------------:|--------:|:-------:|\n{v158}\n## Matched pair\n\n| trunk | metric |\n|-------|--------|\n| 0.5B base | {v439(v138)} |\n| 0.5B-Instruct + chat template | {v439(v139)} |\n| Δ sem 20-way | {v33['matched_pair']['delta_20way_sem']} |\n| Δ fp_only 20-way | {v33['matched_pair'].v199('delta_20way_fp_only')}\n\n## Ladder\n\n| id | result |\n|----|--------|\n" + ''.v438((f'| {v71} | {v439(v123.v199(v71))} |\n' for v71 in ('qwen05_base', 'qwen05_instruct', 'qwen15_instruct', 'qwen3_instruct'))) + '\n## Word-vote arms (0 train)\n\n' + '| arm | top1 | median | 20-way | silence | low-ov silence | low-ov\\|vote |\n' + '|-----|-----:|-------:|-------:|--------:|---------------:|-------------:|\n' + f"| surface words | {v121['top1']:.3f} | {v121['median_rank']:.1f} | {v121['acc_20way']:.3f} | {v121['tie_at_zero_frac']:.3f} | {v121['silence'].v199('tie_at_zero_frac_low_overlap', v182('nan')):.3f} | {v121['silence'].v199('top1_low_overlap_given_vote', v182('nan')):.3f} |\n" + (f"| prompted keywords | {v126['top1']:.3f} | {v126['median_rank']:.1f} | {v126['acc_20way']:.3f} | {v126['tie_at_zero_frac']:.3f} | {v126['silence'].v199('tie_at_zero_frac_low_overlap', v182('nan')):.3f} | {v126['silence'].v199('top1_low_overlap_given_vote', v182('nan')):.3f} |\n" if v145 else '| prompted keywords | n/a | n/a | n/a | n/a | n/a | n/a |\n') + (f"| surface ∪ keywords | {v127['top1']:.3f} | {v127['median_rank']:.1f} | {v127['acc_20way']:.3f} | {v127['tie_at_zero_frac']:.3f} | {v127['silence'].v199('tie_at_zero_frac_low_overlap', v182('nan')):.3f} | {v127['silence'].v199('top1_low_overlap_given_vote', v182('nan')):.3f} |\n" if v146 else '| surface ∪ keywords | n/a | n/a | n/a | n/a | n/a | n/a |\n') + (f"| paraphrase novel | {v128['top1']:.3f} | {v128['median_rank']:.1f} | {v128['acc_20way']:.3f} | {v128['tie_at_zero_frac']:.3f} | {v128['silence'].v199('tie_at_zero_frac_low_overlap', v182('nan')):.3f} | {v128['silence'].v199('top1_low_overlap_given_vote', v182('nan')):.3f} |\n" if v147 else '| paraphrase novel | n/a | n/a | n/a | n/a | n/a | n/a |\n') + (f"| surface ∪ paraphrase | {v129['top1']:.3f} | {v129['median_rank']:.1f} | {v129['acc_20way']:.3f} | {v129['tie_at_zero_frac']:.3f} | {v129['silence'].v199('tie_at_zero_frac_low_overlap', v182('nan')):.3f} | {v129['silence'].v199('top1_low_overlap_given_vote', v182('nan')):.3f} |\n" if v148 else '| surface ∪ paraphrase | n/a | n/a | n/a | n/a | n/a | n/a |\n') + (f"| trained W (matched Instruct) | {v139.v199('top1_sem', v182('nan')):.3f} | — | {v144:.3f} | — | — | — |\n" if v141 else '') + (f"\nParaphrase bridge: novel_on_tape={(v130 or {}).v199('novel_on_tape_frac', v182('nan')):.3f}, queries_with_bridge={(v130 or {}).v199('queries_with_bridge_word_frac', v182('nan')):.3f}, woken_silent={((v129 or {}).v199('surface_silent_woken_frac', v182('nan')) if v148 else v182('nan')):.3f}\n" if v147 or v148 else '') + f"\n## Gates\n\n- G_instruct_beats_base_matched: **{v269}** ({v270})\n- G_ladder_monotone: **{v271}** ({v272})\n- G_prompted_query: **{v273}**\n- G_prompted_beats_surface: **{v274}** (headline: top1/20-way)\n- G_prompted_median_better: **{v275}**\n- G_union_beats_surface: **{v277}**\n- G_mind_refines: **{v279}**\n- G_paraphrase_breaks_silence: **{v282}**\n- G_paraphrase_novel_on_tape: **{v283}**\n- G_paraphrase_useless: **{v155}**\n- G_words_crush_learned: **{v153}**\n- G_any_words_suffice: **{v154}**\n- G_mixer_overfit: **{v33['gates']['G_mixer_overfit']}**\n- G_harvest_helped: **{v151}**\n"
    v2.v242(v159, encoding='utf-8')
    v208(v401.v369({'overall': v284, 'remap_261': v285, 'gates': v33['gates']}, indent=2))
    v208(f'wrote {v1} wall={v244.v244() - v110:.0f}s')
    return 0
if v160 == '__main__':
    raise v289(v370())