"""
Stage 256 — Glue layer: slot-bias decoding (228c fp-decode x free-form head).

228c showed the fp path picks the right value 1.0 of the time, but only as a CONSTRAINED
choice over a 4-way candidate set. The head generates freely but never sees which slots were
retrieved (228a: HEAD_LEXICAL_PRIOR_ONLY, sensitivity ~0.036). This stage glues them:

  p'_t = (1 - g_t) * p_LM(t) + g_t * p_copy(t | tape, q_t)

  q_t      = W_q(ctx_fp(prefix))            queries move, tape KEYS stay frozen canonical
  p_copy   = span-aware distribution over the next token of top-k retrieved slot values
  g_t      = sigmoid(MLP([h_t, max_sim, mean_topk, entropy, coverage]))  "read the tape now?"

Mixing in PROBABILITY space, not as an additive logit bonus, is what makes the gate honest: an
additive bias has to out-shout logits of order ~10, and leaving the gate open costs nothing, so
it saturates at 1.0 and the tape stays decorative. Under a mixture, g_t=1 means "answer purely
from the tape", so on ordinary prose (where p_copy puts ~0 on the true next token) an open gate
is paid for directly in CE. A small L1 on g_t over prose keeps it from drifting back up.

Trunk is FROZEN. Only the glue trains: W_q + gate MLP + tau. Values live in the tape
only — the CE text has the fact sentence replaced by a placeholder, so the gradient toward the
right value can flow ONLY through the bias path. That keeps 244-style unlearning honest: delete
the slot and the answer dies.

Ablations that make the test strong (not just "number looks good"):
  head_only       glue off
  shuffle_tape    permute keys, breaking key<->value pairing
  slot_delete     drop the target slot, check target dies and retained survive
  empty_tape      no slots at all (parametric leak floor)
  prose gate      mean g_t on ordinary wiki windows must stay low

  python _stage256_slot_bias_decode.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
v0 = v19('results')
v1 = v0 / 'stage256_decision.json'
v2 = v0 / 'stage256_mini.md'
v3 = v0 / 'stage256_decode_miss_audit.md'
v4 = v0 / 'stage256_decode_miss_audit.json'
v5 = v0 / '_stage256_log.txt'
v6 = v19('checkpoints/stage191_p1_curve.pt')
v7 = v19('checkpoints/stage253_joint_l02.pt')
v8 = v19('checkpoints/stage256_slot_bias.pt')
v9 = v19('data/_wikitext103_train.txt')
v10 = 256
v11 = '{S} was appointed director of {V} in the regional chronicle of 1987 .'
v12 = '{S} was appointed director of'
v13 = 'The chronicle continues with routine administrative detail .'

def log(v20: v68) -> None:
    v21 = v20 if v20.v270('\n') else v20 + '\n'
    try:
        v271(v21, end='', flush=True)
    except v149:
        v271(v21.v399('ascii', 'replace').v361('ascii'), end='', flush=True)
    v5.v272.v150(parents=True, exist_ok=True)
    with v5.v273('a', encoding='utf-8') as v42:
        v42.v274(v21)
import _inprint_glue as glue_lib
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, DEFAULT_RETRIEVE_MODE, RetrieveStats, SlotBias, TapeView, VOTES_AUTO_MIN_SLOTS, copy_dist, hidden_and_logits, mix_logprob, raw_query, full_bank_cue_summary, retrieve_topk, slot_query_words
from _retrieval_modes import vote_scores
v12 = v14
v11 = v15

def nce_loss(v22: v151, v23: v175.v152, v24: v175.v152, v25: v175.v152, v26: v153):
    """InfoNCE over the whole bank: pull the adapted cue query onto its slot, push off the rest.

    CE through the copy mixture only nudges retrieval second-hand (it can lower the loss by
    sharpening tau instead), so W_q needs a direct retrieval objective.
    """
    v27 = v275.v154(v22.v276(v23), dim=-1)
    v28 = v27 @ v25.v48() / v26
    v29 = v175.v345(v24, v28, v175.v394(v28, -10000.0)).v155(dim=-1)
    return (v28.v155(dim=-1) - v29).v156()

def fact_batch(v22, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39: v18, v40: v68=v41):
    """Teacher-forced CE on the value tokens, logits corrected by the gated slot bias."""
    v157, v158 = ([], [])
    for v42 in v35:
        v52 = [v170 for v170 in v32.v399(v12.v408(S=v42['S'])).v43 if v170 != v36]
        v69 = [v170 for v170 in v32.v399(' ' + v42['value']).v43 if v170 != v36]
        if not v52 or not v69:
            continue
        v46 = (v52 + v69)[-v395:]
        v159 = v185(v46) - v185(v69)
        v43 = v175.v277([v46], dtype=v175.v346, device=v38)
        v160, v161 = v162(v30, v31, v43, v36)
        for v242, v278 in v218(v69):
            v48 = v159 + v242 - 1
            if v48 < 0 or v48 >= v161.v424(1):
                break
            v165 = v46[:v48 + 1]
            v164 = v161[0, v48]
            v166 = v279(v40, v22, v33, v32, v34, v165, v52, v39)
            if v166 is None:
                v169 = v175.v203(v275.v299(v164, -1) + 1e-09)
                v168 = v175.v396((), device=v38)
            else:
                v28, v280 = v166
                v167 = v153(-(v275.v299(v164, -1) * v275.v441(v164, -1)).v220())
                v281, v282 = v283(v22, v34, v28, v280, v165, v37, v38)
                v168 = v22.v284(v160[0, v48], v153(v28.v339()), v153(v28.v156()), v167, v282)
                v169 = v285(v164, v168, v281, v282)
            v157.v286(-v169[v278])
            v158.v286(v153(v168))
    if not v157:
        return (None, v153('nan'))
    return (v175.v397(v157).v156(), v153(v398.v156(v158)))

def prose_batch(v22, v30, v31, v32, v33, v34, v43: v175.v152, v36, v37, v38, v39: v18, v44: v153, v45: v16=True, v40: v68=v41):
    """Same glue on ordinary text. Under a mixture an open gate directly costs CE here; the L1 term
    only stops it from drifting up where the LM happens to be uncertain anyway."""
    v160, v161 = v162(v30, v31, v43, v36)
    v157, v158 = ([], [])
    v46 = v43[0].v163()
    v47 = [v48 for v48 in v172(v185(v46) - 1) if v46[v48] != v36 and v46[v48 + 1] != v36]
    if not v47:
        return (None, v153('nan'))
    for v48 in v47[::v339(1, v185(v47) // 8)]:
        v164 = v161[0, v48]
        v165 = v46[:v48 + 1]
        if not v45:
            v157.v286(-v175.v203(v275.v299(v164, -1) + 1e-09)[v46[v48 + 1]])
            v158.v286(0.0)
            continue
        v166 = v279(v40, v22, v33, v32, v34, v165, None, v39)
        if v166 is None:
            v157.v286(-v175.v203(v275.v299(v164, -1) + 1e-09)[v46[v48 + 1]])
            v158.v286(0.0)
            continue
        v28, v280 = v166
        v167 = v153(-(v275.v299(v164, -1) * v275.v441(v164, -1)).v220())
        v281, v282 = v283(v22, v34, v28, v280, v165, v37, v38)
        v168 = v22.v284(v160[0, v48], v153(v28.v339()), v153(v28.v156()), v167, v282)
        v169 = v285(v164, v168, v281, v282)
        v157.v286(-v169[v46[v48 + 1]] + v44 * v168)
        v158.v286(v153(v168))
    if not v157:
        return (None, v153('nan'))
    return (v175.v397(v157).v156(), v153(v398.v156(v158)))

@v175.v54()
def free_decode(v22, v30, v31, v32, v33, v34, v49, v36, v37, v38, v39: v18, v50: v18, v45: v16, v40: v68=v41, v51: v287 | None=None) -> v55[v68, v153]:
    """Greedy free-form continuation of the cue; no candidate set anywhere."""
    v52 = [v170 for v170 in v32.v399(v12.v408(S=v49['S'])).v43 if v170 != v36]
    v46 = v57(v52)
    v171, v158 = ([], [])
    for v53 in v172(v50):
        v43 = v175.v277([v46[-v395:]], dtype=v175.v346, device=v38)
        v160, v161 = v162(v30, v31, v43, v36)
        v164 = v161[0, -1]
        v173 = v175.v203(v275.v299(v164, -1) + 1e-09)
        if v45:
            v166 = v279(v40, v22, v33, v32, v34, v46, v52, v39, stats=v51)
            if v166 is not None:
                v28, v280 = v166
                v167 = v153(-(v275.v299(v164, -1) * v275.v441(v164, -1)).v220())
                v281, v282 = v283(v22, v34, v28, v280, v46, v37, v38)
                v168 = v22.v284(v160[0, -1], v153(v28.v339()), v153(v28.v156()), v167, v282)
                v173 = v285(v164, v168, v281, v282)
                v158.v286(v153(v168))
        v174 = v18(v173.v347())
        v171.v286(v174)
        v46.v286(v174)
    return (v32.v361(v171).v189(), v153(v398.v156(v158)) if v158 else v153('nan'))

@v175.v54()
def retrieval_report(v22, v33, v32, v34: v176, v35, v36, v39: v18) -> v57[v17]:
    """At the cue (the exact state free decode starts from): where does the gold slot rank?"""
    v56 = []
    for v42 in v35:
        v52 = [v170 for v170 in v32.v399(v12.v408(S=v42['S'])).v43 if v170 != v36]
        v177 = [v288 for v288, v400 in v218(v34.v353) if v400 == v42['value']]
        if v34.v348 is not None and v34.v125() >= v349:
            v61 = v350(v32.v361(v52))
            v289 = v351(v61, v34.v348.v348, v34.v348.v352)
            v290 = v339((v289.v362(v288, 0.0) for v288 in v177), default=0.0)
            v291 = 1 + v220((1 for v400 in v289.v353() if v400 > v290))
            v292 = v339(v289, key=v289.v362) if v289 else 0
            v293 = v34.v353[v292]
            v294 = v290
            v166 = v279(v41, v22, v33, v32, v34, v52, v52, v39)
            v184 = v22.v355(v166[0]) if v166 is not None else v175.v396(0)
        else:
            from _inprint_glue import ctx_query
            v27 = v354(v22, v33, v32, v52, anchor_ids=v52)
            if v27 is None:
                v56.v286({'S': v42['S'], 'rank': None})
                continue
            v28 = v34.v25 @ v27
            v294 = v153(v28[v177].v339()) if v177 else v153('-inf')
            v291 = 1 + v18((v28 > v294).v220())
            v293 = v34.v353[v18(v28.v347())]
            v184 = v22.v355(v175.v89(v28, v409(v39, v28.v442()))[0])
        v56.v286({'S': v42['S'], 'gold': v42['value'], 'rank': v291, 'top1': v293, 'gold_sim': v294, 'w_max': v153(v184.v339())})
    return v56

def exact_match(v58: v68, v59: v68) -> v16:
    return v58.v189().v401(' ')[0].v189(' .,;:') == v59 if v58 else False

def match_in_window(v58: v68, v59: v68, v60: v18=3) -> v16:
    """Value in first n words (257 em_window3). Catches metric misses like Sharaif Shara."""
    if not v58:
        return False
    v61 = [v184.v189(' .,;:') for v184 in v58.v189().v401(' ')[:v60]]
    return v59 in v61

def _token_rank(v62: v175.v152, v63: v18) -> v55[v153, v18]:
    v64 = v153(v62[v63])
    return (v64, 1 + v18((v62 > v64).v220().v402()))

def _rand_values(v60: v18, v65: v295.v178, v66: v228[v68]) -> v57[v68]:
    """Nonsense strings — not English dictionary words; BPE usually multi-token."""
    v179, v180 = ('aeiou', 'bcdfghjklmnpqrstvwxyz')
    v67, v181 = ([], v228())
    while v185(v67) < v60:
        v182 = v65.v296(6, 11)
        v183 = []
        for v170 in v172(v182):
            v183.v286(v65.v403(v180 if v170 % 2 == 0 else v179))
        v184 = ''.v229(v183).v297()
        if v184 in v66 or v184 in v181 or v185(v184) < 5:
            continue
        v181.v298(v184)
        v67.v286(v184)
    return v67

@v175.v54()
def decode_step_audit(v22, v30, v31, v32, v33, v34, v49, v36, v37, v38, v39: v18, v50: v18=6, v40: v68=v41) -> v17:
    """Per-step gate / p_copy / mix during greedy decode — copy has no end-of-value state."""
    v52 = [v170 for v170 in v32.v399(v12.v408(S=v49['S'])).v43 if v170 != v36]
    v69 = [v170 for v170 in v32.v399(' ' + v49['value']).v43 if v170 != v36]
    if not v52 or not v69:
        return {'S': v49['S'], 'error': 'empty cue or value'}
    v46 = v57(v52)
    v171, v83 = ([], [])
    v70 = v185(v69)
    v71 = False
    for v48 in v172(v50):
        v43 = v175.v277([v46[-v395:]], dtype=v175.v346, device=v38)
        v160, v161 = v162(v30, v31, v43, v36)
        v164 = v161[0, -1]
        v186 = v275.v299(v164, dim=-1)
        v187 = v18(v186.v347())
        v188 = v69[v48] if v48 < v70 else None
        v166 = v279(v40, v22, v33, v32, v34, v46, v52, v39)
        if v166 is None:
            v174 = v187
            v83.v286({'t': v48, 'gate': 0.0, 'p_copy_gold': None, 'copy_rank_gold': None, 'mix_top': v32.v361([v174]), 'lm_top': v32.v361([v187]), 'gold_tok': v32.v361([v188]) if v188 is not None else None, 'past_value_end': v48 >= v70})
        else:
            v28, v280 = v166
            v167 = v153(-(v186 * v275.v441(v164, -1)).v220())
            v281, v282 = v283(v22, v34, v28, v280, v46, v37, v38)
            v168 = v153(v22.v284(v160[0, -1], v153(v28.v339()), v153(v28.v156()), v167, v282))
            v173 = v285(v164, v168, v281, v282)
            v300 = v18(v173.v347())
            v301 = v69[0]
            v356, v357 = v358(v281, v301)
            if v48 >= v70 and v357 == 1 and (v168 >= 0.35):
                v71 = True
            v359, v360 = (None, None)
            if v188 is not None:
                v359, v360 = v358(v281, v188)
            v83.v286({'t': v48, 'gate': v168, 'cov': v153(v282), 'p_copy_gold': v359, 'copy_rank_gold': v360, 'p_copy_first_val': v356, 'copy_rank_first_val': v357, 'mix_top': v32.v361([v300]), 'lm_top': v32.v361([v187]), 'gold_tok': v32.v361([v188]) if v188 is not None else None, 'past_value_end': v48 >= v70})
            v174 = v300
        v171.v286(v174)
        v46.v286(v174)
    v72 = v32.v361(v171).v189()
    v73 = v190(v72, v49['value'])
    v74 = v191(v72, v49['value'], 3)
    v75 = v83[0]['gate'] if v83 else v153('nan')
    v76 = v153(v398.v156([v425['gate'] for v425 in v83])) if v83 else v153('nan')
    v77 = v83[0] if v83 else {}
    v78 = v77.v362('copy_rank_gold') == 1 and v77.v362('gate', 0) >= 0.35
    if v73:
        v192 = 'ok'
    elif v74 and (not v73):
        v192 = 'metric_first_word'
    elif v75 is not None and v75 < 0.35:
        v192 = 'gate_low'
    elif v77.v362('copy_rank_gold', 999) not in (1, None) and v77.v362('copy_rank_gold'):
        v192 = 'readout_copy'
    elif v78 and (not v73):
        v192 = 'copy_no_span_lock'
    else:
        v192 = 'other'
    return {'S': v49['S'], 'gold': v49['value'], 'n_val_tokens': v70, 'got': v72, 'em_ok': v73, 'em_window3': v74, 'gate_step0': v75, 'gate_mean_decode': v76, 'copy_restart_after_value': v71, 'diagnosis': v192, 'steps': v83}

@v175.v54()
def em_over(v22, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v50, v45=True, v79=None, v40: v68=v41, v51: v287 | None=None):
    v193, v194 = (0, [])
    for v42 in v35:
        v72, v284 = v302(v22, v30, v31, v32, v33, v34, v42, v36, v37, v38, v39, v50, v45, retrieve_mode=v40, stats=v51)
        v193 += v18(v190(v72, v42['value']))
        if not v404.v363(v284):
            v194.v286(v284)
        if v79 is not None and v185(v79) < 6:
            v79.v286({'cue_S': v42['S'], 'gold': v42['value'], 'got': v72, 'gate': v284})
    return (v193 / v339(1, v185(v35)), v153(v398.v156(v194)) if v194 else v153('nan'))

def main() -> v18:
    v80 = v303.v195()
    v80.v196('--smoke', action='store_true')
    v80.v196('--steps', type=v18, default=0)
    v80.v196('--topk', type=v18, default=8)
    v80.v196('--gate-l1', type=v153, default=0.02, help='L1 on g_t over prose steps')
    v80.v196('--nce-w', type=v153, default=1.0, help='weight of the retrieval InfoNCE term')
    v80.v196('--nce-tau', type=v153, default=0.05)
    v80.v196('--retrieve-mode', default=v41, choices=('auto', 'cosine', 'votes'), help='glue retrieval during train/eval (eval also logs all three after train)')
    v80.v196('--nce-pool', choices=('wiki', 'facts'), default='wiki', help='train W_q on bank-wide (prefix->slot) pairs, or overfit the fit facts (ablation)')
    v80.v196('--eval-only', action='store_true', help='load glue ckpt, skip training (audit retrieve paths)')
    v80.v196('--decode-audit', action='store_true', help='with eval-only: per-step gate/p_copy audit + miss diagnosis for held-out facts')
    v80.v196('--random-values', action='store_true', help='planted values = nonsense strings (control: EM should fall to single-token fraction)')
    v80.v196('--facts', type=v18, default=0)
    v80.v196('--distractor-slots', type=v18, default=0, help='real wiki entities added as bank noise')
    v81 = v80.v197()
    v5.v198('', encoding='utf-8')
    v38 = v175.v38('cuda' if v175.v405.v364() else 'cpu')
    v65 = v295.v178(v10)
    v175.v199(v10)
    v82 = v200.v200()
    v83 = 0 if v81.v108 else v81.v83 or (200 if v81.v202 else 800)
    v84 = v81.v35 or (8 if v81.v202 else 48)
    v85 = v81.v201 or (150 if v81.v202 else 1200)
    v50 = 4 if v81.v202 else 6
    v86 = 4 if v81.v202 else 12
    v87 = 40 if v81.v202 else 120
    v88 = 400 if v81.v202 else 6000
    v39 = v81.v89
    v203(f'Stage256 slot-bias glue start {v434.v420(v435.v421).v340()} device={v38} steps={v83} facts={v84} distractors={v85} topk={v39}')
    v53, v53, v204, v205 = v206()
    v32 = v304.v207(v68(v365.v305))
    v37 = v32.v208()
    v36 = v32.v306(v307) or 0
    v31 = v406.v366(v32, v204, v36, v37).v209(v38)
    v90 = v7 if v7.v308() else v6
    v30 = v367(v205, v37).v209(v38)
    v30.v210(v175.v320(v90, map_location=v38, weights_only=False)['model'])
    v30.v211()
    for v64 in v30.v212():
        v64.v309(False)
    v203(f'  trunk={v90.v268} (frozen)')
    v91 = v367(v205, v37).v209(v38)
    v91.v210(v175.v320(v6, map_location=v38, weights_only=False)['model'])
    v91.v211()
    for v64 in v91.v212():
        v64.v309(False)
    v92 = v213(v91, v204, v38)
    with v9.v273('r', encoding='utf-8', errors='ignore') as v42:
        v214 = v42.v310(1000000 if v81.v202 else 6000000)
    v93 = v57(v17.v311((v20.v371(1) for v20 in v370.v314(v214) if v185(v20.v371(1)) >= 5)))
    v65.v215(v93)
    v94 = [v368.v189() for v368 in v214.v401('\n') if v185(v368.v189()) >= 60][:v88]
    v95 = [v184 for v184 in v407(v228(v93), v65, v84 + 30) if v185(v184) >= 5][:v84]
    if v81.v96:
        v216 = v312(v84, v65, v228(v93) | v228(v95))
        v203(f'  random-values control: {v84} nonsense strings (not wiki entities)')
    else:
        v216 = v93[:v84]
    v35 = []
    for v170, v217 in v218(v95):
        v219 = v216[v170]
        v35.v286({'S': v217, 'value': v219, 'sent': v11.v408(S=v217, V=v219), 'fid': f'f{v170}', 'glue_train': v170 % 2 == 0})
    v97 = [v42 for v42 in v35 if v42['glue_train']]
    v98 = [v42 for v42 in v35 if not v42['glue_train']]
    v99 = v220((1 for v42 in v98 if v185([v170 for v170 in v32.v399(' ' + v42['value']).v43 if v170 != v36]) == 1))
    v203(f'  facts: fit={v185(v97)} held_out={v185(v98)} eval_single_token_values={v99}/{v185(v98)}')
    if v81.v96 and v81.v108:
        v203('random-values + eval-only: refuse (needs train on new tape values)')
        return 1
    v221, v222, v223 = ([], [], [])
    v224, v225 = ([], [])
    for v42 in v35:
        v226 = v92.v369([v42['S']])[0]
        v227 = v92.v313(v42['sent'], exclude=v42['value'])
        v221.v286(v275.v154(v226 + v227, dim=-1) if v227 is not None else v226)
        v222.v286(v42['value'])
        v223.v286(v350(v42['sent'], exclude=v42['value']))
    v100 = v228(v222)
    for v101 in v94:
        if v185(v222) >= v84 + v85:
            break
        for v20 in v370.v314(v101):
            v167 = v20.v371(1)
            if v185(v167) < 5 or v167 in v100:
                continue
            v372, v373 = (v339(0, v20.v436() - 120), v409(v185(v101), v20.v437() + 120))
            v227 = v92.v313(v101[v372:v373], exclude=v167)
            if v227 is None:
                continue
            v315 = [v184 for v184 in v438.v426(v101[v372:v20.v436()]) if v184 != v167]
            if not v315:
                continue
            v221.v286(v275.v154(v92.v369([v315[-1]])[0] + v227, dim=-1))
            v223.v286(v350(v101[v372:v373], exclude=v167))
            v316 = v92.v313(v101[v372:v20.v436()])
            if v316 is not None:
                v224.v286(v275.v154(v92.v369([v315[-1]])[0] + v316, dim=-1))
                v225.v286(v185(v222))
            v222.v286(v167)
            v100.v298(v167)
            if v185(v222) >= v84 + v85:
                break
    v34 = v176(v175.v397(v221, 0).v209(v38), v222, v32, v36, ctxw=v223)
    v203(f'  tape slots={v185(v222)} ({v185(v35)} planted + {v185(v222) - v185(v35)} wiki noise) retrieve=auto (votes if >={v349})')
    v102 = '\n'.v229(v94 + [v13] * v409(v185(v35), v185(v94) // 4))
    v230, v231 = v317.v232(v102, v32, v36, max_lines=v88 + 64, min_line_len=20)
    v103 = v185(v231) - 1
    v104 = v57(v172(v339(1, v103 - v339(2, v103 // 20)), v103))
    v105 = v57(v172(0, v104[0]))
    v106 = v318.v233(v230, v231, v104, v36, v86, v10 + 5)
    v107 = v319.v234(v87)
    v203(f'  prose docs={v103} train={v185(v105)} hold={v185(v104)}')
    v22 = v151(2 * (v30.v410.v374 // 2), v38)
    if v81.v108:
        if not v8.v308():
            v203(f'eval-only: missing {v8}')
            return 1
        v235 = v175.v320(v8, map_location=v38, weights_only=False)
        v22.v276.v210(v235['W_q_glue'] if 'W_q_glue' in v235 else v235['W_q'])
        v22.v375.v210(v235['gate'])
        with v175.v54():
            v22.v419.v411.v376(v235['log_tau'].v209(v38).v412(v22.v419.v411))
        v22.v211()
        v203(f'eval-only: loaded glue from {v8.v268}')
    v109 = v175.v321.v236(v22.v322(), lr=0.003, weight_decay=0.01)
    if v81.v237 == 'facts':
        with v175.v54():
            v323, v324 = ([], [])
            for v42 in v97:
                v52 = [v170 for v170 in v32.v399(v12.v408(S=v42['S'])).v43 if v170 != v36]
                v377 = v413(v92, v32, v52, anchor_ids=v52)
                if v377 is None:
                    continue
                v323.v286(v377)
                v324.v286(v222.v427(v42['value']))
        v224, v225 = (v323, v324)
    v110 = v175.v397(v224).v209(v38).v153() if v224 else None
    v111 = v175.v277(v225, device=v38) if v225 else None
    v112 = v34.v25.v153()
    v203(f'  W_q training pairs={(0 if v110 is None else v110.v424(0))} (pool={v81.v237})')
    v113 = v318.v238(v30, v106, v31, v36, v38)
    v114 = v319.v239(v30, v31, v36, v107, v38)
    v240, v53 = v241(v22, v30, v31, v32, v92, v34, v98, v36, v37, v38, v39, v50, use_glue=False)
    v203(f'baseline hold_ce={v113:.3f} exam={v114:.3f} EM(head_only)={v240:.3f}')
    v115 = []
    v40 = v81.v40
    if v83 > 0:
        for v242 in v172(1, v83 + 1):
            v325 = [v97[v65.v428(v185(v97))] for v53 in v172(v409(4, v185(v97)))]
            v378, v379 = v380(v22, v30, v31, v32, v92, v34, v325, v36, v37, v38, v39, v40)
            v43 = v319.v429(v230, v231, 1, v65, v36, v105).v209(v38)
            v381, v382 = v383(v22, v30, v31, v32, v92, v34, v43, v36, v37, v38, v39, v81.v44, retrieve_mode=v40)
            v326 = None
            if v110 is not None and v81.v414 > 0:
                v384 = v175.v296(0, v110.v424(0), (v409(64, v110.v424(0)),), device=v38)
                v177 = v275.v439(v111[v384], v112.v424(0)).v16()
                v326 = v81.v414 * v430(v22, v110[v384], v177, v112, v81.v431)
            v327 = [v385 for v385 in (v378, v381, v326) if v385 is not None]
            if not v327:
                continue
            v328 = v327[0]
            for v64 in v327[1:]:
                v328 = v328 + v64
            v109.v386(set_to_none=True)
            v328.v387()
            v175.v432.v415.v388(v22.v322(), 1.0)
            v109.v242()
            if v242 % v339(1, v83 // 6) == 0:
                v115.v286({'step': v242, 'loss_fact': v153(v378) if v378 is not None else None, 'loss_prose': v153(v381) if v381 is not None else None, 'loss_nce': v153(v326) if v326 is not None else None, 'gate_fact': v379, 'gate_prose': v382, 'tau': v153(v175.v418(v22.v419))})
                v203(f"  step {v242}/{v83} fact={(v153(v378) if v378 is not None else v153('nan')):.3f} prose={(v153(v381) if v381 is not None else v153('nan')):.3f} nce={(v153(v326) if v326 is not None else v153('nan')):.3f} g_fact={v379:.3f} g_prose={v382:.3f} tau={v153(v175.v418(v22.v419)):.3f} ({v200.v200() - v82:.0f}s)")
    v22.v211()
    v116: v57[v17] = []
    if v81.v117:
        v116 = [v389(v22, v30, v31, v32, v92, v34, v42, v36, v37, v38, v39, v50, v40) for v42 in v98]
        v243 = [v122 for v122 in v116 if not v122.v362('em_ok')]
        v244 = [v122 for v122 in v243 if v122.v362('diagnosis') != 'metric_first_word']
        v245: v17[v68, v18] = {}
        for v122 in v243:
            v129 = v122.v362('diagnosis', '?')
            v245[v129] = v245.v362(v129, 0) + 1
        v246 = v220((1 for v122 in v116 if v122.v362('copy_restart_after_value')))
        v247 = v220((1 for v122 in v116 if v122.v362('em_window3')))
        v203(f'decode step audit: EM-miss {v185(v243)}/{v185(v98)} (mechanism {v185(v244)}; metric_only {v185(v243) - v185(v244)}); em_window3={v247}/{v185(v98)}; copy_restart={v246}/{v185(v98)}')
        for v122 in v243:
            v329 = (v122.v362('steps') or [{}])[0]
            v203(f"  MISS {v122['S']} gold={v122['gold']} got={v122.v362('got')} g0={v122.v362('gate_step0', v153('nan')):.3f} g_mean={v122.v362('gate_mean_decode', v153('nan')):.3f} n_tok={v122.v362('n_val_tokens')} restart={v122.v362('copy_restart_after_value')} -> {v122.v362('diagnosis')}")
            for v255 in (v122.v362('steps') or [])[:4]:
                v203(f"    t={v255['t']} g={v255['gate']:.3f} past_end={v255.v362('past_value_end')} copy_rank_gold={v255.v362('copy_rank_gold')} copy_rank_1st={v255.v362('copy_rank_first_val')} mix={v255.v362('mix_top')!r} gold={v255.v362('gold_tok')!r}")
        v248 = {'n_eval': v185(v98), 'n_miss_em': v185(v243), 'n_miss_mechanism': v185(v244), 'n_em_window3': v247, 'n_copy_restart': v246, 'diagnosis_counts': v245, 'rows': v116, 'note': 'Tape supplies ~first BPE; rest is LM spelling prior. copy_restart_after_value = after value tokens exhausted, p_copy still ranks first value token #1 (no span-lock). metric_first_word = em_window3 would pass (first-word EM too strict).'}
        v249 = v0 / ('stage256_decode_miss_audit_random_values.md' if v81.v96 else 'stage256_decode_miss_audit.md')
        v250 = v0 / ('stage256_decode_miss_audit_random_values.json' if v81.v96 else 'stage256_decode_miss_audit.json')
        v250.v198(v393.v341(v248, indent=2), encoding='utf-8')
        v251 = ['# Stage 256 — decode audit (per-step)\n\n', f'Held-out **{v185(v98)}** · first-word EM miss **{v185(v243)}** · mechanism miss **{v185(v244)}** · em_window3 **{v247}** · copy_restart **{v246}** · retrieve `{v40}`' + (' · **random-values**' if v81.v96 else '') + '\n\n', 'Retrieval @ cue was rank **1.0**. Gate opens on step 0; copy has **no end-of-value** (restarts at first token). Fix = **257 span-lock**.\n\n', '## Diagnosis counts\n\n']
        for v129, v227 in v330(v245.v107()):
            v251.v286(f'- **{v129}**: {v227}\n')
        v251.v286('\n| S | gold | got | g0 | g_mean | n_tok | restart | diagnosis |\n|---|------|-----|---:|-------:|------:|:-------:|----------|\n')
        for v122 in v243:
            v251.v286(f"| {v122['S']} | {v122['gold']} | {v122.v362('got', '')} | {v122.v362('gate_step0', v153('nan')):.3f} | {v122.v362('gate_mean_decode', v153('nan')):.3f} | {v122.v362('n_val_tokens')} | {v122.v362('copy_restart_after_value')} | {v122.v362('diagnosis')} |\n")
        v252 = [v122 for v122 in v116 if v122.v362('em_ok') and v122.v362('copy_restart_after_value')][:4]
        if v252:
            v251.v286('\n## OK but copy restart (same disease)\n\n')
            for v122 in v252:
                v251.v286(f"- **{v122['S']}** `{v122['gold']}` → `{v122['got']}` (g_mean={v122['gate_mean_decode']:.3f})\n")
        v249.v198(''.v229(v251), encoding='utf-8')
        v203(f'  wrote {v249.v268} ({v185(v243)} EM-miss / {v185(v244)} mechanism)')
    v118 = v253(v22, v92, v32, v34, v98, v36, v39)
    v119 = v253(v22, v92, v32, v34, v97, v36, v39)
    v120 = [v122['rank'] for v122 in v118 if v122.v362('rank')]
    v121 = [v122['rank'] for v122 in v119 if v122.v362('rank')]
    v203(f'retrieval at cue: held-out top1={v398.v156([v122 == 1 for v122 in v120]):.2f} median_rank={v398.v416(v120):.0f} | fit top1={v398.v156([v122 == 1 for v122 in v121]):.2f}')
    for v122 in v118[:4]:
        v203(f'    {v122}')
    v123: v57[v17] = []
    v124: v17[v68, v17] = {}
    v125 = v34.v125()
    v203(f"retrieve audit: n_live={v125} auto_eff={v433.v417('auto', v125)} postings={('yes' if v34.v348 else 'no')}")
    v126: v17[v68, v153] = {}
    v131, v254 = (v153('nan'), v153('nan'))
    for v127 in ('auto', 'cosine', 'votes'):
        v255 = v287()
        v256, v76 = v241(v22, v30, v31, v32, v92, v34, v98, v36, v37, v38, v39, v50, retrieve_mode=v127, stats=v255, samples=v123 if v127 == v40 else None)
        v126[v127] = v256
        v124[v127] = v255.v331()
        if v127 == v40:
            v131, v254 = (v256, v76)
    v203(f'retrieve EM (same glue, eval decode): {v393.v341(v126)}')
    v203(f'retrieve decode steps: {v393.v341(v124)}')
    v128: v17[v68, v17] = {}
    for v127 in ('auto', 'cosine', 'votes'):
        v128[v127] = v332(v127, v22, v92, v32, v34, v98, v36, cue_tmpl=v12)
    v203(f'full_bank at cue (held-out): {v393.v341(v128)}')
    for v129 in v123[:4]:
        v203(f'    decode {v129}')
    v257, v53 = v241(v22, v30, v31, v32, v92, v34.v333(v10 + 1), v98, v36, v37, v38, v39, v50)
    v258, v53 = v241(v22, v30, v31, v32, v92, v34.v334(), v98, v36, v37, v38, v39, v50)
    v259, v260 = ([], [])
    for v42 in v98:
        v261 = v34.v335()
        v261.v336(v42['value'])
        v337, v53 = v241(v22, v30, v31, v32, v92, v261, [v42], v36, v37, v38, v39, v50)
        v259.v286(v337)
        v262 = [v338 for v338 in v98 if v338 is not v42]
        if v262:
            v390, v53 = v241(v22, v30, v31, v32, v92, v261, v262, v36, v37, v38, v39, v50)
            v260.v286(v390)
    v130 = v131
    v132 = v153(v398.v156(v259)) if v259 else v153('nan')
    v133 = v153(v398.v156(v260)) if v260 else v153('nan')
    with v175.v54():
        v264, v265, v266 = ([], [], [])
        v263 = v295.v178(v10 + 99)
        for v53 in v172(12):
            v43 = v319.v429(v230, v231, 1, v263, v36, v104).v209(v38)
            v391, v284 = v383(v22, v30, v31, v32, v92, v34, v43, v36, v37, v38, v39, 0.0, True)
            v392, v53 = v383(v22, v30, v31, v32, v92, v34, v43, v36, v37, v38, v39, 0.0, False)
            if v391 is not None and v392 is not None:
                v265.v286(v153(v391))
                v266.v286(v153(v392))
            if not v404.v363(v284):
                v264.v286(v284)
    v134 = v153(v398.v156(v264)) if v264 else v153('nan')
    v135 = v153(v398.v156(v265)) if v265 else v153('nan')
    v136 = v153(v398.v156(v266)) if v266 else v153('nan')
    v137 = v318.v238(v30, v106, v31, v36, v38)
    v138 = v131 >= 0.6
    v139 = v131 >= v240 + 0.2
    v140 = v257 <= v339(0.1, v131 - 0.4)
    v141 = v130 >= 0.4 and v132 <= 0.1 and (v133 >= 0.7 * v131)
    v142 = v258 <= 0.1
    v143 = not v404.v363(v135) and (not v404.v363(v136)) and (v135 <= v136 + 0.05)
    v144 = not v404.v363(v254) and (not v404.v363(v134)) and (v254 >= v134 + 0.2)
    v145 = v138 and v139 and v140 and v142 and v143
    if v145 and v141 and v144:
        v267 = 'SLOT_BIAS_GLUE_OK'
    elif v139 and v140 and v142 and v143:
        v267 = 'SLOT_BIAS_GLUE_PARTIAL'
    else:
        v267 = 'SLOT_BIAS_GLUE_NO'
    v67 = {'stage': 256, 'overall': v267, 'trunk': v90.v268, 'topk': v39, 'steps': v83, 'n_facts': v185(v35), 'n_fit': v185(v97), 'n_eval': v185(v98), 'tape_slots': v185(v222), 'random_values': v16(v81.v96), 'gates': {'G_freeform_value': v138, 'G_beats_head_only': v139, 'G_tape_causal': v140, 'G_slot_delete_clean': v141, 'G_no_param_leak': v142, 'G_lang_intact': v143, 'G_gate_selective': v144}, 'summary': {'em_head_only': v240, 'em_glue': v131, 'em_shuffled_tape': v257, 'em_empty_tape': v258, 'em_target_before_delete': v130, 'em_target_after_delete': v132, 'em_retained_after_delete': v133, 'gate_mean_fact': v254, 'gate_mean_prose': v134, 'prose_ce_glue_on': v135, 'prose_ce_glue_off': v136, 'hold_ce_base': v113, 'hold_ce_after': v137, 'exam_base': v114, 'tau': v153(v175.v418(v22.v419)), 'gate_l1': v81.v44, 'retrieve_mode_train': v40, 'retrieve_em_eval': v126, 'retrieve_decode_steps': v124, 'full_bank_cue_eval': v128, 'fp_version': v6.v268, 'eval_single_token_values': v99, 'eval_single_token_frac': v99 / v339(1, v185(v98))}, 'curve': v115, 'retrieval_at_cue': {'held_out_top1': v153(v398.v156([v122 == 1 for v122 in v120])) if v120 else None, 'held_out_median_rank': v153(v398.v416(v120)) if v120 else None, 'fit_top1': v153(v398.v156([v122 == 1 for v122 in v121])) if v121 else None, 'rows': v118[:8]}, 'decode_samples': v123, 'note': "Glue only: trunk frozen, W_q + gate MLP + tau trained. Copy mixture p' = (1-g)p_LM + g*p_copy, so an open gate is paid for in CE. Values exist in the tape only, so CE toward the right value can flow only through the bias path. EM is free-form greedy decode (no candidate set); scored on facts the glue never fit.", 'timestamp': v434.v420(v435.v421).v340(), 'wall_s': v200.v200() - v82}
    v146 = v0 / ('stage256_decision_random_values.json' if v81.v96 else 'stage256_decision.json')
    v146.v198(v393.v341(v67, indent=2), encoding='utf-8')
    v147 = v0 / ('stage256_mini_random_values.md' if v81.v96 else 'stage256_mini.md')
    v147.v198(f'# Stage 256 slot-bias glue\n\n**{v267}** trunk={v90.v268} slots={v185(v222)} eval_facts={v185(v98)}\n\n- EM free-form: head_only **{v240:.3f}** -> glue **{v131:.3f}**\n- causal: shuffled **{v257:.3f}**, empty **{v258:.3f}**\n- slot delete: target {v130:.2f} -> {v132:.2f}, retained {v133:.2f}\n- gate: fact **{v254:.3f}** vs prose **{v134:.3f}**\n- prose CE glue off {v136:.3f} -> on {v135:.3f} (hold CE {v113:.3f})\n', encoding='utf-8')
    v203(v393.v341({'overall': v267, 'gates': v67['gates'], 'summary': v67['summary']}, indent=2))
    if not v81.v202 and (not v81.v96):
        v8.v272.v150(exist_ok=True)
        v175.v342({'W_q': v22.v276.v422(), 'W_q_glue': v22.v276.v422(), 'gate': v22.v375.v422(), 'log_tau': v22.v419.v440().v423(), 'stage': 256}, v8)
    elif v81.v96 and (not v81.v202):
        v343 = v19('checkpoints/stage256_slot_bias_random_values.pt')
        v343.v272.v150(exist_ok=True)
        v175.v342({'W_q': v22.v276.v422(), 'W_q_glue': v22.v276.v422(), 'gate': v22.v375.v422(), 'log_tau': v22.v419.v440().v423(), 'stage': 256, 'random_values': True}, v343)
        v203(f'  saved random-values glue -> {v343.v268} (did not overwrite {v8.v268})')
    return 0
if v148 == '__main__':
    raise v269(v344())