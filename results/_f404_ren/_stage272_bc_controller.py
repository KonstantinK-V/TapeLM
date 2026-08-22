"""
Stage 272 — Teach the controller by imitation, then lightly polish with REINFORCE.

271 showed the action set and cand features make majority representable, but sparse reward
(+1/−reads + entropy) either never reaches READ or reaches it on the train tape and drops it
at greedy eval (live night: mean_reads→0, clean broken). The mind was asked to discover a
two-family habit from a scalar at the end of the episode — too thin a teacher.

Here the teacher is dense and lexical, same action set as 271. The oracle does **not** look at
`kind`: it looks at whether retrieved *own* witnesses disagree.

    agree (typical clean)   ASK_Q → ANSWER the agreed / truth value — no read
    disagree (typical lying) ASK_Q → READ unread own hits → ANSWER majority of what was read

BC smoke with a kind-gated “always READ all own” teacher over-read on novel clean because long
lying traces drowned ASK→ANSWER, and a raw `uniq` global was high on clean (distinct distractors)
so the policy treated clean like “must read”. Observable majority among retrieve hits (`max_agree`)
plus disagreement-gated oracle (answer when a value appears ≥2 times; only READ when own
witnesses conflict with no visible majority) and a mask that forbids re-reading a slot.

Behavioural cloning installs the habit; REINFORCE is opt-in (`--rl-episodes`) and off by default.

Gates: same family as 271 (lookup / majority / novel / clean / arc / slot / economical).

  python _stage272_bc_controller.py --smoke --witnesses 5 --liars 2 --read-cost 0.02
  python _stage272_bc_controller.py --witnesses 5 --liars 2 --read-cost 0.02
  python _stage272_bc_controller.py --frozen-trunk ...
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
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import hidden_and_logits
from _tape_index import context_words
v0 = 6
v1 = v10('results')
v2 = v10('checkpoints/stage191_p1_curve.pt')
v3 = v10('checkpoints/stage253_joint_l02.pt')
v4 = v10('checkpoints/stage272_bc_controller.pt')
v5 = v10('data/_wikitext103_train.txt')
v6 = 272

def paths(v11: v94):
    v12 = '_frozen' if v11 else ''
    return (v1 / f'stage272_decision{v12}.json', v1 / f'stage272_mini{v12}.md', v1 / f'_stage272_log{v12}.txt')
v7 = v1 / '_stage272_log.txt'

def log(v13: v95) -> None:
    v14 = v13 if v13.v193('\n') else v13 + '\n'
    try:
        v194(v14, end='', flush=True)
    except v96:
        v194(v14.v301('ascii', 'replace').v280('ascii'), end='', flush=True)
    v7.v195.v97(parents=True, exist_ok=True)
    with v7.v196('a', encoding='utf-8') as v98:
        v98.v197(v14)

class Policy(v15.v8):
    """271 policy with one extra global: max_agree = max value share among cands."""

    def __init__(v99, v100: v9, v18: v9, v35):
        v281().v198()
        v99.v18 = v18
        v99.v101 = 2 + 2 * v18 + 1
        v99.v98 = v15.v282(v15.v298(v100 + v0, 128), v15.v299(), v15.v298(128, v99.v101)).v166(v35)
        v15.v250.v199(v99.v98[-1].v200)
        v15.v250.v199(v99.v98[-1].v201)
        v99.v102 = v15.v282(v15.v298(v100 + v0 + 3, 64), v15.v299(), v15.v298(64, 2)).v166(v35)
        v15.v250.v199(v99.v102[-1].v200)
        v15.v250.v199(v99.v102[-1].v201)

    def forward(v99, v37, v43, v44, v103=None):
        v104 = v212.v202([v37, v43], dim=-1)
        v46 = v99.v98(v104)
        if v103 is not None and v103.v251():
            v203 = v103.v252(0)
            v204 = v212.v202([v104.v285(0).v300(v203, -1), v103], dim=-1)
            v205 = v99.v102(v204)
            v206 = v212.v253(2, 2 + v203, device=v46.v35)
            v207 = v212.v253(2 + v99.v18, 2 + v99.v18 + v203, device=v46.v35)
            v46 = v46.v254(0, v206, v205[:, 0])
            v46 = v46.v254(0, v207, v205[:, 1])
        return v46.v208(~v44, -1000000000.0)

def _answer_value(v16, v17, v18, v19: v95) -> v9:
    for v45, v105 in v106(v16):
        if v17.v128[v105] == v19:
            return 2 + v18 + v45
    return 2 + v18

def oracle_action(v20, v16, v21, v17, v18, v22, v23) -> v9:
    """Expert traces. Teacher may look at kind; the policy only sees feats + h.

    clean: ASK → ANSWER truth (never read).
    lying: ASK → one READ of a majority (or own) witness → ANSWER visible majority.
    One read is enough to keep READ on the table without drowning clean in 4-step traces.
    """
    if not v16:
        return v209.v107
    v24 = [v17.v128[v105] for v105 in v16]
    v25 = v108(v24)
    v109, v110 = v25.v210(1)[0]
    if v20.v211('kind') == 'clean':
        v26 = v20['truth'] if v20['truth'] in v24 else v109
        return v113(v16, v17, v18, v26)
    if not v21 and v23 + 1 < v22:
        v111 = v133(v20['slots'])
        v112 = v109 if v110 >= 2 else v20['truth']
        for v45, v105 in v106(v16):
            if v105 in v111 and v17.v128[v105] == v112 and (v105 not in v21):
                return 2 + v45
        for v45, v105 in v106(v16):
            if v105 not in v21:
                return 2 + v45
    v26 = v109 if v110 >= 2 else v20['truth'] if v20['truth'] in v24 else v109
    return v113(v16, v17, v18, v26)

def _state_tensors(v27, v28, v29, v30, v31, v32, v16, v21, v33, v23, v34, v35, v18, v22):
    v17 = v31['tape']
    v36 = [v45 for v45 in v30.v301(v32).v36 if v45 != v34][-v255:]
    if not v36:
        return None
    v12 = v212.v114([v36], dtype=v212.v213, device=v35)
    v37, v54 = v115(v28, v29, v12, v34)
    v37 = v37[0, -1]
    v38 = [v31.v211('_sc', {}).v211(v105, 0.0) for v105 in v16]
    v39 = v157(v38) if v38 else 0.0
    v40 = v256(v38, reverse=True)[1] if v215(v38) > 1 else 0.0
    v41 = [v17.v128[v105] for v105 in v16] if v16 else []
    v42 = v157(v108(v41).v128()) / v215(v41) if v41 else 0.0
    v43 = v212.v114([v39, v39 - v40, v129(v215(v16)) / v157(1, v18), v129(v23) / v22, v129(v94(v33)), v42], device=v35, dtype=v37.v214)
    v44 = v212.v116(v27.v101, dtype=v212.v94, device=v35)
    v44[v209.v107] = True
    v44[v209.v117] = v94(v33)
    for v45 in v118(v215(v16)):
        v44[2 + v45] = v16[v45] not in v21
        v44[2 + v18 + v45] = True
    v44[-1] = True
    if v16:
        v119 = v108(v41)
        v120 = v157(v38) if v38 and v157(v38) > 0 else 1.0
        v121 = [[v38[v45] / v120, v119[v41[v45]] / v215(v16), 1.0 if v16[v45] in v21 else 0.0] for v45 in v118(v215(v16))]
        v103 = v212.v114(v121, device=v35, dtype=v37.v214)
    else:
        v103 = None
    v46 = v27(v37, v43, v44, v103)
    return (v46, v44)

def _apply(v47, *, v20, v31, v48, v32, v16, v33, v21, v23, v49, v50, v18, v27):
    v17, v122, v123 = (v31['tape'], v31['postings'], v31['idf'])
    if v47 in (v209.v107, v209.v117):
        v124 = v48 if v47 == v209.v107 else v33
        v16, v125 = v209.v216(v124, v122, v123, v18)
        v31['_sc'] = v125
        return (v32, v16, v33, v21, v23, v49, v50, False)
    if v47 == v27.v101 - 1:
        return (v32, v16, v33, v21, v23, v49, v50, True)
    if v47 < 2 + v18:
        v45 = v47 - 2
        if v45 >= v215(v16):
            return (v32, v16, v33, v21, v23, v49, v50, True)
        v126 = v16[v45]
        v127 = v31['texts'][v126]
        v32 = (v32 + ' | ' + v127)[-2000:]
        v33 = v131(v127, exclude=v17.v128[v126])
        v21 = v133(v21) | {v126}
        return (v32, v16, v33, v21, v23 + 1, v49, v50, False)
    v45 = v47 - 2 - v18
    if v45 >= v215(v16):
        return (v32, v16, v33, v21, v23, v49, v50, True)
    v49 = v17.v128[v16[v45]]
    v50 = 1.0 if v49 == v20['truth'] else 0.0
    return (v32, v16, v33, v21, v23, v49, v50, True)

def bc_episode(v27, v28, v29, v30, v31, v20, v34, v35, *, v18, v22, v51, v52: v129=5.0):
    """Teacher-forced CE. ANSWER after READ is upweighted — that step was stuck on ANSWER_0."""
    v17 = v31['tape']
    v53 = v209.v217.v130(S=v20['S'])
    v48 = v131(v53)
    v32 = v53
    v16: v132[v9] = []
    v33: v132[v95] = []
    v21: v133[v9] = v133()
    v23, v49, v50 = (0, None, 0.0)
    v55, v134, v135 = ([], [], [])
    for v54 in v118(v22):
        v136 = v218(v27, v28, v29, v30, v31, v32, v16, v21, v33, v23, v34, v35, v18, v22)
        if v136 is None:
            break
        v46, v54 = v136
        v47 = v219(v20, v16, v21, v17, v18, v22, v23)
        if not v212.v283(v46[v47]) or v46[v47] < -100000000.0:
            break
        v137 = 2 <= v47 < 2 + v18
        v138 = 2 + v18 <= v47 < 2 + 2 * v18
        if v137:
            v220 = 1.0
        elif v138 and v23 > 0:
            v220 = v129(v52)
        elif v138:
            v220 = 2.5
        else:
            v220 = 2.0
        v55.v221(v284.v257(v46.v285(0), v212.v114([v47], device=v35)))
        v134.v221(v220)
        v135.v221(v209.v245(v18)[v47])
        v32, v16, v33, v21, v23, v49, v50, v139 = v222(v47, item=v20, pack=v31, qwords=v48, transcript=v32, cands=v16, last_read_words=v33, seen_reads=v21, n_reads=v23, answered=v49, reward=v50, k=v18, policy=v27)
        if v139:
            break
    v50 -= v51 * v23
    if v55:
        v140 = v212.v114(v134, device=v35, dtype=v55[0].v214)
        v141 = (v212.v304(v55) * v140).v258() / v140.v258()
    else:
        v141 = v212.v116((), device=v35)
    return {'loss': v141, 'reward': v50, 'correct': v9(v49 == v20['truth']), 'n_reads': v23, 'trace': v135, 'kind': v20.v211('kind'), 'answer_is_slot': v49 is None or v49 in v133(v17.v128)}

def run_episode(v27, v28, v29, v30, v31, v20, v34, v35, *, v18, v22, v51, v56=True):
    """Policy rollout with 272 feats (for eval / optional RL)."""
    v17 = v31['tape']
    v53 = v209.v217.v130(S=v20['S'])
    v48 = v131(v53)
    v32 = v53
    v16: v132[v9] = []
    v33: v132[v95] = []
    v21: v133[v9] = v133()
    v142, v143, v135 = ([], [], [])
    v23, v49, v50 = (0, None, 0.0)
    for v54 in v118(v22):
        v136 = v218(v27, v28, v29, v30, v31, v32, v16, v21, v33, v23, v34, v35, v18, v22)
        if v136 is None:
            break
        v46, v54 = v136
        v144 = v212.v259.v223(logits=v46)
        v47 = v9(v46.v286()) if v56 else v9(v144.v287())
        v142.v221(v144.v260(v212.v114(v47, device=v35)))
        v143.v221(v144.v261())
        v135.v221(v209.v245(v18)[v47])
        v32, v16, v33, v21, v23, v49, v50, v139 = v222(v47, item=v20, pack=v31, qwords=v48, transcript=v32, cands=v16, last_read_words=v33, seen_reads=v21, n_reads=v23, answered=v49, reward=v50, k=v18, policy=v27)
        if v139:
            break
    v50 -= v51 * v23
    return {'logps': v142, 'entropy': v143, 'reward': v50, 'correct': v9(v49 == v20['truth']), 'answered': v49, 'n_reads': v23, 'trace': v135, 'answer_is_slot': v49 is None or v49 in v133(v17.v128), 'kind': v20.v211('kind')}

def main() -> v9:
    v57 = v224.v145()
    v57.v146('--smoke', action='store_true')
    v57.v146('--bc-episodes', type=v9, default=0)
    v57.v146('--rl-episodes', type=v9, default=0)
    v57.v146('--tape-period', type=v9, default=0)
    v57.v146('--clean', type=v9, default=6)
    v57.v146('--lying', type=v9, default=6)
    v57.v146('--witnesses', type=v9, default=5)
    v57.v146('--liars', type=v9, default=2)
    v57.v146('--distractor-slots', type=v9, default=0)
    v57.v146('--topk', type=v9, default=4)
    v57.v146('--max-steps', type=v9, default=6)
    v57.v146('--read-cost', type=v129, default=0.02)
    v57.v146('--entropy-bonus', type=v129, default=0.01)
    v57.v146('--answer-after-read-weight', type=v129, default=5.0, help='CE weight for ANSWER_* steps that follow a READ (majority index)')
    v57.v146('--lr-policy', type=v129, default=0.001)
    v57.v146('--lr-upper', type=v129, default=3e-05)
    v57.v146('--frozen-trunk', action='store_true')
    v58 = v57.v147()
    global LOG_PATH
    v148, v149, v7 = v150(v58.v151)
    v7.v195.v97(parents=True, exist_ok=True)
    v7.v152('', encoding='utf-8')
    v35 = v212.v35('cuda' if v212.v288.v262() else 'cpu')
    v59 = v225.v153(v6)
    v212.v154(v6)
    v60 = v155.v155()
    v61 = v58.v156 or (400 if v58.v188 else 4000)
    v62 = v157(0, v58.v158)
    v63 = v58.v63 or (50 if v58.v188 else 200)
    v64 = v58.v159 or (150 if v58.v188 else 1000)
    v18 = v58.v65
    v66 = 'none' if v58.v151 else 'upper'
    v160(f'Stage272 BC-controller start {v306.v296(v307.v297).v247()} device={v35} bc={v61} rl={v62} tape_period={v63} clean={v58.v263} lying={v58.v264} wit={v58.v190} liars={v58.v191} k={v18} mode={v66}')
    v54, v54, v161, v162 = v163()
    v30 = v226.v164(v95(v265.v227))
    v67 = v30.v165()
    v34 = v30.v228(v229) or 0
    v29 = v289.v266(v30, v161, v34, v67).v166(v35)
    v68 = v3 if v3.v230() else v2
    v28 = v267(v162, v67).v166(v35)
    v28.v167(v212.v268(v68, map_location=v35, weights_only=False)['model'])
    v231.v168(v28, v66)
    v69 = v209.v169(v28)
    v70 = v267(v162, v67).v166(v35)
    v70.v167(v212.v268(v2, map_location=v35, weights_only=False)['model'])
    v70.v170()
    for v71 in v70.v171():
        v71.v232(False)
    v72 = v172(v70, v161, v35)
    v160(f'  trunk={v68.v269} mode={v66} fp_version={v209.v246()} arc={v69[:12]}…')
    with v5.v196('r', encoding='utf-8', errors='ignore') as v98:
        v173 = v98.v233(1500000 if v58.v188 else 8000000)
    v73 = v132(v270.v234((v13.v290(1) for v13 in v308.v302(v173) if v215(v13.v290(1)) >= 5)))
    v59.v174(v73)
    v74 = [v272.v271() for v272 in v173.v291('\n') if v215(v272.v271()) >= 60][:400 if v58.v188 else 6000]
    v27 = v175(2 * (v28.v292.v273 // 2), v18, v35)
    v75 = [v71 for v71 in v28.v171() if v71.v235]
    v76 = v212.v236.v176([{'params': v27.v171(), 'lr': v58.v293}] + ([{'params': v75, 'lr': v58.v303}] if v75 else []), weight_decay=0.01)
    v77: v133[v95] = v133()
    v31 = None
    v78 = 0.0
    v79 = []

    def new_tape(v177):
        return v209.v237(bank=v72, tok=v30, pad_id=v34, device=v35, rng=v177, pool=v73, lines=v74, used=v77, n_clean=v58.v263, n_lying=v58.v264, n_wit=v58.v190, n_liars=v58.v191, n_dist=v64)
    v27.v178()
    v28.v178(v66 != 'none')
    for v80 in v118(1, v61 + 1):
        if v31 is None or (v80 - 1) % v63 == 0:
            v31 = v185(v59)
        v20 = v31['items'][v59.v274(v215(v31['items']))]
        v92 = v238(v27, v28, v29, v30, v31, v20, v34, v35, k=v18, max_steps=v58.v22, read_cost=v58.v51, answer_after_read_weight=v58.v52)
        v76.v239(set_to_none=True)
        v92['loss'].v240()
        v212.v15.v275.v241(v132(v27.v171()) + v75, 1.0)
        v76.v242()
        if v80 % v157(1, v61 // 10) == 0:
            v79.v221({'phase': 'bc', 'episode': v80, 'loss': v129(v92['loss']), 'reward': v92['reward'], 'trace': v92['trace']})
            v160(f"  bc {v80}/{v61} loss={v129(v92['loss']):.3f} last_trace={v92['trace']} ({v155.v155() - v60:.0f}s)")
    for v80 in v118(1, v62 + 1):
        if v31 is None or (v80 - 1) % v63 == 0:
            v31 = v185(v59)
        v20 = v31['items'][v59.v274(v215(v31['items']))]
        v92 = v243(v27, v28, v29, v30, v31, v20, v34, v35, k=v18, max_steps=v58.v22, read_cost=v58.v51, greedy=False)
        if not v92['logps']:
            continue
        v78 = 0.99 * v78 + 0.01 * v92['reward']
        v179 = v92['reward'] - v78
        v180 = v212.v304(v92['entropy']).v258() if v92['entropy'] else v212.v116((), device=v35)
        v141 = -v179 * v212.v304(v92['logps']).v258() - v58.v189 * v180
        v76.v239(set_to_none=True)
        v141.v240()
        v212.v15.v275.v241(v132(v27.v171()) + v75, 1.0)
        v76.v242()
        if v80 % v157(1, v62 // 10) == 0 or v80 == v62:
            v79.v221({'phase': 'rl', 'episode': v80, 'baseline': v78, 'reward': v92['reward'], 'trace': v92['trace']})
            v160(f"  rl {v80}/{v62} baseline={v78:.3f} last_trace={v92['trace']} ({v155.v155() - v60:.0f}s)")
    v27.v170()
    v28.v170()
    v81 = v209.v169(v28)

    @v212.v183()
    def evaluate(v71):
        v181 = {'clean': [], 'lying': [], 'reads': [], 'reads_clean': [], 'reads_lying': [], 'slot_ok': [], 'lookup': {'clean': [], 'lying': []}, 'major': {'clean': [], 'lying': []}, 'traces': []}
        for v182 in v71['items']:
            v244 = v243(v27, v28, v29, v30, v71, v182, v34, v35, k=v18, max_steps=v58.v22, read_cost=v58.v51, greedy=True)
            v181[v182['kind']].v221(v244['correct'])
            v181['reads'].v221(v244['n_reads'])
            v181[f"reads_{v182['kind']}"].v221(v244['n_reads'])
            v181['slot_ok'].v221(v9(v244['answer_is_slot']))
            v181['lookup'][v182['kind']].v221(v209.v294(v71, v182, v18))
            v181['major'][v182['kind']].v221(v209.v295(v71, v182, v18))
            v181['traces'].v221({'kind': v182['kind'], 'trace': v244['trace'], 'correct': v244['correct']})
        v13 = lambda v276: v129(v309.v305(v276)) if v276 else v129('nan')
        return {'policy_clean': v13(v181['clean']), 'policy_lying': v13(v181['lying']), 'lookup_clean': v13(v181['lookup']['clean']), 'lookup_lying': v13(v181['lookup']['lying']), 'majority_lying': v13(v181['major']['lying']), 'mean_reads': v13(v181['reads']), 'mean_reads_clean': v13(v181['reads_clean']), 'mean_reads_lying': v13(v181['reads_lying']), 'answer_is_slot': v13(v181['slot_ok']), 'n': v215(v71['items']), 'traces': v181['traces']}
    v82 = v184(v31)
    v83 = v185(v225.v153(v6 + 99))
    v84 = v184(v83)
    v160(f'  TRAIN {v278.v248(v82)}')
    v160(f'  NOVEL {v278.v248(v84)}')
    v85 = v69 == v81
    v86 = v84['answer_is_slot'] >= 0.99
    v87 = v84['policy_lying'] >= v84['lookup_lying'] + 0.1
    v88 = v84['policy_lying'] >= v84['majority_lying'] - 0.05
    v89 = v84['policy_clean'] >= 0.7
    v90 = v84['policy_lying'] >= v82['policy_lying'] - 0.1
    v91 = v84['mean_reads'] <= v58.v22 * 0.6
    if not (v85 and v86):
        v186 = 'CONTROLLER_INVALID'
    elif v87 and v89 and v90:
        v186 = 'CONTROLLER_OK'
    elif v87 or v89:
        v186 = 'CONTROLLER_PARTIAL'
    else:
        v186 = 'CONTROLLER_NO'
    v212.v187({'policy': v27.v277(), 'model': v28.v277(), 'stage': 272, 'arc_enc_hash': v81}, v4)
    v92 = {'stage': 272, 'overall': v186, 'frozen_trunk': v58.v151, 'trunk_mode': v66, 'smoke': v58.v188, 'seed': v6, 'bc_episodes': v61, 'rl_episodes': v62, 'tape_period': v63, 'actions': v209.v245(v18), 'topk': v18, 'max_steps': v58.v22, 'read_cost': v58.v51, 'entropy_bonus': v58.v189, 'answer_after_read_weight': v58.v52, 'witnesses': v58.v190, 'liars': v58.v191, 'fp_version': v209.v246(), 'used_pool_final': v215(v77), 'gates': {'G_arc_enc_frozen': v85, 'G_answer_is_slot': v86, 'G_beats_lookup': v87, 'G_beats_majority': v88, 'G_clean_kept': v89, 'G_novel_tape': v90, 'G_reads_economical': v91}, 'train_tape': v82, 'novel_tape': v84, 'arc_enc_hash_before': v69, 'arc_enc_hash_after': v81, 'curve': v79, 'note': 'Same lexical controller as 271, taught by BC: clean ASK→ANSWER; lying one READ then ANSWER visible majority. CE weight on ANSWER-after-READ (default 5) targets the ANSWER_0 collapse. max_agree feat, no re-read mask. REINFORCE opt-in.', 'timestamp': v306.v296(v307.v297).v247(), 'wall_s': v155.v155() - v60}
    v1.v97(parents=True, exist_ok=True)
    v148.v152(v278.v248(v92, indent=2), encoding='utf-8')
    v149.v152(f"# Stage 272 BC controller{(' (frozen trunk)' if v58.v151 else '')}\n\n**{v186}** · bc={v61} rl={v62} · actions={v215(v209.v245(v18))}{(' · SMOKE' if v58.v188 else '')}\n\n| arm | clean | lying |\n|---|---:|---:|\n| policy (novel tape) | **{v84['policy_clean']:.3f}** | **{v84['policy_lying']:.3f}** |\n| fixed lookup | {v84['lookup_clean']:.3f} | {v84['lookup_lying']:.3f} |\n| fixed majority | — | {v84['majority_lying']:.3f} |\n\n- mean reads {v84['mean_reads']:.2f} of {v58.v22} (clean {v84['mean_reads_clean']:.2f} / lying {v84['mean_reads_lying']:.2f})\n- train tape lying {v82['policy_lying']:.3f} → novel {v84['policy_lying']:.3f}\n\n## Gates\n\n" + ''.v279((f'- {v310}: **{v311}**\n' for v310, v311 in v92['gates'].v312())), encoding='utf-8')
    v160(v278.v248({'overall': v186, 'gates': v92['gates']}, indent=2))
    return 0
if v93 == '__main__':
    raise v192(v249())