"""
Stage 271 — The controller: one policy over the actions the project already built.

Everything from 264 to 270 is an action, not a system. Votes retrieve (264), the loop reads and
re-asks (267), span-lock emits a value verbatim (265), ACT halts (257), a lying tape needs several
slots weighed (270). Each was measured alone, and each time the trained part was a projection that
lost to something with no parameters. What was never built is the thing that chooses among them.

So the mind here is a policy over a small, discrete, fully-lexical action set:

    ASK_Q          retrieve with the question's own words
    ASK_READ       retrieve with the words of what was last read
    READ_i         read candidate i — its slot text enters the transcript
    ANSWER_i       answer with candidate i's value, verbatim
    STOP           refuse

Nothing in that list is a vector. State is the transcript of what has been read, plus five scalar
retrieval features. The weights hold one thing: which action next, and — through ANSWER_i — whom
to believe when slots disagree. No fact can live there: the tape is rebuilt every episode, so
between two episodes not one value survives, and the answer must be a slot's value copied out, so
the weights cannot author content even if they wanted to.

Episodes mix two task families on purpose, because a mind that can only aggregate is not a mind:

    clean   one witness per subject           — the policy should ASK once and ANSWER
    lying   several witnesses, one shared lie — ANSWER_0 is wrong, reading is required

If the controller learns a single fixed habit it will lose on one family or the other, and the
per-family split in the decision says which.

Reward is correctness minus reads, so "when to stop" is learned rather than fixed as in 257.
Trained with REINFORCE and a running baseline — no critic, because the action space is five wide
and the episode is under ten steps.

Gates:
  G_beats_lookup     against the fixed policy ASK_Q then ANSWER_0 (what 270 measured)
  G_beats_majority   on the lying family, against unweighted majority over retrieved witnesses
  G_novel_tape       a tape whose entities never appeared in training
  G_reads_economical  fewer reads than the read-everything policy at equal accuracy
  G_arc_enc_frozen   hash unchanged
  G_answer_is_slot   every answer is some slot's value, verbatim — true by construction, asserted

  python _stage271_controller.py --smoke
  python _stage271_controller.py                    # night
  python _stage271_controller.py --frozen-trunk     # paired control, policy head only
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
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, TapeView, hidden_and_logits
from _tape_index import context_words
v0 = v12('results')
v1 = v12('checkpoints/stage191_p1_curve.pt')
v2 = v12('checkpoints/stage253_joint_l02.pt')
v3 = v12('checkpoints/stage271_controller.pt')
v4 = v12('data/_wikitext103_train.txt')
v5 = 271
v6 = '{S} was appointed director of'
v7 = ('{S} was appointed director of {V} in the regional chronicle of 1987 .', 'The county register lists {S} as appointed director of {V} that year .', 'According to the parish record , {S} was appointed director of {V} .', '{S} , appointed director of {V} , appears in the 1987 civil roll .', 'A ledger entry names {S} as the appointed director of {V} .')

def paths(v13: v98):
    v14 = '_frozen' if v13 else ''
    return (v0 / f'stage271_decision{v14}.json', v0 / f'stage271_mini{v14}.md', v0 / f'_stage271_log{v14}.txt')
v8 = v0 / '_stage271_log.txt'

def log(v15: v9) -> None:
    v16 = v15 if v15.v204('\n') else v15 + '\n'
    try:
        v205(v16, end='', flush=True)
    except v99:
        v205(v16.v336('ascii', 'replace').v303('ascii'), end='', flush=True)
    v8.v206.v100(parents=True, exist_ok=True)
    with v8.v207('a', encoding='utf-8') as v101:
        v101.v208(v16)

def arc_enc_hash(v17: v102) -> v9:
    v18 = v209.v103()
    for v64, v14 in v104(v17.v324.v300().v114()):
        v18.v210(v14.v347().v346().v343().v325().v267())
    return v18.v105()

def fp_version() -> v9:
    v19 = v106(v107, 'canonical_fp_version', None)
    return v9(v19()) if v211(v19) else v1.v108

def build_episode_tape(*, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32):
    v33 = [v47 for v47 in v25 if v47 not in v27 and v217(v47) >= 5]
    v24.v109(v33)
    v34 = [v47 for v47 in v268(v128(v27) | v128(v33), v24, v28 + v29 + 60) if v217(v47) >= 5 and v47 not in v27]
    v34 = v110(v129.v212(v34))
    v35 = v28 + v29 * 2
    if v217(v34) < v28 + v29 or v217(v33) < v35:
        raise v213(f'pool exhausted: subs={v217(v34)} vals={v217(v33)} need={v35}')
    v111, v112, v113, v114 = ([], [], [], [])
    v36 = 0

    def add(v115, v116, v117):
        v118 = v7[v117 % v217(v7)].v144(S=v115, V=v116)
        v119 = v20.v214(v118, exclude=v116)
        v120 = v20.v269([v115])[0]
        v111.v215(v326.v304(v120 + v119, dim=-1) if v119 is not None else v120)
        v112.v215(v116)
        v113.v215(v118)
        return v217(v112) - 1
    for v37 in v121(v28):
        v115 = v34[v37]
        v122 = v33[v36]
        v36 += 1
        v123 = v216(v115, v122, 0)
        v114.v215({'S': v115, 'truth': v122, 'slots': [v123], 'kind': 'clean'})
        v27.v216(v122)
        v27.v216(v115)
    for v37 in v121(v29):
        v115 = v34[v28 + v37]
        v124 = v33[v36]
        v36 += 1
        v125 = v33[v36]
        v36 += 1
        v126 = [v124] * (v30 - v31) + [v125] * v31
        v24.v109(v126)
        v127 = [v216(v115, v122, v270) for v270, v122 in v132(v126)]
        v114.v215({'S': v115, 'truth': v124, 'lie': v125, 'slots': v127, 'kind': 'lying'})
        v27.v216(v124)
        v27.v216(v125)
        v27.v216(v115)
    v38 = v128(v112)
    v39 = v217(v112) + v32
    for v40 in v26:
        if v217(v112) >= v39:
            break
        for v15 in v271.v218(v40):
            v219 = v15.v272(1)
            if v217(v219) < 5 or v219 in v38:
                continue
            v273, v274 = (v281(0, v15.v337() - 120), v305(v217(v40), v15.v338() + 120))
            v119 = v20.v214(v40[v273:v274], exclude=v219)
            if v119 is None:
                continue
            v220 = [v47 for v47 in v339.v327(v40[v273:v15.v337()]) if v47 != v219]
            if not v220:
                continue
            v111.v215(v326.v304(v20.v269([v220[-1]])[0] + v119, dim=-1))
            v112.v215(v219)
            v113.v215(v40[v273:v274])
            v38.v216(v219)
            if v217(v112) >= v39:
                break
    v41: v129[v9, v110[v11]] = v130(v110)
    for v131, (v122, v14) in v132(v221(v112, v113)):
        for v47 in v145(v14, exclude=v122):
            v41[v47].v215(v131)
    v42 = {v47: 1.0 / v306.v172(2.0 + v217(v41[v47])) for v47 in v41}
    v43 = v133(v243.v307(v111, 0).v178(v23), v112, v21, v22)
    return {'tape': v43, 'texts': v113, 'items': v114, 'postings': v41, 'idf': v42}

def vote(v44, v41, v42, v45):
    v46: v129[v11, v134] = v130(v134)
    for v47 in v44:
        for v131 in v41.v222(v47, ()):
            v46[v131] += v42.v222(v47, 0.0)
    return ([v131 for v131, v64 in v104(v46.v114(), key=lambda v345: -v345[1])[:v45]], v46)

class Policy(v48.v10):
    """h(transcript) + 5 global scalars -> action logits, plus a per-candidate scorer.

    The first smoke settled on ASK_Q then ANSWER_0 and never read anything, and that was not the
    read cost: with only global features, ANSWER_i can be chosen by position alone, so agreement
    between witnesses is not representable and reading cannot pay. `agreement` — the share of
    retrieved candidates carrying the same value as candidate i — is what makes majority a
    function the head can express at all.
    """

    def __init__(v135, v136: v11, v45: v11, v23):
        v308().v223()
        v135.v45 = v45
        v135.v137 = 2 + 2 * v45 + 1
        v135.v101 = v48.v309(v48.v328(v136 + 5, 128), v48.v329(), v48.v328(128, v135.v137)).v178(v23)
        v48.v275.v224(v135.v101[-1].v225)
        v48.v275.v224(v135.v101[-1].v226)
        v135.v138 = v48.v309(v48.v328(v136 + 5 + 3, 64), v48.v329(), v48.v328(64, 2)).v178(v23)
        v48.v275.v224(v135.v138[-1].v225)
        v48.v275.v224(v135.v138[-1].v226)

    def forward(v135, v18, v139, v140, v141=None):
        v142 = v243.v227([v18, v139], dim=-1)
        v143 = v135.v101(v142)
        if v141 is not None and v141.v276():
            v228 = v141.v277(0)
            v229 = v243.v227([v142.v344(0).v330(v228, -1), v141], dim=-1)
            v230 = v135.v138(v229)
            v231 = v243.v278(2, 2 + v228, device=v143.v23)
            v232 = v243.v278(2 + v135.v45, 2 + v135.v45 + v228, device=v143.v23)
            v143 = v143.v279(0, v231, v230[:, 0])
            v143 = v143.v279(0, v232, v230[:, 1])
        return v143.v233(~v140, -1000000000.0)
v49, v50 = (0, 1)

def act_names(v45):
    return ['ASK_Q', 'ASK_READ'] + [f'READ_{v37}' for v37 in v121(v45)] + [f'ANSWER_{v37}' for v37 in v121(v45)] + ['STOP']

def episode(v51, v17, v52, v21, v53, v54, v22, v23, *, v45, v55, v56, v57=False):
    """One question. Returns logprobs, reward, and a trace of what the policy did."""
    v43, v41, v42 = (v53['tape'], v53['postings'], v53['idf'])
    v58 = v6.v144(S=v54['S'])
    v59 = v145(v58)
    v60 = v58
    v61: v110[v11] = []
    v62: v110[v9] = []
    v63: v128[v11] = v128()
    v146, v147, v148 = ([], [], [])
    v65, v149, v150 = (0.0, 0, None)
    for v64 in v121(v55):
        v151 = [v37 for v37 in v21.v336(v60).v151 if v37 != v22][-v310:]
        if not v151:
            break
        v14 = v243.v234([v151], dtype=v243.v280, device=v23)
        v18, v64 = v235(v17, v52, v14, v22)
        v18 = v18[0, -1]
        v152 = [v53.v222('_sc', {}).v222(v119, 0.0) for v119 in v61]
        v153 = v281(v152) if v152 else 0.0
        v154 = v104(v152, reverse=True)[1] if v217(v152) > 1 else 0.0
        v139 = v243.v234([v153, v153 - v154, v134(v217(v61)) / v281(1, v45), v134(v149) / v55, v134(v98(v62))], device=v23, dtype=v18.v282)
        v140 = v243.v236(v51.v137, dtype=v243.v98, device=v23)
        v140[v49] = True
        v140[v50] = v98(v62)
        for v37 in v121(v217(v61)):
            v140[2 + v37] = True
            v140[2 + v45 + v37] = True
        v140[-1] = True
        if v61:
            v237 = [v43.v286[v119] for v119 in v61]
            v238 = v158(v237)
            v239 = v281(v152) if v152 and v281(v152) > 0 else 1.0
            v240 = []
            for v37 in v121(v217(v61)):
                v240.v215([v152[v37] / v239, v238[v237[v37]] / v217(v61), 1.0 if v61[v37] in v63 else 0.0])
            v141 = v243.v234(v240, device=v23, dtype=v18.v282)
        else:
            v141 = None
        v143 = v51(v18, v139, v140, v141)
        v155 = v243.v283.v241(logits=v143)
        v156 = v11(v143.v311()) if v57 else v11(v155.v312())
        v146.v215(v155.v284(v243.v234(v156, device=v23)))
        v147.v215(v155.v285())
        v148.v215(v262(v45)[v156])
        if v156 in (v49, v50):
            v44 = v59 if v156 == v49 else v62
            v61, v46 = v157(v44, v41, v42, v45)
            v53['_sc'] = v46
        elif v156 == v51.v137 - 1:
            break
        elif v156 < 2 + v45:
            v37 = v156 - 2
            if v37 >= v217(v61):
                break
            v313 = v61[v37]
            v314 = v53['texts'][v313]
            v60 = (v60 + ' | ' + v314)[-2000:]
            v62 = v145(v314, exclude=v43.v286[v313])
            v63.v216(v313)
            v149 += 1
        else:
            v37 = v156 - 2 - v45
            if v37 >= v217(v61):
                break
            v150 = v43.v286[v61[v37]]
            v65 = 1.0 if v150 == v54['truth'] else 0.0
            break
    v65 -= v56 * v149
    return {'logps': v146, 'entropy': v147, 'reward': v65, 'correct': v11(v150 == v54['truth']), 'answered': v150, 'n_reads': v149, 'trace': v148, 'answer_is_slot': v150 is None or v150 in v128(v43.v286), 'kind': v54.v222('kind')}

def fixed_lookup(v53, v54, v45):
    v61, v64 = v157(v145(v6.v144(S=v54['S'])), v53['postings'], v53['idf'], v45)
    return v11(v98(v61) and v53['tape'].v286[v61[0]] == v54['truth'])

def fixed_majority(v53, v54, v45):
    v61, v64 = v157(v145(v6.v144(S=v54['S'])), v53['postings'], v53['idf'], v45)
    v66 = [v119 for v119 in v61 if v119 in v128(v54['slots'])]
    if not v66:
        return 0
    v67 = v158((v53['tape'].v286[v119] for v119 in v66))
    return v11(v67.v331(1)[0][0] == v54['truth'])

def main() -> v11:
    v68 = v242.v159()
    v68.v160('--smoke', action='store_true')
    v68.v160('--episodes', type=v11, default=0)
    v68.v160('--tape-period', type=v11, default=0)
    v68.v160('--clean', type=v11, default=6)
    v68.v160('--lying', type=v11, default=6)
    v68.v160('--witnesses', type=v11, default=5)
    v68.v160('--liars', type=v11, default=2, help='3-vs-2 keeps lookup near 0.6 while majority stays 1.0; 1-of-4 left only 0.167 of headroom and the policy took the cheap habit')
    v68.v160('--distractor-slots', type=v11, default=0)
    v68.v160('--topk', type=v11, default=4)
    v68.v160('--max-steps', type=v11, default=6)
    v68.v160('--read-cost', type=v134, default=0.02)
    v68.v160('--entropy-bonus', type=v134, default=0.01)
    v68.v160('--lr-policy', type=v134, default=0.001)
    v68.v160('--lr-upper', type=v134, default=3e-05)
    v68.v160('--frozen-trunk', action='store_true', help='policy head only (paired control)')
    v69 = v68.v161()
    global LOG_PATH
    v162, v163, v8 = v164(v69.v165)
    v8.v206.v100(parents=True, exist_ok=True)
    v8.v166('', encoding='utf-8')
    v23 = v243.v23('cuda' if v243.v315.v287() else 'cpu')
    v24 = v244.v167(v5)
    v243.v168(v5)
    v70 = v169.v169()
    v71 = v69.v170 or (300 if v69.v199 else 6000)
    v72 = v69.v72 or (50 if v69.v199 else 200)
    v32 = v69.v171 or (150 if v69.v199 else 1000)
    v45 = v69.v73
    v74 = 'none' if v69.v165 else 'upper'
    v172(f'Stage271 controller start {v334.v322(v335.v323).v264()} device={v23} episodes={v71} tape_period={v72} clean={v69.v288} lying={v69.v289} wit={v69.v201} liars={v69.v202} k={v45} mode={v74}')
    v64, v64, v173, v174 = v175()
    v21 = v245.v176(v9(v290.v246))
    v75 = v21.v177()
    v22 = v21.v247(v248) or 0
    v52 = v316.v291(v21, v173, v22, v75).v178(v23)
    v76 = v2 if v2.v249() else v1
    v17 = v102(v174, v75).v178(v23)
    v17.v179(v243.v292(v76, map_location=v23, weights_only=False)['model'])
    v250.v180(v17, v74)
    v77 = v181(v17)
    v78 = v102(v174, v75).v178(v23)
    v78.v179(v243.v292(v1, map_location=v23, weights_only=False)['model'])
    v78.v182()
    for v79 in v78.v183():
        v79.v251(False)
    v20 = v184(v78, v173, v23)
    v172(f'  trunk={v76.v108} mode={v74} fp_version={v263()} arc={v77[:12]}…')
    with v4.v207('r', encoding='utf-8', errors='ignore') as v101:
        v185 = v101.v252(1500000 if v69.v199 else 8000000)
    v25 = v110(v129.v212((v15.v272(1) for v15 in v271.v218(v185) if v217(v15.v272(1)) >= 5)))
    v24.v109(v25)
    v26 = [v294.v293() for v294 in v185.v317('\n') if v217(v294.v293()) >= 60][:400 if v69.v199 else 6000]
    v51 = v186(2 * (v17.v318.v295 // 2), v45, v23)
    v80 = [v79 for v79 in v17.v183() if v79.v253]
    v81 = v243.v254.v187([{'params': v51.v183(), 'lr': v69.v319}] + ([{'params': v80, 'lr': v69.v332}] if v80 else []), weight_decay=0.01)
    v27: v128[v9] = v128()
    v53 = None
    v82 = 0.0
    v83 = []

    def new_tape(v188):
        return v255(bank=v20, tok=v21, pad_id=v22, device=v23, rng=v188, pool=v25, lines=v26, used=v27, n_clean=v69.v288, n_lying=v69.v289, n_wit=v69.v201, n_liars=v69.v202, n_dist=v32)
    for v84 in v121(1, v71 + 1):
        if v53 is None or (v84 - 1) % v72 == 0:
            v53 = v196(v24)
        v54 = v53['items'][v24.v296(v217(v53['items']))]
        v96 = v256(v51, v17, v52, v21, v53, v54, v22, v23, k=v45, max_steps=v69.v55, read_cost=v69.v56)
        if not v96['logps']:
            continue
        v82 = 0.99 * v82 + 0.01 * v96['reward']
        v189 = v96['reward'] - v82
        v190 = v243.v307(v96['entropy']).v297() if v96['entropy'] else v243.v236((), device=v23)
        v191 = -v189 * v243.v307(v96['logps']).v297() - v69.v200 * v190
        v81.v257(set_to_none=True)
        v191.v258()
        v243.v48.v298.v259(v110(v51.v183()) + v80, 1.0)
        v81.v260()
        if v84 % v281(1, v71 // 10) == 0:
            v83.v215({'episode': v84, 'baseline': v82, 'reward': v96['reward'], 'trace': v96['trace']})
            v172(f"  ep {v84}/{v71} baseline={v82:.3f} last_trace={v96['trace']} ({v169.v169() - v70:.0f}s)")
    v51.v182()
    v17.v182()
    v85 = v181(v17)

    @v243.v194()
    def evaluate(v79):
        v192 = {'clean': [], 'lying': [], 'reads': [], 'slot_ok': [], 'lookup': {'clean': [], 'lying': []}, 'major': {'clean': [], 'lying': []}}
        for v193 in v79['items']:
            v261 = v256(v51, v17, v52, v21, v79, v193, v22, v23, k=v45, max_steps=v69.v55, read_cost=v69.v56, greedy=True)
            v192[v193['kind']].v215(v261['correct'])
            v192['reads'].v215(v261['n_reads'])
            v192['slot_ok'].v215(v11(v261['answer_is_slot']))
            v192['lookup'][v193['kind']].v215(v320(v79, v193, v45))
            v192['major'][v193['kind']].v215(v321(v79, v193, v45))
        v15 = lambda v299: v134(v340.v333(v299)) if v299 else v134('nan')
        return {'policy_clean': v15(v192['clean']), 'policy_lying': v15(v192['lying']), 'lookup_clean': v15(v192['lookup']['clean']), 'lookup_lying': v15(v192['lookup']['lying']), 'majority_lying': v15(v192['major']['lying']), 'mean_reads': v15(v192['reads']), 'answer_is_slot': v15(v192['slot_ok']), 'n': v217(v79['items'])}
    v86 = v195(v53)
    v87 = v196(v244.v167(v5 + 99))
    v88 = v195(v87)
    v172(f'  TRAIN {v301.v265(v86)}')
    v172(f'  NOVEL {v301.v265(v88)}')
    v89 = v77 == v85
    v90 = v88['answer_is_slot'] >= 0.99
    v91 = v88['policy_lying'] >= v88['lookup_lying'] + 0.1
    v92 = v88['policy_lying'] >= v88['majority_lying'] - 0.05
    v93 = v88['policy_clean'] >= 0.7
    v94 = v88['policy_lying'] >= v86['policy_lying'] - 0.1
    v95 = v88['mean_reads'] <= v69.v55 * 0.6
    if not (v89 and v90):
        v197 = 'CONTROLLER_INVALID'
    elif v91 and v93 and v94:
        v197 = 'CONTROLLER_OK'
    elif v91 or v93:
        v197 = 'CONTROLLER_PARTIAL'
    else:
        v197 = 'CONTROLLER_NO'
    v243.v198({'policy': v51.v300(), 'model': v17.v300(), 'stage': 271, 'arc_enc_hash': v85}, v3)
    v96 = {'stage': 271, 'overall': v197, 'frozen_trunk': v69.v165, 'trunk_mode': v74, 'smoke': v69.v199, 'seed': v5, 'episodes': v71, 'tape_period': v72, 'actions': v262(v45), 'topk': v45, 'max_steps': v69.v55, 'read_cost': v69.v56, 'entropy_bonus': v69.v200, 'witnesses': v69.v201, 'liars': v69.v202, 'fp_version': v263(), 'used_pool_final': v217(v27), 'gates': {'G_arc_enc_frozen': v89, 'G_answer_is_slot': v90, 'G_beats_lookup': v91, 'G_beats_majority': v92, 'G_clean_kept': v93, 'G_novel_tape': v94, 'G_reads_economical': v95}, 'train_tape': v86, 'novel_tape': v88, 'arc_enc_hash_before': v77, 'arc_enc_hash_after': v85, 'curve': v83, 'note': "One policy over five lexical actions: ASK with the question's words, ASK with what was just read, READ a candidate, ANSWER a candidate verbatim, STOP. No vector leaves the tape and no fact can live in the weights — the tape is rebuilt every tape_period episodes and the answer must be some slot's value. Clean and lying subjects share every bank on purpose: a policy that only aggregates loses the clean family, and the per-family split says which habit it settled into. Reward is correctness minus reads, so halting is learned rather than fixed as in 257. REINFORCE with a running baseline.", 'timestamp': v334.v322(v335.v323).v264(), 'wall_s': v169.v169() - v70}
    v0.v100(parents=True, exist_ok=True)
    v162.v166(v301.v265(v96, indent=2), encoding='utf-8')
    v163.v166(f"# Stage 271 controller{(' (frozen trunk)' if v69.v165 else '')}\n\n**{v197}** · episodes={v71} · actions={v217(v262(v45))}{(' · SMOKE' if v69.v199 else '')}\n\n| arm | clean | lying |\n|---|---:|---:|\n| policy (novel tape) | **{v88['policy_clean']:.3f}** | **{v88['policy_lying']:.3f}** |\n| fixed lookup | {v88['lookup_clean']:.3f} | {v88['lookup_lying']:.3f} |\n| fixed majority | — | {v88['majority_lying']:.3f} |\n\n- mean reads {v88['mean_reads']:.2f} of {v69.v55}\n- train tape lying {v86['policy_lying']:.3f} → novel {v88['policy_lying']:.3f}\n\n## Gates\n\n" + ''.v302((f'- {v341}: **{v342}**\n' for v341, v342 in v96['gates'].v114())), encoding='utf-8')
    v172(v301.v265({'overall': v197, 'gates': v96['gates']}, indent=2))
    return 0
if v97 == '__main__':
    raise v203(v266())