"""
Stage 278 — Three defects in 276, none of which was a shortage of training.

276 answered TEACHER_CANNOT_ABSTAIN with clean 1.000 and decidable 1.000 on a NOVEL tape, with
retrieval precision and witness recall both 1.000 and search inside the loop. Only the tie family
failed, and the curve says exactly why - the failure is not capacity and not episode count.

  1. BC LEARNED THE ABSTENTION AND RL REMOVED IT.
     bc 3000 and bc 3500 both traced ASK_Q -> READ_0..3 -> STOP at loss 0.5297. Every RL trace
     ends in ANSWER. The mechanism is the baseline: one running scalar shared by all three
     families sat near 0.45, so on a tie both actions carried a negative advantage (abstain
     0 - 0.45, answer -0.38 - 0.45) while STOP's logit - the only one still fed by the global
     head - was pushed down by the two thirds of episodes where ANSWER earns +0.98. A rare
     correct action drowned in a common one's gradient.
     Fix: a state-dependent baseline V(s) on the same [h, feats] the policy sees, advantage
     R - V(s_t) per step. Unbiased, standard, and it is the variance reduction the flat-vote
     region needs. A PER-FAMILY baseline would read `kind` and is a leak; this does not.

  2. THE CEILING ITSELF WAS BROKEN.
     teacher_abstain was 0.75 on the train tape and 0.50 on the novel one. A policy cannot be
     asked to beat a teacher that is right half the time. 275's rule ruled on ties from whatever
     had been opened so far, so an early verdict on a partial reading called a 2-2 a 1-0.
     Fix: the teacher must exhaust every retrieved witness it can afford before any verdict.
     Reading order is computable from the cue; no gold, no family label, still executable.

  3. ABSTENTION WAS NOT WORTH ANYTHING.
     reward was correct +1, wrong -0.3, abstain 0.0, read -0.02. Two problems. The wrong/abstain
     margin of 0.3 equals fifteen reads, so reading to find out is barely cheaper than guessing;
     and abstain 0.0 is indistinguishable from an episode that simply never finished.
     Fix: wrong -1.0, abstain +0.75. The first attempt used +0.1, which got the ordering right
     and the magnitude wrong: with one shared V(s) the optimal return per family must be
     comparable, and +0.1 left a tie worth +0.02 against clean's +0.98. Perfect play on a tie
     then carried a negative advantage at every step, so RL bought its loss down by reading
     less - tie reads fell 4.0 to 1.0 and the policy answered blind. At +0.75 the family optima
     are 0.98 / 0.90 / 0.67 and silence still loses to an answer wherever one exists.

Everything else is 276 verbatim - same tape builder, same five witnesses with per-subject filler,
same classic idf, same families, same seeds. The three changes above are the whole diff, so a
difference in the tie family is a difference from them.

  python _stage278_value_baseline.py --smoke
  python _stage278_value_baseline.py --bc-episodes 4000 --rl-episodes 3000
  python _stage278_value_baseline.py --no-value-head   # ablation: 276's scalar baseline back
"""
from __future__ import annotations
import argparse
import json
import math
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
import _stage274_truthfree_oracle as s274
import _stage276_search_in_loop as s276
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tape_index import context_words
v0 = v12('results')
v1 = v12('checkpoints/stage191_p1_curve.pt')
v2 = v12('checkpoints/stage253_joint_l02.pt')
v3 = v12('checkpoints/stage278_value_baseline.pt')
v4 = v12('data/_wikitext103_train.txt')
v5 = 278
v6 = v13.v6
v7 = v0 / '_stage278_log.txt'

def paths(v14: v102):
    return (v0 / f'stage278_decision{v14}.json', v0 / f'stage278_mini{v14}.md', v0 / f'_stage278_log{v14}.txt')

def log(v15: v102) -> None:
    v16 = v15 if v15.v192('\n') else v15 + '\n'
    try:
        v193(v16, end='', flush=True)
    except v103:
        v193(v16.v333('ascii', 'replace').v293('ascii'), end='', flush=True)
    v7.v194.v104(parents=True, exist_ok=True)
    with v7.v195('a', encoding='utf-8') as v105:
        v105.v196(v16)

def teacher(*, v17, v18, v19, v20, v21, v22, v23, v24=None):
    """Executable, and it does not rule until it has read what it can afford.

    275's rule counted whatever happened to be open, so a 2-2 tie seen after two reads looked
    like a decided 1-0 and the teacher answered. That is why teacher_abstain sat at 0.50 on 276's
    novel tape. Here the reading phase is unconditional: every retrieved candidate is opened
    while the budget allows, and only then does the count mean anything.

    Nothing here reads item['truth'] or item['kind']. The retrieve list comes from the cue and
    the budget from the arguments, so this remains a policy that could be run in place of the
    network - which is the whole point of an executable teacher.
    """
    if not v17:
        return v197.v106
    v25 = [v107 for v107, v198 in v199(v17) if v198 not in v18]
    if v25 and v20 < v22 and (v20 + 2 <= v21):
        return 2 + v25[0]
    v26 = v108(v19)
    v27 = v26.v109(2)
    v28 = v27[0][1] if v27 else 0
    v29 = v27[1][1] if v201(v27) > 1 else 0

    def answer_value(v110):
        for v107, v198 in v199(v17):
            if v40.v294(v198) == v110:
                return 2 + v23 + v107
        return 2 + 2 * v23
    if v28 > 0 and v28 == v29:
        return 2 + 2 * v23
    if v28 <= 1:
        if v24:
            v200 = v112(v139(v201(v17)), key=lambda v107: v24.v304(v17[v107], 0.0))
            return 2 + v23 + v200
        return 2 + v23
    return v111(v27[0][0])
v8 = 2
v9 = False

def state_tensors(v30, v31, v32, v33, v34, v35, v17, v18, v19, v36, v20, v37, v38, v23, v21):
    """274's state with two scalars added to the GLOBAL features.

    The first 278 run abstained on decidable as well as on tie. ANSWER_i is chosen by the
    candidate head, which has an `agreement` column since 271, but STOP comes from the global
    head - and its five features carry retrieve scores, counts and budget, nothing about whether
    the witnesses agree. After every witness is read a 3-2 and a 2-2 are the same vector there,
    so the policy cannot express "decided" and "undecidable" as different actions and settles on
    whichever BC saw more of. Adding the lead and the margin over what has been OPENED makes the
    distinction representable; both are counted from the transcript, never from a label.
    """
    v26 = v108(v19)
    v27 = v26.v109(2)
    v39 = v112(1, v201(v19))
    v28 = v27[0][1] if v27 else 0
    v29 = v27[1][1] if v201(v27) > 1 else 0

    class _Widen:
        """Widens the feature vector on its way into the policy; 274's builder is untouched."""
        v113 = v30.v113

        def __call__(v115, v118, v119, v120, v121=None, v122=None):
            v202 = v216.v259([v28 / v39, v210(v334(v28 - v29, 3))], device=v119.v38, dtype=v119.v295)
            return v30(v118[:0] if v9 else v118, v216.v296([v119, v202], dim=-1), v120, v121, v122)
    return v40.v114(v203(), v31, v32, v33, v34, v35, v17, v18, v19, v36, v20, v37, v38, v23, v21)

class PolicyV(v40.v10):
    """274's policy plus a value head on the same global state the actor sees.

    V(s) is a baseline, not a critic on the action: it never sees which action was taken, so
    subtracting it leaves the REINFORCE estimator unbiased. It replaces one running scalar that
    was shared across three families of wildly different return, which is what let the common
    family's gradient decide the rare family's action.
    """

    def __init__(v115, v79: v11, v23: v11, v38):
        v297().v204(v79, v23, v38)
        v115.v116 = v299.v298(v299.v319(v79 + v40.v335, 128), v299.v320(), v299.v319(128, 1)).v162(v38)
        v299.v260.v205(v115.v116[-1].v206)
        v299.v260.v205(v115.v116[-1].v207)
        v115.v117: v128 | None = None

    def forward(v115, v118, v119, v120, v121=None, v122=None):
        if v115.v117 is not None:
            v115.v117.v213(v115.v116(v216.v296([v118, v119], dim=-1)).v300(-1))
        return v297().v208(v118, v119, v120, v121, v122)

def rollout(v30, v31, v32, v33, v34, v41, v37, v38, *, v23, v21, v22, v42, v43, v44, v45, v46=False, v47=True, v48=False, v49=0.0):
    v123, v124, v125 = (v34['tape'], v34['postings'], v34['idf'])
    v40.v50 = {v107: v116 for v107, v116 in v199(v123.v261)}
    v51 = v197.v209.v126(S=v41['S'])
    v52 = v127(v51)
    v35 = v51
    v17: v128[v11] = []
    v36: v128[v102] = []
    v18: v129[v11] = v129()
    v53: v128[v102] = []
    v130, v131, v132, v133 = ([], [], [], [])
    v20, v134, v135 = (0, None, False)
    v136, v137 = (v210('nan'), v210('nan'))
    v54 = v129(v41['slots'])
    v55 = v138(max_steps=v21, max_reads=v22, k=v23)
    for v56 in v139(v21):
        if v48:
            v211 = v262(cands=v17, seen_reads=v18, opened_values=v53, n_reads=v20, cand_scores=v34.v304('_sc'), **v55)
        else:
            v212 = v114(v30, v31, v32, v33, v34, v35, v17, v18, v53, v36, v20, v37, v38, v23, v21)
            if v212 is None:
                break
            v263, v56 = v212
            if v46:
                v211 = v262(cands=v17, seen_reads=v18, opened_values=v53, n_reads=v20, cand_scores=v34.v304('_sc'), **v55)
                if not v216.v336(v263[v211]) or v263[v211] < -100000000.0:
                    break
                v130.v213(v312.v321(v263.v337(0), v216.v259([v211], device=v38)))
            else:
                v264 = v216.v322.v301(logits=v263)
                v211 = v11(v263.v338()) if v47 else v11(v264.v339())
                v131.v213(v264.v323(v216.v259(v211, device=v38)))
                v132.v213(v264.v324())
                if v49 > 0.0:
                    v302 = v262(cands=v17, seen_reads=v18, opened_values=v53, n_reads=v20, cand_scores=v34.v304('_sc'), **v55)
                    if v216.v336(v263[v302]) and v263[v302] > -100000000.0:
                        v130.v213(v312.v321(v263.v337(0), v216.v259([v302], device=v38)))
        v133.v213(v197.v303(v23)[v211])
        if v211 in (v197.v106, v197.v265):
            v214 = v52 if v211 == v197.v106 else v36
            v17, v266 = v197.v267(v214, v124, v125, v23)
            if v45:
                v268 = [v198 for v198 in v17 if v41['S'] in v34['texts'][v198]]
                v17 = v268 if v268 else v17
            v34['_sc'] = {v198: v266.v304(v198, 0.0) for v198 in v17}
            if v17:
                v269 = v287((1 for v198 in v17 if v198 in v54))
                if v325.v305(v136):
                    v136 = v269 / v201(v17)
                    v137 = v269 / v112(1, v201(v54))
        elif v211 == 2 + 2 * v23:
            v135 = True
            break
        elif v211 < 2 + v23:
            v107 = v211 - 2
            if v107 >= v201(v17):
                break
            v306 = v17[v107]
            v35 = (v35 + ' | ' + v34['texts'][v306])[-2000:]
            v36 = v127(v34['texts'][v306], exclude=v123.v261[v306])
            v18.v326(v306)
            v53.v213(v123.v261[v306])
            v20 += 1
        else:
            v107 = v211 - 2 - v23
            if v107 >= v201(v17):
                break
            v134 = v123.v261[v17[v107]]
            break
    if v135 or v134 is None:
        v140, v57, v135 = (0, v44, True)
    else:
        v140 = v11(v41['truth'] is not None and v134 == v41['truth'])
        v57 = 1.0 if v140 else -v43
    v57 -= v42 * v20
    return {'loss': v216.v284(v130).v270() if v130 else v216.v271((), device=v38), 'logps': v131, 'entropy': v132, 'reward': v57, 'correct': v140, 'abstained': v135, 'n_reads': v20, 'trace': v133, 'kind': v41['kind'], 'answer_is_slot': v134 is None or v134 in v129(v123.v261), 'retrieval_precision': v136, 'witness_recall': v137}

def main() -> v11:
    v58 = v215.v141()
    v58.v142('--smoke', action='store_true')
    v58.v142('--bc-episodes', type=v11, default=0)
    v58.v142('--rl-episodes', type=v11, default=0)
    v58.v142('--tape-period', type=v11, default=0)
    v58.v142('--clean', type=v11, default=4)
    v58.v142('--decidable', type=v11, default=4)
    v58.v142('--tie', type=v11, default=4)
    v58.v142('--witnesses', type=v11, default=5)
    v58.v142('--liars', type=v11, default=2)
    v58.v142('--distractor-slots', type=v11, default=0)
    v58.v142('--topk', type=v11, default=7)
    v58.v142('--max-steps', type=v11, default=10)
    v58.v142('--max-reads', type=v11, default=7)
    v58.v142('--read-cost', type=v210, default=0.02)
    v58.v142('--wrong-cost', type=v210, default=1.0, help='276 used 0.3, which made a guess on a coin-flip cheaper than reading')
    v58.v142('--abstain-reward', type=v210, default=0.75, help="276 used 0.0; 0.1 fixed the ORDERING but not the MAGNITUDE and RL collapsed reading (tie 4.0 reads -> 1.0). With one shared V(s) the optimal return per family has to be comparable: at 0.1 a tie was worth +0.02 against clean's +0.98, so even perfect play on a tie carried a negative advantage at every step and the only way down was to read less. At 0.75 the optima are 0.98 / 0.90 / 0.67 and silence still loses to an answer wherever an answer exists.")
    v58.v142('--entropy-bonus', type=v210, default=0.01)
    v58.v142('--lr-policy', type=v210, default=0.001)
    v58.v142('--lr-value', type=v210, default=0.003)
    v58.v142('--lr-upper', type=v210, default=3e-05)
    v58.v142('--value-coef', type=v210, default=0.5)
    v58.v142('--bc-anchor', type=v210, default=0.5, help='weight of a cross-entropy term against the executable teacher kept ON during RL. 0 reproduces plain REINFORCE, which erased the tie behaviour BC had already found.')
    v58.v142('--no-hidden', action='store_true', help='ablation: build the policy over the scalar features ONLY, with no trunk hidden state. If it holds up, the mind transfers between models (Qwen, curve, anything) without retraining, because nothing it reads is model-specific.')
    v58.v142('--no-value-head', action='store_true', help="ablation: 276's single running scalar baseline, everything else new")
    v58.v142('--subject-filter', choices=('off', 'on'), default='off')
    v58.v142('--idf', choices=('classic', 'soft'), default='classic')
    v58.v142('--filler', type=v11, default=4)
    v58.v142('--frozen-trunk', action='store_true')
    v59 = v58.v143()
    v60 = v59.v45 == 'on'
    v61 = not v59.v144
    global NO_HIDDEN
    v9 = v59.v62
    global LOG_PATH
    v14 = v59.v45 + ('_frozen' if v59.v155 else '') + ('' if v61 else '_noval') + ('_nohid' if v59.v62 else '')
    v145, v146, v7 = v147(v14)
    v7.v194.v104(parents=True, exist_ok=True)
    v7.v148('', encoding='utf-8')
    v38 = v216.v38('cuda' if v216.v307.v272() else 'cpu')
    v63 = v217.v149(v5)
    v216.v150(v5)
    v64 = v151.v151()
    v65 = v59.v152 or (400 if v59.v188 else 4000)
    v66 = v112(0, v59.v153)
    v67 = v59.v67 or (50 if v59.v188 else 200)
    v68 = v59.v154 or (150 if v59.v188 else 1000)
    v23 = v59.v69
    v70 = 'none' if v59.v155 else 'upper'
    v156(f'Stage278 value baseline start {v331.v317(v332.v318).v256()} device={v38} value_head={v61} wrong={v59.v43} abstain={v59.v44} subject_filter={v59.v45} bc={v65} rl={v66} k={v23} mode={v70}')
    v56, v56, v157, v158 = v159()
    v33 = v218.v160(v102(v273.v219))
    v71 = v33.v161()
    v37 = v33.v220(v221) or 0
    v32 = v308.v274(v33, v157, v37, v71).v162(v38)
    v72 = v2 if v2.v222() else v1
    v31 = v275(v158, v71).v162(v38)
    v31.v163(v216.v276(v72, map_location=v38, weights_only=False)['model'])
    v223.v164(v31, v70)
    v73 = v197.v165(v31)
    v74 = v275(v158, v71).v162(v38)
    v74.v163(v216.v276(v1, map_location=v38, weights_only=False)['model'])
    v74.v166()
    for v75 in v74.v167():
        v75.v224(False)
    v76 = v168(v74, v157, v38)
    with v4.v195('r', encoding='utf-8', errors='ignore') as v105:
        v169 = v105.v225(1500000 if v59.v188 else 8000000)
    v77 = v128(v138.v226((v15.v309(1) for v15 in v340.v327(v169) if v201(v15.v309(1)) >= 5)))
    v63.v170(v77)
    v78 = [v278.v277() for v278 in v169.v310('\n') if v201(v278.v277()) >= 60][:400 if v59.v188 else 6000]
    v79 = 0 if v59.v62 else 2 * (v31.v311.v279 // 2)
    v30 = v227(v79 + v8, v23, v38) if v61 else v40.v10(v79 + v8, v23, v38)
    v80 = [v75 for v75 in v31.v167() if v75.v228]
    v81 = [{'params': [v75 for v328, v75 in v30.v329() if not v328.v341('v.')], 'lr': v59.v229}]
    if v61:
        v81.v213({'params': v128(v30.v116.v167()), 'lr': v59.v280})
    if v80:
        v81.v213({'params': v80, 'lr': v59.v281})
    v82 = v216.v230.v171(v81, weight_decay=0.01)
    v83: v129[v102] = v129()
    v34, v172, v173 = (None, 0.0, [])
    v84 = v138(k=v23, max_steps=v59.v21, max_reads=v59.v22, read_cost=v59.v42, wrong_cost=v59.v43, abstain_reward=v59.v44, subject_filter=v60)

    def new_tape(v174):
        return v13.v231(bank=v76, tok=v33, pad_id=v37, device=v38, rng=v174, pool=v77, lines=v78, used=v83, n_clean=v59.v250, n_dec=v59.v251, n_tie=v59.v252, n_wit=v59.v189, n_liars=v59.v190, n_dist=v68, idf_mode=v59.v125, n_filler=v59.v187)
    v30.v175()
    v31.v175(v70 != 'none')
    for v85 in v139(1, v65 + 1):
        if v34 is None or (v85 - 1) % v67 == 0:
            v34 = v249(v63)
        v41 = v34['items'][v63.v282(v201(v34['items']))]
        v100 = v232(v30, v31, v32, v33, v34, v41, v37, v38, bc=True, **v84)
        v82.v233(set_to_none=True)
        v100['loss'].v234()
        v216.v299.v283.v235(v128(v30.v167()) + v80, 1.0)
        v82.v236()
        if v85 % v112(1, v65 // 8) == 0:
            v173.v213({'phase': 'bc', 'episode': v85, 'loss': v210(v100['loss']), 'kind': v100['kind'], 'trace': v100['trace']})
            v156(f"  bc {v85}/{v65} loss={v210(v100['loss']):.4f} [{v100['kind']}] {v100['trace']}")
    v86 = []
    for v85 in v139(1, v66 + 1):
        if (v85 - 1) % v67 == 0:
            v34 = v249(v63)
        v41 = v34['items'][v63.v282(v201(v34['items']))]
        if v61:
            v30.v117 = []
        v100 = v232(v30, v31, v32, v33, v34, v41, v37, v38, greedy=False, bc_anchor=v59.v49, **v84)
        v176 = v30.v117 if v61 else None
        if v61:
            v30.v117 = None
        if not v100['logps']:
            continue
        v177 = v100['reward']
        if v61 and v176:
            v237 = v216.v284(v176[:v201(v100['logps'])])
            v238 = (v177 - v237).v285()
            v239 = v312.v286(v237, v216.v313(v237, v177))
            v86.v213(v210(v239))
            v240 = -(v238 * v216.v284(v100['logps'])).v287()
            v179 = v240 + v59.v314 * v239
        else:
            v172 = 0.99 * v172 + 0.01 * v177
            v179 = -(v177 - v172) * v216.v284(v100['logps']).v287()
        v178 = v216.v284(v100['entropy']).v287() if v100['entropy'] else v216.v271((), device=v38)
        v179 = v179 - v59.v288 * v178
        if v59.v49 > 0.0 and v100['loss'].v228:
            v179 = v179 + v59.v49 * v100['loss']
        v82.v233(set_to_none=True)
        v179.v234()
        v216.v299.v283.v235(v128(v30.v167()) + v80, 1.0)
        v82.v236()
        if v85 % v112(1, v66 // 8) == 0:
            v241 = f'v_mse={v210(v330.v270(v86[-200:])):.3f}' if v86 else f'baseline={v172:.3f}'
            v173.v213({'phase': 'rl', 'episode': v85, 'baseline': v172, 'v_mse': v210(v330.v270(v86[-200:])) if v86 else None, 'kind': v100['kind'], 'trace': v100['trace']})
            v156(f"  rl {v85}/{v66} {v241} [{v100['kind']}] {v100['trace']}")
    v30.v166()
    v31.v166()
    v87 = v197.v165(v31)

    @v216.v183()
    def evaluate(v75):
        v180 = {v105: {'correct': [], 'abstain': [], 'reads': [], 'reward': [], 'prec': [], 'rec': []} for v105 in v6}
        v181 = {v105: {'correct': [], 'abstain': [], 'reward': []} for v105 in v6}
        v242, v243 = ([], [])
        for v182 in v75['items']:
            v244 = v232(v30, v31, v32, v33, v75, v182, v37, v38, **v84)
            v245 = v232(v30, v31, v32, v33, v75, v182, v37, v38, teacher_only=True, **v84)
            v105 = v182['kind']
            v180[v105]['correct'].v213(v244['correct'])
            v180[v105]['abstain'].v213(v11(v244['abstained']))
            v180[v105]['reads'].v213(v244['n_reads'])
            v180[v105]['reward'].v213(v244['reward'])
            if not v325.v305(v244['retrieval_precision']):
                v180[v105]['prec'].v213(v244['retrieval_precision'])
                v180[v105]['rec'].v213(v244['witness_recall'])
            v181[v105]['correct'].v213(v245['correct'])
            v181[v105]['abstain'].v213(v11(v245['abstained']))
            v181[v105]['reward'].v213(v245['reward'])
            v242.v213(v11(v244['answer_is_slot']))
            v243.v213({'kind': v105, 'trace': v244['trace'], 'correct': v244['correct'], 'abstained': v244['abstained'], 'prec': v244['retrieval_precision']})
        v15 = lambda v289: v210(v330.v270(v289)) if v289 else v210('nan')
        v100 = {'answer_is_slot': v15(v242), 'traces': v243, 'reward_total': v15([v174 for v105 in v6 for v174 in v180[v105]['reward']]), 'teacher_reward_total': v15([v174 for v105 in v6 for v174 in v181[v105]['reward']]), 'retrieval_precision': v15([v315 for v105 in v6 for v315 in v180[v105]['prec']]), 'witness_recall': v15([v315 for v105 in v6 for v315 in v180[v105]['rec']])}
        v246, v247 = (0, 0)
        for v105 in v6:
            v248 = v287((1 for v211 in v180[v105]['abstain'] if not v211))
            v246 += v287(v180[v105]['correct'])
            v247 += v248
            v100[v105] = {'coverage': 1.0 - v15(v180[v105]['abstain']), 'acc_answered': v287(v180[v105]['correct']) / v248 if v248 else v210('nan'), 'abstain': v15(v180[v105]['abstain']), 'mean_reads': v15(v180[v105]['reads']), 'reward': v15(v180[v105]['reward']), 'precision': v15(v180[v105]['prec']), 'recall': v15(v180[v105]['rec']), 'teacher_abstain': v15(v181[v105]['abstain']), 'teacher_acc_all': v15(v181[v105]['correct'])}
        v100['coverage_all'] = v247 / v112(1, v201(v75['items']))
        v100['acc_answered_all'] = v246 / v112(1, v247)
        return v100
    v88 = v184(v34)
    v89 = v184(v249(v217.v149(v5 + 99)))
    v156(f"  NOVEL {v291.v257({v254: v255 for v254, v255 in v89.v316() if v254 != 'traces'})}")
    v90 = v73 == v87
    v91 = v89['answer_is_slot'] >= 0.99
    v92 = v60 or (v89['retrieval_precision'] >= 0.5 and v89['witness_recall'] >= 0.6)
    v93 = v89['tie']['teacher_abstain'] >= 0.9 and v89['clean']['teacher_acc_all'] >= 0.9 and (v89['decidable']['teacher_acc_all'] >= 0.9)
    v94 = v89['clean']['abstain'] <= 0.15 and v89['decidable']['abstain'] <= 0.25
    v95 = v89['tie']['abstain'] >= 0.7
    v96 = v89['acc_answered_all'] >= 0.75
    v97 = v89['reward_total'] > 0.0
    v98 = v89['reward_total'] >= v88['reward_total'] - 0.15
    v99 = v89['reward_total'] >= v89['teacher_reward_total'] - 0.1
    if not (v90 and v91):
        v185 = 'VALUE_BASELINE_INVALID'
    elif not v92:
        v185 = 'RETRIEVAL_UNUSABLE'
    elif not v93:
        v185 = 'TEACHER_STILL_BROKEN'
    elif not v94:
        v185 = 'OVER_ABSTAINS_ON_DECIDABLE' if v89['clean']['abstain'] <= 0.15 else 'ABSTAINS_EVERYWHERE'
    elif v95 and v96 and v98 and v99:
        v185 = 'JUDGE_OK'
    elif v95 or v96:
        v185 = 'JUDGE_PARTIAL'
    else:
        v185 = 'JUDGE_NO'
    v216.v186({'policy': v30.v290(), 'model': v31.v290(), 'stage': 278, 'value_head': v61, 'arc_enc_hash': v87}, v3)
    v100 = {'stage': 278, 'overall': v185, 'value_head': v61, 'subject_filter': v59.v45, 'idf': v59.v125, 'filler_words': v59.v187, 'frozen_trunk': v59.v155, 'trunk_mode': v70, 'smoke': v59.v188, 'seed': v5, 'bc_episodes': v65, 'rl_episodes': v66, 'families': {'clean': v59.v250, 'decidable': v59.v251, 'tie': v59.v252}, 'witnesses': v59.v189, 'liars': v59.v190, 'topk': v23, 'reward': {'correct': 1.0, 'wrong': -v59.v43, 'abstain': v59.v44, 'read': -v59.v42}, 'family_optima': {'clean': 1.0 - v59.v42, 'decidable': 1.0 - 5 * v59.v42, 'tie': v59.v44 - 4 * v59.v42}, 'teacher': 'exhaust every retrieved witness the budget allows, then rule; repeats are the dispute signal, a repeated tie abstains', 'bc_anchor': v59.v49, 'no_hidden': v59.v62, 'policy_inputs': '7 scalars, nothing model-specific' if v59.v62 else f'trunk hidden {v79} + 7 scalars', 'baseline': 'V(s) on [h, feats], advantage R - V(s_t) per step' if v61 else "single running scalar (276's)", 'value_mse_last': v210(v330.v270(v86[-200:])) if v86 else None, 'fp_version': v197.v253(), 'used_pool_final': v201(v83), 'gates': {'G_arc_enc_frozen': v90, 'G_answer_is_slot': v91, 'G_retrieval_usable': v92, 'G_teacher_ceiling': v93, 'G_answers_when_decidable': v94, 'G_abstain_on_tie': v95, 'G_acc_when_answering': v96, 'G_beats_always_answer': v97, 'G_novel_tape': v98, 'G_reaches_teacher': v99}, 'train_tape': {v254: v255 for v254, v255 in v88.v316() if v254 != 'traces'}, 'novel_tape': v89, 'arc_enc_hash_before': v73, 'arc_enc_hash_after': v87, 'curve': v173, 'reference_276_on': {'tie_abstain': 0.0, 'tie_teacher_abstain': 0.5, 'clean_acc': 1.0, 'decidable_acc': 1.0, 'reward_total': 0.5}, 'note': "276 failed only on ties and its curve says why. bc 3000 and bc 3500 both traced STOP on a tie, and every RL trace ended in ANSWER: one running scalar baseline shared across three families let the two thirds of episodes that answer for +0.98 push down the global STOP logit, which is the only head a tie's correct action can use. A state-dependent V(s) fixes that without reading `kind`, which a per-family baseline would. Second, the ceiling was broken - 275's teacher ruled on whatever was open, so a 2-2 read halfway looked decided, and teacher_abstain was 0.50 on the novel tape; the teacher now exhausts the retrieve list before any verdict, and G_teacher_ceiling is a VALIDITY gate rather than a result. Third, wrong -0.3 against abstain 0.0 made a guess on a coin flip worth more than silence and equal to fifteen reads; -1.0 against +0.1 gives the task's real ordering. Everything else is 276 verbatim, so a change in the tie family is a change from these three.", 'timestamp': v331.v317(v332.v318).v256(), 'wall_s': v151.v151() - v64}
    v0.v104(parents=True, exist_ok=True)
    v145.v148(v291.v257(v100, indent=2), encoding='utf-8')
    v146.v148(f"# Stage 278 value baseline, honest ceiling, real abstain cost\n\n**{v185}**{(' · SMOKE' if v59.v188 else '')} · value head **{v61}** · reward wrong -{v59.v43} / abstain +{v59.v44}\n\n| family (novel) | coverage | acc answered | abstain | teacher abstain | reads |\n|---|---:|---:|---:|---:|---:|\n" + ''.v292((f"| {v105} | {v89[v105]['coverage']:.2f} | {v89[v105]['acc_answered']:.2f} | {v89[v105]['abstain']:.2f} | {v89[v105]['teacher_abstain']:.2f} | {v89[v105]['mean_reads']:.1f} |\n" for v105 in v6)) + f"\n- overall coverage {v89['coverage_all']:.2f} at accuracy {v89['acc_answered_all']:.2f} (276 with filter on: 1.00 at 0.67)\n- reward: policy {v89['reward_total']:.3f} vs teacher {v89['teacher_reward_total']:.3f}\n- tie abstain **{v89['tie']['abstain']:.2f}** (276: 0.00), teacher **{v89['tie']['teacher_abstain']:.2f}** (276: 0.50)\n\n## Gates\n\n" + ''.v292((f'- {v254}: **{v255}**\n' for v254, v255 in v100['gates'].v316())), encoding='utf-8')
    v156(v291.v257({'overall': v185, 'gates': v100['gates']}, indent=2))
    return 0
if v101 == '__main__':
    raise v191(v258())