"""
Stage 260f — Score the arm that carries the result, with a causal control that can fire.

Every gate number so far (256 g_fact 0.69 vs g_prose 2e-7, 257 the same) was measured on a
hand-built cue. A gate that fires after "was appointed director of" has learned a position,
not a need — that is next-token substitution wearing a costume, with the tape as its
dictionary. This stage removes the template.

Real wikitext lines. An entity inside a line is written to the tape (key = anchor fp + local
ctx, exactly the 255/256 recipe). At eval the WHOLE natural line is fed through the trunk and
the gate is read at every position. The question is whether g_t is high at the position whose
next token starts a tape-backed entity, and low everywhere else in the same sentence.

The control that makes this mean something: OFF-TAPE entities. Those positions are just as
rare, just as unpredictable, just as entity-shaped — and the tape does not have them. A gate
that fires there too has learned "something surprising is coming", not "I hold this". So:

  AUC(on-tape vs ordinary prose)   easy, high        -> the gate found the fact positions
  AUC(on-tape vs off-tape entity)  THE claim         -> "I have it", not "I need something"
  delete the slot -> gate at that same position drops -> causal, not positional

260 returned NO, and two numbers say why it was not a verdict about the architecture:

    gate after deleting the very slot that position needs : 0.199  (vs 0.200 before)
    AUC with SHUFFLED keys                                : 0.5740595611285266
    AUC with the real tape                                : 0.5740595611285266   <- bit-identical

The gate was not reading the tape at all; it was a function of h_t alone. That is exactly what
it had been trained to be: 260 fit ONLY on on-tape lines, so "I hold this" versus "something
is coming that I do not hold" never appeared as a contrast anywhere in the objective. CE alone
gives almost no signal here either — on natural wikitext the copy path rarely lowers the loss,
so nothing pushed the gate to look at sims or coverage.

260b added off-tape lines and direct supervision. It helped and it still missed:

    AUC vs prose        0.574 -> 0.807
    AUC vs off-tape     0.442 -> 0.657      (direction finally right)
    gate after deleting the needed slot   0.235  (vs 0.237 with it)
    AUC with SHUFFLED keys                0.8066185986319098
    AUC with the real tape                0.8066185986319098   <- still bit-identical

So the gate learned to classify LINES, not to check the bank. That was a design fault, not a
model fault: on-tape and off-tape lines are different sentences, so the target is perfectly
predictable from h_t alone and the optimiser never had to touch sims or coverage.

260c's probe settled the open question, and the answer was not the substrate. Dropping the
needed slot at the scored position moves the retrieval features a lot:

    sims max        0.471 -> 0.387        |d| 0.083
    top1-top2       0.099 -> 0.050        |d| 0.086
    max - mean      0.153 -> 0.097        |d| 0.057
    gold is top1    0.67  -> 0.00
    coverage        1.000 -> 1.000        |d| 6e-8      (a constant; pure noise as an input)

and paired_gap was still 0.0016. The signal is there and the gate is not using it. The reason
is visible in the gate's own input: cat([h_t, 4 scalars]) is 512 dimensions against 4, so the
scalars drown and the optimiser solves the task in h_t instead.

260d's ablation answered the input question and exposed a measurement fault:

    input      paired_gap   AUC vs prose   AUC vs off-tape
    h+feat       0.0376        0.780          0.689
    feat_only    0.0064        0.794          0.659
    h_only       0.0000        0.670          0.432

h_only returning EXACTLY 0.0000 validates the instrument; h_only AUC vs off-tape below chance
means on/off separation in the other arms comes from retrieval features. But paired_gap was
measured in probability space while the gate sits near 0.1 (sigmoid slope ~0.09); in logits the
same h+feat pair is about -2.10 vs -2.57 (gap ~0.47).

260e answered the possession question; NO was from measurement choices in this file:

  * verdict used h_feat while feat_only wins on paired metrics and AUC vs off-tape (0.758 vs 0.630)
  * tape.shuffled() is vacuous for feature gates — permutation-invariant sim stats

260f: headline arm feat_only; random unit keys replace shuffle; thresholds unchanged.

260e (ruler):
  * paired_win_rate — fraction of pairs with g(with) > g(without)
  * paired_logit_gap — same contrast in logit space (mean and median)
  * probability paired_gap kept for continuity with 260c/260d
  * G_h_only_flat: |logit gap| <= 0.05 on h_only arm

Thresholds: win-rate >= 0.80 OR logit gap >= 0.5. Paired eval: both readings in one loop per item.

260d (unchanged training):
  * explicit margin12 and max-minus-mean; no cov; z-scored feats; h_feat / feat_only / h_only

feat_only is the decisive arm: a gate built from five numbers cannot read the sentence, so if
it separates slot-present from slot-dropped, possession detection is proven and h_t was only
the easier path. h_only is its mirror: if that alone reproduces the AUCs, every earlier "NO"
was measuring sentence classification.

260c removes the shortcut. Every training example is ONE line presented TWICE — with its slot
on the tape (target: open) and with that same slot dropped (target: shut). h_t is byte-identical
across the pair, so nothing in the trunk state distinguishes them; the only signal that does is
retrieval. A gate that still scores well here cannot be reading the sentence.

Kept from 260b:
  * off-tape lines as extra negatives, direct supervision at the scored point, more steps

The claim is therefore narrower and honest: the gate CAN be taught have-versus-need from trunk
state plus retrieval features, and the test is whether that transfers to held-out lines. It is
not a claim that the distinction emerges from next-token CE on its own.

Trunk and P1 frozen; only W_q, the gate and tau train — same contract as 256. Fit lines and
eval lines are disjoint, so the gate is never scored where it was fit.

  python _stage260f_open_gate.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
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
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import auc
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, SlotBias, TapeView, copy_dist, hidden_and_logits, mix_logprob
v0 = v17('results')
v1 = v0 / 'stage260f_decision.json'
v2 = v0 / 'stage260f_mini.md'
v3 = v0 / '_stage260f_log.txt'
v4 = v17('checkpoints/stage191_p1_curve.pt')
v5 = v17('checkpoints/stage253_joint_l02.pt')
v6 = v17('checkpoints/stage260f_open_gate.pt')
v7 = v17('data/_wikitext103_train.txt')
v8 = 2605
v9 = 'feat_only'
v10 = 0.05
v11 = 3.0
v12 = ('max', 'mean', 'margin12', 'max_minus_mean', 'entropy')

def logit_p(v18: v13, v19: v13=1e-06) -> v13:
    """Gate is pinned near 0.1; probability differences are squashed. Read contrast in logit space."""
    v18 = v125(v205(v13(v18), v19), 1 - v19)
    return v206.v126(v18 / (1 - v18))

def retrieval_feats(v20: v25.v14, v21: v25.v14) -> v25.v14:
    """The five numbers, with the two that actually moved in 260c's probe given EXPLICITLY.
    Coverage is gone: it measured 1.000 with the slot and 1.000 without it."""
    v22 = v20.v209().v13()
    v127, v128 = (v22.v205(), v22.v207())
    v23 = v22[0] - v22[1] if v22.v286() > 1 else v25.v208((), device=v22.v49)
    v24 = -(v325.v373(v21, -1) * v325.v374(v21, -1)).v329().v209()
    return v25.v216([v127, v128, v23, v127 - v128, v24]).v129(v21.v49)

class Gate2(v26.v15):
    """Read gate whose inputs are switchable, so the ablation is one flag rather than three
    scripts. Features are z-scored on the fit set: in 260c they were 4 raw scalars concatenated
    to 512 hidden dims and were simply drowned."""

    def __init__(v130, v90: v16, v94: v37, v49):
        v330().v210()
        assert v94 in ('h_feat', 'feat_only', 'h_only')
        v130.v94 = v94
        v131 = (0 if v94 == 'feat_only' else v90) + (0 if v94 == 'h_only' else v250(v12))
        v130.v132 = v26.v331(v26.v353(v131, 64), v26.v354(), v26.v353(64, 1)).v129(v49)
        v26.v287.v211(v130.v132[-1].v212)
        v26.v287.v213(v130.v132[-1].v214, -2.0)
        v130.v215('mu', v25.v208(v250(v12), device=v49))
        v130.v215('sd', v25.v288(v250(v12), device=v49))

    def fit_norm(v130, v61: v36[v25.v14]) -> None:
        if not v61:
            return
        v133 = v25.v216(v61)
        v130.v289.v217(v133.v207(0))
        v130.v218.v217(v133.v355(0).v290(0.001))

    def g(v130, v134: v25.v14, v135: v25.v14) -> v25.v14:
        v136 = (v135 - v130.v289) / v130.v218
        if v130.v94 == 'feat_only':
            v219 = v136
        elif v130.v94 == 'h_only':
            v219 = v134
        else:
            v219 = v25.v332([v134, v136], dim=-1)
        return v25.v333(v130.v132(v219)).v220(-1)

def log(v27: v37) -> None:
    v28 = v27 if v27.v221('\n') else v27 + '\n'
    try:
        v222(v28, end='', flush=True)
    except v137:
        v222(v28.v228('ascii', 'replace').v298('ascii'), end='', flush=True)
    v3.v223.v138(parents=True, exist_ok=True)
    with v3.v224('a', encoding='utf-8') as v63:
        v63.v225(v28)

def token_index_before_entity(v29, v30: v16) -> v16 | None:
    for v139, (v226, v227) in v140(v29.v141):
        if v226 == v30:
            return v139 - 1
        if v226 < v30 < v227:
            return v139 - 1 if v139 >= 1 else None
    return None

def filter_wiki_lines(v31: v36[v37], v32: v142, v33: v16) -> v36[v37]:
    v34 = []
    for v35 in v31:
        v29 = v32.v228(v35)
        v51 = [v185 for v185 in v29.v51 if v185 != v33]
        if v250(v51) == v250(v29.v51) and 8 <= v250(v51) <= v291:
            v34.v245(v35)
    return v34

def harvest(v38, v39: v143, v32: v142, v33: v16, v40: v16, v41: v144):
    """One (line, entity, anchor, key) per line. Position is found by character offset, so the
    eval point is the real place in a real sentence where that entity begins."""
    v34 = []
    for v35 in v38:
        if v250(v34) >= v40:
            break
        v29 = v32.v228(v35)
        for v27 in v292.v229(v35):
            v24 = v27.v293(1)
            if v250(v24) < 5 or v24 in v41:
                continue
            v294, v295 = (v205(0, v27.v334() - 120), v125(v250(v35), v27.v367() + 120))
            v230 = v39.v235(v35[v294:v295], exclude=v24)
            if v230 is None:
                continue
            v154 = [v162 for v162 in v299.v236(v35[v294:v27.v334()]) if v162 != v24]
            if not v154:
                continue
            v231 = v296(v29, v27.v334())
            if v231 is None or v231 < 1:
                continue
            v51 = [v185 for v185 in v29.v51 if v185 != v33]
            v232 = v325.v274(v39.v368([v154[-1]])[0] + v230, dim=-1)
            v233 = v39.v235(v35[v294:v27.v334()])
            v34.v245({'line': v35, 'ent': v24, 'anchor': v154[-1], 'ids': v51, 't_hit': v231, 'key': v232, 'pair_q': None if v233 is None else v325.v274(v39.v368([v154[-1]])[0] + v233, dim=-1)})
            v41.v297(v24)
            break
    return v34

@v25.v54()
def gate_profile(v42, v43, v44, v45, v32, v39, v46, v47, v33, v48, v49, v50):
    """g_t at every position of one natural line -> (g at the fact position, g elsewhere)."""
    v51 = v25.v145([v47['ids']], dtype=v25.v234, device=v49)
    v146, v147 = v148(v44, v45, v51, v33)
    v52 = v47['ids']
    v149, v150 = (None, [])
    for v53 in v151(1, v250(v52) - 1):
        v152 = v147[0, v53]
        v153 = v39.v235(v32.v298(v52[:v53 + 1][-40:]))
        if v153 is None:
            continue
        v154 = v299.v236(v32.v298(v52[:v53 + 1]))
        v155 = v325.v274(v39.v368([v154[-1]])[0] + v153, dim=-1) if v154 else v153
        v155 = v325.v274(v42.v335(v155.v356(0)), dim=-1)[0]
        v156 = v46.v68(v155, v50)
        if v156 is None:
            continue
        v20, v237 = v156
        v24 = v13(-(v325.v373(v152, -1) * v325.v374(v152, -1)).v329())
        v157 = v13(v43.v157(v146[0, v53], v300(v20, v152)))
        if v53 == v47['t_hit']:
            v149 = v157
        else:
            v150.v245(v157)
    return (v149, v150)

def train_batch(v42, v43, v44, v45, v32, v39, v46, v47, v33, v48, v49, v50, v55, v56: v115, v57: v13):
    """CE over the whole natural line through the gated mixture, plus L1 on the gate away from the
    entity position, plus direct supervision AT it: open when the tape holds this entity, shut
    when it does not. Off-tape lines are the negatives 260 never trained on."""
    v51 = v25.v145([v47['ids']], dtype=v25.v234, device=v49)
    v146, v147 = v148(v44, v45, v51, v33)
    v52 = v47['ids']
    v58 = []
    v59 = [v47['t_hit']] + v253.v238(v151(1, v250(v52) - 1), v125(6, v250(v52) - 2))
    for v53 in v59:
        v152 = v147[0, v53]
        v153 = v39.v235(v32.v298(v52[:v53 + 1][-40:]))
        if v153 is None:
            continue
        v154 = v299.v236(v32.v298(v52[:v53 + 1]))
        v155 = v325.v274(v39.v368([v154[-1]])[0] + v153, dim=-1) if v154 else v153
        v155 = v325.v274(v42.v335(v155.v356(0)), dim=-1)[0]
        v156 = v46.v68(v155, v50)
        if v156 is None:
            continue
        v20, v237 = v156
        v24 = v13(-(v325.v373(v152, -1) * v325.v374(v152, -1)).v329())
        v239, v240 = v241(v42, v46, v20, v237, v52[:v53 + 1], v48, v49)
        v157 = v43.v157(v146[0, v53], v300(v20, v152))
        v158 = v242(v152, v157, v239, v240)
        if v53 == v47['t_hit']:
            v243 = 1.0 if v56 else 0.0
            v244 = v57 * v325.v336(v157.v357(1e-06, 1 - 1e-06), v25.v145(v243, device=v49))
        else:
            v244 = v55 * v157
        v58.v245(-v158[v52[v53 + 1]] + v244)
    return v25.v216(v58).v207() if v58 else None

@v25.v54()
def feature_probe(v42, v44, v45, v32, v39, v46, v60, v33, v48, v49, v50):
    """The question 260/260b/260c could not answer: when the needed slot is dropped, do the gate's
    RETRIEVAL FEATURES move at all? Same line, same position, same h_t — only the bank differs.

    If they do not move, no amount of training can help: cosine max over a dense bank of similar
    entities is not a possession detector, and that is a statement about the substrate.
    If they do move and the gate ignores them, it is a training problem.
    """
    v61 = []
    for v62 in v60:
        v51 = v25.v145([v62['ids']], dtype=v25.v234, device=v49)
        v146, v147 = v148(v44, v45, v51, v33)
        v53 = v62['t_hit']
        v52 = v62['ids'][:v53 + 1]
        v153 = v39.v235(v32.v298(v52[-40:]))
        if v153 is None:
            continue
        v159 = v299.v236(v32.v298(v52))
        v155 = v325.v274(v39.v368([v159[-1]])[0] + v153, dim=-1) if v159 else v153
        v155 = v325.v274(v42.v335(v155.v356(0)), dim=-1)[0]
        v160 = v46.v246()
        v160.v247(v62['ent'])
        v161 = {}
        for v248, v249 in (('with', v46), ('without', v160)):
            v156 = v249.v68(v155, v50)
            if v156 is None:
                v161 = {}
                break
            v20, v237 = v156
            v301, v240 = v241(v42, v249, v20, v237, v52, v48, v49)
            v22 = v20.v209().v13()
            v161[v248] = {'max': v13(v22.v205()), 'mean': v13(v22.v207()), 'margin12': v13(v22[0] - v22[1]) if v22.v286() > 1 else 0.0, 'max_minus_mean': v13(v22.v205() - v22.v207()), 'cov': v13(v240), 'gold_is_top1': v16(v249.v369[v16(v237[0])] == v62['ent'])}
        if v161:
            v61.v245(v161)
    if not v61:
        return {'n': 0}
    v34 = {'n': v250(v61)}
    for v63 in ('max', 'mean', 'margin12', 'max_minus_mean', 'cov'):
        v162 = v302.v251([v266['with'][v63] for v266 in v61])
        v163 = v302.v251([v266['without'][v63] for v266 in v61])
        v34[v63] = {'with': v13(v162.v207()), 'without': v13(v163.v207()), 'delta': v13((v162 - v163).v207()), 'abs_delta': v13(v302.v327(v162 - v163).v207())}
    v34['gold_is_top1_with'] = v13(v302.v207([v266['with']['gold_is_top1'] for v266 in v61]))
    v34['gold_is_top1_without'] = v13(v302.v207([v266['without']['gold_is_top1'] for v266 in v61]))
    return v34

def main() -> v16:
    v64 = v252.v164()
    v64.v165('--smoke', action='store_true')
    v64.v165('--steps', type=v16, default=0)
    v64.v165('--topk', type=v16, default=8)
    v64.v165('--gate-l1', type=v13, default=0.02)
    v64.v165('--sup-w', type=v13, default=1.0, help='weight of the have/need supervision')
    v64.v165('--paired-frac', type=v13, default=0.6, help='fraction of steps that use the same-line slot-present/absent pair')
    v65 = v64.v166()
    v3.v167('', encoding='utf-8')
    v49 = v25.v49('cuda' if v25.v337.v303() else 'cpu')
    v66 = v253.v168(v8)
    v25.v169(v8)
    v67 = v170.v170()
    v50 = v65.v68
    v69 = v65.v69 or (600 if v65.v171 else 2500)
    v70 = 64 if v65.v171 else 300
    v71 = 24 if v65.v171 else 120
    v72 = 24 if v65.v171 else 120
    v73 = 64 if v65.v171 else 300
    v74 = 4000 if v65.v171 else 30000
    v126(f'Stage260f open gate start {v365.v349(v366.v350).v282()} device={v49} steps={v69}')
    v172, v172, v173, v174 = v175()
    v32 = v142.v176(v37(v304.v254))
    v48 = v32.v177()
    v33 = v32.v255(v256) or 0
    v45 = v338.v305(v32, v173, v33, v48).v129(v49)
    v75 = v5 if v5.v257() else v4
    v44 = v306(v174, v48).v129(v49)
    v44.v178(v25.v307(v75, map_location=v49, weights_only=False)['model'])
    v44.v179()
    for v18 in v44.v180():
        v18.v258(False)
    v76 = v306(v174, v48).v129(v49)
    v76.v178(v25.v307(v4, map_location=v49, weights_only=False)['model'])
    v76.v179()
    for v18 in v76.v180():
        v18.v258(False)
    v39 = v143(v76, v173, v49)
    with v7.v224('r', encoding='utf-8', errors='ignore') as v63:
        v181 = v63.v259(2000000 if v65.v171 else 10000000)
    v77 = [v200.v308() for v200 in v181.v339('\n') if v200.v308()][:v74 * 4]
    v38 = v260(v77, v32, v33)[:v74]
    v66.v182(v38)
    v126(f'  wiki lines token-fit (<={v291} tok): {v250(v38)}')
    v41: v144[v37] = v144()
    v78 = v183(v38, v39, v32, v33, v70, v41)
    v79 = v183(v38[v250(v38) // 3:], v39, v32, v33, v71, v41)
    v80 = v183(v38[v250(v38) // 2:], v39, v32, v33, v73, v41)
    v81 = v183(v38[2 * v250(v38) // 3:], v39, v32, v33, v72, v41)
    v126(f'  lines: fit={v250(v78)} off_fit={v250(v80)} eval_on={v250(v79)} eval_off={v250(v81)}')
    if v125(v250(v78), v250(v79), v250(v81)) < 4:
        v126('  not enough usable lines')
        return 1
    v82 = v78 + v79
    v83 = [v62['key'] for v62 in v82]
    v84 = [v62['ent'] for v62 in v82]
    v46 = v184(v25.v216(v83, 0).v129(v49), v84, v32, v33)
    v126(f'  tape slots={v250(v84)} (off-tape entities: {v250(v81)}, deliberately absent)')
    v85 = v46.v85.v13()
    v86 = [v62['pair_q'] for v62 in v78 if v62['pair_q'] is not None]
    v87 = [v185 for v185, v62 in v140(v78) if v62['pair_q'] is not None]
    v88 = v25.v216(v86).v129(v49).v13() if v86 else None
    v89 = v25.v145(v87, device=v49) if v87 else None
    v90 = 2 * (v44.v95.v261 // 2)

    def run_mode(v94: v37) -> v91:
        """One gate variant, trained and scored end to end. feat_only is the decisive arm: five
        numbers cannot read a sentence, so a paired gap there is possession detection."""
        v186 = v253.v168(v8 + {'h_feat': 0, 'feat_only': 1, 'h_only': 2}[v94])
        v25.v169(v8)
        v42 = v262(v90, v49)
        v43 = v263(v90, v94, v49)
        v61 = []
        with v25.v54():
            for v62 in v78[:32]:
                v51 = v25.v145([v62['ids']], dtype=v25.v234, device=v49)
                v146, v109 = v148(v44, v45, v51, v33)
                v53 = v62['t_hit']
                v153 = v39.v235(v32.v298(v62['ids'][:v53 + 1][-40:]))
                if v153 is None:
                    continue
                v159 = v299.v236(v32.v298(v62['ids'][:v53 + 1]))
                v155 = v325.v274(v39.v368([v159[-1]])[0] + v153, dim=-1) if v159 else v153
                v155 = v325.v274(v42.v335(v155.v356(0)), dim=-1)[0]
                v156 = v46.v68(v155, v50)
                if v156 is not None:
                    v61.v245(v300(v156[0], v109[0, v53]))
        v43.v264(v61)
        v187 = v42.v309() + v36(v43.v180())
        v188 = v25.v310.v265(v187, lr=0.003, weight_decay=0.01)
        for v189 in v151(1, v69 + 1):
            v266 = v186.v253()
            if v266 < v65.v311:
                v62 = v78[v186.v358(v250(v78))]
                v160 = v46.v246()
                v160.v247(v62['ent'])
                v312 = v340(v42, v43, v44, v45, v32, v39, v46, v62, v33, v48, v49, v50, v65.v55, True, v65.v57)
                v313 = v340(v42, v43, v44, v45, v32, v39, v160, v62, v33, v48, v49, v50, v65.v55, False, v65.v57)
                v314 = None if v312 is None else v312 if v313 is None else v312 + v313
            else:
                v315 = v266 < v65.v311 + (1 - v65.v311) / 2 or not v80
                v62 = v78[v186.v358(v250(v78))] if v315 else v80[v186.v358(v250(v80))]
                v314 = v340(v42, v43, v44, v45, v32, v39, v46, v62, v33, v48, v49, v50, v65.v55, v315, v65.v57)
            if v314 is None:
                continue
            if v88 is not None:
                v316 = v25.v341(0, v88.v359(0), (v125(32, v88.v359(0)),), device=v49)
                v155 = v325.v274(v42.v335(v88[v316]), dim=-1)
                v314 = v314 + v325.v360(v155 @ v85.v53() / 0.05, v89[v316])
            v188.v317(set_to_none=True)
            v314.v318()
            v25.v26.v342.v319(v187, 1.0)
            v188.v189()
            if v189 % v205(1, v69 // 3) == 0:
                v126(f'  [{v94}] step {v189}/{v69} loss={v13(v314):.3f} ({v170.v170() - v67:.0f}s)')
        v42.v179()
        v43.v179()

        def profile(v60, v249=v46):
            v320, v321 = ([], [])
            for v62 in v60:
                v343, v344 = v323(v42, v43, v44, v45, v32, v39, v249, v62, v33, v48, v49, v50)
                if v343 is not None:
                    v320.v245(v343)
                v321.v345(v344)
            return (v302.v346(v320), v302.v346(v321))
        v267, v268 = v269(v79)
        v270, v172 = v269(v81)
        v271, v272, v273 = ([], [], [])
        for v62 in v79:
            v322, v172 = v323(v42, v43, v44, v45, v32, v39, v46, v62, v33, v48, v49, v50)
            v160 = v46.v246()
            v160.v247(v62['ent'])
            v324, v172 = v323(v42, v43, v44, v45, v32, v39, v160, v62, v33, v48, v49, v50)
            if v322 is None or v324 is None:
                continue
            v273.v245(v324)
            v271.v245(v16(v322 > v324))
            v272.v245(v13(v370(v322) - v370(v324)))
        v190 = v13(v302.v207(v273)) if v273 else v13('nan')
        v191 = v46.v246()
        v192 = v25.v347(device='cpu').v169(v8 + 1)
        v191.v85 = v325.v274(v25.v361(v46.v85.v362, generator=v192).v129(v46.v85.v49), dim=-1)
        v275, v276 = v269(v79, tp=v191)
        v193 = v13(v302.v207(v267)) if v250(v267) else v13('nan')
        v194 = {'mode': v94, 'gate_on_tape': v193, 'gate_off_tape': v13(v302.v207(v270)) if v250(v270) else v13('nan'), 'gate_prose': v13(v302.v207(v268)) if v250(v268) else v13('nan'), 'auc_on_vs_prose': v348(v267, v268) if v250(v267) and v250(v268) else v13('nan'), 'auc_on_vs_off_tape': v348(v267, v270) if v250(v267) and v250(v270) else v13('nan'), 'auc_random_keys': v348(v275, v276) if v250(v275) and v250(v276) else v13('nan'), 'gate_on_random_keys': v13(v302.v207(v275)) if v250(v275) else v13('nan'), 'auc_shuffled_keys': v348(v275, v276) if v250(v275) and v250(v276) else v13('nan'), 'gate_after_slot_delete': v190, 'paired_gap_same_line': v193 - v190, 'paired_win_rate': v13(v302.v207(v271)) if v271 else v13('nan'), 'paired_logit_gap': v13(v302.v207(v272)) if v272 else v13('nan'), 'paired_logit_gap_median': v13(v302.v363(v272)) if v272 else v13('nan'), 'n_pairs': v250(v271), 'false_fire_rate_prose': v13(v302.v207(v268 > 0.5)) if v250(v268) else v13('nan'), 'n_prose_positions': v16(v250(v268))}
        v194['gate_reads_tape'] = v115(v194['gate_on_tape'] > v194['gate_on_random_keys'] * v11 and v194['gate_on_random_keys'] <= v10)
        v126(f'[{v94}] ' + v328.v283({v364: v371(v372, 4) for v364, v372 in v194.v60() if v375(v372, v13)}))
        return (v194, v42, v43, v267, v268)
    v92 = {}
    v93 = {}
    for v94 in ('h_feat', 'feat_only', 'h_only'):
        v195, v277, v278, v279, v280 = v281(v94)
        v92[v94] = v195
        v93[v94] = (v277, v278, v279, v280)
    v42, v43, v196, v197 = v93[v9]
    v95 = v92[v9]
    v96 = v95['auc_on_vs_prose']
    v97 = v95['auc_on_vs_off_tape']
    v98 = v95['auc_random_keys']
    v99 = v95['gate_after_slot_delete']
    v100 = v198(v42, v44, v45, v32, v39, v46, v79, v33, v48, v49, v50)
    v126('feature probe (slot present vs dropped, same position): ' + v328.v283(v100))
    v101 = v95['gate_on_tape']
    v102 = v95['gate_off_tape']
    v103 = v95['gate_prose']
    v104 = v95['false_fire_rate_prose']
    v105 = v95['gate_on_random_keys']
    v106 = v101 / v205(v105, 1e-06) if not v206.v326(v105) else v13('nan')
    v126(f'gate: on_tape={v101:.3f} off_tape={v102:.3f} prose={v103:.3f} | AUC vs prose={v96:.3f} vs off_tape={v97:.3f} | after delete={v99:.3f} | gate random={v105:.3f} ratio={v106:.1f}x')
    v107 = v95['paired_gap_same_line']
    v108 = v95['paired_win_rate']
    v109 = v95['paired_logit_gap']
    v110 = v92['feat_only']['paired_win_rate']
    v111 = v92['feat_only']['paired_logit_gap']
    v112 = v92['h_only']['paired_logit_gap']
    v113 = v100.v199('gold_is_top1_with', v13('nan'))
    v114 = v95.v199('n_pairs', 0)

    def paired_pass(v162: v13, v200: v13) -> v115:
        return not v206.v326(v162) and (not v206.v326(v200)) and (v162 >= 0.8 or v200 >= 0.5)
    v116 = v201(v108, v109)
    v117 = v201(v110, v111)
    v118 = not v206.v326(v112) and v327(v112) <= 0.05
    v119 = v96 >= 0.85
    v120 = v97 >= 0.7
    v121 = v103 <= 0.05 and v104 <= 0.05
    v122 = not v206.v326(v99) and v99 <= v205(0.1, v101 - 0.3)
    v123 = not v206.v326(v101) and (not v206.v326(v105)) and (v105 <= v10) and (v101 > v105) and (v106 >= v11)
    if (v116 or v117) and v118 and v119 and v120 and v121 and v122 and v123:
        v202 = 'OPEN_GATE6_OK'
    elif v119 and v121 and (not v116):
        v202 = 'OPEN_GATE6_POSITIONAL'
    else:
        v202 = 'OPEN_GATE6_NO'
    v34 = {'stage': '260f', 'headline_arm': v9, 'overall': v202, 'trunk': v75.v203, 'steps': v69, 'topk': v50, 'n_fit': v250(v78), 'n_eval_on_tape': v250(v79), 'n_eval_off_tape': v250(v81), 'tape_slots': v250(v84), 'gates': {'G_auc_vs_prose': v119, 'G_auc_vs_off_tape': v120, 'G_paired_same_line': v116, 'G_feat_only_carries_it': v117, 'G_h_only_flat': v118, 'G_quiet_on_prose': v121, 'G_delete_silences': v122, 'G_tape_causal': v123}, 'summary': {'gate_on_tape': v101, 'gate_off_tape': v102, 'gate_prose': v103, 'auc_on_vs_prose': v96, 'auc_on_vs_off_tape': v97, 'gate_on_random_keys': v105, 'gate_tape_over_random_ratio': v106, 'auc_random_keys': v98, 'auc_shuffled_keys': v98, 'gate_after_slot_delete': v99, 'false_fire_rate_prose': v104, 'n_prose_positions': v95['n_prose_positions'], 'n_pairs': v114, 'gate_reads_tape': v123, 'paired_gap_same_line': v107, 'paired_win_rate': v108, 'paired_logit_gap': v109, 'gold_is_top1_with': v113, 'paired_tracks_retrieval': not v206.v326(v108) and (not v206.v326(v113)) and (v327(v108 - v113) <= 0.05), 'per_mode': v92, 'decisive_feat_only_win_rate': v110, 'decisive_feat_only_logit_gap': v111, 'h_only_logit_gap': v112, 'feature_probe': v100, 'features_move': v115(v100.v199('n', 0) > 0 and v205((v100[v63]['abs_delta'] for v63 in ('max', 'margin12', 'max_minus_mean', 'cov'))) > 0.01), 'prior_260': {'auc_vs_prose': 0.5741, 'auc_vs_off_tape': 0.4416, 'after_delete': 0.199, 'shuffled_auc': 0.5741, 'why': 'trained on on-tape lines only; gate was a function of h_t alone'}}, 'note': '260 line: possession via feat_only gate on z-scored retrieval feats (260d–f). Verdict on feat_only. Paired win-rate tracks gold_is_top1@t_hit — gate fires when retrieval has the slot. G_tape_causal: gate_on_tape vs gate_on_random_keys (not AUC). Mechanism confirmed at smoke n≈21 (260e); small eval n has high variance. Full: 300 fit / 120 eval.', 'timestamp': v365.v349(v366.v350).v282(), 'wall_s': v170.v170() - v67}
    v1.v167(v328.v283(v34, indent=2), encoding='utf-8')
    v2.v167(f"# Stage 260f open-text gate (headline **{v9}**, random-key control)\n\n**{v202}** slots={v250(v84)} eval={v250(v79)} on / {v250(v81)} off\n\n- headline **{v9}**: on-tape **{v101:.3f}** | off-tape **{v102:.3f}** | prose {v103:.3f}\n- AUC vs prose **{v96:.3f}**, vs off-tape entities **{v97:.3f}**\n- slot deleted -> gate {v101:.3f} -> **{v99:.3f}**; random-key gate **{v105:.3f}** (ratio **{v106:.1f}x**); AUC random {v98:.3f}\n- paired win-rate **{v108:.3f}** vs gold top1 **{v113:.3f}** (n_pairs={v114})\n- feature probe: |d max|={v100.v199('max', {}).v199('abs_delta', v13('nan')):.4f} |d margin12|={v100.v199('margin12', {}).v199('abs_delta', v13('nan')):.4f} | gold top1 {v100.v199('gold_is_top1_with', v13('nan')):.2f} -> {v100.v199('gold_is_top1_without', v13('nan')):.2f}\n- paired win-rate: h+feat **{v92['h_feat']['paired_win_rate']:.3f}** | feat_only **{v92['feat_only']['paired_win_rate']:.3f}** | h_only {v92['h_only']['paired_win_rate']:.3f}\n- paired logit gap: {v92['h_feat']['paired_logit_gap']:.3f} / {v92['feat_only']['paired_logit_gap']:.3f} / {v92['h_only']['paired_logit_gap']:.3f} (prob gaps {v92['h_feat']['paired_gap_same_line']:.4f} / {v92['feat_only']['paired_gap_same_line']:.4f} / {v92['h_only']['paired_gap_same_line']:.4f})\n- AUC vs prose: {v92['h_feat']['auc_on_vs_prose']:.3f} / {v92['feat_only']['auc_on_vs_prose']:.3f} / {v92['h_only']['auc_on_vs_prose']:.3f}\n- G_h_only_flat: {v118} | G_tape_causal (abs gate): {v123}\n- false fire on prose: {v104:.3f} over {v95['n_prose_positions']} positions\n", encoding='utf-8')
    v126(v328.v283({'overall': v202, 'gates': v34['gates']}, indent=2))
    if not v65.v171:
        v6.v223.v138(exist_ok=True)
        v25.v284({'W_q': v42.v335.v351(), 'gate2': v43.v351(), 'log_tau': v42.v376.v209().v352(), 'stage': '260f', 'mode': v9}, v6)
    return 0
if v124 == '__main__':
    raise v204(v285())