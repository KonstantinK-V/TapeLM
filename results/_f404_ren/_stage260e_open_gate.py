"""
Stage 260e — Measure the paired contrast where it actually lives.

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

260e changes the ruler, not the model (same Gate2 / training as 260d):
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

  python _stage260e_open_gate.py [--smoke]
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
v0 = v14('results')
v1 = v0 / 'stage260e_decision.json'
v2 = v0 / 'stage260e_mini.md'
v3 = v0 / '_stage260e_log.txt'
v4 = v14('checkpoints/stage191_p1_curve.pt')
v5 = v14('checkpoints/stage253_joint_l02.pt')
v6 = v14('checkpoints/stage260e_open_gate.pt')
v7 = v14('data/_wikitext103_train.txt')
v8 = 2604
v9 = ('max', 'mean', 'margin12', 'max_minus_mean', 'entropy')

def logit_p(v15: v10, v16: v10=1e-06) -> v10:
    """Gate is pinned near 0.1; probability differences are squashed. Read contrast in logit space."""
    v15 = v118(v195(v10(v15), v16), 1 - v16)
    return v196.v119(v15 / (1 - v15))

def retrieval_feats(v17: v22.v11, v18: v22.v11) -> v22.v11:
    """The five numbers, with the two that actually moved in 260c's probe given EXPLICITLY.
    Coverage is gone: it measured 1.000 with the slot and 1.000 without it."""
    v19 = v17.v199().v10()
    v120, v121 = (v19.v195(), v19.v197())
    v20 = v19[0] - v19[1] if v19.v275() > 1 else v22.v198((), device=v19.v46)
    v21 = -(v324.v362(v18, -1) * v324.v363(v18, -1)).v318().v199()
    return v22.v206([v120, v121, v20, v120 - v121, v21]).v122(v18.v46)

class Gate2(v23.v12):
    """Read gate whose inputs are switchable, so the ablation is one flag rather than three
    scripts. Features are z-scored on the fit set: in 260c they were 4 raw scalars concatenated
    to 512 hidden dims and were simply drowned."""

    def __init__(v123, v87: v13, v91: v34, v46):
        v319().v200()
        assert v91 in ('h_feat', 'feat_only', 'h_only')
        v123.v91 = v91
        v124 = (0 if v91 == 'feat_only' else v87) + (0 if v91 == 'h_only' else v240(v9))
        v123.v125 = v23.v320(v23.v344(v124, 64), v23.v345(), v23.v344(64, 1)).v122(v46)
        v23.v276.v201(v123.v125[-1].v202)
        v23.v276.v203(v123.v125[-1].v204, -2.0)
        v123.v205('mu', v22.v198(v240(v9), device=v46))
        v123.v205('sd', v22.v277(v240(v9), device=v46))

    def fit_norm(v123, v58: v33[v22.v11]) -> None:
        if not v58:
            return
        v126 = v22.v206(v58)
        v123.v278.v207(v126.v197(0))
        v123.v208.v207(v126.v346(0).v279(0.001))

    def g(v123, v127: v22.v11, v128: v22.v11) -> v22.v11:
        v129 = (v128 - v123.v278) / v123.v208
        if v123.v91 == 'feat_only':
            v209 = v129
        elif v123.v91 == 'h_only':
            v209 = v127
        else:
            v209 = v22.v321([v127, v129], dim=-1)
        return v22.v322(v123.v125(v209)).v210(-1)

def log(v24: v34) -> None:
    v25 = v24 if v24.v211('\n') else v24 + '\n'
    try:
        v212(v25, end='', flush=True)
    except v130:
        v212(v25.v218('ascii', 'replace').v288('ascii'), end='', flush=True)
    v3.v213.v131(parents=True, exist_ok=True)
    with v3.v214('a', encoding='utf-8') as v60:
        v60.v215(v25)

def token_index_before_entity(v26, v27: v13) -> v13 | None:
    for v132, (v216, v217) in v133(v26.v134):
        if v216 == v27:
            return v132 - 1
        if v216 < v27 < v217:
            return v132 - 1 if v132 >= 1 else None
    return None

def filter_wiki_lines(v28: v33[v34], v29: v135, v30: v13) -> v33[v34]:
    v31 = []
    for v32 in v28:
        v26 = v29.v218(v32)
        v48 = [v178 for v178 in v26.v48 if v178 != v30]
        if v240(v48) == v240(v26.v48) and 8 <= v240(v48) <= v280:
            v31.v235(v32)
    return v31

def harvest(v35, v36: v136, v29: v135, v30: v13, v37: v13, v38: v137):
    """One (line, entity, anchor, key) per line. Position is found by character offset, so the
    eval point is the real place in a real sentence where that entity begins."""
    v31 = []
    for v32 in v35:
        if v240(v31) >= v37:
            break
        v26 = v29.v218(v32)
        for v24 in v281.v219(v32):
            v21 = v24.v282(1)
            if v240(v21) < 5 or v21 in v38:
                continue
            v283, v284 = (v195(0, v24.v323() - 120), v118(v240(v32), v24.v356() + 120))
            v220 = v36.v225(v32[v283:v284], exclude=v21)
            if v220 is None:
                continue
            v147 = [v155 for v155 in v289.v226(v32[v283:v24.v323()]) if v155 != v21]
            if not v147:
                continue
            v221 = v285(v26, v24.v323())
            if v221 is None or v221 < 1:
                continue
            v48 = [v178 for v178 in v26.v48 if v178 != v30]
            v222 = v324.v286(v36.v357([v147[-1]])[0] + v220, dim=-1)
            v223 = v36.v225(v32[v283:v24.v323()])
            v31.v235({'line': v32, 'ent': v21, 'anchor': v147[-1], 'ids': v48, 't_hit': v221, 'key': v222, 'pair_q': None if v223 is None else v324.v286(v36.v357([v147[-1]])[0] + v223, dim=-1)})
            v38.v287(v21)
            break
    return v31

@v22.v51()
def gate_profile(v39, v40, v41, v42, v29, v36, v43, v44, v30, v45, v46, v47):
    """g_t at every position of one natural line -> (g at the fact position, g elsewhere)."""
    v48 = v22.v138([v44['ids']], dtype=v22.v224, device=v46)
    v139, v140 = v141(v41, v42, v48, v30)
    v49 = v44['ids']
    v142, v143 = (None, [])
    for v50 in v144(1, v240(v49) - 1):
        v145 = v140[0, v50]
        v146 = v36.v225(v29.v288(v49[:v50 + 1][-40:]))
        if v146 is None:
            continue
        v147 = v289.v226(v29.v288(v49[:v50 + 1]))
        v148 = v324.v286(v36.v357([v147[-1]])[0] + v146, dim=-1) if v147 else v146
        v148 = v324.v286(v39.v325(v148.v347(0)), dim=-1)[0]
        v149 = v43.v65(v148, v47)
        if v149 is None:
            continue
        v17, v227 = v149
        v21 = v10(-(v324.v362(v145, -1) * v324.v363(v145, -1)).v318())
        v150 = v10(v40.v150(v139[0, v50], v290(v17, v145)))
        if v50 == v44['t_hit']:
            v142 = v150
        else:
            v143.v235(v150)
    return (v142, v143)

def train_batch(v39, v40, v41, v42, v29, v36, v43, v44, v30, v45, v46, v47, v52, v53: v108, v54: v10):
    """CE over the whole natural line through the gated mixture, plus L1 on the gate away from the
    entity position, plus direct supervision AT it: open when the tape holds this entity, shut
    when it does not. Off-tape lines are the negatives 260 never trained on."""
    v48 = v22.v138([v44['ids']], dtype=v22.v224, device=v46)
    v139, v140 = v141(v41, v42, v48, v30)
    v49 = v44['ids']
    v55 = []
    v56 = [v44['t_hit']] + v243.v228(v144(1, v240(v49) - 1), v118(6, v240(v49) - 2))
    for v50 in v56:
        v145 = v140[0, v50]
        v146 = v36.v225(v29.v288(v49[:v50 + 1][-40:]))
        if v146 is None:
            continue
        v147 = v289.v226(v29.v288(v49[:v50 + 1]))
        v148 = v324.v286(v36.v357([v147[-1]])[0] + v146, dim=-1) if v147 else v146
        v148 = v324.v286(v39.v325(v148.v347(0)), dim=-1)[0]
        v149 = v43.v65(v148, v47)
        if v149 is None:
            continue
        v17, v227 = v149
        v21 = v10(-(v324.v362(v145, -1) * v324.v363(v145, -1)).v318())
        v229, v230 = v231(v39, v43, v17, v227, v49[:v50 + 1], v45, v46)
        v150 = v40.v150(v139[0, v50], v290(v17, v145))
        v151 = v232(v145, v150, v229, v230)
        if v50 == v44['t_hit']:
            v233 = 1.0 if v53 else 0.0
            v234 = v54 * v324.v326(v150.v348(1e-06, 1 - 1e-06), v22.v138(v233, device=v46))
        else:
            v234 = v52 * v150
        v55.v235(-v151[v49[v50 + 1]] + v234)
    return v22.v206(v55).v197() if v55 else None

@v22.v51()
def feature_probe(v39, v41, v42, v29, v36, v43, v57, v30, v45, v46, v47):
    """The question 260/260b/260c could not answer: when the needed slot is dropped, do the gate's
    RETRIEVAL FEATURES move at all? Same line, same position, same h_t — only the bank differs.

    If they do not move, no amount of training can help: cosine max over a dense bank of similar
    entities is not a possession detector, and that is a statement about the substrate.
    If they do move and the gate ignores them, it is a training problem.
    """
    v58 = []
    for v59 in v57:
        v48 = v22.v138([v59['ids']], dtype=v22.v224, device=v46)
        v139, v140 = v141(v41, v42, v48, v30)
        v50 = v59['t_hit']
        v49 = v59['ids'][:v50 + 1]
        v146 = v36.v225(v29.v288(v49[-40:]))
        if v146 is None:
            continue
        v152 = v289.v226(v29.v288(v49))
        v148 = v324.v286(v36.v357([v152[-1]])[0] + v146, dim=-1) if v152 else v146
        v148 = v324.v286(v39.v325(v148.v347(0)), dim=-1)[0]
        v153 = v43.v236()
        v153.v237(v59['ent'])
        v154 = {}
        for v238, v239 in (('with', v43), ('without', v153)):
            v149 = v239.v65(v148, v47)
            if v149 is None:
                v154 = {}
                break
            v17, v227 = v149
            v291, v230 = v231(v39, v239, v17, v227, v49, v45, v46)
            v19 = v17.v199().v10()
            v154[v238] = {'max': v10(v19.v195()), 'mean': v10(v19.v197()), 'margin12': v10(v19[0] - v19[1]) if v19.v275() > 1 else 0.0, 'max_minus_mean': v10(v19.v195() - v19.v197()), 'cov': v10(v230), 'gold_is_top1': v13(v239.v358[v13(v227[0])] == v59['ent'])}
        if v154:
            v58.v235(v154)
    if not v58:
        return {'n': 0}
    v31 = {'n': v240(v58)}
    for v60 in ('max', 'mean', 'margin12', 'max_minus_mean', 'cov'):
        v155 = v292.v241([v256['with'][v60] for v256 in v58])
        v156 = v292.v241([v256['without'][v60] for v256 in v58])
        v31[v60] = {'with': v10(v155.v197()), 'without': v10(v156.v197()), 'delta': v10((v155 - v156).v197()), 'abs_delta': v10(v292.v316(v155 - v156).v197())}
    v31['gold_is_top1_with'] = v10(v292.v197([v256['with']['gold_is_top1'] for v256 in v58]))
    v31['gold_is_top1_without'] = v10(v292.v197([v256['without']['gold_is_top1'] for v256 in v58]))
    return v31

def main() -> v13:
    v61 = v242.v157()
    v61.v158('--smoke', action='store_true')
    v61.v158('--steps', type=v13, default=0)
    v61.v158('--topk', type=v13, default=8)
    v61.v158('--gate-l1', type=v10, default=0.02)
    v61.v158('--sup-w', type=v10, default=1.0, help='weight of the have/need supervision')
    v61.v158('--paired-frac', type=v10, default=0.6, help='fraction of steps that use the same-line slot-present/absent pair')
    v62 = v61.v159()
    v3.v160('', encoding='utf-8')
    v46 = v22.v46('cuda' if v22.v327.v293() else 'cpu')
    v63 = v243.v161(v8)
    v22.v162(v8)
    v64 = v163.v163()
    v47 = v62.v65
    v66 = v62.v66 or (600 if v62.v164 else 2500)
    v67 = 64 if v62.v164 else 300
    v68 = 24 if v62.v164 else 120
    v69 = 24 if v62.v164 else 120
    v70 = 64 if v62.v164 else 300
    v71 = 4000 if v62.v164 else 30000
    v119(f'Stage260e open gate start {v354.v339(v355.v340).v271()} device={v46} steps={v66}')
    v165, v165, v166, v167 = v168()
    v29 = v135.v169(v34(v294.v244))
    v45 = v29.v170()
    v30 = v29.v245(v246) or 0
    v42 = v328.v295(v29, v166, v30, v45).v122(v46)
    v72 = v5 if v5.v247() else v4
    v41 = v296(v167, v45).v122(v46)
    v41.v171(v22.v297(v72, map_location=v46, weights_only=False)['model'])
    v41.v172()
    for v15 in v41.v173():
        v15.v248(False)
    v73 = v296(v167, v45).v122(v46)
    v73.v171(v22.v297(v4, map_location=v46, weights_only=False)['model'])
    v73.v172()
    for v15 in v73.v173():
        v15.v248(False)
    v36 = v136(v73, v166, v46)
    with v7.v214('r', encoding='utf-8', errors='ignore') as v60:
        v174 = v60.v249(2000000 if v62.v164 else 10000000)
    v74 = [v190.v298() for v190 in v174.v329('\n') if v190.v298()][:v71 * 4]
    v35 = v250(v74, v29, v30)[:v71]
    v63.v175(v35)
    v119(f'  wiki lines token-fit (<={v280} tok): {v240(v35)}')
    v38: v137[v34] = v137()
    v75 = v176(v35, v36, v29, v30, v67, v38)
    v76 = v176(v35[v240(v35) // 3:], v36, v29, v30, v68, v38)
    v77 = v176(v35[v240(v35) // 2:], v36, v29, v30, v70, v38)
    v78 = v176(v35[2 * v240(v35) // 3:], v36, v29, v30, v69, v38)
    v119(f'  lines: fit={v240(v75)} off_fit={v240(v77)} eval_on={v240(v76)} eval_off={v240(v78)}')
    if v118(v240(v75), v240(v76), v240(v78)) < 4:
        v119('  not enough usable lines')
        return 1
    v79 = v75 + v76
    v80 = [v59['key'] for v59 in v79]
    v81 = [v59['ent'] for v59 in v79]
    v43 = v177(v22.v206(v80, 0).v122(v46), v81, v29, v30)
    v119(f'  tape slots={v240(v81)} (off-tape entities: {v240(v78)}, deliberately absent)')
    v82 = v43.v82.v10()
    v83 = [v59['pair_q'] for v59 in v75 if v59['pair_q'] is not None]
    v84 = [v178 for v178, v59 in v133(v75) if v59['pair_q'] is not None]
    v85 = v22.v206(v83).v122(v46).v10() if v83 else None
    v86 = v22.v138(v84, device=v46) if v84 else None
    v87 = 2 * (v41.v92.v251 // 2)

    def run_mode(v91: v34) -> v88:
        """One gate variant, trained and scored end to end. feat_only is the decisive arm: five
        numbers cannot read a sentence, so a paired gap there is possession detection."""
        v179 = v243.v161(v8 + {'h_feat': 0, 'feat_only': 1, 'h_only': 2}[v91])
        v22.v162(v8)
        v39 = v252(v87, v46)
        v40 = v253(v87, v91, v46)
        v58 = []
        with v22.v51():
            for v59 in v75[:32]:
                v48 = v22.v138([v59['ids']], dtype=v22.v224, device=v46)
                v139, v104 = v141(v41, v42, v48, v30)
                v50 = v59['t_hit']
                v146 = v36.v225(v29.v288(v59['ids'][:v50 + 1][-40:]))
                if v146 is None:
                    continue
                v152 = v289.v226(v29.v288(v59['ids'][:v50 + 1]))
                v148 = v324.v286(v36.v357([v152[-1]])[0] + v146, dim=-1) if v152 else v146
                v148 = v324.v286(v39.v325(v148.v347(0)), dim=-1)[0]
                v149 = v43.v65(v148, v47)
                if v149 is not None:
                    v58.v235(v290(v149[0], v104[0, v50]))
        v40.v254(v58)
        v180 = v39.v299() + v33(v40.v173())
        v181 = v22.v300.v255(v180, lr=0.003, weight_decay=0.01)
        for v182 in v144(1, v66 + 1):
            v256 = v179.v243()
            if v256 < v62.v301:
                v59 = v75[v179.v349(v240(v75))]
                v153 = v43.v236()
                v153.v237(v59['ent'])
                v302 = v330(v39, v40, v41, v42, v29, v36, v43, v59, v30, v45, v46, v47, v62.v52, True, v62.v54)
                v303 = v330(v39, v40, v41, v42, v29, v36, v153, v59, v30, v45, v46, v47, v62.v52, False, v62.v54)
                v304 = None if v302 is None else v302 if v303 is None else v302 + v303
            else:
                v305 = v256 < v62.v301 + (1 - v62.v301) / 2 or not v77
                v59 = v75[v179.v349(v240(v75))] if v305 else v77[v179.v349(v240(v77))]
                v304 = v330(v39, v40, v41, v42, v29, v36, v43, v59, v30, v45, v46, v47, v62.v52, v305, v62.v54)
            if v304 is None:
                continue
            if v85 is not None:
                v306 = v22.v331(0, v85.v350(0), (v118(32, v85.v350(0)),), device=v46)
                v148 = v324.v286(v39.v325(v85[v306]), dim=-1)
                v304 = v304 + v324.v351(v148 @ v82.v50() / 0.05, v86[v306])
            v181.v307(set_to_none=True)
            v304.v308()
            v22.v23.v332.v309(v180, 1.0)
            v181.v182()
            if v182 % v195(1, v66 // 3) == 0:
                v119(f'  [{v91}] step {v182}/{v66} loss={v10(v304):.3f} ({v163.v163() - v64:.0f}s)')
        v39.v172()
        v40.v172()

        def profile(v57, v239=v43):
            v310, v311 = ([], [])
            for v59 in v57:
                v333, v334 = v313(v39, v40, v41, v42, v29, v36, v239, v59, v30, v45, v46, v47)
                if v333 is not None:
                    v310.v235(v333)
                v311.v335(v334)
            return (v292.v336(v310), v292.v336(v311))
        v257, v258 = v259(v76)
        v260, v165 = v259(v78)
        v261, v262, v263 = ([], [], [])
        for v59 in v76:
            v312, v165 = v313(v39, v40, v41, v42, v29, v36, v43, v59, v30, v45, v46, v47)
            v153 = v43.v236()
            v153.v237(v59['ent'])
            v314, v165 = v313(v39, v40, v41, v42, v29, v36, v153, v59, v30, v45, v46, v47)
            if v312 is None or v314 is None:
                continue
            v263.v235(v314)
            v261.v235(v13(v312 > v314))
            v262.v235(v10(v359(v312) - v359(v314)))
        v183 = v10(v292.v197(v263)) if v263 else v10('nan')
        v264, v265 = v259(v76, tp=v43.v337(v8 + 1))
        v184 = v10(v292.v197(v257)) if v240(v257) else v10('nan')
        v185 = {'mode': v91, 'gate_on_tape': v184, 'gate_off_tape': v10(v292.v197(v260)) if v240(v260) else v10('nan'), 'gate_prose': v10(v292.v197(v258)) if v240(v258) else v10('nan'), 'auc_on_vs_prose': v338(v257, v258) if v240(v257) and v240(v258) else v10('nan'), 'auc_on_vs_off_tape': v338(v257, v260) if v240(v257) and v240(v260) else v10('nan'), 'auc_shuffled_keys': v338(v264, v265) if v240(v264) and v240(v265) else v10('nan'), 'gate_after_slot_delete': v183, 'paired_gap_same_line': v184 - v183, 'paired_win_rate': v10(v292.v197(v261)) if v261 else v10('nan'), 'paired_logit_gap': v10(v292.v197(v262)) if v262 else v10('nan'), 'paired_logit_gap_median': v10(v292.v352(v262)) if v262 else v10('nan'), 'n_pairs': v240(v261), 'false_fire_rate_prose': v10(v292.v197(v258 > 0.5)) if v240(v258) else v10('nan'), 'n_prose_positions': v13(v240(v258))}
        v185['gate_reads_tape'] = v108(v316(v185['auc_on_vs_prose'] - v185['auc_shuffled_keys']) > 1e-06)
        v119(f'[{v91}] ' + v317.v272({v353: v360(v361, 4) for v353, v361 in v185.v57() if v364(v361, v10)}))
        return (v185, v39, v40, v257, v258)
    v89 = {}
    v90 = {}
    for v91 in ('h_feat', 'feat_only', 'h_only'):
        v186, v266, v267, v268, v269 = v270(v91)
        v89[v91] = v186
        v90[v91] = (v266, v267, v268, v269)
    v39, v40, v187, v188 = v90['h_feat']
    v92 = v89['h_feat']
    v93 = v92['auc_on_vs_prose']
    v94 = v92['auc_on_vs_off_tape']
    v95 = v92['auc_shuffled_keys']
    v96 = v92['gate_after_slot_delete']
    v97 = v189(v39, v41, v42, v29, v36, v43, v76, v30, v45, v46, v47)
    v119('feature probe (slot present vs dropped, same position): ' + v317.v272(v97))
    v98 = v92['gate_on_tape']
    v99 = v92['gate_off_tape']
    v100 = v92['gate_prose']
    v101 = v92['false_fire_rate_prose']
    v119(f'gate: on_tape={v98:.3f} off_tape={v99:.3f} prose={v100:.3f} | AUC vs prose={v93:.3f} vs off_tape={v94:.3f} | after delete={v96:.3f} | shuffled AUC={v95:.3f}')
    v102 = v92['paired_gap_same_line']
    v103 = v92['paired_win_rate']
    v104 = v92['paired_logit_gap']
    v105 = v89['feat_only']['paired_win_rate']
    v106 = v89['feat_only']['paired_logit_gap']
    v107 = v89['h_only']['paired_logit_gap']

    def paired_pass(v155: v10, v190: v10) -> v108:
        return not v196.v315(v155) and (not v196.v315(v190)) and (v155 >= 0.8 or v190 >= 0.5)
    v109 = v191(v103, v104)
    v110 = v191(v105, v106)
    v111 = not v196.v315(v107) and v316(v107) <= 0.05
    v112 = v93 >= 0.85
    v113 = v94 >= 0.7
    v114 = v100 <= 0.05 and v101 <= 0.05
    v115 = not v196.v315(v96) and v96 <= v195(0.1, v98 - 0.3)
    v116 = not v196.v315(v95) and v95 <= 0.65
    if (v109 or v110) and v111 and v112 and v113 and v114 and v115 and v116:
        v192 = 'OPEN_GATE5_OK'
    elif v112 and v114 and (not v109):
        v192 = 'OPEN_GATE5_POSITIONAL'
    else:
        v192 = 'OPEN_GATE5_NO'
    v31 = {'stage': '260e', 'overall': v192, 'trunk': v72.v193, 'steps': v66, 'topk': v47, 'n_fit': v240(v75), 'n_eval_on_tape': v240(v76), 'n_eval_off_tape': v240(v78), 'tape_slots': v240(v81), 'gates': {'G_auc_vs_prose': v112, 'G_auc_vs_off_tape': v113, 'G_paired_same_line': v109, 'G_feat_only_carries_it': v110, 'G_h_only_flat': v111, 'G_quiet_on_prose': v114, 'G_delete_silences': v115, 'G_tape_causal': v116}, 'summary': {'gate_on_tape': v98, 'gate_off_tape': v99, 'gate_prose': v100, 'auc_on_vs_prose': v93, 'auc_on_vs_off_tape': v94, 'auc_shuffled_keys': v95, 'gate_after_slot_delete': v96, 'false_fire_rate_prose': v101, 'n_prose_positions': v92['n_prose_positions'], 'gate_reads_tape': v108(v316(v93 - v95) > 1e-06), 'paired_gap_same_line': v102, 'paired_win_rate': v103, 'paired_logit_gap': v104, 'per_mode': v89, 'decisive_feat_only_win_rate': v105, 'decisive_feat_only_logit_gap': v106, 'h_only_logit_gap': v107, 'feature_probe': v97, 'features_move': v108(v97.v341('n', 0) > 0 and v195((v97[v60]['abs_delta'] for v60 in ('max', 'margin12', 'max_minus_mean', 'cov'))) > 0.01), 'prior_260': {'auc_vs_prose': 0.5741, 'auc_vs_off_tape': 0.4416, 'after_delete': 0.199, 'shuffled_auc': 0.5741, 'why': 'trained on on-tape lines only; gate was a function of h_t alone'}}, 'note': '260e: same Gate2/training as 260d; paired_win_rate and paired_logit_gap per arm. Pass paired: win≥0.80 OR logit_gap≥0.5. G_h_only_flat: |h_only logit gap|≤0.05. Prob paired_gap kept for 260c/260d continuity.', 'timestamp': v354.v339(v355.v340).v271(), 'wall_s': v163.v163() - v64}
    v1.v160(v317.v272(v31, indent=2), encoding='utf-8')
    v2.v160(f"# Stage 260e open-text gate (paired win-rate + logit gap)\n\n**{v192}** slots={v240(v81)} eval={v240(v76)} on / {v240(v78)} off\n\n- gate: on-tape **{v98:.3f}** | off-tape **{v99:.3f}** | prose {v100:.3f}\n- AUC vs prose **{v93:.3f}**, vs off-tape entities **{v94:.3f}**\n- slot deleted -> gate {v98:.3f} -> **{v96:.3f}**; shuffled keys AUC {v95:.3f}\n- paired gap prob-space **{v102:.4f}** | win-rate **{v103:.3f}** | logit gap **{v104:.3f}**\n- feature probe: |d max|={v97.v341('max', {}).v341('abs_delta', v10('nan')):.4f} |d margin12|={v97.v341('margin12', {}).v341('abs_delta', v10('nan')):.4f} | gold top1 {v97.v341('gold_is_top1_with', v10('nan')):.2f} -> {v97.v341('gold_is_top1_without', v10('nan')):.2f}\n- paired win-rate: h+feat **{v89['h_feat']['paired_win_rate']:.3f}** | feat_only **{v89['feat_only']['paired_win_rate']:.3f}** | h_only {v89['h_only']['paired_win_rate']:.3f}\n- paired logit gap: {v89['h_feat']['paired_logit_gap']:.3f} / {v89['feat_only']['paired_logit_gap']:.3f} / {v89['h_only']['paired_logit_gap']:.3f} (prob gaps {v89['h_feat']['paired_gap_same_line']:.4f} / {v89['feat_only']['paired_gap_same_line']:.4f} / {v89['h_only']['paired_gap_same_line']:.4f})\n- AUC vs prose: {v89['h_feat']['auc_on_vs_prose']:.3f} / {v89['feat_only']['auc_on_vs_prose']:.3f} / {v89['h_only']['auc_on_vs_prose']:.3f}\n- G_h_only_flat: {v111}\n- false fire on prose: {v101:.3f} over {v92['n_prose_positions']} positions\n", encoding='utf-8')
    v119(v317.v272({'overall': v192, 'gates': v31['gates']}, indent=2))
    if not v62.v164:
        v6.v213.v131(exist_ok=True)
        v22.v273({'W_q': v39.v325.v342(), 'gate2': v40.v342(), 'log_tau': v39.v365.v199().v343(), 'stage': '260e', 'mode': 'h_feat'}, v6)
    return 0
if v117 == '__main__':
    raise v194(v274())