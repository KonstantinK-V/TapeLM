"""
Stage 260d — Give the gate features it can use, then prove which input carries the answer.

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

260d does three things:
  * gives the discriminative features explicitly - top1-top2 margin and max-minus-mean, which
    move most - and drops coverage, which is constant 1.0 here
  * z-scores the retrieval features over the fit set so they cannot be drowned by scale
  * runs the gate in three variants - h+feat, FEAT_ONLY, h_only - as one ablation

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

  python _stage260d_open_gate.py [--smoke]
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
v0 = v13('results')
v1 = v0 / 'stage260d_decision.json'
v2 = v0 / 'stage260d_mini.md'
v3 = v0 / '_stage260d_log.txt'
v4 = v13('checkpoints/stage191_p1_curve.pt')
v5 = v13('checkpoints/stage253_joint_l02.pt')
v6 = v13('checkpoints/stage260d_open_gate.pt')
v7 = v13('data/_wikitext103_train.txt')
v8 = 2603
v9 = ('max', 'mean', 'margin12', 'max_minus_mean', 'entropy')

def retrieval_feats(v14: v19.v10, v15: v19.v10) -> v19.v10:
    """The five numbers, with the two that actually moved in 260c's probe given EXPLICITLY.
    Coverage is gone: it measured 1.000 with the slot and 1.000 without it."""
    v16 = v14.v191().v111()
    v112, v113 = (v16.v188(), v16.v189())
    v17 = v16[0] - v16[1] if v16.v265() > 1 else v19.v190((), device=v16.v43)
    v18 = -(v313.v351(v15, -1) * v313.v352(v15, -1)).v307().v191()
    return v19.v198([v112, v113, v17, v112 - v113, v18]).v114(v15.v43)

class Gate2(v20.v11):
    """Read gate whose inputs are switchable, so the ablation is one flag rather than three
    scripts. Features are z-scored on the fit set: in 260c they were 4 raw scalars concatenated
    to 512 hidden dims and were simply drowned."""

    def __init__(v115, v85: v12, v89: v31, v43):
        v308().v192()
        assert v89 in ('h_feat', 'feat_only', 'h_only')
        v115.v89 = v89
        v116 = (0 if v89 == 'feat_only' else v85) + (0 if v89 == 'h_only' else v232(v9))
        v115.v117 = v20.v309(v20.v335(v116, 64), v20.v336(), v20.v335(64, 1)).v114(v43)
        v20.v266.v193(v115.v117[-1].v194)
        v20.v266.v195(v115.v117[-1].v196, -2.0)
        v115.v197('mu', v19.v190(v232(v9), device=v43))
        v115.v197('sd', v19.v267(v232(v9), device=v43))

    def fit_norm(v115, v55: v30[v19.v10]) -> None:
        if not v55:
            return
        v118 = v19.v198(v55)
        v115.v268.v199(v118.v189(0))
        v115.v200.v199(v118.v337(0).v269(0.001))

    def g(v115, v119: v19.v10, v120: v19.v10) -> v19.v10:
        v121 = (v120 - v115.v268) / v115.v200
        if v115.v89 == 'feat_only':
            v201 = v121
        elif v115.v89 == 'h_only':
            v201 = v119
        else:
            v201 = v19.v310([v119, v121], dim=-1)
        return v19.v311(v115.v117(v201)).v202(-1)

def log(v21: v31) -> None:
    v22 = v21 if v21.v203('\n') else v21 + '\n'
    try:
        v204(v22, end='', flush=True)
    except v122:
        v204(v22.v210('ascii', 'replace').v278('ascii'), end='', flush=True)
    v3.v205.v123(parents=True, exist_ok=True)
    with v3.v206('a', encoding='utf-8') as v57:
        v57.v207(v22)

def token_index_before_entity(v23, v24: v12) -> v12 | None:
    for v124, (v208, v209) in v125(v23.v126):
        if v208 == v24:
            return v124 - 1
        if v208 < v24 < v209:
            return v124 - 1 if v124 >= 1 else None
    return None

def filter_wiki_lines(v25: v30[v31], v26: v127, v27: v12) -> v30[v31]:
    v28 = []
    for v29 in v25:
        v23 = v26.v210(v29)
        v45 = [v172 for v172 in v23.v45 if v172 != v27]
        if v232(v45) == v232(v23.v45) and 8 <= v232(v45) <= v270:
            v28.v227(v29)
    return v28

def harvest(v32, v33: v128, v26: v127, v27: v12, v34: v12, v35: v129):
    """One (line, entity, anchor, key) per line. Position is found by character offset, so the
    eval point is the real place in a real sentence where that entity begins."""
    v28 = []
    for v29 in v32:
        if v232(v28) >= v34:
            break
        v23 = v26.v210(v29)
        for v21 in v271.v211(v29):
            v18 = v21.v272(1)
            if v232(v18) < 5 or v18 in v35:
                continue
            v273, v274 = (v188(0, v21.v312() - 120), v243(v232(v29), v21.v346() + 120))
            v212 = v33.v217(v29[v273:v274], exclude=v18)
            if v212 is None:
                continue
            v139 = [v148 for v148 in v279.v218(v29[v273:v21.v312()]) if v148 != v18]
            if not v139:
                continue
            v213 = v275(v23, v21.v312())
            if v213 is None or v213 < 1:
                continue
            v45 = [v172 for v172 in v23.v45 if v172 != v27]
            v214 = v313.v276(v33.v347([v139[-1]])[0] + v212, dim=-1)
            v215 = v33.v217(v29[v273:v21.v312()])
            v28.v227({'line': v29, 'ent': v18, 'anchor': v139[-1], 'ids': v45, 't_hit': v213, 'key': v214, 'pair_q': None if v215 is None else v313.v276(v33.v347([v139[-1]])[0] + v215, dim=-1)})
            v35.v277(v18)
            break
    return v28

@v19.v48()
def gate_profile(v36, v37, v38, v39, v26, v33, v40, v41, v27, v42, v43, v44):
    """g_t at every position of one natural line -> (g at the fact position, g elsewhere)."""
    v45 = v19.v130([v41['ids']], dtype=v19.v216, device=v43)
    v131, v132 = v133(v38, v39, v45, v27)
    v46 = v41['ids']
    v134, v135 = (None, [])
    for v47 in v136(1, v232(v46) - 1):
        v137 = v132[0, v47]
        v138 = v33.v217(v26.v278(v46[:v47 + 1][-40:]))
        if v138 is None:
            continue
        v139 = v279.v218(v26.v278(v46[:v47 + 1]))
        v140 = v313.v276(v33.v347([v139[-1]])[0] + v138, dim=-1) if v139 else v138
        v140 = v313.v276(v36.v314(v140.v338(0)), dim=-1)[0]
        v141 = v40.v62(v140, v44)
        if v141 is None:
            continue
        v14, v219 = v141
        v18 = v111(-(v313.v351(v137, -1) * v313.v352(v137, -1)).v307())
        v142 = v111(v37.v142(v131[0, v47], v280(v14, v137)))
        if v47 == v41['t_hit']:
            v134 = v142
        else:
            v135.v227(v142)
    return (v134, v135)

def train_batch(v36, v37, v38, v39, v26, v33, v40, v41, v27, v42, v43, v44, v49, v50: v143, v51: v111):
    """CE over the whole natural line through the gated mixture, plus L1 on the gate away from the
    entity position, plus direct supervision AT it: open when the tape holds this entity, shut
    when it does not. Off-tape lines are the negatives 260 never trained on."""
    v45 = v19.v130([v41['ids']], dtype=v19.v216, device=v43)
    v131, v132 = v133(v38, v39, v45, v27)
    v46 = v41['ids']
    v52 = []
    v53 = [v41['t_hit']] + v235.v220(v136(1, v232(v46) - 1), v243(6, v232(v46) - 2))
    for v47 in v53:
        v137 = v132[0, v47]
        v138 = v33.v217(v26.v278(v46[:v47 + 1][-40:]))
        if v138 is None:
            continue
        v139 = v279.v218(v26.v278(v46[:v47 + 1]))
        v140 = v313.v276(v33.v347([v139[-1]])[0] + v138, dim=-1) if v139 else v138
        v140 = v313.v276(v36.v314(v140.v338(0)), dim=-1)[0]
        v141 = v40.v62(v140, v44)
        if v141 is None:
            continue
        v14, v219 = v141
        v18 = v111(-(v313.v351(v137, -1) * v313.v352(v137, -1)).v307())
        v221, v222 = v223(v36, v40, v14, v219, v46[:v47 + 1], v42, v43)
        v142 = v37.v142(v131[0, v47], v280(v14, v137))
        v144 = v224(v137, v142, v221, v222)
        if v47 == v41['t_hit']:
            v225 = 1.0 if v50 else 0.0
            v226 = v51 * v313.v315(v142.v339(1e-06, 1 - 1e-06), v19.v130(v225, device=v43))
        else:
            v226 = v49 * v142
        v52.v227(-v144[v46[v47 + 1]] + v226)
    return v19.v198(v52).v189() if v52 else None

@v19.v48()
def feature_probe(v36, v38, v39, v26, v33, v40, v54, v27, v42, v43, v44):
    """The question 260/260b/260c could not answer: when the needed slot is dropped, do the gate's
    RETRIEVAL FEATURES move at all? Same line, same position, same h_t — only the bank differs.

    If they do not move, no amount of training can help: cosine max over a dense bank of similar
    entities is not a possession detector, and that is a statement about the substrate.
    If they do move and the gate ignores them, it is a training problem.
    """
    v55 = []
    for v56 in v54:
        v45 = v19.v130([v56['ids']], dtype=v19.v216, device=v43)
        v131, v132 = v133(v38, v39, v45, v27)
        v47 = v56['t_hit']
        v46 = v56['ids'][:v47 + 1]
        v138 = v33.v217(v26.v278(v46[-40:]))
        if v138 is None:
            continue
        v145 = v279.v218(v26.v278(v46))
        v140 = v313.v276(v33.v347([v145[-1]])[0] + v138, dim=-1) if v145 else v138
        v140 = v313.v276(v36.v314(v140.v338(0)), dim=-1)[0]
        v146 = v40.v228()
        v146.v229(v56['ent'])
        v147 = {}
        for v230, v231 in (('with', v40), ('without', v146)):
            v141 = v231.v62(v140, v44)
            if v141 is None:
                v147 = {}
                break
            v14, v219 = v141
            v281, v222 = v223(v36, v231, v14, v219, v46, v42, v43)
            v16 = v14.v191().v111()
            v147[v230] = {'max': v111(v16.v188()), 'mean': v111(v16.v189()), 'margin12': v111(v16[0] - v16[1]) if v16.v265() > 1 else 0.0, 'max_minus_mean': v111(v16.v188() - v16.v189()), 'cov': v111(v222), 'gold_is_top1': v12(v231.v348[v12(v219[0])] == v56['ent'])}
        if v147:
            v55.v227(v147)
    if not v55:
        return {'n': 0}
    v28 = {'n': v232(v55)}
    for v57 in ('max', 'mean', 'margin12', 'max_minus_mean', 'cov'):
        v148 = v282.v233([v249['with'][v57] for v249 in v55])
        v149 = v282.v233([v249['without'][v57] for v249 in v55])
        v28[v57] = {'with': v111(v148.v189()), 'without': v111(v149.v189()), 'delta': v111((v148 - v149).v189()), 'abs_delta': v111(v282.v328(v148 - v149).v189())}
    v28['gold_is_top1_with'] = v111(v282.v189([v249['with']['gold_is_top1'] for v249 in v55]))
    v28['gold_is_top1_without'] = v111(v282.v189([v249['without']['gold_is_top1'] for v249 in v55]))
    return v28

def main() -> v12:
    v58 = v234.v150()
    v58.v151('--smoke', action='store_true')
    v58.v151('--steps', type=v12, default=0)
    v58.v151('--topk', type=v12, default=8)
    v58.v151('--gate-l1', type=v111, default=0.02)
    v58.v151('--sup-w', type=v111, default=1.0, help='weight of the have/need supervision')
    v58.v151('--paired-frac', type=v111, default=0.6, help='fraction of steps that use the same-line slot-present/absent pair')
    v59 = v58.v152()
    v3.v153('', encoding='utf-8')
    v43 = v19.v43('cuda' if v19.v316.v283() else 'cpu')
    v60 = v235.v154(v8)
    v19.v155(v8)
    v61 = v156.v156()
    v44 = v59.v62
    v63 = v59.v63 or (600 if v59.v157 else 2500)
    v64 = 64 if v59.v157 else 300
    v65 = 24 if v59.v157 else 120
    v66 = 24 if v59.v157 else 120
    v67 = 64 if v59.v157 else 300
    v68 = 4000 if v59.v157 else 30000
    v158(f'Stage260d open gate start {v344.v330(v345.v331).v261()} device={v43} steps={v63}')
    v159, v159, v160, v161 = v162()
    v26 = v127.v163(v31(v284.v236))
    v42 = v26.v164()
    v27 = v26.v237(v238) or 0
    v39 = v317.v285(v26, v160, v27, v42).v114(v43)
    v69 = v5 if v5.v239() else v4
    v38 = v286(v161, v42).v114(v43)
    v38.v165(v19.v287(v69, map_location=v43, weights_only=False)['model'])
    v38.v166()
    for v70 in v38.v167():
        v70.v240(False)
    v71 = v286(v161, v42).v114(v43)
    v71.v165(v19.v287(v4, map_location=v43, weights_only=False)['model'])
    v71.v166()
    for v70 in v71.v167():
        v70.v240(False)
    v33 = v128(v71, v160, v43)
    with v7.v206('r', encoding='utf-8', errors='ignore') as v57:
        v168 = v57.v241(2000000 if v59.v157 else 10000000)
    v72 = [v289.v288() for v289 in v168.v318('\n') if v289.v288()][:v68 * 4]
    v32 = v242(v72, v26, v27)[:v68]
    v60.v169(v32)
    v158(f'  wiki lines token-fit (<={v270} tok): {v232(v32)}')
    v35: v129[v31] = v129()
    v73 = v170(v32, v33, v26, v27, v64, v35)
    v74 = v170(v32[v232(v32) // 3:], v33, v26, v27, v65, v35)
    v75 = v170(v32[v232(v32) // 2:], v33, v26, v27, v67, v35)
    v76 = v170(v32[2 * v232(v32) // 3:], v33, v26, v27, v66, v35)
    v158(f'  lines: fit={v232(v73)} off_fit={v232(v75)} eval_on={v232(v74)} eval_off={v232(v76)}')
    if v243(v232(v73), v232(v74), v232(v76)) < 4:
        v158('  not enough usable lines')
        return 1
    v77 = v73 + v74
    v78 = [v56['key'] for v56 in v77]
    v79 = [v56['ent'] for v56 in v77]
    v40 = v171(v19.v198(v78, 0).v114(v43), v79, v26, v27)
    v158(f'  tape slots={v232(v79)} (off-tape entities: {v232(v76)}, deliberately absent)')
    v80 = v40.v80.v111()
    v81 = [v56['pair_q'] for v56 in v73 if v56['pair_q'] is not None]
    v82 = [v172 for v172, v56 in v125(v73) if v56['pair_q'] is not None]
    v83 = v19.v198(v81).v114(v43).v111() if v81 else None
    v84 = v19.v130(v82, device=v43) if v82 else None
    v85 = 2 * (v38.v90.v244 // 2)

    def run_mode(v89: v31) -> v86:
        """One gate variant, trained and scored end to end. feat_only is the decisive arm: five
        numbers cannot read a sentence, so a paired gap there is possession detection."""
        v173 = v235.v154(v8 + {'h_feat': 0, 'feat_only': 1, 'h_only': 2}[v89])
        v19.v155(v8)
        v36 = v245(v85, v43)
        v37 = v246(v85, v89, v43)
        v55 = []
        with v19.v48():
            for v56 in v73[:32]:
                v45 = v19.v130([v56['ids']], dtype=v19.v216, device=v43)
                v131, v319 = v133(v38, v39, v45, v27)
                v47 = v56['t_hit']
                v138 = v33.v217(v26.v278(v56['ids'][:v47 + 1][-40:]))
                if v138 is None:
                    continue
                v145 = v279.v218(v26.v278(v56['ids'][:v47 + 1]))
                v140 = v313.v276(v33.v347([v145[-1]])[0] + v138, dim=-1) if v145 else v138
                v140 = v313.v276(v36.v314(v140.v338(0)), dim=-1)[0]
                v141 = v40.v62(v140, v44)
                if v141 is not None:
                    v55.v227(v280(v141[0], v319[0, v47]))
        v37.v247(v55)
        v174 = v36.v290() + v30(v37.v167())
        v175 = v19.v291.v248(v174, lr=0.003, weight_decay=0.01)
        for v176 in v136(1, v63 + 1):
            v249 = v173.v235()
            if v249 < v59.v292:
                v56 = v73[v173.v340(v232(v73))]
                v146 = v40.v228()
                v146.v229(v56['ent'])
                v293 = v320(v36, v37, v38, v39, v26, v33, v40, v56, v27, v42, v43, v44, v59.v49, True, v59.v51)
                v294 = v320(v36, v37, v38, v39, v26, v33, v146, v56, v27, v42, v43, v44, v59.v49, False, v59.v51)
                v295 = None if v293 is None else v293 if v294 is None else v293 + v294
            else:
                v296 = v249 < v59.v292 + (1 - v59.v292) / 2 or not v75
                v56 = v73[v173.v340(v232(v73))] if v296 else v75[v173.v340(v232(v75))]
                v295 = v320(v36, v37, v38, v39, v26, v33, v40, v56, v27, v42, v43, v44, v59.v49, v296, v59.v51)
            if v295 is None:
                continue
            if v83 is not None:
                v297 = v19.v321(0, v83.v341(0), (v243(32, v83.v341(0)),), device=v43)
                v140 = v313.v276(v36.v314(v83[v297]), dim=-1)
                v295 = v295 + v313.v342(v140 @ v80.v47() / 0.05, v84[v297])
            v175.v298(set_to_none=True)
            v295.v299()
            v19.v20.v322.v300(v174, 1.0)
            v175.v176()
            if v176 % v188(1, v63 // 3) == 0:
                v158(f'  [{v89}] step {v176}/{v63} loss={v111(v295):.3f} ({v156.v156() - v61:.0f}s)')
        v36.v166()
        v37.v166()

        def profile(v54, v231=v40):
            v301, v302 = ([], [])
            for v56 in v54:
                v303, v323 = v304(v36, v37, v38, v39, v26, v33, v231, v56, v27, v42, v43, v44)
                if v303 is not None:
                    v301.v227(v303)
                v302.v324(v323)
            return (v282.v325(v301), v282.v325(v302))
        v250, v251 = v252(v74)
        v253, v159 = v252(v76)
        v177 = []
        for v56 in v74:
            v146 = v40.v228()
            v146.v229(v56['ent'])
            v303, v159 = v304(v36, v37, v38, v39, v26, v33, v146, v56, v27, v42, v43, v44)
            if v303 is not None:
                v177.v227(v303)
        v178 = v111(v282.v189(v177)) if v177 else v111('nan')
        v254, v255 = v252(v74, tp=v40.v326(v8 + 1))
        v179 = v111(v282.v189(v250)) if v232(v250) else v111('nan')
        v180 = {'mode': v89, 'gate_on_tape': v179, 'gate_off_tape': v111(v282.v189(v253)) if v232(v253) else v111('nan'), 'gate_prose': v111(v282.v189(v251)) if v232(v251) else v111('nan'), 'auc_on_vs_prose': v327(v250, v251) if v232(v250) and v232(v251) else v111('nan'), 'auc_on_vs_off_tape': v327(v250, v253) if v232(v250) and v232(v253) else v111('nan'), 'auc_shuffled_keys': v327(v254, v255) if v232(v254) and v232(v255) else v111('nan'), 'gate_after_slot_delete': v178, 'paired_gap_same_line': v179 - v178, 'false_fire_rate_prose': v111(v282.v189(v251 > 0.5)) if v232(v251) else v111('nan'), 'n_prose_positions': v12(v232(v251))}
        v180['gate_reads_tape'] = v143(v328(v180['auc_on_vs_prose'] - v180['auc_shuffled_keys']) > 1e-06)
        v158(f'[{v89}] ' + v306.v262({v343: v349(v350, 4) for v343, v350 in v180.v54() if v353(v350, v111)}))
        return (v180, v36, v37, v250, v251)
    v87 = {}
    v88 = {}
    for v89 in ('h_feat', 'feat_only', 'h_only'):
        v181, v256, v257, v258, v259 = v260(v89)
        v87[v89] = v181
        v88[v89] = (v256, v257, v258, v259)
    v36, v37, v182, v183 = v88['h_feat']
    v90 = v87['h_feat']
    v91 = v90['auc_on_vs_prose']
    v92 = v90['auc_on_vs_off_tape']
    v93 = v90['auc_shuffled_keys']
    v94 = v90['gate_after_slot_delete']
    v95 = v184(v36, v38, v39, v26, v33, v40, v74, v27, v42, v43, v44)
    v158('feature probe (slot present vs dropped, same position): ' + v306.v262(v95))
    v96 = v90['gate_on_tape']
    v97 = v90['gate_off_tape']
    v98 = v90['gate_prose']
    v99 = v90['false_fire_rate_prose']
    v158(f'gate: on_tape={v96:.3f} off_tape={v97:.3f} prose={v98:.3f} | AUC vs prose={v91:.3f} vs off_tape={v92:.3f} | after delete={v94:.3f} | shuffled AUC={v93:.3f}')
    v100 = v90['paired_gap_same_line']
    v101 = v87['feat_only']['paired_gap_same_line']
    v102 = v87['h_only']['paired_gap_same_line']
    v103 = not v329.v305(v100) and v100 >= 0.15
    v104 = not v329.v305(v101) and v101 >= 0.15
    v105 = v91 >= 0.85
    v106 = v92 >= 0.7
    v107 = v98 <= 0.05 and v99 <= 0.05
    v108 = not v329.v305(v94) and v94 <= v188(0.1, v96 - 0.3)
    v109 = not v329.v305(v93) and v93 <= 0.65
    if (v103 or v104) and v105 and v106 and v107 and v108 and v109:
        v185 = 'OPEN_GATE4_OK'
    elif v105 and v107 and (not v103 and (not v104)):
        v185 = 'OPEN_GATE4_POSITIONAL'
    else:
        v185 = 'OPEN_GATE4_NO'
    v28 = {'stage': '260d', 'overall': v185, 'trunk': v69.v186, 'steps': v63, 'topk': v44, 'n_fit': v232(v73), 'n_eval_on_tape': v232(v74), 'n_eval_off_tape': v232(v76), 'tape_slots': v232(v79), 'gates': {'G_auc_vs_prose': v105, 'G_auc_vs_off_tape': v106, 'G_paired_same_line': v103, 'G_feat_only_carries_it': v104, 'G_quiet_on_prose': v107, 'G_delete_silences': v108, 'G_tape_causal': v109}, 'summary': {'gate_on_tape': v96, 'gate_off_tape': v97, 'gate_prose': v98, 'auc_on_vs_prose': v91, 'auc_on_vs_off_tape': v92, 'auc_shuffled_keys': v93, 'gate_after_slot_delete': v94, 'false_fire_rate_prose': v99, 'n_prose_positions': v90['n_prose_positions'], 'gate_reads_tape': v143(v328(v91 - v93) > 1e-06), 'paired_gap_same_line': v100, 'per_mode': v87, 'decisive_feat_only_gap': v87['feat_only']['paired_gap_same_line'], 'h_only_gap': v87['h_only']['paired_gap_same_line'], 'feature_probe': v95, 'features_move': v143(v95.v332('n', 0) > 0 and v188((v95[v57]['abs_delta'] for v57 in ('max', 'margin12', 'max_minus_mean', 'cov'))) > 0.01), 'prior_260': {'auc_vs_prose': 0.5741, 'auc_vs_off_tape': 0.4416, 'after_delete': 0.199, 'shuffled_auc': 0.5741, 'why': 'trained on on-tape lines only; gate was a function of h_t alone'}}, 'note': '260d: Gate2 with z-scored retrieval feats (max, mean, margin12, max-minus-mean, entropy; no cov). Three input ablations in one run: h_feat, feat_only, h_only. G_feat_only_carries_it: feat_only paired_gap≥0.15 proves possession without h_t. If h_only AUC≈h_feat, earlier NOs were line classification. feature_probe from 260c.', 'timestamp': v344.v330(v345.v331).v261(), 'wall_s': v156.v156() - v61}
    v1.v153(v306.v262(v28, indent=2), encoding='utf-8')
    v2.v153(f"# Stage 260d open-text gate (feature fix + input ablation)\n\n**{v185}** slots={v232(v79)} eval={v232(v74)} on / {v232(v76)} off\n\n- gate: on-tape **{v96:.3f}** | off-tape **{v97:.3f}** | prose {v98:.3f}\n- AUC vs prose **{v91:.3f}**, vs off-tape entities **{v92:.3f}**\n- slot deleted -> gate {v96:.3f} -> **{v94:.3f}**; shuffled keys AUC {v93:.3f}\n- paired gap (on - delete) **{v100:.4f}**\n- feature probe, slot present vs dropped: |d max|={v95.v332('max', {}).v332('abs_delta', v111('nan')):.4f} |d margin12|={v95.v332('margin12', {}).v332('abs_delta', v111('nan')):.4f} |d cov|={v95.v332('cov', {}).v332('abs_delta', v111('nan')):.4f} | gold top1 {v95.v332('gold_is_top1_with', v111('nan')):.2f} -> {v95.v332('gold_is_top1_without', v111('nan')):.2f}\n- paired gap by input: h+feat **{v87['h_feat']['paired_gap_same_line']:.4f}** | **feat_only {v87['feat_only']['paired_gap_same_line']:.4f}** | h_only {v87['h_only']['paired_gap_same_line']:.4f}\n- AUC vs prose by input: {v87['h_feat']['auc_on_vs_prose']:.3f} / {v87['feat_only']['auc_on_vs_prose']:.3f} / {v87['h_only']['auc_on_vs_prose']:.3f}\n- false fire on prose: {v99:.3f} over {v90['n_prose_positions']} positions\n", encoding='utf-8')
    v158(v306.v262({'overall': v185, 'gates': v28['gates']}, indent=2))
    if not v59.v157:
        v6.v205.v123(exist_ok=True)
        v19.v263({'W_q': v36.v314.v333(), 'gate2': v37.v333(), 'log_tau': v36.v354.v191().v334(), 'stage': '260d', 'mode': 'h_feat'}, v6)
    return 0
if v110 == '__main__':
    raise v187(v264())