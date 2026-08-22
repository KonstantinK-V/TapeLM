"""
Stage 260c — Open-text gate, with the ONE contrast h_t cannot solve.

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

  python _stage260c_open_gate.py [--smoke]

After eval, feature_probe logs retrieval features (max, margin12, cov, …) with vs without the
needed slot at t_hit; summary.features_move reads the verdict (|delta| > 0.01 on key feats).
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
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import auc
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, SlotBias, TapeView, copy_dist, hidden_and_logits, mix_logprob
v0 = v11('results')
v1 = v0 / 'stage260c_decision.json'
v2 = v0 / 'stage260c_mini.md'
v3 = v0 / '_stage260c_log.txt'
v4 = v11('checkpoints/stage191_p1_curve.pt')
v5 = v11('checkpoints/stage253_joint_l02.pt')
v6 = v11('checkpoints/stage260c_open_gate.pt')
v7 = v11('data/_wikitext103_train.txt')
v8 = 2602

def log(v12: v22) -> None:
    v13 = v12 if v12.v175('\n') else v12 + '\n'
    try:
        v176(v13, end='', flush=True)
    except v101:
        v176(v13.v182('ascii', 'replace').v251('ascii'), end='', flush=True)
    v3.v177.v102(parents=True, exist_ok=True)
    with v3.v178('a', encoding='utf-8') as v47:
        v47.v179(v13)

def token_index_before_entity(v14, v15: v10) -> v10 | None:
    for v103, (v180, v181) in v104(v14.v105):
        if v180 == v15:
            return v103 - 1
        if v180 < v15 < v181:
            return v103 - 1 if v103 >= 1 else None
    return None

def filter_wiki_lines(v16: v21[v22], v17: v106, v18: v10) -> v21[v22]:
    v19 = []
    for v20 in v16:
        v14 = v17.v182(v20)
        v35 = [v157 for v157 in v14.v35 if v157 != v18]
        if v209(v35) == v209(v14.v35) and 8 <= v209(v35) <= v243:
            v19.v201(v20)
    return v19

def harvest(v23, v24: v107, v17: v106, v18: v10, v25: v10, v26: v108):
    """One (line, entity, anchor, key) per line. Position is found by character offset, so the
    eval point is the real place in a real sentence where that entity begins."""
    v19 = []
    for v20 in v23:
        if v209(v19) >= v25:
            break
        v14 = v17.v182(v20)
        for v12 in v244.v183(v20):
            v121 = v12.v245(1)
            if v209(v121) < 5 or v121 in v26:
                continue
            v246, v247 = (v254(0, v12.v276() - 120), v221(v209(v20), v12.v302() + 120))
            v184 = v24.v189(v20[v246:v247], exclude=v121)
            if v184 is None:
                continue
            v118 = [v129 for v129 in v252.v190(v20[v246:v12.v276()]) if v129 != v121]
            if not v118:
                continue
            v185 = v248(v14, v12.v276())
            if v185 is None or v185 < 1:
                continue
            v35 = [v157 for v157 in v14.v35 if v157 != v18]
            v186 = v277.v249(v24.v303([v118[-1]])[0] + v184, dim=-1)
            v187 = v24.v189(v20[v246:v12.v276()])
            v19.v201({'line': v20, 'ent': v121, 'anchor': v118[-1], 'ids': v35, 't_hit': v185, 'key': v186, 'pair_q': None if v187 is None else v277.v249(v24.v303([v118[-1]])[0] + v187, dim=-1)})
            v26.v250(v121)
            break
    return v19

@v123.v38()
def gate_profile(v27, v28, v29, v17, v24, v30, v31, v18, v32, v33, v34):
    """g_t at every position of one natural line -> (g at the fact position, g elsewhere)."""
    v35 = v123.v109([v31['ids']], dtype=v123.v188, device=v33)
    v110, v111 = v112(v28, v29, v35, v18)
    v36 = v31['ids']
    v113, v114 = (None, [])
    for v37 in v115(1, v209(v36) - 1):
        v116 = v111[0, v37]
        v117 = v24.v189(v17.v251(v36[:v37 + 1][-40:]))
        if v117 is None:
            continue
        v118 = v252.v190(v17.v251(v36[:v37 + 1]))
        v119 = v277.v249(v24.v303([v118[-1]])[0] + v117, dim=-1) if v118 else v117
        v119 = v277.v249(v27.v278(v119.v296(0)), dim=-1)[0]
        v120 = v30.v54(v119, v34)
        if v120 is None:
            continue
        v191, v192 = v120
        v121 = v124(-(v277.v306(v116, -1) * v277.v307(v116, -1)).v279())
        v193, v194 = v195(v27, v30, v191, v192, v36[:v37 + 1], v32, v33)
        v122 = v124(v27.v122(v110[0, v37], v124(v191.v254()), v124(v191.v202()), v121, v194))
        if v37 == v31['t_hit']:
            v113 = v122
        else:
            v114.v201(v122)
    return (v113, v114)

def train_batch(v27, v28, v29, v17, v24, v30, v31, v18, v32, v33, v34, v39, v40: v9, v41: v124):
    """CE over the whole natural line through the gated mixture, plus L1 on the gate away from the
    entity position, plus direct supervision AT it: open when the tape holds this entity, shut
    when it does not. Off-tape lines are the negatives 260 never trained on."""
    v35 = v123.v109([v31['ids']], dtype=v123.v188, device=v33)
    v110, v111 = v112(v28, v29, v35, v18)
    v36 = v31['ids']
    v42 = []
    v43 = [v31['t_hit']] + v213.v196(v115(1, v209(v36) - 1), v221(6, v209(v36) - 2))
    for v37 in v43:
        v116 = v111[0, v37]
        v117 = v24.v189(v17.v251(v36[:v37 + 1][-40:]))
        if v117 is None:
            continue
        v118 = v252.v190(v17.v251(v36[:v37 + 1]))
        v119 = v277.v249(v24.v303([v118[-1]])[0] + v117, dim=-1) if v118 else v117
        v119 = v277.v249(v27.v278(v119.v296(0)), dim=-1)[0]
        v120 = v30.v54(v119, v34)
        if v120 is None:
            continue
        v191, v192 = v120
        v121 = v124(-(v277.v306(v116, -1) * v277.v307(v116, -1)).v279())
        v197, v194 = v195(v27, v30, v191, v192, v36[:v37 + 1], v32, v33)
        v122 = v27.v122(v110[0, v37], v124(v191.v254()), v124(v191.v202()), v121, v194)
        v125 = v198(v116, v122, v197, v194)
        if v37 == v31['t_hit']:
            v199 = 1.0 if v40 else 0.0
            v200 = v41 * v277.v280(v122.v297(1e-06, 1 - 1e-06), v123.v109(v199, device=v33))
        else:
            v200 = v39 * v122
        v42.v201(-v125[v36[v37 + 1]] + v200)
    return v123.v281(v42).v202() if v42 else None

@v123.v38()
def feature_probe(v27, v28, v29, v17, v24, v30, v44, v18, v32, v33, v34):
    """Same line, same t_hit, same h_t — only the bank differs (slot present vs dropped)."""
    v45 = []
    for v46 in v44:
        v35 = v123.v109([v46['ids']], dtype=v123.v188, device=v33)
        v203, v204 = v112(v28, v29, v35, v18)
        v37 = v46['t_hit']
        v36 = v46['ids'][:v37 + 1]
        v117 = v24.v189(v17.v251(v36[-40:]))
        if v117 is None:
            continue
        v126 = v252.v190(v17.v251(v36))
        v119 = v277.v249(v24.v303([v126[-1]])[0] + v117, dim=-1) if v126 else v117
        v119 = v277.v249(v27.v278(v119.v296(0)), dim=-1)[0]
        v127 = v30.v205()
        v127.v206(v46['ent'])
        v128 = {}
        for v207, v159 in (('with', v30), ('without', v127)):
            v120 = v159.v54(v119, v34)
            if v120 is None:
                v128 = {}
                break
            v191, v192 = v120
            v193, v194 = v195(v27, v159, v191, v192, v36, v32, v33)
            v208 = v191.v298().v124()
            v128[v207] = {'max': v124(v208.v254()), 'mean': v124(v208.v202()), 'margin12': v124(v208[0] - v208[1]) if v208.v304() > 1 else 0.0, 'max_minus_mean': v124(v208.v254() - v208.v202()), 'cov': v124(v194), 'gold_is_top1': v10(v159.v305[v10(v192[0])] == v46['ent'])}
        if v128:
            v45.v201(v128)
    if not v45:
        return {'n': 0}
    v19 = {'n': v209(v45)}
    for v47 in ('max', 'mean', 'margin12', 'max_minus_mean', 'cov'):
        v129 = v253.v210([v158['with'][v47] for v158 in v45])
        v130 = v253.v210([v158['without'][v47] for v158 in v45])
        v19[v47] = {'with': v124(v129.v202()), 'without': v124(v130.v202()), 'delta': v124((v129 - v130).v202()), 'abs_delta': v124(v253.v270(v129 - v130).v202())}
    v19['gold_is_top1_with'] = v124(v253.v202([v158['with']['gold_is_top1'] for v158 in v45]))
    v19['gold_is_top1_without'] = v124(v253.v202([v158['without']['gold_is_top1'] for v158 in v45]))
    return v19

def probe_features_move(v48: v97, *, v49: v124=0.01) -> v9:
    if v48.v211('n', 0) <= 0:
        return False
    try:
        return v254((v48[v47]['abs_delta'] for v47 in ('max', 'margin12', 'max_minus_mean', 'cov'))) > v49
    except v131:
        return False

def main() -> v10:
    v50 = v212.v132()
    v50.v133('--smoke', action='store_true')
    v50.v133('--steps', type=v10, default=0)
    v50.v133('--topk', type=v10, default=8)
    v50.v133('--gate-l1', type=v124, default=0.02)
    v50.v133('--sup-w', type=v124, default=1.0, help='weight of the have/need supervision')
    v50.v133('--paired-frac', type=v124, default=0.6, help='fraction of steps that use the same-line slot-present/absent pair')
    v51 = v50.v134()
    v3.v135('', encoding='utf-8')
    v33 = v123.v33('cuda' if v123.v282.v255() else 'cpu')
    v52 = v213.v136(v8)
    v123.v137(v8)
    v53 = v138.v138()
    v34 = v51.v54
    v55 = v51.v55 or (600 if v51.v139 else 2500)
    v56 = 64 if v51.v139 else 300
    v57 = 24 if v51.v139 else 120
    v58 = 24 if v51.v139 else 120
    v59 = 64 if v51.v139 else 300
    v60 = 4000 if v51.v139 else 30000
    v140(f'Stage260c open gate start {v299.v292(v300.v293).v239()} device={v33} steps={v55} paired_frac={v51.v173}')
    v141, v141, v142, v143 = v144()
    v17 = v106.v145(v22(v256.v214))
    v32 = v17.v146()
    v18 = v17.v215(v216) or 0
    v29 = v283.v257(v17, v142, v18, v32).v147(v33)
    v61 = v5 if v5.v217() else v4
    v28 = v258(v143, v32).v147(v33)
    v28.v148(v123.v259(v61, map_location=v33, weights_only=False)['model'])
    v28.v149()
    for v62 in v28.v150():
        v62.v218(False)
    v63 = v258(v143, v32).v147(v33)
    v63.v148(v123.v259(v4, map_location=v33, weights_only=False)['model'])
    v63.v149()
    for v62 in v63.v150():
        v62.v218(False)
    v24 = v107(v63, v142, v33)
    with v7.v178('r', encoding='utf-8', errors='ignore') as v47:
        v151 = v47.v219(2000000 if v51.v139 else 10000000)
    v64 = [v261.v260() for v261 in v151.v284('\n') if v261.v260()][:v60 * 4]
    v23 = v220(v64, v17, v18)[:v60]
    v52.v152(v23)
    v140(f'  wiki lines token-fit (<={v243} tok): {v209(v23)}')
    v26: v108[v22] = v108()
    v65 = v153(v23, v24, v17, v18, v56, v26)
    v66 = v153(v23[v209(v23) // 3:], v24, v17, v18, v57, v26)
    v67 = v153(v23[v209(v23) // 2:], v24, v17, v18, v59, v26)
    v68 = v153(v23[2 * v209(v23) // 3:], v24, v17, v18, v58, v26)
    v140(f'  lines: fit={v209(v65)} off_fit={v209(v67)} eval_on={v209(v66)} eval_off={v209(v68)}')
    if v221(v209(v65), v209(v66), v209(v68)) < 4:
        v140('  not enough usable lines')
        return 1
    v69 = v65 + v66
    v70 = [v46['key'] for v46 in v69]
    v71 = [v46['ent'] for v46 in v69]
    v30 = v154(v123.v281(v70, 0).v147(v33), v71, v17, v18)
    v140(f'  tape slots={v209(v71)} (off-tape entities: {v209(v68)}, deliberately absent)')
    v27 = v155(2 * (v28.v285.v262 // 2), v33)
    v72 = v123.v222.v156(v27.v223(), lr=0.003, weight_decay=0.01)
    v73 = v30.v73.v124()
    v74 = [v46['pair_q'] for v46 in v65 if v46['pair_q'] is not None]
    v75 = [v157 for v157, v46 in v104(v65) if v46['pair_q'] is not None]
    v76 = v123.v281(v74).v147(v33).v124() if v74 else None
    v77 = v123.v109(v75, device=v33) if v75 else None
    for v78 in v115(1, v55 + 1):
        v158 = v52.v213()
        if v158 < v51.v173:
            v46 = v65[v52.v286(v209(v65))]
            v127 = v30.v205()
            v127.v206(v46['ent'])
            v224 = v263(v27, v28, v29, v17, v24, v30, v46, v18, v32, v33, v34, v51.v39, True, v51.v41)
            v225 = v263(v27, v28, v29, v17, v24, v127, v46, v18, v32, v33, v34, v51.v39, False, v51.v41)
            v226 = None if v224 is None else v224 if v225 is None else v224 + v225
        else:
            v227 = v158 < v51.v173 + (1 - v51.v173) / 2 or not v67
            v46 = v65[v52.v286(v209(v65))] if v227 else v67[v52.v286(v209(v67))]
            v226 = v263(v27, v28, v29, v17, v24, v30, v46, v18, v32, v33, v34, v51.v39, v227, v51.v41)
        if v226 is None:
            continue
        if v76 is not None:
            v228 = v123.v264(0, v76.v287(0), (v221(32, v76.v287(0)),), device=v33)
            v119 = v277.v249(v27.v278(v76[v228]), dim=-1)
            v226 = v226 + v277.v288(v119 @ v73.v37() / 0.05, v77[v228])
        v72.v229(set_to_none=True)
        v226.v230()
        v123.v289.v265.v231(v27.v223(), 1.0)
        v72.v78()
        if v78 % v254(1, v55 // 5) == 0:
            v140(f'  step {v78}/{v55} loss={v124(v226):.3f} ({v138.v138() - v53:.0f}s)')
    v27.v149()

    def profile(v44, v159=v30):
        v232, v233 = ([], [])
        for v46 in v44:
            v235, v266 = v236(v27, v28, v29, v17, v24, v159, v46, v18, v32, v33, v34)
            if v235 is not None:
                v232.v201(v235)
            v233.v267(v266)
        return (v253.v268(v232), v253.v268(v233))
    v160, v161 = v162(v66)
    v163, v141 = v162(v68)
    v79 = v234(v160, v161) if v209(v160) and v209(v161) else v124('nan')
    v80 = v234(v160, v163) if v209(v160) and v209(v163) else v124('nan')
    v81 = []
    for v46 in v66:
        v127 = v30.v205()
        v127.v206(v46['ent'])
        v235, v141 = v236(v27, v28, v29, v17, v24, v127, v46, v18, v32, v33, v34)
        if v235 is not None:
            v81.v201(v235)
    v82 = v124(v253.v202(v81)) if v81 else v124('nan')
    v48 = v164(v27, v28, v29, v17, v24, v30, v66, v18, v32, v33, v34)
    v83 = v165(v48)
    v140('feature probe (slot present vs dropped, same position): ' + v275.v240(v48))
    v166, v167 = v162(v66, tp=v30.v269(v8 + 1))
    v84 = v234(v166, v167) if v209(v166) and v209(v167) else v124('nan')
    v85 = v124(v253.v202(v160)) if v209(v160) else v124('nan')
    v86 = v124(v253.v202(v163)) if v209(v163) else v124('nan')
    v87 = v124(v253.v202(v161)) if v209(v161) else v124('nan')
    v88 = v124(v253.v202(v161 > 0.5)) if v209(v161) else v124('nan')
    v140(f'gate: on_tape={v85:.3f} off_tape={v86:.3f} prose={v87:.3f} | AUC vs prose={v79:.3f} vs off_tape={v80:.3f} | after delete={v82:.3f} | shuffled AUC={v84:.3f}')
    v89 = v85 - v82
    v90 = v9(v270(v79 - v84) > 1e-06)
    v91 = not v290.v271(v82) and v89 >= 0.15
    v92 = v79 >= 0.85
    v93 = v80 >= 0.7
    v94 = v87 <= 0.05 and v88 <= 0.05
    v95 = not v290.v271(v82) and v82 <= v254(0.1, v85 - 0.3)
    v96 = not v290.v271(v84) and v84 <= 0.65
    if v91 and v92 and v93 and v94 and v95 and v96:
        v168 = 'OPEN_GATE3_OK'
    elif v92 and v94 and (not v91 or not v90):
        v168 = 'OPEN_GATE3_POSITIONAL'
    else:
        v168 = 'OPEN_GATE3_NO'

    def _prior(v169: v11, v170: v97) -> v97:
        if not v169.v217():
            return v170
        try:
            v237 = v275.v272(v169.v291(encoding='utf-8'))
            v238 = v237.v211('summary') or {}
            return {'overall': v237.v211('overall'), 'auc_vs_prose': v238.v211('auc_on_vs_prose', v170.v211('auc_vs_prose')), 'auc_vs_off_tape': v238.v211('auc_on_vs_off_tape', v170.v211('auc_vs_off_tape')), 'after_delete': v238.v211('gate_after_slot_delete', v170.v211('after_delete')), 'shuffled_auc': v238.v211('auc_shuffled_keys', v170.v211('shuffled_auc')), 'paired_gap': v238.v211('paired_gap_same_line', v238.v211('paired_gap_same_line')), 'gate_reads_tape': v238.v211('gate_reads_tape')}
        except (v275.v273, v274):
            return v170
    v98 = v171(v0 / 'stage260_decision.json', {'auc_vs_prose': 0.5741, 'auc_vs_off_tape': 0.4416, 'after_delete': 0.199, 'shuffled_auc': 0.5741})
    v99 = v171(v0 / 'stage260b_decision.json', {'auc_vs_prose': 0.807, 'auc_vs_off_tape': 0.657, 'after_delete': 0.235, 'shuffled_auc': 0.8066, 'paired_gap': 0.002, 'gate_reads_tape': False})
    v19 = {'stage': '260c', 'overall': v168, 'trunk': v61.v172, 'steps': v55, 'topk': v34, 'paired_frac': v51.v173, 'n_fit': v209(v65), 'n_eval_on_tape': v209(v66), 'n_eval_off_tape': v209(v68), 'tape_slots': v209(v71), 'gates': {'G_auc_vs_prose': v92, 'G_auc_vs_off_tape': v93, 'G_paired_same_line': v91, 'G_quiet_on_prose': v94, 'G_delete_silences': v95, 'G_tape_causal': v96}, 'summary': {'gate_on_tape': v85, 'gate_off_tape': v86, 'gate_prose': v87, 'auc_on_vs_prose': v79, 'auc_on_vs_off_tape': v80, 'auc_shuffled_keys': v84, 'gate_after_slot_delete': v82, 'false_fire_rate_prose': v88, 'n_prose_positions': v10(v209(v161)), 'gate_reads_tape': v90, 'paired_gap_same_line': v89, 'feature_probe': v48, 'features_move': v83, 'prior_260': v98, 'prior_260b': v99}, 'note': '260c: paired same-line train (h_t fixed); feature_probe at eval t_hit with vs without the needed slot. features_move=false => cosine features do not detect possession (substrate limit; bears on 212b). features_move=true but paired_gap~0 => train the gate. gold_is_top1_with low => retrieval misses gold even with slot present — rebuild exam. G_paired_same_line ≥ 0.15; gate_reads_tape = real vs shuffled AUC differ.', 'timestamp': v299.v292(v300.v293).v239(), 'wall_s': v138.v138() - v53}
    v1.v135(v275.v240(v19, indent=2), encoding='utf-8')
    v2.v135(f"# Stage 260c open-text gate (paired same-line contrast)\n\n**{v168}** slots={v209(v71)} eval={v209(v66)} on / {v209(v68)} off\n\n- gate: on-tape **{v85:.3f}** | off-tape **{v86:.3f}** | prose {v87:.3f}\n- AUC vs prose **{v79:.3f}**, vs off-tape entities **{v80:.3f}**\n- slot deleted -> gate {v85:.3f} -> **{v82:.3f}**; shuffled keys AUC {v84:.3f}\n- paired gap (on − delete) **{v89:.4f}** | gate_reads_tape={v90}\n- **features_move={v83}** | probe |d max|={v48.v211('max', {}).v211('abs_delta', v124('nan')):.4f} |d margin12|={v48.v211('margin12', {}).v211('abs_delta', v124('nan')):.4f} |d cov|={v48.v211('cov', {}).v211('abs_delta', v124('nan')):.4f} | gold top1 {v48.v211('gold_is_top1_with', v124('nan')):.2f} → {v48.v211('gold_is_top1_without', v124('nan')):.2f}\n- false fire on prose: {v88:.3f} over {v209(v161)} positions\n", encoding='utf-8')
    v140(v275.v240({'overall': v168, 'gates': v19['gates']}, indent=2))
    if not v51.v139:
        v6.v177.v102(exist_ok=True)
        v123.v241({'W_q': v27.v278.v294(), 'gate': v27.v301.v294(), 'log_tau': v27.v308.v298().v295(), 'stage': 260}, v6)
    return 0
if v100 == '__main__':
    raise v174(v242())