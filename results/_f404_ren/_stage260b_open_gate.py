"""
Stage 260b — Open-text gate, with the contrast the first run never saw.

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

260b adds the missing half of the exam:
  * off-tape lines are TRAINED on too, with the gate pushed shut at their entity position
  * the gate gets direct supervision (open at on-tape hits, shut at off-tape hits) alongside CE
  * more fit lines and steps — 260's loss never converged (3.37 -> 1.52 -> 4.51 -> 4.63 -> 2.89)

The claim is therefore narrower and honest: the gate CAN be taught have-versus-need from trunk
state plus retrieval features, and the test is whether that transfers to held-out lines. It is
not a claim that the distinction emerges from next-token CE on its own.

Trunk and P1 frozen; only W_q, the gate and tau train — same contract as 256. Fit lines and
eval lines are disjoint, so the gate is never scored where it was fit.

  python _stage260b_open_gate.py [--smoke]
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
v0 = v10('results')
v1 = v0 / 'stage260b_decision.json'
v2 = v0 / 'stage260b_mini.md'
v3 = v0 / '_stage260b_log.txt'
v4 = v10('checkpoints/stage191_p1_curve.pt')
v5 = v10('checkpoints/stage253_joint_l02.pt')
v6 = v10('checkpoints/stage260b_open_gate.pt')
v7 = v10('data/_wikitext103_train.txt')
v8 = 2601

def log(v11: v21) -> None:
    v12 = v11 if v11.v158('\n') else v11 + '\n'
    try:
        v159(v12, end='', flush=True)
    except v90:
        v159(v12.v165('ascii', 'replace').v225('ascii'), end='', flush=True)
    v3.v160.v91(parents=True, exist_ok=True)
    with v3.v161('a', encoding='utf-8') as v92:
        v92.v162(v12)

def token_index_before_entity(v13, v14: v9) -> v9 | None:
    for v93, (v163, v164) in v94(v13.v95):
        if v163 == v14:
            return v93 - 1
        if v163 < v14 < v164:
            return v93 - 1 if v93 >= 1 else None
    return None

def filter_wiki_lines(v15: v20[v21], v16: v96, v17: v9) -> v20[v21]:
    v18 = []
    for v19 in v15:
        v13 = v16.v165(v19)
        v34 = [v142 for v142 in v13.v34 if v142 != v17]
        if v209(v34) == v209(v13.v34) and 8 <= v209(v34) <= v216:
            v18.v184(v19)
    return v18

def harvest(v22, v23: v97, v16: v96, v17: v9, v24: v9, v25: v98):
    """One (line, entity, anchor, key) per line. Position is found by character offset, so the
    eval point is the real place in a real sentence where that entity begins."""
    v18 = []
    for v19 in v22:
        if v209(v18) >= v24:
            break
        v13 = v16.v165(v19)
        for v11 in v217.v166(v19):
            v111 = v11.v218(1)
            if v209(v111) < 5 or v111 in v25:
                continue
            v219, v220 = (v242(0, v11.v247() - 120), v194(v209(v19), v11.v275() + 120))
            v167 = v23.v172(v19[v219:v220], exclude=v111)
            if v167 is None:
                continue
            v108 = [v221 for v221 in v226.v173(v19[v219:v11.v247()]) if v221 != v111]
            if not v108:
                continue
            v168 = v222(v13, v11.v247())
            if v168 is None or v168 < 1:
                continue
            v34 = [v142 for v142 in v13.v34 if v142 != v17]
            v169 = v248.v223(v23.v276([v108[-1]])[0] + v167, dim=-1)
            v170 = v23.v172(v19[v219:v11.v247()])
            v18.v184({'line': v19, 'ent': v111, 'anchor': v108[-1], 'ids': v34, 't_hit': v168, 'key': v169, 'pair_q': None if v170 is None else v248.v223(v23.v276([v108[-1]])[0] + v170, dim=-1)})
            v25.v224(v111)
            break
    return v18

@v113.v37()
def gate_profile(v26, v27, v28, v16, v23, v29, v30, v17, v31, v32, v33):
    """g_t at every position of one natural line -> (g at the fact position, g elsewhere)."""
    v34 = v113.v99([v30['ids']], dtype=v113.v171, device=v32)
    v100, v101 = v102(v27, v28, v34, v17)
    v35 = v30['ids']
    v103, v104 = (None, [])
    for v36 in v105(1, v209(v35) - 1):
        v106 = v101[0, v36]
        v107 = v23.v172(v16.v225(v35[:v36 + 1][-40:]))
        if v107 is None:
            continue
        v108 = v226.v173(v16.v225(v35[:v36 + 1]))
        v109 = v248.v223(v23.v276([v108[-1]])[0] + v107, dim=-1) if v108 else v107
        v109 = v248.v223(v26.v249(v109.v269(0)), dim=-1)[0]
        v110 = v29.v47(v109, v33)
        if v110 is None:
            continue
        v174, v175 = v110
        v111 = v115(-(v248.v278(v106, -1) * v248.v279(v106, -1)).v250())
        v176, v177 = v178(v26, v29, v174, v175, v35[:v36 + 1], v31, v32)
        v112 = v115(v26.v112(v100[0, v36], v115(v174.v242()), v115(v174.v185()), v111, v177))
        if v36 == v30['t_hit']:
            v103 = v112
        else:
            v104.v184(v112)
    return (v103, v104)

def train_batch(v26, v27, v28, v16, v23, v29, v30, v17, v31, v32, v33, v38, v39: v114, v40: v115):
    """CE over the whole natural line through the gated mixture, plus L1 on the gate away from the
    entity position, plus direct supervision AT it: open when the tape holds this entity, shut
    when it does not. Off-tape lines are the negatives 260 never trained on."""
    v34 = v113.v99([v30['ids']], dtype=v113.v171, device=v32)
    v100, v101 = v102(v27, v28, v34, v17)
    v35 = v30['ids']
    v41 = []
    v42 = [v30['t_hit']] + v187.v179(v105(1, v209(v35) - 1), v194(6, v209(v35) - 2))
    for v36 in v42:
        v106 = v101[0, v36]
        v107 = v23.v172(v16.v225(v35[:v36 + 1][-40:]))
        if v107 is None:
            continue
        v108 = v226.v173(v16.v225(v35[:v36 + 1]))
        v109 = v248.v223(v23.v276([v108[-1]])[0] + v107, dim=-1) if v108 else v107
        v109 = v248.v223(v26.v249(v109.v269(0)), dim=-1)[0]
        v110 = v29.v47(v109, v33)
        if v110 is None:
            continue
        v174, v175 = v110
        v111 = v115(-(v248.v278(v106, -1) * v248.v279(v106, -1)).v250())
        v180, v177 = v178(v26, v29, v174, v175, v35[:v36 + 1], v31, v32)
        v112 = v26.v112(v100[0, v36], v115(v174.v242()), v115(v174.v185()), v111, v177)
        v116 = v181(v106, v112, v180, v177)
        if v36 == v30['t_hit']:
            v182 = 1.0 if v39 else 0.0
            v183 = v40 * v248.v251(v112.v270(1e-06, 1 - 1e-06), v113.v99(v182, device=v32))
        else:
            v183 = v38 * v112
        v41.v184(-v116[v35[v36 + 1]] + v183)
    return v113.v252(v41).v185() if v41 else None

def main() -> v9:
    v43 = v186.v117()
    v43.v118('--smoke', action='store_true')
    v43.v118('--steps', type=v9, default=0)
    v43.v118('--topk', type=v9, default=8)
    v43.v118('--gate-l1', type=v115, default=0.02)
    v43.v118('--sup-w', type=v115, default=1.0, help='weight of the have/need supervision')
    v44 = v43.v119()
    v3.v120('', encoding='utf-8')
    v32 = v113.v32('cuda' if v113.v253.v227() else 'cpu')
    v45 = v187.v121(v8)
    v113.v122(v8)
    v46 = v123.v123()
    v33 = v44.v47
    v48 = v44.v48 or (600 if v44.v124 else 2500)
    v49 = 64 if v44.v124 else 300
    v50 = 24 if v44.v124 else 120
    v51 = 24 if v44.v124 else 120
    v52 = 64 if v44.v124 else 300
    v53 = 4000 if v44.v124 else 30000
    v125(f'Stage260b open gate start {v272.v265(v273.v266).v212()} device={v32} steps={v48}')
    v126, v126, v127, v128 = v129()
    v16 = v96.v130(v21(v228.v188))
    v31 = v16.v131()
    v17 = v16.v189(v190) or 0
    v28 = v254.v229(v16, v127, v17, v31).v132(v32)
    v54 = v5 if v5.v154() else v4
    v27 = v230(v128, v31).v132(v32)
    v27.v133(v113.v231(v54, map_location=v32, weights_only=False)['model'])
    v27.v134()
    for v55 in v27.v135():
        v55.v191(False)
    v56 = v230(v128, v31).v132(v32)
    v56.v133(v113.v231(v4, map_location=v32, weights_only=False)['model'])
    v56.v134()
    for v55 in v56.v135():
        v55.v191(False)
    v23 = v97(v56, v127, v32)
    with v7.v161('r', encoding='utf-8', errors='ignore') as v92:
        v136 = v92.v192(2000000 if v44.v124 else 10000000)
    v57 = [v233.v232() for v233 in v136.v255('\n') if v233.v232()][:v53 * 4]
    v22 = v193(v57, v16, v17)[:v53]
    v45.v137(v22)
    v125(f'  wiki lines token-fit (<={v216} tok): {v209(v22)}')
    v25: v98[v21] = v98()
    v58 = v138(v22, v23, v16, v17, v49, v25)
    v59 = v138(v22[v209(v22) // 3:], v23, v16, v17, v50, v25)
    v60 = v138(v22[v209(v22) // 2:], v23, v16, v17, v52, v25)
    v61 = v138(v22[2 * v209(v22) // 3:], v23, v16, v17, v51, v25)
    v125(f'  lines: fit={v209(v58)} off_fit={v209(v60)} eval_on={v209(v59)} eval_off={v209(v61)}')
    if v194(v209(v58), v209(v59), v209(v61)) < 4:
        v125('  not enough usable lines')
        return 1
    v62 = v58 + v59
    v63 = [v75['key'] for v75 in v62]
    v64 = [v75['ent'] for v75 in v62]
    v29 = v139(v113.v252(v63, 0).v132(v32), v64, v16, v17)
    v125(f'  tape slots={v209(v64)} (off-tape entities: {v209(v61)}, deliberately absent)')
    v26 = v140(2 * (v27.v256.v234 // 2), v32)
    v65 = v113.v195.v141(v26.v196(), lr=0.003, weight_decay=0.01)
    v66 = v29.v66.v115()
    v67 = [v75['pair_q'] for v75 in v58 if v75['pair_q'] is not None]
    v68 = [v142 for v142, v75 in v94(v58) if v75['pair_q'] is not None]
    v69 = v113.v252(v67).v132(v32).v115() if v67 else None
    v70 = v113.v99(v68, device=v32) if v68 else None
    for v71 in v105(1, v48 + 1):
        v143 = v45.v187() < 0.5 or not v60
        v75 = v58[v45.v257(v209(v58))] if v143 else v60[v45.v257(v209(v60))]
        v144 = v197(v26, v27, v28, v16, v23, v29, v75, v17, v31, v32, v33, v44.v38, v143, v44.v40)
        if v144 is None:
            continue
        if v69 is not None:
            v198 = v113.v235(0, v69.v258(0), (v194(32, v69.v258(0)),), device=v32)
            v109 = v248.v223(v26.v249(v69[v198]), dim=-1)
            v144 = v144 + v248.v259(v109 @ v66.v36() / 0.05, v70[v198])
        v65.v199(set_to_none=True)
        v144.v200()
        v113.v260.v236.v201(v26.v196(), 1.0)
        v65.v71()
        if v71 % v242(1, v48 // 5) == 0:
            v125(f'  step {v71}/{v48} loss={v115(v144):.3f} ({v123.v123() - v46:.0f}s)')
    v26.v134()

    def profile(v145, v146=v29):
        v202, v203 = ([], [])
        for v75 in v145:
            v207, v237 = v208(v26, v27, v28, v16, v23, v146, v75, v17, v31, v32, v33)
            if v207 is not None:
                v202.v184(v207)
            v203.v238(v237)
        return (v261.v239(v202), v261.v239(v203))
    v147, v148 = v149(v59)
    v150, v126 = v149(v61)
    v72 = v204(v147, v148) if v209(v147) and v209(v148) else v115('nan')
    v73 = v204(v147, v150) if v209(v147) and v209(v150) else v115('nan')
    v74 = []
    for v75 in v59:
        v151 = v29.v205()
        v151.v206(v75['ent'])
        v207, v126 = v208(v26, v27, v28, v16, v23, v151, v75, v17, v31, v32, v33)
        if v207 is not None:
            v74.v184(v207)
    v76 = v115(v261.v185(v74)) if v74 else v115('nan')
    v152, v153 = v149(v59, tp=v29.v240(v8 + 1))
    v77 = v204(v152, v153) if v209(v152) and v209(v153) else v115('nan')
    v78 = v115(v261.v185(v147)) if v209(v147) else v115('nan')
    v79 = v115(v261.v185(v150)) if v209(v150) else v115('nan')
    v80 = v115(v261.v185(v148)) if v209(v148) else v115('nan')
    v81 = v115(v261.v185(v148 > 0.5)) if v209(v148) else v115('nan')
    v125(f'gate: on_tape={v78:.3f} off_tape={v79:.3f} prose={v80:.3f} | AUC vs prose={v72:.3f} vs off_tape={v73:.3f} | after delete={v76:.3f} | shuffled AUC={v77:.3f}')
    v82 = v72 >= 0.85
    v83 = v73 >= 0.7
    v84 = v80 <= 0.05 and v81 <= 0.05
    v85 = not v262.v241(v76) and v76 <= v242(0.1, v78 - 0.3)
    v86 = not v262.v241(v77) and v77 <= 0.65
    v87 = {'auc_vs_prose': 0.5741, 'auc_vs_off_tape': 0.4416, 'after_delete': 0.199, 'shuffled_auc': 0.5741, 'why': 'trained on on-tape lines only; gate was a function of h_t alone'}
    v88 = v0 / 'stage260_decision.json'
    if v88.v154():
        try:
            v210 = v246.v243(v88.v263(encoding='utf-8'))
            v211 = v210.v264('summary') or {}
            v87 = {'overall': v210.v264('overall'), 'auc_vs_prose': v211.v264('auc_on_vs_prose', v87['auc_vs_prose']), 'auc_vs_off_tape': v211.v264('auc_on_vs_off_tape', v87['auc_vs_off_tape']), 'after_delete': v211.v264('gate_after_slot_delete', v87['after_delete']), 'shuffled_auc': v211.v264('auc_shuffled_keys', v87['shuffled_auc'])}
        except (v246.v244, v245):
            pass
    if v82 and v83 and v84 and v85 and v86:
        v155 = 'OPEN_GATE2_OK'
    elif v82 and v84 and v86:
        v155 = 'OPEN_GATE2_POSITIONAL'
    else:
        v155 = 'OPEN_GATE2_NO'
    v18 = {'stage': '260b', 'overall': v155, 'trunk': v54.v156, 'steps': v48, 'topk': v33, 'n_fit': v209(v58), 'n_eval_on_tape': v209(v59), 'n_eval_off_tape': v209(v61), 'tape_slots': v209(v64), 'gates': {'G_auc_vs_prose': v82, 'G_auc_vs_off_tape': v83, 'G_quiet_on_prose': v84, 'G_delete_silences': v85, 'G_tape_causal': v86}, 'summary': {'gate_on_tape': v78, 'gate_off_tape': v79, 'gate_prose': v80, 'auc_on_vs_prose': v72, 'auc_on_vs_off_tape': v73, 'auc_shuffled_keys': v77, 'gate_after_slot_delete': v76, 'false_fire_rate_prose': v81, 'n_prose_positions': v9(v209(v148)), 'gate_reads_tape': v114(v271(v72 - v77) > 1e-06), 'prior_260': v87}, 'note': "260b adds the negatives 260 lacked: off-tape lines are trained on with the gate pushed shut at their entity position, plus direct have/need supervision at the scored point. gate_reads_tape is the first thing to check - in 260 the real and shuffled AUC were bit-identical, which meant the gate never looked at the bank. Natural wikitext lines, no cue template anywhere. The gate is read at every position of a real sentence; the scored point is the position whose next token starts a tape-backed entity. Off-tape entities are the control that separates 'I hold this' from 'something surprising is coming' — they are equally rare and equally entity-shaped, and the tape simply does not have them. OPEN_GATE2_POSITIONAL is the honest verdict when the gate finds fact positions but fires on off-tape entities too. Fit and eval lines are disjoint; trunk and P1 frozen; only W_q, gate and tau train.", 'timestamp': v272.v265(v273.v266).v212(), 'wall_s': v123.v123() - v46}
    v1.v120(v246.v213(v18, indent=2), encoding='utf-8')
    v2.v120(f'# Stage 260b open-text gate (with negatives)\n\n**{v155}** slots={v209(v64)} eval={v209(v59)} on / {v209(v61)} off\n\n- gate: on-tape **{v78:.3f}** | off-tape **{v79:.3f}** | prose {v80:.3f}\n- AUC vs prose **{v72:.3f}**, vs off-tape entities **{v73:.3f}**\n- slot deleted -> gate {v78:.3f} -> **{v76:.3f}**; shuffled keys AUC {v77:.3f}\n- false fire on prose: {v81:.3f} over {v209(v148)} positions\n', encoding='utf-8')
    v125(v246.v213({'overall': v155, 'gates': v18['gates']}, indent=2))
    if not v44.v124:
        v6.v160.v91(exist_ok=True)
        v113.v214({'W_q': v26.v249.v267(), 'gate': v26.v274.v267(), 'log_tau': v26.v280.v277().v268(), 'stage': 260}, v6)
    return 0
if v89 == '__main__':
    raise v157(v215())