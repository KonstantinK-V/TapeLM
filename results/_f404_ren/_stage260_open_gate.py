"""
Stage 260 — Does the read gate fire in the right place on OPEN text?

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

Trunk and P1 frozen; only W_q, the gate and tau train — same contract as 256. Fit lines and
eval lines are disjoint, so the gate is never scored where it was fit.

  python _stage260_open_gate.py [--smoke]
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
v1 = v0 / 'stage260_decision.json'
v2 = v0 / 'stage260_mini.md'
v3 = v0 / '_stage260_log.txt'
v4 = v10('checkpoints/stage191_p1_curve.pt')
v5 = v10('checkpoints/stage253_joint_l02.pt')
v6 = v10('checkpoints/stage260_open_gate.pt')
v7 = v10('data/_wikitext103_train.txt')
v8 = 260

def log(v11: v21) -> None:
    v12 = v11 if v11.v150('\n') else v11 + '\n'
    try:
        v151(v12, end='', flush=True)
    except v84:
        v151(v12.v157('ascii', 'replace').v214('ascii'), end='', flush=True)
    v3.v152.v85(parents=True, exist_ok=True)
    with v3.v153('a', encoding='utf-8') as v86:
        v86.v154(v12)

def token_index_before_entity(v13, v14: v9) -> v9 | None:
    """Map entity char start to t_hit = index of position whose *next* token covers the entity."""
    for v87, (v155, v156) in v88(v13.v89):
        if v155 == v14:
            return v87 - 1
        if v155 < v14 < v156:
            return v87 - 1 if v87 >= 1 else None
    return None

def filter_wiki_lines(v15: v20[v21], v16: v90, v17: v9) -> v20[v21]:
    """Keep natural lines that fit the trunk window (MAX_ARCS) without dropping tokens."""
    v18 = []
    for v19 in v15:
        v13 = v16.v157(v19)
        v34 = [v136 for v136 in v13.v34 if v136 != v17]
        if v200(v34) == v200(v13.v34) and 8 <= v200(v34) <= v205:
            v18.v174(v19)
    return v18

def harvest(v22, v23: v91, v16: v90, v17: v9, v24: v9, v25: v92):
    """One (line, entity, anchor, key) per line. Position is found by character offset, so the
    eval point is the real place in a real sentence where that entity begins."""
    v18 = []
    for v19 in v22:
        if v200(v18) >= v24:
            break
        v13 = v16.v157(v19)
        for v11 in v206.v158(v19):
            v105 = v11.v207(1)
            if v200(v105) < 5 or v105 in v25:
                continue
            v208, v209 = (v232(0, v11.v234() - 120), v185(v200(v19), v11.v256() + 120))
            v159 = v23.v164(v19[v208:v209], exclude=v105)
            if v159 is None:
                continue
            v102 = [v210 for v210 in v215.v165(v19[v208:v11.v234()]) if v210 != v105]
            if not v102:
                continue
            v160 = v211(v13, v11.v234())
            if v160 is None or v160 < 1:
                continue
            v34 = [v136 for v136 in v13.v34 if v136 != v17]
            v161 = v235.v212(v23.v257([v102[-1]])[0] + v159, dim=-1)
            v162 = v23.v164(v19[v208:v11.v234()])
            v18.v174({'line': v19, 'ent': v105, 'anchor': v102[-1], 'ids': v34, 't_hit': v160, 'key': v161, 'pair_q': None if v162 is None else v235.v212(v23.v257([v102[-1]])[0] + v162, dim=-1)})
            v25.v213(v105)
            break
    return v18

@v107.v37()
def gate_profile(v26, v27, v28, v16, v23, v29, v30, v17, v31, v32, v33):
    """g_t at every position of one natural line -> (g at the fact position, g elsewhere)."""
    v34 = v107.v93([v30['ids']], dtype=v107.v163, device=v32)
    v94, v95 = v96(v27, v28, v34, v17)
    v35 = v30['ids']
    v97, v98 = (None, [])
    for v36 in v99(1, v200(v35) - 1):
        v100 = v95[0, v36]
        v101 = v23.v164(v16.v214(v35[:v36 + 1][-40:]))
        if v101 is None:
            continue
        v102 = v215.v165(v16.v214(v35[:v36 + 1]))
        v103 = v235.v212(v23.v257([v102[-1]])[0] + v101, dim=-1) if v102 else v101
        v103 = v235.v212(v26.v236(v103.v252(0)), dim=-1)[0]
        v104 = v29.v45(v103, v33)
        if v104 is None:
            continue
        v166, v167 = v104
        v105 = v135(-(v235.v259(v100, -1) * v235.v260(v100, -1)).v237())
        v168, v169 = v170(v26, v29, v166, v167, v35[:v36 + 1], v31, v32)
        v106 = v135(v26.v106(v94[0, v36], v135(v166.v232()), v135(v166.v175()), v105, v169))
        if v36 == v30['t_hit']:
            v97 = v106
        else:
            v98.v174(v106)
    return (v97, v98)

def train_batch(v26, v27, v28, v16, v23, v29, v30, v17, v31, v32, v33, v38):
    """CE over the whole natural line through the gated mixture, plus L1 on the gate everywhere
    except the fact position — the only pressure to open it is the fact itself."""
    v34 = v107.v93([v30['ids']], dtype=v107.v163, device=v32)
    v94, v95 = v96(v27, v28, v34, v17)
    v35 = v30['ids']
    v39 = []
    v40 = [v30['t_hit']] + v177.v171(v99(1, v200(v35) - 1), v185(6, v200(v35) - 2))
    for v36 in v40:
        v100 = v95[0, v36]
        v101 = v23.v164(v16.v214(v35[:v36 + 1][-40:]))
        if v101 is None:
            continue
        v102 = v215.v165(v16.v214(v35[:v36 + 1]))
        v103 = v235.v212(v23.v257([v102[-1]])[0] + v101, dim=-1) if v102 else v101
        v103 = v235.v212(v26.v236(v103.v252(0)), dim=-1)[0]
        v104 = v29.v45(v103, v33)
        if v104 is None:
            continue
        v166, v167 = v104
        v105 = v135(-(v235.v259(v100, -1) * v235.v260(v100, -1)).v237())
        v172, v169 = v170(v26, v29, v166, v167, v35[:v36 + 1], v31, v32)
        v106 = v26.v106(v94[0, v36], v135(v166.v232()), v135(v166.v175()), v105, v169)
        v108 = v173(v100, v106, v172, v169)
        v109 = 0.0 if v36 == v30['t_hit'] else v38 * v106
        v39.v174(-v108[v35[v36 + 1]] + v109)
    return v107.v238(v39).v175() if v39 else None

def main() -> v9:
    v41 = v176.v110()
    v41.v111('--smoke', action='store_true')
    v41.v111('--steps', type=v9, default=0)
    v41.v111('--topk', type=v9, default=8)
    v41.v111('--gate-l1', type=v135, default=0.02)
    v42 = v41.v112()
    v3.v113('', encoding='utf-8')
    v32 = v107.v32('cuda' if v107.v239.v216() else 'cpu')
    v43 = v177.v114(v8)
    v107.v115(v8)
    v44 = v116.v116()
    v33 = v42.v45
    v46 = v42.v46 or (200 if v42.v117 else 800)
    v47 = 24 if v42.v117 else 160
    v48 = 16 if v42.v117 else 80
    v49 = 16 if v42.v117 else 80
    v50 = 1500 if v42.v117 else 12000
    v118(f'Stage260 open gate start {v253.v248(v254.v249).v201()} device={v32} steps={v46}')
    v119, v119, v120, v121 = v122()
    v16 = v90.v123(v21(v217.v178))
    v31 = v16.v124()
    v17 = v16.v179(v180) or 0
    v28 = v240.v218(v16, v120, v17, v31).v125(v32)
    v51 = v5 if v5.v181() else v4
    v27 = v219(v121, v31).v125(v32)
    v27.v126(v107.v220(v51, map_location=v32, weights_only=False)['model'])
    v27.v127()
    for v52 in v27.v128():
        v52.v182(False)
    v53 = v219(v121, v31).v125(v32)
    v53.v126(v107.v220(v4, map_location=v32, weights_only=False)['model'])
    v53.v127()
    for v52 in v53.v128():
        v52.v182(False)
    v23 = v91(v53, v120, v32)
    with v7.v153('r', encoding='utf-8', errors='ignore') as v86:
        v129 = v86.v183(2000000 if v42.v117 else 10000000)
    v54 = [v222.v221() for v222 in v129.v241('\n') if v222.v221()][:v50 * 4]
    v22 = v184(v54, v16, v17)[:v50]
    v43.v130(v22)
    v118(f'  wiki lines token-fit (<={v205} tok): {v200(v22)}')
    v25: v92[v21] = v92()
    v55 = v131(v22, v23, v16, v17, v47, v25)
    v56 = v131(v22[v200(v22) // 3:], v23, v16, v17, v48, v25)
    v57 = v131(v22[2 * v200(v22) // 3:], v23, v16, v17, v49, v25)
    v118(f'  lines: fit={v200(v55)} eval_on_tape={v200(v56)} eval_off_tape={v200(v57)}')
    if v185(v200(v55), v200(v56), v200(v57)) < 4:
        v118('  not enough usable lines')
        return 1
    v58 = v55 + v56
    v59 = [v71['key'] for v71 in v58]
    v60 = [v71['ent'] for v71 in v58]
    v29 = v132(v107.v238(v59, 0).v125(v32), v60, v16, v17)
    v118(f'  tape slots={v200(v60)} (off-tape entities: {v200(v57)}, deliberately absent)')
    v26 = v133(2 * (v27.v242.v223 // 2), v32)
    v61 = v107.v186.v134(v26.v187(), lr=0.003, weight_decay=0.01)
    v62 = v29.v62.v135()
    v63 = [v71['pair_q'] for v71 in v55 if v71['pair_q'] is not None]
    v64 = [v136 for v136, v71 in v88(v55) if v71['pair_q'] is not None]
    v65 = v107.v238(v63).v125(v32).v135() if v63 else None
    v66 = v107.v93(v64, device=v32) if v64 else None
    for v67 in v99(1, v46 + 1):
        v71 = v55[v43.v224(v200(v55))]
        v137 = v188(v26, v27, v28, v16, v23, v29, v71, v17, v31, v32, v33, v42.v38)
        if v137 is None:
            continue
        if v65 is not None:
            v189 = v107.v225(0, v65.v243(0), (v185(32, v65.v243(0)),), device=v32)
            v103 = v235.v212(v26.v236(v65[v189]), dim=-1)
            v137 = v137 + v235.v244(v103 @ v62.v36() / 0.05, v66[v189])
        v61.v190(set_to_none=True)
        v137.v191()
        v107.v245.v226.v192(v26.v187(), 1.0)
        v61.v67()
        if v67 % v232(1, v46 // 5) == 0:
            v118(f'  step {v67}/{v46} loss={v135(v137):.3f} ({v116.v116() - v44:.0f}s)')
    v26.v127()

    def profile(v138, v139=v29):
        v193, v194 = ([], [])
        for v71 in v138:
            v198, v227 = v199(v26, v27, v28, v16, v23, v139, v71, v17, v31, v32, v33)
            if v198 is not None:
                v193.v174(v198)
            v194.v228(v227)
        return (v246.v229(v193), v246.v229(v194))
    v140, v141 = v142(v56)
    v143, v119 = v142(v57)
    v68 = v195(v140, v141) if v200(v140) and v200(v141) else v135('nan')
    v69 = v195(v140, v143) if v200(v140) and v200(v143) else v135('nan')
    v70 = []
    for v71 in v56:
        v144 = v29.v196()
        v144.v197(v71['ent'])
        v198, v119 = v199(v26, v27, v28, v16, v23, v144, v71, v17, v31, v32, v33)
        if v198 is not None:
            v70.v174(v198)
    v72 = v135(v246.v175(v70)) if v70 else v135('nan')
    v145, v146 = v142(v56, tp=v29.v230(v8 + 1))
    v73 = v195(v145, v146) if v200(v145) and v200(v146) else v135('nan')
    v74 = v135(v246.v175(v140)) if v200(v140) else v135('nan')
    v75 = v135(v246.v175(v143)) if v200(v143) else v135('nan')
    v76 = v135(v246.v175(v141)) if v200(v141) else v135('nan')
    v77 = v135(v246.v175(v141 > 0.5)) if v200(v141) else v135('nan')
    v118(f'gate: on_tape={v74:.3f} off_tape={v75:.3f} prose={v76:.3f} | AUC vs prose={v68:.3f} vs off_tape={v69:.3f} | after delete={v72:.3f} | shuffled AUC={v73:.3f}')
    v78 = v68 >= 0.85
    v79 = v69 >= 0.7
    v80 = v76 <= 0.05 and v77 <= 0.05
    v81 = not v247.v231(v72) and v72 <= v232(0.1, v74 - 0.3)
    v82 = not v247.v231(v73) and v73 <= 0.65
    if v78 and v79 and v80 and v81 and v82:
        v147 = 'OPEN_GATE_OK'
    elif v78 and v80 and v82:
        v147 = 'OPEN_GATE_POSITIONAL'
    else:
        v147 = 'OPEN_GATE_NO'
    v18 = {'stage': 260, 'overall': v147, 'trunk': v51.v148, 'steps': v46, 'topk': v33, 'n_fit': v200(v55), 'n_eval_on_tape': v200(v56), 'n_eval_off_tape': v200(v57), 'tape_slots': v200(v60), 'gates': {'G_auc_vs_prose': v78, 'G_auc_vs_off_tape': v79, 'G_quiet_on_prose': v80, 'G_delete_silences': v81, 'G_tape_causal': v82}, 'summary': {'gate_on_tape': v74, 'gate_off_tape': v75, 'gate_prose': v76, 'auc_on_vs_prose': v68, 'auc_on_vs_off_tape': v69, 'auc_shuffled_keys': v73, 'gate_after_slot_delete': v72, 'false_fire_rate_prose': v77, 'n_prose_positions': v9(v200(v141))}, 'note': "Natural wikitext lines, no cue template anywhere. The gate is read at every position of a real sentence; the scored point is the position whose next token starts a tape-backed entity. Off-tape entities are the control that separates 'I hold this' from 'something surprising is coming' — they are equally rare and equally entity-shaped, and the tape simply does not have them. OPEN_GATE_POSITIONAL is the honest verdict when the gate finds fact positions but fires on off-tape entities too. Fit and eval lines are disjoint; trunk and P1 frozen; only W_q, gate and tau train.", 'timestamp': v253.v248(v254.v249).v201(), 'wall_s': v116.v116() - v44}
    v1.v113(v233.v202(v18, indent=2), encoding='utf-8')
    v2.v113(f'# Stage 260 open-text gate\n\n**{v147}** slots={v200(v60)} eval={v200(v56)} on / {v200(v57)} off\n\n- gate: on-tape **{v74:.3f}** | off-tape **{v75:.3f}** | prose {v76:.3f}\n- AUC vs prose **{v68:.3f}**, vs off-tape entities **{v69:.3f}**\n- slot deleted -> gate {v74:.3f} -> **{v72:.3f}**; shuffled keys AUC {v73:.3f}\n- false fire on prose: {v77:.3f} over {v200(v141)} positions\n', encoding='utf-8')
    v118(v233.v202({'overall': v147, 'gates': v18['gates']}, indent=2))
    if not v42.v117:
        v6.v152.v85(exist_ok=True)
        v107.v203({'W_q': v26.v236.v250(), 'gate': v26.v255.v250(), 'log_tau': v26.v261.v258().v251(), 'stage': 260}, v6)
    return 0
if v83 == '__main__':
    raise v149(v204())