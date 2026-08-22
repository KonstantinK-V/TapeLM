"""
Stage 273 — Read must inform the answer (SOTE course correction).

272's confirmatory READ before ANSWER-by-agreement was a ritual: majority lived in cand
`agreement` / `max_agree`, so the policy could ignore h(transcript) and still look like it
"read". That is lookup-with-features wearing a READ — off course for a neural mind over tape.

Here the answer branch is denied the majority channel:

    READ bonus  <- [score, agreement, was_read]   (whom to open)
    ANSWER bonus <- [score, was_said, was_read]    (was_said = value appears in transcript)

Global feats drop `max_agree`. After a READ, choosing ANSWER_i must track what entered the
transcript — copy from what was read — not count siblings in the retrieve list.

Oracle (teacher may use kind; policy does not):
    clean   ASK_Q → ANSWER truth          (no read; score ranks the single witness)
    lying   ASK_Q → READ a truth witness → ANSWER that value (now was_said)

BC only by default. Same gates as 271/272.

  python _stage273_read_must_inform.py --smoke --witnesses 5 --liars 2 --read-cost 0.02
  python _stage273_read_must_inform.py --witnesses 5 --liars 2 --read-cost 0.02
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
v0 = v10('results')
v1 = v10('checkpoints/stage191_p1_curve.pt')
v2 = v10('checkpoints/stage253_joint_l02.pt')
v3 = v10('checkpoints/stage273_read_must_inform.pt')
v4 = v10('data/_wikitext103_train.txt')
v5 = 273
v6 = 5

def paths(v11: v95):
    v12 = '_frozen' if v11 else ''
    return (v0 / f'stage273_decision{v12}.json', v0 / f'stage273_mini{v12}.md', v0 / f'_stage273_log{v12}.txt')
v7 = v0 / '_stage273_log.txt'

def log(v13: v96) -> None:
    v14 = v13 if v13.v189('\n') else v13 + '\n'
    try:
        v190(v14, end='', flush=True)
    except v97:
        v190(v14.v301('ascii', 'replace').v278('ascii'), end='', flush=True)
    v7.v191.v98(parents=True, exist_ok=True)
    with v7.v192('a', encoding='utf-8') as v99:
        v99.v193(v14)

class Policy(v15.v8):
    """Separate READ vs ANSWER cand scorers — ANSWER never sees agreement."""

    def __init__(v100, v101: v9, v18: v9, v35):
        v279().v194()
        v100.v18 = v18
        v100.v102 = 2 + 2 * v18 + 1
        v100.v99 = v15.v280(v15.v298(v101 + v6, 128), v15.v299(), v15.v298(128, v100.v102)).v162(v35)
        v15.v249.v195(v100.v99[-1].v196)
        v15.v249.v195(v100.v99[-1].v197)
        v100.v103 = v15.v280(v15.v298(v101 + v6 + 3, 64), v15.v299(), v15.v298(64, 1)).v162(v35)
        v15.v249.v195(v100.v103[-1].v196)
        v15.v249.v195(v100.v103[-1].v197)
        v100.v104 = v15.v280(v15.v298(v101 + v6 + 3, 64), v15.v299(), v15.v298(64, 1)).v162(v35)
        v15.v249.v195(v100.v104[-1].v196)
        v15.v249.v195(v100.v104[-1].v197)

    def forward(v100, v37, v41, v42, v44=None, v45=None):
        v105 = v205.v198([v37, v41], dim=-1)
        v46 = v100.v99(v105)
        if v44 is not None and v44.v250():
            v199 = v44.v251(0)
            v200 = v205.v198([v105.v285(0).v300(v199, -1), v44], dim=-1)
            v46 = v46.v252(0, v205.v281(2, 2 + v199, device=v46.v35), v100.v103(v200).v282(-1))
        if v45 is not None and v45.v250():
            v199 = v45.v251(0)
            v200 = v205.v198([v105.v285(0).v300(v199, -1), v45], dim=-1)
            v46 = v46.v252(0, v205.v281(2 + v100.v18, 2 + v100.v18 + v199, device=v46.v35), v100.v104(v200).v282(-1))
        return v46.v201(~v42, -1000000000.0)

def _answer_value(v16, v17, v18, v19: v96) -> v9:
    for v43, v106 in v107(v16):
        if v17.v125[v106] == v19:
            return 2 + v18 + v43
    return 2 + v18

def oracle_action(v20, v16, v21, v17, v18, v22, v23, v24: v96) -> v9:
    if not v16:
        return v202.v108
    v25 = [v17.v125[v106] for v106 in v16]
    if v20.v203('kind') == 'clean':
        v27 = v20['truth'] if v20['truth'] in v25 else v25[0]
        return v110(v16, v17, v18, v27)
    v26 = v109(v20['slots'])
    if not v204((v17.v125[v106] == v20['truth'] and v106 in v21 for v106 in v16)):
        if v23 + 1 < v22:
            for v43, v106 in v107(v16):
                if v106 in v26 and v17.v125[v106] == v20['truth'] and (v106 not in v21):
                    return 2 + v43
            for v43, v106 in v107(v16):
                if v106 in v26 and v106 not in v21:
                    return 2 + v43
            for v43, v106 in v107(v16):
                if v106 not in v21:
                    return 2 + v43
    v27 = v20['truth'] if v20['truth'] in v25 else v25[0]
    return v110(v16, v17, v18, v27)

def _state_tensors(v28, v29, v30, v31, v32, v24, v16, v21, v33, v23, v34, v35, v18, v22):
    v17 = v32['tape']
    v36 = [v43 for v43 in v31.v301(v24).v36 if v43 != v34][-v253:]
    if not v36:
        return None
    v12 = v205.v111([v36], dtype=v205.v206, device=v35)
    v37, v54 = v112(v29, v30, v12, v34)
    v37 = v37[0, -1]
    v38 = [v32.v203('_sc', {}).v203(v106, 0.0) for v106 in v16]
    v39 = v153(v38) if v38 else 0.0
    v40 = v254(v38, reverse=True)[1] if v208(v38) > 1 else 0.0
    v41 = v205.v111([v39, v39 - v40, v126(v208(v16)) / v153(1, v18), v126(v23) / v22, v126(v95(v33))], device=v35, dtype=v37.v207)
    v42 = v205.v113(v28.v102, dtype=v205.v95, device=v35)
    v42[v202.v108] = True
    v42[v202.v114] = v95(v33)
    for v43 in v115(v208(v16)):
        v42[2 + v43] = v16[v43] not in v21
        v42[2 + v18 + v43] = True
    v42[-1] = True
    v44 = v45 = None
    if v16:
        v116 = [v17.v125[v106] for v106 in v16]
        v117 = v209(v116)
        v118 = v153(v38) if v38 and v153(v38) > 0 else 1.0
        v210, v211 = ([], [])
        for v43, v106 in v107(v16):
            v122 = v38[v43] / v118
            v212 = v117[v116[v43]] / v208(v16)
            v213 = 1.0 if v106 in v21 else 0.0
            v214 = 1.0 if v116[v43] and v116[v43] in v24 else 0.0
            v210.v220([v122, v212, v213])
            v211.v220([v122, v214, v213])
        v44 = v205.v111(v210, device=v35, dtype=v37.v207)
        v45 = v205.v111(v211, device=v35, dtype=v37.v207)
    v46 = v28(v37, v41, v42, v44, v45)
    return (v46, v42)

def _apply(v47, *, v20, v32, v48, v24, v16, v33, v21, v23, v49, v50, v18, v28):
    v17, v119, v120 = (v32['tape'], v32['postings'], v32['idf'])
    if v47 in (v202.v108, v202.v114):
        v121 = v48 if v47 == v202.v108 else v33
        v16, v122 = v202.v215(v121, v119, v120, v18)
        v32['_sc'] = v122
        return (v24, v16, v33, v21, v23, v49, v50, False)
    if v47 == v28.v102 - 1:
        return (v24, v16, v33, v21, v23, v49, v50, True)
    if v47 < 2 + v18:
        v43 = v47 - 2
        if v43 >= v208(v16):
            return (v24, v16, v33, v21, v23, v49, v50, True)
        v123 = v16[v43]
        v124 = v32['texts'][v123]
        v24 = (v24 + ' | ' + v124)[-2000:]
        v33 = v128(v124, exclude=v17.v125[v123])
        v21 = v109(v21) | {v123}
        return (v24, v16, v33, v21, v23 + 1, v49, v50, False)
    v43 = v47 - 2 - v18
    if v43 >= v208(v16):
        return (v24, v16, v33, v21, v23, v49, v50, True)
    v49 = v17.v125[v16[v43]]
    v50 = 1.0 if v49 == v20['truth'] else 0.0
    return (v24, v16, v33, v21, v23, v49, v50, True)

def bc_episode(v28, v29, v30, v31, v32, v20, v34, v35, *, v18, v22, v51, v52: v126=5.0):
    v17 = v32['tape']
    v53 = v202.v216.v127(S=v20['S'])
    v48 = v128(v53)
    v24 = v53
    v16: v129[v9] = []
    v33: v129[v96] = []
    v21: v109[v9] = v109()
    v23, v49, v50 = (0, None, 0.0)
    v55, v130, v131 = ([], [], [])
    for v54 in v115(v22):
        v132 = v217(v28, v29, v30, v31, v32, v24, v16, v21, v33, v23, v34, v35, v18, v22)
        if v132 is None:
            break
        v46, v54 = v132
        v47 = v218(v20, v16, v21, v17, v18, v22, v23, v24)
        if not v205.v283(v46[v47]) or v46[v47] < -100000000.0:
            break
        v133 = 2 <= v47 < 2 + v18
        v134 = 2 + v18 <= v47 < 2 + 2 * v18
        if v133:
            v219 = 1.0
        elif v134 and v23 > 0:
            v219 = v126(v52)
        elif v134:
            v219 = 2.5
        else:
            v219 = 2.0
        v55.v220(v284.v255(v46.v285(0), v205.v111([v47], device=v35)))
        v130.v220(v219)
        v131.v220(v202.v244(v18)[v47])
        v24, v16, v33, v21, v23, v49, v50, v135 = v221(v47, item=v20, pack=v32, qwords=v48, transcript=v24, cands=v16, last_read_words=v33, seen_reads=v21, n_reads=v23, answered=v49, reward=v50, k=v18, policy=v28)
        if v135:
            break
    v50 -= v51 * v23
    if v55:
        v136 = v205.v111(v130, device=v35, dtype=v55[0].v207)
        v137 = (v205.v304(v55) * v136).v256() / v136.v256()
    else:
        v137 = v205.v113((), device=v35)
    return {'loss': v137, 'reward': v50, 'correct': v9(v49 == v20['truth']), 'n_reads': v23, 'trace': v131, 'kind': v20.v203('kind'), 'answer_is_slot': v49 is None or v49 in v109(v17.v125)}

def run_episode(v28, v29, v30, v31, v32, v20, v34, v35, *, v18, v22, v51, v56=True):
    v17 = v32['tape']
    v53 = v202.v216.v127(S=v20['S'])
    v48 = v128(v53)
    v24 = v53
    v16: v129[v9] = []
    v33: v129[v96] = []
    v21: v109[v9] = v109()
    v138, v139, v131 = ([], [], [])
    v23, v49, v50 = (0, None, 0.0)
    for v54 in v115(v22):
        v132 = v217(v28, v29, v30, v31, v32, v24, v16, v21, v33, v23, v34, v35, v18, v22)
        if v132 is None:
            break
        v46, v54 = v132
        v140 = v205.v257.v222(logits=v46)
        v47 = v9(v46.v286()) if v56 else v9(v140.v287())
        v138.v220(v140.v258(v205.v111(v47, device=v35)))
        v139.v220(v140.v259())
        v131.v220(v202.v244(v18)[v47])
        v24, v16, v33, v21, v23, v49, v50, v135 = v221(v47, item=v20, pack=v32, qwords=v48, transcript=v24, cands=v16, last_read_words=v33, seen_reads=v21, n_reads=v23, answered=v49, reward=v50, k=v18, policy=v28)
        if v135:
            break
    v50 -= v51 * v23
    return {'logps': v138, 'entropy': v139, 'reward': v50, 'correct': v9(v49 == v20['truth']), 'answered': v49, 'n_reads': v23, 'trace': v131, 'answer_is_slot': v49 is None or v49 in v109(v17.v125), 'kind': v20.v203('kind')}

def main() -> v9:
    v57 = v223.v141()
    v57.v142('--smoke', action='store_true')
    v57.v142('--bc-episodes', type=v9, default=0)
    v57.v142('--rl-episodes', type=v9, default=0)
    v57.v142('--tape-period', type=v9, default=0)
    v57.v142('--clean', type=v9, default=6)
    v57.v142('--lying', type=v9, default=6)
    v57.v142('--witnesses', type=v9, default=5)
    v57.v142('--liars', type=v9, default=2)
    v57.v142('--distractor-slots', type=v9, default=0)
    v57.v142('--topk', type=v9, default=4)
    v57.v142('--max-steps', type=v9, default=6)
    v57.v142('--read-cost', type=v126, default=0.02)
    v57.v142('--entropy-bonus', type=v126, default=0.01)
    v57.v142('--answer-after-read-weight', type=v126, default=5.0)
    v57.v142('--lr-policy', type=v126, default=0.001)
    v57.v142('--lr-upper', type=v126, default=3e-05)
    v57.v142('--frozen-trunk', action='store_true')
    v58 = v57.v143()
    global LOG_PATH
    v144, v145, v7 = v146(v58.v147)
    v7.v191.v98(parents=True, exist_ok=True)
    v7.v148('', encoding='utf-8')
    v35 = v205.v35('cuda' if v205.v288.v260() else 'cpu')
    v59 = v224.v149(v5)
    v205.v150(v5)
    v60 = v151.v151()
    v61 = v58.v152 or (400 if v58.v184 else 4000)
    v62 = v153(0, v58.v154)
    v63 = v58.v63 or (50 if v58.v184 else 200)
    v64 = v58.v155 or (150 if v58.v184 else 1000)
    v18 = v58.v65
    v66 = 'none' if v58.v147 else 'upper'
    v156(f'Stage273 read-must-inform start {v306.v296(v307.v297).v246()} device={v35} bc={v61} rl={v62} tape_period={v63} clean={v58.v261} lying={v58.v262} wit={v58.v186} liars={v58.v187} k={v18} mode={v66}')
    v54, v54, v157, v158 = v159()
    v31 = v225.v160(v96(v263.v226))
    v67 = v31.v161()
    v34 = v31.v227(v228) or 0
    v30 = v289.v264(v31, v157, v34, v67).v162(v35)
    v68 = v2 if v2.v229() else v1
    v29 = v265(v158, v67).v162(v35)
    v29.v163(v205.v266(v68, map_location=v35, weights_only=False)['model'])
    v230.v164(v29, v66)
    v69 = v202.v165(v29)
    v70 = v265(v158, v67).v162(v35)
    v70.v163(v205.v266(v1, map_location=v35, weights_only=False)['model'])
    v70.v166()
    for v71 in v70.v167():
        v71.v231(False)
    v72 = v168(v70, v157, v35)
    v156(f'  trunk={v68.v267} mode={v66} fp_version={v202.v245()} arc={v69[:12]}…')
    with v4.v192('r', encoding='utf-8', errors='ignore') as v99:
        v169 = v99.v232(1500000 if v58.v184 else 8000000)
    v73 = v129(v268.v233((v13.v290(1) for v13 in v308.v302(v169) if v208(v13.v290(1)) >= 5)))
    v59.v170(v73)
    v74 = [v270.v269() for v270 in v169.v291('\n') if v208(v270.v269()) >= 60][:400 if v58.v184 else 6000]
    v28 = v171(2 * (v29.v292.v271 // 2), v18, v35)
    v75 = [v71 for v71 in v29.v167() if v71.v234]
    v76 = v205.v235.v172([{'params': v28.v167(), 'lr': v58.v293}] + ([{'params': v75, 'lr': v58.v303}] if v75 else []), weight_decay=0.01)
    v77: v109[v96] = v109()
    v32 = None
    v78 = 0.0
    v79 = []

    def new_tape(v173):
        return v202.v236(bank=v72, tok=v31, pad_id=v34, device=v35, rng=v173, pool=v73, lines=v74, used=v77, n_clean=v58.v261, n_lying=v58.v262, n_wit=v58.v186, n_liars=v58.v187, n_dist=v64)
    v28.v174()
    v29.v174(v66 != 'none')
    for v80 in v115(1, v61 + 1):
        if v32 is None or (v80 - 1) % v63 == 0:
            v32 = v181(v59)
        v20 = v32['items'][v59.v272(v208(v32['items']))]
        v93 = v237(v28, v29, v30, v31, v32, v20, v34, v35, k=v18, max_steps=v58.v22, read_cost=v58.v51, answer_after_read_weight=v58.v52)
        v76.v238(set_to_none=True)
        v93['loss'].v239()
        v205.v15.v273.v240(v129(v28.v167()) + v75, 1.0)
        v76.v241()
        if v80 % v153(1, v61 // 10) == 0:
            v79.v220({'phase': 'bc', 'episode': v80, 'loss': v126(v93['loss']), 'reward': v93['reward'], 'trace': v93['trace']})
            v156(f"  bc {v80}/{v61} loss={v126(v93['loss']):.3f} last_trace={v93['trace']} ({v151.v151() - v60:.0f}s)")
    for v80 in v115(1, v62 + 1):
        if v32 is None or (v80 - 1) % v63 == 0:
            v32 = v181(v59)
        v20 = v32['items'][v59.v272(v208(v32['items']))]
        v93 = v242(v28, v29, v30, v31, v32, v20, v34, v35, k=v18, max_steps=v58.v22, read_cost=v58.v51, greedy=False)
        if not v93['logps']:
            continue
        v78 = 0.99 * v78 + 0.01 * v93['reward']
        v175 = v93['reward'] - v78
        v176 = v205.v304(v93['entropy']).v256() if v93['entropy'] else v205.v113((), device=v35)
        v137 = -v175 * v205.v304(v93['logps']).v256() - v58.v185 * v176
        v76.v238(set_to_none=True)
        v137.v239()
        v205.v15.v273.v240(v129(v28.v167()) + v75, 1.0)
        v76.v241()
        if v80 % v153(1, v62 // 10) == 0 or v80 == v62:
            v79.v220({'phase': 'rl', 'episode': v80, 'baseline': v78, 'reward': v93['reward'], 'trace': v93['trace']})
            v156(f"  rl {v80}/{v62} baseline={v78:.3f} last_trace={v93['trace']} ({v151.v151() - v60:.0f}s)")
    v28.v166()
    v29.v166()
    v81 = v202.v165(v29)

    @v205.v179()
    def evaluate(v71):
        v177 = {'clean': [], 'lying': [], 'reads': [], 'reads_clean': [], 'reads_lying': [], 'slot_ok': [], 'lookup': {'clean': [], 'lying': []}, 'major': {'clean': [], 'lying': []}, 'traces': []}
        for v178 in v71['items']:
            v243 = v242(v28, v29, v30, v31, v71, v178, v34, v35, k=v18, max_steps=v58.v22, read_cost=v58.v51, greedy=True)
            v177[v178['kind']].v220(v243['correct'])
            v177['reads'].v220(v243['n_reads'])
            v177[f"reads_{v178['kind']}"].v220(v243['n_reads'])
            v177['slot_ok'].v220(v9(v243['answer_is_slot']))
            v177['lookup'][v178['kind']].v220(v202.v294(v71, v178, v18))
            v177['major'][v178['kind']].v220(v202.v295(v71, v178, v18))
            v177['traces'].v220({'kind': v178['kind'], 'trace': v243['trace'], 'correct': v243['correct']})
        v13 = lambda v274: v126(v309.v305(v274)) if v274 else v126('nan')
        return {'policy_clean': v13(v177['clean']), 'policy_lying': v13(v177['lying']), 'lookup_clean': v13(v177['lookup']['clean']), 'lookup_lying': v13(v177['lookup']['lying']), 'majority_lying': v13(v177['major']['lying']), 'mean_reads': v13(v177['reads']), 'mean_reads_clean': v13(v177['reads_clean']), 'mean_reads_lying': v13(v177['reads_lying']), 'answer_is_slot': v13(v177['slot_ok']), 'n': v208(v71['items']), 'traces': v177['traces']}
    v82 = v180(v32)
    v83 = v181(v224.v149(v5 + 99))
    v84 = v180(v83)
    v156(f'  TRAIN {v276.v247(v82)}')
    v156(f'  NOVEL {v276.v247(v84)}')
    v85 = v69 == v81
    v86 = v84['answer_is_slot'] >= 0.99
    v87 = v84['policy_lying'] >= v84['lookup_lying'] + 0.1
    v88 = v84['policy_lying'] >= v84['majority_lying'] - 0.05
    v89 = v84['policy_clean'] >= 0.7
    v90 = v84['policy_lying'] >= v82['policy_lying'] - 0.1
    v91 = v84['mean_reads'] <= v58.v22 * 0.6
    v92 = v84['mean_reads_lying'] >= 0.5 and v84['mean_reads_clean'] <= v84['mean_reads_lying'] + 0.25
    if not (v85 and v86):
        v182 = 'READ_INFORM_INVALID'
    elif v87 and v89 and v90 and v92:
        v182 = 'READ_INFORM_OK'
    elif v89 or v87 or v92:
        v182 = 'READ_INFORM_PARTIAL'
    else:
        v182 = 'READ_INFORM_NO'
    v205.v183({'policy': v28.v275(), 'model': v29.v275(), 'stage': 273, 'arc_enc_hash': v81}, v3)
    v93 = {'stage': 273, 'overall': v182, 'frozen_trunk': v58.v147, 'trunk_mode': v66, 'smoke': v58.v184, 'seed': v5, 'bc_episodes': v61, 'rl_episodes': v62, 'tape_period': v63, 'actions': v202.v244(v18), 'topk': v18, 'max_steps': v58.v22, 'read_cost': v58.v51, 'entropy_bonus': v58.v185, 'answer_after_read_weight': v58.v52, 'witnesses': v58.v186, 'liars': v58.v187, 'fp_version': v202.v245(), 'used_pool_final': v208(v77), 'gates': {'G_arc_enc_frozen': v85, 'G_answer_is_slot': v86, 'G_beats_lookup': v87, 'G_beats_majority': v88, 'G_clean_kept': v89, 'G_novel_tape': v90, 'G_reads_economical': v91, 'G_read_informed': v92}, 'train_tape': v82, 'novel_tape': v84, 'arc_enc_hash_before': v69, 'arc_enc_hash_after': v81, 'curve': v79, 'note': 'Course correction after 272: ANSWER cand scorer sees was_said (value in transcript) instead of agreement; max_agree removed from globals. READ is the only way to light was_said for the truth on lying items — majority-in-retrieve is not an answer feature.', 'timestamp': v306.v296(v307.v297).v246(), 'wall_s': v151.v151() - v60}
    v0.v98(parents=True, exist_ok=True)
    v144.v148(v276.v247(v93, indent=2), encoding='utf-8')
    v145.v148(f"# Stage 273 read-must-inform{(' (frozen trunk)' if v58.v147 else '')}\n\n**{v182}** · bc={v61} rl={v62} · actions={v208(v202.v244(v18))}{(' · SMOKE' if v58.v184 else '')}\n\n| arm | clean | lying |\n|---|---:|---:|\n| policy (novel tape) | **{v84['policy_clean']:.3f}** | **{v84['policy_lying']:.3f}** |\n| fixed lookup | {v84['lookup_clean']:.3f} | {v84['lookup_lying']:.3f} |\n| fixed majority | — | {v84['majority_lying']:.3f} |\n\n- mean reads {v84['mean_reads']:.2f} (clean {v84['mean_reads_clean']:.2f} / lying {v84['mean_reads_lying']:.2f})\n- train tape lying {v82['policy_lying']:.3f} → novel {v84['policy_lying']:.3f}\n\n## Gates\n\n" + ''.v277((f'- {v310}: **{v311}**\n' for v310, v311 in v93['gates'].v312())), encoding='utf-8')
    v156(v276.v247({'overall': v182, 'gates': v93['gates']}, indent=2))
    return 0
if v94 == '__main__':
    raise v188(v248())