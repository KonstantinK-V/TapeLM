"""
Stage 270 — A tape that lies: the first exam a lookup cannot pass.

Everything measured so far has been retrieval. The tape holds one truth per key, the question is
which key, and a good index wins — which is why zero-train word votes keep beating a trained
query (266: 0.199 vs 0.062). No stage so far has posed a question where finding the slot is not
the same as knowing the answer.

Here several slots speak about the same subject and they disagree. Three witnesses say the value
is X, one says Y, and every witness is keyed just as well as the others — same subject fp, same
sentence shape, only the value differs. Top-1 retrieval therefore lands on the liar about as often
as chance allows, and the only route to the answer is to read several slots and weigh them.

That is the first place in this project where memory alone is provably insufficient, and it is
what "mind separate from memory" has to mean: the tape can be wrong, and something else judges.
No RAG setup asks this — there the retrieved passage is true by definition.

270 does not train anything. It asks whether the machinery already built can survive a tape that
lies, and it establishes the numbers a trained mind would have to beat:

    A  lookup      value of the single highest-similarity slot        must FAIL
    B  majority    unweighted vote over the witnesses retrieved       the honest target
    C  idf_weight  witnesses weighted by key similarity               does sharpening help or hurt
    D  glue        265 span-lock decode, untouched                    what today's pipeline does

Gates are written so the stage can fail informatively. G_lookup_fails is a VALIDITY gate: if
single-slot lookup already answers contradicted subjects, the exam has a leak — most likely the
liar's key is distinguishable — and every other number is meaningless.

  python _stage270_lying_tape.py --smoke
  python _stage270_lying_tape.py --witnesses 4 --liars 1
  python _stage270_lying_tape.py --liars 2          # 3-vs-2, near the aggregation limit
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
import _stage265_span_lock as s265
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import SlotBias, TapeView, ctx_query, load_glue
v0 = v14('results')
v1 = v0 / 'stage270_decision.json'
v2 = v0 / 'stage270_mini.md'
v3 = v0 / '_stage270_log.txt'
v4 = v14('checkpoints/stage191_p1_curve.pt')
v5 = v14('checkpoints/stage253_joint_l02.pt')
v6 = v14('checkpoints/stage268_mind_learns_tape.pt')
v7 = v14('checkpoints/stage265_span_lock.pt')
v8 = v14('data/_wikitext103_train.txt')
v9 = 270
v10 = '{S} was appointed director of'
v11 = ('{S} was appointed director of {V} in the regional chronicle of 1987 .', 'The county register lists {S} as appointed director of {V} that year .', 'According to the parish record , {S} was appointed director of {V} .', '{S} , appointed director of {V} , appears in the 1987 civil roll .', 'A ledger entry names {S} as the appointed director of {V} .')

def log(v15: v12) -> None:
    v16 = v15 if v15.v158('\n') else v15 + '\n'
    try:
        v159(v16, end='', flush=True)
    except v83:
        v159(v16.v230('ascii', 'replace').v227('ascii'), end='', flush=True)
    v3.v160.v84(parents=True, exist_ok=True)
    with v3.v161('a', encoding='utf-8') as v85:
        v85.v162(v16)

def fp_version() -> v12:
    v17 = v86(v87, 'canonical_fp_version', None)
    if v88(v17):
        try:
            return v12(v17())
        except v163:
            pass
    return v4.v18

def build_lying_tape(*, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29):
    """Each contradicted subject gets n_wit witnesses: n_wit-n_liars agree, n_liars dissent.

    A control population of plain single-witness subjects rides along, so "the machinery broke on
    a lying tape" can be told apart from "the machinery broke".
    """
    v30 = [v89 for v89 in v24 if v187(v89) >= 5]
    v23.v90(v30)
    v31 = [v89 for v89 in v206(v109(v30), v23, v26 * 2 + 80) if v187(v89) >= 5]
    v31 = v91(v183.v164(v31))
    v32 = v26 * (1 + v28) + v26
    if v187(v31) < v26 * 2 or v187(v30) < v32:
        raise v165(f'pool too small: subs={v187(v31)} vals={v187(v30)} need={v32}')
    v92, v93, v94 = ([], [], [])
    v45, v95 = ([], [])
    v33 = 0

    def add_slot(v39, v96, v97):
        v98 = v11[v97 % v187(v11)].v166(S=v39, V=v96)
        v99 = v19.v207([v39])[0]
        v100 = v19.v167(v98, exclude=v96)
        v92.v168(v239.v228(v99 + v100, dim=-1) if v100 is not None else v99)
        v93.v168(v96)
        v94.v168(v98)
        return v187(v93) - 1
    for v34 in v101(v26):
        v39 = v31[v34]
        v102 = v30[v33]
        v33 += 1
        v103 = v30[v33]
        v33 += 1
        v104 = [v103] * v28
        v105 = v27 - v28
        v169, v170 = ([], {})
        v106 = [v102] * v105 + v104
        v23.v90(v106)
        for v171, v107 in v172(v106):
            v108 = v173(v39, v107, v171)
            v169.v168(v108)
            v170[v108] = v107
        v45.v168({'S': v39, 'truth': v102, 'lies': v104, 'slots': v169, 'slot_val': v170, 'n_true': v105, 'n_liars': v28})
    for v34 in v101(v26):
        v39 = v31[v26 + v34]
        v107 = v30[v33]
        v33 += 1
        v108 = v173(v39, v107, 0)
        v95.v168({'S': v39, 'truth': v107, 'lies': [], 'slots': [v108], 'slot_val': {v108: v107}, 'n_true': 1, 'n_liars': 0})
    v35 = v109(v93)
    from _inprint_glue import ANCHOR_RE
    for v36 in v25:
        if v187(v93) >= v26 * (v27 + 1) + v29:
            break
        for v15 in v208.v174(v36):
            v175 = v15.v209(1)
            if v187(v175) < 5 or v175 in v35:
                continue
            v210, v211 = (v126(0, v15.v247() - 120), v178(v187(v36), v15.v248() + 120))
            v100 = v19.v167(v36[v210:v211], exclude=v175)
            if v100 is None:
                continue
            v176 = [v89 for v89 in v249.v240(v36[v210:v15.v247()]) if v89 != v175]
            if not v176:
                continue
            v92.v168(v239.v228(v19.v207([v176[-1]])[0] + v100, dim=-1))
            v93.v168(v175)
            v94.v168(v36[v210:v211])
            v35.v212(v175)
            if v187(v93) >= v26 * (v27 + 1) + v29:
                break
    v37 = v110(v115.v229(v92, 0).v144(v22), v93, v20, v21)
    return (v37, v45, v95, v94)

@v115.v44()
def retrieve(v38, v19, v20, v37, v39, v21, v40):
    v41 = [v34 for v34 in v20.v230(v10.v166(S=v39)).v121 if v34 != v21]
    v42 = v111(v38, v19, v20, v41, anchor_ids=v41)
    if v42 is None:
        return (None, None)
    v43 = (v37.v213 @ v42).v112(~v37.v177, -10000.0)
    v107, v113 = v115.v114(v43, v178(v40, v13(v37.v177.v214())))
    return (v107, v113)

def arm_scores(v38, v19, v20, v37, v45, v21, v40):
    """A lookup / B majority / C similarity-weighted, plus how often the liar outranks the truth."""
    v116, v117, v118, v119, v120 = ([], [], [], [], [])
    v46 = []
    for v47 in v45:
        v43, v113 = v179(v38, v19, v20, v37, v47['S'], v21, v40)
        if v113 is None:
            v116.v168(0)
            v117.v168(0)
            v118.v168(0)
            v119.v168(0)
            v120.v168(0.0)
            continue
        v121 = v113.v180()
        v122 = v43.v180()
        v123 = [(v171, v186) for v171, v186 in v231(v121, v122) if v171 in v47['slot_val']]
        v120.v168(v187(v123) / v126(1, v187(v47['slots'])))
        v124 = v37.v181[v121[0]]
        v116.v168(v13(v124 == v47['truth']))
        v119.v168(v13(v124 in v47['lies']))
        v125 = v182((v37.v181[v171] for v171, v138 in v123))
        v117.v168(v13(v241(v125) and v125.v254(1)[0][0] == v47['truth']))
        v89: v183[v12, v185] = v184(v185)
        for v171, v186 in v123:
            v89[v37.v181[v171]] += v126(0.0, v186)
        v118.v168(v13(v241(v89) and v126(v89.v45(), key=lambda v255: v255[1])[0] == v47['truth']))
        v46.v168({'S': v47['S'], 'truth': v47['truth'], 'top1': v124, 'witnesses_retrieved': v187(v123), 'counts': v183(v125)})
    v48 = v126(1, v187(v45))
    return {'lookup_top1': v214(v116) / v48, 'majority': v214(v117) / v48, 'sim_weighted': v214(v118) / v48, 'liar_is_top1': v214(v119) / v48, 'witness_recall': v185(v242.v232(v120)) if v120 else v185('nan'), 'n': v187(v45), 'rows': v46[:8]}

def main() -> v13:
    v49 = v188.v127()
    v49.v128('--smoke', action='store_true')
    v49.v128('--subjects', type=v13, default=0)
    v49.v128('--witnesses', type=v13, default=4)
    v49.v128('--liars', type=v13, default=1, help='witnesses repeating ONE shared lie; --liars 2 of 4 is a tie')
    v49.v128('--distractor-slots', type=v13, default=0)
    v49.v128('--topk', type=v13, default=0, help='0 = witnesses + 4')
    v49.v128('--glue-ckpt', type=v12, default='', help='268 or 265 checkpoint; empty = auto')
    v50 = v49.v129()
    v3.v130('', encoding='utf-8')
    v22 = v115.v22('cuda' if v115.v233.v215() else 'cpu')
    v23 = v189.v131(v9)
    v115.v132(v9)
    v51 = v133.v133()
    v26 = v50.v134 or (12 if v50.v136 else 60)
    v27 = v50.v52
    v28 = v50.v53
    v29 = v50.v135 or (200 if v50.v136 else 1200)
    v40 = v50.v114 or v27 + 4
    v54 = 6 if v50.v136 else 12
    v55 = 600 if v50.v136 else 6000
    if v28 >= v27 - v28:
        v137(f'  WARNING: {v28} liars vs {v27 - v28} truths — majority is not defined')
    v137(f'Stage270 lying-tape start {v245.v237(v246.v238).v203()} device={v22} subjects={v26} witnesses={v27} liars={v28} topk={v40} dist={v29}')
    v138, v138, v139, v140 = v141()
    v20 = v190.v142(v12(v216.v191))
    v56 = v20.v143()
    v21 = v20.v192(v193) or 0
    v57 = v234.v217(v20, v139, v21, v56).v144(v22)
    v58 = v5 if v5.v194() else v4
    v59 = v218(v140, v56).v144(v22)
    v59.v145(v115.v219(v58, map_location=v22, weights_only=False)['model'])
    v59.v146()
    for v60 in v59.v147():
        v60.v195(False)
    v61 = v218(v140, v56).v144(v22)
    v61.v145(v115.v219(v4, map_location=v22, weights_only=False)['model'])
    v61.v146()
    for v60 in v61.v147():
        v60.v195(False)
    v19 = v148(v61, v139, v22)
    v62 = v14(v50.v149) if v50.v149 else v6 if v6.v150() else v7
    v38 = None
    if v62.v150():
        try:
            v38 = v220(v59, v22, v62)
        except (v221, v165):
            v38 = None
        if v38 is None:
            v196 = v115.v219(v62, map_location=v22, weights_only=False)
            v38 = v197(2 * (v59.v244.v235 // 2), v22)
            with v115.v44():
                if 'glue' in v196 and v243(v196['glue'], v183):
                    v38.v145(v196['glue'], strict=False)
                    v137(f'  loaded glue from {v62.v18} via nested state_dict')
                elif 'W_q_glue' in v196:
                    v38.v251.v145(v196['W_q_glue'])
                    v38.v252.v145(v196['gate'])
                    v38.v253.v250(v196['log_tau'].v144(v22))
                    v137(f'  loaded glue from {v62.v18} via W_q_glue')
                elif 'W_q' in v196:
                    v38.v251.v145(v196['W_q'])
                    v38.v252.v145(v196['gate'])
                    v38.v253.v250(v196['log_tau'].v144(v22))
                    v137(f'  loaded glue from {v62.v18} via W_q')
                else:
                    v38 = None
                    v137(f'  {v62.v18} has no readable glue — arm D would be meaningless')
            if v38 is not None:
                v38.v146()
    if v38 is None:
        v38 = v197(2 * (v59.v244.v235 // 2), v22)
        v38.v146()
        v137(f'  no glue checkpoint at {v62} — running with an UNTRAINED glue (W_q ~ identity)')
    else:
        v137(f'  glue={v62.v18} trunk={v58.v18} fp_version={v200()}')
    with v8.v161('r', encoding='utf-8', errors='ignore') as v85:
        v151 = v85.v198(1000000 if v50.v136 else 6000000)
    v24 = v91(v183.v164((v15.v209(1) for v15 in v208.v174(v151) if v187(v15.v209(1)) >= 5)))
    v23.v90(v24)
    v25 = [v223.v222() for v223 in v151.v236('\n') if v187(v223.v222()) >= 60][:v55]
    v37, v45, v95, v138 = v152(bank_can=v19, tok=v20, pad_id=v21, device=v22, rng=v23, values_pool=v24, lines=v25, n_subj=v26, n_wit=v27, n_liars=v28, n_dist=v29)
    v137(f'  tape slots={v187(v37.v181)} contradicted={v187(v45)} clean={v187(v95)}')
    v63 = v153(v38, v19, v20, v37, v45, v21, v40)
    v64 = v153(v38, v19, v20, v37, v95, v21, v40)
    v137(f"  contradicted: lookup={v63['lookup_top1']:.3f} majority={v63['majority']:.3f} sim_w={v63['sim_weighted']:.3f} liar_top1={v63['liar_is_top1']:.3f} witness_recall={v63['witness_recall']:.3f}")
    v137(f"  clean       : lookup={v64['lookup_top1']:.3f}")
    v65 = [{'S': v47['S'], 'value': v47['truth'], 'kind': 'wiki'} for v47 in v45]
    v66 = v199.v154(v38, v59, v57, v20, v19, v37, v65, v21, v56, v22, v40, v54, locked=True)
    v67 = [{'S': v47['S'], 'value': v47['truth'], 'kind': 'wiki'} for v47 in v95]
    v68 = v199.v154(v38, v59, v57, v20, v19, v37, v67, v21, v56, v22, v40, v54, locked=True)
    v137(f"  glue span-lock: contradicted EM={v66['em']:.3f} clean EM={v68['em']:.3f}")
    v69 = v37.v155()
    v70 = 0
    for v47 in v45:
        for v103 in v109(v47['lies']):
            v70 += v69.v224(v103)
    v71 = v153(v38, v19, v20, v69, v45, v21, v40)
    v137(f"  liars removed ({v70} slots): lookup={v71['lookup_top1']:.3f}")
    v72 = 1.0 / v126(1, v27)
    v73 = v63['lookup_top1'] <= 0.6
    v74 = v71['lookup_top1'] >= v63['lookup_top1'] + 0.2
    v75 = v64['lookup_top1'] >= 0.7
    v76 = v63['witness_recall'] >= 0.75
    v77 = v63['majority'] >= 0.8
    v78 = v63['majority'] >= v63['lookup_top1'] + 0.2
    v79 = v66['em'] >= v63['majority'] - 0.1
    v80 = v75 and v76 and v74
    if not v80:
        v156 = 'LYING_TAPE_INVALID'
    elif not v73:
        v156 = 'LOOKUP_SUFFICES'
    elif v77 and v78 and v79:
        v156 = 'GLUE_ALREADY_AGGREGATES'
    elif v77 and v78:
        v156 = 'AGGREGATION_NEEDED'
    else:
        v156 = 'LYING_TAPE_NO'
    v81 = {'stage': 270, 'overall': v156, 'smoke': v50.v136, 'seed': v9, 'trunk': v58.v18, 'glue_ckpt': v62.v18 if v62.v150() else None, 'fp_version': v200(), 'trained_parameters': 0, 'n_subjects': v26, 'witnesses_per_subject': v27, 'liars_per_subject': v28, 'topk': v40, 'tape_slots': v187(v37.v181), 'chance_per_witness': v72, 'gates': {'G_clean_ok': v75, 'G_witnesses_reachable': v76, 'G_liar_causal': v74, 'G_lookup_fails': v73, 'G_majority_works': v77, 'G_aggregation_beats_lookup': v78, 'G_glue_aggregates': v79}, 'contradicted': {v201: v202 for v201, v202 in v63.v45() if v201 != 'rows'}, 'clean': {v201: v202 for v201, v202 in v64.v45() if v201 != 'rows'}, 'liars_removed': {v201: v202 for v201, v202 in v71.v45() if v201 != 'rows'}, 'glue_span_lock': {'em_contradicted': v66['em'], 'em_clean': v68['em'], 'verbatim': v66['verbatim'], 'open_recall': v66['open_recall']}, 'samples': v63['rows'], 'note': "Several slots speak about one subject and disagree; every witness is keyed alike, so top-1 retrieval lands on the liar as often as the geometry allows and only reading several slots answers the question. G_lookup_fails is a validity gate, not a result: if lookup already answers, the liar's key is distinguishable and the exam leaks. Nothing is trained here — the point is to establish the number an aggregating mind would have to beat, and to see whether the 265/268 pipeline aggregates by accident.", 'timestamp': v245.v237(v246.v238).v203(), 'wall_s': v133.v133() - v51}
    v0.v84(parents=True, exist_ok=True)
    v1.v130(v225.v204(v81, indent=2), encoding='utf-8')
    v2.v130(f"# Stage 270 lying tape\n\n**{v156}** · {v27} witnesses, {v28} lying · {v26} subjects · {v187(v37.v181)} slots · trained params **0**{(' · SMOKE' if v50.v136 else '')}\n\n| arm | contradicted | clean |\n|---|---:|---:|\n| A lookup (top-1 slot) | **{v63['lookup_top1']:.3f}** | {v64['lookup_top1']:.3f} |\n| B majority over witnesses | **{v63['majority']:.3f}** | — |\n| C similarity-weighted | {v63['sim_weighted']:.3f} | — |\n| D glue span-lock | {v66['em']:.3f} | {v68['em']:.3f} |\n\n- liar is top-1: **{v63['liar_is_top1']:.3f}**, witness recall {v63['witness_recall']:.3f}\n- liars removed → lookup {v71['lookup_top1']:.3f} (was {v63['lookup_top1']:.3f})\n\n## Gates (read G_lookup_fails first — it is validity, not result)\n\n" + ''.v226((f'- {v201}: **{v202}**\n' for v201, v202 in v81['gates'].v45())), encoding='utf-8')
    v137(v225.v204({'overall': v156, 'gates': v81['gates']}, indent=2))
    return 0
if v82 == '__main__':
    raise v157(v205())