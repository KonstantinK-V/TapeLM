"""
Stage 274 — A teacher that knows nothing the policy does not.

273 reached 1.000 on the training tape and 0.667 on a novel one — exactly its lookup baseline. The
reason is in the teacher, not the policy. 273's oracle picked which witness to open using `kind`
and `truth`, so it demonstrated a behaviour the policy has no way to reproduce: at test time
nothing in the state distinguishes a truthful witness from a lying one. The policy learned "read
something, then answer what was said" and stopped there, because "whom to read" was not learnable
from anything it could see.

The fix is not a better policy. It is a teacher that is itself executable:

    if no candidates            ASK_Q
    if read fewer than R and budget remains   READ the first unopened candidate
    otherwise                   ANSWER the value said most often IN THE TRANSCRIPT

Nothing there consults the gold value or the family of the question. Every branch is computable
from what the policy already has, so behaviour cloning now targets something the policy can
actually carry to a tape it has never seen.

The answer feature changes with it. 273 gave ANSWER a binary `was_said`, which says only that a
value appeared; agreement lived in the retrieve list, where it is free. Here it is a count over
the transcript:

    n_said_i = how many opened slots asserted candidate i's value

Free agreement is gone — the count is one until something is read, so a policy that skips reading
holds a useless feature, and a ritual read of a single slot buys nothing either.

Because the teacher is executable it is also a baseline, and it is reported as one. If the trained
policy never beats `teacher_*`, behaviour cloning bought speed and nothing else, and the decision
says so rather than celebrating a number the teacher already had.

  python _stage274_truthfree_oracle.py --smoke --witnesses 5 --liars 2
  python _stage274_truthfree_oracle.py --witnesses 5 --liars 2 --rl-episodes 2000
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
v0 = v13('results')
v1 = v13('checkpoints/stage191_p1_curve.pt')
v2 = v13('checkpoints/stage253_joint_l02.pt')
v3 = v13('checkpoints/stage274_truthfree_oracle.pt')
v4 = v13('data/_wikitext103_train.txt')
v5 = 274
v6 = 5
v14, v15 = (0, 1)

def paths(v16: v98):
    v17 = '_frozen' if v16 else ''
    return (v0 / f'stage274_decision{v17}.json', v0 / f'stage274_mini{v17}.md', v0 / f'_stage274_log{v17}.txt')
v7 = v0 / '_stage274_log.txt'

def log(v18: v99) -> None:
    v19 = v18 if v18.v191('\n') else v18 + '\n'
    try:
        v192(v19, end='', flush=True)
    except v100:
        v192(v19.v306('ascii', 'replace').v280('ascii'), end='', flush=True)
    v7.v193.v101(parents=True, exist_ok=True)
    with v7.v194('a', encoding='utf-8') as v102:
        v102.v195(v19)

class Policy(v20.v8):
    """READ sees the retrieve list; ANSWER sees only what has been opened."""

    def __init__(v103, v104: v12, v27: v12, v41):
        v281().v196()
        v103.v27 = v27
        v103.v105 = 2 + 2 * v27 + 1
        v103.v102 = v20.v282(v20.v303(v104 + v6, 128), v20.v304(), v20.v303(128, v103.v105)).v163(v41)
        v103.v106 = v20.v282(v20.v303(v104 + v6 + 3, 64), v20.v304(), v20.v303(64, 1)).v163(v41)
        v103.v107 = v20.v282(v20.v303(v104 + v6 + 3, 64), v20.v304(), v20.v303(64, 1)).v163(v41)
        for v18 in (v103.v102, v103.v106, v103.v107):
            v20.v283.v247(v18[-1].v248)
            v20.v283.v247(v18[-1].v249)

    def forward(v103, v44, v48, v49, v51=None, v52=None):
        v108 = v205.v197([v44, v48], dim=-1)
        v109 = v103.v102(v108)
        v110 = v205.v198(v109)
        for v111 in (v14, v15, v103.v105 - 1):
            v110 = v110.v250(0, v205.v120([v111], device=v109.v41), v109[v111].v284(1))
        for v199, v200, v201 in ((v103.v106, 2, v51), (v103.v107, 2 + v103.v27, v52)):
            if v201 is not None and v201.v285():
                v251 = v201.v286(0)
                v252 = v205.v197([v108.v322(0).v319(v251, -1), v201], dim=-1)
                v253 = v205.v287(v200, v200 + v251, device=v110.v41)
                v110 = v110.v288(0, v253, v199(v252).v305(-1))
        return v110.v202(~v49, -1000000000.0)

def said_counts(v21: v134[v99]) -> v9:
    return v9(v21)

def truthfree_oracle(*, v22, v23, v21, v24, v25, v26, v27):
    """Executable teacher. Receives no gold value and no question family — by signature."""
    if not v22:
        return v203.v112
    v28 = [v50 for v50, v32 in v116(v22) if v32 not in v23]
    v29 = v113(v21)
    v30 = [v32 for v60, v32 in v29.v254(2)]
    v31 = v98(v21) and (v208(v30) == 1 or v30[0] > v30[1])
    if v28 and (not v31) and (v24 < v26) and (v24 + 2 <= v25):
        return 2 + v28[0]
    v114, v115 = (0, (-1.0, -1.0))
    for v50, v32 in v116(v22):
        v117 = v204(v32)
        v118 = (v218(v29.v119(v117, 0)), -v218(v50))
        if v118 > v115:
            v115, v114 = (v118, v50)
    return 2 + v27 + v114
v10: v11 = {}

def _cand_value(v32):
    return v10.v119(v32)

def state_tensors(v33, v34, v35, v36, v37, v38, v22, v23, v21, v39, v24, v40, v41, v27, v25):
    v42 = v37['tape']
    v43 = [v50 for v50 in v36.v306(v38).v43 if v50 != v40][-v255:]
    if not v43:
        return None
    v17 = v205.v120([v43], dtype=v205.v206, device=v41)
    v44, v60 = v121(v34, v35, v17, v40)
    v44 = v44[0, -1]
    v45 = [v37.v119('_sc', {}).v119(v32, 0.0) for v32 in v22]
    v46 = v154(v45) if v45 else 0.0
    v47 = v256(v45, reverse=True)[1] if v208(v45) > 1 else 0.0
    v48 = v205.v120([v46, v46 - v47, v218(v208(v22)) / v154(1, v27), v218(v24) / v25, v218(v98(v39))], device=v41, dtype=v44.v207)
    v49 = v205.v122(v33.v105, dtype=v205.v98, device=v41)
    v49[v203.v112] = True
    v49[v203.v123] = v98(v39)
    for v50 in v124(v208(v22)):
        v49[2 + v50] = v22[v50] not in v23
        v49[2 + v27 + v50] = True
    v49[-1] = True
    v51 = v52 = None
    if v22:
        v125 = [v42.v257[v32] for v32 in v22]
        v126 = v9(v125)
        v127 = v113(v21)
        v128 = v154(v45) if v45 and v154(v45) > 0 else 1.0
        v129 = v154(1, v208(v21))
        v209, v210 = ([], [])
        for v50, v32 in v116(v22):
            v199 = v45[v50] / v128
            v211 = 1.0 if v32 in v23 else 0.0
            v209.v214([v199, v126[v125[v50]] / v208(v22), v211])
            v210.v214([v199, v127.v119(v125[v50], 0) / v129, v211])
        v51 = v205.v120(v209, device=v41, dtype=v44.v207)
        v52 = v205.v120(v210, device=v41, dtype=v44.v207)
    return (v33(v44, v48, v49, v51, v52), v49)

def rollout(v33, v34, v35, v36, v37, v53, v40, v41, *, v27, v25, v54, v26, v55=False, v56=True, v57=False):
    """One question. teacher=True clones the executable oracle; teacher_only runs it alone."""
    v42, v130, v131 = (v37['tape'], v37['postings'], v37['idf'])
    global _VALUE_OF
    v10 = {v50: v117 for v50, v117 in v116(v42.v257)}
    v58 = v203.v212.v132(S=v53['S'])
    v59 = v133(v58)
    v38 = v58
    v22: v134[v12] = []
    v39: v134[v99] = []
    v23: v135[v12] = v135()
    v21: v134[v99] = []
    v136, v137, v138, v139 = ([], [], [], [])
    v24, v140, v141 = (0, None, 0)
    for v60 in v124(v25):
        if v57:
            v111 = v258(cands=v22, seen_reads=v23, opened_values=v21, n_reads=v24, max_steps=v25, min_reads=v26, k=v27)
        else:
            v213 = v259(v33, v34, v35, v36, v37, v38, v22, v23, v21, v39, v24, v40, v41, v27, v25)
            if v213 is None:
                break
            v110, v60 = v213
            if v55:
                v111 = v258(cands=v22, seen_reads=v23, opened_values=v21, n_reads=v24, max_steps=v25, min_reads=v26, k=v27)
                if not v205.v320(v110[v111]) or v110[v111] < -100000000.0:
                    break
                v136.v214(v321.v307(v110.v322(0), v205.v120([v111], device=v41)))
            else:
                v260 = v205.v308.v289(logits=v110)
                v111 = v12(v110.v323()) if v56 else v12(v260.v324())
                v137.v214(v260.v309(v205.v120(v111, device=v41)))
                v138.v214(v260.v310())
        v139.v214(v203.v290(v27)[v111])
        if v111 in (v203.v112, v203.v123):
            v215 = v59 if v111 == v203.v112 else v39
            v22, v199 = v203.v261(v215, v130, v131, v27)
            v37['_sc'] = v199
        elif v111 == v33.v105 - 1:
            break
        elif v111 < 2 + v27:
            v50 = v111 - 2
            if v50 >= v208(v22):
                break
            v291 = v22[v50]
            v292 = v37['texts'][v291]
            v38 = (v38 + ' | ' + v292)[-2000:]
            v39 = v133(v292, exclude=v42.v257[v291])
            v23.v311(v291)
            v21.v214(v42.v257[v291])
            v24 += 1
        else:
            v50 = v111 - 2 - v27
            if v50 >= v208(v22):
                break
            v140 = v42.v257[v22[v50]]
            v141 = v12(v140 == v53['truth'])
            break
    v61 = v205.v293(v136).v216() if v136 else v205.v122((), device=v41)
    return {'loss': v61, 'logps': v137, 'entropy': v138, 'reward': v141 - v54 * v24, 'correct': v141, 'n_reads': v24, 'trace': v139, 'kind': v53.v119('kind'), 'answer_is_slot': v140 is None or v140 in v135(v42.v257)}

def main() -> v12:
    v62 = v217.v142()
    v62.v143('--smoke', action='store_true')
    v62.v143('--bc-episodes', type=v12, default=0)
    v62.v143('--rl-episodes', type=v12, default=0)
    v62.v143('--tape-period', type=v12, default=0)
    v62.v143('--clean', type=v12, default=6)
    v62.v143('--lying', type=v12, default=6)
    v62.v143('--witnesses', type=v12, default=5)
    v62.v143('--liars', type=v12, default=2)
    v62.v143('--distractor-slots', type=v12, default=0)
    v62.v143('--topk', type=v12, default=4)
    v62.v143('--max-steps', type=v12, default=6)
    v62.v143('--min-reads', type=v12, default=3, help='reads the teacher takes before answering')
    v62.v143('--read-cost', type=v218, default=0.02)
    v62.v143('--entropy-bonus', type=v218, default=0.01)
    v62.v143('--lr-policy', type=v218, default=0.001)
    v62.v143('--lr-upper', type=v218, default=3e-05)
    v62.v143('--frozen-trunk', action='store_true')
    v63 = v62.v144()
    global LOG_PATH
    v145, v146, v7 = v147(v63.v148)
    v7.v193.v101(parents=True, exist_ok=True)
    v7.v149('', encoding='utf-8')
    v41 = v205.v41('cuda' if v205.v294.v262() else 'cpu')
    v64 = v219.v150(v5)
    v205.v151(v5)
    v65 = v152.v152()
    v66 = v63.v153 or (400 if v63.v187 else 4000)
    v67 = v154(0, v63.v155)
    v68 = v63.v68 or (50 if v63.v187 else 200)
    v69 = v63.v156 or (150 if v63.v187 else 1000)
    v27 = v63.v70
    v71 = 'none' if v63.v148 else 'upper'
    v157(f'Stage274 truthfree-oracle start {v317.v301(v318.v302).v244()} device={v41} bc={v66} rl={v67} wit={v63.v188} liars={v63.v189} min_reads={v63.v26} k={v27} mode={v71}')
    v60, v60, v158, v159 = v160()
    v36 = v220.v161(v99(v263.v221))
    v72 = v36.v162()
    v40 = v36.v222(v223) or 0
    v35 = v295.v264(v36, v158, v40, v72).v163(v41)
    v73 = v2 if v2.v224() else v1
    v34 = v265(v159, v72).v163(v41)
    v34.v164(v205.v266(v73, map_location=v41, weights_only=False)['model'])
    v225.v165(v34, v71)
    v74 = v203.v166(v34)
    v75 = v265(v159, v72).v163(v41)
    v75.v164(v205.v266(v1, map_location=v41, weights_only=False)['model'])
    v75.v167()
    for v76 in v75.v168():
        v76.v226(False)
    v77 = v169(v75, v158, v41)
    with v4.v194('r', encoding='utf-8', errors='ignore') as v102:
        v170 = v102.v227(1500000 if v63.v187 else 8000000)
    v78 = v134(v11.v228((v18.v296(1) for v18 in v325.v312(v170) if v208(v18.v296(1)) >= 5)))
    v64.v171(v78)
    v79 = [v268.v267() for v268 in v170.v297('\n') if v208(v268.v267()) >= 60][:400 if v63.v187 else 6000]
    v33 = v172(2 * (v34.v298.v269 // 2), v27, v41)
    v80 = [v76 for v76 in v34.v168() if v76.v229]
    v81 = v205.v230.v173([{'params': v33.v168(), 'lr': v63.v299}] + ([{'params': v80, 'lr': v63.v313}] if v80 else []), weight_decay=0.01)
    v82: v135[v99] = v135()
    v37, v174, v175 = (None, 0.0, [])

    def new_tape(v176):
        return v203.v231(bank=v77, tok=v36, pad_id=v40, device=v41, rng=v176, pool=v78, lines=v79, used=v82, n_clean=v63.v270, n_lying=v63.v271, n_wit=v63.v188, n_liars=v63.v189, n_dist=v69)
    v83 = v11(k=v27, max_steps=v63.v25, read_cost=v63.v54, min_reads=v63.v26)
    v33.v177()
    v34.v177(v71 != 'none')
    for v84 in v124(1, v66 + 1):
        if v37 is None or (v84 - 1) % v68 == 0:
            v37 = v242(v64)
        v53 = v37['items'][v64.v272(v208(v37['items']))]
        v96 = v232(v33, v34, v35, v36, v37, v53, v40, v41, teacher=True, **v83)
        v81.v233(set_to_none=True)
        v96['loss'].v234()
        v205.v20.v273.v235(v134(v33.v168()) + v80, 1.0)
        v81.v236()
        if v84 % v154(1, v66 // 8) == 0:
            v175.v214({'phase': 'bc', 'episode': v84, 'loss': v218(v96['loss']), 'trace': v96['trace']})
            v157(f"  bc {v84}/{v66} loss={v218(v96['loss']):.4f} {v96['trace']} ({v152.v152() - v65:.0f}s)")
    for v84 in v124(1, v67 + 1):
        if (v84 - 1) % v68 == 0:
            v37 = v242(v64)
        v53 = v37['items'][v64.v272(v208(v37['items']))]
        v96 = v232(v33, v34, v35, v36, v37, v53, v40, v41, greedy=False, **v83)
        if not v96['logps']:
            continue
        v174 = 0.99 * v174 + 0.01 * v96['reward']
        v178 = v205.v293(v96['entropy']).v274() if v96['entropy'] else v205.v122((), device=v41)
        v61 = -(v96['reward'] - v174) * v205.v293(v96['logps']).v274() - v63.v275 * v178
        v81.v233(set_to_none=True)
        v61.v234()
        v205.v20.v273.v235(v134(v33.v168()) + v80, 1.0)
        v81.v236()
        if v84 % v154(1, v67 // 8) == 0:
            v175.v214({'phase': 'rl', 'episode': v84, 'baseline': v174, 'trace': v96['trace']})
            v157(f"  rl {v84}/{v67} baseline={v174:.3f} {v96['trace']} ({v152.v152() - v65:.0f}s)")
    v33.v167()
    v34.v167()
    v85 = v203.v166(v34)

    @v205.v183()
    def evaluate(v76):
        v179 = {'clean': [], 'lying': []}
        v180 = {'clean': [], 'lying': []}
        v181 = {'clean': [], 'lying': []}
        v237, v238, v239, v240 = ([], {'clean': [], 'lying': []}, [], [])
        for v182 in v76['items']:
            v241 = v232(v33, v34, v35, v36, v76, v182, v40, v41, **v83)
            v17 = v232(v33, v34, v35, v36, v76, v182, v40, v41, teacher_only=True, **v83)
            v179[v182['kind']].v214(v241['correct'])
            v180[v182['kind']].v214(v17['correct'])
            v181[v182['kind']].v214(v241['n_reads'])
            v237.v214(v12(v241['answer_is_slot']))
            v238[v182['kind']].v214(v203.v300(v76, v182, v27))
            if v182['kind'] == 'lying':
                v239.v214(v203.v314(v76, v182, v27))
            v240.v214({'kind': v182['kind'], 'trace': v241['trace'], 'correct': v241['correct']})
        v18 = lambda v276: v218(v326.v216(v276)) if v276 else v218('nan')
        return {'policy_clean': v18(v179['clean']), 'policy_lying': v18(v179['lying']), 'teacher_clean': v18(v180['clean']), 'teacher_lying': v18(v180['lying']), 'lookup_clean': v18(v238['clean']), 'lookup_lying': v18(v238['lying']), 'majority_lying': v18(v239), 'mean_reads_clean': v18(v181['clean']), 'mean_reads_lying': v18(v181['lying']), 'answer_is_slot': v18(v237), 'n': v208(v76['items']), 'traces': v240}
    v86 = v184(v37)
    v87 = v184(v242(v219.v150(v5 + 99)))
    v157(f"  TRAIN {v278.v245({v315: v316 for v315, v316 in v86.v327() if v315 != 'traces'})}")
    v157(f"  NOVEL {v278.v245({v315: v316 for v315, v316 in v87.v327() if v315 != 'traces'})}")
    v88 = v74 == v85
    v89 = v87['answer_is_slot'] >= 0.99
    v90 = v87['teacher_lying'] >= v87['lookup_lying'] + 0.1
    v91 = v87['policy_lying'] >= v87['teacher_lying'] - 0.1
    v92 = v87['policy_lying'] >= v87['lookup_lying'] + 0.1
    v93 = v87['policy_clean'] >= 0.7
    v94 = v87['policy_lying'] >= v86['policy_lying'] - 0.1
    v95 = v87['mean_reads_lying'] >= 1.0
    if not (v88 and v89):
        v185 = 'TRUTHFREE_INVALID'
    elif not v90:
        v185 = 'TEACHER_NO_BETTER_THAN_LOOKUP'
    elif v92 and v91 and v93 and v94:
        v185 = 'TRUTHFREE_ORACLE_OK'
    elif v92 or v91:
        v185 = 'TRUTHFREE_ORACLE_PARTIAL'
    else:
        v185 = 'TRUTHFREE_ORACLE_NO'
    v205.v186({'policy': v33.v277(), 'model': v34.v277(), 'stage': 274, 'arc_enc_hash': v85}, v3)
    v96 = {'stage': 274, 'overall': v185, 'frozen_trunk': v63.v148, 'trunk_mode': v71, 'smoke': v63.v187, 'seed': v5, 'bc_episodes': v66, 'rl_episodes': v67, 'witnesses': v63.v188, 'liars': v63.v189, 'min_reads': v63.v26, 'topk': v27, 'max_steps': v63.v25, 'read_cost': v63.v54, 'teacher': 'executable: ASK_Q, READ until min_reads, ANSWER argmax n_said(transcript)', 'teacher_sees_gold': False, 'teacher_sees_kind': False, 'fp_version': v203.v243(), 'used_pool_final': v208(v82), 'gates': {'G_arc_enc_frozen': v88, 'G_answer_is_slot': v89, 'G_teacher_useful': v90, 'G_policy_matches_teacher': v91, 'G_beats_lookup': v92, 'G_clean_kept': v93, 'G_novel_tape': v94, 'G_reads_informed': v95}, 'train_tape': v86, 'novel_tape': v87, 'arc_enc_hash_before': v74, 'arc_enc_hash_after': v85, 'curve': v175, 'note': "273's teacher chose which witness to open using the gold value and the question's family, so it demonstrated a behaviour the policy could not reproduce — novel-tape accuracy fell back to lookup exactly. This teacher is executable: it consults nothing the policy lacks, so it is both a cloning target and a baseline, and it is reported as both. Agreement now counts opened slots rather than the retrieve list, so it is one until something is read and a ritual read buys nothing.", 'timestamp': v317.v301(v318.v302).v244(), 'wall_s': v152.v152() - v65}
    v0.v101(parents=True, exist_ok=True)
    v145.v149(v278.v245(v96, indent=2), encoding='utf-8')
    v146.v149(f"# Stage 274 truth-free oracle\n\n**{v185}** · bc={v66} rl={v67} · {v63.v188} witnesses, {v63.v189} lying{(' · SMOKE' if v63.v187 else '')}\n\n| arm (novel tape) | clean | lying |\n|---|---:|---:|\n| policy | **{v87['policy_clean']:.3f}** | **{v87['policy_lying']:.3f}** |\n| teacher (executable) | {v87['teacher_clean']:.3f} | {v87['teacher_lying']:.3f} |\n| fixed lookup | {v87['lookup_clean']:.3f} | {v87['lookup_lying']:.3f} |\n| fixed majority | — | {v87['majority_lying']:.3f} |\n\n- reads: clean {v87['mean_reads_clean']:.2f}, lying {v87['mean_reads_lying']:.2f}\n- train lying {v86['policy_lying']:.3f} → novel {v87['policy_lying']:.3f}\n\n## Gates\n\n" + ''.v279((f'- {v315}: **{v316}**\n' for v315, v316 in v96['gates'].v327())), encoding='utf-8')
    v157(v278.v245({'overall': v185, 'gates': v96['gates']}, indent=2))
    return 0
if v97 == '__main__':
    raise v190(v246())