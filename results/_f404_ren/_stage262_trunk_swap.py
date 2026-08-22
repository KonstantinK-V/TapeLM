"""
Stage 262 — Does the semantic channel survive a trunk swap?

The claim the whole architecture rests on is that memory and reasoning are separable: keys live
in frozen character fp, understanding lives in the trunk, and the two meet only through W_sem.
If that is real, the trunk is replaceable. 258 showed the channel works with the curve trunk and
beats matched GPT-2 on unseen paraphrases (0.646 vs 0.276). This runs the SAME exam with a third
trunk - any HuggingFace causal LM - and changes nothing else.

Nothing about the tape moves. Keys are canonical frozen P1 fp, written from characters, so the
external model's tokenizer never touches them. The only coupling is h_t -> W_sem -> key space,
and W_sem is a single Linear whose input dim is read off the model. Retrieval only: no decode,
so there is no vocabulary to reconcile at all.

Read it as:
  external >= curve   the interface transfers; the trunk is rentable and 209's scale wall stops
                      being the blocking problem for the product story
  external >> curve   understanding was the bottleneck, exactly as predicted
  external <= curve   the channel is tuned to curve states specifically - a real negative, and
                      the one that would sink the "any reasoner" claim

  python _stage262_trunk_swap.py --model Qwen/Qwen2.5-0.5B [--smoke]
  python _stage262_trunk_swap.py --model sshleifer/tiny-gpt2 --smoke      # wiring check
"""
from __future__ import annotations
import argparse
import json
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
import _stage24x_lib as L
import _stage258_semantic_query as s258
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE
v0 = v9('results')
v1 = v0 / 'stage262_decision.json'
v2 = v0 / 'stage262_mini.md'
v3 = v0 / '_stage262_log.txt'
v4 = v9('checkpoints/stage191_p1_curve.pt')
v5 = v9('checkpoints/stage253_joint_l02.pt')
v6 = v9('data/_wikitext103_train.txt')
v7 = 262

def log(v10: v62) -> None:
    v11 = v10 if v10.v129('\n') else v10 + '\n'
    try:
        v130(v11, end='', flush=True)
    except v63:
        v130(v11.v206('ascii', 'replace').v189('ascii'), end='', flush=True)
    v3.v131.v64(parents=True, exist_ok=True)
    with v3.v132('a', encoding='utf-8') as v65:
        v65.v133(v11)

class ExternalTrunk:
    """Any HF causal LM, frozen, read for its last hidden state. Its tokenizer is used ONLY to
    feed itself - the tape is keyed on characters through P1 and never sees it."""

    def __init__(v66, v67: v62, v14, v68=v134.v69):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        v66.v21 = v169.v135(v67)
        v66.v70 = v199.v135(v67, torch_dtype=v68 if v14.v212 == 'cuda' else v134.v207).v92(v14)
        v66.v70.v94()
        for v27 in v66.v70.v95():
            v27.v146(False)
        v66.v14 = v14
        v66.v71 = v8(v66.v70.v170.v136)

    @v134.v74()
    def state(v66, v72: v62) -> v134.v75 | None:
        v73 = v66.v21(v72, return_tensors='pt', truncation=True, max_length=256)
        v73 = {v137: v117.v92(v66.v14) for v137, v117 in v73.v190()}
        if v73['input_ids'].v171() == 0:
            return None
        v60 = v66.v70(**v73, output_hidden_states=True)
        return v60.v213[-1][0, -1].v191().v110()

def main() -> v8:
    v12 = v138.v76()
    v12.v77('--model', type=v62, required=True, help='HF causal LM id, frozen')
    v12.v77('--smoke', action='store_true')
    v12.v77('--steps', type=v8, default=0)
    v12.v77('--subjects', type=v8, default=0)
    v12.v77('--distractor-slots', type=v8, default=0)
    v12.v77('--tau', type=v110, default=0.05)
    v12.v77('--lr', type=v110, default=0.002)
    v13 = v12.v78()
    v3.v79('', encoding='utf-8')
    v14 = v134.v14('cuda' if v134.v192.v172() else 'cpu')
    v15 = v139.v80(v140.v7)
    v134.v81(v140.v7)
    v16 = v82.v82()
    v17 = v13.v17 or (150 if v13.v84 else 600)
    v18 = v13.v32 or (12 if v13.v84 else 64)
    v19 = v13.v83 or (150 if v13.v84 else 1200)
    v20 = 400 if v13.v84 else 6000
    v85(f'Stage262 trunk swap start {v204.v197(v205.v198).v166()} model={v13.v70}')
    v86, v86, v87, v88 = v89()
    v21 = v141.v90(v62(v173.v142))
    v22 = v21.v91()
    v23 = v21.v143(v144) or 0
    v24 = v193.v174(v21, v87, v23, v22).v92(v14)
    v25 = v5 if v5.v145() else v4
    v26 = v175(v88, v22).v92(v14)
    v26.v93(v134.v176(v25, map_location=v14, weights_only=False)['model'])
    v26.v94()
    for v27 in v26.v95():
        v27.v146(False)
    v28 = v175(v88, v22).v92(v14)
    v28.v93(v134.v176(v4, map_location=v14, weights_only=False)['model'])
    v28.v94()
    for v27 in v28.v95():
        v27.v146(False)
    v29 = v96(v28, v87, v14)
    try:
        v97 = v147(v13.v70, v14)
        v85(f'  external trunk loaded: {v13.v70} hidden={v97.v71}')
    except v98 as e:
        v85(f'  could not load {v13.v70}: {v212(v153).v61}: {v153}')
        return 1
    with v6.v132('r', encoding='utf-8', errors='ignore') as v65:
        v99 = v65.v148(1000000 if v13.v84 else 6000000)
    v30 = v100(v111.v149((v10.v181(1) for v10 in v180.v152(v99) if v109(v10.v181(1)) >= 5)))
    v15.v101(v30)
    v31 = [v178.v177() for v178 in v99.v194('\n') if v109(v178.v177()) >= 60][:v20]
    v32 = v140.v102(v30, v15, v18, v150(v109(v30), 400))
    v33 = v18 // 2
    v103, v104 = (v32[:v33], v32[v33:])
    v105, v106, v107 = v140.v108(v29, v32, v14)
    v34 = v109(v106)
    v35 = {v179['S'] for v179 in v32} | v151(v106)
    for v36 in v31:
        if v109(v106) >= v34 + v19:
            break
        for v10 in v180.v152(v36):
            v153 = v10.v181(1)
            if v109(v153) < 5 or v153 in v35:
                continue
            v182, v183 = (v195(0, v10.v208() - 120), v150(v109(v36), v10.v209() + 120))
            v154 = v29.v184(v36[v182:v183], exclude=v153)
            if v154 is None:
                continue
            v155 = [v185 for v185 in v210.v200(v36[v182:v10.v208()]) if v185 != v153]
            if not v155:
                continue
            v105.v186(v201.v196(v29.v214([v155[-1]])[0] + v154, dim=-1))
            v106.v186(v153)
            v107.v186((None, None))
            v35.v187(v153)
            if v109(v106) >= v34 + v19:
                break
    v37 = v134.v202(v105, 0).v92(v14).v110()
    v38: v111[v62, v100] = {}
    for v112, (v156, v157) in v113(v107):
        if v156 is not None:
            v38.v203(v156, []).v186((v157, v112))
    v39 = {(v156, v157): v112 for v112, (v156, v157) in v113(v107) if v156 is not None}
    v40 = v140.v114(v103, v140.v115, 'para') + v140.v114(v103, v140.v115, 'para_b') + v140.v114(v103, v140.v115, 'anchored')
    for v41 in v40:
        v41['gold_idx'] = v39[v41['sid'], v41['rel']]
    v42 = v140.v114(v104, v140.v115, 'para')
    v43 = v140.v114(v104, v140.v115, 'para_hold')
    v44 = v140.v114(v104, v100(v140.v158), 'anchored')
    v45 = v116({v41['text'] for v41 in v40 + v42 + v43 + v44})
    v85(f'  tape={v109(v106)} ({v34} subject facts + {v109(v106) - v34} noise) fit={v109(v40)} seen={v109(v42)} unseen={v109(v43)} chance={v140.v127:.3f}')
    v46 = {}
    for v47 in v45:
        v117 = v140.v159(v26, v24, v21, v23, v14, v47)
        if v117 is not None:
            v46[v47] = v117.v110()
    v48 = {}
    for v47 in v45:
        v117 = v97.v160(v47)
        if v117 is not None:
            v48[v47] = v117
    v85(f'  cached states: curve {v109(v46)} (d={v211(v215(v46.v106())).v171()}) | external {v109(v48)} (d={v97.v71})')

    def run(v118, v119):
        v120 = v165.v123(v14)
        v121 = v140.v161(v8(v211(v215(v118.v106())).v171()), v14)
        v122 = v140.v162(v120, v121, v29, v37, v40, v118, None, None, v17, v13.v163, v13.v164, v15, v119)
        return {'loss': v122, 'seen_rel': v140.v124(v120, v121, v29, v37, v38, v42, v118, True), 'unseen_para': v140.v124(v120, v121, v29, v37, v38, v43, v118, True), 'anchored': v140.v124(v120, v121, v29, v37, v38, v44, v118, True)}
    v49 = v165.v123(v14)
    v50 = v140.v124(v49, None, v29, v37, v38, v42, v46, False)
    v51 = v140.v124(v49, None, v29, v37, v38, v43, v46, False)
    v85(f"fp-only: seen={v50['sel_acc']:.3f} unseen={v51['sel_acc']:.3f}")
    v52 = v125(v46, 'curve')
    v85(f"curve : seen={v52['seen_rel']['sel_acc']:.3f} unseen={v52['unseen_para']['sel_acc']:.3f} anchored={v52['anchored']['sel_acc']:.3f}")
    v53 = v125(v48, 'external')
    v85(f"extern: seen={v53['seen_rel']['sel_acc']:.3f} unseen={v53['unseen_para']['sel_acc']:.3f} anchored={v53['anchored']['sel_acc']:.3f}")
    v54 = v52['unseen_para']['sel_acc']
    v55 = v53['unseen_para']['sel_acc']
    v56 = v51['sel_acc'] <= v140.v127 + 0.1
    v57 = v55 >= v140.v127 + 0.2
    v58 = v55 >= v54 - 0.05
    v59 = v55 >= v54 + 0.1
    if not v56:
        v126 = 'TRUNK_SWAP_INVALID'
    elif v57 and v59:
        v126 = 'TRUNK_SWAP_BETTER'
    elif v57 and v58:
        v126 = 'TRUNK_SWAP_OK'
    elif v57:
        v126 = 'TRUNK_SWAP_PARTIAL'
    else:
        v126 = 'TRUNK_SWAP_NO'
    v60 = {'stage': 262, 'overall': v126, 'external_model': v13.v70, 'external_hidden': v97.v71, 'curve_trunk': v25.v67, 'steps': v17, 'n_subjects': v109(v32), 'tape_slots': v109(v106), 'subject_slots': v34, 'chance': v140.v127, 'fit_rels': v140.v115, 'exam_holdout': 'para_hold (258 unseen_para)', 'gates': {'G_exam_valid': v56, 'G_external_works': v57, 'G_interface_transfers': v58, 'G_external_better': v59}, 'summary': {'fp_only': {'seen_rel': v50, 'unseen_para': v51}, 'curve_trunk': v52, 'external_trunk': v53}, 'note': "258's exam verbatim - same seed, same subjects, same relations, same helpers - with the trunk swapped for an external frozen causal LM. Nothing about the tape moves: keys are canonical P1 fp written from characters, so the external tokenizer never touches them, and the only coupling is h_t -> W_sem -> key space. Retrieval only, so there is no vocabulary to reconcile. TRUNK_SWAP_NO would mean the channel is tuned to curve states specifically, which is the result that sinks the 'any reasoner' claim.", 'timestamp': v204.v197(v205.v198).v166(), 'wall_s': v82.v82() - v16}
    v1.v79(v188.v167(v60, indent=2), encoding='utf-8')
    v2.v79(f"# Stage 262 trunk swap\n\n**{v126}** external={v13.v70} (d={v97.v71}) slots={v109(v106)} chance={v140.v127:.3f}\n\n- unseen paraphrase: fp-only **{v51['sel_acc']:.3f}** | curve **{v54:.3f}** | external **{v55:.3f}**\n- seen paraphrase: curve {v52['seen_rel']['sel_acc']:.3f} | external {v53['seen_rel']['sel_acc']:.3f}\n- anchored: curve {v52['anchored']['sel_acc']:.3f} | external {v53['anchored']['sel_acc']:.3f}\n", encoding='utf-8')
    v85(v188.v167({'overall': v126, 'gates': v60['gates']}, indent=2))
    return 0
if v61 == '__main__':
    raise v128(v168())