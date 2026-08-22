"""
Stage 259 — Hot swap: change a fact in memory, get the new answer, zero gradient steps.

256 proved the head USES the tape. This asks the question that separates "knowledge lives in
memory" from "knowledge lives in weights" in one move a reader can check without a benchmark:

    edit the slot -> ask again -> the model says the new thing, immediately, no training

Keys are written as norm( fp(anchor) + ctx_fp(sentence, exclude=value) ), so a value never
enters its own key. Replacing it leaves the key BIT-IDENTICAL — asserted here, not assumed.
That is what makes this an update rather than a re-index, and it is why the edit costs no
gradient: nothing in the geometry moved.

Nothing trains. The glue (W_q + gate + tau) is loaded from stage 256 and its parameters are
snapshotted and compared bit for bit at the end, so "zero-train" is a measured claim.

What would break the story, and is therefore tested:
  old value survives    the two answers coexist -> the edit did not take
  neighbours die        editing one fact damages others -> not a local update
  keys moved            it was a re-index, and the editability claim is empty
  params moved          something trained; the demo is a lie
  second edit ignored   first write wins -> a cache, not a memory
  empty tape answers    the value was in the weights all along

Data is rebuilt with stage 256's seed and call order so the loaded glue matches the tape it
was fit on. Requires checkpoints/stage256_slot_bias.pt — run 256 first.

  python _stage259_hot_swap.py [--smoke]
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, TapeView, free_decode_value, load_glue, value_exact_match
v0 = v11('results')
v1 = v0 / 'stage259_decision.json'
v2 = v0 / 'stage259_mini.md'
v3 = v0 / '_stage259_log.txt'
v4 = v11('checkpoints/stage191_p1_curve.pt')
v5 = v11('checkpoints/stage253_joint_l02.pt')
v6 = v11('checkpoints/stage256_slot_bias.pt')
v7 = v11('data/_wikitext103_train.txt')
v8 = 256

def log(v12: v9) -> None:
    v13 = v12 if v12.v134('\n') else v12 + '\n'
    try:
        v135(v13, end='', flush=True)
    except v76:
        v135(v13.v210('ascii', 'replace').v191('ascii'), end='', flush=True)
    v3.v136.v77(parents=True, exist_ok=True)
    with v3.v137('a', encoding='utf-8') as v27:
        v27.v138(v13)

def rebuild_256_tape(v14: v78, v15: v79, v16: v10, v17, v18: v80):
    """Mirror of stage 256's data build — same seed, same call order, same loops."""
    v19 = v139.v81(v8)
    v20 = 8 if v18 else 48
    v21 = 150 if v18 else 1200
    v22 = 400 if v18 else 6000
    with v7.v137('r', encoding='utf-8', errors='ignore') as v27:
        v82 = v27.v140(1000000 if v18 else 6000000)
    v23 = v83(v171.v141((v12.v176(1) for v12 in v175.v145(v82) if v160(v12.v176(1)) >= 5)))
    v19.v84(v23)
    v24 = [v173.v172() for v173 in v82.v192('\n') if v160(v173.v172()) >= 60][:v22]
    v25 = [v142 for v142 in v193(v92(v23), v19, v20 + 30) if v160(v142) >= 5][:v20]
    v26 = []
    for v85, v86 in v87(v25):
        v26.v143({'S': v86, 'value': v23[v85], 'sent': v204.v194(S=v86, V=v23[v85]), 'fid': f'f{v85}', 'glue_train': v85 % 2 == 0})
    v88, v89 = ([], [])
    for v27 in v26:
        v90 = v14.v174([v27['S']])[0]
        v91 = v14.v144(v27['sent'], exclude=v27['value'])
        v88.v143(v205.v195(v90 + v91, dim=-1) if v91 is not None else v90)
        v89.v143(v27['value'])
    v28 = v92(v89)
    for v29 in v24:
        if v160(v89) >= v20 + v21:
            break
        for v12 in v175.v145(v29):
            v146 = v12.v176(1)
            if v160(v146) < 5 or v146 in v28:
                continue
            v177, v178 = (v150(0, v12.v211() - 120), v196(v160(v29), v12.v212() + 120))
            v91 = v14.v144(v29[v177:v178], exclude=v146)
            if v91 is None:
                continue
            v147 = [v142 for v142 in v213.v206(v29[v177:v12.v211()]) if v142 != v146]
            if not v147:
                continue
            v88.v143(v205.v195(v14.v174([v147[-1]])[0] + v91, dim=-1))
            v89.v143(v146)
            v28.v179(v146)
            if v160(v89) >= v20 + v21:
                break
    v30 = v93(v96.v197(v88, 0).v112(v17), v89, v15, v16)
    v31 = [v94 for v94 in v23 if v94 not in v28]
    return (v26, v30, v31)

@v96.v40()
def em_against(v32, v33, v34, v15, v14, v30, v35, v16, v36, v17, v37, v38):
    """pairs: (subject, expected_value). Free-form greedy decode, no candidate set."""
    v39 = 0
    for v86, v95 in v35:
        v148, v106 = v149(v32, v33, v34, v15, v14, v30, v86, v16, v36, v17, k=v37, max_new=v38)
        v39 += v10(v180(v148, v95))
    return v39 / v150(1, v160(v35))

def param_fingerprint(v32) -> v9:
    """Bit-level snapshot of everything trainable, so 'nothing trained' is checkable."""
    import hashlib
    v41 = v151.v97()
    for v42 in v83(v32.v215.v214().v167()) + v83(v32.v216.v214().v167()) + [v32.v198.v181()]:
        v41.v152(v42.v181().v217().v207().v182())
    return v41.v98()

def main() -> v10:
    v43 = v153.v99()
    v43.v100('--smoke', action='store_true')
    v43.v100('--topk', type=v10, default=8)
    v44 = v43.v101()
    v3.v102('', encoding='utf-8')
    v17 = v96.v17('cuda' if v96.v199.v183() else 'cpu')
    v96.v103(v8)
    v45 = v104.v104()
    v37 = v44.v46
    v38 = 4 if v44.v18 else 6
    if not v6.v154():
        v105(f'missing {v6} — run _stage256_slot_bias_decode.py first')
        return 1
    v105(f'Stage259 hot swap start {v208.v202(v209.v203).v168()} device={v17}')
    v106, v106, v107, v108 = v109()
    v15 = v79.v110(v9(v184.v155))
    v36 = v15.v111()
    v16 = v15.v156(v157) or 0
    v34 = v200.v185(v15, v107, v16, v36).v112(v17)
    v47 = v5 if v5.v158() else v4
    v33 = v186(v108, v36).v112(v17)
    v33.v113(v96.v187(v47, map_location=v17, weights_only=False)['model'])
    v33.v114()
    for v48 in v33.v115():
        v48.v159(False)
    v49 = v186(v108, v36).v112(v17)
    v49.v113(v96.v187(v4, map_location=v17, weights_only=False)['model'])
    v49.v114()
    for v48 in v49.v115():
        v48.v159(False)
    v14 = v78(v49, v107, v17)
    v26, v30, v31 = v116(v14, v15, v16, v17, v44.v18)
    v50 = [v27 for v27 in v26 if not v27['glue_train']]
    v32 = v117(v33, v17, v6)
    if v32 is None:
        v105('glue failed to load')
        return 1
    v105(f'  trunk={v47.v132} glue={v6.v132} slots={v160(v30.v167)} eval_facts={v160(v50)}')
    if v160(v31) < 2 * v160(v50):
        v105(f'  not enough unused values ({v160(v31)}) for two rounds of edits')
        return 1
    v51 = v118(v32)
    v52 = v30.v161.v119()
    v53 = v120(v32, v33, v34, v15, v14, v30, [(v27['S'], v27['value']) for v27 in v50], v16, v36, v17, v37, v38)
    v105(f'before edit: EM={v53:.3f}')
    v54 = {v27['fid']: v31[v85] for v85, v27 in v87(v50)}
    v121, v122, v123, v124 = ([], [], [], [])
    for v27 in v50:
        v125 = v104.v162()
        v126 = v30.v163(v27['value'], v54[v27['fid']], v15, v16)
        v124.v143((v104.v162() - v125) * 1000000.0)
        v121.v143(v120(v32, v33, v34, v15, v14, v126, [(v27['S'], v54[v27['fid']])], v16, v36, v17, v37, v38))
        v122.v143(v120(v32, v33, v34, v15, v14, v126, [(v27['S'], v27['value'])], v16, v36, v17, v37, v38))
        v127 = [(v201['S'], v201['value']) for v201 in v50 if v201 is not v27][:4]
        if v127:
            v123.v143(v120(v32, v33, v34, v15, v14, v126, v127, v16, v36, v17, v37, v38))
    v55 = v128(v188.v164(v121))
    v56 = v128(v188.v164(v122))
    v57 = v128(v188.v164(v123)) if v123 else v128('nan')
    v105(f'after edit:  new={v55:.3f}  old={v56:.3f}  neighbours={v57:.3f} (edit {v188.v164(v124):.0f} us)')
    v58 = {v27['fid']: v31[v160(v50) + v85] for v85, v27 in v87(v50)}
    v129, v130 = ([], [])
    for v27 in v50:
        v126 = v30.v163(v27['value'], v54[v27['fid']], v15, v16)
        v126 = v126.v163(v54[v27['fid']], v58[v27['fid']], v15, v16)
        v129.v143(v120(v32, v33, v34, v15, v14, v126, [(v27['S'], v58[v27['fid']])], v16, v36, v17, v37, v38))
        v130.v143(v120(v32, v33, v34, v15, v14, v126, [(v27['S'], v54[v27['fid']])], v16, v36, v17, v37, v38))
    v59 = v128(v188.v164(v129))
    v60 = v128(v188.v164(v130))
    v105(f'second edit: newest={v59:.3f}  superseded={v60:.3f}')
    v61 = [(v27['S'], v54[v27['fid']]) for v27 in v50]
    v62 = v120(v32, v33, v34, v15, v14, v30.v165(), v61, v16, v36, v17, v37, v38)
    v63 = v80(v96.v166(v30.v161, v52))
    v64 = v118(v32) == v51
    v65 = v53 >= 0.4
    v66 = v55 >= 0.4 and v55 >= v53 - 0.15
    v67 = v56 <= 0.1
    v68 = not v188.v189(v57) and v57 >= 0.7 * v53
    v69 = v63
    v70 = v64
    v71 = v59 >= 0.4 and v60 <= 0.1
    v72 = v62 <= 0.1
    v73 = v65 and v66 and v67 and v69 and v70 and v72
    if v73 and v68 and v71:
        v131 = 'HOT_SWAP_OK'
    elif v73:
        v131 = 'HOT_SWAP_PARTIAL'
    else:
        v131 = 'HOT_SWAP_NO'
    v74 = {'stage': 259, 'overall': v131, 'trunk': v47.v132, 'glue': v6.v132, 'topk': v37, 'tape_slots': v160(v30.v167), 'n_eval_facts': v160(v50), 'gates': {'G_baseline_alive': v65, 'G_answer_follows_edit': v66, 'G_old_answer_dies': v67, 'G_edit_is_local': v68, 'G_keys_untouched': v69, 'G_zero_gradient_steps': v70, 'G_latest_write_wins': v71, 'G_no_param_leak': v72}, 'summary': {'em_before': v53, 'em_new_value_after_edit': v55, 'em_old_value_after_edit': v56, 'em_neighbours_after_edit': v57, 'em_newest_after_second_edit': v59, 'em_superseded_after_second_edit': v60, 'em_empty_tape': v62, 'edit_wall_us_mean': v128(v188.v164(v124)), 'edit_wall_us_note': 'includes a defensive copy of the value list (O(slots)); an in-place product edit is a tokenize plus two assignments', 'keys_bit_identical': v63, 'glue_params_bit_identical': v64, 'gradient_steps': 0}, 'note': 'No training anywhere: the 256 glue is loaded and its parameters are hashed before and after, so zero-train is measured rather than asserted. A value never enters its own key (ctx_fp excludes it at write time), so the edit leaves keys bit-identical — an update, not a re-index. Each fact is edited on its own view of the tape and scored free-form with no candidate set, on the half of the facts 256 never fit. The second edit checks the tape tracks the latest write instead of behaving like a write-once cache.', 'timestamp': v208.v202(v209.v203).v168(), 'wall_s': v104.v104() - v45}
    v1.v102(v190.v169(v74, indent=2), encoding='utf-8')
    v2.v102(f'# Stage 259 hot swap\n\n**{v131}** glue={v6.v132} slots={v160(v30.v167)} facts={v160(v50)} gradient steps **0**\n\n- before edit **{v53:.3f}** -> after edit, new value **{v55:.3f}**, old value **{v56:.3f}**\n- neighbours untouched: {v57:.3f}\n- edited again: newest **{v59:.3f}**, superseded {v60:.3f}\n- empty tape (leak floor): {v62:.3f}\n- keys bit-identical: {v63} | glue params bit-identical: {v64}\n- edit cost: {v188.v164(v124):.0f} us, 0 gradient steps\n', encoding='utf-8')
    v105(v190.v169({'overall': v131, 'gates': v74['gates'], 'summary': v74['summary']}, indent=2))
    return 0
if v75 == '__main__':
    raise v133(v170())