"""
Stage 250 — Long masked-only night: understanding CE without binding leak.

Single arm (no full-bindings control) so wall-clock goes into useful steps.
Periodically probe next_tok + slot mem; optional hop-gated extra writes mid-stream.
Saves checkpoint for resume.

  python _stage250_masked_night.py [--smoke] [--steps N] [--resume]
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage225_family_fork as s225
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data, sample_windows, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
v0 = v12('results')
v1 = v12('checkpoints/stage191_p1_curve.pt')
v2 = v12('checkpoints/stage250_masked_night.pt')
v3 = v12('checkpoints/stage248_masked_upper.pt')
v4 = v0 / 'stage250_decision.json'
v5 = v0 / 'stage250_mini.md'
v6 = v0 / '_stage250_log.txt'
v7 = v12('data/_wikitext103_train.txt')
v8 = v12('data/stage191_exam_v3.jsonl')
v9 = 250

def log(v13: v10) -> None:
    v14 = v13 if v13.v137('\n') else v13 + '\n'
    try:
        v138(v14, end='', flush=True)
    except v73:
        v138(v14.v223('ascii', 'replace').v199('ascii'), end='', flush=True)
    v6.v139.v74(parents=True, exist_ok=True)
    with v6.v140('a', encoding='utf-8') as v18:
        v18.v141(v14)

def mask_facts(v15: v10, v16) -> v10:
    v17 = v15
    for v18 in v16:
        v17 = v17.v142(v18['sent'], 'The chronicle continues without naming any director.')
        v17 = v17.v142(v18['S'], 'Someone')
        v17 = v17.v142(v18['value'], 'somewhere')
    return v17

def next_tok(v19, v20, v21, v22, v23):
    if not v22:
        return v143('nan')
    v24 = 0
    for v25 in v22:
        v26 = [v179(v19, v20, v21, v25['ctx_ids'], v103, v23) for v103 in v25['cand_ids']]
        v24 += v11(v11(v215(v26)) == v25['gold_idx'])
    return v24 / v144(v22)

def np_argmax(v26):
    import numpy as np
    return v11(v180.v145(v26))

def main() -> v11:
    v27 = v146.v75()
    v27.v76('--smoke', action='store_true')
    v27.v76('--steps', type=v11, default=0)
    v27.v76('--resume', action='store_true')
    v28 = v27.v77()
    v6.v78('', encoding='utf-8')
    v23 = v147.v23('cuda' if v147.v200.v181() else 'cpu')
    v29 = v148.v79(v9)
    v147.v80(v9)
    v30 = v81.v81()
    v31 = v28.v31 or (200 if v28.v82 else 60000)
    v32 = 12 if v28.v82 else 48
    v33 = 50 if v28.v82 else 5000
    v34 = 30 if v28.v82 else 300
    v35 = 30 if v28.v82 else 400
    v36 = 200 if v28.v82 else 12000
    v37 = 40 if v28.v82 else 300
    v38 = 30 if v28.v82 else 100
    v83(f'Stage250 start {v221.v213(v222.v214).v176()} steps={v31} resume={v28.v91}')
    v84, v84, v85, v86 = v87()
    v39 = v149.v88(v10(v182.v150))
    v40 = v39.v89()
    v21 = v39.v151(v152) or 0
    v20 = v201.v183(v39, v85, v21, v40).v90(v23)
    v19 = v184(v86, v40).v90(v23)
    v41 = v3 if v28.v91 and v3.v111() else v1
    if v28.v91 and v2.v111():
        v41 = v2
    v19.v92(v147.v185(v41, map_location=v23, weights_only=False)['model'])
    v19.v93()
    v83(f'loaded {v41}')
    v42 = v94(v19, v85, v23)
    with v7.v140('r', encoding='utf-8', errors='ignore') as v18:
        v15 = v18.v153(3000000 if v28.v82 else 25000000)
    v43 = v95(v186.v154((v13.v202(1) for v13 in v224.v216(v15) if v144(v13.v202(1)) >= 5)))
    v29.v96(v43)
    v44 = [v62.v155() for v62 in v15.v187('\n') if v144(v62.v155()) > 150]
    v45 = v95(v186.v154((v156 for v156 in v226.v225('[A-Za-z][a-z]{2,}', v15) if v144(v156) <= 14)))[:v37]
    v46 = [v156 for v156 in v203(v217(v43), v29, v32 + 40) if v144(v156) >= 5][:v32]
    v16 = []
    for v97, v98 in v99(v46):
        v100 = v43[v97]
        v16.v157({'S': v98, 'value': v100, 'sent': f'{v98} was appointed director of {v100} in 1987 .', 'fid': v97})
    v47 = [v18['value'] for v18 in v16] + v43[v32:v32 + 80]
    v48 = v42.v101('In the report the organization appointed a new director of governance.')
    if v48 is None:
        v48 = v42.v188(['organization'])[0]
    v49 = []
    for v18 in v16:
        v102 = v42.v188([v18['S']])[0]
        v103 = v42.v101(v18['sent'], exclude=v18['value'])
        v104 = v190.v189(v102 + v103, dim=-1) if v103 is not None else v102
        v49.v157((v143((v104 * v48).v218()), v18))
    v49.v105(key=lambda v204: -v204[0])
    v50 = [v18 for v84, v18 in v49[:v210(2, v144(v16) // 2)]]
    v51 = []
    for v97, v18 in v99(v16):
        if v97 < v144(v44):
            v51.v157(v44[v97][:280])
        v51.v157(v18['sent'])
    v52 = ' '.v106(v51 + v44[v144(v16):v144(v16) + 80])
    v53 = v107(v52, v16)
    v108, v109 = v158.v110(v53, v39, v21, max_lines=v36, min_line_len=16)
    v22 = []
    if v8.v111():
        with v8.v140(encoding='utf-8') as v18:
            for v14 in v18:
                v25 = v198.v205(v14)
                if v25.v219('type') == 'next_tok':
                    v22.v157(v25)
                if v144(v22) >= v38:
                    break
    v54 = v184(v86, v40).v90(v23)
    v54.v92(v147.v185(v1, map_location=v23, weights_only=False)['model'])
    v54.v93()
    v55 = v94(v54, v85, v23)
    v112, v113 = v159.v114(v55, v50)
    v56 = v160.v115(v55, v45)
    v13 = v161.v116(v19)
    v158.v117(v13, 'upper')
    v57 = [v62 for v62 in v13.v126() if v62.v162]
    v58 = v147.v163.v118(v57, lr=0.0003, weight_decay=0.01)
    v59 = v148.v79(v9 + 3)
    v60 = []
    for v61 in v119(1, v31 + 1):
        v120 = v206(v108, v109, v207, v59, v21).v90(v23)
        v121 = v120 == v21
        v164, v84, v165 = v13.v166(v20[v120], v121, ids=v120)
        v122 = v120[:, 1:]
        v123 = ~v121[:, :-1] & ~v121[:, 1:]
        v124 = v190.v167(v164[:, :-1][v123], v122[v123])
        v125 = v124 + v191 * v165[~v121].v208()
        v58.v168(set_to_none=True)
        v125.v169()
        v147.v209.v192.v170(v57, 1.0)
        v58.v61()
        if v61 % v210(100, v31 // 20) == 0:
            v83(f'  night step {v61}/{v31}: ce={v143(v124):.3f}')
        if v61 % v33 == 0 or v61 == v31:
            v13.v93()
            v171 = v193(v13, v20, v21, v22, v23)
            v172 = v159.v134(v50, v47, v55, v112, v113, v9)
            v173 = v211.v194(v13, v108, v109, v20, v21, v23, v148.v79(v9 + v61), 8 if v28.v82 else 16)
            v60.v157({'step': v61, 'ce': v143(v124), 'next_tok': v171, 'mem': v172, 'domain_win': v173})
            v83(f"  probe@{v61}: nt={v171:.3f} mem={v172['four_way']:.3f} fb_top1={v172['full_bank_top1']:.3f} win={v173:.3f}")
            v12('checkpoints').v74(exist_ok=True)
            v147.v195({'model': v13.v220(), 'stage': 250, 'step': v61, 'curve': v60}, v2)
            v13.v196()
            v158.v117(v13, 'upper')
    v13.v93()
    for v62 in v13.v126():
        v62.v174(False)
    v63 = v175.v127(v148.v79(v9 + 1), v28.v82)
    v128, v129 = v158.v110(v63, v39, v21, max_lines=v197(v36, 8000), min_line_len=20)
    v64 = v160.v130(v13, v128, v129, v20, v21, v23, v34, v9 + 9)
    v65 = v94(v64, v85, v23)
    v131, v132 = v160.v133(v212(256).v90(v23), v160.v115(v65, v45), v56, v29, v35, v23)
    v66 = v159.v134(v50, v47, v65, v112, v113, v9, W_bwd=v131)
    v67 = v60[-1]['next_tok'] if v60 else v143('nan')
    v68 = v60[0]['next_tok'] if v60 else v143('nan')
    v69 = v66['four_way'] >= 0.8
    v70 = v67 == v67 and v67 >= (v68 - 0.03 if v68 == v68 else 0.5)
    v71 = v144(v60) >= 2
    if v69 and v70:
        v135 = 'MASKED_NIGHT_OK'
    elif v69 or v70:
        v135 = 'MASKED_NIGHT_PARTIAL'
    else:
        v135 = 'MASKED_NIGHT_NO'
    v17 = {'stage': 250, 'overall': v135, 'steps': v31, 'gates': {'G_mem_cf_ge_0p80': v69, 'G_under_stable': v70, 'G_probes': v71}, 'curve': v60, 'mem_after_code_W': v66, 'W_align': v132, 'n_hop_facts': v144(v50), 'ckpt': v10(v2), 'timestamp': v221.v213(v222.v214).v176(), 'wall_s': v81.v81() - v30}
    v4.v78(v198.v177(v17, indent=2), encoding='utf-8')
    v5.v78(f"# Stage 250 masked night\n\n**{v135}** steps={v31} mem_cf={v66['four_way']:.3f} fb_top1={v66['full_bank_top1']:.3f} nt_final={v67}\n", encoding='utf-8')
    v83(v198.v177({'overall': v135, 'mem_cf': v66, 'probes': v144(v60)}, indent=2))
    return 0
if v72 == '__main__':
    raise v136(v178())