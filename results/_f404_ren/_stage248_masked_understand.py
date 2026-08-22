"""
Stage 248 — Masked-CE understanding growth while facts live only in slots.

Branch from 247: CE on binding-stripped domain text (upper layers, frozen arc_enc);
novel facts written to canonical slots (+ optional hop gate). Measure:
  - understanding: next_tok exam (+ domain window acc)
  - memory: slot recall before/after long CE and after code-shift+W
  - edit: slot overwrite collateral
  - control arm: CE on FULL text with bindings (facts leak into weights via upper)

  python _stage248_masked_understand.py [--smoke] [--steps N]
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
import numpy as np
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
v0 = v10('results')
v1 = v0 / 'stage248_decision.json'
v2 = v0 / 'stage248_mini.md'
v3 = v0 / '_stage248_log.txt'
v4 = v10('checkpoints/stage191_p1_curve.pt')
v5 = v10('data/_wikitext103_train.txt')
v6 = v10('data/stage191_exam_v3.jsonl')
v7 = 248

def log(v11: v8) -> None:
    v12 = v11 if v11.v148('\n') else v11 + '\n'
    try:
        v149(v12, end='', flush=True)
    except v80:
        v149(v12.v224('ascii', 'replace').v203('ascii'), end='', flush=True)
    v3.v150.v81(parents=True, exist_ok=True)
    with v3.v151('a', encoding='utf-8') as v16:
        v16.v152(v12)

def mask_facts(v13: v8, v14) -> v8:
    v15 = v13
    for v16 in v14:
        v15 = v15.v153(v16['sent'], 'The chronicle continues without naming any director.')
        v15 = v15.v153(v16['S'], 'Someone')
        v15 = v15.v153(v16['value'], 'somewhere')
    return v15

def ce_upper(v17, v18, v19, v20, v21, v22, v23, v24, v25):
    v11 = v154.v82(v17)
    v155.v83(v11, 'upper')
    v26 = [v30 for v30 in v11.v94() if v30.v156]
    v27 = v170.v157.v84(v26, lr=0.0003, weight_decay=0.01)
    v28 = v158.v85(v24)
    for v29 in v86(1, v23 + 1):
        v87 = v204(v18, v19, v205, v28, v21).v110(v22)
        v88 = v87 == v21
        v159, v104, v160 = v11.v161(v20[v87], v88, ids=v87)
        v89 = v87[:, 1:]
        v90 = ~v88[:, :-1] & ~v88[:, 1:]
        v91 = v191.v162(v159[:, :-1][v90], v89[v90])
        v92 = v91 + v192 * v160[~v88].v206()
        v27.v163(set_to_none=True)
        v92.v164()
        v170.v207.v193.v165(v26, 1.0)
        v27.v29()
        if v29 % v208(100, v23 // 10) == 0:
            v103(f'  {v25} step {v29}/{v23}: ce={v167(v91):.3f}')
    v11.v93()
    for v30 in v11.v94():
        v30.v166(False)
    return v11

def next_tok(v17, v20, v21, v31, v22):
    if not v31:
        return v167('nan')
    v32 = 0
    for v33 in v31:
        v95 = [v194(v17, v20, v21, v33['ctx_ids'], v72, v22) for v72 in v33['cand_ids']]
        v32 += v9(v9(v225.v218(v95)) == v33['gold_idx'])
    return v32 / v168(v31)

def main() -> v9:
    v34 = v169.v96()
    v34.v97('--smoke', action='store_true')
    v34.v97('--steps', type=v9, default=0)
    v35 = v34.v98()
    v3.v99('', encoding='utf-8')
    v22 = v170.v22('cuda' if v170.v209.v195() else 'cpu')
    v28 = v158.v85(v7)
    v170.v100(v7)
    v36 = v101.v101()
    v23 = v35.v23 or (150 if v35.v102 else 8000)
    v37 = 12 if v35.v102 else 40
    v38 = 40 if v35.v102 else 400
    v39 = 40 if v35.v102 else 500
    v40 = 250 if v35.v102 else 10000
    v41 = 50 if v35.v102 else 300
    v42 = 40 if v35.v102 else 120
    v103(f'Stage248 start {v222.v215(v223.v216).v187()} steps={v23}')
    v104, v104, v105, v106 = v107()
    v43 = v171.v108(v8(v196.v172))
    v44 = v43.v109()
    v21 = v43.v173(v174) or 0
    v20 = v210.v197(v43, v105, v21, v44).v110(v22)
    v45 = v198(v106, v44).v110(v22)
    v45.v111(v170.v199(v4, map_location=v22, weights_only=False)['model'])
    v45.v93()
    for v30 in v45.v94():
        v30.v166(False)
    v46 = v112(v45, v105, v22)
    with v5.v151('r', encoding='utf-8', errors='ignore') as v16:
        v13 = v16.v175(4000000 if v35.v102 else 20000000)
    v47 = v113(v144.v176((v11.v211(1) for v11 in v226.v219(v13) if v168(v11.v211(1)) >= 5)))
    v28.v114(v47)
    v48 = [v30.v177() for v30 in v13.v200('\n') if v168(v30.v177()) > 160]
    v49 = v113(v144.v176((v179 for v179 in v228.v227('[A-Za-z][a-z]{2,}', v13) if v168(v179) <= 14)))[:v41]
    v50 = v178.v115(v46, v49)
    v51 = [v179 for v179 in v212(v220(v47), v28, v37 + 40) if v168(v179) >= 5][:v37]
    v14 = []
    for v116, v117 in v118(v51):
        v119 = v47[v116]
        v14.v180({'S': v117, 'value': v119, 'sent': f'{v117} was appointed director of {v119} in 1987 .', 'fid': v116})
    v52 = [v16['value'] for v16 in v14] + v47[v37:v37 + 80]
    v53 = []
    for v116, v16 in v118(v14):
        if v116 < v168(v48):
            v53.v180(v48[v116][:300])
        v53.v180(v16['sent'])
    v54 = ' '.v120(v53 + v48[v168(v14):v168(v14) + 40])
    v55 = v121(v54, v14)
    v122, v123 = v155.v124(v54, v43, v21, max_lines=v40, min_line_len=16)
    v125, v126 = v155.v124(v55, v43, v21, max_lines=v40, min_line_len=16)
    v31 = []
    if v6.v127():
        with v6.v151(encoding='utf-8') as v16:
            for v12 in v16:
                v33 = v202.v213(v12)
                if v33.v221('type') == 'next_tok':
                    v31.v180(v33)
                if v168(v31) >= v42:
                    break
    v128, v129 = v181.v130(v46, v14)
    v56 = v181.v131(v14, v52, v46, v128, v129, v7)
    v57 = v132(v45, v20, v21, v31, v22)
    v103('arm MASKED: slots + CE without bindings')
    v58 = v133(v45, v125, v126, v20, v21, v22, v23, v7 + 2, 'masked')
    v59 = v181.v131(v14, v52, v46, v128, v129, v7)
    v60 = v132(v58, v20, v21, v31, v22)
    v61 = v182.v134(v58, v125, v126, v20, v21, v22, v158.v85(v7 + 3), 12 if v35.v102 else 24)
    v103('arm FULL: CE with bindings (leak control)')
    v62 = v133(v45, v122, v123, v20, v21, v22, v23, v7 + 4, 'full')
    v63 = v112(v62, v105, v22)
    v64 = v181.v131(v14, v52, v63, v128, v129, v7)
    v65 = v132(v62, v20, v21, v31, v22)
    v66 = v183.v135(v158.v85(v7 + 1), v35.v102)
    v136, v137 = v155.v124(v66, v43, v21, max_lines=v40, min_line_len=20)
    v67 = v178.v138(v58, v136, v137, v20, v21, v22, v38, v7 + 5)
    v68 = v112(v67, v105, v22)
    v139, v140 = v178.v141(v214(256).v110(v22), v178.v115(v68, v49), v50, v28, v39, v22)
    v69 = v181.v131(v14, v52, v68, v128, v129, v7, W_bwd=v139)
    v142, v143 = (v128.v184(), v113(v129))
    v143[0] = v47[v37 + 2]
    v70 = v144(v14[0])
    v70['value'] = v143[0]
    v70['sent'] = f"{v70['S']} was appointed director of {v143[0]} in 1987 ."
    v71 = v46.v185([v70['S']])[0]
    v72 = v46.v145(v70['sent'], exclude=v143[0])
    v142[0] = v191.v186(v71 + v72, dim=-1) if v72 is not None else v71
    v73 = v181.v131([v70], v52 + [v143[0]], v46, v142, v143, v7)
    v74 = v181.v131(v14[1:], v52, v46, v142, v143, v7)
    v75 = v69 >= 0.8
    v76 = v60 >= v57 - 0.02
    v77 = v59 >= v64 - 0.05 and v60 + 0.02 >= v65
    v78 = v73 >= 0.8 and v201(v74 - v181.v131(v14[1:], v52, v46, v128, v129, v7)) <= 0.05
    if v75 and v76 and v78:
        v146 = 'MASKED_UNDERSTAND_OK'
    elif v75 and (v76 or v78):
        v146 = 'MASKED_UNDERSTAND_PARTIAL'
    else:
        v146 = 'MASKED_UNDERSTAND_NO'
    v15 = {'stage': 248, 'overall': v146, 'steps': v23, 'gates': {'G_mem_after_shift_ge_0p80': v75, 'G_under_not_worse': v76, 'G_masked_vs_full_ok': v77, 'G_edit_clean': v78}, 'baseline': {'mem': v56, 'next_tok': v57}, 'masked': {'mem': v59, 'next_tok': v60, 'domain_win': v61, 'mem_after_code_W': v69, 'W_align': v140}, 'full_bindings_CE': {'mem_query_drifted': v64, 'next_tok': v65}, 'edit': {'new': v73, 'retained': v74}, 'timestamp': v222.v215(v223.v216).v187(), 'wall_s': v101.v101() - v36}
    v1.v99(v202.v188(v15, indent=2), encoding='utf-8')
    v2.v99(f'# Stage 248 masked understand\n\n**{v146}** steps={v23} mem_cf={v69:.3f} nt_m={v60:.3f} nt_full={v65:.3f}\n', encoding='utf-8')
    v103(v202.v188({'overall': v146, 'mem_cf': v69, 'nt_m': v60}, indent=2))
    if not v35.v102 and v23 >= 2000:
        v10('checkpoints').v81(exist_ok=True)
        v170.v189({'model': v58.v217(), 'stage': 248, 'steps': v23}, 'checkpoints/stage248_masked_upper.pt')
    return 0
if v79 == '__main__':
    raise v147(v190())