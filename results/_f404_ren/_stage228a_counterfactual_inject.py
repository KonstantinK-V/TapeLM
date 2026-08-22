"""
Stage 228a — Counterfactual inject probe (226 boundary).

Tests whether head_code *follows* injected values vs lexical prior only:
  - none / gold_inject / wrong_inject (code-comment form)
  - 4-way rank of gold
  - span logprob of gold string
  - sensitivity: P(gold|gold_inject) - P(gold|wrong_inject)

  python _stage228a_counterfactual_inject.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE
v0 = v10('results')
v1 = v0 / 'stage228a_decision.json'
v2 = v0 / 'stage228a_mini.md'
v3 = v10('checkpoints/stage191_p1_curve.pt')
v4 = v10('data/_wikitext103_train.txt')
v5 = 2281

def log(v11: v6) -> None:
    v60(v11, flush=True)

def prompt_code_comment(v12: v6, v13: v6 | None) -> v6:
    if v13 is None:
        return f'# TODO\ndef org_of_{v12}():\n    return '
    return f'# org[{v12}] = {v13}\ndef org_of_{v12}():\n    return '

@v65.v23()
def argmax_token(v14, v15, v16, v17, v18, v19: v6) -> v7:
    v20 = v15.v107(v19).v20
    if not v20:
        return -1
    v21 = v65.v61([v20], dtype=v65.v108, device=v18)
    v22 = v21 == v16
    v62, v63, v63 = v14.v64(v17[v21], v22, ids=v21)
    return v7(v62[0, -1].v109())

@v65.v23()
def span_logprob(v14, v15, v16, v17, v18, v24: v6, v25: v6) -> v8:
    """Logprob of target token ids continuing prefix (teacher-forced on target)."""
    v26 = v15.v107(v24).v20
    v27 = v15.v107(v25).v20
    if not v26 or not v27:
        return v8('-inf')
    v28 = v26 + v27
    v21 = v65.v61([v28], dtype=v65.v108, device=v18)
    v22 = v21 == v16
    v62, v63, v63 = v14.v64(v17[v21], v22, ids=v21)
    v29 = 0.0
    for v66, v67 in v68(v27):
        v69 = v111(v26) - 1 + v66
        if v69 < 0 or v69 >= v62.v144[1]:
            break
        v70 = v145.v132(v62[0, v69], dim=-1)[v67]
        v29 += v8(v70)
    return v29 / v100(1, v111(v27))

@v65.v23()
def rank4_last(v14, v15, v16, v17, v18, v19: v6, v30: v6, v31: v110[v6], v32) -> v9:
    v20 = v15.v107(v19).v20
    if not v20:
        return False
    v21 = v65.v61([v20], dtype=v65.v108, device=v18)
    v22 = v21 == v16
    v62, v63, v63 = v14.v64(v17[v21], v22, ids=v21)
    v33 = v62[0, -1]
    v34 = [v30] + [v127 for v127 in v31 if v127 != v30][:3]
    while v111(v34) < 4:
        v34.v112(v31[0])
    v32.v71(v34)
    v35 = [v8(v33[v15.v107(v113).v20[0]]) if v15.v107(v113).v20 else -1000000000.0 for v113 in v34]
    return v34[v7(v151.v109(v35))] == v30

def main() -> v7:
    v36 = v114.v72()
    v36.v73('--smoke', action='store_true')
    v37 = v36.v74()
    v18 = v65.v18('cuda' if v65.v146.v133() else 'cpu')
    v38 = 80 if v37.v75 else 600
    v39 = 400 if v37.v75 else 8000
    v40 = 12 if v37.v75 else 40
    v32 = v115.v76(v5)
    v77, v78, v79, v80 = v81()
    v15 = v116.v82(v6(v134.v117))
    v16 = v15.v118(v119) or 0
    v17 = v147.v135(v15, v79, v16, v15.v148()).v83(v18)
    v41 = v136(v80, v15.v148()).v83(v18)
    v41.v84(v65.v137(v3, map_location=v18, weights_only=False)['model'])
    v41.v85()
    with v4.v120('r', encoding='utf-8', errors='ignore') as v86:
        v87 = v110(v149.v138((v11.v154(1) for v11 in v158.v157(v86.v159(4000000)) if v111(v11.v154(1)) >= 5)))
    v42 = v121(v139(v87), v32, v40 + 10)[:v40]
    v43 = v87[:v40]
    v44 = v122.v88(v115.v76(v5 + 1), v37.v75)
    v89, v90 = v123.v91(v44, v15, v16, max_lines=v39, min_line_len=20)
    v45 = v122.v92(v41, v89, v90, v17, v16, v18, v38, v5 + 2)
    v46 = {'none': 0, 'gold': 0, 'wrong': 0}
    v47 = 0
    v48 = 0
    v49 = []
    v50 = []
    v51 = []
    v52 = 0
    for v66, (v12, v30) in v68(v124(v42, v43)):
        v93 = v43[(v66 + 7) % v111(v43)]
        if v93 == v30:
            v93 = v43[(v66 + 3) % v111(v43)]
        v31 = v43
        v94 = v125(v12, None)
        v95 = v125(v12, v30)
        v96 = v125(v12, v93)
        for v126, v127 in [('none', v94), ('gold', v95), ('wrong', v96)]:
            v46[v126] += v7(v150(v45, v15, v16, v17, v18, v127, v30, v31, v32))
        v97 = v15.v107(v30).v20[0] if v15.v107(v30).v20 else -1
        v98 = v128(v45, v15, v16, v17, v18, v95)
        v99 = v128(v45, v15, v16, v17, v18, v96)
        v48 += v7(v98 == v97)
        v47 += v7(v99 == v15.v107(v93).v20[0] if v15.v107(v93).v20 else -2)
        v51.v112(v140(v45, v15, v16, v17, v18, v94, v30))
        v49.v112(v140(v45, v15, v16, v17, v18, v95, v30))
        v50.v112(v140(v45, v15, v16, v17, v18, v96, v30))
        v52 += 1
    v52 = v100(1, v52)
    v101, v102, v103 = (v46['none'] / v52, v46['gold'] / v52, v46['wrong'] / v52)
    v53 = v102 - v103
    v54 = v8(v151.v141(v49) - v151.v141(v50))
    v55 = v102 - v101
    v56 = v53 >= 0.08 or v54 >= 0.5
    v57 = v142(v53) < 0.03 and v142(v55) < 0.03
    if v56:
        v104 = 'HEAD_READS_INJECT_YES'
    elif v57:
        v104 = 'HEAD_LEXICAL_PRIOR_ONLY'
    else:
        v104 = 'HEAD_INJECT_PARTIAL'
    v58 = {'stage': '228a', 'overall': v104, 'rank4_gold_target': {'none': v101, 'gold_inject': v102, 'wrong_inject': v103}, 'sensitivity_rank_gold_minus_wrong': v53, 'span_logprob_gold': {'mean_given_none': v8(v151.v141(v51)), 'mean_given_gold_inject': v8(v151.v141(v49)), 'mean_given_wrong_inject': v8(v151.v141(v50)), 'sensitivity_gold_minus_wrong': v54}, 'argmax_follows_inject_rate': {'gold_prompt': v48 / v52, 'wrong_prompt': v47 / v52}, 'interpretation': 'If sensitivity ~0: head ignores inject; use fp-guided decode or head_mem train.' if v57 or not v56 else 'Head partially follows inject; tune format / joint mem templates.', 'n_items': v52, 'timestamp': v155.v152(v156.v153).v129()}
    v1.v105(v143.v130(v58, indent=2), encoding='utf-8')
    v2.v105(f'# Stage 228a counterfactual inject\n\n**{v104}** rank none/g/w={v101:.3f}/{v102:.3f}/{v103:.3f} sens={v53:.3f} lp_sens={v54:.3f}\n', encoding='utf-8')
    v60(v143.v130(v58, indent=2))
    return 0
if v59 == '__main__':
    raise v106(v131())