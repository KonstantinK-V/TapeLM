"""
Stage 244 — Forget-A cleanliness: slot delete vs GPT gradient unlearn.

Subset of 205 framing: delete half the facts from TapeLM slots (O(1)); GPT gradient-ascent
unlearn with early stop. Compare target forget + retained collateral + next_tok.

  python _stage244_forget_clean.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import time
from datetime import datetime, timezone
import torch
import _stage24x_lib as L
from _stage196_tapelm import load_gpt
v0 = 244
v1 = v62.v6 / 'stage244_decision.json'
v2 = v62.v6 / 'stage244_mini.md'
v3 = v62.v6 / '_stage244_log.txt'
v4 = 0.25

def recall_subset(v7, v8, v9, v10, v11, v12, v13, v14, v15=None, v16=None, v17=None):
    if v8 == 'tape':
        return v62.v88(v9, v10, v15, v16, v17, v14)
    return v62.v63(v7, v11, v12, v9, v10, v13, v14)

def main() -> v5:
    v18 = v105.v64()
    v18.v65('--smoke', action='store_true')
    v19 = v18.v66()
    v3.v67('', encoding='utf-8')
    v20 = v62.v68(v3)
    v13 = v106.v13('cuda' if v106.v125.v119() else 'cpu')
    v21 = v120('random').v69(v0)
    v106.v70(v0)
    v22 = v71.v71()
    v23 = 16 if v19.v72 else 40
    v24 = 6 if v19.v72 else 16
    v25 = 200 if v19.v72 else 1600
    v26 = 30 if v19.v72 else 60
    v27 = 40 if v19.v72 else 120
    v73, v74, v75 = (8, 64, 0.0003)
    v28 = 5e-05
    v29 = 0.72
    v20(f'Stage244 start {v130.v127(v131.v128).v117()}')
    v76, v76, v76, v76, v11, v76, v12, v77, v78, v15 = v62.v79(v13)
    v76, v80, v76, v81 = v62.v82(v19.v72, 60 if v19.v72 else 400, v21)
    v83, v10 = v62.v84(v23, v80, v21)
    v30 = v83[:v24]
    v31 = v83[v24:]
    v32 = v62.v85(v27)
    v16, v86 = v62.v87(v15, v83)
    v33 = v62.v88(v30, v10, v15, v16, v86, v0)
    v34 = v62.v88(v31, v10, v15, v16, v86, v0)
    v35 = v62.v89(v78, v77, v12, v32, v13)
    v36 = [v90 for v90, v107 in v121(v83) if v107 not in v30]
    v37 = {v107['fid'] for v107 in v30}
    v38 = [v90 for v90, v107 in v121(v83) if v107['fid'] not in v37]
    v39 = v16[v38]
    v40 = [v86[v90] for v90 in v38]
    v41 = v62.v88(v30, v10, v15, v39, v40, v0)
    v42 = v62.v88(v31, v10, v15, v39, v40, v0)
    v43 = v62.v89(v78, v77, v12, v32, v13)
    v20(f'tape delete: tgt {v33:.3f}->{v41:.3f} ret {v34:.3f}->{v42:.3f}')
    v44 = v108.v91(v109(v13))
    v92, v93, v76 = v62.v94(v44, v11, v12, v83, v10, v81, v13, v0, v25, v73, v74, v75, v29, 40 if v19.v72 else 100, v20)
    v45 = v62.v63(v44, v11, v12, v30, v10, v13, v0)
    v46 = v62.v63(v44, v11, v12, v31, v10, v13, v0)
    v47 = v62.v95(v44, v32, v13)
    v48 = [[v90 for v90 in v11.v129(v107['sent']).v122 if v90 != v12] for v107 in v30]
    v49 = v106.v110.v96(v44.v111(), lr=v28)
    v50 = v120('random').v69(v0 + 13)
    v44.v97()
    v51 = 0
    for v52 in v98(1, v26 + 1):
        v99 = v62.v112(v50, v48, [], v73, v74, v13, mix_real=False)
        v100 = -v44(input_ids=v99, labels=v99).v100
        v49.v113(set_to_none=True)
        v100.v114()
        v106.v126.v123.v115(v44.v111(), 1.0)
        v49.v52()
        v51 = v52
        if v52 % 10 == 0:
            v44.v101()
            v116 = v62.v63(v44, v11, v12, v30, v10, v13, v0)
            v20(f'  gpt unlearn {v52}: tgt={v116:.3f}')
            if v116 <= v4 + 0.05:
                break
            v44.v97()
    v44.v101()
    v53 = v62.v63(v44, v11, v12, v30, v10, v13, v0)
    v54 = v62.v63(v44, v11, v12, v31, v10, v13, v0)
    v55 = v62.v95(v44, v32, v13)
    v20(f'gpt unlearn ({v51}): tgt {v45:.3f}->{v53:.3f} ret {v46:.3f}->{v54:.3f}')
    v56 = v41 <= v4 + 0.05
    v57 = v124(v42 - v34) <= 0.02 and v124(v43 - v35) < 1e-09
    v58 = v124(v54 - v46) > 0.02 or v124(v55 - v47) > 0.02
    v59 = v53 <= v45 - 0.15 or v53 <= v4 + 0.15
    if v56 and v57 and v58:
        v102 = 'FORGET_CLEAN_OK'
    elif v56 and v57:
        v102 = 'FORGET_CLEAN_PARTIAL'
    else:
        v102 = 'FORGET_CLEAN_NO'
    v60 = {'stage': 244, 'overall': v102, 'gates': {'G_tape_forget_to_chance': v56, 'G_tape_no_collateral': v57, 'G_gpt_shows_collateral': v58, 'G_gpt_forgot_some': v59}, 'tape': {'tgt_before': v33, 'tgt_after': v41, 'ret_before': v34, 'ret_after': v42, 'next_tok_before': v35, 'next_tok_after': v43}, 'gpt': {'tgt_before': v45, 'tgt_after': v53, 'ret_before': v46, 'ret_after': v54, 'next_tok_before': v47, 'next_tok_after': v55, 'unlearn_steps': v51, 'memorize_steps': v92}, 'note': 'Capability vs parametric GPT; architectural vs GPT+RAG index delete.', 'timestamp': v130.v127(v131.v128).v117(), 'wall_s': v71.v71() - v22}
    v62.v103(v1, v2, v60, 'Stage 244 forget cleanliness')
    v20(v102)
    return 0
if v61 == '__main__':
    raise v104(v118())