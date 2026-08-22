"""
Stage 240 — CF A→B: TapeLM vs GPT+RAG (frozen embedding index) vs parametric GPT.

Same A facts / code-B adapt as 239. Extra arm: freeze GPT embedding keys after memorize A;
after code CE, query with post-B GPT against frozen index (fair RAG).

Expected: RAG keeps A ≈ TapeLM (architectural). Surprise: query drift breaks RAG.

  python _stage240_cf_vs_rag.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage194_fp_fact_memory import FpBank
from _stage196_tapelm import load_gpt
from _tapelm_ext import DomainAdapter
v0 = 240
v1 = v42.v5 / 'stage240_decision.json'
v2 = v42.v5 / 'stage240_mini.md'
v3 = v42.v5 / '_stage240_log.txt'

def main() -> v4:
    v6 = v105.v43()
    v6.v44('--smoke', action='store_true')
    v7 = v6.v45()
    v3.v46('', encoding='utf-8')
    v8 = v42.v47(v3)
    v9 = v106.v9('cuda' if v106.v119.v117() else 'cpu')
    v10 = v118('random').v48(v0)
    v106.v49(v0)
    v11 = v50.v50()
    v12 = 12 if v7.v51 else 40
    v13 = 240 if v7.v51 else 2400
    v14 = 400 if v7.v51 else 1600
    v15 = 60 if v7.v51 else v107.v52
    v16 = 80 if v7.v51 else v107.v53
    v17 = 60 if v7.v51 else 400
    v18 = 40 if v7.v51 else 120
    v54, v55, v56, v57 = (8, 64, 0.0003, 0.0005)
    v19 = 0.72
    v8(f'Stage240 start {v124.v122(v125.v123).v114()} device={v9}')
    v58, v58, v59, v60, v61, v62, v63, v64, v65, v66 = v42.v67(v9)
    v58, v68, v69, v70 = v42.v71(v7.v51, v17, v10)
    v72, v73 = v42.v74(v12, v68, v10)
    v20 = v42.v75(v18)
    v76, v77 = v42.v78(v66, v72)
    v21 = v42.v79(v72, v73, v66, v76, v77, v0)
    v8(f'tape A write recall={v21:.3f}')
    v22 = v108.v80(v109(v9))
    v81, v82, v58 = v42.v83(v22, v61, v63, v72, v73, v70, v9, v0, v13, v54, v55, v56, v19, 40 if v7.v51 else 100, v8)
    v23 = v42.v84(v22, v61, v63, v72, v73, v9, v0)
    v85, v86 = v42.v87(v22, v61, v63, v9, v72)
    v24 = v42.v88(v22, v61, v63, v9, v72, v73, v85, v86, v0)
    v8(f'gpt memorize ({v81}): param={v23:.3f} rag={v24:.3f}')
    v25 = v110.v89(v118('random').v48(v0 + 1), v7.v51)
    v90, v91 = v111.v92(v25, v61, v63, max_lines=300 if v7.v51 else 8000, min_line_len=20)
    v26 = [v93 for v93 in v61.v120(v25[:200000]).v112 if v93 != v63]
    v27 = v107.v94(v66, v69)
    v28 = v107.v95(v65, v90, v91, v64, v63, v9, v15, v0 + 7)
    v29 = v96(v28, v59, v9)
    v97, v98 = v107.v99(v121(256).v113(v9), v107.v94(v29, v69), v27, v10, v16, v9)
    v30 = v42.v79(v72, v73, v29, v76, v77, v0, W_bwd=v97)
    v31 = v42.v79(v72, v73, v29, v76, v77, v0, W_bwd=None)
    v42.v100(v22, v26, v54, v55, v57, v14, v9, v0, v8)
    v32 = v42.v84(v22, v61, v63, v72, v73, v9, v0)
    v33 = v42.v88(v22, v61, v63, v9, v72, v73, v85, v86, v0)
    v34 = v42.v101(v22, v20, v9)
    v8(f'after B: tape_W={v30:.3f} rag={v33:.3f} param={v32:.3f} ({v50.v50() - v11:.0f}s)')
    v35 = v21 >= 0.7 and v23 >= 0.7 and (v24 >= 0.7)
    v36 = v30 >= 0.8
    v37 = v33 >= 0.8
    v38 = v23 - v32 >= 0.15
    v39 = v33 < v30 - 0.15
    if v35 and v36 and v38 and v39 and (not v37):
        v102 = 'CF_VS_RAG_SURPRISE'
    elif v35 and v36 and v37 and v38:
        v102 = 'CF_VS_RAG_ARCHITECTURAL'
    elif v35 and v36 and (v37 or v38):
        v102 = 'CF_VS_RAG_PARTIAL'
    else:
        v102 = 'CF_VS_RAG_NO'
    v40 = {'stage': 240, 'overall': v102, 'gates': {'G_memorize_ge_0p70': v35, 'G_tape_keep_ge_0p80': v36, 'G_rag_keep_ge_0p80': v37, 'G_param_drop_ge_0p15': v38, 'G_rag_surprise_gap': v39}, 'tape': {'A0': v21, 'A1_W': v30, 'A1_raw': v31, 'W_align': v98}, 'rag': {'A0': v24, 'A1': v33, 'drop': v24 - v33}, 'param_gpt': {'A0': v23, 'A1': v32, 'drop': v23 - v32, 'next_tok_after_B': v34}, 'note': 'Frozen GPT emb index after A; queries with post-B GPT. Architectural if RAG~TapeLM.', 'timestamp': v124.v122(v125.v123).v114()}
    v42.v103(v1, v2, v40, 'Stage 240 CF vs GPT+RAG')
    v8(v115(v40['overall']))
    return 0
if v41 == '__main__':
    raise v104(v116())