"""
Stage 226b — Diagnose 226 NO: reconcile recall vs 227 + utilization inject forms.

Hypothesis check:
  H1: 0.60 vs 0.95 is exam/seed variance (same factual query form as 227, not code-gen query).
  H2: gold_inject == no_inject because head_code ignores prose; code-native comment may help (path C).

Does NOT do joint SFT (path B) — keeps zero-train substrate narrative.

  python _stage226b_joint_diag.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
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
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
v0 = v8('results')
v1 = v0 / 'stage226b_decision.json'
v2 = v0 / 'stage226b_mini.md'
v3 = v8('checkpoints/stage191_p1_curve.pt')
v4 = v8('data/_wikitext103_train.txt')
v5 = 2262

def log(v9: v57) -> None:
    v58(v9, flush=True)

def w_apply(v10):
    return lambda v112: v143.v113(v10.v144(v112), dim=-1)

def recall_qmap(v11, v12, v13, v14, v15, v16, v17):
    v18 = v19 = 0
    for v59, v60 in v61(v15, v16):
        v62 = v13.v114(f'In the report {v59} was linked to the organization.', exclude=v60)
        if v62 is None:
            continue
        v63 = v159(v14)(v62.v160(0))[0]
        v64 = []
        v65 = [v60] + [v16[(v145 + 1) % v149(v16)] for v145 in v173(3)]
        v17.v115(v65)
        v66 = v65.v116(v60)
        for v67 in v65:
            v117 = [v145 for v145, v174 in v175(v12) if v174 == v67]
            v64.v137(v6((v11[v117] @ v63).v109()) if v117 else -1.0)
        v18 += v7(v176.v161(v64) == v66)
        v19 += 1
    return v18 / v109(1, v19)

@v75.v26()
def rank4(v20, v21, v22, v23, v24, v25: v118[v146[v57, v57, v118[v57]]], v17) -> v6:
    """(prompt_text, gold, distractor_pool) → 4-way rank of first BPE of gold."""
    v18 = v19 = 0
    for v68, v60, v69 in v25:
        v70 = v21.v147(v68).v70
        if not v70:
            continue
        v71 = v75.v119([v70], dtype=v75.v148, device=v24)
        v72 = v71 == v22
        v120, v121, v121 = v20.v122(v23[v71], v72, ids=v71)
        v73 = v120[0, -1]
        v65 = [v60] + [v162 for v162 in v69 if v162 != v60][:3]
        while v149(v65) < 4:
            v65.v137(v69[v149(v65) % v149(v69)])
        v17.v115(v65)
        v74 = []
        for v67 in v65:
            v123 = v21.v147(v67).v70
            v74.v137(v6(v73[v123[0]]) if v123 else -1000000000.0)
        v18 += v7(v65[v7(v176.v161(v74))] == v60)
        v19 += 1
    return v18 / v109(1, v19)

def main() -> v7:
    v27 = v124.v76()
    v27.v77('--smoke', action='store_true')
    v28 = v27.v78()
    v24 = v75.v24('cuda' if v75.v163.v150() else 'cpu')
    v29 = 80 if v28.v79 else v125.v80
    v30 = 80 if v28.v79 else 600
    v31 = 100 if v28.v79 else v125.v81
    v32 = 80 if v28.v79 else v125.v82
    v33 = 12 if v28.v79 else 60
    v34 = 400 if v28.v79 else 8000
    v83, v84, v85, v86 = v87()
    v21 = v126.v88(v57(v151.v127))
    v22 = v21.v128(v129) or 0
    v23 = v164.v152(v21, v85, v22, v21.v165()).v89(v24)
    with v4.v130('r', encoding='utf-8', errors='ignore') as v90:
        v68 = v90.v131(2000000)
    v35 = v118(v166.v153((v167 for v167 in v182.v180('[A-Za-z][a-z]{2,}', v68) if v149(v167) <= 14)))[:v32]
    v36 = v154(v86, v21.v165()).v89(v24)
    v36.v91(v75.v155(v3, map_location=v24, weights_only=False)['model'])
    v36.v92()
    v37 = v93(v36, v85, v24)
    v38 = v125.v94(v37, v35)
    with v4.v130('r', encoding='utf-8', errors='ignore') as v90:
        v95 = v118(v166.v153((v9.v177(1) for v9 in v183.v181(v90.v131(4000000)) if v149(v9.v177(1)) >= 5)))
    v39 = v132.v96(227)
    v15 = v133(v156(v95), v39, v33 + 10)[:v33]
    v16 = v95[:v33]
    v97, v12 = v125.v98(v37, v15, v16, v39)
    v40 = v134.v99(v132.v96(227 + 1), v28.v79)
    v100, v101 = v135.v102(v40, v21, v22, max_lines=v34, min_line_len=20)
    v41 = v125.v103(v36, v100, v101, v23, v22, v24, v29, 227 + 3)
    v42 = v93(v41, v85, v24)
    v43 = v125.v94(v42, v35)
    v44 = v6((v38 * v43).v168(-1).v136())
    v14, v104 = v125.v105(v169(256).v89(v24), v43, v38, v39, v31, v24)
    v45 = v106(v97, v12, v42, v14, v15, v16, v39)
    v18 = v19 = 0
    for v59, v60 in v61(v15, v16):
        v62 = v42.v114(f'In the report {v59} was linked to the organization.', exclude=v60)
        if v62 is None:
            continue
        v65 = [v60] + [v16[(v145 + 1) % v149(v16)] for v145 in v173(3)]
        v39.v115(v65)
        v66 = v65.v116(v60)
        v64 = [v6((v97[[v145 for v145, v174 in v175(v12) if v174 == v67]] @ v62).v109()) if v170((v174 == v67 for v174 in v12)) else -1.0 for v67 in v65]
        v18 += v7(v176.v161(v64) == v66)
        v19 += 1
    v46 = v18 / v109(1, v19)
    v47 = v134.v107(v36, v100, v101, v23, v22, v24, v30, v5 + 3)
    v17 = v132.v96(v5)
    v48 = []
    for v59, v60 in v61(v15, v16):
        v69 = v16
        v48.v137((v59, v60, v69))

    def pack(v108: v57):
        v55 = []
        for v59, v60, v69 in v48:
            if v108 == 'none':
                v68 = f'# TODO\ndef org_of_{v59}():\n    return '
            elif v108 == 'prose':
                v68 = f'# Note: {v59} was director of {v60}.\ndef org_of_{v59}():\n    return '
            elif v108 == 'code_comment':
                v68 = f'# org[{v59}] = {v60}\ndef org_of_{v59}():\n    return '
            elif v108 == 'assignment':
                v68 = f'ORG = {v60!r}\ndef org_of_{v59}():\n    return '
            else:
                raise v184(v108)
            v55.v137((v68, v60, v69))
        return v55
    v49 = {'none': v138(v47, v21, v22, v23, v24, v157('none'), v17), 'prose_inject': v138(v47, v21, v22, v23, v24, v157('prose'), v17), 'code_comment_inject': v138(v47, v21, v22, v23, v24, v157('code_comment'), v17), 'assignment_inject': v138(v47, v21, v22, v23, v24, v157('assignment'), v17)}
    v50 = {'none': v138(v36, v21, v22, v23, v24, v157('none'), v17), 'assignment_inject': v138(v36, v21, v22, v23, v24, v157('assignment'), v17)}
    v51 = v45 >= 0.85
    v52 = v109(v49.v139())
    v53 = v52 >= v49['none'] + 0.08
    v54 = 'RETRIEVAL_OK_UTIL_BOUNDARY'
    if v51 and v53:
        v54 = 'JOINT_DIAG_UTIL_WIN'
    elif v51:
        v54 = 'RETRIEVAL_OK_UTIL_BOUNDARY'
    v55 = {'stage': '226b', 'overall': v54, 'H1_recall_vs_227': {'protocol': 'SEED227 facts + factual query + code shift + qmap', 'mean_cos_code': v44, 'align_W_bwd': v104, 'recall_qmap': v45, 'recall_no_W': v46, 'reconciles_with_227_cross_0p95': v51, 'note': "226's 0.60 used SEED226 / n=40 — not code-gen query; same factual template"}, 'H2_utilization_inject_forms': {'head_code': v49, 'head_P1_ref': v50, 'best_minus_none': v52 - v49['none'], 'path_C_code_native_helps': v49['code_comment_inject'] >= v49['prose_inject'] + 0.03 or v49['assignment_inject'] >= v49['prose_inject'] + 0.03}, 'contract_note': 'One canonical bank + W@read (227). Utilization is head/policy layer; path A boundary unless inject form or joint train (B) helps.', 'timestamp': v178.v171(v179.v172).v140()}
    v1.v110(v158.v141(v55, indent=2), encoding='utf-8')
    v2.v110(f"# Stage 226b joint diag\n\n**{v54}** qmap={v45:.3f} util none/prose/code/assign={v49['none']:.3f}/{v49['prose_inject']:.3f}/{v49['code_comment_inject']:.3f}/{v49['assignment_inject']:.3f}\n", encoding='utf-8')
    v58(v158.v141(v55, indent=2))
    return 0
if v56 == '__main__':
    raise v111(v142())