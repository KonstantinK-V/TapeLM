"""
Stage 224 — far-domain arc_enc shifts + cross-W matrix (Stories vs code vs med).

Tests whether W is domain-specific (registry) or ~canonical (universal unwarp).

  python _stage224_far_shift.py [--smoke]
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
v0 = v13('results')
v1 = v13('data')
v2 = v0 / 'stage224_decision.json'
v3 = v0 / 'stage224_mini.md'
v4 = v1 / '_stage224_code_corpus.txt'
v5 = v13('checkpoints/stage191_p1_curve.pt')
v6 = v13('data/external_tinystories_100k_85.txt')
v7 = v13('data/_wikitext103_train.txt')
v8 = 224
v9 = v62.v14('\\b(patient|patients|clinical|diagnosis|treatment|therapy|disease|symptoms|hospital|physician|medical|cancer|cardiac|infection|chronic|acute)\\b', v62.v15)

def log(v16: v10) -> None:
    v63(v16, flush=True)

def ensure_code_corpus(v17: v126.v64, v18: v12=12000) -> v10:
    if v4.v127() and v4.v186().v128 > 10000:
        return v4.v129(encoding='utf-8')
    v19: v65[v10] = []
    for v20 in v66(v18):
        v67 = f'process_{v17.v187(0, 99999)}'
        v68 = f'arg_{v17.v187(0, 9999)}'
        v69 = f'm{v20 % 200}'
        v19.v130(f'def {v67}({v68}: int) -> int:\n    import numpy as np  # {v69}\n    return np.abs({v68}) + {v17.v187(0, 512)}')
        v19.v130(f'class Handler_{v20 % 500}(object):\n    def __init__(self, {v68}=None):\n        self.{v68} = {v68}')
    v21 = '\n'.v70(v19)
    v4.v71(v21, encoding='utf-8')
    return v21

def ensure_med_corpus(v22: v12=8000) -> v10:
    v23 = v1 / '_stage224_med_corpus.txt'
    if v23.v127() and v23.v186().v128 > 10000:
        return v23.v129(encoding='utf-8')
    v19: v65[v10] = []
    with v7.v131('r', encoding='utf-8', errors='ignore') as v72:
        for v73 in v72:
            v73 = v73.v168()
            if v132(v73) < 48 or not v9.v200(v73):
                continue
            v19.v130(v73)
            if v132(v19) >= v22:
                break
    if v132(v19) < 200:
        raise v133('med corpus too small; check WIKI path')
    v21 = '\n'.v70(v19)
    v23.v71(v21, encoding='utf-8')
    v74(f'med corpus lines={v132(v19)}')
    return v21

def flat_from_domain(v21: v10, v24: v75, v25: v12, v22: v12, v26: v12=32):
    return v134.v76(v21, v24, v25, max_lines=v22, min_line_len=v26)

def recall_k(v27, v28, v29, v30, v31, v17, v32):
    v77, v78 = (0, 0)
    for v79, v80 in v81(v30, v31):
        v82 = v29.v135(f'In the report {v79} was linked to the organization.', exclude=v80)
        if v82 is None:
            continue
        v83 = v32(v27)
        v84 = [v80] + [v31[(v20 + 1) % v132(v31)] for v20 in v66(3)]
        v17.v136(v84)
        v85 = v84.v137(v80)
        v86 = []
        for v87 in v84:
            v138 = [v20 for v20, v184 in v111(v28) if v184 == v87]
            v86.v130(v140((v83[v138] @ v82).v122()) if v138 else -1.0)
        v77 += v12(v201.v188(v86) == v85)
        v78 += 1
    return (v77 / v122(1, v78), v78)

def w_distance(v33: v139.v88, v34: v139.v88) -> v11:
    v35 = v33 - v34
    return {'frobenius': v140(v35.v210(2).v178().v169()), 'cos_flat': v140(v189.v170(v33.v207().v190(0), v34.v207().v190(0)))}

def main() -> v12:
    v36 = v141.v89()
    v36.v90('--smoke', action='store_true')
    v37 = v36.v91()
    v38 = v139.v38('cuda' if v139.v191.v171() else 'cpu')
    v39 = 80 if v37.v92 else v142.v93
    v40 = 100 if v37.v92 else v142.v94
    v41 = 80 if v37.v92 else v142.v95
    v42 = 12 if v37.v92 else 60
    v22 = 400 if v37.v92 else 8000
    v17 = v126.v64(v8)
    v96, v97, v98, v99 = v100()
    v24 = v75.v101(v10(v172.v143))
    v25 = v24.v144(v145) or 0
    v43 = v192.v173(v24, v98, v25, v24.v193()).v102(v38)
    with v7.v131('r', encoding='utf-8', errors='ignore') as v72:
        v103 = v72.v146(2000000)
    v44 = v65(v11.v174((v194 for v194 in v62.v208('[A-Za-z][a-z]{2,}', v103) if v132(v194) <= 14)))[:v41]
    v45 = v175(v99, v24.v193()).v102(v38)
    v45.v104(v139.v176(v5, map_location=v38, weights_only=False)['model'])
    v45.v105()
    v46 = v106(v45, v98, v38)
    v47 = v142.v107(v46, v44)
    with v7.v131('r', encoding='utf-8', errors='ignore') as v72:
        v108 = v65(v11.v174((v203.v202(1) for v203 in v211.v209(v72.v146(4000000)) if v132(v203.v202(1)) >= 5)))
    v30 = v147(v177(v108), v17, v42 + 10)[:v42]
    v31 = v108[:v42]
    v109, v28 = v142.v110(v46, v30, v31, v17)
    v48: v11[v10, v148] = {}
    v49 = {'stories': v6.v129(encoding='utf-8', errors='ignore'), 'code': v149(v126.v64(v8 + 11), n_lines=3000 if v37.v92 else 12000), 'med': v150(max_lines=v22)}
    v50: v11[v10, v151] = {}
    v51: v11[v10, v106] = {}
    v52: v11[v10, v140] = {}
    for v20, (v152, v21) in v111(v49.v153()):
        v74(f'arc shift domain={v152} ...')
        v154, v155 = v156(v21, v24, v25, max_lines=v22)
        v112 = v142.v157(v45, v154, v155, v43, v25, v38, v39, v8 + 10 + v20)
        v113 = v106(v112, v98, v38)
        v114 = v142.v107(v113, v44)
        v52[v152] = v140((v47 * v114).v204(-1).v178())
        v115, v158 = v142.v159(v151(256).v102(v38), v47, v114, v17, v40, v38)
        v50[v152] = v115
        v51[v152] = v113
        v48[v152] = v112
    v53 = v65(v49.v160())

    def tr(v69: v151):
        return lambda v27: v189.v179(v69.v195(v27), dim=-1)
    v54: v11[v10, v11[v10, v140]] = {}
    for v35 in v53:
        v54[v35] = {}
        for v161, v162 in v50.v153():
            v163, v158 = v180(v109, v28, v51[v35], v30, v31, v17, v196(v162))
            v54[v35][v161] = v163
    v55 = {}
    for v35 in v53:
        v116 = v54[v35][v35]
        v117 = [v54[v35][v181] for v181 in v53 if v181 != v35]
        v118 = v122(v117) if v117 else v116
        v119 = v123(v117) if v117 else v116
        v55[v35] = {'matched': v116, 'best_wrong': v118, 'worst_wrong': v119, 'drop_vs_best_wrong': v116 - v118, 'drop_vs_worst_wrong': v116 - v119}
    v56 = {v78: v50[v78].v194.v182.v164() for v78 in v53}
    v57 = {}
    for v20, v120 in v111(v53):
        for v121 in v53[v20 + 1:]:
            v57[f'{v120}_vs_{v121}'] = v183(v56[v120], v56[v121])
    v58 = v122((v55[v35]['drop_vs_best_wrong'] for v35 in v53))
    v59 = v123((v55[v35]['drop_vs_best_wrong'] for v35 in v53))
    v60 = v122((v184['frobenius'] for v184 in v57.v197()))
    if v58 >= 0.2:
        v124 = 'W_REGISTRY_NEEDED'
    elif v58 < 0.05 and v59 >= -0.02:
        v124 = 'CANONICAL_W_CANDIDATE'
    else:
        v124 = 'W_DOMAIN_PARTIAL'
    v23 = {'stage': 224, 'overall': v124, 'gates': {'G_cross_drop_ge_0p20_any': v58 >= 0.2, 'G_cross_drop_lt_0p05_all': v58 < 0.05}, 'mean_cos_shift_per_domain': v52, 'recall_matrix_query_domain__W_adapter': v54, 'cross_drops': v55, 'W_frobenius_cos_pairs': v57, 'summary': {'max_cross_drop_vs_best_wrong_W': v58, 'min_cross_drop_vs_best_wrong_W': v59, 'max_W_pair_frobenius': v60}, 'domains': {'stories': 'TinyStories prose', 'code': v10(v4), 'med': 'Wiki lines filtered medical lexicon'}, 'note': 'Rows=encoder domain for query; cols=W trained after shift on that domain', 'timestamp': v205.v198(v206.v199).v165()}
    v2.v71(v185.v166(v23, indent=2), encoding='utf-8')
    v3.v71(f'# Stage 224 far shift\n\n**{v124}** max_drop={v58:.3f} shifts={v52}\n', encoding='utf-8')
    v63(v185.v166(v23, indent=2))
    return 0
if v61 == '__main__':
    raise v125(v167())