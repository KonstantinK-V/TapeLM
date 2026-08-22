"""
Stage 225 — family fork for W + multi-head generation with frozen arc_enc.

Tests domain-bundle modularity on one shared substrate:
  shared: frozen arc_enc → one fp geometry
  bundle: {W_family, head_family}  (slots versioned later)

A) W_prose reuse on legal-ish wiki lines vs freshly learned W_legal (fork gate drop>=0.05).
B) Train head_prose / head_code with arc_enc frozen; matched vs cross next_tok on domain windows;
   fp drift must stay ~0 (shared map).

  python _stage225_family_fork.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
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
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, WFamilyPolicy
v0 = v13('results')
v1 = v13('data')
v2 = v0 / 'stage225_decision.json'
v3 = v0 / 'stage225_mini.md'
v4 = v13('checkpoints/stage191_p1_curve.pt')
v5 = v13('data/external_tinystories_100k_85.txt')
v6 = v13('data/_wikitext103_train.txt')
v7 = v13('data/_stage224_code_corpus.txt')
v8 = 225
v9 = v87.v14('\\b(court|law|legal|plaintiff|defendant|statute|contract|jurisdiction|legislation|attorney|verdict|constitution|amendment)\\b', v87.v15)

def log(v16: v10) -> None:
    v88(v16, flush=True)

def ensure_legal_corpus(v17: v12) -> v10:
    v18 = v1 / '_stage225_legal_corpus.txt'
    if v18.v165() and v18.v227().v166 > 5000:
        return v18.v94(encoding='utf-8')
    v19: v89[v10] = []
    with v6.v167('r', encoding='utf-8', errors='ignore') as v90:
        for v91 in v90:
            v91 = v91.v205()
            if v168(v91) < 48 or not v9.v240(v91):
                continue
            v19.v206(v91)
            if v168(v19) >= v17:
                break
    if v168(v19) < 80:
        raise v169('legal corpus too small')
    v18.v92('\n'.v170(v19), encoding='utf-8')
    v93(f'legal corpus lines={v168(v19)}')
    return v18.v94(encoding='utf-8')

def ensure_code(v20: v171.v95, v21: v96) -> v10:
    if v7.v165() and v7.v227().v166 > 10000:
        return v7.v94(encoding='utf-8')
    import _stage224_far_shift as s224
    return v172.v97(v20, n_lines=2000 if v21 else 12000)

@v104.v32()
def window_next_tok_acc(v22, v23, v24, v25, v26, v27, v20, v28=20) -> v11:
    v29 = v30 = 0
    for v31 in v98(v28):
        v99 = v228(v23, v24, v229, v20, v26).v134(v27)
        v100 = v99 == v26
        v173, v31, v31 = v22.v174(v25[v99], v100, ids=v99)
        v101 = v173[:, :-1].v175(-1)
        v102 = v99[:, 1:]
        v103 = ~v100[:, :-1] & ~v100[:, 1:]
        v29 += v12((v101[v103] == v102[v103]).v207())
        v30 += v12(v103.v207())
    return v29 / v176(1, v30)

def train_upper(v22, v23, v24, v25, v26, v27, v33, v34):
    from _stage191_night import LR
    v16 = v177.v105(v22)
    v178.v106(v16, 'upper')
    v35 = [v107 for v107 in v16.v208() if v107.v179]
    v36 = v104.v180.v108(v35, lr=v209 * 0.5)
    v37 = v171.v95(v34)
    for v38 in v98(1, v33 + 1):
        for v109 in v36.v110:
            v109['lr'] = v210(v38, v33)
        v99 = v228(v23, v24, v229, v37, v26).v134(v27)
        v100 = v99 == v26
        v173, v31, v181 = v16.v174(v25[v99], v100, ids=v99)
        v102 = v99[:, 1:]
        v103 = ~v100[:, :-1] & ~v100[:, 1:]
        v111 = v211.v182(v173[:, :-1][v103], v102[v103])
        v112 = v111 + v212 * v181[~v100].v199()
        v36.v183(set_to_none=True)
        v112.v184()
        v104.v230.v213.v185(v35, 1.0)
        v36.v38()
    v16.v113()
    v16.v186.v113()
    return v16

def recall_k(v39, v40, v41, v42, v43, v20, v44):
    v29, v114 = (0, 0)
    for v115, v116 in v117(v42, v43):
        v118 = v41.v187(f'In the report {v115} was linked to the organization.', exclude=v116)
        if v118 is None:
            continue
        v119 = v44(v39)
        v120 = [v116] + [v43[(v221 + 1) % v168(v43)] for v221 in v98(3)]
        v20.v188(v120)
        v109 = v120.v189(v116)
        v121 = [v11((v119[[v221 for v221, v245 in v222(v40) if v245 == v214]] @ v118).v176()) if v231((v245 == v214 for v245 in v40)) else -1.0 for v214 in v120]
        v29 += v12(v241.v175(v121) == v109)
        v114 += 1
    return v29 / v176(1, v114)

def main() -> v12:
    v45 = v190.v122()
    v45.v123('--smoke', action='store_true')
    v46 = v45.v124()
    v27 = v104.v27('cuda' if v104.v232.v215() else 'cpu')
    v47 = 80 if v46.v21 else v191.v125
    v48 = 80 if v46.v21 else 600
    v49 = 100 if v46.v21 else v191.v126
    v50 = 80 if v46.v21 else v191.v127
    v51 = 12 if v46.v21 else 60
    v17 = 400 if v46.v21 else 8000
    v20 = v171.v95(v8)
    v128, v129, v130, v131 = v132()
    v52 = v192.v133(v10(v216.v193))
    v26 = v52.v194(v195) or 0
    v25 = v233.v217(v52, v130, v26, v52.v234()).v134(v27)
    with v6.v167('r', encoding='utf-8', errors='ignore') as v90:
        v135 = v90.v196(2000000)
    v53 = v89(v235.v218((v139 for v139 in v87.v246('[A-Za-z][a-z]{2,}', v135) if v168(v139) <= 14)))[:v50]
    v54 = v219(v131, v52.v234()).v134(v27)
    v54.v136(v104.v220(v4, map_location=v27, weights_only=False)['model'])
    v54.v113()
    v55 = v137(v54, v130, v27)
    v56 = v191.v138(v55, v53)
    v57 = {v139: v56[v221].v197() for v221, v139 in v222(v53[:v247(32, v168(v53))])}
    with v6.v167('r', encoding='utf-8', errors='ignore') as v90:
        v140 = v89(v235.v218((v16.v242(1) for v16 in v249.v248(v90.v196(4000000)) if v168(v16.v242(1)) >= 5)))
    v42 = v198(v223(v140), v20, v51 + 10)[:v51]
    v43 = v140[:v51]
    v141, v40 = v191.v142(v55, v42, v43, v20)
    v93('A: W_prose vs legal fork …')
    v58 = v5.v94(encoding='utf-8', errors='ignore')
    v143, v144 = v178.v145(v58, v52, v26, max_lines=v17)
    v59 = v191.v146(v54, v143, v144, v25, v26, v27, v47, v8 + 1)
    v60 = v191.v138(v137(v59, v130, v27), v53)
    v61 = v11((v56 * v60).v207(-1).v199())
    v147, v31 = v191.v148(v236(256).v134(v27), v56, v60, v20, v49, v27)
    v62 = v149(v17)
    v150, v151 = v178.v145(v62, v52, v26, max_lines=v17, min_line_len=40)
    v63 = v191.v146(v54, v150, v151, v25, v26, v27, v47, v8 + 2)
    v64 = v137(v63, v130, v27)
    v65 = v191.v138(v64, v53)
    v66 = v11((v56 * v65).v207(-1).v199())
    v152, v31 = v191.v148(v236(256).v134(v27), v56, v65, v20, v49, v27)

    def tr(v153):
        return lambda v39: v211.v224(v153.v237(v39), dim=-1)
    v67 = v154(v141, v40, v64, v42, v43, v20, v200(v147))
    v68 = v154(v141, v40, v64, v42, v43, v20, v200(v152))
    v69 = v156.v155(v68, v67, drop_tol=0.05)
    v70 = v156(registry={'prose': v147}, cos_identity=0.85, cos_family_floor=0.65)
    v71 = v70.v157(v66, 'prose')
    v93('B: multi-head freeze arc_enc …')
    v72 = v158(v171.v95(v8 + 3), v46.v21)
    v159, v160 = v178.v145(v72, v52, v26, max_lines=v17, min_line_len=20)
    v73 = v161(v54, v143, v144, v25, v26, v27, v48, v8 + 4)
    v74 = v161(v54, v159, v160, v25, v26, v27, v48, v8 + 5)
    v75 = v191.v138(v137(v73, v130, v27), v89(v57.v225()))
    v76 = v191.v138(v137(v74, v130, v27), v89(v57.v225()))
    v77 = v104.v162([v57[v139] for v139 in v57.v225()])
    v78 = v11(1.0 - (v75 * v77).v207(-1).v199())
    v79 = v11(1.0 - (v76 * v77).v207(-1).v199())
    v80 = v171.v95(v8 + 9)
    v81 = 8 if v46.v21 else 24
    v82 = {'stories_with_head_prose': v201(v73, v143, v144, v25, v26, v27, v80, v81), 'stories_with_head_code': v201(v74, v143, v144, v25, v26, v27, v80, v81), 'code_with_head_code': v201(v74, v159, v160, v25, v26, v27, v80, v81), 'code_with_head_prose': v201(v73, v159, v160, v25, v26, v27, v80, v81), 'baseline_stories_P1': v201(v54, v143, v144, v25, v26, v27, v80, v81), 'baseline_code_P1': v201(v54, v159, v160, v25, v26, v27, v80, v81)}
    v82['cross_drop_stories'] = v82['stories_with_head_prose'] - v82['stories_with_head_code']
    v82['cross_drop_code'] = v82['code_with_head_code'] - v82['code_with_head_prose']
    v83 = v78 < 1e-05 and v79 < 1e-05
    v84 = v82['cross_drop_stories'] >= 0.02 or v82['cross_drop_code'] >= 0.02
    if v83 and v84:
        v163 = 'DOMAIN_BUNDLE_OK'
    elif v83:
        v163 = 'DOMAIN_BUNDLE_PARTIAL'
    else:
        v163 = 'DOMAIN_BUNDLE_NO'
    v85 = {'stage': 225, 'overall': v163, 'architecture': {'shared': 'frozen arc_enc → one fp R^d', 'bundle': '{W_family, head_family, slots_family*}', 'note': '*slots versioning deferred; A-era bank used for W tests'}, 'A_W_family_fork': {'cos_stories_shift': v61, 'cos_legal_shift': v66, 'recall_legal_query_W_prose_REUSE': v67, 'recall_legal_query_W_legal_MATCHED': v68, 'drop_matched_minus_reuse': v68 - v67, 'fork_W_family': v69, 'policy_decide_legal': v71}, 'B_multi_head_frozen_arc': {'fp_drift_head_prose': v78, 'fp_drift_head_code': v79, 'generation': v82, 'gates': {'G_fp_shared': v83, 'G_head_specializes': v84}}, 'timestamp': v243.v238(v244.v239).v202()}
    v2.v92(v226.v203(v85, indent=2), encoding='utf-8')
    v3.v92(f"# Stage 225 domain bundle\n\n**{v163}** fork={v69} reuse={v67:.3f} matchedW={v68:.3f} fp_drift={v78:.2e}/{v79:.2e} gen_cross_stories={v82['cross_drop_stories']:.3f} code={v82['cross_drop_code']:.3f}\n", encoding='utf-8')
    v88(v226.v203(v85, indent=2))
    return 0
if v86 == '__main__':
    raise v164(v204())