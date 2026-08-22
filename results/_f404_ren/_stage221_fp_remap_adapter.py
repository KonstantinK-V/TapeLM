"""
Stage 221 — fp-remap adapter: arc_enc shift + tiny W on core vocab.

Protocol:
  1. fp_old = frozen P1 arc_enc on core vocab + old fact bank keys.
  2. Finetune arc_enc only on domain B (control shift) -> fp_new.
  3. Train W (d×d, optional bottleneck) so normalize(W @ fp_old) ≈ fp_new on core words.
  4. Gates:
     G_align  mean cos(W fp_old, fp_new) >= 0.85 on core vocab
     G_recall recall old facts with W @ key_old >= 0.80 * oracle reindex recall

  python _stage221_fp_remap_adapter.py [--smoke]
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
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage213_arc_enc_freeze_finetune as s213
from _stage191_night import LR, MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import BottleneckRemap, DomainAdapter
v0 = v14('results')
v1 = v14('checkpoints/stage191_p1_curve.pt')
v2 = v14('data/external_tinystories_100k_85.txt')
v3 = v14('data/_wikitext103_train.txt')
v4 = v0 / 'stage221_decision.json'
v5 = v0 / 'stage221_mini.md'
v6 = 221
v7 = 800
v8 = 800
v9 = 1200
v10 = 32

def log(v15: v68) -> None:
    v69(v15, flush=True)

@v19.v18()
def fp_matrix(v16: v70, v17: v145[v68]) -> v19.v11:
    return v16.v71(v17)

def build_fact_bank(v16: v70, v20, v21, v22):
    v72, v54 = ([], [])
    for v73, v74 in v75(v20, v21):
        v76 = f'Official records state {v73} was director of {v74} in 1987 .'
        v77 = v16.v71([v73])[0]
        v78 = v16.v146(v76, exclude=v74)
        if v78 is None:
            continue
        v72.v147(v190.v180(v77 + v78, dim=-1))
        v54.v147(v74)
    return (v19.v148(v72, 0), v54)

def recall_at(v23: v19.v11, v24: v145[v68], v16: v70, v20, v21, v22, v25=None):
    v79, v33 = (0, 0)
    for v73, v80 in v75(v20, v21):
        v81 = v16.v146(f'In the report {v73} was linked to the organization.', exclude=v80)
        if v81 is None:
            continue
        if v25 is not None:
            v149 = v25(v23)
            v150 = v25(v81.v213(0))[0]
        else:
            v149, v150 = (v23, v81)
        v82 = [v80] + [v21[(v184 + 1) % v88(v21)] for v184 in v89(3)]
        v22.v151(v82)
        v83 = v82.v152(v80)
        v84 = []
        for v78 in v82:
            v153 = [v184 for v184, v74 in v214(v24) if v74 == v78]
            v84.v147(v130((v149[v153] @ v150).v142()) if v153 else -1.0)
        v79 += v13(v215.v201(v84) == v83)
        v33 += 1
    return (v79 / v142(1, v33), v33)

def train_remap(v26: v154.v85, v27, v28, v22, v29: v13, v30, v31: v86=True):
    v32 = v19.v155.v87(v26.v156(), lr=0.002)
    v33 = v88(v27)
    for v34 in v89(1, v29 + 1):
        v90 = v22.v157(v89(v33), v185(64, v33))
        v158, v159 = (v27[v90], v28[v90])
        v91 = v26(v158)
        v92 = (1.0 - (v91 * v159).v209(-1)).v160()
        if v31 and v186(v26, v187):
            v92 = v92 + 0.02 * (v26.v208.v224 @ v26.v208.v224.v225 - v19.v226(256, device=v30)).v221(2).v160()
        v32.v161(set_to_none=True)
        v92.v162()
        v32.v34()
    with v19.v18():
        v93 = v130((v26(v27) * v28).v209(-1).v160())
    return (v26, v93)

def finetune_arc_enc(v35: v12, v36, v37, v38, v39: v13, v30: v19.v30, v29: v13, v40: v13) -> v12:
    """In-place arc_enc-only finetune (domain shift control)."""
    v15 = v163.v94(v35)
    v164.v95(v15, 'arc_enc')
    v41 = [v96 for v96 in v15.v156() if v96.v165]
    v32 = v19.v155.v87(v41, lr=v188 * 0.5)
    v42 = v166.v97(v40)
    for v34 in v89(1, v29 + 1):
        for v83 in v32.v98:
            v83['lr'] = v189(v34, v29)
        v99 = v202(v36, v37, v203, v42, v39).v113(v30)
        v100 = v99 == v39
        v167, v122, v168 = v15.v169(v38[v99], v100, ids=v99)
        v101 = v99[:, 1:]
        v102 = ~v100[:, :-1] & ~v100[:, 1:]
        v103 = v190.v170(v167[:, :-1][v102], v101[v102])
        v92 = v103 + v191 * v168[~v100].v160()
        v32.v161(set_to_none=True)
        v92.v162()
        v19.v154.v192.v171(v41, 1.0)
        v32.v34()
    v15.v104()
    return v15

def main() -> v13:
    v43 = v172.v105()
    v43.v106('--smoke', action='store_true')
    v44 = v43.v107()
    v30 = v19.v30('cuda' if v19.v204.v193() else 'cpu')
    v22 = v166.v97(v6)
    v45 = 80 if v44.v108 else v8
    v46 = 100 if v44.v108 else v9
    v47 = 80 if v44.v108 else v7
    v48 = 15 if v44.v108 else 60
    v36, v37, v109, v110 = v111()
    v49 = v173.v112(v68(v194.v174))
    v39 = v49.v175(v176) or 0
    import _stage185_tape_read as s185
    v38 = v205.v195(v49, v109, v39, v49.v206()).v113(v30)
    with v3.v177('r', encoding='utf-8', errors='ignore') as v114:
        v115 = v114.v178(2000000)
    v50 = v145(v207.v196((v208 for v208 in v222.v219('[A-Za-z][a-z]{2,}', v115) if v88(v208) <= 14)))[:v47]
    v51 = v12(v110, v49.v206()).v113(v30)
    v51.v116(v19.v197(v1, map_location=v30, weights_only=False)['model'])
    v51.v104()
    v52 = v70(v51, v109, v30)
    v27 = v117(v52, v50)
    with v3.v177('r', encoding='utf-8', errors='ignore') as v114:
        v118 = v145(v207.v196((v15.v216(1) for v15 in v223.v220(v114.v178(4000000)) if v88(v15.v216(1)) >= 5)))
    v53 = v179(v198(v118), v22, v48 + 10)[:v48]
    v54 = v118[:v48]
    v119, v24 = v120(v52, v53, v54, v22)
    v121, v122 = v123(v119, v24, v52, v53, v54, v22, None)
    v124(f'baseline recall (old fp, old bank) acc={v121:.3f}')
    v55 = v2.v125(encoding='utf-8', errors='ignore')
    v126, v127 = v164.v128(v55, v49, v39, max_lines=500 if v44.v108 else 8000)
    v56 = v129(v51, v126, v127, v38, v39, v30, v45, v6 + 1)
    v57 = v70(v56, v109, v30)
    v28 = v117(v57, v50)
    v58 = v130((v27 * v28).v209(-1).v160())
    v124(f'after arc_enc shift: mean cos(fp_old,fp_new) on core = {v58:.3f}')
    v131, v122 = v120(v57, v53, v54, v22)
    v132, v122 = v123(v131, v24, v57, v53, v54, v22, None)
    v124(f'oracle reindex (new bank new fp) acc={v132:.3f}')
    v133, v134 = v135(v187(256).v113(v30), v27, v28, v22, v46, v30)

    def transform_full(v23):
        return v190.v180(v133.v199(v23), dim=-1)
    v136, v122 = v123(v119, v24, v52, v53, v54, v22, v137)
    v124(f'full 256x256: align={v134:.3f} recall={v136:.3f}')
    v59 = 16 if v44.v108 else v10
    v138, v139 = v135(v210(256, v59).v113(v30), v27, v28, v22, v46, v30, orth=False)

    def transform_bot(v23):
        return v190.v180(v138.v199(v23), dim=-1)
    v140, v122 = v123(v119, v24, v52, v53, v54, v22, v141)
    v124(f'bottleneck d-{v59}-d: align={v139:.3f} recall={v140:.3f} params={v59 * 256 * 2}')
    v60 = v142(v136, v140)
    v61 = v134 if v136 >= v140 else v139
    v62 = 'full' if v136 >= v140 else f'bottleneck_r{v59}'
    v63 = v61 >= 0.85
    v64 = v60 >= 0.8 * v142(v132, 0.01)
    v65 = 'FP_REMAP_ADAPTER_YES' if v63 and v64 else 'FP_REMAP_PARTIAL' if v60 > v121 + 0.05 else 'FP_REMAP_NO'
    v66 = {'stage': 221, 'overall': v65, 'gates': {'G_align_core': v63, 'G_recall_W_keys': v64}, 'core_vocab_n': v88(v50), 'mean_cos_before_shift': v58, 'mean_cos_after_W': v61, 'recall_old_fp': v121, 'recall_oracle_new': v132, 'recall_W_remapped': v60, 'remap_full': {'align': v134, 'recall': v136, 'params': 256 * 256}, 'remap_bottleneck': {'r': v59, 'align': v139, 'recall': v140, 'params': v59 * 256 * 2}, 'best_remap': v62, 'note': 'arc_enc domain shift then W on core vocab; old slots use W @ key_old', 'timestamp': v217.v211(v218.v212).v181()}
    v4.v143(v200.v182(v66, indent=2), encoding='utf-8')
    v5.v143(f'# Stage221 fp-remap\n\n**{v65}** align_W={v61:.3f} recall W={v60:.3f} oracle={v132:.3f}\n', encoding='utf-8')
    v124(f'VERDICT {v65}')
    return 0
if v67 == '__main__':
    raise v144(v183())