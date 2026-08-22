"""
Stage 221-probe — characterise W-remap (geometry, OOV, multi-domain, incremental).

Runs after the same protocol as 221; writes results/stage221_probe_decision.json.

  python _stage221_probe.py [--smoke]
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
v0 = v11('results')
v1 = v0 / 'stage221_probe_decision.json'
v2 = v0 / 'stage221_probe_mini.md'
v3 = v11('checkpoints/stage191_p1_curve.pt')
v4 = v11('data/external_tinystories_100k_85.txt')
v5 = v11('data/_wikitext103_train.txt')
v6 = 2211
v7 = ['Zorblax', 'Qjxtrv', 'pneumonoultramicroscopicsilicovolcanoconiosis', 'floccinaucinihilipilification', 'xyzabc']

@v80.v16()
def align_pairs(v12: v77, v13: v80.v78, v14: v80.v78) -> v8:
    v15 = v138.v79(v12.v139(v13), dim=-1)
    return v8((v15 * v14).v177(-1).v140())

@v80.v16()
def w_gram_stats(v17: v80.v78) -> v9:
    """W is (d,d) linear map fp' = W @ fp (column convention: Linear weight)."""
    v18 = v17.v81 @ v17
    v19 = v18.v82[0]
    v20 = v80.v83(v19, device=v18.v28, dtype=v18.v141)
    v21 = v8((v18 - v20).v189(2).v140().v142())
    v22 = v8((v18 - v18.v81).v189(2).v140().v142())
    v23 = v80.v166.v183(v18).v165().v84()
    v24 = v80.v166.v184(v17).v165().v84()
    v25 = v8(v80.v166.v25(v17))
    return {'WtW_frobenius_to_I': v21, 'WtW_asymmetry': v22, 'WtW_eig_min': v8(v23.v167()), 'WtW_eig_max': v8(v23.v106()), 'WtW_eig_mean': v8(v23.v140()), 'singular_value_min': v8(v24.v167()), 'singular_value_max': v8(v24.v106()), 'singular_value_mean': v8(v24.v140()), 'det_W': v25, 'interpretation': 'near_orthogonal' if v21 < 0.15 and v22 < 0.05 else 'general_linear_warp'}

def train_w_on_indices(v13: v80.v78, v14: v80.v78, v26: v123[v10], v27: v10, v28: v80.v28, v29: v10) -> v32[v77, v8]:
    v30 = v143.v85(v29)
    v31 = v77(256).v86(v28)
    v87, v88 = (v13[v26], v14[v26])
    v31, v89 = v144.v90(v31, v87, v88, v30, v27, v28, orth=True)
    return (v31, v122(v31, v87, v88))

def main() -> v10:
    v33 = v145.v91()
    v33.v92('--smoke', action='store_true')
    v34 = v33.v93()
    v28 = v80.v28('cuda' if v80.v178.v168() else 'cpu')
    v35 = 80 if v34.v94 else v144.v95
    v36 = 100 if v34.v94 else v144.v96
    v37 = 120 if v34.v94 else v144.v97
    v38 = 12 if v34.v94 else 60
    v30 = v143.v85(v6)
    v98, v99, v100, v101 = v102()
    v39 = v146.v103(v147(v169.v148))
    v40 = v39.v149(v150) or 0
    v41 = v179.v170(v39, v100, v40, v39.v180()).v86(v28)
    with v5.v151('r', encoding='utf-8', errors='ignore') as v104:
        v105 = v104.v152(2000000)
    v42 = v123(v9.v157((v155 for v155 in v192.v190('[A-Za-z][a-z]{2,}', v105) if v173(v155) <= 14)))[:v37]
    v43 = v106(20, v37 // 4)
    v44 = v42[:-v43]
    v45 = v42[-v43:]
    v46 = v171(v101, v39.v180()).v86(v28)
    v46.v107(v80.v172(v3, map_location=v28, weights_only=False)['model'])
    v46.v108()
    v47 = v109(v46, v100, v28)
    v48 = v144.v110(v47, v42)
    v49 = v4.v111(encoding='utf-8', errors='ignore')
    v112, v113 = v153.v114(v49, v39, v40, max_lines=500 if v34.v94 else 8000)
    v115, v116 = (v98, v99)
    v50 = v144.v117(v46, v112, v113, v41, v40, v28, v35, v6 + 1)
    v51 = v109(v50, v100, v28)
    v52 = v144.v110(v51, v42)
    v53 = [v42.v154(v155) for v155 in v44]
    v54 = [v42.v154(v155) for v155 in v45]
    v118, v89 = v119(v48, v52, v53, v36, v28, v6 + 2)
    v17 = v118.v155.v156.v120()
    v55 = v121(v17)
    v56 = v122(v118, v48[v53], v52[v53])
    v57 = v122(v118, v48[v54], v52[v54])
    v58 = v123(v9.v157(v7 + v45[:5]))
    v59 = v144.v110(v47, v58)
    v60 = v144.v110(v51, v58)
    v61 = v122(v118, v59, v60)
    v62 = v8((v59 * v60).v177(-1).v140())
    v63 = v8((v48[v54] * v52[v54]).v177(-1).v140())
    v64 = [100, 200, 400, 800] if not v34.v94 else [30, 60, 90]
    v65 = []
    for v66 in v64:
        if v66 > v173(v53):
            continue
        v124 = v53[:v66]
        v158, v89 = v119(v48, v52, v124, v36, v28, v6 + 100 + v66)
        v65.v159({'n_train_words': v66, 'align_train_subset': v122(v158, v48[v124], v52[v124]), 'align_hold_out': v122(v158, v48[v54], v52[v54])})
    with v5.v151('r', encoding='utf-8', errors='ignore') as v104:
        v125 = v123(v9.v157((v186.v185(1) for v186 in v193.v191(v104.v152(4000000)) if v173(v186.v185(1)) >= 5)))
    v67 = v160(v174(v125), v30, v38 + 10)[:v38]
    v68 = v125[:v38]
    v126, v127 = v144.v128(v47, v67, v68, v30)

    def tr(v129):
        return lambda v175: v138.v79(v129.v139(v175), dim=-1)
    v130, v89 = v144.v131(v126, v127, v51, v67, v68, v30, None)
    v132, v89 = v144.v131(v126, v127, v47, v67, v68, v30, v161(v118))
    v69 = v144.v117(v46, v115, v116, v41, v40, v28, v35, v6 + 3)
    v70 = v109(v69, v100, v28)
    v71 = v144.v110(v70, v42)
    v133, v89 = v119(v48, v71, v53, v36, v28, v6 + 4)
    v134, v89 = v144.v131(v126, v127, v47, v67, v68, v30, v161(v133))
    v72 = v61 >= 0.8 * v56 and v61 >= v57 - 0.05
    v73 = v132 >= 0.7 and v134 >= 0.7
    v74 = v173(v65) >= 2 and v65[-1]['align_hold_out'] >= v65[0]['align_hold_out'] - 0.08
    v75 = {'stage': '221-probe', 'overall': 'W_REMAP_CHARACTERIZED', 'q1_W_linear_gram': v55, 'q2_oov': {'train_align': v56, 'holdout_align': v57, 'holdout_mean_cos_raw': v63, 'oov_align_after_W': v61, 'oov_mean_cos_raw': v62, 'oov_words': v58, 'generalizes_beyond_core': v72, 'note': 'holdout = core words not used in W train; OOV includes nonsense/long forms'}, 'q3_multi_W': {'recall_W_B_stories_shift': v132, 'recall_W_C_wiki_shift': v134, 'recall_mismatch_old_keys_new_queries_no_W': v130, 'note': 'W_B/W_C each trained for its domain shift; same A-era slot keys; 221-style W on keys+queries', 'one_bank_two_projections_both_ge_0p70': v73}, 'q4_incremental': {'curve': v65, 'monotone_enough': v74}, 'prior_test_validation': {'mean_cos_word_fp_after_B_shift': v8((v48 * v52).v177(-1).v140()), 'note': '221 exam uses 4-way MC + composite slot keys; word cos ~0.68 is real but not same as slot recall'}, 'timestamp': v187.v181(v188.v182).v162()}
    v1.v135(v176.v163(v75, indent=2), encoding='utf-8')
    v2.v135(f"# Stage 221-probe\n\nOOV={v75['q2_oov']['oov_align_after_W']:.3f} multi={v75['q3_multi_W']['one_bank_two_projections_both_ge_0p70']}\n", encoding='utf-8')
    v136(v176.v163(v75, indent=2))
    return 0
if v76 == '__main__':
    raise v137(v164())