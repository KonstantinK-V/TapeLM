"""
Stage 201 — B-track on YOUR GPU: crack form-dominance with minimal-pair hard negatives.

199 showed a CPC head on FROZEN features can't move B (car≈cat baked into the substrate).
So here we touch the SUBSTRATE — but on a COPY of the encoder, leaving the product P1 frozen
and intact. Objective directly attacks form-dominance:
  - HARD NEGATIVES: edit-distance-1 word pairs mined from the corpus (car/cat, cold/bold,
    door/book-like) are pushed APART in fp-space.
  - ANCHOR: every word's new fp is kept near its original P1 fp (prevents collapse / preserves
    the structure that gives parity + memory).
Then measure B on the 179 sentence pairs through the fine-tuned encoder's fast channel.
Success = hard pairs drop below paraphrases (INVERSION), i.e. meaning finally beats spelling.

Fits a 3050: word-level contrastive on arc_enc only (fast/head frozen), few hundred steps.

Gates:
  G_invert  para_new > hard_new                                   -> SEM_HARDNEG_YES
  G_trend   hard_new <= baseline_hard - 0.10 and para_new >= 0.60 -> SEM_HARDNEG_TREND
  report    next_tok on the COPY (generation cost of touching the substrate; product P1 untouched)

  python _stage201_semantic_hardneg.py
"""
from __future__ import annotations
import copy
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data, score_items, span_logprob_x
v0 = v20('results')
v1 = v20('checkpoints/stage191_p1_curve.pt')
v2 = v20('data/_wikitext103_train.txt')
v3 = v20('data/stage191_exam_v3.jsonl')
v4 = v0 / 'stage201_decision.json'
v5 = v0 / 'stage201_mini.md'
v6 = v0 / '_stage201_log.txt'
v7 = 201
v8 = 60000000
v9 = 40000
v10 = 20000
v11 = 500
v12 = 256
v13 = 256
v14 = 0.0002
v15 = 1.0
v16 = v21.v17
v18 = 64

def log(v22: v56) -> None:
    v23 = v22 if v22.v115('\n') else v22 + '\n'
    try:
        v116(v23, end='', flush=True)
    except v57:
        v116(v23.v191('ascii', 'replace').v172('ascii'), end='', flush=True)
    v6.v117.v58(parents=True, exist_ok=True)
    with v6.v118('a', encoding='utf-8') as v59:
        v59.v119(v23)

def mine_minimal_pairs(v24: v30[v56], v25: v120.v60) -> v30[v68[v56, v56]]:
    """edit-distance-1 substitution pairs via wildcard buckets (car/cat, cold/bold)."""
    v26: v61[v56, v30[v56]] = v62(v30)
    for v27 in v24:
        for v63 in v101(v151(v27)):
            v26[v27[:v63] + '*' + v27[v63 + 1:]].v155(v27)
    v28 = v64()
    for v29 in v26.v65():
        if v151(v29) < 2:
            continue
        for v66 in v101(v151(v29)):
            for v121 in v101(v66 + 1, v151(v29)):
                v28.v173((v29[v66], v29[v121]))
                if v151(v28) >= v10 * 2:
                    break
    v28 = v30(v28)
    v25.v67(v28)
    return v28[:v10]

def main() -> v19:
    v0.v58(parents=True, exist_ok=True)
    v6.v69('', encoding='utf-8')
    v70(f'Stage201 start {v189.v182(v190.v183).v150()}')
    v70('B-track: minimal-pair hard negatives on an encoder COPY (product P1 stays frozen)')
    v31 = v122.v31('cuda' if v122.v174.v156() else 'cpu')
    v25 = v120.v60(v7)
    v32 = v71.v71()
    v72, v73, v74, v75 = v76()
    v33 = v123.v77(v56(v21.v124))
    v34 = v33.v78()
    v35 = v33.v125(v126) or 0
    v36 = v175.v157(v33, v74, v35, v34).v79(v31)
    v37 = v158(v75, v34).v79(v31)
    v37.v80(v122.v159(v1, map_location=v31, weights_only=False)['model'])
    v37.v81()
    for v38 in v37.v82():
        v38.v127(False)
    v39 = v128.v83(v37)
    for v38 in v39.v82():
        v38.v127(False)
    for v38 in v39.v129.v82():
        v38.v127(True)
    v39.v129.v84()
    v70(f'copy made; arc_enc trainable ({v184((v38.v194() for v38 in v39.v129.v82())) / 1000.0:.0f}k params)')

    def word_rows(v24):
        v85 = v122.v130(v151(v24), 1, v16, dtype=v122.v160)
        for v63, v27 in v131(v24):
            for v161, v162 in v131(v27[:v16]):
                v85[v63, 0, v161] = v74.v176(v162, 0)
        return v85.v79(v31)

    def word_fp(v24, v86):
        return v163.v132(v86.v129(v185(v24))[:, 0], dim=-1)
    with v2.v118('r', encoding='utf-8', errors='ignore') as v59:
        v41 = v59.v133(v8)
    v40 = v87(v164.v134('[a-z]{3,}', v41.v165()))
    del v41
    v24 = [v27 for v27, v162 in v40.v166(v9)]
    v28 = v88(v24, v25)
    v70(f'vocab={v151(v24)} minimal_pairs={v151(v28)} (e.g. {v28[:5]}) ({v71.v71() - v32:.0f}s)')
    v42 = v24[:8000]
    with v122.v96():
        v89 = {}
        for v63 in v101(0, v151(v42), 4096):
            v135 = v42[v63:v63 + 4096]
            v136 = v141(v135, v37)
            for v27, v167 in v168(v135, v136):
                v89[v27] = v167
    v70(f'anchor fps={v151(v89)} ({v71.v71() - v32:.0f}s)')

    @v122.v96()
    def pooled_text(v90: v56, v86):
        v91 = [v63 for v63 in v33.v191(v90).v91 if v63 != v35][:v18]
        v92 = v122.v137([v91], device=v31)
        v93 = v92 == v35
        v94 = v86.v138(v36[v92], v92)
        v95 = v86.v95(v94, pad_mask=v93)
        return v163.v132(v95.v142(1)[0], dim=-1)

    def measure_B(v86):
        v97 = v139(v177.v142([v139(v163.v192(v195(v66, v86), v195(v121, v86), dim=-1)) for v66, v121 in v193.v186]))
        v98 = v139(v177.v142([v139(v163.v192(v195(v66, v86), v195(v121, v86), dim=-1)) for v66, v121 in v193.v187]))
        return {'para': v97, 'hard': v98, 'inversion': v97 > v98}
    v43 = v99(v37)
    v70(f"P1 baseline B: para={v43['para']:.3f} hard={v43['hard']:.3f} inv={v43['inversion']}")
    v44 = v122.v140.v100(v39.v129.v82(), lr=v14)
    v45 = None
    for v46 in v101(1, v11 + 1):
        v102 = [v28[v25.v178(0, v151(v28) - 1)] for v169 in v101(v12)]
        v103 = [v38[0] for v38 in v102]
        v104 = [v38[1] for v38 in v102]
        v105 = v141(v103, v39)
        v106 = v141(v104, v39)
        v107 = (v105 * v106).v184(-1).v179(min=-1, max=1).v142()
        v108 = [v42[v25.v178(0, v151(v42) - 1)] for v169 in v101(v13)]
        v109 = v141(v108, v39)
        v110 = v122.v143([v89[v27] for v27 in v108], 0)
        v111 = (1.0 - (v109 * v110).v184(-1)).v142()
        v112 = v107 + v15 * v111
        v44.v144(set_to_none=True)
        v112.v145()
        v44.v46()
        v45 = v139(v112) if v45 is None else 0.97 * v45 + 0.03 * v139(v112)
        if v46 % 100 == 0 or v46 == v11:
            v121 = v99(v39)
            v70(f"  step {v46}: loss~{v45:.3f} (neg~{v139(v107):.3f} anchor~{v139(v111):.3f}) | para={v121['para']:.3f} hard={v121['hard']:.3f} inv={v121['inversion']} ({v71.v71() - v32:.0f}s)")
    v39.v129.v81()
    v47 = v99(v39)
    v48 = [v170.v146(v147) for v147 in v3.v188(encoding='utf-8').v171()]
    v49 = [v148 for v148 in v48 if v148['type'] == 'next_tok'][:120]
    v50 = v149(lambda v162, v180: v181(v39, v36, v35, v162, v180, v31), v49, 'next_tok')['next_tok_acc']
    v51 = v149(lambda v162, v180: v181(v37, v36, v35, v162, v180, v31), v49, 'next_tok')['next_tok_acc']
    v70(f'next_tok: P1(product)={v51:.3f}  copy(fine-tuned)={v50:.3f}')
    v52 = v47['para'] > v47['hard']
    v53 = v47['hard'] <= v43['hard'] - 0.1 and v47['para'] >= 0.6
    if v52:
        v113 = 'SEM_HARDNEG_YES'
    elif v53:
        v113 = 'SEM_HARDNEG_TREND'
    else:
        v113 = 'SEM_HARDNEG_NO'
    v54 = {'timestamp': v189.v182(v190.v183).v150(), 'protocol': 'semantic_hardneg_201', 'overall': v113, 'baseline_B': v43, 'after_B': v47, 'next_tok_product_p1': v51, 'next_tok_finetuned_copy': v50, 'minimal_pairs': v151(v28), 'gates': {'g_invert': v52, 'g_trend': v53}, 'note': 'arc_enc COPY fine-tuned with edit-distance-1 hard negatives + anchor-to-P1; product P1 frozen'}
    v4.v69(v170.v152(v54, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v69('\n'.v153(['# Stage201 — B via minimal-pair hard negatives (encoder copy)', '', f'**Overall:** `{v113}`', '', f"- P1 baseline: para {v43['para']:.3f} / hard {v43['hard']:.3f} (inv={v43['inversion']})", f"- after hard-neg: para {v47['para']:.3f} / hard {v47['hard']:.3f} (**inv={v47['inversion']}**)", f'- next_tok: product P1 {v51:.3f} | fine-tuned copy {v50:.3f}', f'- minimal pairs mined: {v151(v28)}', '']), encoding='utf-8')
    v70(f"[201] {v113} | para {v43['para']:.2f}->{v47['para']:.2f} hard {v43['hard']:.2f}->{v47['hard']:.2f} nt_copy={v50:.2f}")
    return 0
if v55 == '__main__':
    raise v114(v154())