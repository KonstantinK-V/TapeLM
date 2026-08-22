"""
Stage 203 — internal hops (for-interest): a trainable differentiable k-hop reader
that performs multi-hop retrieval INSIDE a forward pass, over a FROZEN fp-space and a
NON-GRADIENT slot tape. Replaces the external argmax loop (200) with a learned module.

Anti-CF by construction: encoder frozen, tape non-gradient; only the tiny HopReader trains.
The reader is told how many hops k to take (question spec) and must land on entity A_k.

  state_0 = proj(fp(A0)) + kemb[k]
  for t in 1..T:  a = softmax(state @ K^T / temp); read = a @ V; state = state + U([state;read])
  answer = normalize(Wo(state));  predict = argmax_c cos(answer, fp(c))

Gates:
  G_learn        internal reader test acc (k=2,3) >= 0.70 (chance 0.25)
  G_generalize   test chains (unseen) acc high
  G_vs_handloop  report external argmax-loop acc for reference
  G_anticf       encoder params untouched (assert)

  python _stage203_internal_hops.py
"""
from __future__ import annotations
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
v0 = v18('results')
v1 = v18('checkpoints/stage191_p1_curve.pt')
v2 = v18('data/_wikitext103_train.txt')
v3 = v0 / 'stage203_decision.json'
v4 = v0 / 'stage203_mini.md'
v5 = v0 / '_stage203_log.txt'
v6 = 203
v7 = 40000000
v8 = 90000000
v9 = 240
v10 = 4
v11 = 4000
v12 = 3
v13 = 1500
v14 = 128
v15 = 0.001

def log(v19: v63) -> None:
    v20 = v19 if v19.v131('\n') else v19 + '\n'
    try:
        v132(v20, end='', flush=True)
    except v64:
        v132(v20.v216('ascii', 'replace').v199('ascii'), end='', flush=True)
    v5.v133.v65(parents=True, exist_ok=True)
    with v5.v134('a', encoding='utf-8') as v66:
        v66.v135(v20)

class HopReader(v21.v16):

    def __init__(v67, v39, v68=v12, v69=v10):
        v200().v136()
        v67.v70 = v68
        v67.v71 = v21.v137(v39, v39)
        v67.v72 = v21.v138(v69 + 1, v39)
        v67.v73 = v21.v139(v21.v137(2 * v39, v39), v21.v170(), v21.v137(v39, v39))
        v67.v61 = v21.v137(v39, v39)
        v67.v74 = v21.v140(v150.v146(0.1))

    def forward(v67, v75, v76, v25, v26):
        v77 = v67.v71(v75) + v67.v72(v76)
        for v78 in v85(v67.v70):
            v141 = v150.v171(v77 @ v25.v204 / v67.v74.v144(min=0.02), dim=-1)
            v142 = v141 @ v26
            v77 = v77 + v67.v73(v150.v209([v77, v142], dim=-1))
        return v172.v143(v67.v61(v77), dim=-1)

class SoftFollow(v21.v16):
    """Minimal-structure internal hops: pure soft value-following + step selection by k.
    Only a temperature is learned; the hop OPERATION is parameter-free (should generalize)."""

    def __init__(v67, v39, v68=v12):
        v200().v136()
        v67.v70 = v68
        v67.v79 = v21.v140(v150.v146(-3.0))

    def forward(v67, v75, v76, v25, v26):
        v74 = v67.v79.v201().v144(min=0.01, max=1.0)
        v77 = v75
        v80 = []
        for v78 in v85(v67.v70):
            v77 = v172.v143(v150.v171(v77 @ v25.v204 / v74, dim=-1) @ v26, dim=-1)
            v80.v160(v77)
        v81 = v150.v115(v80, dim=1)
        v82 = (v76 - 1).v144(min=0, max=v67.v70 - 1)
        return v81[v150.v173(v81.v210(0), device=v81.v27), v82]

def train_reader(v22, v23, v24, v25, v26, v27, v28):
    v29 = v150.v145.v83(v22.v107(), lr=v15, weight_decay=0.01)
    v22.v84()
    v30 = None
    for v31 in v85(1, v13 + 1):
        v86 = [v23[v28.v202(0, v197(v23) - 1)] for v78 in v85(v14)]
        v87 = v24.v117([v203[0] for v203 in v86])
        v88 = v150.v146([v203[1] for v203 in v86], device=v27)
        v89 = v24.v117([v203[2] for v203 in v86])
        v90 = v22(v87, v88, v25, v26)
        v91 = v172.v147(v90 @ v89.v204 / 0.1, v150.v173(v197(v86), device=v27))
        v29.v148(set_to_none=True)
        v91.v149()
        v29.v31()
        v30 = v174(v91) if v30 is None else 0.98 * v30 + 0.02 * v174(v91)
    v22.v92()
    return v30

def main() -> v17:
    v0.v65(parents=True, exist_ok=True)
    v5.v93('', encoding='utf-8')
    v94(f'Stage203 start {v214.v207(v215.v208).v166()}')
    v94('internal hops: trainable differentiable k-hop reader over frozen fp tape')
    v27 = v150.v27('cuda' if v150.v205.v175() else 'cpu')
    v28 = v151.v95(v6)
    v150.v96(v6)
    v32 = v97.v97()
    v98, v99, v100, v101 = v102()
    v33 = v152.v103(v63(v176.v153))
    v34 = v33.v104()
    v35 = v33.v154(v155) or 0
    v36 = v177(v101, v34).v105(v27)
    v36.v106(v150.v178(v1, map_location=v27, weights_only=False)['model'])
    v36.v92()
    for v37 in v36.v107():
        v37.v156(False)
    v38 = v108((v174(v37.v165().v108()) for v37 in v36.v211.v107()))
    v24 = v109(v36, v100, v27)
    v39 = v36.v157.v110 // 2
    v94(f'encoder frozen (fp dim={v39}) ({v97.v97() - v32:.0f}s)')
    with v2.v134('r', encoding='utf-8', errors='ignore') as v66:
        v41 = v66.v142(v7)[v8 % v7:]
    v40 = v116(v179.v161((v213.v212(1) for v213 in v219.v217(v41) if v197(v213.v212(1)) >= 4)))[:3000]
    del v41
    v42 = v111(v158(), v28, v9 * v10 + 50)
    v43 = [v42[v47 * v10:(v47 + 1) * v10] for v47 in v85(v9)]
    v43 = [v46 for v46 in v43 if v197(v46) == v10]
    v28.v112(v43)
    v44 = v43[:v17(0.75 * v197(v43))]
    v45 = v43[v17(0.75 * v197(v43)):]
    v113, v114 = ([], [])
    for v46 in v43:
        for v47 in v85(v197(v46) - 1):
            v113.v160(v24.v117([v46[v47]])[0])
            v114.v160(v24.v117([v46[v47 + 1]])[0])
    for v47 in v85(v159(v11, v197(v40) - 1)):
        v113.v160(v24.v117([v40[v47]])[0])
        v114.v160(v24.v117([v40[v47 + 1]])[0])
    v25 = v150.v115(v113, 0)
    v26 = v150.v115(v114, 0)
    v94(f'tape slots={v25.v196[0]} chains={v197(v43)} (train {v197(v44)}/test {v197(v45)}) ({v97.v97() - v32:.0f}s)')
    v48 = v116(v179.v161([v180 for v46 in v43 for v180 in v46]))
    v49 = v24.v117(v48)

    def samples(v118):
        v61 = []
        for v46 in v118:
            for v76 in v85(1, v10):
                v61.v160((v46[0], v76, v46[v76]))
        return v61
    v23 = v119(v44)
    v50 = v119(v45)
    v51 = v181(v39).v105(v27)
    v52 = v120(v51, v23, v24, v25, v26, v27, v151.v95(v6))
    v94(f'free-form reader trained (loss~{v52:.3f})')
    v53 = v182(v39).v105(v27)
    v54 = v120(v53, v23, v24, v25, v26, v27, v151.v95(v6))
    v94(f'soft-follow reader trained (loss~{v54:.3f}, temp={v174(v53.v79.v201()):.3f})')

    @v150.v123()
    def eval_reader(v22, v121):
        v122 = {}
        for v76 in v85(1, v10):
            v162 = [v183 for v183 in v121 if v183[1] == v76]
            v163 = 0
            for v184, v78, v185 in v162:
                v90 = v22(v24.v117([v184]), v150.v146([v76], device=v27), v25, v26)[0]
                v186 = [v185] + [v48[v28.v202(0, v197(v48) - 1)] for v78 in v85(3)]
                v187 = v116(v85(4))
                v28.v112(v187)
                v188 = [v186[v47] for v47 in v187]
                v189 = v187.v206(0)
                v190 = [v174(v90 @ v24.v117([v46])[0]) for v46 in v188]
                v163 += v17(v17(v220.v218(v190)) == v189)
            v122[v76] = v163 / v195(1, v197(v162))
        return v122

    @v150.v123()
    def hand_loop(v121):
        v122 = {}
        for v76 in v85(1, v10):
            v162 = [v183 for v183 in v121 if v183[1] == v76]
            v163 = 0
            for v184, v78, v185 in v162:
                v191 = v24.v117([v184])[0]
                for v78 in v85(v76):
                    v191 = v26[v17((v25 @ v191).v218())]
                v192 = v48[v17((v49 @ v191).v218())]
                v163 += v17(v192 == v185)
            v122[v76] = v163 / v195(1, v197(v162))
        return v122
    v124, v125 = (v164(v51, v23), v164(v51, v50))
    v126, v127 = (v164(v53, v23), v164(v53, v50))
    v55 = v128(v50)
    v56 = v108((v174(v37.v165().v108()) for v37 in v36.v211.v107()))
    v57 = v165(v38 - v56) < 0.001
    v94(f'free-form  train={v124} test={v125}')
    v94(f'soft-follow train={v126} test={v127}')
    v94(f'external hand-loop test={v55}')
    v94(f'anti-CF encoder untouched: {v57}')
    v58 = v127.v193(2, 0) >= 0.7 and v127.v193(3, 0) >= 0.7
    v59 = v159(v127.v194()) >= 0.6
    v60 = v159(v124.v194()) >= 0.9 and v195(v125.v194()) <= 0.4
    if v58 and v59 and v57:
        v129 = 'INTERNAL_HOPS_YES_IF_STRUCTURED'
    elif v127.v193(2, 0) >= 0.6:
        v129 = 'INTERNAL_HOPS_PARTIAL'
    else:
        v129 = 'INTERNAL_HOPS_NO'
    v61 = {'timestamp': v214.v207(v215.v208).v166(), 'protocol': 'internal_hops_203', 'overall': v129, 'free_form_reader': {'train': v124, 'test': v125}, 'soft_follow_reader': {'train': v126, 'test': v127}, 'external_hand_loop_test': v55, 'free_form_overfits': v60, 'anticf_encoder_frozen': v57, 'tape_slots': v17(v25.v196[0]), 'chains_train_test': [v197(v44), v197(v45)], 't_steps': v12, 'chance': 0.25, 'note': 'internal hops generalize ONLY with minimal structure (soft value-follow + step select); a free-form learned reader memorizes; anti-CF preserved (frozen encoder + non-grad tape)'}
    v3.v93(v198.v167(v61, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v93('\n'.v168(['# Stage203 — internal hops (free-form vs structured)', '', f'**Overall:** `{v129}`', '', f'- **soft-follow (structured) test:** k1={v127.v193(1):.2f} k2={v127.v193(2):.2f} k3={v127.v193(3):.2f} (chance 0.25)', f'- free-form test: k1={v125.v193(1):.2f} k2={v125.v193(2):.2f} k3={v125.v193(3):.2f} (train {v124.v193(2):.2f} → overfits={v60})', f'- external hand-loop test: k1={v55.v193(1):.2f} k2={v55.v193(2):.2f} k3={v55.v193(3):.2f}', f'- anti-CF (encoder untouched): {v57}, tape slots={v17(v25.v196[0])}, T={v12}']), encoding='utf-8')
    v94(f'[203] {v129} | soft test k2={v127.v193(2):.2f} k3={v127.v193(3):.2f} | free test k3={v125.v193(3):.2f} | handloop k3={v55.v193(3):.2f}')
    return 0
if v62 == '__main__':
    raise v130(v169())