"""
Stage 207 MAX — same falsification protocol as 207, scaled to full WikiText-103 train.

Streaming corpus (no full-text RAM hold): two-pass word-rank array.
Defaults tuned for ~500MB wiki + long training on RTX 3050.

  python _stage207_max.py
  python _stage207_max.py --steps 12000 --v-lex 60000
"""
from __future__ import annotations
import argparse
import json
import random
import re
import time
import warnings
from collections import Counter
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
from _stage194_fp_fact_memory import FpBank
from _stage207_curve_thinking import D_MODEL, MAXLEN, N_EVAL, N_HEAD, N_LAYER, TEMP, Trunk, log
v0 = v13('results')
v1 = v13('checkpoints/stage191_p1_curve.pt')
v2 = v13('data/_wikitext103_train.txt')
v3 = v0 / 'stage207_max_decision.json'
v4 = v0 / 'stage207_max_mini.md'
v5 = v0 / '_stage207_max_log.txt'
v6 = 2071
v7 = 8000000
v8 = 8000
v9 = 512
v10 = v0 / '_stage207_max_ranks.mmap'
v11 = v74.v14('[a-z]{2,}')

def stream_words(v15: v13):
    with v15.v169('r', encoding='utf-8', errors='ignore') as v61:
        while True:
            v170 = v61.v225(v7)
            if not v170:
                break
            yield from v11.v251(v170.v264())

def build_ranks_safe(v15: v13, v16: v12):
    v75('pass 1: word frequencies (streaming)…')
    v17 = v76.v76()
    v18 = v77(v80(v15))
    v19 = [v27 for v27, v114 in v18.v226(v16)]
    v20 = {v27: v39 for v39, v27 in v227(v19)}
    v21 = v16
    v75(f'  distinct={v106(v18):,} lexicon={v106(v19):,} ({v76.v76() - v17:.0f}s)')
    v75('pass 2: rank sequence (streaming, memmap)…')
    v22 = v76.v76()
    v23 = v78((1 for v114 in v80(v15)))
    v75(f'  counting done: {v23:,} tokens ({v76.v76() - v22:.0f}s)')
    v24 = v76.v76()
    v25 = v171.v79(v10, dtype=v171.v172, mode='w+', shape=(v23,))
    v26 = 0
    for v27 in v80(v15):
        v25[v26] = v20.v173(v27, v21)
        v26 += 1
        if v26 % 10000000 == 0:
            v25.v81()
            v75(f'  wrote {v26:,}/{v23:,} ({v76.v76() - v24:.0f}s)')
    v25.v81()
    v75(f'  memmap {v10.v228} ({v76.v76() - v24:.0f}s)')
    return (v25, v19, v20, v21, v106(v18), v23)

def main() -> v12:
    v28 = v174.v82()
    v28.v83('--steps', type=v12, default=20000)
    v28.v83('--batch', type=v12, default=48)
    v28.v83('--lr', type=v155, default=0.0003)
    v28.v83('--v-lex', type=v12, default=80000)
    v28.v83('--log-every', type=v12, default=1000)
    v29 = v28.v84()
    v0.v85(parents=True, exist_ok=True)
    v175.v86('ignore', message='.*nested_tensor.*')
    v5.v87('', encoding='utf-8')
    v75(f'Stage207-MAX start {v268.v261(v269.v262).v221()}')
    v75(f'full wiki stream | steps={v29.v148} batch={v29.v191} V_LEX={v29.v16}')
    v30 = v176.v30('cuda' if v176.v252.v229() else 'cpu')
    v31 = v177.v88(v6)
    v176.v89(v6)
    v17 = v76.v76()
    if not v2.v178():
        v75(f'ERROR: missing {v2}')
        return 1
    v75(f'wiki bytes={v2.v263().v222:,}')
    v90, v91, v92, v93 = v94()
    v32 = v179.v95(v180(v230.v181))
    v33 = v32.v96()
    v34 = v231(v93, v33).v97(v30)
    v34.v98(v176.v232(v1, map_location=v30, weights_only=False)['model'])
    v34.v99()
    for v35 in v34.v100():
        v35.v182(False)
    v36 = v101(v34, v92, v30)
    v102, v19, v20, v103, v104, v23 = v105(v2, v29.v16)
    v37 = v106(v19)
    v38 = []
    for v39 in v107(0, v37, 2048):
        v38.v183(v36.v233(v19[v39:v39 + 2048]))
        if v39 // 2048 % 10 == 0 and v39:
            v75(f'  fp lexicon encode {v39}/{v37} ({v76.v76() - v17:.0f}s)')
    v40 = v176.v108(v38, 0)
    v41 = v176.v109(256, device=v30)
    v75(f'fp table {v253(v40.v247)} ({v76.v76() - v17:.0f}s)')
    v42 = v12(0.9 * v23)
    v43 = v42 - v184 - 1
    v44 = v102[v42:]
    v45 = v23 - v42

    def input_fp(v110):
        v111 = v176.v254(v110).v97(v30)
        return (v176.v234((v111 == v103).v240(-1), v41.v255(*v111.v247, 256), v40[v111.v265(max=v37 - 1)]), v111)

    def draw_train(v112):
        v113 = []
        for v114 in v107(v112):
            v149 = v31.v207(v43)
            v113.v183(v171.v208(v102[v149:v149 + v184 + 1], dtype=v171.v246))
        v115 = v171.v160(v113, 0)
        return (v115[:, :-1], v115[:, 1:])
    v46 = v235(d_out=256).v97(v30)
    v47 = v235(d_out=v8 + 1).v97(v30)
    v48 = v78((v35.v236() for v35 in v46.v100()))
    v49 = v176.v185.v116(v46.v100(), lr=v29.v186, weight_decay=0.01)
    v50 = v176.v185.v116(v47.v100(), lr=v29.v186, weight_decay=0.01)

    def ce_target(v117):
        return v176.v254(v171.v234(v117 < v8, v117, v8)).v97(v30)
    v51 = v52 = None
    v53 = v171.v75(v187(2, v29.v191 * v184))
    for v54 in v107(1, v29.v148 + 1):
        v188, v189 = v190(v29.v191)
        v192, v114 = v193(v188)
        v118 = v237.v194(v46(v192), dim=-1)
        v119 = v176.v254(v189).v97(v30)
        v120 = v119 != v103
        v121 = v118[v120]
        v122 = v119[v120]
        if v121.v238(0) < 2:
            continue
        if v121.v238(0) > v9:
            v195 = v176.v256(v121.v238(0), device=v30)[:v9]
            v121 = v121[v195]
            v122 = v122[v195]
        v123 = v40[v122]
        v124 = v121 @ v123.v239 / v196
        v125 = v122.v240(0) == v122.v240(1)
        v126 = v176.v126(v125.v238(0), dtype=v176.v241, device=v30)
        v124 = v124.v197(v125 & ~v126, v155('-inf'))
        v127 = v237.v198(v124, v176.v242(v121.v238(0), device=v30))
        v49.v199(set_to_none=True)
        v127.v200()
        v49.v54()
        v128 = v47(v192)
        v129 = v237.v198(v128.v243(-1, v8 + 1), v266(v189).v243(-1))
        v50.v199(set_to_none=True)
        v129.v200()
        v50.v54()
        v51 = v155(v127) if v51 is None else 0.995 * v51 + 0.005 * v155(v127)
        v52 = v155(v129) if v52 is None else 0.995 * v52 + 0.005 * v155(v129)
        if v54 % v29.v257 == 0 or v54 == v29.v148:
            v75(f'  step {v54}/{v29.v148}: curve_nce~{v51:.3f} ce~{v52:.3f} (floor~{v53:.2f}) ({v76.v76() - v17:.0f}s)')
    v46.v99()
    v47.v99()

    @v176.v138()
    def eval_rank(v130, v131, v132):
        v133 = v134 = v135 = 0
        v136 = v177.v88(v6 + 99)
        v137 = 0
        while v135 < v132 and v137 < v132 * 50:
            v137 += 1
            v149 = v136.v207(v45 - v184 - 1)
            v201 = v171.v208(v44[v149:v149 + v184 + 1], dtype=v171.v246)
            v202 = v12(v201[-1])
            if not v130 <= v202 < v131:
                continue
            v188 = v201[:-1][None, :]
            v192, v114 = v193(v188)
            v118 = v237.v194(v46(v192), dim=-1)[0, -1]
            v128 = v47(v192)[0, -1]
            v203 = [v202]
            while v106(v203) < 4:
                v244 = v136.v207(v130, v131)
                if v244 != v202 and v244 not in v203:
                    v203.v183(v244)
            v166 = v220(v107(4))
            v136.v219(v166)
            v167 = [v203[v39] for v39 in v166]
            v204 = v166.v245(0)
            v205 = [v155(v118 @ v40[v244]) for v244 in v167]
            v206 = [v155(v128[v244 if v244 < v8 else v8]) + 1e-06 * v136.v177() for v244 in v167]
            v133 += v12(v12(v171.v258(v205)) == v204)
            v134 += v12(v12(v171.v258(v206)) == v204)
            v135 += 1
        return (v133 / v187(1, v135), v134 / v187(1, v135), v135)
    v139, v140, v141 = v142(0, v8, v143)
    v144, v145, v146 = v142(v8, v37, v143)

    @v176.v138()
    def free_run(v147, v148=50):
        v149 = v31.v207(v45 - v184 - 1)
        v150 = v171.v208(v44[v149:v149 + 16], dtype=v171.v246)
        v151 = v193(v150[None, :])[0][0]
        v209, v210 = ([], [])
        for v114 in v107(v148):
            v118 = v237.v194(v46(v151.v240(0)), dim=-1)[0, -1]
            v162 = v40 @ v118
            v163 = v12(v162.v258())
            v209.v183(1.0 - v155(v162[v163]))
            v210.v183(v163)
            v119 = v40[v163] if v147 else v118
            v151 = v176.v108([v151, v119.v240(0)], 0)[-v184:]
        return (v209, v210)
    v152, v114 = v153(snap=False)
    v114, v154 = v153(snap=True)
    v55 = v155(v171.v211(v152[:10]))
    v56 = v155(v171.v211(v152[-10:]))
    v57 = v78((v12(v154[v39] == v154[v39 - 1]) for v39 in v107(1, v106(v154)))) / (v106(v154) - 1)

    @v176.v138()
    def trunk_key(v156):
        v110 = v171.v212([[v20.v173(v27, v103) for v27 in v156][:v184]], dtype=v171.v246)
        if v110.v247[1] < 2:
            return None
        v192, v114 = v193(v110)
        return v237.v194(v46.v270(v192)[0].v211(0), dim=-1)
    v58 = [v27 for v27 in v259(v267(v19), v31, 120) if v106(v27) >= 5][:80]
    v59 = v19[100:180]
    v60 = [{'S': v58[v39], 'V': v59[v39]} for v39 in v107(v260(v106(v58), v106(v59)))]
    v157, v158 = ([], [])
    for v61 in v60:
        v159 = v213(f"{v61['S']} was appointed director of {v61['V']} in 1987".v248())
        if v159 is not None:
            v157.v183(v159)
            v158.v183(v61['V'])
    v62 = v176.v160(v157, 0)
    v63 = v177.v88(v6 + 5)
    v64 = 0
    for v61 in v60[:v106(v158)]:
        v161 = v213(f"{v61['S']} was appointed director of".v248())
        if v161 is None:
            continue
        v162 = (v62 @ v161).v214()
        v163 = {}
        for v215, v216 in v217(v158, v162):
            v163[v215] = v187(v163.v173(v215, -9.9), v216)
        v164 = [v218 for v218 in v158 if v218 != v61['V']]
        v63.v219(v164)
        v165 = [v61['V']] + v164[:3]
        v166 = v220(v107(v106(v165)))
        v63.v219(v166)
        v167 = [v165[v26] for v26 in v166]
        v64 += v12(v12(v171.v258([v163.v173(v244, -9.9) for v244 in v167])) == v166.v245(0))
    v65 = v64 / v187(1, v106(v158))
    v66 = v139 >= v140 - 0.05
    v67 = v56 <= v55 + 0.15
    v68 = v144 >= v145 + 0.2 and v144 >= 0.5
    v69 = v65 >= 0.8
    v70 = v78([v66, v67, v68, v69])
    v71 = 'CURVE_THINKING_YES' if v66 and v68 and v67 and v69 else 'CURVE_THINKING_PARTIAL' if v66 and v68 else 'CURVE_THINKING_NO'
    v72 = {'timestamp': v268.v261(v269.v262).v221(), 'protocol': 'curve_as_thinking_207_max', 'overall': v71, 'gates_passed': f'{v70}/4', 'G1_quality_invocab': {'curve': v139, 'ce': v140, 'n': v141, 'pass': v66}, 'G3_open_vocab_oov': {'curve': v144, 'ce': v145, 'n': v146, 'chance': 0.25, 'pass': v68}, 'G2_drift': {'raw_first10': v55, 'raw_last10': v56, 'snap_repetition': v57, 'pass': v67}, 'G4_unified_memory': {'recall': v65, 'pass': v69}, 'config': {'wiki_bytes': v2.v263().v222, 'train_tokens': v12(v42), 'eval_tokens': v12(v45), 'n_distinct': v104, 'V_CE': v8, 'V_LEX': v37, 'STEPS': v29.v148, 'BATCH': v29.v191, 'params_each_M': v249(v48 / 1000000.0, 2)}, 'compare_baseline_207_smoke': {'note': '207 used 25M chars / 3500 steps / V_LEX 40k; this run is full wiki stream'}}
    v3.v87(v250.v223(v72, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v87(f'# Stage207 MAX\n\n**Overall:** `{v71}` ({v70}/4)\n\n- tokens train/eval: {v42:,} / {v45:,}\n- G1 curve/ce: {v139:.3f} / {v140:.3f}\n- G3 curve/ce: {v144:.3f} / {v145:.3f}\n- G2 drift: {v55:.3f} -> {v56:.3f}\n- G4 memory: {v65:.3f}\n', encoding='utf-8')
    v75(f'[207-MAX] {v71} ({v70}/4) | G1 {v139:.3f}/{v140:.3f} | G3 {v144:.3f}/{v145:.3f} | G2 {v55:.3f}->{v56:.3f} | G4 {v65:.3f} | train_tokens={v42:,}')
    return 0
if v73 == '__main__':
    raise v168(v224())