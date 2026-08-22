"""
Stage 207 — "curve as thinking": generative model whose OUTPUT space is the fp-space.

Same frozen char arc-encoder supplies word fingerprints for BOTH models; both share the same
causal-transformer trunk architecture and training budget. They differ ONLY in the output:

  CURVE : predicts the next word's FINGERPRINT in R^d, trained with in-batch InfoNCE (contrastive
          next-arc, NOT L2 -> no mean-collapse), decoded by SNAP to the nearest lexicon fp.
          Open metric vocabulary: can score/emit ANY word for which a fingerprint exists.
  CE    : predicts the next word ID via softmax over a CLOSED vocab (top-8k + UNK), trained with
          cross-entropy. Words outside the table collapse to UNK by construction.

Gates:
  G1 quality      curve k-way next-word acc within 0.05 of CE (on IN-VOCAB targets, fair to both)
  G2 drift (kill) free-run 50 steps: raw predicted fp must NOT walk off the lexicon manifold
                  (last-10 drift not >> first-10), i.e. snap is corrective, not a crutch for garbage
  G3 open-vocab   on OOV-for-CE targets (rank 8k..40k, real words): curve >> CE (CE ~ chance 0.25)
  G4 unification  the SAME trunk's hidden states work as memory keys for fact recall (>= 0.80)

  python _stage207_curve_thinking.py
"""
from __future__ import annotations
import json
import math
import random
import re
import time
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
from _stage194_fp_fact_memory import FpBank
from _stage192_fp_lexicon import gen_fakes
v0 = v23('results')
v1 = v23('checkpoints/stage191_p1_curve.pt')
v2 = v23('data/_wikitext103_train.txt')
v3 = v0 / 'stage207_decision.json'
v4 = v0 / 'stage207_mini.md'
v5 = v0 / '_stage207_log.txt'
v6 = 207
v7 = 25000000
v8 = 8000
v9 = 40000
v10 = v9
v11 = 48
v12 = 256
v13 = 4
v14 = 4
v15 = 3500
v16 = 48
v17 = 0.0003
v18 = 0.07
v19 = 800
v20 = v74.v24('[a-z]{2,}')

def log(v25: v75) -> None:
    v26 = v25 if v25.v186('\n') else v25 + '\n'
    try:
        v187(v26, end='', flush=True)
    except v76:
        v187(v26.v285('ascii', 'replace').v268('ascii'), end='', flush=True)
    v5.v188.v77(parents=True, exist_ok=True)
    with v5.v189('a', encoding='utf-8') as v62:
        v62.v190(v26)

class Trunk(v27.v21):

    def __init__(v78, v79, v80=256, v81=v12, v82=v13, v83=v14):
        v269().v191()
        v78.v84 = v27.v192(v80, v81)
        v78.v85 = v27.v193(v198.v270(1, v11, v81) * 0.02)
        v86 = v27.v194(v81, v83, dim_feedforward=4 * v81, batch_first=True, activation='gelu', norm_first=True)
        v78.v87 = v27.v195(v86, v82)
        v78.v88 = v27.v192(v81, v79)

    def hidden(v78, v89):
        v90 = v89.v196(1)
        v91 = v78.v84(v89) + v78.v85[:, :v90]
        v92 = v198.v197(v198.v245((v90, v90), v169('-inf'), device=v89.v28), diagonal=1)
        return v78.v87(v91, mask=v92)

    def forward(v78, v89):
        return v78.v88(v78.v246(v89))

def main() -> v22:
    v0.v77(parents=True, exist_ok=True)
    v5.v93('', encoding='utf-8')
    v94(f'Stage207 start {v283.v279(v284.v280).v241()}')
    v94('curve-as-thinking: generate next fingerprint (InfoNCE + snap) vs closed-vocab token CE')
    v28 = v198.v28('cuda' if v198.v271.v247() else 'cpu')
    v29 = v199.v95(v6)
    v198.v96(v6)
    v30 = v97.v97()
    v98, v99, v100, v101 = v102()
    v31 = v200.v103(v75(v248.v201))
    v32 = v31.v104()
    v33 = v31.v202(v203) or 0
    v34 = v249(v101, v32).v105(v28)
    v34.v106(v198.v250(v1, map_location=v28, weights_only=False)['model'])
    v34.v107()
    for v35 in v34.v108():
        v35.v204(False)
    v36 = v109(v34, v100, v28)
    v94(f'frozen arc-encoder loaded ({v97.v97() - v30:.0f}s)')
    with v2.v189('r', encoding='utf-8', errors='ignore') as v62:
        v38 = v62.v272(v7).v205()
    v37 = v20.v110(v38)
    del v38
    v39 = v111(v37)
    v40 = [v112 for v112, v125 in v39.v251(v9)]
    v41 = {v112: v43 for v43, v112 in v176(v40)}
    v94(f'words={v206(v37):,} distinct={v206(v39):,} lexicon={v206(v40):,} ({v97.v97() - v30:.0f}s)')
    v42 = []
    for v43 in v113(0, v206(v40), 4096):
        v42.v207(v36.v252(v40[v43:v43 + 4096]))
    v44 = v198.v114(v42, 0)
    v45 = v198.v115(256, device=v28)
    v94(f'fp lexicon table {v273(v44.v264)} built ({v97.v97() - v30:.0f}s)')
    v46 = v208.v116((v41.v253(v112, v10) for v112 in v37), dtype=v208.v209, count=v206(v37))
    v47 = v22(0.9 * v206(v46))
    v117, v118 = (v46[:v47], v46[v47:])

    def input_fp(v119):
        v120 = v198.v274(v119).v105(v28)
        v72 = v198.v210((v120 == v10).v254(-1), v45.v255(*v120.v264, 256), v44[v120.v275(max=v9 - 1)])
        return (v72, v120)

    def draw(v121, v122):
        v123 = []
        v124 = v206(v121) - v11 - 1
        for v125 in v113(v122):
            v160 = v29.v228(v124)
            v123.v207(v121[v160:v160 + v11 + 1])
        v126 = v208.v175(v123, 0)
        return (v126[:, :-1], v126[:, 1:])
    v48 = v256(d_out=256).v105(v28)
    v49 = v256(d_out=v8 + 1).v105(v28)
    v50 = v127((v35.v257() for v35 in v48.v108()))
    v94(f'trunk params each ~{v50 / 1000000.0:.2f}M ({v97.v97() - v30:.0f}s)')
    v51 = v198.v211.v128(v48.v108(), lr=v17, weight_decay=0.01)
    v52 = v198.v211.v128(v49.v108(), lr=v17, weight_decay=0.01)

    def ce_target(v129):
        v130 = v198.v274(v208.v210(v129 < v8, v129, v8)).v105(v28)
        return v130
    v53 = v54 = None
    for v55 in v113(1, v15 + 1):
        v212, v213 = v214(v117, v16)
        v215, v125 = v216(v212)
        v131 = v258.v217(v48(v215), dim=-1)
        v132 = v198.v274(v213).v105(v28)
        v133 = v132 != v10
        v134 = v131[v133]
        v135 = v132[v133]
        v136 = v44[v135]
        v137 = v134 @ v136.v90 / v18
        v138 = v135.v254(0) == v135.v254(1)
        v139 = v198.v139(v138.v196(0), dtype=v198.v259, device=v28)
        v137 = v137.v218(v138 & ~v139, v169('-inf'))
        v140 = v258.v219(v137, v198.v260(v134.v196(0), device=v28))
        v51.v220(set_to_none=True)
        v140.v221()
        v51.v55()
        v141 = v49(v215)
        v142 = v258.v219(v141.v261(-1, v8 + 1), v281(v213).v261(-1))
        v52.v220(set_to_none=True)
        v142.v221()
        v52.v55()
        v53 = v169(v140) if v53 is None else 0.98 * v53 + 0.02 * v169(v140)
        v54 = v169(v142) if v54 is None else 0.98 * v54 + 0.02 * v169(v142)
        if v55 % 500 == 0 or v55 == v15:
            v94(f'  step {v55}: curve_nce~{v53:.3f} ce~{v54:.3f} ({v97.v97() - v30:.0f}s)')
    v48.v107()
    v49.v107()

    @v198.v150()
    def eval_rank(v143, v124, v144):
        """positions whose next-word rank in [lo,hi); 4-way, candidates drawn from [lo,hi)."""
        v145 = v146 = v147 = 0
        v148 = v199.v95(v6 + 99)
        v149 = 0
        while v147 < v144 and v149 < v144 * 40:
            v149 += 1
            v160 = v148.v228(v206(v118) - v11 - 1)
            v222 = v118[v160:v160 + v11 + 1]
            v223 = v22(v222[-1])
            if not v143 <= v223 < v124:
                continue
            v212 = v222[:-1][None, :]
            v215, v125 = v216(v212)
            v131 = v258.v217(v48(v215), dim=-1)[0, -1]
            v141 = v49(v215)[0, -1]
            v224 = [v223]
            while v206(v224) < 4:
                v262 = v148.v228(v143, v124)
                if v262 != v223 and v262 not in v224:
                    v224.v207(v262)
            v182 = v239(v113(4))
            v148.v238(v182)
            v183 = [v224[v43] for v43 in v182]
            v225 = v182.v263(0)
            v226 = [v169(v131 @ v44[v262]) for v262 in v183]
            v227 = [v169(v141[v262 if v262 < v8 else v8]) + 1e-06 * v148.v199() for v262 in v183]
            v145 += v22(v22(v208.v276(v226)) == v225)
            v146 += v22(v22(v208.v276(v227)) == v225)
            v147 += 1
        return (v145 / v240(1, v147), v146 / v240(1, v147), v147)
    v151, v152, v153 = v154(0, v8, v19)
    v155, v156, v157 = v154(v8, v9, v19)
    v94(f'G1 in-vocab (n={v153}): curve={v151:.3f} ce={v152:.3f}')
    v94(f'G3 OOV-for-CE (n={v157}): curve={v155:.3f} ce={v156:.3f} (chance 0.25)')

    @v198.v150()
    def free_run(v158, v159=50):
        v160 = v29.v228(v206(v118) - v11 - 1)
        v161 = v118[v160:v160 + 16]
        v162 = v216(v161[None, :])[0][0]
        v229, v230 = ([], [])
        v163 = v162
        for v125 in v113(v159):
            v131 = v258.v217(v48(v163.v254(0)), dim=-1)[0, -1]
            v178 = v44 @ v131
            v179 = v22(v178.v276())
            v229.v207(1.0 - v169(v178[v179]))
            v230.v207(v179)
            v132 = v44[v179] if v158 else v131
            v163 = v198.v114([v163, v132.v254(0)], 0)[-v11:]
        return (v229, v230)
    v164, v165 = v166(snap=False)
    v167, v168 = v166(snap=True)
    v56 = v169(v208.v231(v164[:10]))
    v57 = v169(v208.v231(v164[-10:]))
    v58 = v127((v22(v168[v43] == v168[v43 - 1]) for v43 in v113(1, v206(v168)))) / (v206(v168) - 1)
    v94(f'G2 raw drift first10={v56:.3f} last10={v57:.3f} | snap repetition={v58:.3f}')

    @v198.v150()
    def trunk_key(v170):
        v119 = v208.v232([[v41.v253(v112, v10) for v112 in v170][:v11]], dtype=v208.v209)
        if v119.v264[1] < 2:
            return None
        v215, v125 = v216(v119)
        v171 = v48.v246(v215)[0]
        return v258.v217(v171.v231(0), dim=-1)
    v59 = [v112 for v112 in v277(v282(v40), v29, 120) if v206(v112) >= 5][:80]
    v60 = v40[100:100 + 80]
    v61 = [{'S': v59[v43], 'V': v60[v43]} for v43 in v113(v278(v206(v59), v206(v60)))]
    v172, v173 = ([], [])
    for v62 in v61:
        v174 = v233(f"{v62['S']} was appointed director of {v62['V']} in 1987".v121())
        if v174 is not None:
            v172.v207(v174)
            v173.v207(v62['V'])
    v63 = v198.v175(v172, 0)
    v64 = v199.v95(v6 + 5)
    v65 = 0
    for v43, v62 in v176(v61[:v206(v173)]):
        v177 = v233(f"{v62['S']} was appointed director of".v121())
        if v177 is None:
            continue
        v178 = (v63 @ v177).v234()
        v179 = {}
        for v235, v236 in v237(v173, v178):
            v179[v235] = v240(v179.v253(v235, -9.9), v236)
        v180 = [v91 for v91 in v173 if v91 != v62['V']]
        v64.v238(v180)
        v181 = [v62['V']] + v180[:3]
        v182 = v239(v113(v206(v181)))
        v64.v238(v182)
        v183 = [v181[v265] for v265 in v182]
        v65 += v22(v22(v208.v276([v179.v253(v262, -9.9) for v262 in v183])) == v182.v263(0))
    v66 = v65 / v240(1, v206(v173))
    v94(f'G4 trunk-hidden memory recall (4-way): {v66:.3f}')
    v67 = v151 >= v152 - 0.05
    v68 = v57 <= v56 + 0.15
    v69 = v155 >= v156 + 0.2 and v155 >= 0.5
    v70 = v66 >= 0.8
    v71 = v127([v67, v68, v69, v70])
    if v67 and v69 and v68 and v70:
        v184 = 'CURVE_THINKING_YES'
    elif v67 and v69:
        v184 = 'CURVE_THINKING_PARTIAL'
    else:
        v184 = 'CURVE_THINKING_NO'
    v72 = {'timestamp': v283.v279(v284.v280).v241(), 'protocol': 'curve_as_thinking_207', 'overall': v184, 'gates_passed': f'{v71}/4', 'G1_quality_invocab': {'curve': v151, 'ce': v152, 'n': v153, 'pass': v67}, 'G3_open_vocab_oov': {'curve': v155, 'ce': v156, 'n': v157, 'chance': 0.25, 'pass': v69}, 'G2_drift': {'raw_first10': v56, 'raw_last10': v57, 'snap_repetition': v58, 'pass': v68}, 'G4_unified_memory': {'recall': v66, 'pass': v70}, 'config': {'V_CE': v8, 'V_LEX': v9, 'MAXLEN': v11, 'D_MODEL': v12, 'N_LAYER': v13, 'STEPS': v15, 'params_each_M': v266(v50 / 1000000.0, 2)}, 'note': "output space is the shared fp lexicon (open metric vocab) vs CE's closed softmax table; contrastive next-arc avoids L2 mean-collapse; snap-to-lexicon is the error-correcting decoder"}
    v3.v93(v267.v242(v72, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v93('\n'.v243(['# Stage207 — curve as thinking (fp-generative vs token CE)', '', f'**Overall:** `{v184}` ({v71}/4 gates)', '', '| gate | curve | CE / baseline | pass |', '|------|-------|---------------|------|', f'| G1 quality (in-vocab, 4-way) | {v151:.3f} | {v152:.3f} | {v67} |', f'| G3 open-vocab OOV (4-way, chance 0.25) | **{v155:.3f}** | {v156:.3f} | {v69} |', f'| G2 drift raw first→last10 | {v56:.3f}→{v57:.3f} | snap rep {v58:.3f} | {v68} |', f'| G4 unified trunk-hidden memory | {v66:.3f} | — | {v70} |', '', f'- shared frozen arc-encoder input; both trunks {v266(v50 / 1000000.0, 2)}M params, {v15} steps.', "- G3 is the essence gate: CE's closed softmax gives every OOV word the same UNK logit → chance;", '  the curve ranks them by fingerprint in an open metric vocabulary.']), encoding='utf-8')
    v94(f'[207] {v184} ({v71}/4) | G1 {v151:.2f}/{v152:.2f} | G3 {v155:.2f}/{v156:.2f} | G2 {v56:.2f}->{v57:.2f} | G4 {v66:.2f}')
    return 0
if v73 == '__main__':
    raise v185(v244())