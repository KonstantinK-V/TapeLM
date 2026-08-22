"""
Stage 208 — hybrid for variant A: a word-level fp RERANKER for rare words, on top of the frozen A stack.

Why this is not a rerun of 207: 207 asked the model to GENERATE the next fingerprint (regress a spelling
code) and failed. Here the fp side is DISCRIMINATIVE — it only has to prefer the right candidate among a
few, using the frozen context state that we already know is informative (191 next_tok 0.867).

  A (baseline)  : mean per-piece log-prob of the candidate word's BPE pieces (frozen CE head)
  fp reranker   : score = <W(h_ctx), fp(word)> ; only W trains, encoder frozen
  combined      : z-scored A + w * z-scored fp
  gated         : w chosen per item by fp-lexicon surprise of the CANDIDATES (read-only, no gold peeking)

Candidates are frequency-matched (exam-v2 discipline) so nothing can be won by unigram frequency.

Gates:
  G1 no_degrade  gated overall acc >= A overall acc - 0.01
  G2 rare_win    gated acc on the RARE band >= A + 0.10
  G3 read_only   encoder/head params unchanged (assert)

  python _stage208_hybrid_rare_head.py
"""
from __future__ import annotations
import json
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
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage194_fp_fact_memory import FpBank
v0 = v20('results')
v1 = v20('checkpoints/stage191_p1_curve.pt')
v2 = v20('data/_wikitext103_train.txt')
v3 = v0 / 'stage208_decision.json'
v4 = v0 / 'stage208_mini.md'
v5 = v0 / '_stage208_log.txt'
v6 = 208
v7 = 12000000
v8 = 0.85
v9 = 8000
v10 = 40000
v11 = 12000
v12 = 700
v13 = 15
v14 = 2500
v15 = 128
v16 = 0.0003
v17 = v72.v21('[A-Za-z]{3,}')
v18 = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

def log(v22: v73) -> None:
    v23 = v22 if v22.v155('\n') else v22 + '\n'
    try:
        v156(v23, end='', flush=True)
    except v74:
        v156(v23.v98('ascii', 'replace').v240('ascii'), end='', flush=True)
    v5.v157.v75(parents=True, exist_ok=True)
    with v5.v158('a', encoding='utf-8') as v76:
        v76.v159(v23)

def zs(v24):
    v25 = v160.v77(v24, dtype=v160.v161)
    v26 = v25.v78()
    return (v25 - v25.v212()) / (v26 if v26 > 1e-09 else 1.0)

def main() -> v19:
    v0.v75(parents=True, exist_ok=True)
    v5.v79('', encoding='utf-8')
    v80(f'Stage208 start {v260.v255(v261.v256).v208()}')
    v80('hybrid: discriminative word-level fp reranker for rare words over frozen A')
    v27 = v162.v27('cuda' if v162.v241.v213() else 'cpu')
    v28 = v163.v81(v6)
    v162.v82(v6)
    v29 = v83.v83()
    v84, v85, v86, v87 = v88()
    v30 = v164.v89(v73(v214.v165))
    v31 = v30.v90()
    v32 = v30.v166(v167) or 0
    v33 = v242.v215(v30, v86, v32, v31).v91(v27)
    v34 = v216(v87, v31).v91(v27)
    v34.v92(v162.v217(v1, map_location=v27, weights_only=False)['model'])
    v34.v93()
    for v35 in v34.v94():
        v35.v168(False)
    v36 = v95((v143(v35.v207().v95()) for v35 in v34.v94()))
    v37 = v96(v34, v86, v27)
    v80(f'frozen A loaded ({v83.v83() - v29:.0f}s)')
    with v2.v158('r', encoding='utf-8', errors='ignore') as v76:
        v97 = v76.v169(v7)
    v38 = v30.v98(v97)
    v99, v100 = (v38.v99, v38.v101)
    v39 = {v25: v102 for v102, (v25, v243) in v201(v100)}
    v40 = v103(v17.v170(v97))
    v41 = [v62 for v62, v218 in v40.v219(v10)]
    v42 = {v62: v102 for v102, v62 in v201(v41)}
    v43 = {'common': v41[:v9], 'rare': v41[v9:v10]}
    v80(f"tokens={v222(v99):,} distinct_words={v222(v40):,} common={v222(v43['common'])} rare={v222(v43['rare'])} ({v83.v83() - v29:.0f}s)")
    v44 = []
    for v45 in v17.v104(v97):
        v26, v171 = (v45.v220(), v45.v221())
        if v26 == 0 or v97[v26 - 1] != ' ':
            continue
        v102 = v39.v172(v26 - 1)
        if v102 is None or v102 < v244 + 1:
            continue
        v62 = v45.v173(0)
        v105 = v42.v172(v62)
        if v105 is None:
            continue
        v44.v174((v102, v62, v105))
    v46 = v19(v8 * v222(v44))
    v106, v107 = (v44[:v46], v44[v46:])
    v80(f'word positions: train={v222(v106):,} eval={v222(v107):,} ({v83.v83() - v29:.0f}s)')

    @v162.v111()
    def ctx_states(v108, v109=64):
        v70 = []
        for v110 in v123(0, v222(v108), v109):
            v175 = v108[v110:v110 + v109]
            v140 = [v180(v99[v102 - v244:v102]) for v102 in v175]
            v176 = v162.v223(v140, dtype=v162.v245, device=v27)
            v177 = v176 == v32
            v178 = v34.v224(v33[v176], v176)
            v179 = v34.v179(v178, pad_mask=v177)
            v225, v218, v218 = v34.v225(v178, v177)
            v70.v174(v162.v116([v179, v225], dim=-1)[:, -1].v143())
        return v162.v116(v70, 0)
    v47: v112[v73, v180[v19]] = {}

    def pieces(v62):
        if v62 not in v47:
            v47[v62] = [v257 for v257 in v30.v98(' ' + v62).v99 if v257 != v32][:8] or [v32]
        return v47[v62]

    def a_scores(v113, v114):
        """Use the established span scorer verbatim (truncate-only, no padding)."""
        v115 = v180(v99[v251(0, v113 - v244):v113])
        return [v226(v34, v33, v32, v115, v246(v62), v27) for v62 in v114]
    v48 = v162.v116([v37.v119(v43['common'][v102:v102 + 4096]) for v102 in v123(0, v9, 4096)], 0)

    def surprise(v117):
        return (1.0 - (v37.v119(v117) @ v48.v265).v251(dim=-1).v262).v212().v181()
    v49 = v106[:v11]
    v50 = v118([v102 for v102, v218, v218 in v49])
    v51 = v37.v119([v62 for v218, v62, v218 in v49])
    v80(f'train states {v247(v50.v248)} ({v83.v83() - v29:.0f}s)')

    class Rerank(v120.v52):

        def __init__(v182, v183, v184=256):
            v258().v227()
            v182.v185 = v120.v228(v120.v249(v183, v183), v120.v250(), v120.v249(v183, v184))

        def forward(v182, v125):
            return v232.v229(v182.v185(v125), dim=-1)
    v53 = v230(v50.v231(1)).v91(v27)
    v54 = v162.v186.v121(v53.v94(), lr=v16, weight_decay=0.01)
    v55 = v160.v122([v105 for v218, v218, v105 in v49])
    v56 = None
    for v57 in v123(1, v14 + 1):
        v124 = v162.v187(0, v50.v231(0), (v15,))
        v125 = v50[v124.v91(v27)]
        v126 = v51[v124.v91(v27)]
        v127 = []
        for v128 in v124.v188():
            v105 = v19(v55[v128])
            v194, v195 = (v251(0, v19(v105 * 0.5)), v252(v10 - 1, v251(v19(v105 * 2), v105 + 20)))
            v127.v174([v41[v28.v187(v194, v195)] for v218 in v123(v13)])
        v129 = v37.v119([v62 for v263 in v127 for v62 in v263]).v189(v15, v13, -1)
        v130 = v53(v125)
        v131 = (v130 * v126).v95(-1, keepdim=True)
        v132 = v162.v190('bd,bnd->bn', v130, v129)
        v133 = v232.v191(v162.v116([v131, v132], 1) / 0.07, v162.v233(v15, dtype=v162.v245, device=v27))
        v54.v192(set_to_none=True)
        v133.v193()
        v54.v57()
        v56 = v143(v133) if v56 is None else 0.98 * v56 + 0.02 * v143(v133)
        if v57 % 500 == 0 or v57 == v14:
            v80(f'  step {v57}: rerank_loss~{v56:.3f} ({v83.v83() - v29:.0f}s)')
    v53.v93()

    def make_band(v134, v135):
        v194, v195 = (0, v9) if v134 == 'common' else (v9, v10)
        v136 = [v196 for v196 in v107 if v194 <= v196[2] < v195]
        v28.v137(v136)
        v70 = []
        for v102, v62, v105 in v136[:v135]:
            v234, v235 = (v251(v194, v19(v105 * 0.5)), v252(v195 - 1, v251(v19(v105 * 2), v105 + 20)))
            v114 = [v62]
            while v222(v114) < 4:
                v236 = v41[v28.v187(v234, v235)]
                if v236 not in v114:
                    v114.v174(v236)
            v197 = v180(v123(4))
            v28.v137(v197)
            v198 = [v114[v200] for v200 in v197]
            v70.v174({'tok_i': v102, 'cands': v198, 'gold': v197.v259(0), 'band': v134})
        return v70
    v58 = v199('common', v12) + v199('rare', v12)
    v28.v137(v58)
    v138, v139 = (v58[:v222(v58) // 2], v58[v222(v58) // 2:])
    v80(f'eval items dev={v222(v138)} test={v222(v139)} ({v83.v83() - v29:.0f}s)')

    @v162.v111()
    def annotate(v140):
        v141 = v118([v105['tok_i'] for v105 in v140])
        v142 = v53(v141)
        for v200, v105 in v201(v140):
            v105['a'] = v237(v105['tok_i'], v105['cands'])
            v105['fp'] = [v143(v142[v200] @ v76) for v76 in v37.v119(v105['cands'])]
            v105['sur'] = v238(v105['cands'])
        return v140
    v138, v139 = (v202(v138), v202(v139))
    v59 = v143(v160.v203([v105['sur'] for v105 in v138]))

    def acc(v140, v144):
        v145 = {'common': [0, 0], 'rare': [0, 0], 'all': [0, 0]}
        for v105 in v140:
            v62 = v144(v105)
            v204 = v253(v105['a']) + v62 * v253(v105['fp'])
            v205 = v19(v19(v160.v264(v204)) == v105['gold'])
            for v206 in (v105['band'], 'all'):
                v145[v206][0] += v205
                v145[v206][1] += 1
        return {v200: v24[0] / v24[1] if v24[1] else v143('nan') for v200, v24 in v145.v254()}
    v60 = v146(v139, lambda v105: 0.0)
    v61 = v146(v139, lambda v105: 1000000.0)
    v147, v148 = (0.0, -1.0)
    for v62 in v18:
        v149 = v146(v138, lambda v105, v62=v62: v62)['all']
        if v149 > v148:
            v148, v147 = (v149, v62)
    v63 = v146(v139, lambda v105: v147)
    v150, v151 = ((0.0, 0.0), -1.0)
    for v64 in v18:
        for v152 in v18:
            v149 = v146(v138, lambda v105, v64=v64, v152=v152: v152 if v105['sur'] > v59 else v64)['all']
            if v149 > v151:
                v151, v150 = (v149, (v64, v152))
    v65 = v146(v139, lambda v105: v150[1] if v105['sur'] > v59 else v150[0])
    v66 = v95((v143(v35.v207().v95()) for v35 in v34.v94()))
    v80(f"A only     : all={v60['all']:.3f} common={v60['common']:.3f} rare={v60['rare']:.3f}")
    v80(f"fp only    : all={v61['all']:.3f} common={v61['common']:.3f} rare={v61['rare']:.3f}")
    v80(f"combined w={v147}: all={v63['all']:.3f} common={v63['common']:.3f} rare={v63['rare']:.3f}")
    v80(f"gated {v150}: all={v65['all']:.3f} common={v65['common']:.3f} rare={v65['rare']:.3f}")
    v67 = v65['all'] >= v60['all'] - 0.01
    v68 = v65['rare'] >= v60['rare'] + 0.1
    v69 = v207(v36 - v66) < 0.001
    if v67 and v68 and v69:
        v153 = 'HYBRID_RARE_WIN'
    elif v67 and v65['rare'] >= v60['rare'] + 0.03 and v69:
        v153 = 'HYBRID_RARE_PARTIAL'
    else:
        v153 = 'HYBRID_NO_GAIN'
    v70 = {'timestamp': v260.v255(v261.v256).v208(), 'protocol': 'hybrid_rare_word_head_208', 'overall': v153, 'A_only': v60, 'fp_only': v61, 'combined': {'weight': v147, **v63}, 'gated': {'weights_low_high': v180(v150), 'surprise_threshold': v59, **v65}, 'gates': {'g1_no_degrade': v67, 'g2_rare_win': v68, 'g3_read_only': v69}, 'config': {'N_TRAIN_POS': v222(v49), 'STEPS': v14, 'N_EVAL_BAND': v12, 'V_COMMON': v9, 'V_RARE': v10, 'chance': 0.25}, 'note': 'candidates are frequency-matched within band (no unigram shortcut); the fp side is a discriminative reranker over the frozen context state, not a generative fp predictor (cf. 207)'}
    v3.v79(v239.v209(v70, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v79('\n'.v210(['# Stage208 — hybrid rare-word fp reranker over frozen A', '', f'**Overall:** `{v153}`', '', '| scorer | all | common band | rare band |', '|--------|-----|-------------|-----------|', f"| A only (BPE CE head) | {v60['all']:.3f} | {v60['common']:.3f} | {v60['rare']:.3f} |", f"| fp reranker only | {v61['all']:.3f} | {v61['common']:.3f} | {v61['rare']:.3f} |", f"| combined (w={v147}) | {v63['all']:.3f} | {v63['common']:.3f} | {v63['rare']:.3f} |", f"| gated by fp-surprise {v150} | **{v65['all']:.3f}** | {v65['common']:.3f} | **{v65['rare']:.3f}** |", '', f'- 4-way, frequency-matched candidates within band, chance 0.25; test n={v222(v139)}', f'- gates: no_degrade={v67} rare_win={v68} read_only={v69}']), encoding='utf-8')
    v80(f"[208] {v153} | A all={v60['all']:.3f} rare={v60['rare']:.3f} -> gated all={v65['all']:.3f} rare={v65['rare']:.3f}")
    return 0
if v71 == '__main__':
    raise v154(v211())