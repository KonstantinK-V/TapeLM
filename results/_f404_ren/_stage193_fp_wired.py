"""
Stage 193 — wire FP-lexicon surprise into the head (the real crossover).

192 proved the signal (AUC 0.991, read-only). Now make it visible in behavior:
  - frozen night P1 curve-XL (nothing in the backbone trains — G1 safe by design)
  - per-position lexical surprise s_t: at each WORD boundary, s = 1 - max_cos(fp(word), lexicon)
  - logits_t / T_t,  T_t = 1 + softplus(w * s_t + b) — only w,b train (2 params, CE)

Gates:
  G1 next_tok drop <= 0.02 vs raw P1 (temperature must not hurt)
  G3 entropy after FAKE entity > after real (the never-passed one, now with real signal)

  python _stage193_fp_wired.py
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
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, sample_windows
v0 = v18('results')
v1 = v18('data')
v2 = v18('checkpoints/stage191_p1_curve.pt')
v3 = v18('data/_wikitext103_train.txt')
v4 = v0 / 'stage193_decision.json'
v5 = v0 / 'stage193_mini.md'
v6 = v0 / '_stage193_log.txt'
v7 = v1 / 'stage191_exam_v3.jsonl'
v8 = 193
v9 = v19.v10
v11 = 200000
v12 = 2
v13 = 600
v14 = 0.05
v15 = 16
v16 = 150

def log(v20: v64) -> None:
    v21 = v20 if v20.v131('\n') else v20 + '\n'
    try:
        v132(v21, end='', flush=True)
    except v65:
        v132(v21.v221('ascii', 'replace').v188('ascii'), end='', flush=True)
    v6.v133.v66(parents=True, exist_ok=True)
    with v6.v134('a', encoding='utf-8') as v67:
        v67.v135(v21)

class LexSurprise:
    """word → 1 - max_cos(fp, lexicon); fp = normalized frozen arc_enc(word chars)."""

    def __init__(v68, v44, v69, v47, v39):
        v68.v44 = v44
        v68.v69 = v69
        v68.v39 = v39
        v68.v70: v136[v64, v34] = {}
        v89('  encoding lexicon fps …')
        v71 = []
        for v72 in v82(0, v80(v47), 4096):
            v71.v175(v68.v184(v47[v72:v72 + 4096]))
        v68.v73 = v77.v137(v71, 0)

    @v77.v76()
    def _fp(v68, v74: v33[v64]) -> v77.v22:
        v75 = v77.v138(v80(v74), 1, v9, dtype=v77.v180)
        for v72, v87 in v139(v74):
            for v181, v182 in v139(v87[:v9]):
                v75[v72, 0, v181] = v68.v69.v204(v182, 0)
        return v183.v140(v68.v44.v205(v75.v98(v68.v39))[:, 0], dim=-1)

    @v77.v76()
    def surprise(v68, v32: v33[v64]) -> v33[v34]:
        v78 = [v87 for v87 in v32 if v87 not in v68.v70]
        if v78:
            v71 = v68.v184(v78)
            v141 = (v71 @ v68.v73.v107).v200(dim=-1).v142
            for v87, v145 in v147(v78, v141):
                v68.v70[v87] = v34(1.0 - v145)
        return [v68.v70[v87] for v87 in v32]

def position_surprise(v23: v33[v17], v24: v33[v64], v25: v79, v26: v17) -> v33[v34]:
    """s_t > 0 at the last piece of each alphabetic word."""
    v27 = v80(v23)
    v28 = [0.0] * v27
    v29 = ''
    v30 = 0
    v32, v81 = ([], [])
    for v31 in v82(v27 + 1):
        v45 = v24[v23[v31]] if v31 < v27 and v23[v31] != v26 else ' '
        v83 = v31 == v27 or v45.v185(' ') or (v45 and (not v45[0].v217()))
        if v83 and v29:
            v143 = v191.v206('[A-Za-z][a-z]+$', v29) or v191.v206('[A-Za-z]+$', v29)
            if v143 and v80(v143.v218(0)) >= 4:
                v32.v175(v143.v218(0))
                v81.v175(v31 - 1)
            v29 = ''
        if v31 < v27 and v23[v31] != v26:
            v29 = v29 + v45 if not v83 else v45
    if v32:
        v84 = v25.v144(v32)
        for v145, v146 in v147(v81, v84):
            v28[v145] = v146
    return v28

def gen_fakes(v35, v36, v27):
    v85, v86 = ('bcdfghklmnprstvz', 'aeiou')
    v37 = []
    v38 = 0
    while v80(v37) < v27 and v38 < 20000:
        v38 += 1
        v87 = ''.v178((v36.v227(v85) + v36.v227(v86) + (v36.v227(v85) if v36.v149() < 0.4 else '') for v173 in v82(v36.v201(2, 4)))).v148()
        if v87.v193() not in v35 and v87 not in v35 and (5 <= v80(v87) <= 12) and (v87 not in v37):
            v37.v175(v87)
    return v37

def main() -> v17:
    v0.v66(parents=True, exist_ok=True)
    v6.v88('', encoding='utf-8')
    v89(f'Stage193 start {v222.v215(v223.v216).v176()}')
    v89('Wire FP-lexicon surprise into head temperature (frozen backbone, 2 trainable params)')
    v39 = v77.v39('cuda' if v77.v207.v186() else 'cpu')
    v36 = v149.v90(v8)
    v40 = v91.v91()
    v92, v93, v69, v94 = v95()
    v41 = v150.v96(v64(v19.v151))
    v42 = v41.v97()
    v26 = v41.v152(v153) or 0
    import _stage185_tape_read as s185
    v43 = v208.v187(v41, v69, v26, v42).v98(v39)
    v24 = [v41.v188([v72], skip_special_tokens=False) or '' for v72 in v82(v42)]
    v44 = v189(v94, v42).v98(v39)
    v44.v99(v77.v190(v2, map_location=v39, weights_only=False)['model'])
    v44.v100()
    for v45 in v44.v101():
        v45.v154(False)
    v89('reading corpus words …')
    with v3.v134('r', encoding='utf-8', errors='ignore') as v67:
        v48 = v67.v155(150000000)
    v46 = v102(v191.v156('[A-Za-z][a-z]+', v48))
    v35 = v157(v46.v192()) | {v87.v193() for v87 in v46}
    v47 = [v87 for v87, v182 in v46.v194(v11) if v182 >= v12]
    del v48
    v25 = v79(v44, v69, v47, v39)
    v89(f'  lexicon={v80(v47)} ({v91.v91() - v40:.0f}s)')
    v49 = v158.v103(v77.v159(10.0, device=v39))
    v50 = v158.v103(v77.v159(-2.0, device=v39))
    v51 = v77.v160.v104([v49, v50], lr=v14)

    def s_tensor(v105: v77.v22) -> v77.v22:
        v75 = v105.v161()
        return v77.v159([v209(v210, v24, v25, v26) for v210 in v75], device=v39)

    def logits_with_T(v105: v77.v22):
        v106 = v105 == v26
        with v77.v76():
            v163, v173, v173 = v44.v195(v43[v105], v106, ids=v105)
        v28 = v162(v105)
        v107 = 1.0 + v183.v219(v49 * v28 + v50).v196(-1)
        return (v163 / v107, v106)
    v89('training temperature (w,b) …')
    for v52 in v82(1, v13 + 1):
        v105 = v211(v92, v93, v15, v36, v26).v98(v39)
        v163, v106 = v164(v105)
        v108 = v105[:, 1:]
        v109 = ~v106[:, :-1] & ~v106[:, 1:]
        v110 = v183.v165(v163[:, :-1][v109], v108[v109])
        v51.v166(set_to_none=True)
        v110.v167()
        v51.v52()
        if v52 % 150 == 0 or v52 == v13:
            v89(f'  step {v52}: ce={v34(v110):.4f} w={v34(v49):.3f} b={v34(v50):.3f}')
    v53 = [v197.v168(v169) for v169 in v7.v220(encoding='utf-8').v198()]

    @v77.v76()
    def span_lp(v111, v112, v113: v170):
        v114 = (v111 + v112)[-v212:]
        v115 = v80(v114) - v80(v112)
        v116 = v77.v159([v114], dtype=v77.v180, device=v39)
        if v113:
            v163, v173 = v164(v116)
            v163 = v163[0]
        else:
            v106 = v116 == v26
            v163 = v44.v195(v43[v116], v106, ids=v116)[0][0]
        v117 = v183.v171(v163, dim=-1)
        return v199((v34(v117[v115 + v224 - 1, v225]) for v224, v225 in v139(v112))) / v200(1, v80(v112))

    def next_tok_acc(v113: v170, v27=200):
        v118 = [v59 for v59 in v53 if v59['type'] == 'next_tok'][:v27]
        v119 = 0
        for v59 in v118:
            v172 = [v213(v59['ctx_ids'], v182, v113) for v182 in v59['cand_ids']]
            v119 += v17(v17(v214.v226(v172)) == v59['gold_idx'])
        return v119 / v80(v118)
    v54 = v120(False)
    v55 = v120(True)
    v89(f'next_tok raw={v54:.3f} temp={v55:.3f}')
    v56 = v121(v35, v36, v16)
    v57 = [v59 for v59 in v53 if v59['type'] == 'entity'][:100]
    v58 = v149.v90(3)

    @v77.v76()
    def entropy_after(v111, v122):
        v114 = (v111 + v122)[-v212:]
        v116 = v77.v159([v114], dtype=v77.v180, device=v39)
        v163, v173 = v164(v116)
        v45 = v183.v174(v163[0, v80(v114) - 1], dim=-1)
        return v34(-(v45 * v77.v89(v45 + 1e-09)).v199())
    v123, v124 = ([], [])
    for v59 in v57:
        v125 = v59['cand_ids'][v59['gold_idx']]
        v126 = v56[v58.v201(0, v80(v56) - 1)]
        v127 = [v72 for v72 in v41.v221(' ' + v126).v105 if v72 != v26]
        v123.v175(v202(v59['ctx_ids'], v125))
        v124.v175(v202(v59['ctx_ids'], v127))
    v128, v129 = (v34(v214.v203(v123)), v34(v214.v203(v124)))
    v60 = v55 >= v54 - 0.02
    v61 = v129 > v128
    v62 = 'FP_WIRED_YES' if v60 and v61 else 'FP_WIRED_PARTIAL_' + ''.v178((v27 for v27, v119 in (('1', v60), ('3', v61)) if not v119))
    v37 = {'timestamp': v222.v215(v223.v216).v176(), 'protocol': 'fp_wired_193', 'overall': v62, 'gates': {'G1': {'next_tok_raw': v54, 'next_tok_temp': v55, 'ok': v60}, 'G3': {'entropy_real': v128, 'entropy_fake': v129, 'ok': v61}}, 'temp': {'w': v34(v49), 'b': v34(v50)}, 'note': 'frozen backbone; only (w,b) trained; surprise = FP-lexicon, read-only'}
    v4.v88(v197.v177(v37, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v88('\n'.v178(['# Stage193 — FP-lexicon wired into head temperature', '', f'**Overall:** `{v62}`', '', f'- G1: next_tok raw={v54:.3f} → temp={v55:.3f} → {v60}', f'- G3: entropy real={v128:.3f} fake={v129:.3f} → {v61}', f'- learned w={v34(v49):.3f} b={v34(v50):.3f}', '']), encoding='utf-8')
    v89(f'[193] {v62} | G1 {v54:.3f}->{v55:.3f} | G3 {v128:.3f} vs {v129:.3f} | w={v34(v49):.2f} b={v34(v50):.2f}')
    return 0
if v63 == '__main__':
    raise v130(v179())