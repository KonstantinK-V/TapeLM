"""
Stage 211 — beyond-window: internal slow tape vs endpoint (pre-publish frontier).

Cross-doc: fact written while reading document A; query on document B (subject S never in B).
Compare retrieval of latest fake value F among 4 candidates.

Methods:
  internal_tape  — surprise-gated (key=ctx_fp local, value=slow_t) logged during read(A)
  endpoint_only  — slow_T after read(A) only (no addressable slots)
  external_slots — explicit fp slots from A (194/198 style, reference ceiling)
  gpt_incontext  — GPT sees only B tail + cue (beyond-window structural fail)
  doc_id_oracle  — key = doc embedding (metadata control; breaks under wrong id + noise)

Gates: see results/pre_publish_frontier.md §211

  python _stage211_internal_tape.py
"""
from __future__ import annotations
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank, build_memory
from _stage196_tapelm import load_gpt
from _stage204_noise_robustness import noisy
v0 = v20('results')
v1 = v20('checkpoints/stage191_p1_curve.pt')
v2 = v20('data/_wikitext103_train.txt')
v3 = v0 / 'stage211_decision.json'
v4 = v0 / 'stage211_mini.md'
v5 = v0 / '_stage211_log.txt'
v6 = 211
v7 = 150000000
v8 = 70000000
v9 = 4000000
v10 = 100
v11 = 48
v12 = 0.22
v13 = 0.3
v14 = 0.25

def log(v21: v87) -> None:
    v22 = v21 if v21.v160('\n') else v21 + '\n'
    try:
        v161(v22, end='', flush=True)
    except v88:
        v161(v22.v230('ascii', 'replace').v220('ascii'), end='', flush=True)
    v5.v162.v89(parents=True, exist_ok=True)
    with v5.v163('a', encoding='utf-8') as v90:
        v90.v164(v22)

def encode_ids(v23: v91, v24: v19, v25: v87) -> v26[v19]:
    return [v115 for v115 in v23.v230(v25).v29 if v115 != v24][-v207:]

@v46.v35()
def forward_slow(v27, v28, v29: v26[v19], v30):
    v31 = v46.v92([v29], dtype=v46.v165, device=v30)
    v32 = v31 == v15
    v33 = v27.v93(v28[v31], v31)
    v34 = v27.v34(v33, pad_mask=v32)
    v94, v95, v96 = v27.v94(v33, v32)
    return (v94[0], v95[0], v34[0])
v15 = 0

@v46.v35()
def build_internal_tape(v27, v28, v36: v97, v25: v87, v30):
    v29 = v98(v18, v15, v25)
    if v166(v29) < 8:
        return (None, None, None)
    v94, v99, v96 = v100(v27, v28, v29, v30)
    v37 = v94.v101(0)
    v38 = v46.v167(v99, k=v168(v11, v37)).v39
    v102, v103 = ([], [])
    for v40 in v38.v104():
        if v50(v99[v40]) < v12:
            continue
        v105 = v19(v40 / v183(1, v37 - 1) * v183(1, v166(v25) - v208))
        v106 = v168(v166(v25), v105 + v208)
        v107 = v25[v105:v106]
        v108 = v36.v169(v107)
        if v108 is None:
            continue
        v102.v170(v108)
        v103.v170(v171.v111(v94[v40], dim=-1))
    if not v102:
        return (v94[-1], None, None)
    return (v94[-1], v46.v128(v102), v46.v128(v103))

@v46.v35()
def retrieve_soft(v41: v46.v16, v42: v46.v16, v43: v46.v16) -> v46.v16:
    v44 = v46.v109(v41 @ v43 * 10.0, dim=0)
    v45 = (v44.v221(0) @ v42).v110(0)
    return v171.v111(v45, dim=-1)

@v46.v35()
def score_candidates(v47: v46.v16, v36: v97, v48: v26[v87]) -> v26[v50]:
    v49 = v36.v112(v48)
    return [v50(v47 @ v49[v115]) for v115 in v176(v166(v48))]

def four_way_acc(v51: v26[v50], v52: v19) -> v17:
    return v19(v222.v209(v51)) == v52

def build_tasks(v53: v26[v87], v54: v26[v87], v55: v172.v113):
    v56 = []
    v57 = 0
    v58 = [v68 for v68 in v53 if 120 < v166(v68) < 900]
    v55.v114(v58)
    for v115, v68 in v116(v58):
        if v166(v56) >= v10 or v57 >= v166(v54):
            break
        v117 = v210.v173(v68)
        if not v117:
            continue
        v118 = v117.v174(1)
        if v166(v118) < 4:
            continue
        v119 = v54[v57]
        v57 += 1
        v105, v106 = (v183(0, v117.v231() - v208), v168(v166(v68), v117.v232() + v208))
        v120 = v68[v105:v106].v175(v118, v118 + ' ' + v119, 1)
        v121 = None
        for v96 in v176(40):
            v177 = v58[v55.v223(0, v166(v58) - 1)]
            if v118 not in v177:
                v121 = v177[:v168(600, v166(v177))]
                break
        if v121 is None:
            continue
        v56.v170({'S': v118, 'value': v119, 'text_a': v120, 'text_b': v121, 'doc_id': v115})
    return v56

def eval_internal(v56, v36, v27, v28, v30, v55, v59: v50=0.0):
    v60 = 0
    v61 = v26({v40['value'] for v40 in v56})
    v62 = v172.v113(v6 + 99)
    for v40 in v56:
        v122 = v40['text_a']
        if v59 > 0:
            v122 = ''.v205((v237(v133, v59, v62) if v133.v236() else v133 for v133 in v122))
        v178, v41, v42 = v179(v27, v28, v36, v122, v30)
        v43 = v36.v112([v40['S']])[0]
        if v41 is not None and v42 is not None:
            v47 = v188(v41, v42, v43)
        else:
            v47 = v171.v111(v178 + v43, dim=-1)
        v123 = [v31 for v31 in v61 if v31 != v40['value']]
        v55.v114(v123)
        v48 = [v40['value']] + v123[:3]
        v124 = v26(v176(4))
        v55.v114(v124)
        v125 = [v48[v115] for v115 in v124]
        v52 = v124.v180(0)
        v126 = v181(v47, v36, v125)
        v60 += v182(v126, v52)
    return v60 / v183(1, v166(v56))

def eval_endpoint(v56, v36, v27, v28, v30, v55, v59: v50=0.0):
    v60 = 0
    v61 = v26({v40['value'] for v40 in v56})
    v62 = v172.v113(v6 + 100)
    for v40 in v56:
        v122 = v40['text_a']
        if v59 > 0:
            v122 = ''.v205((v237(v133, v59, v62) if v133.v236() else v133 for v133 in v122))
        v178, v96, v96 = v179(v27, v28, v36, v122, v30)
        v43 = v36.v112([v40['S']])[0]
        v47 = v171.v111(v178 + v43, dim=-1)
        v123 = [v31 for v31 in v61 if v31 != v40['value']]
        v55.v114(v123)
        v48 = [v40['value']] + v123[:3]
        v124 = v26(v176(4))
        v55.v114(v124)
        v125 = [v48[v115] for v115 in v124]
        v52 = v124.v180(0)
        v126 = v181(v47, v36, v125)
        v60 += v182(v126, v52)
    return v60 / v183(1, v166(v56))

def eval_external(v56, v36, v55, v59: v50=0.0):
    v60 = 0
    v61 = v26({v40['value'] for v40 in v56})
    v62 = v172.v113(v6 + 101)
    for v40 in v56:
        v122 = v40['text_a']
        if v59 > 0:
            v122 = ''.v205((v237(v133, v59, v62) if v133.v236() else v133 for v133 in v122))
        v41, v103 = v184([v122], v36, '211')
        v43 = v36.v112([v40['S']])[0]
        v127 = {}
        for v115, v45 in v116(v103):
            v185 = v50(v41[v115] @ v43)
            v127[v45] = v183(v127.v187(v45, -9.0), v185)
        v123 = [v31 for v31 in v61 if v31 != v40['value']]
        v55.v114(v123)
        v48 = [v40['value']] + v123[:3]
        v124 = v26(v176(4))
        v55.v114(v124)
        v125 = [v48[v115] for v115 in v124]
        v52 = v124.v180(0)
        v126 = [v127.v187(v133, -9.0) for v133 in v125]
        v60 += v182(v126, v52)
    return v60 / v183(1, v166(v56))

def eval_doc_id_oracle(v56, v36, v63: v186[v19, v46.v16], v55, v64: v17):
    """Global doc-id keyed store; query uses doc embedding (wrong id on noisy test)."""
    v102, v103 = ([], [])
    for v40 in v56:
        v102.v170(v63[v40['doc_id']])
        v103.v170(v36.v112([v40['value']])[0])
    v41 = v46.v128(v102)
    v42 = v46.v128(v103)
    v60 = 0
    v61 = v26({v40['value'] for v40 in v56})
    for v40 in v56:
        v129 = v40['doc_id'] + (999 if v64 else 0)
        v43 = v63.v187(v129, v63[v40['doc_id']])
        v47 = v188(v41, v42, v43)
        v123 = [v31 for v31 in v61 if v31 != v40['value']]
        v55.v114(v123)
        v48 = [v40['value']] + v123[:3]
        v124 = v26(v176(4))
        v55.v114(v124)
        v125 = [v48[v115] for v115 in v124]
        v52 = v124.v180(0)
        v126 = v181(v47, v36, v125)
        v60 += v182(v126, v52)
    return v60 / v183(1, v166(v56))

@v46.v35()
def eval_gpt_ic(v56, v65, v23, v24, v30, v55):
    v60 = 0
    v61 = v26({v40['value'] for v40 in v56})
    for v40 in v56:
        v130 = v98(v23, v24, v40['text_b'])
        v131 = [v115 for v115 in v23.v230(' ' + v40['S'] + ' is').v29 if v115 != v24]
        v132 = (v130 + v131)[-v207:]
        v123 = [v31 for v31 in v61 if v31 != v40['value']]
        v55.v114(v123)
        v48 = [v40['value']] + v123[:3]
        v124 = v26(v176(4))
        v55.v114(v124)
        v125 = [v48[v115] for v115 in v124]
        v52 = v124.v180(0)
        v126 = []
        for v133 in v125:
            v189 = [v115 for v115 in v23.v230(' ' + v133).v29 if v115 != v24]
            v190 = (v132 + v189)[-v207:]
            v191 = v166(v190) - v166(v189)
            v31 = v46.v92([v190], device=v30)
            v192 = v171.v211(v65(input_ids=v31).v224[0], dim=-1)
            v126.v170(v147((v50(v192[v191 + v108 - 1, v238]) for v108, v238 in v116(v189))) / v183(1, v166(v189)))
        v60 += v182(v126, v52)
    return v60 / v183(1, v166(v56))
v18 = None

def main() -> v19:
    global pad_id_global, tok_global
    v0.v89(parents=True, exist_ok=True)
    v5.v134('', encoding='utf-8')
    v135(f'Stage211 start {v234.v228(v235.v229).v203()}')
    v30 = v46.v30('cuda' if v46.v225.v212() else 'cpu')
    v55 = v172.v113(v6)
    v66 = v136.v136()
    v137, v138, v139, v140 = v141()
    v23 = v91.v142(v87(v213.v193))
    v18 = v23
    v42 = v23.v143()
    v24 = v23.v194(v195) or 0
    v15 = v24
    v28 = v226.v214(v23, v139, v24, v42).v144(v30)
    v27 = v215(v140, v42).v144(v30)
    v27.v145(v46.v216(v1, map_location=v30, weights_only=False)['model'])
    v27.v146()
    v67 = v147((v50(v68.v202().v147()) for v68 in v27.v233.v148()))
    for v68 in v27.v148():
        v68.v196(False)
    v65 = v149(v30)
    v36 = v97(v27, v139, v30)
    v135(f'P1 frozen ({v136.v136() - v66:.0f}s)')
    with v2.v163('r', encoding='utf-8', errors='ignore') as v90:
        v25 = v90.v197(v7)
    v69 = v25[v8:v8 + v9]
    v53 = [v68.v198() for v68 in v69.v217('\n') if 120 < v166(v68.v198()) < 1000]
    v70 = v199.v150('[A-Za-z][a-z]+', v25)
    del v25
    v54 = v151(v200(v70), v55, v10 + 20)
    v56 = v152(v53, v54, v55)
    v135(f'tasks={v166(v56)} ({v136.v136() - v66:.0f}s)')
    v63 = {}
    for v40 in v56:
        if v40['doc_id'] not in v63:
            v63[v40['doc_id']] = v171.v111(v46.v201(v27.v227.v218 // 2, device=v30), dim=-1)
    v63[v40['doc_id'] + 999] = v171.v111(v46.v201(v27.v227.v218 // 2, device=v30), dim=-1)
    v71 = v172.v113(v6 + 1)
    v72 = v153(v56, v36, v27, v28, v30, v71, 0.0)
    v73 = v154(v56, v36, v27, v28, v30, v172.v113(v6 + 2), 0.0)
    v74 = v155(v56, v36, v172.v113(v6 + 3), 0.0)
    v75 = v156(v56, v65, v23, v24, v30, v172.v113(v6 + 4))
    v135(f'clean internal={v72:.3f} endpoint={v73:.3f} external={v74:.3f} gpt_ic={v75:.3f}')
    v76 = v153(v56, v36, v27, v28, v30, v172.v113(v6 + 5), v13)
    v77 = v157(v56, v36, v63, v172.v113(v6 + 6), wrong_id=True)
    v135(f'noisy p={v13} internal={v76:.3f} doc_id_wrong={v77:.3f}')
    v78 = v147((v50(v68.v202().v147()) for v68 in v27.v233.v148()))
    v79 = v202(v67 - v78) < 0.001
    v80 = v72 - v73 >= 0.25
    v81 = v72 >= v74 - 0.1
    v82 = v75 <= 0.35
    v83 = v76 >= v77 - 0.05
    v84 = v79
    if v80 and v81 and v82 and v83 and v84:
        v158 = 'THESIS_YES'
    elif v80 and v81 and v84:
        v158 = 'ENGINEERING_ONLY'
    else:
        v158 = 'THESIS_NO_AT_SCALE'
    v85 = {'timestamp': v234.v228(v235.v229).v203(), 'protocol': 'internal_slow_tape_211', 'overall': v158, 'clean': {'internal_tape': v72, 'endpoint_only': v73, 'external_slots': v74, 'gpt_incontext': v75}, 'noisy': {'p': v13, 'internal_tape': v76, 'doc_id_oracle_wrong_id': v77}, 'gates': {'g1_beats_endpoint': v80, 'g2_near_external': v81, 'g3_beyond_window': v82, 'g4_not_metadata': v83, 'g5_anticf': v84}, 'n_tasks': v166(v56), 'chance': v14, 'anticf_encoder_frozen': v79}
    v3.v134(v219.v204(v85, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v134('\n'.v205(['# Stage211 — internal slow tape vs endpoint (cross-doc)', '', f'**Overall:** `{v158}`', '', f'- clean: internal **{v72:.3f}** endpoint {v73:.3f} external {v74:.3f} gpt_ic {v75:.3f}', f'- noisy: internal **{v76:.3f}** doc_id_wrong {v77:.3f} (p={v13})', f"- gates: {v85['gates']}"]), encoding='utf-8')
    v135(f'[211] {v158} | internal={v72:.3f} endpoint={v73:.3f} ext={v74:.3f} ({v136.v136() - v66:.0f}s)')
    return 0
if v86 == '__main__':
    raise v159(v206())