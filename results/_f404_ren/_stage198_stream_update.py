"""
Stage 198 — streaming, beyond-window, update-heavy memory under budget.
The last piece for AND_DISTINCT: does online in-space write beat a rebuilt RAG index?

Regime: a long stream of write-events over time (>> attention window). Some subject
entities S get an initial fake value F1, then a LATER update F2 (overwrite). Many
boilerplate events (real entities) create budget pressure. Memory budget B < stream.
At end, query each fact entity by its subject anchor; want the CURRENT (latest) value.

Systems (all query by subject-anchor fp(S), best mech from 197):
  tape_gated   : online write, admission by fp-lexicon surprise (in-space, free), recency for updates
  rag_uniform  : GPT-embedding store, admission by recency/ingestion order (no in-space novelty prior)
  rag_novelty  : GPT-embedding store, admission by the SAME fp-surprise (bolted-on) — honesty control
  gpt_incontext: vanilla GPT sees only last window of the concatenated stream (beyond-window fail)

Gates:
  G_beyond  tape - gpt_incontext >= 0.30     (beats in-context; needs external memory)
  G_budget  tape - rag_uniform  >= 0.15      (in-space write policy > ingestion order at same budget)
  G_update  tape latest-value acc (updated) >= 0.60
  distinctness type:
    tape > rag_novelty + 0.05  -> STREAM_CAPABILITY_DISTINCT (RAG can't match even when handed the signal)
    else                       -> STREAM_ARCHITECTURAL_DISTINCT (capability reachable, but only by bolting
                                   the tape's own signal onto RAG; tape gets it built-in/free)

  python _stage198_stream_update.py
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank
from _stage196_tapelm import GptBank, load_gpt
v0 = v17('results')
v1 = v17('checkpoints/stage191_p1_curve.pt')
v2 = v17('data/_wikitext103_train.txt')
v3 = v0 / 'stage198_decision.json'
v4 = v0 / 'stage198_mini.md'
v5 = v0 / '_stage198_log.txt'
v6 = 198
v7 = 150000000
v8 = 70000000
v9 = 4000000
v10 = 2
v11 = 200000
v12 = 90
v13 = 0.5
v14 = 4
v15 = 64

def log(v18: v74) -> None:
    v19 = v18 if v18.v148('\n') else v18 + '\n'
    try:
        v149(v19, end='', flush=True)
    except v75:
        v149(v19.v217('ascii', 'replace').v209('ascii'), end='', flush=True)
    v5.v150.v76(parents=True, exist_ok=True)
    with v5.v151('a', encoding='utf-8') as v77:
        v77.v152(v19)

def build_stream(v20, v21, v22):
    """events: list of dicts {t, S, value, text, novel}; facts: {S -> latest value}."""
    v23 = []
    for v24 in v20:
        v78 = v135(v137)
        for v79 in v190.v153(v24):
            v78[v79.v158(1)].v156((v79.v222(), v79.v223()))
        for v65, v154 in v78.v155():
            if v170(v65) >= 4 and v170(v154) >= 1:
                v210, v211 = v154[0]
                v83 = v24[v186(0, v210 - v213):v191(v170(v24), v211 + v213)]
                if v65 in v83 and v170(v230.v107(v83)) >= 4:
                    v23.v156((v65, v83))
    v21.v80(v23)
    v23 = v23[:v12]
    v25 = 0
    v81, v82 = ([], {})
    for v65, v83 in v23:
        v84 = v22[v25]
        v25 += 1
        v81.v156({'S': v65, 'value': v84, 'text': v83.v212(v65, v65 + ' ' + v84, 1), 'novel': True})
        v82[v65] = v84
    v26 = v16(v170(v23) * v13)
    for v65, v83 in v23[:v26]:
        v85 = v22[v25]
        v25 += 1
        v81.v156({'S': v65, 'value': v85, 'text': v83.v212(v65, v65 + ' ' + v85, 1), 'novel': True})
        v82[v65] = v85
    v27 = []
    for v24 in v20:
        if v170(v27) >= v170(v81) * v14:
            break
        v79 = v190.v157(v24)
        if not v79:
            continue
        v86 = v79.v158(1)
        v159, v160 = (v186(0, v79.v222() - v213), v191(v170(v24), v79.v223() + v213))
        v27.v156({'S': v86, 'value': v86, 'text': v24[v159:v160], 'novel': False})
    v28 = v81 + v27
    v21.v80(v28)
    for v87, v88 in v89(v28):
        v88['t'] = v87
    return (v28, v82, v26)

def main() -> v16:
    v0.v76(parents=True, exist_ok=True)
    v5.v90('', encoding='utf-8')
    v91(f'Stage198 start {v227.v220(v228.v221).v187()}')
    v91('streaming beyond-window update-heavy memory under budget')
    v29 = v161.v29('cuda' if v161.v214.v192() else 'cpu')
    v21 = v162.v92(v6)
    v30 = v93.v93()
    v94, v95, v96, v97 = v98()
    v31 = v163.v99(v74(v193.v164))
    v32 = v31.v100()
    v33 = v31.v165(v166) or 0
    v34 = v194(v97, v32).v101(v29)
    v34.v102(v161.v195(v1, map_location=v29, weights_only=False)['model'])
    v34.v103()
    v35 = v104(v29)
    v36 = v105(v34, v96, v29)
    v37 = v106(v35, v31, v33, v29)
    v91(f'models loaded ({v93.v93() - v30:.0f}s)')
    with v2.v151('r', encoding='utf-8', errors='ignore') as v77:
        v41 = v77.v167(v7)
    v38 = v41[v8:v8 + v9]
    v39 = [v24.v196() for v24 in v38.v215('\n') if 120 < v170(v24.v196()) < 1000][:2000]
    v40 = v168.v107('[A-Za-z][a-z]+', v41)
    del v41
    v42 = v108(v40)
    v43 = v109(v42.v169())
    v44 = [v110 for v110, v119 in v42.v197(v11) if v119 >= v10]
    v45 = []
    for v46 in v111(0, v170(v44), 4096):
        v45.v156(v36.v198(v44[v46:v46 + 4096]))
    v47 = v161.v112(v45, 0)
    v91(f'mid_paras={v170(v39)} lexicon={v170(v44)} ({v93.v93() - v30:.0f}s)')
    v22 = v113(v43, v21, v12 * 3)
    v28, v82, v26 = v114(v39, v21, v22)
    v48 = v115((1 for v60 in v28 if v60['novel']))
    v91(f'stream={v170(v28)} fact_events(budget)={v48} updated_entities={v26} query_entities={v170(v82)}')
    v49 = [v60['value'] for v60 in v28]
    v50 = (1.0 - (v36.v198(v49) @ v47.v233).v186(dim=-1).v224).v199().v116()
    for v60, v117 in v118(v28, v50):
        v60['sur'] = v171(v117)

    def key_curve(v60):
        v119 = v36.v172(v60['text'], exclude=v60['value'])
        v120 = v36.v198([v60['S']])[0]
        return v200.v175(v120 + v119, dim=-1) if v119 is not None else v120

    def query_curve(v65):
        return v36.v198([v65])[0]

    def admit(v121):
        if v121 == 'gated':
            v56 = v136(v28, key=lambda v60: -v60['sur'])[:v48]
        elif v121 == 'uniform':
            v56 = v136(v28, key=lambda v60: -v60['t'])[:v48]
        return v56

    def eval_curve(v121):
        v56 = v134(v121)
        v57 = [(v207(v60), v60['value'], v60['t']) for v60 in v56]
        v62 = v63 = 0
        v64 = v137({v60['value'] for v60 in v28 if v60['novel']})
        for v65, v138 in v82.v155():
            v139 = v183(v65)
            v140 = {}
            for v179, v184, v87 in v57:
                v117 = v171(v179 @ v139) + 0.001 * (v87 / v170(v28))
                v140[v184] = v186(v140.v219(v184, -9.9), v117)
            v141 = [v123 for v123 in v64 if v123 != v138]
            v21.v80(v141)
            v142 = [v138] + v141[:3]
            v143 = v137(v111(v170(v142)))
            v21.v80(v143)
            v144 = [v142[v46] for v46 in v143]
            v145 = v143.v185(0)
            v62 += v16(v16(v232.v226([v140.v219(v119, -9.9) for v119 in v144])) == v145)
            v63 += 1
        return v62 / v186(1, v63)

    @v161.v125()
    def gpt_emb(v122):
        v122 = [v46 for v46 in v122 if v46 != v33][-v15:]
        if not v122:
            return None
        v123 = v161.v173([v122], device=v29)
        v124 = v35.v229(input_ids=v123).v216[0].v174(0)
        return v200.v175(v124, dim=-1)

    def gpt_word_fp(v110):
        return v176(v31.v217(' ' + v110).v122)

    def gpt_ctx_fp(v126, v127=None):
        v128 = [v110 for v110 in v230.v107(v126) if v110 != v127][:40]
        if v170(v128) < 3:
            return None
        return v176(v31.v217(' '.v180(v128)).v122)

    def key_rag(v60):
        v120 = v177(v60['S'])
        v119 = v178(v60['text'], exclude=v60['value'])
        if v120 is None:
            return v119
        return v200.v175(v120 + v119, dim=-1) if v119 is not None else v120

    def query_rag(v65):
        return v177(v65)

    def eval_rag(v121):
        v56 = v134(v121)
        v57 = []
        for v60 in v56:
            v179 = v201(v60)
            if v179 is not None:
                v57.v156((v179, v60['value'], v60['t']))
        v62 = v63 = 0
        v64 = v137({v60['value'] for v60 in v28 if v60['novel']})
        for v65, v138 in v82.v155():
            v139 = v202(v65)
            if v139 is None:
                continue
            v140 = {}
            for v179, v184, v87 in v57:
                v117 = v171(v179 @ v139) + 0.001 * (v87 / v170(v28))
                v140[v184] = v186(v140.v219(v184, -9.9), v117)
            v141 = [v123 for v123 in v64 if v123 != v138]
            v21.v80(v141)
            v142 = [v138] + v141[:3]
            v143 = v137(v111(v170(v142)))
            v21.v80(v143)
            v144 = [v142[v46] for v46 in v143]
            v145 = v143.v185(0)
            v62 += v16(v16(v232.v226([v140.v219(v119, -9.9) for v119 in v144])) == v145)
            v63 += 1
        return v62 / v186(1, v63)

    @v161.v125()
    def eval_gpt_incontext():
        v129 = ' '.v180((v60['text'] for v60 in v136(v28, key=lambda v60: v60['t'])))
        v130 = [v46 for v46 in v31.v217(v129).v122 if v46 != v33][-v15 + 6:]
        v62 = v63 = 0
        v64 = v137({v60['value'] for v60 in v28 if v60['novel']})
        for v65, v138 in v82.v155():
            v181 = [v46 for v46 in v31.v217(' ' + v65 + ' is').v122 if v46 != v33]
            v182 = (v130 + v181)[-v15:]
            v141 = [v123 for v123 in v64 if v123 != v138]
            v21.v80(v141)
            v142 = [v138] + v141[:3]
            v143 = v137(v111(v170(v142)))
            v21.v80(v143)
            v144 = [v142[v46] for v46 in v143]
            v145 = v143.v185(0)
            v140 = []
            for v119 in v144:
                v203 = [v46 for v46 in v31.v217(' ' + v119).v122 if v46 != v33]
                v204 = (v182 + v203)[-v15:]
                v205 = v170(v204) - v170(v203)
                v123 = v161.v173([v204], device=v29)
                v206 = v200.v218(v35(input_ids=v123).v225[0], dim=-1)
                v140.v156(v115((v171(v206[v205 + v179 - 1, v234]) for v179, v234 in v89(v203))) / v186(1, v170(v203)))
            v62 += v16(v16(v232.v226(v140)) == v145)
            v63 += 1
        return v62 / v186(1, v63)
    v51 = v131('gated')
    v52 = v132('uniform')
    v53 = v132('gated')
    v54 = v133()
    v91(f'  tape_gated={v51:.3f} rag_uniform={v52:.3f} rag_novelty={v53:.3f} gpt_incontext={v54:.3f} ({v93.v93() - v30:.0f}s)')
    v55 = [v60['S'] for v60 in v28 if v60['novel']]
    v55 = [v65 for v65 in v82 if v55.v231(v65) >= 2 or True][:v26]
    v56 = v134('gated')
    v57 = [(v207(v60), v60['value'], v60['t']) for v60 in v56]
    v58 = v109()
    v59 = v135(v16)
    for v60 in v136(v28, key=lambda v60: v60['t']):
        if v60['novel']:
            v59[v60['S']] += 1
    v61 = [v65 for v65, v119 in v59.v155() if v119 >= 2]
    v62 = v63 = 0
    v64 = v137({v60['value'] for v60 in v28 if v60['novel']})
    for v65 in v61:
        v138 = v82[v65]
        v139 = v183(v65)
        v140 = {}
        for v179, v184, v87 in v57:
            v117 = v171(v179 @ v139) + 0.001 * (v87 / v170(v28))
            v140[v184] = v186(v140.v219(v184, -9.9), v117)
        v141 = [v123 for v123 in v64 if v123 != v138]
        v21.v80(v141)
        v142 = [v138] + v141[:3]
        v143 = v137(v111(v170(v142)))
        v21.v80(v143)
        v144 = [v142[v46] for v46 in v143]
        v145 = v143.v185(0)
        v62 += v16(v16(v232.v226([v140.v219(v119, -9.9) for v119 in v144])) == v145)
        v63 += 1
    v66 = v62 / v186(1, v63)
    v91(f'  tape latest-value acc on updated entities={v66:.3f} (n={v63})')
    v67 = v51 - v54 >= 0.3
    v68 = v51 - v52 >= 0.15
    v69 = v66 >= 0.6
    if v67 and v68 and v69:
        if v51 > v53 + 0.05:
            v146 = 'STREAM_CAPABILITY_DISTINCT'
        else:
            v146 = 'STREAM_ARCHITECTURAL_DISTINCT'
    elif v67 and v69:
        v146 = 'STREAM_BEYOND_WINDOW_ONLY'
    else:
        v146 = 'STREAM_PARTIAL'
    v70 = {'g_beyond': v67, 'g_budget': v68, 'g_update': v69}
    v71 = {'tape_gated': v51, 'rag_uniform': v52, 'rag_novelty_bolted': v53, 'gpt_incontext': v54, 'tape_update_latest': v66, 'budget': v48, 'stream_len': v170(v28), 'updated_entities': v170(v61), 'chance': 0.25}
    v72 = {'timestamp': v227.v220(v228.v221).v187(), 'protocol': 'stream_update_198', 'overall': v146, 'gates': v70, 'results': v71, 'note': 'online in-space write (fp-surprise admission + subject-anchor + recency) vs rebuilt RAG index; rag_novelty control quantifies whether distinctness is capability or architecture'}
    v3.v90(v208.v188(v72, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v90('\n'.v180(['# Stage198 — streaming / beyond-window / update-heavy under budget', '', f'**Overall:** `{v146}`', '', f'- tape (gated, in-space): **{v51:.3f}**', f'- rag_uniform (ingestion order): {v52:.3f}', f'- rag_novelty (RAG + bolted fp-surprise): {v53:.3f}', f'- gpt_incontext (beyond-window): {v54:.3f}  (chance 0.25)', f'- tape latest-value on updated entities: {v66:.3f}', '', f'budget={v48}, stream={v170(v28)}, updated={v170(v61)}', f'gates: {v70}']), encoding='utf-8')
    v91(f'[198] {v146} | tape={v51:.3f} rag_u={v52:.3f} rag_n={v53:.3f} gpt={v54:.3f}')
    return 0
if v73 == '__main__':
    raise v147(v189())