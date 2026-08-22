"""
Stage 197 — knowledge edit for the tape: make one-shot fact acquisition a CLEAN
win vs vanilla GPT, and find the one in-space advantage over GPT+RAG.

196 left edit weak (0.417, mean-ctx key, disjoint windows). Here we test stronger
zero-training write mechanisms and the surprise-gated write policy.

Setup: synthetic one-shot facts on a held-out corpus slice. Each fact ties a brand
-new fake object F to an EXISTING subject entity S: WRITE window mentions "S F",
disjoint QUERY window mentions S only. Recall F by anchoring on S (4-way, chance 0.25).

Mechanisms (all zero-train, one frozen fp-space):
  M1 mean-ctx   : key=fp(context words)                         (196 baseline)
  M2 subject    : key=fp(S)                                     (variant 2)
  M3 blend      : key=norm(fp(S)+fp(context))                   (variant 3)
  M4 multi-key  : slots on fp(S) + top-2 rarest context words   (variant 3+4, fp-lexicon anchors)

Controls:
  GPT parametric : logprob(" "+F | query)  -> chance (F unseen, cannot ingest write)
  GPT+RAG        : key=GPT embed(write), query=GPT embed(query) -> nearest rival

Selective write (the in-space advantage over generic RAG):
  stream of facts = few NOVEL (fake F, high fp-lexicon surprise) + many boilerplate (real, low).
  budget B slots. gated = write top-B by fp-lexicon surprise; uniform = write first-B (ingestion order).
  metric = recall of NOVEL facts. A generic store has no in-space novelty prior; the tape gets it free.

Gates:
  G_acquire  best curve mech >= 0.60 and > gpt_param + 0.20   (clean win over vanilla GPT)
  G_vs_rag   best curve mech >= gpt_rag - 0.03                 (not worse than nearest rival)
  G_select   gated_recall > uniform_recall + 0.15             (free write policy from fp-space)

  python _stage197_knowledge_edit.py
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank
from _stage196_tapelm import GptBank, load_gpt
v0 = v18('results')
v1 = v18('checkpoints/stage191_p1_curve.pt')
v2 = v18('data/_wikitext103_train.txt')
v3 = v0 / 'stage197_decision.json'
v4 = v0 / 'stage197_mini.md'
v5 = v0 / '_stage197_log.txt'
v6 = 197
v7 = 150000000
v8 = 60000000
v9 = 4000000
v10 = 2
v11 = 200000
v12 = 100
v13 = 64
v14 = v19.v15

def log(v20: v81) -> None:
    v21 = v20 if v20.v149('\n') else v20 + '\n'
    try:
        v150(v21, end='', flush=True)
    except v82:
        v150(v21.v236('ascii', 'replace').v213('ascii'), end='', flush=True)
    v5.v151.v83(parents=True, exist_ok=True)
    with v5.v152('a', encoding='utf-8') as v32:
        v32.v153(v21)

def build_acq_facts(v22, v23, v24) -> v28[v29]:
    v25 = []
    v26 = [0]

    def next_fake():
        v32 = v24[v26[0] % v146(v24)]
        v26[0] += 1
        return v32
    for v27 in v22:
        if v146(v25) >= v12:
            break
        v84 = v154(v28)
        for v85 in v193.v155(v27):
            v84[v85.v187(1)].v159((v85.v223(), v85.v224()))
        v86 = [(v194, v156) for v194, v156 in v84.v214() if v146(v156) >= 2 and v146(v194) >= 4]
        if not v86:
            continue
        v96, v156 = v86[v23.v195(0, v146(v86) - 1)]
        (v196, v197), (v198, v199) = (v156[0], v156[-1])
        if v198 - v197 < v157:
            continue
        v87 = v158()
        v88 = v27[v141(0, v196 - v157):v197]
        v88 = v88 + ' ' + v87 + ' ' + v27[v197:v215(v146(v27), v197 + 20)]
        v89 = v27[v141(0, v198 - v157):v215(v146(v27), v199 + v157)]
        if v87 in v89 or v96 not in v89 or v146(v225.v132(v89)) < 4:
            continue
        v25.v159({'S': v96, 'F': v87, 'write': v88, 'query': v89})
    return v25

class Mech:
    """A write/query key generator over a fp-bank exposing .fp(list) and .ctx_fp(text, exclude)."""

    def __init__(v90, v58, v45, v91=None, v55=None):
        v90.v58 = v58
        v90.v45 = v45
        v90.v91 = v91
        v90.v55 = v55

    def _rare_anchors(v90, v49, v92, v93=2):
        v94 = [v135 for v135 in v225.v132(v49) if v135 != v92 and v135[0:1].v237() is False]
        v94 = v28(v29.v200(v94))
        if not v94 or v90.v55 is None:
            return []
        v53 = v90.v45.v160(v94)
        v68 = 1.0 - (v53 @ v90.v55.v228).v141(dim=-1).v104
        v95 = v113.v226(v68, v215(v93, v146(v94))).v201.v161()
        return [v94[v54] for v54 in v95]

    def write_keys(v90, v32):
        v96, v162 = (v32['S'], v32['F'])
        if v90.v58 == 'M1_meanctx':
            v93 = v90.v45.v176(v32['write'], exclude=v162)
            return [v93] if v93 is not None else []
        if v90.v58 == 'M2_subject':
            return [v90.v45.v160([v96])[0]]
        if v90.v58 == 'M3_blend':
            v111 = v90.v45.v176(v32['write'], exclude=v162)
            v163 = v90.v45.v160([v96])[0]
            return [v217.v227(v163 + v111, dim=-1)] if v111 is not None else [v163]
        if v90.v58 == 'M4_multikey':
            v164 = [v90.v45.v160([v96])[0]]
            for v165 in v90.v202(v32['write'], v162):
                v164.v159(v90.v45.v160([v165])[0])
            return v164
        raise v166(v90.v58)

    def query_keys(v90, v32):
        v96 = v32['S']
        if v90.v58 == 'M1_meanctx':
            v93 = v90.v45.v176(v32['query'])
            return [v93] if v93 is not None else []
        if v90.v58 == 'M2_subject':
            return [v90.v45.v160([v96])[0]]
        if v90.v58 == 'M3_blend':
            v111 = v90.v45.v176(v32['query'])
            v163 = v90.v45.v160([v96])[0]
            return [v217.v227(v163 + v111, dim=-1)] if v111 is not None else [v163]
        if v90.v58 == 'M4_multikey':
            v164 = [v90.v45.v160([v96])[0]]
            for v165 in v90.v202(v32['query'], None):
                v164.v159(v90.v45.v160([v165])[0])
            return v164
        raise v166(v90.v58)

def score_mech(v30: v97, v25, v23, v31) -> v16:
    v98, v99 = ([], [])
    for v32 in v25:
        for v93 in v30.v167(v32):
            v98.v159(v93)
            v99.v159(v32['F'])
    if not v98:
        return 0.0
    v33 = v113.v100(v98, 0)
    v34 = [v32['F'] for v32 in v25]
    v35 = v36 = 0
    for v32 in v25:
        v101 = v30.v168(v32)
        if not v101:
            continue
        v102 = v113.v100(v101, 0)
        v103 = (v102 @ v33.v228).v141(0).v104
        v105 = [v169 for v169 in v34 if v169 != v32['F']]
        v23.v145(v105)
        v106 = [v32['F']] + v105[:3]
        v107 = v28(v136(v146(v106)))
        v23.v145(v107)
        v108 = [v106[v54] for v54 in v107]
        v109 = v107.v170(0)
        v110 = []
        for v111 in v108:
            v171 = [v54 for v54, v229 in v230(v99) if v229 == v111]
            v110.v159(v16(v103[v171].v141()) if v171 else -1.0)
        v35 += v17(v17(v238.v231(v110)) == v109)
        v36 += 1
    return v35 / v141(1, v36)

@v113.v40()
def gpt_param_edit(v37, v38, v39, v31, v25, v23) -> v16:
    v34 = [v32['F'] for v32 in v25]
    v35 = v36 = 0
    for v32 in v25:
        v105 = [v169 for v169 in v34 if v169 != v32['F']]
        v23.v145(v105)
        v106 = [v32['F']] + v105[:3]
        v107 = v28(v136(v146(v106)))
        v23.v145(v107)
        v108 = [v106[v54] for v54 in v107]
        v109 = v107.v170(0)
        v112 = [v54 for v54 in v38.v236(v32['query']).v216 if v54 != v39][-v13:]
        v110 = []
        for v111 in v108:
            v172 = [v54 for v54 in v38.v236(' ' + v111).v216 if v54 != v39]
            v173 = (v112 + v172)[-v13:]
            v174 = v146(v173) - v146(v172)
            v169 = v113.v203([v173], device=v31)
            v175 = v217.v204(v37(input_ids=v169).v218[0], dim=-1)
            v110.v159(v232((v16(v175[v174 + v93 - 1, v239]) for v93, v239 in v230(v172))) / v141(1, v146(v172)))
        v35 += v17(v17(v238.v231(v110)) == v109)
        v36 += 1
    return v35 / v141(1, v36)

def score_gpt_rag(v41: v114, v25, v23) -> v16:
    v98, v99 = ([], [])
    for v32 in v25:
        v93 = v41.v176(v32['write'], exclude=v32['F'])
        if v93 is not None:
            v98.v159(v93)
            v99.v159(v32['F'])
    if not v98:
        return 0.0
    v33 = v113.v100(v98, 0)
    v34 = [v32['F'] for v32 in v25]
    v35 = v36 = 0
    for v32 in v25:
        v115 = v41.v176(v32['query'])
        if v115 is None:
            continue
        v103 = v33 @ v115
        v105 = [v169 for v169 in v34 if v169 != v32['F']]
        v23.v145(v105)
        v106 = [v32['F']] + v105[:3]
        v107 = v28(v136(v146(v106)))
        v23.v145(v107)
        v108 = [v106[v54] for v54 in v107]
        v109 = v107.v170(0)
        v110 = []
        for v111 in v108:
            v171 = [v54 for v54, v229 in v230(v99) if v229 == v111]
            v110.v159(v16(v103[v171].v141()) if v171 else -1.0)
        v35 += v17(v17(v238.v231(v110)) == v109)
        v36 += 1
    return v35 / v141(1, v36)

def main() -> v17:
    v0.v83(parents=True, exist_ok=True)
    v5.v116('', encoding='utf-8')
    v117(f'Stage197 start {v234.v221(v235.v222).v189()}')
    v117('knowledge edit: anchored/multikey write + surprise-gated policy')
    v31 = v113.v31('cuda' if v113.v219.v205() else 'cpu')
    v23 = v177.v118(v6)
    v42 = v119.v119()
    v120, v121, v122, v123 = v124()
    v38 = v178.v125(v81(v19.v179))
    v43 = v38.v126()
    v39 = v38.v180(v181) or 0
    v44 = v206(v123, v43).v127(v31)
    v44.v128(v113.v207(v1, map_location=v31, weights_only=False)['model'])
    v44.v129()
    v37 = v130(v31)
    v45 = v131(v44, v122, v31)
    v117(f'models loaded ({v119.v119() - v42:.0f}s)')
    with v2.v152('r', encoding='utf-8', errors='ignore') as v32:
        v49 = v32.v182(v7)
    v46 = v49[v8:v8 + v9]
    v47 = [v27.v208() for v27 in v46.v220('\n') if 120 < v146(v27.v208()) < 1000][:1500]
    v48 = v183.v132('[A-Za-z][a-z]+', v49)
    del v49
    v50 = v133(v48)
    v51 = v134(v50.v164())
    v52 = [v135 for v135, v111 in v50.v209(v11) if v111 >= v10]
    v53 = []
    for v54 in v136(0, v146(v52), 4096):
        v53.v159(v45.v160(v52[v54:v54 + 4096]))
    v55 = v113.v137(v53, 0)
    v117(f'mid_paras={v146(v47)} lexicon={v146(v52)} ({v119.v119() - v42:.0f}s)')
    v56 = v138(v51, v23, v12 * 3)
    v25 = v139(v47, v23, v56)
    v117(f'acquisition facts={v146(v25)}')
    v57 = {'acquire': {}}
    for v58 in ('M1_meanctx', 'M2_subject', 'M3_blend', 'M4_multikey'):
        v140 = v184(v97(v58, v45, lex=v55), v25, v177.v118(v6), v31)
        v57['acquire'][v58] = v140
        v117(f'  [{v58}] acc={v140:.3f} ({v119.v119() - v42:.0f}s)')
    v59 = v141(v57['acquire'].v104())
    v60 = v141(v57['acquire'], key=v57['acquire'].v185)
    v61 = v142(v37, v38, v39, v31, v25, v177.v118(v6))
    v41 = v114(v37, v38, v39, v31)
    v62 = v143(v41, v25, v177.v118(v6))
    v57['controls'] = {'gpt_parametric': v61, 'gpt_rag': v62, 'chance': 0.25}
    v117(f'  [gpt_param]={v61:.3f} [gpt_rag]={v62:.3f} ({v119.v119() - v42:.0f}s)')
    v63 = [{'S': v32['S'], 'F': v32['F'], 'write': v32['write'], 'query': v32['query'], 'novel': True} for v32 in v25]
    v64 = []
    for v27 in v47:
        if v146(v64) >= v146(v63) * 5:
            break
        v85 = v193.v186(v27)
        if not v85:
            continue
        v144 = v85.v187(1)
        if v144 in v51 or v50.v185(v144.v233(), 0) > 0:
            v210, v211 = (v141(0, v85.v223() - v157), v215(v146(v27), v85.v224() + v157))
            v64.v159({'S': v144, 'F': v144, 'write': v27[v210:v211], 'query': v27[v210:v211], 'novel': False})
    v65 = v63 + v64
    v23.v145(v65)
    v66 = v146(v63)
    v67 = [v188['F'] for v188 in v65]
    v68 = 1.0 - (v45.v160(v67) @ v55.v228).v141(dim=-1).v104
    v69 = v134(v113.v226(v68, v66).v201.v161())
    v70 = {v65[v54]['F'] for v54 in v69 if v65[v54]['novel']}
    v71 = {v188['F'] for v188 in v65[:v66] if v188['novel']}
    v72 = v146(v63)
    v73 = v146(v70) / v141(1, v72)
    v74 = v146(v71) / v141(1, v72)
    v57['selective'] = {'budget': v66, 'n_novel': v72, 'n_stream': v146(v65), 'gated_novel_kept': v73, 'uniform_novel_kept': v74}
    v117(f'  [selective] gated_kept={v73:.3f} uniform_kept={v74:.3f}')
    v75 = v59 >= 0.6 and v59 > v61 + 0.2
    v76 = v59 >= v62 - 0.03
    v77 = v73 > v74 + 0.15
    if v75 and v77 and v76:
        v147 = 'EDIT_CLEAN_WIN'
    elif v75 and v76:
        v147 = 'EDIT_ACQUIRE_FIXED'
    elif v75:
        v147 = 'EDIT_ACQUIRE_FIXED_RAG_AHEAD'
    else:
        v147 = 'EDIT_STILL_WEAK'
    v78 = {'g_acquire': v75, 'g_vs_rag': v76, 'g_select': v77}
    v79 = {'timestamp': v234.v221(v235.v222).v189(), 'protocol': 'knowledge_edit_197', 'overall': v147, 'best_mechanism': v60, 'best_curve_acc': v59, 'gates': v78, 'axes': v57, 'note': 'zero-training; anchored/multikey write on one frozen fp-space; selective-write policy = fp-lexicon surprise (free, in-space) vs ingestion order'}
    v3.v116(v212.v190(v79, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v116('\n'.v191(['# Stage197 — knowledge edit (anchored write + surprise-gated policy)', '', f'**Overall:** `{v147}`  (best mech: `{v60}` = {v59:.3f})', '', 'acquisition (4-way, chance 0.25):', *[f'- {v93}: {v229:.3f}' for v93, v229 in v57['acquire'].v214()], f'- gpt_parametric: {v61:.3f}  |  gpt+rag: {v62:.3f}', '', f'selective write (budget={v66}, novel={v72}): gated kept {v73:.3f} vs uniform {v74:.3f}', '', f'gates: {v78}']), encoding='utf-8')
    v117(f'[197] {v147} | best {v60}={v59:.3f} gpt={v61:.3f} rag={v62:.3f}')
    return 0
if v80 == '__main__':
    raise v148(v192())