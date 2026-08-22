"""
Stage 200 — do we OPERATE on facts (in-space vectors) or just READ them (index)?

User's question: our facts live inside the model's own fp-space; RAG's live in an
external text index. Does in-space storage give a real compositional/chaining
advantage over index lookup?

Test: planted k-hop chains of NOVEL entities (A0->A1->...->Ak), each edge stated in a
distinctive sentence spread across the corpus (beyond the attention window). Answer:
given A0, follow k hops to Ak. Memory also holds thousands of real-edge distractors.

Systems (fair subject-anchoring for both):
  curve_string : per hop  q=fp(current) -> argmax over keys -> value STRING -> next   (fp encoder)
  curve_vector : per hop  q=value FP (vector-native) -> argmax -> value FP -> ...      (never decodes;
                 pure operable-vector chaining — the thing RAG structurally cannot do)
  rag          : per hop  q=gpt_word_embed(current) -> argmax -> value STRING -> next  (GPT encoder = index)
  gpt_incontext: sees only last window of the concatenated edge sentences (beyond-window control)

Also: binding one-shot 2-hop (old SOTE edge_fp) — answer a 2-hop query with composed vectors,
no sequential re-retrieval.

Gates:
  G_external   curve & rag at k>=2 >> gpt_incontext        (external memory required)
  G_chain      curve_string at k=3 >= 0.50
  G_vs_rag     curve_string(k=3) - rag(k=3) >= 0.10         (fp is a better chainer -> operable win)
  G_vectornative curve_vector at k=3 >= 0.50                (chaining survives with NO decoding)

  python _stage200_fact_composition.py
"""
from __future__ import annotations
import json
import random
import time
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
from _stage196_tapelm import load_gpt
v0 = v15('results')
v1 = v15('checkpoints/stage191_p1_curve.pt')
v2 = v15('data/_wikitext103_train.txt')
v3 = v0 / 'stage200_decision.json'
v4 = v0 / 'stage200_mini.md'
v5 = v0 / '_stage200_log.txt'
v6 = 200
v7 = 150000000
v8 = 80000000
v9 = 4000000
v10 = 60
v11 = 3
v12 = 6000
v13 = 64

def log(v16: v59) -> None:
    v17 = v16 if v16.v118('\n') else v16 + '\n'
    try:
        v119(v17, end='', flush=True)
    except v60:
        v119(v17.v188('ascii', 'replace').v170('ascii'), end='', flush=True)
    v5.v120.v61(parents=True, exist_ok=True)
    with v5.v121('a', encoding='utf-8') as v62:
        v62.v122(v17)

def main() -> v14:
    v0.v61(parents=True, exist_ok=True)
    v5.v63('', encoding='utf-8')
    v64(f'Stage200 start {v186.v180(v187.v181).v148()}')
    v64('fact composition: in-space chaining vs index-RAG vs vanilla GPT')
    v18 = v84.v18('cuda' if v84.v171.v152() else 'cpu')
    v19 = v123.v65(v6)
    v20 = v66.v66()
    v67, v68, v69, v70 = v71()
    v21 = v124.v72(v59(v153.v125))
    v22 = v21.v73()
    v23 = v21.v126(v127) or 0
    v24 = v154(v70, v22).v74(v18)
    v24.v75(v84.v155(v1, map_location=v18, weights_only=False)['model'])
    v24.v76()
    v25 = v77(v18)
    v26 = v78(v24, v69, v18)
    v64(f'models loaded ({v66.v66() - v20:.0f}s)')

    @v84.v83()
    def gpt_word(v79: v59) -> v84.v27:
        v80 = [v92 for v92 in v21.v188(' ' + v79).v80 if v92 != v23][:v13]
        if not v80:
            v80 = [v23]
        v81 = v84.v128([v80], device=v18)
        v82 = v25.v189(input_ids=v81).v172[0].v129(0)
        return v156.v130(v82, dim=-1)
    with v2.v121('r', encoding='utf-8', errors='ignore') as v62:
        v30 = v62.v131(v7)
    v28 = v30[v8:v8 + v9]
    v29 = [v31.v157() for v31 in v28.v173('\n') if 120 < v146(v31.v157()) < 1000][:3000]
    del v30
    v64(f'mid_paras={v146(v29)} ({v66.v66() - v20:.0f}s)')
    v85, v86, v87, v88 = ([], [], [], [])

    def add_slot(v89, v90):
        v85.v132(v26.v98([v89])[0])
        v86.v132(v90)
        v87.v132(v26.v98([v90])[0])
        v88.v132(v104(v89))
    for v31 in v29:
        if v146(v86) >= v12:
            break
        v91 = v97(v159.v134((v141.v182(1) for v141 in v193.v190(v31))))
        for v92 in v94(v146(v91) - 1):
            if v146(v86) >= v12:
                break
            if v146(v91[v92]) >= 4 and v146(v91[v92 + 1]) >= 4:
                v158(v91[v92], v91[v92 + 1])
    v64(f'distractor slots={v146(v86)} ({v66.v66() - v20:.0f}s)')
    v32 = v93(v133(), v19, v10 * (v11 + 1) + 50)
    v33 = []
    v34 = 0
    for v35 in v94(v10):
        v95 = v32[v34:v34 + v11 + 1]
        v34 += v11 + 1
        if v146(v95) < v11 + 1:
            break
        v33.v132(v95)
    v36 = []
    for v37 in v33:
        for v92 in v94(v146(v37) - 1):
            v89, v90 = (v37[v92], v37[v92 + 1])
            v158(v89, v90)
            v36.v132((v26.v98([v89])[0], v26.v98([v90])[0]))
    v38 = v84.v96(v85, 0)
    v39 = v84.v96(v88, 0)
    v40 = v84.v96(v87, 0)
    v64(f'total slots={v146(v86)} chains={v146(v33)} ({v66.v66() - v20:.0f}s)')
    v41 = v97(v159.v134([v114 for v37 in v33 for v114 in v37]))
    v42 = v26.v98(v41)

    def chain_string(v99, v100):
        v101 = {}
        for v102 in v94(1, v11 + 1):
            v113 = 0
            for v37 in v33:
                v160 = v37[0]
                for v35 in v94(v102):
                    v174 = v100(v160)
                    v175 = v14((v99 @ v174).v184())
                    v160 = v86[v175]
                v113 += v14(v160 == v37[v102])
            v101[v102] = v113 / v146(v33)
        return v101

    def chain_vector():
        v101 = {}
        for v102 in v94(1, v11 + 1):
            v113 = 0
            for v37 in v33:
                v90 = v26.v98([v37[0]])[0]
                for v35 in v94(v102):
                    v175 = v14((v38 @ v90).v184())
                    v90 = v40[v175]
                v161 = v41[v14((v42 @ v90).v184())]
                v113 += v14(v161 == v37[v102])
            v101[v102] = v113 / v146(v33)
        return v101
    v43 = v103(v38, lambda v79: v26.v98([v79])[0])
    v44 = v103(v39, v104)
    v45 = v105()
    v64(f'curve_string={v43}')
    v64(f'rag={v44}')
    v64(f'curve_vector={v45}')
    v46 = v84.v96([v156.v130(v176 * v177, dim=-1) for v176, v177 in v36], 0)
    v47 = 0
    v48 = 0
    for v37 in v33:
        v135, v136 = (v37[0], v37[2])
        v106 = [v136] + [v41[v19.v183(0, v146(v41) - 1)] for v35 in v94(3)]
        v107 = v97(v94(4))
        v19.v137(v107)
        v108 = [v106[v92] for v92 in v107]
        v109 = v107.v138(0)
        v110 = v26.v98([v135])[0]
        v111 = []
        for v112 in v108:
            v139 = v26.v98([v112])[0]
            v140 = -9.9
            for v141 in v41:
                v162 = v26.v98([v141])[0]
                v163 = v178((v46 @ v156.v130(v110 * v162, dim=-1)).v142())
                v164 = v178((v46 @ v156.v130(v162 * v139, dim=-1)).v142())
                v140 = v142(v140, v163 + v164)
            v111.v132(v140)
        v47 += v14(v14(v191.v184(v111)) == v109)
        v48 += 1
    v49 = v47 / v142(1, v48)
    v64(f'binding one-shot 2-hop acc={v49:.3f} (chance 0.25)')

    @v84.v83()
    def gpt_incontext_2hop():
        v113 = v114 = 0
        for v37 in v33:
            v143 = [v92 for v92 in v21.v188(' ' + v37[0] + ' leads to').v80 if v92 != v23]
            v144 = v143[-v13:]
            v106 = [v37[2]] + [v41[v19.v183(0, v146(v41) - 1)] for v35 in v94(3)]
            v107 = v97(v94(4))
            v19.v137(v107)
            v108 = [v106[v92] for v92 in v107]
            v109 = v107.v138(0)
            v111 = []
            for v145 in v108:
                v165 = [v92 for v92 in v21.v188(' ' + v145).v80 if v92 != v23]
                v166 = (v144 + v165)[-v13:]
                v167 = v146(v166) - v146(v165)
                v81 = v84.v128([v166], device=v18)
                v168 = v156.v179(v25(input_ids=v81).v185[0], dim=-1)
                v111.v132(v192((v178(v168[v167 + v194 - 1, v195]) for v194, v195 in v196(v165))) / v142(1, v146(v165)))
            v113 += v14(v14(v191.v184(v111)) == v109)
            v114 += 1
        return v113 / v142(1, v114)
    v50 = v115()
    v64(f'gpt_incontext 2-hop acc={v50:.3f}')
    v51 = v43[2] - v50 >= 0.3 and v44[2] - v50 >= 0.0
    v52 = v43[v11] >= 0.5
    v53 = v43[v11] - v44[v11] >= 0.1
    v54 = v45[v11] >= 0.5
    if v52 and v51 and v53 and v54:
        v116 = 'COMPOSE_OPERABLE_WIN'
    elif v52 and v51 and v54:
        v116 = 'COMPOSE_CHAINS_BUT_RAG_PARITY'
    elif v52 and v51:
        v116 = 'COMPOSE_CHAINS_STRING_ONLY'
    else:
        v116 = 'COMPOSE_WEAK'
    v55 = {'g_external': v51, 'g_chain': v52, 'g_vs_rag': v53, 'g_vectornative': v54}
    v56 = {'curve_string': v43, 'curve_vector': v45, 'rag_index': v44, 'binding_2hop': v49, 'gpt_incontext_2hop': v50, 'slots': v146(v86), 'chains': v146(v33), 'K': v11, 'chance_exact': v147(1.0 / v146(v41), 4)}
    v57 = {'timestamp': v186.v180(v187.v181).v148(), 'protocol': 'fact_composition_200', 'overall': v116, 'gates': v55, 'results': v56, 'note': 'in-space fp chaining (string & vector-native) + binding one-shot vs GPT-embedding index-RAG and vanilla in-context GPT; tests whether facts are OPERABLE vectors vs READ-by-index documents'}
    v3.v63(v169.v149(v57, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v63('\n'.v150(['# Stage200 — fact composition: operate vs read-by-index', '', f'**Overall:** `{v116}`', '', f'- curve_string k1/k2/k3: {v43[1]:.2f} / {v43[2]:.2f} / {v43[3]:.2f}', f'- curve_vector (no decode) k1/k2/k3: {v45[1]:.2f} / {v45[2]:.2f} / {v45[3]:.2f}', f'- rag_index k1/k2/k3: {v44[1]:.2f} / {v44[2]:.2f} / {v44[3]:.2f}', f'- binding one-shot 2-hop: {v49:.3f} (chance 0.25)', f'- gpt_incontext 2-hop: {v50:.3f}', '', f'slots={v146(v86)} chains={v146(v33)} K={v11}', f'gates: {v55}']), encoding='utf-8')
    v64(f'[200] {v116} | curve_str k3={v43[3]:.2f} vec k3={v45[3]:.2f} rag k3={v44[3]:.2f} gpt={v50:.2f}')
    return 0
if v58 == '__main__':
    raise v117(v151())