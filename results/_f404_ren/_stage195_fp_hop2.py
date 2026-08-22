"""
Stage 195 — hop2 chains: retest of old SOTE's death place (collision-bound @ d64).

Task: question context = paragraph P1 around entity B (B masked). Gold answer C
co-occurs with B in ANOTHER paragraph P2 (and never appears in P1). Distractors
never co-occur with B anywhere. Direct hop1 retrieval cannot solve this — the
chain is required: ctx(P1) → B → B's other contexts → C.

Two mechanisms tested:
  CHAIN   : B̂ = retrieve(q1); score(c) = max cos(key[B̂ slots ∉ P1], key[c slots])
  BINDING : edge memory e=norm(fp(A)⊙fp(B)) per co-occurring pair (old SOTE edge_fp);
            score(c) = max cos(norm(fp(B̂)⊙fp(c)), E)   ← the d64-collision victim, now d256

Gates:
  G1 chain acc >= 0.50 (chance 0.25)
  G2 direct hop1 scoring <= 0.35 (no shortcut — else items are broken)
  report: binding acc, oracle-B chain acc (upper bound)

  python _stage195_fp_hop2.py
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
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, FpBank
v0 = v14('results')
v1 = v14('data')
v2 = v14('checkpoints/stage191_p1_curve.pt')
v3 = v14('data/_wikitext103_train.txt')
v4 = v0 / 'stage195_decision.json'
v5 = v0 / 'stage195_mini.md'
v6 = v0 / '_stage195_log.txt'
v7 = 195
v8 = 150000000
v9 = 3000000
v10 = 120
v11 = 4
v12 = 30

def log(v15: v52) -> None:
    v16 = v15 if v15.v102('\n') else v15 + '\n'
    try:
        v103(v16, end='', flush=True)
    except v53:
        v103(v16.v177('ascii', 'replace').v159('ascii'), end='', flush=True)
    v6.v104.v54(parents=True, exist_ok=True)
    with v6.v105('a', encoding='utf-8') as v55:
        v55.v106(v16)

def main() -> v13:
    v0.v54(parents=True, exist_ok=True)
    v6.v56('', encoding='utf-8')
    v57(f'Stage195 start {v175.v170(v176.v171).v128()}')
    v57('hop2 chains + edge-binding retest (old SOTE failure, now d256)')
    v17 = v107.v17('cuda' if v107.v160.v131() else 'cpu')
    v18 = v108.v58(v7)
    v19 = v59.v59()
    v60, v61, v62, v63 = v64()
    v20 = v109.v65(v52(v132.v110))
    v21 = v20.v66()
    v22 = v133(v63, v21).v67(v17)
    v22.v68(v107.v134(v2, map_location=v17, weights_only=False)['model'])
    v22.v69()
    v23 = v70(v22, v62, v17)
    with v3.v105('r', encoding='utf-8', errors='ignore') as v55:
        v26 = v55.v111(v8)
    v24 = v26[-v9:]
    v25 = [v76.v135() for v76 in v24.v161('\n') if 120 < v92(v76.v135()) < 1000][:1200]
    del v26
    v57(f'paras={v92(v25)} ({v59.v59() - v19:.0f}s)')
    v71, v72, v73 = ([], [], [])
    v27: v74[v74[v52]] = []
    for v75, v76 in v77(v25):
        v78 = []
        for v79 in v136.v112(v76):
            v113 = v79.v137(1)
            v138, v139 = (v157(0, v79.v178() - v172), v162(v92(v76), v79.v179() + v172))
            v114 = v23.v122(v76[v138:v139], exclude=v113)
            if v114 is not None:
                v71.v115(v114)
                v72.v115(v113)
                v73.v115(v75)
                v78.v115(v113)
        v27.v115(v74(v82.v163(v78)))
    v28 = v107.v80(v71, 0)
    v29 = v116.v81(v73)
    v57(f'memory slots={v92(v72)} ({v59.v59() - v19:.0f}s)')
    v30: v82[v52, v74[v13]] = {}
    for v83, v84 in v77(v72):
        v30.v164(v84, []).v115(v83)
    v31: v82[v52, v140[v13]] = {}
    for v83, v84 in v77(v72):
        v31.v164(v84, v140()).v117(v73[v83])
    v32: v82[v52, v140[v52]] = {}
    for v33 in v27:
        for v85 in v33:
            v32.v164(v85, v140()).v141((v86 for v86 in v33 if v86 != v85))
    v34 = v74(v30.v71())
    v35 = []
    v36 = [v86 for v86 in v34 if v92(v31[v86]) >= 2]
    v18.v87(v36)
    for v37 in v36:
        if v92(v35) >= v10:
            break
        v88 = v118(v31[v37])
        v18.v87(v88)
        v89 = False
        for v90 in v88:
            for v119 in v88:
                if v90 == v119:
                    continue
                v142 = v140(v27[v90])
                v143 = [v97 for v97 in v27[v119] if v97 != v37 and v97 not in v142 and (v97 in v30)]
                if not v143:
                    continue
                v144 = v143[v18.v173(0, v92(v143) - 1)]
                v145 = []
                for v146 in (6, 15, 40):
                    for v165 in v120(v157(0, v119 - v146), v162(v92(v25), v119 + v146 + 1)):
                        if v165 in (v90, v119):
                            continue
                        for v174 in v27[v165]:
                            if v174 not in (v37, v144) and v174 not in v32.v154(v37, v140()) and (v174 not in v142) and (v174 not in v27[v119]) and (v174 not in v145):
                                v145.v115(v174)
                    if v92(v145) >= v11 - 1:
                        break
                v18.v87(v145)
                v147 = v145[:v11 - 1]
                if v92(v147) < v11 - 1:
                    continue
                v79 = None
                for v148 in v136.v112(v25[v90]):
                    if v148.v137(1) == v37:
                        v79 = v148
                        break
                if v79 is None:
                    continue
                v138, v139 = (v157(0, v79.v178() - v172), v162(v92(v25[v90]), v79.v179() + v172))
                v149 = [v144] + v147
                v150 = v74(v120(v11))
                v18.v87(v150)
                v35.v115({'ctx_text': v25[v90][v138:v139], 'B': v37, 'p1': v90, 'cands': [v149[v114] for v114 in v150], 'gold_idx': v150.v180(0)})
                v89 = True
                break
            if v89:
                break
    v57(f'hop2 items={v92(v35)}')
    v38 = []
    for v33 in v27:
        v91 = 0
        for v83 in v120(v92(v33)):
            for v121 in v120(v83 + 1, v92(v33)):
                if v91 >= v12:
                    break
                v151 = v23.v155([v33[v83]])[0]
                v152 = v23.v155([v33[v121]])[0]
                v38.v115(v168.v156(v151 * v152, dim=-1))
                v91 += 1
    v39 = v107.v80(v38, 0)
    v57(f'edge memory={v92(v38)} pairs')
    v40 = v92(v35)
    v41 = v42 = v43 = v44 = 0
    v45 = 0
    for v46 in v35:
        v93 = v23.v122(v46['ctx_text'], exclude=v46['B'])
        if v93 is None:
            v40 -= 1
            continue
        v94 = v28 @ v93
        v95 = v72[v13(v94.v166())]
        v45 += v13(v95 == v46['B'])

        def chain_score(v123):
            v124 = []
            v125 = [v83 for v83 in v30.v154(v123, []) if v73[v83] != v46['p1']]
            for v97 in v46['cands']:
                v126 = v30.v154(v97, [])
                if not v125 or not v126:
                    v124.v115(-1.0)
                    continue
                v153 = v28[v125] @ v28[v126].v167
                v124.v115(v169(v153.v157()))
            return v124
        v41 += v13(v13(v116.v166(v181(v95))) == v46['gold_idx'])
        v44 += v13(v13(v116.v166(v181(v46['B']))) == v46['gold_idx'])
        v96 = []
        for v97 in v46['cands']:
            v126 = v30.v154(v97, [])
            v96.v115(v169((v28[v126] @ v93).v157()) if v126 else -1.0)
        v42 += v13(v13(v116.v166(v96)) == v46['gold_idx'])
        v98 = v23.v155([v95])[0]
        v99 = []
        for v97 in v46['cands']:
            v86 = v168.v156(v98 * v23.v155([v97])[0], dim=-1)
            v99.v115(v169((v39 @ v86).v157()))
        v43 += v13(v13(v116.v166(v99)) == v46['gold_idx'])
    v47 = {'n': v40, 'hop1_B_acc': v45 / v157(1, v40), 'chain_acc': v41 / v157(1, v40), 'chain_acc_oracle_B': v44 / v157(1, v40), 'direct_shortcut_acc': v42 / v157(1, v40), 'binding_acc': v43 / v157(1, v40), 'chance': 1 / v11}
    v57(v158.v127(v47, indent=2))
    v48 = v47['chain_acc'] >= 0.5
    v49 = v47['direct_shortcut_acc'] <= 0.35
    if v48 and v49:
        v100 = 'HOP2_CHAIN_YES'
    elif v47['chain_acc'] >= 0.35 and v49:
        v100 = 'HOP2_CHAIN_WEAK'
    elif not v49:
        v100 = 'HOP2_ITEMS_LEAKY'
    else:
        v100 = 'HOP2_CHAIN_NO'
    if v47['binding_acc'] >= 0.5 and v49:
        v100 += '+BINDING_YES'
    v50 = {'timestamp': v175.v170(v176.v171).v128(), 'protocol': 'fp_hop2_195', 'overall': v100, 'results': v47, 'slots': v92(v72), 'edges': v92(v38), 'note': 'zero training; chain = two cosine hops over slot memory; binding = old SOTE edge_fp at d256'}
    v4.v56(v158.v127(v50, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v56('\n'.v129(['# Stage195 — hop2 chains + edge binding (d256 retest)', '', f'**Overall:** `{v100}`', '', f"- chain acc={v47['chain_acc']:.3f} (oracle-B {v47['chain_acc_oracle_B']:.3f}), chance 0.25", f"- direct shortcut={v47['direct_shortcut_acc']:.3f} (must be low)", f"- binding (old edge_fp)={v47['binding_acc']:.3f}", f"- hop1 B retrieval={v47['hop1_B_acc']:.3f}, n={v47['n']}", '']), encoding='utf-8')
    v57(f'[195] {v100}')
    return 0
if v51 == '__main__':
    raise v101(v130())