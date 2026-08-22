"""
Stage 204 — W1: substrate robustness to noise / OOV (the axis where BPE is weak BY CONSTRUCTION).

Two fair, space-agnostic tests (ranking-based, so curve-cos and GPT-cos never compared directly):

  A. identity retrieval under char noise: query = typo(w) -> retrieve clean w among a pool.
     reported separately for SEEN (real corpus entities) and OOV (novel pronounceable fakes).
  B. downstream fact recall with NOISY queries: planted facts, subject-anchored keys,
     curve fp memory vs fair GPT+RAG mirror (identical key/query recipe, only encoder differs).
  C. mechanism stat: BPE pieces per word, clean vs noisy (why BPE breaks).

Gates:
  G_idA     curve identity acc at p=0.2 & 0.3 beats GPT by >= +0.10
  G_factB   curve fact recall at p=0.3 beats fair RAG by >= +0.10
  G_degrade curve relative drop (p0 -> p0.3) < RAG relative drop

  python _stage204_noise_robustness.py
"""
from __future__ import annotations
import json
import random
import re
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
from _stage194_fp_fact_memory import ENT_RE, WORD_RE, FpBank
from _stage196_tapelm import load_gpt
v0 = v18('results')
v1 = v18('checkpoints/stage191_p1_curve.pt')
v2 = v18('data/_wikitext103_train.txt')
v3 = v0 / 'stage204_decision.json'
v4 = v0 / 'stage204_mini.md'
v5 = v0 / '_stage204_log.txt'
v6 = 204
v7 = 30000000
v8 = 64
v9 = 700
v10 = 300
v11 = 150
v12 = 400
v13 = 100
v14 = [0.0, 0.1, 0.2, 0.3]

def log(v19: v16) -> None:
    v20 = v19 if v19.v127('\n') else v19 + '\n'
    try:
        v128(v20, end='', flush=True)
    except v63:
        v128(v20.v205('ascii', 'replace').v202('ascii'), end='', flush=True)
    v5.v129.v64(parents=True, exist_ok=True)
    with v5.v130('a', encoding='utf-8') as v65:
        v65.v131(v20)
v15 = 'abcdefghijklmnopqrstuvwxyz'

def noisy(v21: v16, v22: v66, v23: v132.v67) -> v16:
    """Char noise (sub/del/ins/swap). Position 0 is preserved so we measure spelling noise,
    not a capitalization artifact of the tokenizer."""
    if v22 <= 0 or v180(v21) < 4:
        return v21
    v24 = [v21[0]]
    v25 = False
    for v26 in v21[1:]:
        if v23.v132() < v22:
            v25 = True
            v133 = v23.v183(('sub', 'del', 'ins', 'swap'))
            if v133 == 'sub':
                v24.v152(v23.v183(v15))
            elif v133 == 'del':
                continue
            elif v133 == 'ins':
                v24.v152(v26)
                v24.v152(v23.v183(v15))
            elif v180(v24) >= 1:
                v24.v225(v180(v24) - 1, v26)
            else:
                v24.v152(v26)
        else:
            v24.v152(v26)
    if not v25:
        v68 = v23.v134(1, v180(v21))
        v24 = v94(v21[:v68]) + [v23.v183(v15)] + v94(v21[v68 + 1:])
    v27 = ''.v69(v24)
    return v27 if v180(v27) >= 3 else v21

def main() -> v17:
    v0.v64(parents=True, exist_ok=True)
    v5.v70('', encoding='utf-8')
    v71(f'Stage204 start {v221.v211(v222.v212).v177()}')
    v71('W1: substrate robustness to noise/OOV — curve fp vs fair GPT mirror')
    v28 = v135.v28('cuda' if v135.v203.v184() else 'cpu')
    v23 = v132.v67(v6)
    v135.v72(v6)
    v29 = v73.v73()
    v74, v75, v76, v77 = v78()
    v30 = v136.v79(v16(v185.v137))
    v31 = v30.v80()
    v32 = v30.v138(v139) or 0
    v33 = v186(v77, v31).v81(v28)
    v33.v82(v135.v187(v1, map_location=v28, weights_only=False)['model'])
    v33.v83()
    for v22 in v33.v84():
        v22.v140(False)
    v34 = v85(v33, v76, v28)
    v35 = v86(v28)
    v71(f'models loaded ({v73.v73() - v29:.0f}s)')

    @v135.v89()
    def gpt_emb(v87):
        v87 = [v68 for v68 in v87 if v68 != v32][-v8:]
        if not v87:
            return None
        v88 = v35.v223(input_ids=v135.v226([v87], device=v28)).v204[0].v141(0)
        return v188.v142(v88, dim=-1)

    def gpt_word(v21):
        return v143(v30.v205(' ' + v21).v87)

    def gpt_ctx(v90, v91=None):
        v92 = [v189 for v189 in v199.v214(v90) if v189 != v91][:40]
        return v143(v30.v205(' '.v69(v92)).v87) if v180(v92) >= 3 else None
    with v2.v130('r', encoding='utf-8', errors='ignore') as v65:
        v93 = v65.v144(v7)
    v36 = [v22.v145() for v22 in v93.v190('\n') if v180(v22.v145()) > 300]
    v37 = v94(v191.v146((v115.v157(1) for v115 in v194.v215(v93) if v180(v115.v157(1)) >= 5)))
    v23.v95(v37)
    v37 = v37[:v9]
    v38 = [v21 for v21 in v206(v216(v37), v23, v10 + 60) if v180(v21) >= 5][:v10]
    v39 = v37 + v38
    v71(f'pool: seen={v180(v37)} oov={v180(v38)} paras={v180(v36)} ({v73.v73() - v29:.0f}s)')
    v40 = v34.v96(v39)
    v41 = v135.v97([v154(v21) for v21 in v39], 0)
    v71(f'pool encoded both spaces ({v73.v73() - v29:.0f}s)')
    v42 = {'curve_seen': {}, 'gpt_seen': {}, 'curve_oov': {}, 'gpt_oov': {}}
    for v22 in v14:
        v98 = v132.v67(v6 + v17(v22 * 100))
        v99 = [v192(v21, v22, v98) for v21 in v39]
        v100 = v34.v96(v99)
        v101 = (v100 @ v40.v224).v207(dim=-1).v147()
        v102 = v135.v97([v154(v21) for v21 in v99], 0)
        v103 = (v102 @ v41.v224).v207(dim=-1).v147()
        for v148, v149, v150 in (('seen', 0, v180(v37)), ('oov', v180(v37), v180(v39))):
            v151 = v193(v149, v150)
            v42[f'curve_{v148}'][v22] = v208((v17(v101[v68] == v68) for v68 in v151)) / v195(1, v150 - v149)
            v42[f'gpt_{v148}'][v22] = v208((v17(v103[v68] == v68) for v68 in v151)) / v195(1, v150 - v149)
        v71(f"  A p={v22:.1f}: curve seen={v42['curve_seen'][v22]:.3f} oov={v42['curve_oov'][v22]:.3f} | gpt seen={v42['gpt_seen'][v22]:.3f} oov={v42['gpt_oov'][v22]:.3f} ({v73.v73() - v29:.0f}s)")
    v43 = [v21 for v21 in v206(v216(v37), v132.v67(v6 + 7), v11 + 80) if v180(v21) >= 5][:v11]
    v44 = v37[:v11]
    v45 = []
    for v104, v105 in v106(v43, v44):
        v107 = v36[v23.v134(v180(v36))][:200]
        v45.v152({'S': v104, 'value': v105, 'text': f'{v107} {v104} was appointed director of {v105} in 1987 .'})
    v108, v109, v110 = ([], [], [])
    for v46 in v45:
        v111 = v34.v96([v46['S']])[0]
        v112 = v34.v153(v46['text'], exclude=v46['value'])
        v108.v152(v188.v142(v111 + v112, dim=-1) if v112 is not None else v111)
        v113 = v154(v46['S'])
        v114 = v155(v46['text'], exclude=v46['value'])
        v109.v152(v188.v142(v113 + v114, dim=-1) if v114 is not None else v113)
        v110.v152(v46['value'])
    for v47 in v36[:v12]:
        v115 = v194.v156(v47)
        if not v115:
            continue
        v116 = v115.v157(1)
        v149, v150 = (v195(0, v115.v217() - v13), v196(v180(v47), v115.v218() + v13))
        v112 = v34.v153(v47[v149:v150], exclude=v116)
        v114 = v155(v47[v149:v150], exclude=v116)
        if v112 is None or v114 is None:
            continue
        v108.v152(v188.v142(v34.v96([v116])[0] + v112, dim=-1))
        v109.v152(v188.v142(v154(v116) + v114, dim=-1))
        v110.v152(v116)
    v48 = v135.v97(v108, 0)
    v49 = v135.v97(v109, 0)
    v71(f'  memory slots={v180(v110)} (facts={v180(v45)}) ({v73.v73() - v29:.0f}s)')
    v50 = v94(v191.v146(v110))

    def score_recall(v117, v118, v22):
        v98 = v132.v67(v6 + 31 + v17(v22 * 100))
        v119 = 0
        for v46 in v45:
            v158 = v118(v192(v46['S'], v22, v98))
            v159 = {}
            for v197, v27 in v198((v117 @ v158).v147()):
                v159[v110[v197]] = v195(v159.v219(v110[v197], -9.9), v27)
            v160 = [v189 for v189 in v50 if v189 != v46['value']]
            v98.v95(v160)
            v161 = [v46['value']] + v160[:3]
            v162 = v94(v193(4))
            v98.v95(v162)
            v163 = [v161[v68] for v68 in v162]
            v119 += v17(v17(v210.v207([v159.v219(v26, -9.9) for v26 in v163])) == v162.v220(0))
        return v119 / v180(v45)
    v51 = {'curve': {}, 'rag': {}}
    for v22 in v14:
        v51['curve'][v22] = v164(v48, lambda v21: v34.v96([v21])[0], v22)
        v51['rag'][v22] = v164(v49, lambda v21: v154(v21), v22)
        v71(f"  B p={v22:.1f}: curve={v51['curve'][v22]:.3f} rag={v51['rag'][v22]:.3f} ({v73.v73() - v29:.0f}s)")
    v52 = [0.0, 0.2, 0.3]
    v53 = 400
    v54 = 8

    def noise_text(v120, v22, v98):
        return v199.v165(lambda v115: v192(v115.v157(0), v22, v98), v120)
    v55 = {'curve': {}, 'rag': {}}
    for v22 in v52:
        v98 = v132.v67(v6 + 71 + v17(v22 * 100))
        v166, v167, v168 = ([], [], [])
        for v46 in v45:
            v169 = v192(v46['S'], v22, v98)
            v170 = v200(v46['text'].v209(v46['S'], v169), v22, v98)
            v112 = v34.v153(v170, exclude=v46['value'])
            v114 = v155(v170, exclude=v46['value'])
            v111, v113 = (v34.v96([v169])[0], v154(v169))
            v166.v152(v188.v142(v111 + v112, dim=-1) if v112 is not None else v111)
            v167.v152(v188.v142(v113 + v114, dim=-1) if v114 is not None else v113)
            v168.v152(v46['value'])
        for v47 in v36[:v53]:
            v115 = v194.v156(v47)
            if not v115:
                continue
            v116 = v115.v157(1)
            v149, v150 = (v195(0, v115.v217() - v13), v196(v180(v47), v115.v218() + v13))
            v170 = v200(v47[v149:v150], v22, v98)
            v112 = v34.v153(v170, exclude=v116)
            v114 = v155(v170, exclude=v116)
            if v112 is None or v114 is None:
                continue
            v166.v152(v188.v142(v34.v96([v192(v116, v22, v98)])[0] + v112, dim=-1))
            v167.v152(v188.v142(v154(v192(v116, v22, v98)) + v114, dim=-1))
            v168.v152(v116)
        v171, v172 = (v135.v97(v166, 0), v135.v97(v167, 0))
        v121 = v94(v191.v146(v168))

        def score2(v117, v118, v173):
            v174 = v132.v67(v6 + 91 + v173 + v17(v22 * 100))
            v119 = 0
            for v46 in v45:
                v158 = v118(v192(v46['S'], v22, v174))
                v159 = {}
                for v197, v27 in v198((v117 @ v158).v147()):
                    v159[v168[v197]] = v195(v159.v219(v168[v197], -9.9), v27)
                v160 = [v189 for v189 in v121 if v189 != v46['value']]
                v174.v95(v160)
                v161 = [v46['value']] + v160[:v54 - 1]
                v162 = v94(v193(v180(v161)))
                v174.v95(v162)
                v163 = [v161[v68] for v68 in v162]
                v119 += v17(v17(v210.v207([v159.v219(v26, -9.9) for v26 in v163])) == v162.v220(0))
            return v119 / v180(v45)
        v55['curve'][v22] = v175(v171, lambda v21: v34.v96([v21])[0], 0)
        v55['rag'][v22] = v175(v172, lambda v21: v154(v21), 1)
        v71(f"  B2 p={v22:.1f} (noisy corpus+query, {v54}-way): curve={v55['curve'][v22]:.3f} rag={v55['rag'][v22]:.3f} ({v73.v73() - v29:.0f}s)")
    v56 = {}
    for v22 in v14:
        v98 = v132.v67(v6 + 5 + v17(v22 * 100))
        v122 = [v180(v30.v205(' ' + v192(v21, v22, v98)).v87) for v21 in v39]
        v56[v22] = v66(v210.v141(v122))
    v71(f"  C BPE pieces/word: {[f'{v22}:{v56[v22]:.2f}' for v22 in v14]}")

    def drop(v123):
        return (v123[0.0] - v123[0.3]) / v195(1e-06, v123[0.0])
    v57 = v124((v42['curve_seen'][v22] >= v42['gpt_seen'][v22] + 0.1 for v22 in (0.2, 0.3)))
    v58 = v124((v42['curve_oov'][v22] >= v42['gpt_oov'][v22] + 0.1 for v22 in (0.2, 0.3)))
    v59 = v51['curve'][0.3] >= v51['rag'][0.3] + 0.1
    v60 = v176(v51['curve']) < v176(v51['rag'])
    v61 = v55['curve'][0.3] >= v55['rag'][0.3] + 0.1
    if (v57 or v58) and v59 and v60 and v61:
        v125 = 'NOISE_ROBUST_WIN'
    elif v57 or v58 or v59:
        v125 = 'NOISE_ROBUST_PARTIAL'
    else:
        v125 = 'NOISE_ROBUST_NO'
    v24 = {'timestamp': v221.v211(v222.v212).v177(), 'protocol': 'noise_oov_robustness_204', 'overall': v125, 'A_identity_retrieval': {v178: {v16(v22): v179 for v22, v179 in v123.v213()} for v178, v123 in v42.v213()}, 'B_fact_recall_noisy_query': {v178: {v16(v22): v179 for v22, v179 in v123.v213()} for v178, v123 in v51.v213()}, 'B_relative_drop_p0_to_p03': {'curve': v176(v51['curve']), 'rag': v176(v51['rag'])}, 'B2_hardened_noisy_corpus_and_query_8way': {v178: {v16(v22): v179 for v22, v179 in v123.v213()} for v178, v123 in v55.v213()}, 'C_bpe_pieces_per_word': {v16(v22): v179 for v22, v179 in v56.v213()}, 'gates': {'g_id_seen': v57, 'g_id_oov': v58, 'g_fact': v59, 'g_degrade': v60, 'g_hard_B2': v61}, 'pool': {'seen': v180(v37), 'oov': v180(v38)}, 'slots': v180(v110), 'chance_B': 0.25, 'note': 'rank-based metrics only (no cross-space cosine comparison); GPT+RAG uses the identical subject-anchored key/query recipe, only the encoder differs'}
    v3.v70(v201.v181(v24, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v70('\n'.v69(['# Stage204 — W1 noise/OOV robustness', '', f'**Overall:** `{v125}`', '', '| noise p | A id seen curve/gpt | A id OOV curve/gpt | B recall curve/rag | BPE pieces/word |', '|---------|---------------------|--------------------|--------------------|-----------------|'] + [f"| {v22:.1f} | {v42['curve_seen'][v22]:.3f} / {v42['gpt_seen'][v22]:.3f} | {v42['curve_oov'][v22]:.3f} / {v42['gpt_oov'][v22]:.3f} | {v51['curve'][v22]:.3f} / {v51['rag'][v22]:.3f} | {v56[v22]:.2f} |" for v22 in v14] + ['', f"- relative drop p0→p0.3 (fact recall): curve {v176(v51['curve']):.3f} vs rag {v176(v51['rag']):.3f}", '- **B2 hardened (noise in stored corpus AND query, 8-way, chance 0.125):** ' + ' · '.v69((f"p={v22:.1f} curve {v55['curve'][v22]:.3f} / rag {v55['rag'][v22]:.3f}" for v22 in v52)), f'- gates: id_seen={v57} id_oov={v58} fact={v59} degrade={v60} hard_B2={v61}', f'- slots={v180(v110)}, chance B=0.25']), encoding='utf-8')
    v71(f"[204] {v125} | A0.3 curve seen={v42['curve_seen'][0.3]:.3f} vs gpt={v42['gpt_seen'][0.3]:.3f} | B0.3 curve={v51['curve'][0.3]:.3f} vs rag={v51['rag'][0.3]:.3f}")
    return 0
if v62 == '__main__':
    raise v126(v182())