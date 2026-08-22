"""
Stage 196 — TapeLM: assemble variant A into ONE inference stack on the frozen
191-P1 curve encoder, and enforce the anti-clone (distinguishability) gate.

Single object `TapeLM` over frozen P1 does all three on ONE held-out slice (exam v3):
  1. GENERATE  : next-piece CE log-prob  (parity = entry ticket, must TIE GPT)
  2. RECALL    : FP episodic fact memory  (194)   — win axis
  3. CALIBRATE : FP-lexicon lexical surprise (192) — win axis (OOD "I don't know")
  4. EDIT      : one-shot knowledge write at read time — win axis (GPT structurally can't)

Anti-clone rule (plan North-star v2): parity is the entry ticket, NOT a win.
Win only counts where BPE-GPT is structurally weak, and there curve must BEAT, not tie.
Nearest real rival = GPT+RAG (same retrieval math, GPT's own embedding as key).

Controls:
  - GPT-XL (191-P2): next_tok parity, parametric entity recall, native BPE OOD surprisal.
  - GPT+RAG        : identical retrieval over same read paras, keyed by GPT mean-pool embedding.

Verdicts:
  parity_hold        = |curve_next - gpt_next| <= 0.03
  recall_win         = curve_recall >= 0.50 and curve_recall > gpt_param + 0.15
  recall_beats_rag   = curve_recall >= gpt_rag - 0.03            (not worse than RAG)
  calib_win          = curve_lexAUC > gpt_bpeAUC and curve_lexAUC >= 0.80
  edit_win           = curve_edit >= 0.50 and curve_edit > gpt_edit + 0.20
  overall:
    parity_hold & recall_win & recall_beats_rag & calib_win & edit_win -> TAPELM_COMPOSES_AND_DISTINCT
    parity_hold & recall_win & calib_win & edit_win & !recall_beats_rag -> TAPELM_DISTINCT_RECALL_RAG_EQUIV
    parity_hold & !(any strict beat over BOTH gpt & gpt+rag)           -> TAPELM_CLONE_RISK
    else                                                              -> TAPELM_PARTIAL

  python _stage196_tapelm.py
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
from transformers import GPT2Config, GPT2LMHeadModel
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, score_items, span_logprob_x
from _stage192_fp_lexicon import auc, gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank, build_memory, score_entity_items
v0 = v20('results')
v1 = v20('checkpoints/stage191_p1_curve.pt')
v2 = v20('checkpoints/stage191_p2_gpt.pt')
v3 = v20('data/_wikitext103_train.txt')
v4 = v20('data/stage191_exam_v3.jsonl')
v5 = v0 / 'stage196_decision.json'
v6 = v0 / 'stage196_mini.md'
v7 = v0 / '_stage196_log.txt'
v8 = 196
v9 = 150000000
v10 = 3000000
v11 = 2
v12 = 200000
v13 = 150
v14 = 60
v15 = v21.v16

def log(v22: v89) -> None:
    v23 = v22 if v22.v174('\n') else v22 + '\n'
    try:
        v175(v23, end='', flush=True)
    except v90:
        v175(v23.v253('ascii', 'replace').v240('ascii'), end='', flush=True)
    v7.v176.v91(parents=True, exist_ok=True)
    with v7.v177('a', encoding='utf-8') as v92:
        v92.v178(v23)

class TapeLM:
    """One inference object over the frozen P1 curve encoder: generate / recall / calibrate / edit."""

    def __init__(v93, v44, v94, v32, v43, v42, v24):
        v93.v44 = v44
        v93.v32 = v32
        v93.v43 = v43
        v93.v42 = v42
        v93.v24 = v24
        v93.v95 = v179(v44, v94, v24)
        v93.v96 = None
        v93.v97 = None
        v93.v98 = None
        v93.v56 = None

    def next_tok_score(v93, v99, v100) -> v17:
        return v180(v93.v44, v93.v43, v93.v42, v99, v100, v93.v24)

    def read(v93, v101, v102='read'):
        v93.v96, v93.v97 = v154(v101, v93.v95, v102)

    @v134.v37()
    def build_lexicon(v93, v56):
        v93.v56 = v56
        v103 = []
        for v104 in v181(0, v194(v56), 4096):
            v103.v187(v93.v95.v183(v56[v104:v104 + 4096]))
        v93.v98 = v134.v182(v103, 0)

    @v134.v37()
    def lex_surprise(v93, v33) -> v38.v18:
        v103 = v93.v95.v183(v33)
        v105 = (v103 @ v93.v98.v251).v197(dim=-1).v106
        return (1.0 - v105).v241().v184()

    @v134.v37()
    def write_fact(v93, v107, v108, v109=None):
        v110 = v93.v95.v185(v107, exclude=v109)
        if v110 is None:
            return False
        v93.v111 = v186(v93, 'edit_K', [])
        v93.v112 = v186(v93, 'edit_V', [])
        v93.v111.v187(v110)
        v93.v112.v187(v108)
        return True

    @v134.v37()
    def recall_fact(v93, v113, v114, v109=None):
        v115 = v93.v95.v185(v113, exclude=v109)
        if v115 is None or not v186(v93, 'edit_K', []):
            return None
        v116 = v134.v188(v93.v111, 0)
        v117 = v116 @ v115
        v105 = {}
        for v104, v123 in v189(v93.v112):
            v105[v123] = v197(v105.v242(v123, -1.0), v17(v117[v104]))
        return v19(v38.v225([v105.v242(v236, -1.0) for v236 in v114]))

class GptBank:
    """GPT+RAG control: same retrieval math as FpBank, key = GPT mean-pool embedding."""

    def __init__(v93, v26, v32, v42, v24):
        v93.v26 = v26
        v93.v32 = v32
        v93.v42 = v42
        v93.v24 = v24
        v93.v118: v190[v89, v134.v226] = {}

    @v134.v37()
    def ctx_fp(v93, v53, v109=None):
        v119 = [v36 for v36 in v257.v147(v53) if v36 != v109][:40]
        if v194(v119) < 3:
            return None
        v120 = ' '.v191(v119)
        if v120 in v93.v118:
            return v93.v118[v120]
        v121 = [v104 for v104 in v93.v32.v253(v120).v121 if v104 != v93.v42][-v229:]
        if not v121:
            return None
        v31 = v134.v128([v121], device=v93.v24)
        v122 = v93.v26.v258(input_ids=v31).v243[0].v192(0)
        v123 = v227.v193(v122, dim=-1)
        v93.v118[v120] = v123
        return v123

def load_gpt(v24):
    v25 = v134.v124(v2, map_location=v24, weights_only=False)
    v26 = v228(v244(**v25['conf'])).v125(v24)
    v26.v126(v25['model'])
    v26.v127()
    return v26

def gpt_span(v26, v24, v27, v28) -> v17:
    v29 = (v27 + v28)[-v229:]
    v30 = v194(v29) - v194(v28)
    v31 = v134.v128([v29], device=v24)
    with v134.v37():
        v129 = v227.v195(v26(input_ids=v31).v230[0], dim=-1)
    return v196((v17(v129[v30 + v110 - 1, v252]) for v110, v252 in v189(v28))) / v197(1, v194(v28))

@v134.v37()
def gpt_word_surprisal(v26, v32, v24, v33) -> v38.v18:
    """GPT's native OOD signal: mean per-piece surprisal of the word given a neutral prefix."""
    v34 = [v104 for v104 in v32.v253(' The ').v121][-8:]
    v35 = []
    for v36 in v33:
        v130 = [v104 for v104 in v32.v253(' ' + v36).v121]
        if not v130:
            v35.v187(0.0)
            continue
        v29 = v34 + v130
        v31 = v134.v128([v29], device=v24)
        v129 = v227.v195(v26(input_ids=v31).v230[0], dim=-1)
        v131 = v194(v34)
        v132 = -v38.v192([v17(v129[v131 + v110 - 1, v252]) for v110, v252 in v189(v130)])
        v35.v187(v17(v132))
    return v38.v133(v35)

def main() -> v19:
    v0.v91(parents=True, exist_ok=True)
    v7.v135('', encoding='utf-8')
    v136(f'Stage196 start {v255.v249(v256.v250).v222()}')
    v136('TapeLM assembled stack + anti-clone distinguishability gate')
    v24 = v134.v24('cuda' if v134.v245.v231() else 'cpu')
    v39 = v198.v137(v8)
    v40 = v138.v138()
    v139, v140, v94, v141 = v142()
    v32 = v199.v143(v89(v21.v200))
    v41 = v32.v144()
    v42 = v32.v201(v202) or 0
    v43 = v246.v232(v32, v94, v42, v41).v125(v24)
    v44 = v233(v141, v41).v125(v24)
    v44.v126(v134.v124(v1, map_location=v24, weights_only=False)['model'])
    v44.v127()
    v26 = v145(v24)
    v136(f'models loaded ({v138.v138() - v40:.0f}s)')
    v45 = v146(v44, v94, v32, v43, v42, v24)
    v46 = [v234.v203(v204) for v204 in v4.v254(encoding='utf-8').v235()]
    v47 = [v65 for v65 in v46 if v65['type'] == 'next_tok']
    v48 = [v65 for v65 in v46 if v65['type'] == 'entity']
    v136(f'exam: next_tok={v194(v47)} entity={v194(v48)}')
    with v3.v177('r', encoding='utf-8', errors='ignore') as v92:
        v53 = v92.v150(v9)
    v49 = v53[-v10:]
    v50 = [v72.v210() for v72 in v49.v247('\n') if 120 < v194(v72.v210()) < 1000][:1200]
    v51 = v53[60000000:60000000 + v10]
    v52 = [v72.v210() for v72 in v51.v247('\n') if 120 < v194(v72.v210()) < 1000][:600]
    import re
    from collections import Counter
    v33 = v205.v147('[A-Za-z][a-z]+', v53[:v9])
    del v53
    v54 = v148(v33)
    v55 = v149(v54.v206())
    v56 = [v36 for v36, v236 in v54.v237(v12) if v236 >= v11]
    v136(f'tail_paras={v194(v50)} mid_paras={v194(v52)} lexicon={v194(v56)} ({v138.v138() - v40:.0f}s)')
    v57 = {}
    v58 = v207(v45.v208, v47, 'next_tok')['next_tok_acc']
    v59 = v207(lambda v236, v248: v239(v26, v24, v236, v248), v47, 'next_tok')['next_tok_acc']
    v57['parity'] = {'curve_next_tok': v58, 'gpt_next_tok': v59, 'delta': v58 - v59}
    v136(f'[1 parity] curve={v58:.3f} gpt={v59:.3f} d={v58 - v59:+.3f} ({v138.v138() - v40:.0f}s)')
    v45.v150(v50, 'read-tail')
    v60 = v209(v46, v32, v42, v45.v95, v45.v96, v45.v97)['acc']
    v61 = v207(lambda v236, v248: v239(v26, v24, v236, v248), v48, 'entity')['entity_acc']
    v62 = v151(v26, v32, v42, v24)
    v152, v153 = v154(v50, v62, 'gpt-rag')
    v63 = v209(v46, v32, v42, v62, v152, v153)['acc']
    v57['recall'] = {'curve_fp': v60, 'gpt_parametric': v61, 'gpt_rag': v63, 'chance': 0.25}
    v136(f'[2 recall] curve_fp={v60:.3f} gpt_param={v61:.3f} gpt_rag={v63:.3f} ({v138.v138() - v40:.0f}s)')
    v45.v155(v56)
    v64 = []
    for v65 in v48:
        v132 = v32.v240(v65['cand_ids'][v65['gold_idx']], skip_special_tokens=False).v210()
        v36 = v205.v147('[A-Za-z][a-z]+', v132)
        if v36 and v36[0] in v54:
            v64.v187(v36[0])
    v64 = v156(v190.v211(v64))
    v66 = v157(v55, v39, v13)
    v67 = v158(v45.v212(v66), v45.v212(v64))
    v68 = v158(v213(v26, v32, v24, v66), v213(v26, v32, v24, v64))
    v57['calibration'] = {'curve_lex_auc': v67, 'gpt_bpe_auc': v68, 'n_real': v194(v64), 'n_fake': v194(v66)}
    v136(f'[3 calib] curve_lexAUC={v67:.3f} gpt_bpeAUC={v68:.3f} ({v138.v138() - v40:.0f}s)')
    v69 = []
    v70 = v157(v55, v39, v14 * 2)
    v71 = 0
    for v72 in v52:
        if v194(v69) >= v14 or v71 >= v194(v70):
            break
        v159 = v238.v214(v72)
        if not v159:
            continue
        v160 = v159.v215(1)
        v161 = v70[v71]
        v71 += 1
        v162 = v72.v216(v160, v161)
        v163 = v194(v162) // 2
        v164 = v162[:v163]
        v165 = v162[v163:]
        if v161 not in v164 or v194(v257.v147(v165)) < 4:
            continue
        v69.v187({'write': v164, 'query': v165, 'F': v161})
    v136(f'knowledge-edit items={v194(v69)}')
    v45.v111, v45.v112 = ([], [])
    v73 = [v74['F'] for v74 in v69]
    for v74 in v69:
        v45.v217(v74['write'], v74['F'], exclude=v74['F'])
    v75 = v76 = 0
    for v74 in v69:
        v166 = [v92 for v92 in v73 if v92 != v74['F']]
        v39.v218(v166)
        v114 = [v74['F']] + v166[:3]
        v167 = v156(v181(v194(v114)))
        v39.v218(v167)
        v168 = [v114[v110] for v110 in v167]
        v169 = v167.v219(0)
        v170 = v45.v220(v74['query'], v168, exclude=v74['F'])
        if v170 is None:
            continue
        v75 += v19(v170 == v169)
        v76 += 1
    v77 = v75 / v197(1, v76)
    v78 = v79 = 0
    for v74 in v69:
        v166 = [v92 for v92 in v73 if v92 != v74['F']]
        v39.v218(v166)
        v114 = [v74['F']] + v166[:3]
        v167 = v156(v181(v194(v114)))
        v39.v218(v167)
        v168 = [v114[v110] for v110 in v167]
        v169 = v167.v219(0)
        v99 = [v104 for v104 in v32.v253(v74['query']).v121 if v104 != v42][-v229:]
        v171 = [v239(v26, v24, v99, [v104 for v104 in v32.v253(' ' + v236).v121 if v104 != v42]) for v236 in v168]
        v78 += v19(v19(v38.v225(v171)) == v169)
        v79 += 1
    v80 = v78 / v197(1, v79)
    v57['edit'] = {'curve': v77, 'gpt': v80, 'n': v76, 'chance': 0.25}
    v136(f'[4 edit] curve={v77:.3f} gpt={v80:.3f} (n={v76}) ({v138.v138() - v40:.0f}s)')
    v81 = v221(v57['parity']['delta']) <= 0.03
    v82 = v60 >= 0.5 and v60 > v61 + 0.15
    v83 = v60 >= v63 - 0.03
    v84 = v67 > v68 and v67 >= 0.8
    v85 = v77 >= 0.5 and v77 > v80 + 0.2
    v86 = v60 > v61 + 0.15 and v60 > v63 or v67 > v68 or v77 > v80 + 0.2
    if v81 and v82 and v83 and v84 and v85:
        v172 = 'TAPELM_COMPOSES_AND_DISTINCT'
    elif v81 and v82 and v84 and v85 and (not v83):
        v172 = 'TAPELM_DISTINCT_RECALL_RAG_EQUIV'
    elif v81 and (not v86):
        v172 = 'TAPELM_CLONE_RISK'
    else:
        v172 = 'TAPELM_PARTIAL'
    v87 = {'parity_hold': v81, 'recall_win': v82, 'recall_beats_rag': v83, 'calib_win': v84, 'edit_win': v85}
    v35 = {'timestamp': v255.v249(v256.v250).v222(), 'protocol': 'tapelm_196', 'overall': v172, 'gates': v87, 'axes': v57, 'note': 'one frozen P1 encoder; generation + fp memory + fp lexicon + one-shot edit share one fp-space; anti-clone: win only where GPT structurally weak, curve must beat GPT AND GPT+RAG'}
    v5.v135(v234.v223(v35, indent=2, ensure_ascii=False), encoding='utf-8')
    v6.v135('\n'.v191(['# Stage196 — TapeLM assembled stack + anti-clone gate', '', f'**Overall:** `{v172}`', '', f'- **parity (entry ticket):** curve {v58:.3f} vs gpt {v59:.3f} (Δ{v58 - v59:+.3f}) — hold={v81}', f'- **recall (win):** curve_fp {v60:.3f} vs gpt_param {v61:.3f} vs **gpt+rag {v63:.3f}** (chance 0.25)', f'- **calibration (win):** curve_lexAUC {v67:.3f} vs gpt_bpeAUC {v68:.3f}', f'- **one-shot edit (win):** curve {v77:.3f} vs gpt {v80:.3f} (n={v76}, chance 0.25)', '', f'gates: {v87}', '', 'One frozen curve encoder serves generation + fact memory + lexical calibration + one-shot edit from a single shared fp-space. Win counted only where BPE-GPT is structurally weak (recall/calib/edit), with GPT+RAG as the nearest rival control.']), encoding='utf-8')
    v136(f'[196] {v172}')
    return 0
if v88 == '__main__':
    raise v173(v224())