"""Shared helpers for unexpected-comparison stages 240–245."""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _tapelm_ext import DomainAdapter
v0 = v11('results')
v1 = v11('checkpoints/stage191_p1_curve.pt')
v2 = v11('checkpoints/stage238_mixed_scratch.pt')
v3 = v11('data/_wikitext103_train.txt')
v4 = v11('data/stage191_exam_v3.jsonl')
v5 = v11('data/external_tinystories_100k_85.txt')
v6 = v109.v12("[A-Za-z][A-Za-z0-9'\\-]{2,}")

def make_logger(v13: v11):
    v13.v165.v110(parents=True, exist_ok=True)

    def log(v111: v10) -> None:
        v112 = v111 if v111.v213('\n') else v111 + '\n'
        try:
            v214(v112, end='', flush=True)
        except v166:
            v214(v112.v239('ascii', 'replace').v253('ascii'), end='', flush=True)
        with v13.v173('a', encoding='utf-8') as v35:
            v35.v215(v112)
    return v14

def load_p1(v15):
    v113, v114, v24, v23 = v115()
    v16 = v167.v116(v10(v216.v168))
    v17 = v16.v117()
    v18 = v16.v169(v170) or 0
    v19 = v241.v217(v16, v24, v18, v17).v118(v15)
    v20 = v218(v23, v17).v118(v15)
    v20.v119(v162.v219(v1, map_location=v15, weights_only=False)['model'])
    v20.v120()
    for v21 in v20.v121():
        v21.v171(False)
    v22 = v122(v20, v24, v15)
    return (v113, v114, v24, v23, v16, v17, v18, v19, v20, v22)

def load_curve_ckpt(v13: v11, v23: v46, v17: v46, v24, v15):
    v20 = v218(v23, v17).v118(v15)
    v20.v119(v162.v219(v13, map_location=v15, weights_only=False)['model'])
    v20.v120()
    for v21 in v20.v121():
        v21.v171(False)
    return (v20, v122(v20, v24, v15))

def wiki_bits(v25: v123, v26: v46, v27: v172.v124):
    with v3.v173('r', encoding='utf-8', errors='ignore') as v35:
        v125 = v35.v174(4000000 if v25 else 20000000)
    v28 = v45(v9.v175((v74.v242(1) for v74 in v268.v254(v125) if v201(v74.v242(1)) >= 5)))
    v27.v126(v28)
    v29 = v45(v9.v175((v101 for v101 in v109.v252('[A-Za-z][a-z]{2,}', v125) if v201(v101) <= 14)))[:v26]
    v30 = [v21.v176() for v21 in v125.v220('\n') if v201(v21.v176()) > 200]
    return (v125, v28, v29, v30)

def make_facts(v31: v46, v28, v27: v172.v124):
    v32 = [v101 for v101 in v243(v255(v28), v27, v31 + 40) if v201(v101) >= 5][:v31]
    v33 = []
    for v127, v128 in v129(v32):
        v130 = v28[v127]
        v33.v177({'S': v128, 'value': v130, 'sent': f'{v128} was appointed director of {v130} in 1987 .', 'fid': v127})
    v34 = [v35['value'] for v35 in v33] + v28[v31:v31 + 80]
    return (v33, v34)

def write_tape_bank(v22: v122, v33):
    v131, v132 = ([], [])
    for v35 in v33:
        v133 = v22.v179([v35['S']])[0]
        v40 = v22.v134(v35['sent'], exclude=v35['value'])
        v131.v177(v208.v161(v133 + v40, dim=-1) if v40 is not None else v133)
        v132.v177(v35['value'])
    return (v162.v178(v131, 0), v132)

def _tape_query(v36, v37, v38=None):
    """Query built like a key: subject fingerprint + context of a prefix that stops where the value
    would go. A context-only query leaves the shared template dominating the similarity, and it also
    does not match the (prefix -> slot) pairs W_q is trained on."""
    v39 = v36.v179([v37['S']])[0]
    v40 = v36.v134(f"In the report {v37['S']} was appointed director of")
    v41 = v208.v161(v39 + v40, dim=-1) if v40 is not None else v39
    if v38 is not None:
        v41 = v208.v161(v38.v244(v41.v256(0)), dim=-1)[0]
    return v41

def init_query_adapter(v15: v162.v15) -> v7:
    """Trainable query warp; tape KEYS stay frozen canonical — only queries move.

    Naming in checkpoints:
      W_q_glue  — stage 256 SlotBias.W_q (decode glue + gate)
      W_q_stream — stage 255 stream ingest W_query (continual query adapter)
    """
    v42 = v7(256).v118(v15)
    with v162.v100():
        v42.v101.v221.v180(v162.v245(256, device=v15) + 0.02 * v162.v257(256, 256, device=v15))
    return v42

def _gold_slot_indices(v43: v10, v44: v45[v10]) -> v45[v46]:
    return [v127 for v127, v143 in v129(v44) if v143 == v43]

def train_query_adapter(v42: v7, v36, v33: v45[v9], v47: v162.v135, v44: v45[v10], v15: v162.v15, v48: v46, v49: v46, v50: v8=0.002, v51: v46=8) -> v8:
    """Contrastive align adapted queries to frozen keys (InfoNCE over full bank)."""
    if not v33 or v47.v222() == 0 or v48 <= 0:
        return v8('nan')
    v42.v136()
    v52 = v162.v181.v137(v42.v121(), lr=v50, weight_decay=0.01)
    v27 = v172.v124(v49 + 17)
    v53 = v47.v223().v118(v15, v162.v138)
    v54: v45[v8] = []
    for v55 in v139(v48):
        v57 = [v33[v27.v235(v201(v33))] for v55 in v139(v258(v51, v201(v33)))]
        v140 = 0.0
        v141 = 0
        for v35 in v57:
            v182 = v224(v35['value'], v44)
            if not v182:
                continue
            v41 = v259(v36, v35, None).v118(v15)
            v41 = v42(v41.v256(0)).v225(0)
            v157 = v53 @ v41
            v142 = -(v162.v260(v157[v182], 0) - v162.v260(v157, 0))
            v140 = v140 + v142
            v141 += 1
        if v141 == 0:
            continue
        v142 = v140 / v141
        v52.v183(set_to_none=True)
        v142.v184()
        v162.v246.v226.v185(v42.v121(), 1.0)
        v52.v94()
        v54.v177(v8(v142))
    v42.v120()
    return v186(v54) / v187(1, v201(v54))

def train_query_adapter_pairs(v42: v7, v56: v45[v9], v47: v162.v135, v44: v45[v10], v15: v162.v15, v48: v46, v49: v46, v50: v8=0.002, v57: v46=64, v58: v8=0.05) -> v8:
    """InfoNCE over the full bank using (prefix -> slot) pairs harvested from the stream itself.

    Fitting W_q on a handful of planted probe facts only teaches it where those facts live, so the
    recall number then says nothing about the rest of the bank. Ingested entities supply thousands
    of pairs, which lets the probe facts stay fully held out.
    """
    if not v56 or v47.v222() == 0 or v48 <= 0:
        return v8('nan')
    v59: v9[v10, v46] = {}
    for v127, v143 in v129(v44):
        v59.v188(v143, v127)
    v60 = [v21 for v21 in v56 if v21['value'] in v59]
    if not v60:
        return v8('nan')
    v53 = v47.v223().v118(v15, v162.v138)
    v61 = v162.v178([v21['q'] for v21 in v60]).v118(v15, v162.v138)
    v62 = v162.v144([v59[v21['value']] for v21 in v60], device=v15)
    v42.v136()
    v52 = v162.v181.v137(v42.v121(), lr=v50, weight_decay=0.01)
    v63 = v162.v227(device='cpu').v145(v49 + 31)
    v54: v45[v8] = []
    for v55 in v139(v48):
        v146 = v162.v247(0, v61.v149(0), (v258(v57, v61.v149(0)),), generator=v63).v118(v15)
        v41 = v208.v161(v42(v61[v146]), dim=-1)
        v142 = v208.v189(v41 @ v53.v261() / v58, v62[v146])
        v52.v183(set_to_none=True)
        v142.v184()
        v162.v246.v226.v185(v42.v121(), 1.0)
        v52.v94()
        v54.v177(v8(v142))
    v42.v120()
    return v186(v54) / v187(1, v201(v54))

def tape_recall_metrics(v33, v34, v36, v47, v44, v49: v46, v38=None, v64: v46=200000, v65=None, v66: v10='auto') -> v9:
    """Fixed-seed 4-way distractors plus full-bank rank metrics (scores are time-invariant if arc_enc frozen)."""
    if not v33 or v47.v222() == 0:
        return {'four_way': v8('nan'), 'top1': v8('nan'), 'mrr': v8('nan'), 'median_rank': v8('nan')}
    from _inprint_glue import VOTES_AUTO_MIN_SLOTS, resolve_retrieve_mode, slot_query_words
    from _retrieval_modes import vote_scores
    v67 = v172.v124(v49 + 3)
    v59: v9[v10, v45[v46]] = {}
    for v147, v143 in v129(v44):
        v59.v188(v143, []).v177(v147)
    v53 = v47.v223().v118('cpu', v162.v138) if v47.v148 else v47.v8()
    v68 = v53.v149(0)
    v69 = v65 is not None and v68 >= v190 and (v228(v66, v68) == 'votes')
    v70 = 0
    v71: v45[v46] = []
    for v35 in v33:
        if v69:
            v191 = v229(f"In the report {v35['S']} was appointed director of")
            v157 = v230(v191, v65.v65, v65.v231)
            v62 = v35['value']
            v192 = v187((v157.v262(v147, 0.0) for v147 in v59.v262(v62, ())), default=0.0)
            v193 = 1 + v186((1 for v143 in v157.v269() if v143 > v192))
        else:
            v194 = v259(v36, v35, v38).v223().v263().v8()
            v195 = []
            for v127 in v139(0, v68, v64):
                v195.v177(v53[v127:v127 + v64] @ v194)
            v196 = v162.v248(v195) if v195 else v162.v249(0)
            v62 = v35['value']
            v197 = v8(v196[v59[v62]].v187()) if v62 in v59 else -1.0
            v193 = 1 + v46((v196 > v197).v186().v264())
        v71.v177(v193)
        v150 = [v159 for v159 in v34 if v159 != v62]
        v67.v126(v150)
        v151 = [v62] + v150[:3]
        v152 = v45(v139(4))
        v67.v126(v152)
        v153 = [v151[v127] for v127 in v152]
        if v69:
            v198 = [v187((v157.v262(v147, 0.0) for v147 in v59.v262(v40, ())), default=-1.0) for v40 in v153]
        else:
            v198 = [v8(v196[v59[v40]].v187()) if v40 in v59 else -1.0 for v40 in v153]
        v70 += v46(v46(v199.v265(v198)) == v152.v250(0))
    v72 = v199.v154(v71, dtype=v199.v200)
    return {'four_way': v70 / v201(v33), 'top1': v8(v199.v160(v72 == 1)), 'mrr': v8(v199.v160(1.0 / v72)), 'median_rank': v8(v199.v232(v72))}

def tape_recall(v33, v34, v36, v47, v44, v49: v46, v38=None) -> v8:
    return v155(v33, v34, v36, v47, v44, v49, W_bwd=v38)['four_way']

def tape_recall_decision(v33, v34, v36, v47, v44, v49: v46, v38=None, **v73) -> v9:
    """Closed-pool 4-way plus open full-bank rank — use in decision JSON."""
    v74 = v155(v33, v34, v36, v47, v44, v49, W_bwd=v38, **v73)
    return {'four_way': v74['four_way'], 'full_bank_top1': v74['top1'], 'full_bank_mrr': v74['mrr'], 'full_bank_median_rank': v74['median_rank']}

def canonical_fp_version() -> v10:
    return v1.v75

def load_next_tok_items(v76: v46):
    v77 = []
    if v4.v156():
        with v4.v173(encoding='utf-8') as v35:
            for v112 in v35:
                v79 = v240.v251(v112)
                if v79.v262('type') == 'next_tok':
                    v77.v177(v79)
                if v201(v77) >= v76:
                    break
    return v77

def curve_next_tok(v20, v19, v18, v77, v15) -> v8:
    if not v77:
        return v8('nan')
    v78 = 0
    for v79 in v77:
        v157 = [v233(v20, v19, v18, v79['ctx_ids'], v40, v15) for v40 in v79['cand_ids']]
        v78 += v46(v46(v199.v265(v157)) == v79['gold_idx'])
    return v78 / v201(v77)

def gpt_next_tok(v80, v77, v15) -> v8:
    if not v77:
        return v8('nan')
    v78 = 0
    for v79 in v77:
        v157 = [v234(v80, v15, v79['ctx_ids'], v40) for v40 in v79['cand_ids']]
        v78 += v46(v46(v199.v265(v157)) == v79['gold_idx'])
    return v78 / v201(v77)

def gpt_fact_recall(v80, v16, v18, v33, v34, v15, v49: v46) -> v8:
    v67 = v172.v124(v49 + 3)
    v78 = 0
    for v35 in v33:
        v158 = [v127 for v127 in v16.v239(f"In the report {v35['S']} was linked to the organization of").v98 if v127 != v18]
        v150 = [v159 for v159 in v34 if v159 != v35['value']]
        v67.v126(v150)
        v151 = [v35['value']] + v150[:3]
        v152 = v45(v139(4))
        v67.v126(v152)
        v153 = [v151[v127] for v127 in v152]
        v157 = [v234(v80, v15, v158, [v127 for v127 in v16.v239(' ' + v40).v98 if v127 != v18]) for v40 in v153]
        v78 += v46(v46(v199.v265(v157)) == v152.v250(0))
    return v78 / v187(1, v201(v33))

def ft_batch(v81, v82, v83, v84, v85, v15, v86=True, v87=0.75):
    v88 = []
    for v55 in v139(v84):
        if not v86 or v81.v172() < v87 or (not v83):
            v202 = []
            while v201(v202) < v85:
                v202 += v82[v81.v235(v201(v82))]
            v88.v177(v202[:v85])
        else:
            v203 = v81.v235(v187(1, v201(v83) - v85 - 1))
            v88.v177(v83[v203:v203 + v85])
    return v162.v144(v88, device=v15)

def memorize_gpt(v80, v16, v18, v33, v34, v30, v15, v49, v89, v84, v85, v90, v91, v92, v14):
    v82 = [[v127 for v127 in v16.v239(v35['sent']).v98 if v127 != v18] for v35 in v33]
    v83 = [v127 for v127 in v16.v239(' '.v267(v30[:400])[:150000]).v98 if v127 != v18]
    v52 = v162.v181.v137(v80.v121(), lr=v90, weight_decay=0.01)
    v81 = v172.v124(v49 + 11)
    v93 = 0
    v80.v136()
    for v94 in v139(1, v89 + 1):
        v159 = v204(v81, v82, v83, v84, v85, v15)
        v142 = v80(input_ids=v159, labels=v159).v142
        v52.v183(set_to_none=True)
        v142.v184()
        v52.v94()
        v93 = v94
        if v94 % v92 == 0:
            v80.v120()
            v205 = v236(v80, v16, v18, v33, v34, v15, v49)
            v14(f'  gpt memorize step {v94}: loss={v8(v142):.3f} recall={v205:.3f}')
            if v205 >= v91:
                v80.v136()
                break
            v80.v136()
    v80.v120()
    return (v93, v82, v83)

def code_ce(v80, v95, v84, v85, v50, v48, v15, v49, v14, v96='B', v82=None, v97=0.0):
    if v201(v95) < v85 + 2:
        raise v206('code corpus too short')
    v52 = v162.v181.v137(v80.v121(), lr=v50, weight_decay=0.01)
    v81 = v172.v124(v49 + 17)
    v80.v136()
    for v94 in v139(1, v48 + 1):
        v88 = []
        for v55 in v139(v84):
            v207 = v82 is not None and v97 > 0 and (v81.v172() < v97)
            if v207:
                v202 = []
                while v201(v202) < v85:
                    v202 += v82[v81.v235(v201(v82))]
                v88.v177(v202[:v85])
            else:
                v203 = v81.v235(v187(1, v201(v95) - v85 - 1))
                v88.v177(v95[v203:v203 + v85])
        v159 = v162.v144(v88, device=v15)
        v142 = v80(input_ids=v159, labels=v159).v142
        v52.v183(set_to_none=True)
        v142.v184()
        v52.v94()
        if v94 % v187(40, v48 // 4) == 0:
            v14(f'  gpt {v96} step {v94}: loss={v8(v142):.3f}')
    v80.v120()

@v162.v100()
def gpt_emb(v80, v16, v18, v15, v98):
    v98 = [v127 for v127 in v98 if v127 != v18][-v237:]
    if not v98:
        return None
    v99 = v80.v266(input_ids=v162.v144([v98], device=v15)).v238[0].v160(0)
    return v208.v161(v99, dim=-1)

def gpt_word(v80, v16, v18, v15, v101):
    return v163(v80, v16, v18, v15, v16.v239(' ' + v101).v98)

def gpt_ctx(v80, v16, v18, v15, v102, v103=None):
    v104 = [v159 for v159 in v6.v252(v102) if v159 != v103][:40]
    return v163(v80, v16, v18, v15, v16.v239(' '.v267(v104)).v98) if v201(v104) >= 3 else None

def write_rag_bank(v80, v16, v18, v15, v33):
    v88, v132 = ([], [])
    for v35 in v33:
        v133 = v209(v80, v16, v18, v15, v35['S'])
        v40 = v210(v80, v16, v18, v15, v35['sent'], exclude=v35['value'])
        v88.v177(v208.v161(v133 + v40, dim=-1) if v40 is not None else v133)
        v132.v177(v35['value'])
    return (v162.v178(v88, 0), v132)

def rag_recall(v80, v16, v18, v15, v33, v34, v47, v44, v49: v46) -> v8:
    v78, v141 = (0, 0)
    v67 = v172.v124(v49 + 3)
    for v35 in v33:
        v41 = v210(v80, v16, v18, v15, f"In the report {v35['S']} was linked to the organization.", exclude=v35['value'])
        if v41 is None:
            v41 = v209(v80, v16, v18, v15, v35['S'])
        v150 = [v159 for v159 in v34 if v159 != v35['value']]
        v67.v126(v150)
        v151 = [v35['value']] + v150[:3]
        v152 = v45(v139(4))
        v67.v126(v152)
        v153 = [v151[v127] for v127 in v152]
        v157 = []
        for v40 in v153:
            v211 = [v147 for v147, v143 in v129(v44) if v143 == v40]
            v157.v177(v8((v47[v211] @ v41).v187()) if v211 else -1.0)
        v78 += v46(v46(v199.v265(v157)) == v152.v250(0))
        v141 += 1
    return v78 / v187(1, v141)

def dump(v105: v11, v106: v11, v107: v9, v108: v10):
    v0.v110(parents=True, exist_ok=True)
    v105.v164(v240.v212(v107, indent=2), encoding='utf-8')
    v106.v164(f"# {v108}\n\n**{v107['overall']}**\n\n```json\n{v240.v212(v107, indent=2)}\n```\n", encoding='utf-8')