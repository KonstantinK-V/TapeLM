"""
Stage 239 — Sequential catastrophic forgetting (A → B) vs fair GPT control.

Protocol:
  1. Plant domain-A facts (novel subjects) into TapeLM slots on frozen P1.
  2. Matched GPT finetunes on A fact sentences until paraphrase-probe recall clears (205-style).
  3. Adapt to domain B (code):
       TapeLM — keep canonical slot keys; arc_enc shift on code + W_bwd qmap @ read (227).
       GPT    — continue CE finetune on code only (no A rehearsal; classic CF).
  4. Measure retained A-fact recall (shared paraphrase probe) + next_tok collateral.

Gates:
  G_memorize   both systems A-recall >= 0.70 after acquire
  G_tape_keep  TapeLM A-recall after B >= 0.80
  G_gpt_drop   GPT A-recall drops by >= 0.15 vs post-memorize
  G_gap        TapeLM_after - GPT_after >= 0.20

Note: vs GPT+RAG this is architectural (index can also keep A); vs parametric GPT it is CF capability.

  python _stage239_cf_sequential.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
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
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _tapelm_ext import DomainAdapter
v0 = v9('results')
v1 = v9('checkpoints/stage191_p1_curve.pt')
v2 = v9('data/_wikitext103_train.txt')
v3 = v9('data/stage191_exam_v3.jsonl')
v4 = v0 / 'stage239_decision.json'
v5 = v0 / 'stage239_mini.md'
v6 = v0 / '_stage239_log.txt'
v7 = 239

def log(v10: v77) -> None:
    v11 = v10 if v10.v153('\n') else v10 + '\n'
    try:
        v154(v11, end='', flush=True)
    except v78:
        v154(v11.v225('ascii', 'replace').v214('ascii'), end='', flush=True)
    try:
        v6.v199.v155(parents=True, exist_ok=True)
        with v6.v165('a', encoding='utf-8') as v40:
            v40.v200(v11)
    except v79:
        pass

def main() -> v8:
    v12 = v156.v80()
    v12.v81('--smoke', action='store_true')
    v13 = v12.v82()
    try:
        v6.v151('', encoding='utf-8')
    except v79:
        pass
    v14 = v157.v14('cuda' if v157.v215.v201() else 'cpu')
    v15 = v158.v83(v7)
    v157.v84(v7)
    v16 = v85.v85()
    v17 = 12 if v13.v86 else 40
    v18 = 240 if v13.v86 else 2400
    v19 = 400 if v13.v86 else 1600
    v20 = 60 if v13.v86 else v159.v87
    v21 = 80 if v13.v86 else v159.v88
    v22 = 60 if v13.v86 else 400
    v23 = 40 if v13.v86 else 120
    v24 = 300 if v13.v86 else 8000
    v89, v90, v91 = (8, 64, 0.0003)
    v25 = 0.0005
    v26 = 0.72
    v92(f'Stage239 start {v235.v227(v236.v228).v196()} device={v14}')
    v93, v94, v95, v96 = v97()
    v27 = v160.v98(v77(v202.v161))
    v28 = v27.v99()
    v29 = v27.v162(v163) or 0
    v30 = v216.v203(v27, v95, v29, v28).v100(v14)
    v31 = v204(v96, v28).v100(v14)
    v31.v101(v157.v205(v1, map_location=v14, weights_only=False)['model'])
    v31.v102()
    for v32 in v31.v103():
        v32.v164(False)
    v33 = v104(v31, v95, v14)
    with v2.v165('r', encoding='utf-8', errors='ignore') as v40:
        v105 = v40.v166(4000000 if v13.v86 else 20000000)
    v34 = v106(v206.v167((v124.v217(1) for v124 in v237.v229(v105) if v192(v124.v217(1)) >= 5)))
    v15.v107(v34)
    v35 = v106(v206.v167((v169 for v169 in v243.v238('[A-Za-z][a-z]{2,}', v105) if v192(v169) <= 14)))[:v22]
    v36 = [v32.v168() for v32 in v105.v207('\n') if v192(v32.v168()) > 200]
    v37 = [v169 for v169 in v218(v230(v34), v15, v17 + 40) if v192(v169) >= 5][:v17]
    v38 = []
    for v108, v109 in v110(v37):
        v111 = v34[v108]
        v38.v170({'S': v109, 'value': v111, 'sent': f'{v109} was appointed director of {v111} in 1987 .', 'fid': v108})
    v39 = [v40['value'] for v40 in v38] + v34[v17:v17 + 80]
    v112, v113 = ([], [])
    for v40 in v38:
        v114 = v33.v208([v40['S']])[0]
        v115 = v33.v171(v40['sent'], exclude=v40['value'])
        v112.v170(v231.v219(v114 + v115, dim=-1) if v115 is not None else v114)
        v113.v170(v40['value'])
    v41 = v157.v116(v112, 0)

    def tape_recall(v117, v118, v119, v120=None) -> v42:
        v125, v172 = (0, 0)
        v121 = v158.v83(v7 + 3)
        v122 = (lambda v220: v231.v219(v120.v232(v220), dim=-1)) if v120 is not None else None
        for v40 in v38:
            v173 = v117.v171(f"In the report {v40['S']} was linked to the organization.", exclude=v40['value'])
            if v173 is None:
                v173 = v117.v208([v40['S']])[0]
            v174 = v122(v173.v239(0))[0] if v122 else v173
            v175 = [v139 for v139 in v39 if v139 != v40['value']]
            v121.v107(v175)
            v176 = [v40['value']] + v175[:3]
            v177 = v106(v138(4))
            v121.v107(v177)
            v178 = [v176[v108] for v108 in v177]
            v179 = []
            for v115 in v178:
                v209 = [v221 for v221, v240 in v110(v119) if v240 == v115]
                v179.v170(v42((v118[v209] @ v174).v210()) if v209 else -1.0)
            v125 += v8(v8(v244.v241(v179)) == v177.v233(0))
            v172 += 1
        return v125 / v210(1, v172)
    v43 = []
    if v3.v123():
        with v3.v165(encoding='utf-8') as v40:
            for v11 in v40:
                v126 = v213.v222(v11)
                if v126.v234('type') == 'next_tok':
                    v43.v170(v126)
                if v192(v43) >= v23:
                    break

    def curve_next_tok(v124) -> v42:
        if not v43:
            return v42('nan')
        v125 = 0
        for v126 in v43:
            v179 = [v223(v124, v30, v29, v126['ctx_ids'], v115, v14) for v115 in v126['cand_ids']]
            v125 += v8(v8(v244.v241(v179)) == v126['gold_idx'])
        return v125 / v192(v43)

    def gpt_next_tok(v46) -> v42:
        if not v43:
            return v42('nan')
        v125 = 0
        for v126 in v43:
            v179 = [v224(v46, v14, v126['ctx_ids'], v115) for v115 in v126['cand_ids']]
            v125 += v8(v8(v244.v241(v179)) == v126['gold_idx'])
        return v125 / v192(v43)
    v44 = v127(v33, v41, v113)
    v45 = v128(v31)
    v92(f'tape AFTER A write: A_recall={v44:.3f} next_tok={v45:.3f} ({v85.v85() - v16:.0f}s)')
    v46 = v180.v129(v181(v14))
    v46.v130()
    v47 = [[v108 for v108 in v27.v225(v40['sent']).v182 if v108 != v29] for v40 in v38]
    v48 = [v108 for v108 in v27.v225(' '.v242(v36[:400])[:150000]).v182 if v108 != v29]

    def ft_batch(v50, v131=None, v132=True):
        v133 = []
        v134 = v131 if v131 is not None else v47
        for v135 in v138(v89):
            if not v132 or v50.v158() < 0.75 or (not v48):
                v211 = []
                while v192(v211) < v90:
                    v211 += v134[v50.v212(v192(v134))]
                v133.v170(v211[:v90])
            else:
                v194 = v50.v212(v210(1, v192(v48) - v90 - 1))
                v133.v170(v48[v194:v194 + v90])
        return v157.v183(v133, device=v14)

    def gpt_fact_recall(v136) -> v42:
        v121 = v158.v83(v7 + 3)
        v125 = 0
        for v40 in v38:
            v184 = [v108 for v108 in v27.v225(f"In the report {v40['S']} was linked to the organization of").v182 if v108 != v29]
            v175 = [v139 for v139 in v39 if v139 != v40['value']]
            v121.v107(v175)
            v176 = [v40['value']] + v175[:3]
            v177 = v106(v138(4))
            v121.v107(v177)
            v178 = [v176[v108] for v108 in v177]
            v179 = [v224(v136, v14, v184, [v108 for v108 in v27.v225(' ' + v115).v182 if v108 != v29]) for v115 in v178]
            v125 += v8(v8(v244.v241(v179)) == v177.v233(0))
        return v125 / v210(1, v192(v38))
    v49 = v157.v185.v137(v46.v103(), lr=v91, weight_decay=0.01)
    v50 = v158.v83(v7 + 11)
    v51 = 0
    v52 = 40 if v13.v86 else 100
    for v53 in v138(1, v18 + 1):
        v139 = v186(v50)
        v140 = v46(input_ids=v139, labels=v139).v140
        v49.v187(set_to_none=True)
        v140.v188()
        v49.v53()
        v51 = v53
        if v53 % v52 == 0:
            v46.v102()
            v189 = v141(v46)
            v92(f'  gpt memorize A step {v53}: loss={v42(v140):.3f} recall={v189:.3f}')
            if v189 >= v26:
                v46.v130()
                break
            v46.v130()
    v46.v102()
    v54 = v141(v46)
    v55 = v142(v46)
    v92(f'gpt AFTER A memorize ({v51} steps): A_recall={v54:.3f} next_tok={v55:.3f} ({v85.v85() - v16:.0f}s)')
    v56 = v190.v143(v158.v83(v7 + 1), v13.v86)
    v144, v145 = v191.v146(v56, v27, v29, max_lines=v24, min_line_len=20)
    v57 = [v108 for v108 in v27.v225(v56[:200000]).v182 if v108 != v29]
    v58 = v159.v147(v33, v35)
    v59 = v159.v148(v31, v144, v145, v30, v29, v14, v20, v7 + 7)
    v60 = v104(v59, v95, v14)
    v120, v149 = v159.v150(v226(256).v100(v14), v159.v147(v60, v35), v58, v15, v21, v14)
    v61 = v127(v60, v41, v113, W_bwd=None)
    v62 = v127(v60, v41, v113, W_bwd=v120)
    v63 = v128(v31)
    v92(f'tape AFTER B (code shift + W): A_raw={v61:.3f} A_W={v62:.3f} align={v149:.3f} next_tok(frozen)={v63:.3f} ({v85.v85() - v16:.0f}s)')
    if v192(v57) < v90 + 2:
        raise v193('code corpus too short for GPT domain-B CE')
    v46.v130()
    v64 = v157.v185.v137(v46.v103(), lr=v25, weight_decay=0.01)
    v65 = v158.v83(v7 + 17)

    def code_batch(v50):
        v133 = []
        for v135 in v138(v89):
            v194 = v50.v212(v210(1, v192(v57) - v90 - 1))
            v133.v170(v57[v194:v194 + v90])
        return v157.v183(v133, device=v14)
    for v53 in v138(1, v19 + 1):
        v139 = v195(v65)
        v140 = v46(input_ids=v139, labels=v139).v140
        v64.v187(set_to_none=True)
        v140.v188()
        v64.v53()
        if v53 % v210(40, v19 // 4) == 0:
            v92(f'  gpt learn B step {v53}: loss={v42(v140):.3f}')
    v46.v102()
    v66 = v141(v46)
    v67 = v142(v46)
    v92(f'gpt AFTER B (code ft): A_recall={v66:.3f} next_tok={v67:.3f} ({v85.v85() - v16:.0f}s)')
    v68 = v54 - v66
    v69 = v62 - v66
    v70 = v44 >= 0.7 and v54 >= 0.7
    v71 = v62 >= 0.8
    v72 = v68 >= 0.15
    v73 = v69 >= 0.2
    v74 = 'CF_SEQUENTIAL_OK' if v70 and v71 and v72 and v73 else 'CF_SEQUENTIAL_PARTIAL' if v70 and v71 and (v72 or v69 >= 0.1) else 'CF_SEQUENTIAL_NO'
    v75 = {'stage': 239, 'overall': v74, 'gates': {'G_memorize_both_ge_0p70': v70, 'G_tape_retain_after_B_ge_0p80': v71, 'G_gpt_A_drop_ge_0p15': v72, 'G_gap_tape_minus_gpt_ge_0p20': v73}, 'n_facts': v17, 'tape': {'A_after_write': v44, 'A_after_B_raw_no_W': v61, 'A_after_B_with_W': v62, 'next_tok_before': v45, 'next_tok_frozen_after': v63, 'W_align': v149}, 'gpt': {'A_after_memorize': v54, 'A_after_B': v66, 'A_drop': v68, 'next_tok_after_memorize': v55, 'next_tok_after_B': v67, 'memorize_steps': v51, 'domain_B_steps': v19}, 'gap_tape_minus_gpt_after_B': v69, 'note': 'vs parametric GPT = CF capability; vs GPT+RAG = architectural (index can keep A)', 'timestamp': v235.v227(v236.v228).v196()}
    v4.v151(v213.v197(v75, indent=2), encoding='utf-8')
    v5.v151(f'# Stage 239 CF sequential A→B\n\n**{v74}** tape_A={v62:.3f} gpt_A={v66:.3f} drop_gpt={v68:.3f} gap={v69:.3f}\n', encoding='utf-8')
    v92(v213.v197(v75, indent=2))
    return 0
if v76 == '__main__':
    raise v152(v198())