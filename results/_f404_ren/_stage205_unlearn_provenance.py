"""
Stage 205 — W3: targeted unlearning, provenance, contradiction audit.

Claim under test: because facts live in explicit fp slots, TapeLM can (a) delete one fact in O(1)
with provably zero collateral, (b) attribute every answer to the slot/source it came from, and
(c) flag contradictions instead of silently answering. A parametric GPT can do none of these
without gradients — and gradient unlearning damages what it should not touch.

Controls:
  - GPT is FIRST fine-tuned to actually memorize the same facts (otherwise "unlearning" is vacuous),
    then unlearned by gradient ascent on the target facts with EARLY STOP (minimal, fairest damage).
  - Collateral measured identically for both: retained-fact recall + next_tok on exam v3 items.
  - Honest note recorded: GPT+RAG could also delete from its index and give provenance, so vs RAG
    this axis is architectural; vs parametric GPT it is capability.

Gates:
  G_forget      curve target recall drops to <= chance+0.05 after slot delete
  G_no_collat   curve retained recall delta <= 0.02 AND curve next_tok delta == 0
  G_gpt_collat  GPT unlearning shows collateral (retained recall or next_tok drop > 0.02)
  G_prov        curve provenance attribution >= 0.90
  G_conflict    conflict detection >= 0.80 with false-positive <= 0.20

  python _stage205_unlearn_provenance.py
"""
from __future__ import annotations
import copy
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
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
v0 = v23('results')
v1 = v23('data')
v2 = v23('checkpoints/stage191_p1_curve.pt')
v3 = v23('data/_wikitext103_train.txt')
v4 = v1 / 'stage191_exam_v3.jsonl'
v5 = v0 / 'stage205_decision.json'
v6 = v0 / 'stage205_mini.md'
v7 = v0 / '_stage205_log.txt'
v8 = 205
v9 = 20000000
v10 = 60
v11 = 20
v12 = 20
v13 = 300
v14 = 120
v15 = 800
v16 = 8
v17 = 64
v18 = 0.0003
v19 = 5e-05
v20 = 60
v21 = 0.25

def log(v24: v91) -> None:
    v25 = v24 if v24.v155('\n') else v24 + '\n'
    try:
        v156(v25, end='', flush=True)
    except v92:
        v156(v25.v237('ascii', 'replace').v228('ascii'), end='', flush=True)
    v7.v157.v93(parents=True, exist_ok=True)
    with v7.v158('a', encoding='utf-8') as v67:
        v67.v159(v25)

def main() -> v22:
    v0.v93(parents=True, exist_ok=True)
    v7.v94('', encoding='utf-8')
    v95(f'Stage205 start {v246.v240(v247.v241).v204()}')
    v95('W3: targeted unlearning / provenance / contradiction audit')
    v26 = v160.v26('cuda' if v160.v229.v208() else 'cpu')
    v27 = v161.v96(v8)
    v160.v97(v8)
    v28 = v98.v98()
    v99, v100, v101, v102 = v103()
    v29 = v162.v104(v91(v209.v163))
    v30 = v29.v105()
    v31 = v29.v164(v165) or 0
    v32 = v230.v210(v29, v101, v31, v30).v106(v26)
    v33 = v211(v102, v30).v106(v26)
    v33.v107(v160.v212(v2, map_location=v26, weights_only=False)['model'])
    v33.v108()
    for v34 in v33.v109():
        v34.v166(False)
    v35 = v110(v33, v101, v26)
    v95(f'curve loaded, frozen ({v98.v98() - v28:.0f}s)')
    with v3.v158('r', encoding='utf-8', errors='ignore') as v67:
        v111 = v67.v167(v9)
    v36 = [v34.v168() for v34 in v111.v213('\n') if v223(v34.v168()) > 300]
    v37 = v112(v214.v169((v127.v179(1) for v127 in v218.v242(v111) if v223(v127.v179(1)) >= 5)))
    v27.v113(v37)
    v38 = [v114 for v114 in v215(v231(v37), v27, v10 + v12 + 60) if v223(v114) >= 5]
    v39 = []
    for v40 in v115(v10):
        v116, v170 = (v38[v40], v37[v40])
        v39.v171({'S': v116, 'value': v170, 'sent': f'{v116} was appointed director of {v170} in 1987 .', 'fid': v40})
    v41 = v39[:v11]
    v42 = v39[v11:]
    v43 = {v67['S'] for v67 in v41}
    v44 = []
    for v45 in v115(v12):
        v116 = v38[v10 + v45]
        v172, v173 = (v37[v10 + 2 * v45], v37[v10 + 2 * v45 + 1])
        v44.v171({'S': v116, 'v1': v172, 'v2': v173})
    v95(f'facts={v223(v39)} (target={v223(v41)} retained={v223(v42)}) conflicts={v223(v44)} ({v98.v98() - v28:.0f}s)')

    def slot_of(v116, v117, v118, v119, v120):
        v121 = v35.v216([v116])[0]
        v122 = v35.v174(v117, exclude=v118)
        v123 = v232.v217(v121 + v122, dim=-1) if v122 is not None else v121
        return {'key': v123, 'value': v118, 'src': v119, 'kind': v120, 'subject': v116}
    v46 = [v175(v67['S'], v67['sent'], v67['value'], v67['fid'], 'fact') for v67 in v39]
    for v45, v124 in v125(v44):
        for v176, v177 in (('v1', v124['v1']), ('v2', v124['v2'])):
            v117 = f"{v124['S']} was appointed director of {v177} in 1987 ."
            v46.v171(v175(v124['S'], v117, v177, 10000 + v45, f'conflict_{v176}'))
    for v40, v126 in v125(v36[:v13]):
        v127 = v218.v178(v126)
        if not v127:
            continue
        v128 = v127.v179(1)
        v122 = v35.v174(v126[:400], exclude=v128)
        if v122 is None:
            continue
        v46.v171({'key': v232.v217(v35.v216([v128])[0] + v122, dim=-1), 'value': v128, 'src': 20000 + v40, 'kind': 'filler', 'subject': v128})
    v95(f'curve slots={v223(v46)} ({v98.v98() - v28:.0f}s)')
    v47 = v112(v214.v169((v137['value'] for v137 in v46)))

    def curve_recall(v129, v130):
        v131 = v160.v180([v137['key'] for v137 in v130], 0)
        v132 = v161.v96(v8 + 3)
        v133 = 0
        for v67 in v129:
            v181 = v35.v216([v67['S']])[0]
            v182 = (v131 @ v181).v219()
            v183 = {}
            for v137, v191 in v220(v130, v182):
                v183[v137['value']] = v193(v183.v243(v137['value'], -9.9), v191)
            v184 = [v149 for v149 in v47 if v149 != v67['value']]
            v132.v113(v184)
            v185 = [v67['value']] + v184[:3]
            v186 = v112(v115(4))
            v132.v113(v186)
            v187 = [v185[v40] for v40 in v186]
            v133 += v22(v22(v249.v233([v183.v243(v122, -9.9) for v122 in v187])) == v186.v244(0))
        return v133 / v193(1, v223(v129))

    def curve_provenance(v129, v130):
        v131 = v160.v180([v137['key'] for v137 in v130], 0)
        v133 = 0
        for v67 in v129:
            v181 = v35.v216([v67['S']])[0]
            v188 = v22((v131 @ v181).v233())
            v133 += v22(v130[v188]['src'] == v67['fid'])
        return v133 / v193(1, v223(v129))
    v48 = [v221.v189(v190) for v190 in v4.v245(encoding='utf-8').v222()]
    v49 = [v134 for v134 in v48 if v134['type'] == 'next_tok'][:v14]

    def curve_next_tok():
        v133 = 0
        for v134 in v49:
            v191 = [v234(v33, v32, v31, v134['ctx_ids'], v122, v26) for v122 in v134['cand_ids']]
            v133 += v22(v22(v249.v233(v191)) == v134['gold_idx'])
        return v133 / v223(v49)

    def gpt_next_tok(v65):
        v133 = 0
        for v134 in v49:
            v191 = [v235(v65, v26, v134['ctx_ids'], v122) for v122 in v134['cand_ids']]
            v133 += v22(v22(v249.v233(v191)) == v134['gold_idx'])
        return v133 / v223(v49)
    v50 = v46
    v51 = v135(v41, v50)
    v52 = v135(v42, v50)
    v53 = v136()
    v95(f'curve BEFORE: target={v51:.3f} retained={v52:.3f} next_tok={v53:.3f} ({v98.v98() - v28:.0f}s)')
    v54 = v98.v98()
    v55 = [v137 for v137 in v50 if not (v137['kind'] == 'fact' and v137['subject'] in v43)]
    v56 = v98.v98() - v54
    v57 = v135(v41, v55)
    v58 = v135(v42, v55)
    v59 = v136()
    v60 = v138(v42, v55)
    v95(f'curve AFTER delete ({v56 * 1000:.1f} ms, {v223(v50) - v223(v55)} slots): target={v57:.3f} retained={v58:.3f} next_tok={v59:.3f} provenance={v60:.3f} ({v98.v98() - v28:.0f}s)')

    def conflict_flags(v130, v139, v140=0.02):
        v131 = v160.v180([v137['key'] for v137 in v130], 0)
        v89 = []
        for v116 in v139:
            v181 = v35.v216([v116])[0]
            v182 = (v131 @ v181).v219()
            v183 = {}
            for v137, v191 in v220(v130, v182):
                v183[v137['value']] = v193(v183.v243(v137['value'], -9.9), v191)
            v188 = v236(v183.v48(), key=lambda v250: -v250[1])[:2]
            v89.v171(v223(v188) == 2 and v188[0][1] - v188[1][1] < v140)
        return v89
    v61 = v141(v55, [v122['S'] for v122 in v44])
    v62 = v141(v55, [v67['S'] for v67 in v42])
    v63 = v192(v61) / v193(1, v223(v61))
    v64 = v192(v62) / v193(1, v223(v62))
    v95(f'conflict audit: detection={v63:.3f} false_positive={v64:.3f} ({v98.v98() - v28:.0f}s)')
    v65 = v142(v26)
    v65 = v194.v143(v65)
    v65.v144()
    v66 = []
    for v67 in v39:
        v66.v171([v40 for v40 in v29.v237(v67['sent']).v195 if v40 != v31])
    v68 = [v40 for v40 in v29.v237(' '.v206(v36[300:600])[:200000]).v195 if v40 != v31]

    def ft_batch(v70, v145=None):
        v146 = []
        for v147 in v115(v16):
            if v145 is not None or v70.v161() < 0.5:
                v224 = v145 if v145 is not None else v66
                v225 = []
                while v223(v225) < v17:
                    v225 += v224[v70.v238(v223(v224))]
                v146.v171(v225[:v17])
            else:
                v137 = v70.v238(v193(1, v223(v68) - v17 - 1))
                v146.v171(v68[v137:v137 + v17])
        return v160.v196(v146, device=v26)

    def gpt_fact_recall(v129):
        v132 = v161.v96(v8 + 3)
        v133 = 0
        for v67 in v129:
            v197 = [v40 for v40 in v29.v237(f"{v67['S']} was appointed director of").v195 if v40 != v31]
            v184 = [v149 for v149 in v47 if v149 != v67['value']]
            v132.v113(v184)
            v185 = [v67['value']] + v184[:3]
            v186 = v112(v115(4))
            v132.v113(v186)
            v187 = [v185[v40] for v40 in v186]
            v191 = [v235(v65, v26, v197, [v40 for v40 in v29.v237(' ' + v122).v195 if v40 != v31]) for v122 in v187]
            v133 += v22(v22(v249.v233(v191)) == v186.v244(0))
        return v133 / v193(1, v223(v129))
    v69 = v160.v198.v148(v65.v109(), lr=v18, weight_decay=0.01)
    v70 = v161.v96(v8 + 11)
    for v71 in v115(1, v15 + 1):
        v149 = v199(v70)
        v150 = v65(input_ids=v149, labels=v149).v150
        v69.v200(set_to_none=True)
        v150.v201()
        v69.v71()
        if v71 % 200 == 0:
            v95(f'  gpt memorize step {v71}: loss={v248(v150):.3f} ({v98.v98() - v28:.0f}s)')
    v65.v108()
    v72 = v151(v41)
    v73 = v151(v42)
    v74 = v152(v65)
    v95(f'gpt AFTER memorize: target={v72:.3f} retained={v73:.3f} next_tok={v74:.3f} ({v98.v98() - v28:.0f}s)')
    v75 = [[v40 for v40 in v29.v237(v67['sent']).v195 if v40 != v31] for v67 in v41]
    v76 = v160.v198.v148(v65.v109(), lr=v19)
    v77 = v161.v96(v8 + 13)
    v65.v144()
    v78 = 0
    v79 = v98.v98()
    for v71 in v115(1, v20 + 1):
        v149 = v199(v77, only=v75)
        v150 = -v65(input_ids=v149, labels=v149).v150
        v76.v200(set_to_none=True)
        v150.v201()
        v160.v239.v226.v202(v65.v109(), 1.0)
        v76.v71()
        v78 = v71
        if v71 % 10 == 0:
            v65.v108()
            v203 = v151(v41)
            v95(f'  gpt unlearn step {v71}: target={v203:.3f} ({v98.v98() - v28:.0f}s)')
            v65.v144()
            if v203 <= v21 + 0.05:
                break
    v80 = v98.v98() - v79
    v65.v108()
    v81 = v151(v41)
    v82 = v151(v42)
    v83 = v152(v65)
    v95(f'gpt AFTER unlearn ({v78} grad steps, {v80:.1f}s): target={v81:.3f} retained={v82:.3f} next_tok={v83:.3f} ({v98.v98() - v28:.0f}s)')
    v84 = v57 <= v21 + 0.05
    v85 = v227(v58 - v52) <= 0.02 and v227(v59 - v53) < 1e-09
    v86 = v73 - v82 > 0.02 or v74 - v83 > 0.02
    v87 = v60 >= 0.9
    v88 = v63 >= 0.8 and v64 <= 0.2
    if v84 and v85 and v87 and v88 and v86:
        v153 = 'UNLEARN_PROVENANCE_WIN'
    elif v84 and v85 and (v87 or v88):
        v153 = 'UNLEARN_PARTIAL'
    else:
        v153 = 'UNLEARN_NO'
    v89 = {'timestamp': v246.v240(v247.v241).v204(), 'protocol': 'unlearn_provenance_audit_205', 'overall': v153, 'curve': {'target_recall': {'before': v51, 'after': v57}, 'retained_recall': {'before': v52, 'after': v58}, 'next_tok': {'before': v53, 'after': v59}, 'provenance_attribution': v60, 'delete_seconds': v56, 'slots_deleted': v223(v50) - v223(v55)}, 'gpt_parametric': {'target_recall': {'after_memorize': v72, 'after_unlearn': v81}, 'retained_recall': {'after_memorize': v73, 'after_unlearn': v82}, 'next_tok': {'after_memorize': v74, 'after_unlearn': v83}, 'unlearn_grad_steps': v78, 'unlearn_seconds': v80}, 'conflict_audit': {'detection': v63, 'false_positive': v64}, 'gates': {'g_forget': v84, 'g_no_collateral': v85, 'g_gpt_collateral': v86, 'g_provenance': v87, 'g_conflict': v88}, 'chance': v21, 'note': 'vs parametric GPT this is capability (no gradient-free deletion, no attribution); vs GPT+RAG it is architectural — a RAG index can also delete and attribute'}
    v5.v94(v221.v205(v89, indent=2, ensure_ascii=False), encoding='utf-8')
    v6.v94('\n'.v206(['# Stage205 — targeted unlearning / provenance / audit', '', f'**Overall:** `{v153}`', '', '| metric | curve before | curve after | GPT after memorize | GPT after unlearn |', '|--------|--------------|-------------|--------------------|-------------------|', f'| target fact recall | {v51:.3f} | **{v57:.3f}** | {v72:.3f} | {v81:.3f} |', f'| retained fact recall | {v52:.3f} | **{v58:.3f}** | {v73:.3f} | {v82:.3f} |', f'| next_tok (collateral) | {v53:.3f} | **{v59:.3f}** | {v74:.3f} | {v83:.3f} |', '', f'- curve delete: {v223(v50) - v223(v55)} slots in {v56 * 1000:.1f} ms, no gradient', f'- GPT unlearn: {v78} gradient steps, {v80:.1f} s', f'- provenance attribution (curve): {v60:.3f}', f'- conflict audit: detection {v63:.3f}, false-positive {v64:.3f}', f'- gates: forget={v84} no_collateral={v85} gpt_collateral={v86} prov={v87} conflict={v88}']), encoding='utf-8')
    v95(f'[205] {v153} | curve tgt {v51:.2f}->{v57:.2f} ret {v52:.2f}->{v58:.2f} nt {v53:.3f}->{v59:.3f} | gpt tgt {v72:.2f}->{v81:.2f} ret {v73:.2f}->{v82:.2f} nt {v74:.3f}->{v83:.3f}')
    return 0
if v90 == '__main__':
    raise v154(v207())