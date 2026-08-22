"""
Stage 268 — Unfreeze the mind: learn the tape *procedure*, not a fixed tape.

265 proved span-lock decode. After 256 the trunk stayed frozen as a *measurement*
discipline (slot delete kills the answer → fact not in weights). That discipline
quietly became architecture. The product invariant is weaker and sharper:

    no fact written *after deployment* enters the weights.

It does not require weights never change. Continual (253–255) already trained
upper layers with arc_enc frozen. 268 restores that mode on the 265 exam:

  - set_train_mode(m, "upper"): fast/slow/head learn; arc_enc frozen (hash check)
  - tape rebuilt every ~200 steps (new subjects, values, keys) — nothing factual
    survives across rebuilds, so memorizing a bank cannot explain novel-tape EM
  - decode = 265 span-lock, unchanged

Gates (priority):
  G_novel_tape        — EM on a never-seen tape ≥ EM on last train tape − 0.05
  G_arc_enc_frozen    — arc_enc hash unchanged
  G_beats_frozen_mind — trained upper beats init-upper with the same glue on novel tape
  G_no_param_leak     — empty tape EM ≤ 0.10
  G_slot_delete       — target dies, others live
  G_lang_intact       — hold CE does not rise

Verdict: MIND_LEARNS_TAPE_OK / _PARTIAL / _NO.

  python _stage268_mind_learns_tape.py --smoke
  python _stage268_mind_learns_tape.py          # night: 8000 steps, ~40 tapes
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
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
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
import _stage265_span_lock as s265
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, SlotBias, TapeView, hidden_and_logits
v0 = v18('results')
v1 = v0 / 'stage268_decision.json'
v2 = v0 / 'stage268_mini.md'
v3 = v0 / '_stage268_log.txt'
v4 = v0 / 'stage265_decision.json'
v5 = v18('checkpoints/stage191_p1_curve.pt')
v6 = v18('checkpoints/stage253_joint_l02.pt')
v7 = v18('checkpoints/stage268_mind_learns_tape.pt')
v8 = v18('data/_wikitext103_train.txt')
v9 = 268
v10 = v11
v12 = v13
v14 = v19.v14

def log(v20: v15) -> None:
    v21 = v20 if v20.v207('\n') else v20 + '\n'
    try:
        v208(v21, end='', flush=True)
    except v119:
        v208(v21.v313('ascii', 'replace').v291('ascii'), end='', flush=True)
    v3.v209.v120(parents=True, exist_ok=True)
    with v3.v210('a', encoding='utf-8') as v47:
        v47.v211(v21)

def fp_version() -> v15:
    v22 = v121(v122, 'canonical_fp_version', None)
    if v123(v22):
        try:
            return v15(v22())
        except v133:
            pass
    return v5.v23

def arc_enc_hash(v24: v124) -> v15:
    v25 = v212.v125()
    for v126, v127 in v128(v24.v307.v279().v213()):
        v25.v214(v127.v319().v318().v317().v308().v260())
    return v25.v129()

def published_265_em() -> v26 | None:
    if not v4.v215():
        return None
    try:
        v130 = v261.v216(v4.v262(encoding='utf-8'))
        v131 = v130.v263('arms') or {}
        v132 = v131.v263('B_soft_locked') or {}
        return v26(v132.v263('em_text', v132.v263('em', v26('nan'))))
    except v133:
        return None

def build_tape(*, v27: v134, v28, v29: v17, v30, v31: v217.v135, v32: v184[v15], v33: v184[v15], v34: v149[v15], v35: v17, v36: v17, v37: v17) -> v16:
    """Fresh planted facts + distractors. Subjects/values drawn from unused pool."""
    v38 = [v136 for v136 in v32 if v136 not in v34 and v218(v136) >= 5]
    if v218(v38) < v35 + v37 // 4:
        v38 = [v136 for v136 in v32 if v218(v136) >= 5]
    v31.v137(v38)
    v39 = [v136 for v136 in v264(v149(v34) | v149(v38), v31, v35 + v36 + 80) if v218(v136) >= 5]
    v39 = [v136 for v136 in v16.v241(v39) if v136 not in v34]
    v40 = [v136 for v136 in v264(v149(v34) | v149(v38) | v149(v39), v31, v36 + 40) if v218(v136) >= 6 and v136 not in v39 and (v136 not in v34)]
    v40 = v184(v16.v241(v40))[:v36]
    if v218(v39) < v35 + v218(v40) or v218(v38) < v35:
        raise v219(f'tape pool exhausted: subs={v218(v39)} avail={v218(v38)} need facts={v35} nonsense={v36}')
    v41 = []
    for v42 in v138(v35):
        v41.v220({'S': v39[v42], 'value': v38[v42], 'sent': v12.v292(S=v39[v42], V=v38[v42]), 'glue_train': v42 % 2 == 0, 'kind': 'wiki'})
        v34.v221(v38[v42])
        v34.v221(v39[v42])
    for v139, v140 in v141(v40):
        v142 = v39[v35 + v139]
        v41.v220({'S': v142, 'value': v140, 'sent': v12.v292(S=v142, V=v140), 'glue_train': False, 'kind': 'nonsense'})
        v34.v221(v140)
        v34.v221(v142)
    v43 = [v47 for v47 in v41 if v47['glue_train']]
    v44 = [v47 for v47 in v41 if not v47['glue_train'] and v47['kind'] == 'wiki']
    v45 = [v47 for v47 in v41 if v47['kind'] == 'nonsense']
    v46 = v44 + v45
    v143, v144 = ([], [])
    v145, v146 = ([], [])
    for v47 in v41:
        v147 = v27.v265([v47['S']])[0]
        v148 = v27.v222(v47['sent'], exclude=v47['value'])
        v143.v220(v309.v293(v147 + v148, dim=-1) if v148 is not None else v147)
        v144.v220(v47['value'])
    v48 = v149(v144)
    for v49 in v33:
        if v218(v144) >= v218(v41) + v37:
            break
        for v20 in v266.v223(v49):
            v224 = v20.v267(1)
            if v218(v224) < 5 or v224 in v48:
                continue
            v268, v269 = (v171(0, v20.v314() - 120), v294(v218(v49), v20.v315() + 120))
            v148 = v27.v222(v49[v268:v269], exclude=v224)
            if v148 is None:
                continue
            v225 = [v136 for v136 in v316.v310(v49[v268:v20.v314()]) if v136 != v224]
            if not v225:
                continue
            v143.v220(v309.v293(v27.v265([v225[-1]])[0] + v148, dim=-1))
            v226 = v27.v222(v49[v268:v20.v314()])
            if v226 is not None:
                v145.v220(v309.v293(v27.v265([v225[-1]])[0] + v226, dim=-1))
                v146.v220(v218(v144))
            v144.v220(v224)
            v48.v221(v224)
            v34.v221(v224)
            if v218(v144) >= v218(v41) + v37:
                break
    v50 = v150(v231.v295(v143, 0).v154(v30), v144, v28, v29)
    v51 = v231.v295(v145).v154(v30).v26() if v145 else None
    v52 = v231.v227(v146, device=v30) if v146 else None
    return {'tape': v50, 'fit_facts': v43, 'eval_facts': v46, 'eval_wiki': v44, 'eval_non': v45, 'nce_q': v51, 'nce_slot': v52, 'n_slots': v218(v144)}

def train_step(v53, v24, v54, v28, v55, v56, v57, v58, v59, v29, v60, v30, *, v61, v62, v63, v64, v31):
    v50 = v56['tape']
    v65 = v56['fit_facts']
    v66 = [v65[v31.v270(v218(v65))] for v126 in v138(v294(4, v218(v65)))]
    v151, v152 = v19.v153(v53, v24, v54, v28, v55, v50, v66, v29, v60, v30, v61, open_only=True)
    v67 = v296.v271(v57, v58, 1, v31, v29, v59).v154(v30)
    v155, v156 = v19.v157(v53, v24, v54, v28, v55, v50, v67, v29, v60, v30, v61, v62)
    v68 = None
    v51, v52 = (v56['nce_q'], v56['nce_slot'])
    if v51 is not None and v63 > 0:
        v158 = v50.v272.v26()
        v159 = v231.v228(0, v51.v273(0), (v294(64, v51.v273(0)),), device=v30)
        v160 = v309.v297(v52[v159], v158.v273(0)).v229()
        v68 = v63 * v19.v274(v53, v51[v159], v160, v158, v64)
    v69 = [v161 for v161 in (v151, v155, v68) if v161 is not None]
    if not v69:
        return None
    v70 = v69[0]
    for v71 in v69[1:]:
        v70 = v70 + v71
    return {'loss': v70, 'loss_fact': v26(v151) if v151 is not None else None, 'loss_prose': v26(v155) if v155 is not None else None, 'gate_fact': v152, 'gate_prose': v156}

def main() -> v17:
    v72 = v230.v162()
    v72.v163('--smoke', action='store_true')
    v72.v163('--steps', type=v17, default=0)
    v72.v163('--tape-period', type=v17, default=0, help='rebuild tape every N steps')
    v72.v163('--topk', type=v17, default=8)
    v72.v163('--gate-l1', type=v26, default=0.02)
    v72.v163('--nce-w', type=v26, default=1.0)
    v72.v163('--nce-tau', type=v26, default=0.05)
    v72.v163('--facts', type=v17, default=0)
    v72.v163('--nonsense-facts', type=v17, default=0)
    v72.v163('--distractor-slots', type=v17, default=0)
    v72.v163('--lr-glue', type=v26, default=0.003)
    v72.v163('--lr-upper', type=v26, default=3e-05)
    v72.v163('--open-thresh', type=v26, default=0.5)
    v72.v163('--reopen-margin', type=v26, default=0.1)
    v72.v163('--max-opens', type=v17, default=1)
    v73 = v72.v164()
    v3.v165('', encoding='utf-8')
    v30 = v231.v30('cuda' if v231.v298.v275() else 'cpu')
    v31 = v217.v135(v9)
    v231.v166(v9)
    v74 = v167.v167()
    v75 = v73.v75 or (400 if v73.v170 else 8000)
    v76 = v73.v76 or (100 if v73.v170 else 200)
    v35 = v73.v41 or (8 if v73.v170 else 48)
    v36 = v73.v168 or (4 if v73.v170 else 16)
    v37 = v73.v169 or (150 if v73.v170 else 1200)
    v77 = 6 if v73.v170 else 12
    v78 = 4 if v73.v170 else 12
    v79 = 400 if v73.v170 else 6000
    v61 = v73.v80
    v81 = v171(1, v75 // v76)
    v172(f'Stage268 mind-learns-tape start {v311.v305(v312.v306).v257()} device={v30} steps={v75} tape_period={v76} (~{v81} tapes) facts={v35} distractors={v37} lr_glue={v73.v204} lr_upper={v73.v205}')
    v126, v126, v173, v174 = v175()
    v28 = v232.v176(v15(v276.v233))
    v60 = v28.v177()
    v29 = v28.v234(v235) or 0
    v54 = v299.v277(v28, v173, v29, v60).v154(v30)
    v82 = v6 if v6.v236() else v5
    v24 = v124(v174, v60).v154(v30)
    v24.v178(v231.v278(v82, map_location=v30, weights_only=False)['model'])
    v237.v179(v24, 'upper')
    v83 = v180(v24)
    v172(f'  trunk={v82.v23} upper=TRAIN arc_enc=FROZEN hash0={v83[:16]}…')
    v84 = v124(v174, v60).v154(v30)
    v84.v178(v253.v238(v24.v279()))
    v84.v181()
    for v71 in v84.v182():
        v71.v239(False)
    v85 = v124(v174, v60).v154(v30)
    v85.v178(v231.v278(v5, map_location=v30, weights_only=False)['model'])
    v85.v181()
    for v71 in v85.v182():
        v71.v239(False)
    v27 = v134(v85, v173, v30)
    v172(f'  fp_version={v256()} (canonical keys always from frozen P1)')
    with v8.v210('r', encoding='utf-8', errors='ignore') as v47:
        v183 = v47.v240(1000000 if v73.v170 else 8000000)
    v32 = v184(v16.v241((v20.v267(1) for v20 in v266.v223(v183) if v218(v20.v267(1)) >= 5)))
    v31.v137(v32)
    v33 = [v281.v280() for v281 in v183.v300('\n') if v218(v281.v280()) >= 60][:v79]
    v172(f'  entity pool={v218(v32)} wiki_lines={v218(v33)}')
    v86 = '\n'.v185(v33 + [v14] * 32)
    v57, v58 = v237.v186(v86, v28, v29, max_lines=v79 + 64, min_line_len=20)
    v87 = v218(v58) - 1
    v88 = v184(v138(v171(1, v87 - v171(2, v87 // 20)), v87))
    v59 = v184(v138(0, v88[0]))
    v89 = v242.v187(v57, v58, v88, v29, v78, v9 + 5)
    v90 = v242.v188(v84, v89, v54, v29, v30)
    v172(f'  hold CE base (init upper)={v90:.4f}')
    v91 = 2 * (v24.v282.v243 // 2)
    v53 = v189(v91, v30)
    v92 = v231.v244.v190(v53.v245(), lr=v73.v204, weight_decay=0.01)
    v93 = v231.v244.v190([v71 for v71 in v24.v182() if v71.v283], lr=v73.v205, weight_decay=0.01)
    v34: v149[v15] = v149()
    v56 = None
    v94 = 0
    v95 = []
    v96 = v26('nan')

    def run_locked(v191, v192, v193, v194):
        return v19.v246(v192, v191, v54, v28, v27, v193, v194, v29, v60, v30, v61, v77, locked=True, open_thresh=v73.v284, reopen_margin=v73.v285, max_opens=v73.v286)
    for v97 in v138(1, v75 + 1):
        if v56 is None or (v97 - 1) % v76 == 0:
            v56 = v196(bank_can=v27, tok=v28, pad_id=v29, device=v30, rng=v31, values_pool=v32, lines=v33, used=v34, n_facts=v35, n_nonsense=v36, n_dist=v37)
            v94 += 1
            v172(f"  tape#{v94} @step {v97}: slots={v56['n_slots']} fit={v218(v56['fit_facts'])} eval={v218(v56['eval_facts'])} used_pool={v218(v34)}")
        v237.v179(v24, 'upper')
        v116 = v247(v53, v24, v54, v28, v27, v56, v57, v58, v59, v29, v60, v30, k=v61, gate_l1=v73.v62, nce_w=v73.v63, nce_tau=v73.v64, rng=v31)
        if v116 is None:
            continue
        v92.v248(set_to_none=True)
        v93.v248(set_to_none=True)
        v116['loss'].v249()
        v231.v301.v287.v250(v184(v53.v245()) + [v71 for v71 in v24.v182() if v71.v283], 1.0)
        v92.v97()
        v93.v97()
        if v97 % v171(1, v75 // 10) == 0 or v97 == v75:
            v24.v181()
            with v231.v302():
                v288 = v195(v24, v53, v56['tape'], v56['eval_facts'])
            v96 = v26(v288['em'])
            v95.v220({'step': v97, 'tape': v94, 'em_train_tape': v96, 'loss_fact': v116['loss_fact'], 'loss_prose': v116['loss_prose'], 'gate_fact': v116['gate_fact'], 'gate_prose': v116['gate_prose']})
            v172(f"  step {v97}/{v75} tape#{v94} em_train={v96:.3f} fact={v116['loss_fact']} prose={v116['loss_prose']} ({v167.v167() - v74:.0f}s)")
            v237.v179(v24, 'upper')
    v53.v181()
    v24.v181()
    v98 = v180(v24)
    v99 = v83 == v98
    v172(f'  arc_enc hash match={v99} ({v83[:12]}… vs {v98[:12]}…)')
    v100 = v195(v24, v53, v56['tape'], v56['eval_facts'])
    v172(f"  last-train-tape EM={v100['em']:.3f} verbatim={v100['verbatim']:.3f}")
    v101 = v196(bank_can=v27, tok=v28, pad_id=v29, device=v30, rng=v217.v135(v9 + 99), values_pool=v32, lines=v33, used=v34, n_facts=v35, n_nonsense=v36, n_dist=v37)
    v102 = v195(v24, v53, v101['tape'], v101['eval_facts'])
    v103 = v195(v84, v53, v101['tape'], v101['eval_facts'])
    v172(f"  novel-tape EM live={v102['em']:.3f} frozen_upper={v103['em']:.3f} slots={v101['n_slots']}")
    v104 = v195(v24, v53, v101['tape'].v251(), v101['eval_facts'])
    v105 = v195(v24, v53, v101['tape'].v252(v9 + 1), v101['eval_facts'])
    v197, v198 = ([], [])
    for v47 in v101['eval_facts']:
        v199 = v101['tape'].v253()
        v199.v254(v47['value'])
        v197.v220(v195(v24, v53, v199, [v47])['em'])
        v200 = [v255 for v255 in v101['eval_facts'] if v255 is not v47]
        if v200:
            v198.v220(v195(v24, v53, v199, v200)['em'])
    v106 = v26(v303.v289(v197)) if v197 else v26('nan')
    v107 = v26(v303.v289(v198)) if v198 else v26('nan')
    v108 = v242.v188(v24, v89, v54, v29, v30)
    v172(f"  hold CE after={v108:.4f} (base={v90:.4f}) empty_em={v104['em']:.3f}")
    v109 = v201()
    v110 = not v304.v290(v102['em']) and (not v304.v290(v100['em'])) and (v102['em'] >= v100['em'] - 0.05)
    v111 = v102['em'] >= v103['em'] + 0.05
    v112 = v104['em'] <= 0.1
    v113 = v102['em'] >= 0.4 and v106 <= 0.1 and (v304.v290(v107) or v107 >= 0.7 * v102['em'])
    v114 = v108 <= v90 + 0.05
    v115 = v105['em'] <= v171(0.1, v102['em'] - 0.4)
    if v110 and v99 and v111 and v112 and v113 and v114:
        v202 = 'MIND_LEARNS_TAPE_OK'
    elif v110 and v99 and (v111 or v112):
        v202 = 'MIND_LEARNS_TAPE_PARTIAL'
    else:
        v202 = 'MIND_LEARNS_TAPE_NO'
    v231.v203({'model': v24.v279(), 'glue': v53.v279(), 'stage': 268, 'steps': v75, 'n_tapes': v94, 'arc_enc_hash': v98}, v7)
    v116 = {'stage': 268, 'overall': v202, 'smoke': v73.v170, 'seed': v9, 'trunk': v82.v23, 'fp_version': v256(), 'steps': v75, 'tape_period': v76, 'n_tapes': v94, 'distractor_slots': v37, 'lr_glue': v73.v204, 'lr_upper': v73.v205, 'gates': {'G_novel_tape': v110, 'G_arc_enc_frozen': v99, 'G_beats_frozen_mind': v111, 'G_no_param_leak': v112, 'G_slot_delete': v113, 'G_lang_intact': v114, 'G_tape_causal': v115}, 'headline': {'em_last_train_tape': v100['em'], 'em_novel_tape': v102['em'], 'em_novel_frozen_upper': v103['em'], 'delta_novel_minus_train': v102['em'] - v100['em'], 'delta_live_minus_frozen': v102['em'] - v103['em'], 'em_265_published': v109}, 'controls': {'em_empty_tape': v104['em'], 'em_shuffled_tape': v105['em'], 'em_target_after_delete': v106, 'em_retained_after_delete': v107, 'hold_ce_base': v90, 'hold_ce_after': v108, 'arc_enc_hash_before': v83, 'arc_enc_hash_after': v98}, 'train_tape': {'em': v100['em'], 'em_span': v100['em_span'], 'em_text': v100['em_text'], 'verbatim': v100['verbatim'], 'open_recall': v100['open_recall'], 'n_eval': v218(v56['eval_facts']), 'n_slots': v56['n_slots']}, 'novel_tape': {'em': v102['em'], 'em_span': v102['em_span'], 'em_text': v102['em_text'], 'verbatim': v102['verbatim'], 'open_recall': v102['open_recall'], 'n_eval': v218(v101['eval_facts']), 'n_slots': v101['n_slots'], 'frozen_upper_em': v103['em']}, 'curve': v95, 'note': 'Upper trunk learns; arc_enc frozen (hash). Tape rebuilt every tape_period steps so no planted fact survives across rebuilds. G_novel_tape is the claim: procedure transfers to a bank never seen in training. G_beats_frozen_mind compares the same glue on novel tape with live vs init upper. Decode is 265 span-lock.', 'timestamp': v311.v305(v312.v306).v257(), 'wall_s': v167.v167() - v74}
    v1.v165(v261.v258(v116, indent=2), encoding='utf-8')
    v117 = f"# Stage 268 mind learns tape\n\n**{v202}** · steps={v75} · tapes={v94} · bank≈{v37}+facts{(' · SMOKE' if v73.v170 else '')}\n\n| exam | EM | frozen-upper EM |\n|------|---:|----------------:|\n| last train tape | {v100['em']:.3f} | — |\n| novel tape | **{v102['em']:.3f}** | {v103['em']:.3f} |\n\n## Gates (read G_novel_tape first)\n\n- G_novel_tape: **{v110}** (novel {v102['em']:.3f} vs train {v100['em']:.3f})\n- G_arc_enc_frozen: **{v99}**\n- G_beats_frozen_mind: **{v111}**\n- G_no_param_leak: **{v112}** (empty={v104['em']:.3f})\n- G_slot_delete: **{v113}**\n- G_lang_intact: **{v114}** (hold {v90:.3f}→{v108:.3f})\n"
    v2.v165(v117, encoding='utf-8')
    v172(v261.v258({'overall': v202, 'gates': v116['gates'], 'headline': v116['headline']}, indent=2))
    v172(f'wrote {v1} wall={v167.v167() - v74:.0f}s')
    return 0
if v118 == '__main__':
    raise v206(v259())