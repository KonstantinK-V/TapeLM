"""
Stage 258 — Semantic query: let the trunk's understanding reach the retrieval.

209 measured something that has been sitting unused: curve states carry MiniLM geometry
at least as well as matched GPT (r 0.256 vs 0.270), and PAWS tracks GPT at every scale
(STRUCTURAL_BLOCK_NO). Meanwhile the retrieval query never sees the trunk at all:

    q = normalize( W_q( ctx_fp(prefix) [+ fp(anchor)] ) )      W_q maps fp -> fp

`ctx_fp` is a MEAN over word fingerprints — a bag of spellings, no word order, no meaning.
`h_t` only reaches the read gate ("should I look at the tape now?"), never the query
("what am I looking for?"). So the tape can only be asked with the literal anchor string.

This stage adds one channel and changes nothing else:

    q_fp  = normalize( W_q(fp query) )                 existing path, untouched
    q_sem = normalize( W_sem(h_t) )                    trunk understanding -> key space
    q     = normalize( (1 - a) * q_fp + a * q_sem )    a = sigmoid(MLP(h_t)), starts ~0.12

Keys stay canonical frozen fp, P1 stays frozen, the trunk stays frozen. Only W_q, W_sem
and the blend train. At a=0 this is EXACTLY stage 256, so the baseline is a special case
of the model rather than a separate implementation.

The exam is built so the fp path CANNOT win, by construction. One subject S carries FOUR
facts with different relations. Every one of those slots has the SAME anchor fp(S), so the
anchor contributes nothing to telling them apart, and the query paraphrases the relation
with NO content word in common with the written sentence:

    written : "{S} was appointed director of {V} in the regional chronicle of 1987 ."
    asked   : "the body that {S} led was named"          (led / body / named vs appointed / director)

A bag of spellings cannot bridge "led" to "appointed director". Trunk semantics might.
Chance is 1/4 and fp-only is PRE-REGISTERED to sit there (G_fp_only_at_chance) — if it
does not, the exam leaks and the rest of the numbers mean nothing.

Held out twice, so a win cannot be four memorised templates:
    seen_rel       unseen SUBJECTS, paraphrase A (fit during training)
    unseen_para    unseen SUBJECTS, paraphrase B of the SAME relations (258c — not held-out
                   relations, which were structurally penalized as InfoNCE negatives)

Matched GPT-2 control (the 210-212 lesson): those stages closed as THESIS_NO on a single
scale with no control, which reads as "impossible" when the evidence only said "not here".
Here the same semantic channel is trained on GPT-2 states too. If curve fails and GPT also
fails, the verdict is SEM_QUERY_NO_AT_SCALE, not SEM_QUERY_NO — a statement about a 3050,
not about the architecture. If GPT succeeds where curve does not, that IS architectural and
the verdict says so.

Retrieval only: whether a won slot reaches the output is already 256's result, so this
stage does not touch decode and cannot break language.

  python _stage258_semantic_query.py [--smoke] [--no-gpt-control]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, hidden_and_logits
v0 = v15('results')
v1 = v0 / 'stage258_decision.json'
v2 = v0 / 'stage258_mini.md'
v3 = v0 / '_stage258_log.txt'
v4 = v15('checkpoints/stage191_p1_curve.pt')
v5 = v15('checkpoints/stage253_joint_l02.pt')
v6 = v15('checkpoints/stage258_semantic_query.pt')
v7 = v15('data/_wikitext103_train.txt')
v8 = 258
v9 = {'lead': {'write': '{S} was appointed director of {V} in the regional chronicle of 1987 .', 'para': 'the body that {S} led was named', 'para_b': 'the organisation under {S} leadership was called', 'para_hold': 'which institution {S} commanded was known as', 'anchored': 'in the ledger {S} was appointed director of'}, 'birth': {'write': '{S} was born in {V} according to the parish register .', 'para': 'the birthplace recorded for {S} is', 'para_b': 'records show {S} entered the world in', 'para_hold': 'the city listed as birthplace of {S} is', 'anchored': 'in the ledger {S} was born in'}, 'death': {'write': '{S} died in {V} at the close of the season .', 'para': 'the place where {S} passed away is', 'para_b': 'the locale of {S} death is given as', 'para_hold': 'where {S} drew their final breath was', 'anchored': 'in the ledger {S} died in'}, 'marriage': {'write': '{S} married in {V} during the spring assembly .', 'para': 'the wedding of {S} was held in', 'para_b': 'the town of {S} nuptials was', 'para_hold': 'the municipality hosting {S} wedding was', 'anchored': 'in the ledger {S} married in'}, 'work': {'write': '{S} worked at {V} for eleven consecutive terms .', 'para': 'the employer of {S} was listed as', 'para_b': 'the workplace associated with {S} was', 'para_hold': 'the firm recorded as employer of {S} was', 'anchored': 'in the ledger {S} worked at'}, 'prison': {'write': '{S} was jailed in {V} following the tribunal .', 'para': 'the prison that confined {S} stood in', 'para_b': 'the gaol holding {S} was situated in', 'para_hold': 'the detention site holding {S} was located in', 'anchored': 'in the ledger {S} was jailed in'}, 'study': {'write': '{S} studied at {V} before the reorganisation .', 'para': 'the school that taught {S} is called', 'para_b': 'the college attended by {S} was', 'para_hold': 'the academy {S} attended was known as', 'anchored': 'in the ledger {S} studied at the'}, 'burial': {'write': '{S} was buried in {V} after the civic ceremony .', 'para': 'the grave of {S} lies in', 'para_b': 'the resting place of {S} is recorded as', 'para_hold': 'the cemetery where {S} was interred is in', 'anchored': 'in the ledger {S} was buried in'}}
v10 = ['lead', 'birth', 'death', 'marriage', 'work', 'prison']
v11 = 1.0 / v139(v9)

def log(v16: v140) -> None:
    v17 = v16 if v16.v230('\n') else v16 + '\n'
    try:
        v231(v17, end='', flush=True)
    except v141:
        v231(v17.v337('ascii', 'replace').v332('ascii'), end='', flush=True)
    v3.v232.v142(parents=True, exist_ok=True)
    with v3.v233('a', encoding='utf-8') as v143:
        v143.v234(v17)

class SemQuery(v18.v12):
    """Trunk state -> key space, plus how much to trust it. Starts near a=0.12 so the fp path
    dominates at init and the semantic channel has to earn its weight against CE."""

    def __init__(v144, v145: v14, v23):
        v333().v235()
        v144.v146 = v18.v334(v145, 256).v205(v23)
        v144.v147 = v18.v335(v18.v334(v145 + 2, 64), v18.v351(), v18.v334(64, 1)).v205(v23)
        v18.v298.v236(v144.v147[-1].v237)
        v18.v298.v238(v144.v147[-1].v239, -2.0)

    def q(v144, v44: v27.v13) -> v27.v13:
        return v260.v240(v144.v146(v44), dim=-1)

    def a(v144, v44: v27.v13, v40: v27.v13) -> v27.v13:
        return v27.v336(v144.v147(v27.v178([v44, v40], dim=-1))).v156(-1)

@v27.v26()
def curve_state(v19, v20, v21, v22, v23, v24: v140) -> v27.v13:
    v25 = [v151 for v151 in v21.v337(v24).v25 if v151 != v22][-v299:]
    if not v25:
        return None
    v44, v148 = v149(v19, v20, v27.v187([v25], device=v23), v22)
    return v44[0, -1].v150()

@v27.v26()
def gpt_state(v28, v21, v22, v23, v24: v140) -> v27.v13:
    v25 = [v151 for v151 in v21.v337(v24).v25 if v151 != v22]
    v29 = v241.v152(v28, v21, v22, v23, v25)
    return None if v29 is None else v29.v150()

def fp_query_raw(v30: v153, v24: v140):
    """Exactly the 256 recipe: anchor fingerprint + bag-of-spellings context."""
    v31 = v30.v154(v24)
    if v31 is None:
        return None
    v32 = v242.v155(v24)
    return v260.v240(v30.v340([v32[-1]])[0] + v31, dim=-1) if v32 else v31

def fp_confidence(v33: v27.v13, v34: v27.v13) -> v27.v13:
    """[top1, margin] of the fp-only query. Raw margins are ~0.02–0.05 — scale so the blend MLP
    can see them next to h_t. Batch callers may z-score on top of this."""
    if v33.v243() == 1:
        v33 = v33.v244(0)
        v156 = True
    else:
        v156 = False
    v35 = v33 @ v34.v111()
    v36 = v27.v245(v35, v175(2, v35.v246(-1)), dim=-1).v37
    v38 = v36[..., 0]
    v39 = v38 - v36[..., 1] if v36.v246(-1) > 1 else v38
    v40 = v27.v157([v38 * 2.0 - 1.0, v39 * 20.0], dim=-1)
    return v40[0] if v156 else v40

def zscore_conf(v40: v27.v13) -> v27.v13:
    """Per-feature z-score over the batch so relative fp-margin drives the blend."""
    if v40.v246(0) < 2:
        return v40
    return (v40 - v40.v300(0, keepdim=True)) / (v40.v301(0, keepdim=True) + 1e-05)

def blended_query(v41, v42: v158, v43, v44, v34=None):
    """(1-a) * W_q(fp) + a * W_sem(h). a=0 reproduces stage 256 bit for bit."""
    v33 = v260.v240(v41(v43.v244(0)), dim=-1)[0]
    if v42 is None or v44 is None:
        return (v33, v27.v261((), device=v33.v23))
    v45 = v42.v45(v44, v247(v33, v34))
    return (v260.v240((1.0 - v45) * v33 + v45 * v42.v62(v44), dim=-1), v45)

def make_subjects(v46, v47, v48: v14, v49: v14):
    """One subject, four relations, four distinct values -> four slots with the same anchor fp."""
    v50 = [v248 for v248 in v338(v282(v46), v47, v48 + 40) if v139(v248) >= 5][:v48]
    v51 = [v249 for v249 in v46 if v139(v249) >= 5][:v49]
    v52 = []
    for v151, v159 in v160(v50):
        v161 = {}
        for v63 in v9:
            v161[v63] = v51[v47.v339(v139(v51))]
        v52.v250({'S': v159, 'facts': v161, 'sid': f's{v151}'})
    return v52

def build_tape(v30: v153, v53, v23):
    v162, v37, v163 = ([], [], [])
    for v54 in v53:
        for v63 in v9:
            v249 = v54['facts'][v63]
            v251 = v9[v63]['write'].v302(S=v54['S'], V=v249)
            v31 = v30.v154(v251, exclude=v249)
            v252 = v30.v340([v54['S']])[0]
            v162.v250(v260.v240(v252 + v31, dim=-1) if v31 is not None else v252)
            v37.v250(v249)
            v163.v250((v54['sid'], v63))
    return (v162, v37, v163)

def queries_for(v53, v55, v56: v140):
    return [{'sid': v54['sid'], 'S': v54['S'], 'rel': v63, 'kind': v56, 'text': v9[v63][v56].v302(S=v54['S']), 'value': v54['facts'][v63]} for v54 in v53 for v63 in v55]

@v27.v26()
def evaluate(v41, v42, v30, v34, v57, v58, v59, v60: v164):
    """Primary metric: among the FOUR slots of this subject, does the asked relation win?
    Chance is 1/4 by construction. Bank-wide rank is reported so a semantic channel that
    wins locally by wrecking global retrieval cannot pass unnoticed."""
    v165, v166, v167, v168 = ([], [], [], 0)
    v61: v169[v140, v169[v140, v14]] = {}
    for v62 in v58:
        v170 = v253(v30, v62['text'])
        v44 = v59.v263(v62['text']) if v60 else None
        if v170 is None:
            v168 += 1
            continue
        v254, v45 = v255(v41, v42 if v60 else None, v170, v44, v34)
        v35 = v34 @ v254
        v171 = v57[v62['sid']]
        v172 = v256(v171, key=lambda v218: v181(v35[v218[1]]))
        v165.v250(v14(v172[0] == v62['rel']))
        v61.v257(v62['rel'], {}).v257(v172[0], 0)
        v61[v62['rel']][v172[0]] += 1
        v65 = [v218 for v63, v218 in v171 if v63 == v62['rel']][0]
        v166.v250(1 + v14((v35 > v35[v65]).v352()))
        v167.v250(v181(v45))
    v63 = v303.v258(v166, dtype=v303.v304) if v166 else v303.v258([v303.v305])
    return {'sel_acc': v181(v303.v300(v165)) if v165 else v181('nan'), 'bank_top1': v181(v303.v300(v63 == 1)), 'bank_mrr': v181(v303.v300(1.0 / v63)), 'alpha': v181(v303.v300(v167)) if v167 else v181('nan'), 'n': v139(v165), 'skipped': v168, 'confusion': v61}

def hard_neg_ce(v64: v27.v13, v65: v27.v13, v66: v14) -> v27.v13:
    """CE over {gold} ∪ top-k hardest bank slots (by current logits). Focuses gradient on
    confusable keys instead of diluting across the whole bank softmax."""
    v173, v174 = v64.v67
    v68 = v175(v256(1, v66), v256(1, v174 - 1))
    v69 = v64.v150().v176()
    v69.v177(1, v65.v259(-1, 1), v181('-inf'))
    v70 = v69.v245(v68, dim=-1).v71
    v72 = v27.v178([v65.v259(-1, 1), v70], dim=1)
    v73 = v64.v179(1, v72)
    return v260.v180(v73, v27.v261(v173, dtype=v27.v306, device=v64.v23))

def train_channel(v41, v42, v30, v34, v74, v59, v75, v76, v77, v78, v79, v47, v80, *, v81: v181=0.0, v66: v14=32):
    """InfoNCE with hard-negative mining over the bank + wiki grounding for W_q."""
    v82 = v210(v42.v208()) + v210(v41.v208())
    v83 = v27.v262.v182(v82, lr=v78, weight_decay=0.01)
    v183, v184, v185, v186 = ([], [], [], [])
    for v62 in v74:
        v170 = v253(v30, v62['text'])
        v44 = v59.v263(v62['text'])
        if v170 is None or v44 is None:
            continue
        v183.v250(v170)
        v184.v250(v44)
        v185.v250(v62['gold_idx'])
        v186.v250(v62.v263('kind', 'para'))
    if not v183:
        return v181('nan')
    v84 = v27.v157(v183)
    v85 = v27.v157(v184)
    v86 = v27.v187(v185, device=v34.v23)
    v87 = v27.v187([v68 == 'anchored' for v68 in v186], device=v34.v23)
    v88 = []
    for v89 in v188(1, v77 + 1):
        v165 = v27.v264(0, v84.v246(0), (v175(32, v84.v246(0)),), device=v34.v23)
        v33 = v260.v240(v41(v84[v165]), dim=-1)
        v40 = v265(v247(v33, v34))
        v45 = v42.v45(v85[v165], v40).v244(-1)
        v62 = v260.v240((1.0 - v45) * v33 + v45 * v42.v62(v85[v165]), dim=-1)
        v64 = v62 @ v34.v111() / v79
        v189 = 0.5 * v260.v180(v64, v86[v165]) + 0.5 * v341(v64, v86[v165], v66)
        if v81 > 0:
            with v27.v26():
                v307 = (v33 @ v34.v111()).v353(dim=-1) == v86[v165]
            v266 = v87[v165] & v307
            if v266.v308():
                v189 = v189 + v81 * v45.v156(-1)[v266].v300()
        if v75 is not None:
            v248 = v27.v264(0, v75.v246(0), (v175(64, v75.v246(0)),), device=v34.v23)
            v267 = v260.v240(v41(v75[v248]), dim=-1)
            v268 = v267 @ v34.v111() / v79
            v189 = v189 + 0.5 * v260.v180(v268, v76[v248]) + 0.5 * v341(v268, v76[v248], v66)
        v83.v269(set_to_none=True)
        v189.v270()
        v27.v18.v309.v271(v82, 1.0)
        v83.v89()
        v88.v250(v181(v189))
        if v89 % v256(1, v77 // 5) == 0:
            v199(f'  [{v80}] step {v89}/{v77} loss={v181(v189):.3f} a={v181(v45.v300()):.3f}')
    return v181(v303.v300(v88[-20:])) if v88 else v181('nan')

def main() -> v14:
    v90 = v272.v190()
    v90.v191('--smoke', action='store_true')
    v90.v191('--steps', type=v14, default=0)
    v90.v191('--subjects', type=v14, default=0)
    v90.v191('--distractor-slots', type=v14, default=0)
    v90.v191('--tau', type=v181, default=0.05)
    v90.v191('--lr', type=v181, default=0.002)
    v90.v191('--blend-l1', type=v181, default=0.25, help='L1 on blend a for anchored fit queries where fp-only already hits gold')
    v90.v191('--k-hard', type=v14, default=32, help='top-k hard negatives for InfoNCE mining')
    v90.v191('--no-gpt-control', action='store_true')
    v91 = v90.v192()
    v3.v193('', encoding='utf-8')
    v23 = v27.v23('cuda' if v27.v342.v310() else 'cpu')
    v47 = v273.v194(v8)
    v27.v195(v8)
    v92 = v196.v196()
    v77 = v91.v77 or (150 if v91.v198 else 600)
    v48 = v91.v53 or (12 if v91.v198 else 64)
    v93 = v91.v197 or (150 if v91.v198 else 1200)
    v94 = 400 if v91.v198 else 6000
    v199(f'Stage258 semantic query start {v354.v348(v355.v349).v294()} device={v23} steps={v77} subjects={v48} distractors={v93} chance={v11:.3f}')
    v148, v148, v200, v201 = v202()
    v21 = v274.v203(v140(v311.v275))
    v95 = v21.v204()
    v22 = v21.v276(v277) or 0
    v20 = v343.v312(v21, v200, v22, v95).v205(v23)
    v96 = v5 if v5.v278() else v4
    v19 = v313(v201, v95).v205(v23)
    v19.v206(v27.v314(v96, map_location=v23, weights_only=False)['model'])
    v19.v207()
    for v97 in v19.v208():
        v97.v279(False)
    v98 = v313(v201, v95).v205(v23)
    v98.v206(v27.v314(v4, map_location=v23, weights_only=False)['model'])
    v98.v207()
    for v97 in v98.v208():
        v97.v279(False)
    v30 = v153(v98, v200, v23)
    v199(f'  trunk={v96.v228} (frozen)  keys=canonical P1 (frozen)')
    with v7.v233('r', encoding='utf-8', errors='ignore') as v143:
        v209 = v143.v280(1000000 if v91.v198 else 6000000)
    v46 = v210(v169.v281((v16.v318(1) for v16 in v317.v283(v209) if v139(v16.v318(1)) >= 5)))
    v47.v211(v46)
    v99 = [v316.v315() for v316 in v209.v344('\n') if v139(v316.v315()) >= 60][:v94]
    v53 = v212(v46, v47, v48, v175(v139(v46), 400))
    v100 = v48 // 2
    v213, v214 = (v53[:v100], v53[v100:])
    v199(f'  subjects: fit={v139(v213)} held_out={v139(v214)}  rels fit={v10}')
    v162, v37, v163 = v215(v30, v53, v23)
    v101 = v139(v37)
    v102 = {v54['S'] for v54 in v53} | v282(v37)
    v216, v217 = ([], [])
    for v103 in v99:
        if v139(v37) >= v101 + v93:
            break
        for v16 in v317.v283(v103):
            v284 = v16.v318(1)
            if v139(v284) < 5 or v284 in v102:
                continue
            v319, v320 = (v256(0, v16.v356() - 120), v175(v139(v103), v16.v357() + 120))
            v31 = v30.v154(v103[v319:v320], exclude=v284)
            if v31 is None:
                continue
            v32 = [v248 for v248 in v242.v155(v103[v319:v16.v356()]) if v248 != v284]
            if not v32:
                continue
            v285 = v30.v340([v32[-1]])[0]
            v162.v250(v260.v240(v285 + v31, dim=-1))
            v286 = v30.v154(v103[v319:v16.v356()])
            if v286 is not None:
                v216.v250(v260.v240(v285 + v286, dim=-1))
                v217.v250(v139(v37))
            v37.v250(v284)
            v163.v250((None, None))
            v102.v321(v284)
            if v139(v37) >= v101 + v93:
                break
    v34 = v27.v157(v162, 0).v205(v23).v181()
    v199(f'  tape slots={v139(v37)} ({v101} subject facts + {v139(v37) - v101} wiki noise)')
    v57: v169[v140, v210] = {}
    for v218, (v287, v63) in v160(v163):
        if v287 is not None:
            v57.v257(v287, []).v250((v63, v218))
    v104 = {(v287, v63): v218 for v218, (v287, v63) in v160(v163) if v287 is not None}
    v74 = v219(v213, v10, 'para') + v219(v213, v10, 'para_b') + v219(v213, v10, 'anchored')
    for v62 in v74:
        v62['gold_idx'] = v104[v62['sid'], v62['rel']]
    v105 = v219(v214, v10, 'para')
    v106 = v219(v214, v10, 'para_hold')
    v107 = v219(v214, v210(v9), 'anchored')
    v199(f'  queries: fit={v139(v74)} seen_rel={v139(v105)} unseen_para={v139(v106)} anchored={v139(v107)}')
    v108 = v74 + v105 + v106 + v107
    v109 = v220({v62['text'] for v62 in v108})
    v110 = {}
    for v111 in v109:
        v221 = v288(v19, v20, v21, v22, v23, v111)
        if v221 is not None:
            v110[v111] = v221.v181()
    v112 = v322(v345(v110.v37())).v222()
    v199(f'  cached curve states: {v139(v110)} (dim {v112})')
    v75 = v27.v157(v216).v205(v23).v181() if v216 else None
    v76 = v27.v187(v217, device=v23) if v217 else None
    v41 = v241.v223(v23)
    v42 = v158(v112, v23)
    v113 = v34.v176()
    v114 = v224(v41, None, v30, v34, v57, v105, v110, use_sem=False)
    v115 = v224(v41, None, v30, v34, v57, v106, v110, use_sem=False)
    v116 = v224(v41, None, v30, v34, v57, v107, v110, use_sem=False)
    v199(f"fp-only (a=0, i.e. stage 256): seen_rel={v114['sel_acc']:.3f} unseen_para={v115['sel_acc']:.3f} anchored={v116['sel_acc']:.3f} (chance {v11:.3f})")
    v117 = v225(v41, v42, v30, v34, v74, v110, v75, v76, v77, v91.v78, v91.v79, v47, 'curve', blend_l1=v91.v81, k_hard=v91.v66)
    v118 = v224(v41, v42, v30, v34, v57, v105, v110, use_sem=True)
    v119 = v224(v41, v42, v30, v34, v57, v106, v110, use_sem=True)
    v120 = v224(v41, v42, v30, v34, v57, v107, v110, use_sem=True)
    v199(f"curve+sem: seen_rel={v118['sel_acc']:.3f} unseen_para={v119['sel_acc']:.3f} anchored={v120['sel_acc']:.3f} | alpha para={v119['alpha']:.3f} anchored={v120['alpha']:.3f} | bank_top1={v119['bank_top1']:.3f}")
    v121 = v164(v27.v289(v34, v113))
    v122 = v34[v27.v290(v34.v246(0), generator=v27.v358().v195(v8 + 1))]
    v123 = v224(v41, v42, v30, v122, v57, v106, v110, use_sem=True)
    v124 = None
    if not v91.v226:
        try:
            v28 = v241.v323(v23)
            v291 = {}
            for v111 in v109:
                v221 = v346(v28, v21, v22, v23, v111)
                if v221 is not None:
                    v291[v111] = v221.v181()
            if v291:
                v324 = v322(v345(v291.v37())).v222()
                v325 = v241.v223(v23)
                v326 = v158(v324, v23)
                v327 = v225(v325, v326, v30, v34, v74, v291, v75, v76, v77, v91.v78, v91.v79, v47, 'gpt', blend_l1=v91.v81, k_hard=v91.v66)
                v328 = v224(v325, v326, v30, v34, v57, v105, v291, use_sem=True)
                v329 = v224(v325, v326, v30, v34, v57, v106, v291, use_sem=True)
                v124 = {'dim': v324, 'loss': v327, 'seen_rel': v328, 'unseen_para': v329}
                v199(f"gpt2+sem: seen_rel={v328['sel_acc']:.3f} unseen_para={v329['sel_acc']:.3f}")
        except v292 as e:
            v199(f'  gpt control unavailable: {v359(v29).v138}: {v29}')
    v125 = v115['sel_acc'] <= v11 + 0.1
    v126 = v119['sel_acc'] >= v115['sel_acc'] + 0.15
    v127 = v118['sel_acc'] >= v114['sel_acc'] + 0.15
    v128 = v119['sel_acc'] >= v11 + 0.2
    v129 = v118['sel_acc'] >= v11 + 0.2
    v130 = v120['sel_acc'] >= v116['sel_acc'] - 0.05
    v131 = v119['bank_top1'] >= 0.5
    v132 = not v347.v330(v119['alpha']) and (not v347.v330(v120['alpha'])) and (v119['alpha'] >= v120['alpha'] + 0.05)
    v133 = v123['sel_acc'] <= v11 + 0.1
    v134 = v121
    v135 = v124 is not None and v124['unseen_para']['sel_acc'] < v11 + 0.2
    v136 = v125 and v134 and v133
    v137 = 'gpt_parity' if v135 else 'curve_gap' if v124 is not None else 'no_control'
    if not v136:
        v227 = 'SEM_QUERY_INVALID'
    elif v126 and v128 and v131 and v130:
        v227 = 'SEM_QUERY_OK'
    elif v127 and v129:
        v227 = 'SEM_QUERY_PARTIAL'
    elif v135:
        v227 = 'SEM_QUERY_NO_AT_SCALE'
    else:
        v227 = 'SEM_QUERY_NO'
    v52 = {'stage': 258, 'overall': v227, 'trunk': v96.v228, 'fp_version': v241.v293(), 'chance': v11, 'steps': v77, 'n_subjects': v139(v53), 'n_fit_subjects': v139(v213), 'n_eval_subjects': v139(v214), 'fit_rels': v10, 'exam_holdout': 'para_hold (same relations, alternate paraphrase on held-out subjects)', 'tape_slots': v139(v37), 'subject_slots': v101, 'curve_state_dim': v112, 'loss_curve': v117, 'unseen_reading': v137, 'gates': {'G_fp_only_at_chance': v125, 'G_sem_beats_fp': v126, 'G_sem_beats_fp_seen': v127, 'G_unseen_para': v128, 'G_seen_rel': v129, 'G_anchored_intact': v130, 'G_bankwide_retrieval': v131, 'G_sem_selective': v132, 'G_tape_causal': v133, 'G_keys_frozen': v134}, 'summary': {'fp_only': {'seen_rel': v114, 'unseen_para': v115, 'anchored': v116}, 'curve_sem': {'seen_rel': v118, 'unseen_para': v119, 'anchored': v120}, 'curve_sem_shuffled_keys': v123, 'gpt_control': v124}, 'note': 'Keys stay canonical frozen fp; P1 and trunk frozen; only W_q, W_sem and the blend train. a=0 reproduces stage 256 exactly, so fp-only is the same code path, not a reimplementation. The exam gives one subject four relations, so every candidate slot shares the anchor fingerprint and the query paraphrases the relation with no shared content word — fp-only is pre-registered at chance (G_fp_only_at_chance); if that gate fails the exam leaks and nothing else here is interpretable. unseen_para is paraphrase B on held-out subjects (258c). Fit training uses para + anchored so the blend sees fp-perfect queries. The matched GPT-2 channel exists so a negative can be told apart from a small GPU: NO_AT_SCALE means GPT did no better, which is what 210-212 never checked.', 'timestamp': v354.v348(v355.v349).v294(), 'wall_s': v196.v196() - v92}
    v1.v193(v331.v295(v52, indent=2), encoding='utf-8')
    v2.v193(f"# Stage 258 semantic query\n\n**{v227}** trunk={v96.v228} slots={v139(v37)} chance={v11:.2f}\n\n- unseen paraphrase: fp-only **{v115['sel_acc']:.3f}** -> +sem **{v119['sel_acc']:.3f}**\n- seen relation:   fp-only {v114['sel_acc']:.3f} -> +sem {v118['sel_acc']:.3f}\n- anchored (must not regress): {v116['sel_acc']:.3f} -> {v120['sel_acc']:.3f}\n- blend a: paraphrase {v119['alpha']:.3f} vs anchored {v120['alpha']:.3f} (blend now sees fp top1-top2 margin)\n- unseen_para reading: **{v137}**; predicted relations {v331.v295(v119.v263('confusion', {}))}\n- bank-wide top1 {v119['bank_top1']:.3f}, shuffled keys {v123['sel_acc']:.3f}\n" + (f"- matched GPT-2 unseen paraphrase: {v124['unseen_para']['sel_acc']:.3f}\n" if v124 else '- matched GPT-2 control: not run\n'), encoding='utf-8')
    v199(v331.v295({'overall': v227, 'gates': v52['gates']}, indent=2))
    if not v91.v198:
        v6.v232.v142(exist_ok=True)
        v27.v296({'W_q': v41.v350(), 'sem': v42.v350(), 'stage': 258}, v6)
    return 0
if v138 == '__main__':
    raise v229(v297())