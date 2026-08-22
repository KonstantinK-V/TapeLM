"""
Stage 257 — Composition: two-hop answers by chasing pointers in fp space.

256 showed the glue can EMIT a retrieved value (EM 0.667 free-form, causal under
shuffle/delete). But every question there was one-hop: cue -> slot -> copy. That is a
pointer-copy mechanism, not composition. This stage asks the next question:

    A --r1--> B   and   B --r2--> C   are on the tape as SEPARATE slots.
    C never co-occurs with A anywhere. Ask for A..r1..r2 and get C.

Mechanism (deliberately NOT a transformer doing latent hops — 210-212 is THESIS_NO,
and NOT retrieved text pasted back into the prompt — that is RAG+CoT):

    q_0 = W_q( fp(anchor) + ctx_fp(prefix) )      anchor from the cue, as in 256
    v_1 = argmax_slot  q_0                        -> should be B
    q_1 = W_q( fp(v_1)  + ctx_fp(prefix) )        RE-ANCHOR on what we just read
    v_2 = argmax_slot  q_1                        -> should be C

The hop is one line: replace the subject anchor with the retrieved value, keep the
question context. Keys stay frozen canonical, P1 stays frozen, the tape is untouched.

Halting is the only new "intelligence", and it is an ACT-style soft mixture so CE can
train it end-to-end without a hard argmax in the loop:

    w_h = (prod_{j<h} (1 - p_j)) * p_h ,   p_h = StopGate([h_t, sims stats, hop, lookahead])
    p_copy = sum_h w_h * copy_dist_h      (last hop absorbs the remainder)

"lookahead" is the honest signal for halting: re-anchor on the current top-1 and see
whether the bank answers. A waypoint has an outgoing edge; an answer does not.

Trunk frozen. Only W_q + read gate + tau + StopGate train — same contract as 256.

Controls (a two-hop number alone proves nothing):
  head_only        glue off                          -> must fail
  max_hops=1       the 256 mechanism, no hop loop    -> must fail (this is THE baseline)
  no_slot1         tape holds only B->C              -> must fail (no shortcut from A)
  delete_middle    drop THIS chain's A->B slot       -> its 2-hop dies, B->C survives
  shuffle_tape     permute keys                      -> must fail
  empty_tape       no slots                          -> parametric leak floor
  unseen_pair      (r1,r2) combination never fit     -> composition as an OPERATION
  stop selectivity expected hops on 1-hop < 2-hop    -> the gate decides, not the schedule

Write templates and query cues are worded DIFFERENTLY on purpose: 256 used one template
for both, so a decoder could ride the template instead of the tape.

  python _stage257_fp_compose.py [--smoke]
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
import _stage213_arc_enc_freeze_finetune as s213
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, SlotBias, TapeView, copy_dist, hidden_and_logits, mix_logprob
v0 = v21('results')
v1 = v0 / 'stage257_decision.json'
v2 = v0 / 'stage257_mini.md'
v3 = v0 / '_stage257_log.txt'
v4 = v21('checkpoints/stage191_p1_curve.pt')
v5 = v21('checkpoints/stage253_joint_l02.pt')
v6 = v21('checkpoints/stage257_compose.pt')
v7 = v21('data/_wikitext103_train.txt')
v8 = 257
v9 = 'The chronicle continues with routine administrative detail .'
v10 = {'director': {'write': '{S} was appointed director of {V} in the regional chronicle of 1987 .', 'q': 'directed'}, 'founder': {'write': '{S} was recorded as founder of {V} in the municipal register of 1954 .', 'q': 'founded'}}
v11 = {'seat': {'write': '{S} kept its registered seat in the city of {V} through the postwar decade .', 'q': 'kept its registered seat in the city of'}, 'archive': {'write': '{S} deposited its archive in the town of {V} during the reorganisation .', 'q': 'deposited its archive in the town of'}}
v12 = [('director', 'seat'), ('founder', 'archive')]
v13 = [('director', 'archive'), ('founder', 'seat')]
v14 = 'In the chronicle the body that {S} {q1} {q2}'
v15 = 'In the chronicle {S} {q1} the body named'
v16 = 'In the chronicle the body {S} {q2}'

def log(v22: v34) -> None:
    v23 = v22 if v22.v314('\n') else v22 + '\n'
    try:
        v315(v23, end='', flush=True)
    except v161:
        v315(v23.v432('ascii', 'replace').v324('ascii'), end='', flush=True)
    v3.v316.v162(parents=True, exist_ok=True)
    with v3.v317('a', encoding='utf-8') as v163:
        v163.v318(v23)

class StopGate(v24.v17):
    """p(stop hopping here). Same shape as the 256 read gate, five retrieval features."""

    def __init__(v164, v124: v20, v41):
        v425().v319()
        v164.v165 = v24.v426(v24.v453(v124 + 5, 64), v24.v454(), v24.v453(64, 1)).v246(v41)
        v24.v394.v320(v164.v165[-1].v321)
        v24.v394.v320(v164.v165[-1].v322)

    def forward(v164, v38, v166: v180, v167: v180, v48: v180, v168: v180, v169: v180):
        v170 = v215.v209([v166, v167, v48, v168, v169], device=v38.v41, dtype=v38.v395)
        return v215.v427(v164.v165(v215.v460([v38, v170], dim=-1))).v323(-1)

def anchored_query(v25: v171, v26: v172, v27: v173, v28: v179[v20], v29: v34 | None):
    """Query built exactly like a slot key: anchor fingerprint + context of the prefix.

    The hop lives here: pass the value we just retrieved as `anchor` and the same question
    context comes back pointed at the next edge.
    """
    v30 = v26.v174(v27.v324(v28[-40:]))
    if v30 is None:
        return None
    v31 = v337.v204(v26.v441([v29])[0] + v30, dim=-1) if v29 else v30
    return v337.v204(v25.v338(v31.v428(0)), dim=-1)[0]

def cue_anchor(v27: v173, v32: v179[v20]) -> v34 | None:
    """Subject of the question, read off the cue text — no oracle knowledge of the chain."""
    v33 = v325.v175(v27.v324(v32))
    return v33[-1] if v33 else None

def hop_mixture(v25: v171, v35: v176, v26: v172, v27: v173, v36: v177, v37: v179[v20], v32: v179[v20], v38: v215.v178, v39: v215.v178, v40: v20, v41, v42: v20, v43: v20, *, v44: v18=False, v45: v20 | None=None):
    """Bounded pointer chase with soft halting. Returns (p_copy, cov, stats, exp_hops, trace).

    Training uses the soft mixture over hops. Decode sets hard_commit=True: commit to the hop
    with largest weight w (ACT inference) so multi-token copy spans are not polluted by earlier
    hops' first-token proposals."""
    v46: v179[v215.v178] = []
    v47: v179[v215.v178] = []
    v48 = v180(-(v337.v455(v39, -1) * v337.v461(v39, -1)).v396())
    v29 = v181(v27, v32)
    v49 = v215.v182((), device=v41)
    v50 = v215.v183(v40, device=v41)
    v51 = v215.v183((), device=v41)
    v52 = v215.v183((), device=v41)
    v53 = v215.v183((), device=v41)
    v54 = v215.v183((), device=v41)
    v55: v179[v19] = []
    for v56 in v184(v43):
        v31 = v326(v25, v26, v27, v37, v29)
        v185 = v36.v239(v31, v42) if v31 is not None else None
        if v185 is None:
            break
        v327, v328 = v185
        v329, v330 = v331(v25, v36, v327, v328, v37, v40, v41)
        v46.v212(v329)
        v47.v212(v330)
        v186 = v36.v332[v20(v328[0])]
        v187 = v326(v25, v26, v27, v37, v186)
        v188 = v36.v239(v187, 1) if v187 is not None else None
        v169 = v180(v188[0][0]) if v188 is not None else 0.0
        v189 = v56 == v43 - 1
        v190 = v215.v182((), device=v41) if v189 else v35(v38, v180(v327.v222()), v180(v327.v335()), v48, v56 / v222(1, v43 - 1), v169)
        v191 = v49 * v190
        v50 = v50 + v191 * v329
        v51 = v51 + v191 * v330
        v52 = v52 + v191 * v180(v327.v222())
        v53 = v53 + v191 * v180(v327.v335())
        v54 = v54 + v191 * (v56 + 1)
        v55.v212({'hop': v56, 'top': v186, 'sim': v180(v327.v222()), 'lookahead': v169, 'p_stop': v180(v190), 'w': v180(v191), 'slot_idx': v20(v328[0])})
        v49 = v49 * (1.0 - v190)
        if v189:
            break
        v29 = v186
    if not v55:
        return (None, v215.v183((), device=v41), (0.0, 0.0), v54, v55)
    if v44:
        v192 = v45 if v45 is not None else v222(v184(v258(v55)), key=lambda v206: v55[v206]['w'])
        v192 = v213(v222(0, v192), v258(v55) - 1)
        v50 = v46[v192]
        v51 = v47[v192]
        v52 = v215.v209(v55[v192]['sim'], device=v41, dtype=v50.v395)
        v53 = v52
        v54 = v215.v209(v180(v192 + 1), device=v41, dtype=v50.v395)
    return (v50, v51, (v180(v52), v180(v53)), v54, v55)

def scored_step(v25, v35, v26, v27, v36, v37, v32, v38, v39, v40, v41, v42, v43, *, v44: v18=False, v45: v20 | None=None):
    """One decode position: hop chain -> copy mixture -> gated mix with the LM head."""
    v193, v194, (v52, v53), v54, v55 = v195(v25, v35, v26, v27, v36, v37, v32, v38, v39, v40, v41, v42, v43, hard_commit=v44, commit_hop=v45)
    if v193 is None:
        return (v215.v240(v337.v455(v39, -1) + 1e-09), v215.v183((), device=v41), v54, v55)
    v48 = v180(-(v337.v455(v39, -1) * v337.v461(v39, -1)).v396())
    v57 = v25.v57(v38, v52, v53, v48, v194)
    return (v333(v39, v57, v193, v194), v57, v54, v55)

def chain_batch(v25, v35, v58, v59, v27, v26, v36, v60, v61, v40, v41, v42, v43):
    """Teacher-forced CE on the answer tokens with soft hop mixture (trains StopGate).

    Decode uses hard-commit + span-lock; training stays soft so p_stop gets gradient.
    """
    v196, v197, v198 = ([], [], [])
    for v62 in v60:
        v32 = [v206 for v206 in v27.v432(v62['cue']).v28 if v206 != v61]
        v77 = [v206 for v206 in v27.v432(' ' + v62['answer']).v28 if v206 != v61]
        if not v32 or not v77:
            continue
        v37 = (v32 + v77)[-v429:]
        v199 = v258(v37) - v258(v77)
        v28 = v215.v209([v37], dtype=v215.v340, device=v41)
        v200, v201 = v202(v58, v59, v28, v61)
        v73 = v20(v62.v339('hops_needed', v43))
        for v133, v334 in v229(v77):
            v67 = v199 + v133 - 1
            if v67 < 0 or v67 >= v201.v444(1):
                break
            v336, v57, v210, v208 = v211(v25, v35, v26, v27, v36, v37[:v67 + 1], v32, v200[0, v67], v201[0, v67], v40, v41, v42, v73, hard_commit=False)
            v196.v212(-v336[v334])
            v197.v212(v180(v57))
            v198.v212(v180(v210))
    if not v196:
        return (None, v180('nan'), v180('nan'))
    return (v215.v430(v196).v335(), v180(v431.v335(v197)), v180(v431.v335(v198)))

def prose_batch(v25, v35, v58, v59, v27, v26, v36, v28, v61, v40, v41, v42, v43, v63, v64, v65=True):
    """Same machinery on ordinary text. An open gate costs CE directly under a mixture; the L1
    terms only stop gate and hop budget from drifting up where the LM is uncertain anyway."""
    v200, v201 = v202(v58, v59, v28, v61)
    v37 = v28[0].v203()
    v66 = [v67 for v67 in v184(v258(v37) - 1) if v37[v67] != v61 and v37[v67 + 1] != v61]
    if not v66:
        return (None, v180('nan'))
    v196, v197 = ([], [])
    for v67 in v66[::v222(1, v258(v66) // 8)]:
        v39 = v201[0, v67]
        if not v65:
            v196.v212(-v215.v240(v337.v455(v39, -1) + 1e-09)[v37[v67 + 1]])
            v197.v212(0.0)
            continue
        v336, v57, v210, v55 = v211(v25, v35, v26, v27, v36, v37[:v67 + 1], v37[:v67 + 1], v200[0, v67], v39, v40, v41, v42, v43)
        if not v55:
            v196.v212(-v215.v240(v337.v455(v39, -1) + 1e-09)[v37[v67 + 1]])
            v197.v212(0.0)
            continue
        v196.v212(-v336[v37[v67 + 1]] + v63 * v57 + v64 * v210)
        v197.v212(v180(v57))
    if not v196:
        return (None, v180('nan'))
    return (v215.v430(v196).v335(), v180(v431.v335(v197)))

def nce_loss(v25: v171, v68: v215.v178, v69: v215.v178, v70: v215.v178, v71: v180):
    """Retrieval objective for W_q on (prefix -> slot) pairs harvested from wiki noise.

    Fitting W_q on the chains themselves would only teach it where those chains live (255 lesson),
    so every chain stays held out from the retrieval objective.
    """
    v31 = v337.v204(v25.v338(v68), dim=-1)
    return v337.v205(v31 @ v70.v67() / v71, v69)

@v215.v78()
def free_decode(v25, v35, v58, v59, v27, v26, v36, v62, v61, v40, v41, v42, v43, v72, v65):
    """Greedy free-form continuation of the cue. No candidate set anywhere.

    After the hop chain commits a slot, the value span is emitted from tape.tok_ids
    (span-lock). Soft mixture still runs once for gate/exp_hops metrics; copy uses the
    answer hop (len(hop_targets)-1), not argmax-w — stop can be early while retrieval is right.
    """
    v32 = [v206 for v206 in v27.v432(v62['cue']).v28 if v206 != v61]
    v37 = v179(v32)
    v73 = v20(v62.v339('hops_needed', v43))
    v207, v197, v198, v74 = ([], [], [], None)
    if not v65:
        for v208 in v184(v72):
            v28 = v215.v209([v37[-v429:]], dtype=v215.v340, device=v41)
            v200, v201 = v202(v58, v59, v28, v61)
            v188 = v20(v201[0, -1].v397())
            v207.v212(v188)
            v37.v212(v188)
        return (v27.v324(v207).v216(), v180('nan'), v180('nan'), None)
    v28 = v215.v209([v37[-v429:]], dtype=v215.v340, device=v41)
    v200, v201 = v202(v58, v59, v28, v61)
    v39 = v201[0, -1]
    v208, v57, v210, v55 = v211(v25, v35, v26, v27, v36, v37, v32, v200[0, -1], v39, v40, v41, v42, v73, hard_commit=False)
    v197.v212(v180(v57))
    v198.v212(v180(v210))
    v74 = v55
    if not v55:
        v188 = v20(v39.v397())
        v207.v212(v188)
        return (v27.v324(v207).v216(), v180(v57), v180(v210), v74)
    v75 = v258(v62.v339('hop_targets') or [v62['answer']]) - 1
    v45 = v213(v75, v258(v55) - 1)
    v76 = v55[v45]['slot_idx']
    v77 = v179(v36.v398[v76] or [])
    if not v77:
        v188 = v20(v39.v397())
        v207.v212(v188)
        return (v27.v324(v207).v216(), v180(v57), v180(v210), v74)
    v207.v214(v77[:v72])
    return (v27.v324(v207).v216(), v180(v431.v335(v197)) if v197 else v180('nan'), v180(v431.v335(v198)) if v198 else v180('nan'), v74)

def exact_match(v79: v34, v80: v34) -> v18:
    """First word equals gold, or BPE truncation: generated prefix of gold / gold prefix of first word."""
    if not v79 or not v80:
        return False
    v81 = v79.v216().v433(' ')[0].v216(' .,;:')
    if v81 == v80:
        return True
    return v18(v81) and (v80.v399(v81) or v81.v399(v80)) and (v213(v258(v81), v258(v80)) >= 3)

def match_in_window(v79: v34, v80: v34, v82: v20=3) -> v18:
    """Value anywhere in the first n generated words. Reported for EVERY arm including the
    baselines, so it can never be used to rescue one number in isolation."""
    if not v79:
        return False
    v83 = [v191.v216(' .,;:') for v191 in v79.v216().v433(' ')[:v82]]
    if v80 in v83:
        return True
    return v217((v400(v191, v80) for v191 in v83 if v191))

@v215.v78()
def em_over(v25, v35, v58, v59, v27, v26, v36, v60, v61, v40, v41, v42, v43, v72, v65=True, v84=None):
    v218, v219, v220, v221 = (0, 0, [], [])
    for v62 in v60:
        v341, v57, v210, v55 = v342(v25, v35, v58, v59, v27, v26, v36, v62, v61, v40, v41, v42, v43, v72, v65)
        v218 += v20(v400(v341, v62['answer']))
        v219 += v20(v401(v341, v62['answer']))
        if not v434.v402(v57):
            v220.v212(v57)
        if not v434.v402(v210):
            v221.v212(v210)
        if v84 is not None and v258(v84) < 6:
            v84.v212({'cue': v62['cue'], 'gold': v62['answer'], 'got': v341, 'gate': v57, 'exp_hops': v210, 'trace': [{v462: v67[v462] for v462 in ('hop', 'top', 'sim', 'p_stop')} for v67 in v55 or []]})
    v82 = v222(1, v258(v60))
    return (v218 / v82, v219 / v82, v180(v431.v335(v220)) if v220 else v180('nan'), v180(v431.v335(v221)) if v221 else v180('nan'))

@v215.v78()
def retrieval_at_cue(v25, v35, v58, v59, v27, v26, v36, v60, v61, v40, v41, v42, v43) -> v19:
    """Does the pointer chase land on the right slots? Measured at the cue, decoder untouched.

    The first 257 run reported EM 0.000 while all three traces had already reached gold at hop 1.
    An end-to-end number cannot tell "the chain is broken" from "the chain is fine and the decode
    protocol lost it", so the mechanism gets its own metric here:

      hop_top1[h]     hop h's top-1 slot == the value that hop should reach
      chain_complete  every hop correct, in order
      answer_reached  the final target is top-1 at SOME hop (chain worked, halting may not have)
      halt_correct    the highest-weight hop is the last one the chain needed
    """
    v223, v224, v225, v226, v82 = ({}, 0, 0, 0, 0)
    for v62 in v60:
        v227 = v62.v339('hop_targets') or [v62['answer']]
        v32 = [v206 for v206 in v27.v432(v62['cue']).v28 if v206 != v61]
        if not v32:
            continue
        v28 = v215.v209([v32[-v429:]], dtype=v215.v340, device=v41)
        v200, v201 = v202(v58, v59, v28, v61)
        v208, v208, v208, v208, v55 = v195(v25, v35, v26, v27, v36, v32, v32, v200[0, -1], v201[0, -1], v40, v41, v42, v43)
        v228 = [v67['top'] for v67 in v55]
        v82 += 1
        for v343, v344 in v229(v227):
            v185 = v20(v343 < v258(v228) and v228[v343] == v344)
            v223.v456(v343, []).v212(v185)
        v224 += v20(v403((v343 < v258(v228) and v228[v343] == v67 for v343, v67 in v229(v227))))
        v225 += v20(v227[-1] in v228)
        if v55:
            v226 += v20(v20(v431.v397([v67['w'] for v67 in v55])) == v258(v227) - 1)
    if not v82:
        return {'n': 0}
    return {'n': v82, 'hop_top1': {f'hop{v343}': v180(v431.v335(v435)) for v343, v435 in v436(v223.v123())}, 'chain_complete': v224 / v82, 'answer_reached': v225 / v82, 'halt_correct': v226 / v82}

def build_chains(v85, v86, v87, v88, v89: v34):
    """A --r1--> B --r2--> C. B and C are unique per chain so slot delete is surgical."""
    v90 = []
    for v206, (v345, v346) in v229(v88):
        v347, v348, v349 = (v85.v404(), v86.v404(), v87.v404())
        v350, v351 = (v10[v345]['q'], v11[v346]['q'])
        v90.v212({'A': v347, 'B': v348, 'C': v349, 'r1': v345, 'r2': v346, 'pair': f'{v345}+{v346}', 'kind': v89, 'sent1': v10[v345]['write'].v437(S=v347, V=v348), 'sent2': v11[v346]['write'].v437(S=v348, V=v349), 'cue': v14.v437(S=v347, q1=v350, q2=v351), 'answer': v349, 'hop_targets': [v348, v349], 'cue_r1': v15.v437(S=v347, q1=v350), 'answer_r1': v348, 'cue_r2': v16.v437(S=v348, q2=v351), 'answer_r2': v349, 'cid': f'{v89}_{v206}'})
    return v90

def as_1hop(v60, v91: v34):
    """Same chains re-framed as single-edge questions (for stop selectivity and delete controls)."""
    return [{**v62, 'cue': v62[f'cue_{v91}'], 'answer': v62[f'answer_{v91}'], 'hops_needed': 1, 'hop_targets': [v62[f'answer_{v91}']]} for v62 in v60]

def main() -> v20:
    v92 = v352.v230()
    v92.v231('--smoke', action='store_true')
    v92.v231('--steps', type=v20, default=0)
    v92.v231('--topk', type=v20, default=8)
    v92.v231('--max-hops', type=v20, default=2)
    v92.v231('--gate-l1', type=v180, default=0.02, help='L1 on the read gate over prose')
    v92.v231('--hop-l1', type=v180, default=0.01, help='L1 on expected hops over prose')
    v92.v231('--nce-w', type=v180, default=1.0)
    v92.v231('--nce-tau', type=v180, default=0.05)
    v92.v231('--chains', type=v20, default=0)
    v92.v231('--distractor-slots', type=v20, default=0)
    v93 = v92.v232()
    v3.v233('', encoding='utf-8')
    v41 = v215.v41('cuda' if v215.v438.v405() else 'cpu')
    v94 = v353.v234(v8)
    v215.v235(v8)
    v95 = v236.v236()
    v96 = v93.v96 or (200 if v93.v238 else 800)
    v97 = v93.v60 or (8 if v93.v238 else 48)
    v98 = v93.v237 or (150 if v93.v238 else 1200)
    v72 = 8 if v93.v238 else 12
    v99 = 4 if v93.v238 else 12
    v100 = 40 if v93.v238 else 120
    v101 = 400 if v93.v238 else 6000
    v42, v43 = (v93.v239, v93.v43)
    v240(f'Stage257 fp compose start {v457.v449(v458.v450).v390()} device={v41} steps={v96} chains={v97} distractors={v98} topk={v42} max_hops={v43}')
    v208, v208, v241, v242 = v243()
    v27 = v173.v244(v34(v406.v354))
    v40 = v27.v245()
    v61 = v27.v355(v356) or 0
    v59 = v439.v407(v27, v241, v61, v40).v246(v41)
    v102 = v5 if v5.v357() else v4
    v58 = v408(v242, v40).v246(v41)
    v58.v247(v215.v409(v102, map_location=v41, weights_only=False)['model'])
    v58.v248()
    for v103 in v58.v249():
        v103.v358(False)
    v240(f'  trunk={v102.v312} (frozen)')
    v104 = v408(v242, v40).v246(v41)
    v104.v247(v215.v409(v4, map_location=v41, weights_only=False)['model'])
    v104.v248()
    for v103 in v104.v249():
        v103.v358(False)
    v105 = v172(v104, v241, v41)
    with v7.v317('r', encoding='utf-8', errors='ignore') as v163:
        v250 = v163.v359(1000000 if v93.v238 else 6000000)
    v106 = v179(v19.v360((v22.v412(1) for v22 in v411.v366(v250) if v258(v22.v412(1)) >= 5)))
    v94.v251(v106)
    v107 = [v410.v216() for v410 in v250.v433('\n') if v258(v410.v216()) >= 60][:v101]
    v85 = [v191 for v191 in v440(v365(v106), v94, v97 + 40) if v258(v191) >= 5][:v97 + 8]
    v252, v253 = (v97 - v97 // 3, v97 // 3)
    v86, v87 = (v106[:v97 + 8], v106[v97 + 8:2 * (v97 + 8)])
    v108 = [v12[v206 % v258(v12)] for v206 in v184(v252)]
    v109 = [v13[v206 % v258(v13)] for v206 in v184(v253)]
    v60 = v254(v85, v86, v87, v108, 'seen')
    v110 = v254(v85, v86, v87, v109, 'unseen')
    for v206, v62 in v229(v60):
        v62['glue_train'] = v206 % 2 == 0
    for v62 in v110:
        v62['glue_train'] = False
    v111 = [v30 for v30 in v60 if v30['glue_train']]
    v112 = [v30 for v30 in v60 if not v30['glue_train']]
    v240(f'  chains: fit={v258(v111)} held_out={v258(v112)} unseen_pair={v258(v110)}')
    v113 = {v255 for v30 in v60 + v110 for v255 in (v30['A'], v30['B'], v30['C'])}
    v114 = [v30['cid'] for v30 in v60 + v110 if v30['C'] in v30['sent1'] or v30['A'] in v30['sent2']]
    v240(f'  structural shortcut check: {v258(v114)} violations')
    v256, v257 = ([], [])
    for v30 in v60 + v110:
        for v29, v361, v362 in ((v30['A'], v30['sent1'], v30['B']), (v30['B'], v30['sent2'], v30['C'])):
            v363 = v105.v174(v361, exclude=v362)
            v364 = v105.v441([v29])[0]
            v256.v212(v337.v204(v364 + v363, dim=-1) if v363 is not None else v364)
            v257.v212(v362)
    v115 = v258(v257)
    v259, v260 = ([], [])
    v116 = v365(v257) | v113
    for v117 in v107:
        if v258(v257) >= v115 + v98:
            break
        for v22 in v411.v366(v117):
            v48 = v22.v412(1)
            if v258(v48) < 5 or v48 in v116:
                continue
            v413, v343 = (v222(0, v22.v463() - 120), v213(v258(v117), v22.v464() + 120))
            v363 = v105.v174(v117[v413:v343], exclude=v48)
            if v363 is None:
                continue
            v367 = [v191 for v191 in v325.v175(v117[v413:v22.v463()]) if v191 != v48]
            if not v367:
                continue
            v368 = v105.v441([v367[-1]])[0]
            v256.v212(v337.v204(v368 + v363, dim=-1))
            v369 = v105.v174(v117[v413:v22.v463()])
            if v369 is not None:
                v259.v212(v337.v204(v368 + v369, dim=-1))
                v260.v212(v258(v257))
            v257.v212(v48)
            v116.v414(v48)
            if v258(v257) >= v115 + v98:
                break
    v36 = v177(v215.v430(v256, 0).v246(v41), v257, v27, v61)
    v240(f'  tape slots={v258(v257)} ({v115} chain edges + {v258(v257) - v115} wiki noise)')
    v118 = '\n'.v261(v107 + [v9] * v213(v258(v60), v258(v107) // 4))
    v262, v263 = v370.v264(v118, v27, v61, max_lines=v101 + 64, min_line_len=20)
    v119 = v258(v263) - 1
    v120 = v179(v184(v222(1, v119 - v222(2, v119 // 20)), v119))
    v121 = v179(v184(0, v120[0]))
    v122 = v371.v265(v262, v263, v120, v61, v99, v8 + 5)
    v123 = v372.v266(v100)
    v240(f'  prose docs={v119} train={v258(v121)} hold={v258(v120)}')
    v124 = 2 * (v58.v415.v373 // 2)
    v25 = v171(v124, v41)
    v35 = v176(v124, v41)
    v125 = v215.v374.v267(v25.v416() + v179(v35.v249()), lr=0.003, weight_decay=0.01)
    v126 = v215.v430(v259).v246(v41).v180() if v259 else None
    v127 = v215.v209(v260, device=v41) if v260 else None
    v128 = v36.v70.v180()
    v240(f'  W_q training pairs={(0 if v126 is None else v126.v444(0))} (wiki noise only)')
    v129 = v371.v268(v58, v122, v59, v61, v41)
    v130 = v372.v269(v58, v59, v61, v123, v41)
    v270, v208, v208, v208 = v271(v25, v35, v58, v59, v27, v105, v36, v112, v61, v40, v41, v42, v43, v72, use_glue=False)
    v240(f'baseline hold_ce={v129:.3f} exam={v130:.3f} EM(head_only)={v270:.3f}')
    v131 = v111 + v375(v111, 'r1') + v375(v111, 'r2')
    v132 = []
    for v133 in v184(1, v96 + 1):
        v272 = [v131[v94.v442(v258(v131))] for v208 in v184(v213(4, v258(v131)))]
        v376, v377, v378 = v379(v25, v35, v58, v59, v27, v105, v36, v272, v61, v40, v41, v42, v43)
        v28 = v372.v443(v262, v263, 1, v94, v61, v121).v246(v41)
        v380, v381 = v382(v25, v35, v58, v59, v27, v105, v36, v28, v61, v40, v41, v42, v43, v93.v63, v93.v64)
        v273 = None
        if v126 is not None and v93.v417 > 0:
            v383 = v215.v418(0, v126.v444(0), (v213(64, v126.v444(0)),), device=v41)
            v273 = v93.v417 * v445(v25, v126[v383], v127[v383], v128, v93.v446)
        v274 = [v255 for v255 in (v376, v380, v273) if v255 is not None]
        if not v274:
            continue
        v275 = v274[0]
        for v103 in v274[1:]:
            v275 = v275 + v103
        v125.v384(set_to_none=True)
        v275.v385()
        v215.v24.v419.v386(v25.v416() + v179(v35.v249()), 1.0)
        v125.v133()
        if v133 % v222(1, v96 // 6) == 0:
            v132.v212({'step': v133, 'loss_chain': v180(v376) if v376 is not None else None, 'loss_prose': v180(v380) if v380 is not None else None, 'loss_nce': v180(v273) if v273 is not None else None, 'gate_chain': v377, 'gate_prose': v381, 'exp_hops_chain': v378, 'tau': v180(v215.v447(v25.v448))})
            v240(f"  step {v133}/{v96} chain={(v180(v376) if v376 is not None else v180('nan')):.3f} prose={(v180(v380) if v380 is not None else v180('nan')):.3f} nce={(v180(v273) if v273 is not None else v180('nan')):.3f} g_chain={v377:.3f} g_prose={v381:.3f} hops={v378:.2f} tau={v180(v215.v447(v25.v448)):.3f} ({v236.v236() - v95:.0f}s)")
    v25.v248()
    v35.v248()

    def em(v276, v277=v36, v198=v43, v65=True, v84=None):
        return v271(v25, v35, v58, v59, v27, v105, v277, v276, v61, v40, v41, v42, v198, v72, use_glue=v65, samples=v84)
    v134: v179[v19] = []
    v278, v279, v280, v281 = v282(v112, samples=v134)
    for v135 in v134[:4]:
        v240(f'    decode {v135}')
    v283, v284, v208, v208 = v282(v112, hops=1)
    v285, v286, v208, v287 = v282(v110)
    v288, v208, v208, v208 = v282(v112, tp=v36.v420(v8 + 1))
    v289, v208, v208, v208 = v282(v112, tp=v36.v421())
    v290, v291 = (v375(v112, 'r1'), v375(v112, 'r2'))
    v292, v293, v208, v294 = v282(v290)
    v295, v296, v208, v208 = v282(v291)

    def ret(v276, v277=v36, v198=v43):
        return v387(v25, v35, v58, v59, v27, v105, v277, v276, v61, v40, v41, v42, v198)
    v136 = v297(v112)
    v137 = v297(v110)
    v298, v299 = (v297(v290), v297(v291))
    v138 = v297(v112, tp=v36.v420(v8 + 1))
    v240(f'retrieval@cue 2hop: {v424.v391(v136)}')
    v240(f'retrieval@cue unseen_pair: {v424.v391(v137)}')
    v240(f'retrieval@cue shuffled: {v424.v391(v138)}')
    v139 = v36.v300()
    for v30 in v112:
        v139.v388(v30['B'])
    v301, v208, v208, v208 = v282(v112, tp=v139)
    v302, v303, v304 = ([], [], [])
    v140 = v353.v234(v8 + 7)
    for v30 in v112:
        v305 = v36.v300()
        v305.v388(v30['B'])
        v302.v212(v282([v30], tp=v305)[0])
        v303.v212(v282(v375([v30], 'r2'), tp=v305)[0])
        v306 = [v389 for v389 in v112 if v389 is not v30]
        if v306:
            v304.v212(v282(v140.v465(v306, v213(4, v258(v306))), tp=v305)[0])
    v141 = v180(v431.v335(v302)) if v302 else v180('nan')
    v142 = v180(v431.v335(v303)) if v303 else v180('nan')
    v143 = v180(v431.v335(v304)) if v304 else v180('nan')
    with v215.v78():
        v308, v309, v310 = ([], [], [])
        v307 = v353.v234(v8 + 99)
        for v208 in v184(12):
            v28 = v372.v443(v262, v263, 1, v307, v61, v120).v246(v41)
            v422, v57 = v382(v25, v35, v58, v59, v27, v105, v36, v28, v61, v40, v41, v42, v43, 0.0, 0.0, True)
            v423, v208 = v382(v25, v35, v58, v59, v27, v105, v36, v28, v61, v40, v41, v42, v43, 0.0, 0.0, False)
            if v422 is not None and v423 is not None:
                v309.v212(v180(v422))
                v310.v212(v180(v423))
            if not v434.v402(v57):
                v308.v212(v57)
    v144 = v180(v431.v335(v308)) if v308 else v180('nan')
    v145 = v180(v431.v335(v309)) if v309 else v180('nan')
    v146 = v180(v431.v335(v310)) if v310 else v180('nan')
    v147 = v278 >= 0.5
    v148 = v278 >= v283 + 0.2
    v149 = v278 >= v270 + 0.2
    v150 = v258(v114) == 0 and v301 <= 0.1
    v151 = v278 >= 0.4 and v141 <= 0.1 and (v142 >= 0.7 * v295) and (v143 >= 0.7 * v278)
    v152 = v288 <= v222(0.1, v278 - 0.4)
    v153 = v289 <= 0.1
    v154 = not v434.v402(v145) and (not v434.v402(v146)) and (v145 <= v146 + 0.05)
    v155 = not v434.v402(v294) and (not v434.v402(v281)) and (v281 >= v294 + 0.3)
    v156 = v285 >= 0.4
    v157 = v136.v339('chain_complete', 0.0) >= 0.5
    v158 = v138.v339('chain_complete', 1.0) <= 0.1
    v159 = v147 and v148 and v149 and v152 and v153 and v154
    if v159 and v150 and v151 and v155 and v156:
        v311 = 'FP_COMPOSE_OK'
    elif v159 and v150 and v151:
        v311 = 'FP_COMPOSE_PARTIAL'
    elif v157 and v158 and v150:
        v311 = 'FP_COMPOSE_MECHANISM_ONLY'
    else:
        v311 = 'FP_COMPOSE_NO'
    v90 = {'stage': 257, 'overall': v311, 'trunk': v102.v312, 'topk': v42, 'max_hops': v43, 'steps': v96, 'n_chains': v258(v60), 'n_fit': v258(v111), 'n_eval': v258(v112), 'n_unseen_pair': v258(v110), 'tape_slots': v258(v257), 'chain_edges': v115, 'fit_pairs': ['+'.v261(v103) for v103 in v12], 'held_pairs': ['+'.v261(v103) for v103 in v13], 'gates': {'G_compose_2hop': v147, 'G_beats_one_hop': v148, 'G_beats_head_only': v149, 'G_no_shortcut': v150, 'G_middle_slot_causal': v151, 'G_tape_causal': v152, 'G_no_param_leak': v153, 'G_lang_intact': v154, 'G_stop_selective': v155, 'G_unseen_pair': v156, 'G_retrieval_chain': v157, 'G_retrieval_causal': v158}, 'summary': {'em_2hop_glue': v278, 'em_2hop_one_hop_only': v283, 'em_2hop_head_only': v270, 'em_2hop_unseen_pair': v285, 'em_2hop_shuffled': v288, 'em_2hop_empty': v289, 'em_2hop_no_edge1_bank': v301, 'em_2hop_after_delete_middle': v141, 'em_r2_after_delete_middle': v142, 'em_2hop_others_after_delete': v143, 'em_1hop_r1': v292, 'em_1hop_r2': v295, 'em_window3': {'2hop_glue': v279, '2hop_one_hop_only': v284, 'unseen_pair': v286, '1hop_r1': v293, '1hop_r2': v296}, 'retrieval_at_cue': {'2hop': v136, 'unseen_pair': v137, '1hop_r1': v298, '1hop_r2': v299, 'shuffled': v138}, 'exp_hops_2hop': v281, 'exp_hops_1hop': v294, 'exp_hops_unseen': v287, 'gate_mean_chain': v280, 'gate_mean_prose': v144, 'prose_ce_glue_on': v145, 'prose_ce_glue_off': v146, 'hold_ce_base': v129, 'exam_base': v130, 'tau': v180(v215.v447(v25.v448)), 'structural_shortcut_violations': v258(v114)}, 'curve': v132, 'decode_samples': v134, 'note': "Retrieval@cue is the mechanism metric and is scored with no decoder in the loop: the first 257 run read EM 0.000 while every trace had already reached gold at hop 1, because the cue ended on a preposition, the LM emitted 'the', and exact_match only reads the first token (same run, same tape: 1-hop r1 ending on 'named' scored 0.667, 1-hop r2 ending on 'in' scored 0.000). The r2 tails now end on 'of' so the value is the immediate next token, and em_window3 is reported for every arm including the baselines. Two-hop by re-anchoring the query on the retrieved value; keys frozen canonical, P1 and trunk frozen, only W_q + read gate + tau + StopGate train. Halting is an ACT-style soft mixture over hops, so CE trains it without a hard argmax. C never co-occurs with A (checked structurally and by the no-edge1 bank control). EM is free-form greedy decode on chains the glue never fit; unseen_pair chains use relation COMBINATIONS never seen in fit.", 'timestamp': v457.v449(v458.v450).v390(), 'wall_s': v236.v236() - v95}
    v1.v233(v424.v391(v90, indent=2), encoding='utf-8')
    v2.v233(f"# Stage 257 fp composition (two-hop)\n\n**{v311}** trunk={v102.v312} slots={v258(v257)} eval_chains={v258(v112)}\n\n- EM 2-hop: head_only **{v270:.3f}** -> one-hop-only **{v283:.3f}** -> hop loop **{v278:.3f}** (value in first 3 tokens: {v279:.3f})\n- retrieval@cue 2-hop: chain **{v136.v339('chain_complete', v180('nan')):.3f}**, answer reached {v136.v339('answer_reached', v180('nan')):.3f}, halt correct {v136.v339('halt_correct', v180('nan')):.3f} (shuffled chain {v138.v339('chain_complete', v180('nan')):.3f})\n- unseen relation pair: **{v285:.3f}**\n- causal: shuffled {v288:.3f}, empty {v289:.3f}, no-edge1 bank {v301:.3f}\n- delete middle edge: 2-hop {v278:.2f} -> {v141:.2f}, its B->C {v295:.2f} -> {v142:.2f}, others {v143:.2f}\n- expected hops: 1-hop q **{v294:.2f}** vs 2-hop q **{v281:.2f}**\n- prose CE glue off {v146:.3f} -> on {v145:.3f}\n", encoding='utf-8')
    v240(v424.v391({'overall': v311, 'gates': v90['gates'], 'summary': v90['summary']}, indent=2))
    if not v93.v238:
        v6.v316.v162(exist_ok=True)
        v215.v392({'W_q': v25.v338.v451(), 'gate': v25.v459.v451(), 'log_tau': v25.v448.v466().v452(), 'stopper': v35.v451(), 'stage': 257}, v6)
    return 0
if v160 == '__main__':
    raise v313(v393())