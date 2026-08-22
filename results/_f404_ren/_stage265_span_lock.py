"""
Stage 265 — Span-lock: the gate decides WHERE a value starts, the tape decides HOW it is spelled.

The 256 decode audit isolated the failure. At every one of the five mechanism misses the copy
channel held the correct next token at rank 1 with p_copy ~= 0.997, and the gate closed anyway:

    Markbreit t=3  copy_rank_gold=1  p_copy=0.997  gate=0.00025  needed "it"  emitted "ch"
    Diavolo   t=3  copy_rank_gold=1  p_copy=0.997  gate=0.0017   needed "ol"  emitted "l"
    Cheese    t=2  copy_rank_gold=1  p_copy=0.997  gate=0.0027   needed "ese" emitted "f"
    Densetsu  t=3  copy_rank_gold=1  p_copy=0.997  gate=0.12     needed "u"   emitted "h"
    Sphinx    t=2  copy_rank_gold=1  p_copy=0.997  gate=0.13     needed "in"  emitted "r"

On each of those steps the emitted token is the LM's own top-1. The gate takes the entropy of the
base logits as an input feature, so it learned "the LM is confident -> hand over the wheel". Outside
a value that is reasonable. Inside one it is always wrong. The same mechanism explains the
--random-values control coming out HIGHER (0.875 vs 0.75): on nonsense strings the LM is never
confident, so the gate never closes.

Deleting the entropy feature would not fix it -- h_t is also an input and the same signal is
recoverable from it. The structural fix is to stop asking the gate that question at all:

    soft   (256):  the value survives only if the gate holds on EVERY token   P ~ p^N
    locked (265):  the gate opens once, then tape.tok_ids is emitted verbatim  P ~ p

That is the scaling argument, and the 256 data already shows the exponent: the five mechanism
misses sit on 5/4/4/4/3-token values while the successes cluster at 2-3. Longer values are exactly
what a real tape holds -- dates, identifiers, names outside English, anything not in BPE's comfort
zone -- so the compounding penalty is a ceiling that scale does not lift.

Span-lock also makes the contract checkable rather than statistical: the emitted span is bit-identical
to the slot, so the weights' contribution to a value is exactly zero, by construction and assertable.
It gets the restart defect (19/24 decodes in the audit re-emitted the value's first token after the
value ended) for free, because the span length comes from the tape instead of a learned stop.

Three arms, so the claim is attributable:

    A  soft train  + soft decode    reproduction of 256          (validity gate)
    B  soft train  + locked decode  does the lock alone fix it   (no retraining at all)
    C  open train  + locked decode  gate trained only on "open"  (full proposal)

Arm C trains the gate ONLY at the value-start step; inside the span g is pinned to 1 with no
gradient, so the entropy shortcut never gets a training signal. 257's lesson is respected --
training stays a soft mixture everywhere else, since hard-commit-only training collapses the
stop gate.

Controls: head-only, shuffled keys, empty tape, per-fact slot delete, prose CE on/off, prose gate,
and a paired nonsense-value exam in the SAME tape -- under the lock, EM must stop caring whether
the value is a dictionary word.

  python _stage265_span_lock.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import Counter
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
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, SlotBias, TapeView, copy_dist, ctx_query, hidden_and_logits, mix_logprob, raw_query
v0 = v20('results')
v1 = v0 / 'stage265_decision.json'
v2 = v0 / 'stage265_mini.md'
v3 = v0 / '_stage265_log.txt'
v4 = v0 / 'stage265_span_trace.json'
v5 = v0 / 'stage256_decision.json'
v6 = v20('checkpoints/stage191_p1_curve.pt')
v7 = v20('checkpoints/stage253_joint_l02.pt')
v8 = v20('checkpoints/stage265_span_lock.pt')
v9 = v20('data/_wikitext103_train.txt')
v10 = 265
v11 = v12
v13 = v14
v15 = 'The chronicle continues with routine administrative detail .'

def log(v21: v16) -> None:
    v22 = v21 if v21.v287('\n') else v21 + '\n'
    try:
        v288(v22, end='', flush=True)
    except v177:
        v288(v22.v382('ascii', 'replace').v297('ascii'), end='', flush=True)
    v3.v289.v178(parents=True, exist_ok=True)
    with v3.v290('a', encoding='utf-8') as v88:
        v88.v291(v22)

def fp_version() -> v16:
    v23 = v179(v180, 'canonical_fp_version', None)
    if v181(v23):
        try:
            return v16(v23())
        except v225:
            pass
    return v6.v24

def match_step(v25: v34[v17], v26: v34[v17]) -> v17:
    """How many tokens of `ids` the prefix already ends with. Mirrors copy_dist exactly."""
    for v27 in v182(v292(v294(v26), v294(v25)), 0, -1):
        if v25[-v27:] == v26[:v27]:
            return v27
    return 0

def span_candidates(v28, v29: v183, v30, v31, v25) -> v34[v187[v17, v109]]:
    """Retrieved slots whose value has not started yet -> (slot, copy weight).

    A span may only OPEN at step 0 of a value. Slots already under way are excluded because the
    lock, not the gate, is what carries a value that is mid-emission.
    """
    v32 = v28.v184(v30)
    v33 = []
    for v86, v185 in v186(v31.v211()):
        v26 = v29.v293[v185]
        if not v26:
            continue
        if v349(v25, v26) == 0:
            v33.v296((v185, v109(v32[v86])))
    return v33

@v195.v68()
def decode(v28, v35, v36, v37, v38, v29: v183, v39, v40, v41, v42, v43: v17, v44: v17, *, v45: v18, v46: v18=True, v47: v109=0.5, v48: v109=0.1, v49: v17=1, v50: v18=True, v51: v34 | None=None):
    """Greedy free-form continuation of the cue. `locked=False` is 256's decode verbatim.

    Under the lock the gate is consulted only while no span is open. Once it opens on a slot the
    slot's tokens are emitted verbatim; the LM cannot overwrite them and no learned stop is needed,
    because the length is the slot's length.

    Hysteresis: after a span closes the lock is disarmed and cannot fire again until the gate has
    fallen below `open_thresh - reopen_margin` for at least one step. Without it the smoke run
    re-opened on a *different* slot as soon as the value ended and there was room left (restart
    0.375) -- 256's restart defect wearing a new coat: the gate is still high because the query has
    not moved, so `no_repeat` on the value string does not catch it.

    Planted-fact exam default `max_opens=1`: after the first open the lock stays disarmed for the
    rest of the decode. That is a property of this exam ("one answer per cue"), not of the
    architecture. Multi-slot answers need `max_opens=0` (unlimited) with hysteresis only.

    Two EM readings under the lock (do not collapse them):
      em_span  — first-word match on the emitted tape span alone (stop-at-boundary for scoring)
      em_text  — first-word match on the full max_new continuation (LM may BPE-glue onto the value)
    Headline ``em`` = em_span. em_text keeps the end-of-value defect visible; the boundary is NOT
    solved — only deferred by the one-answer exam. Same family as restart: the model does not know
    the value has ended.
    """
    v52 = [v126 for v126 in v37.v382(v11.v400(S=v39['S'])).v26 if v126 != v40]
    v53 = v34(v52)
    v54: v34[v17] = []
    v55: v34[v109] = []
    v56: v34[v19] = []
    v57: v188[v16] = v188()
    v58 = True
    v59: v17 | None = None
    while v294(v54) < v44:
        v26 = v195.v295([v53[-v388:]], dtype=v195.v350, device=v42)
        v208, v209 = v210(v35, v36, v26, v40)
        v189 = v209[0, -1]
        v190 = v195.v236(v299.v383(v189, -1) + 1e-09)
        if v46:
            v85 = v302(v28, v38, v37, v53, anchor_ids=v52)
            v213 = v29.v117(v85, v43) if v85 is not None else None
            if v213 is not None:
                v30, v31 = v213
                v214 = v109(-(v299.v383(v189, -1) * v299.v423(v189, -1)).v353())
                v303, v304 = v305(v28, v29, v30, v31, v53, v41, v42)
                v215 = v28.v306(v208[0, -1], v109(v30.v197()), v109(v30.v204()), v214, v304)
                v55.v296(v109(v215))
                if v49 <= 0 or v294(v56) < v49:
                    if v109(v215) < v47 - v48:
                        v58 = True
                if v45 and v58 and (v109(v215) >= v47) and (v49 <= 0 or v294(v56) < v49):
                    v384 = [v256 for v256 in v422(v28, v29, v30, v31, v53) if not (v50 and v29.v194[v256[0]] in v57)]
                    if v384:
                        v185 = v197(v384, key=lambda v256: v256[1])[0]
                        v75 = v34(v29.v293[v185])
                        v406 = v44 - v294(v54)
                        v407 = v75[:v406]
                        v56.v296({'at': v294(v54), 'slot': v185, 'value': v29.v194[v185], 'g': v109(v215), 'n_tok': v294(v75), 'truncated': v294(v407) < v294(v75)})
                        v57.v373(v29.v194[v185])
                        v54.v417(v407)
                        v53.v417(v407)
                        v58 = False
                        if v59 is None:
                            v59 = v294(v54)
                        continue
                v190 = v385(v189, v215, v303, v304)
        v191 = v17(v190.v351())
        v54.v296(v191)
        v53.v296(v191)
    v60 = v37.v297(v54).v192()
    v61 = v37.v297(v54[:v59]).v192() if v59 is not None else v60
    v62 = None
    for v185, v193 in v186(v29.v194):
        if v193 == v39['value']:
            v62 = v29.v293[v185]
            break
    v63 = v56[0] if v56 else None
    v64 = v37.v297(v62).v192() if v62 else None
    v65 = v60.v192().v399(' ')[0].v192(' .,;:') if v60 else ''
    v66 = v61.v192().v399(' ')[0].v192(' .,;:') if v61 else ''
    v67 = {'S': v39['S'], 'gold': v39['value'], 'got': v61 if v45 and v59 is not None else v60, 'got_span': v61, 'got_text': v60, 'em_first_word': v66 if v45 and v59 is not None else v65, 'em_first_word_text': v65, 'gold_decode': v64, 'gate_mean': v109(v391.v204(v55)) if v55 else v109('nan'), 'n_opens': v294(v56), 'opened_value': v63['value'] if v63 else None, 'opened_at': v63['at'] if v63 else None, 'opened_correct': v18(v63 and v63['value'] == v39['value'] and (v63['at'] == 0)), 'truncated': v18(v63 and v63['truncated']), 'verbatim': v18(v62 is not None and v63 is not None and (v63['value'] == v39['value']) and (not v63['truncated']) and (v54[v63['at']:v63['at'] + v294(v62)] == v62)), 'n_val_tokens': v294(v62) if v62 else None, 'opens': v56, 'span_end': v59}
    if v51 is not None and v294(v51) < 48:
        v51.v296(v67)
    return v67

def exact_match(v69: v16, v70: v16) -> v18:
    return v69.v192().v399(' ')[0].v192(' .,;:') == v70 if v69 else False

def em_window3(v69: v16, v70: v16) -> v18:
    if not v69:
        return False
    return v70 in [v32.v192(" .,;:'") for v32 in v69.v192().v399(' ')[:3]]

@v195.v68()
def exam(v28, v35, v36, v37, v38, v29, v71, v40, v41, v42, v43, v44, *, v45, v46=True, v47=0.5, v48=0.1, v49=1, v51=None):
    v72 = [v297(v28, v35, v36, v37, v38, v29, v88, v40, v41, v42, v43, v44, locked=v45, use_glue=v46, open_thresh=v47, reopen_margin=v48, max_opens=v49, trace=v51) for v88 in v71]
    for v80, v88 in v196(v72, v71):
        v80['em_span'] = v298(v80.v377('got_span') or v80['got'], v88['value'])
        v80['em_text'] = v298(v80.v377('got_text') or v80['got'], v88['value'])
        v80['em'] = v80['em_span'] if v45 else v80['em_text']
        v80['em3'] = v352(v80.v377('got_span') or v80['got'], v88['value']) if v45 else v352(v80['got'], v88['value'])
        v80['glue_bpe'] = v18(v80['em_span'] and (not v80['em_text']))
    v73 = v197(1, v294(v72))
    v74 = [v80['gate_mean'] for v80 in v72 if not v401.v378(v80['gate_mean'])]
    v33 = {'em': v353((v80['em'] for v80 in v72)) / v73, 'em_span': v353((v80['em_span'] for v80 in v72)) / v73, 'em_text': v353((v80['em_text'] for v80 in v72)) / v73, 'em3': v353((v80['em3'] for v80 in v72)) / v73, 'glue_bpe_rate': v353((v80['glue_bpe'] for v80 in v72)) / v73, 'gate_mean': v109(v391.v204(v74)) if v74 else v109('nan'), 'rows': v72}
    v75 = {'verbatim': v353((v80['verbatim'] for v80 in v72)) / v73, 'open_recall': v353((v80['opened_correct'] for v80 in v72)) / v73, 'open_precision': v353((v80['opened_correct'] for v80 in v72)) / v197(1, v353((v80['n_opens'] > 0 for v80 in v72))), 'restart_rate': v353((v80['n_opens'] > 1 for v80 in v72)) / v73, 'truncated_rate': v353((v80['truncated'] for v80 in v72)) / v73}
    v33.v198(v75 if v45 else v19.v331(v75, None))
    return v33

def em_by_length(v72: v34[v19], v27: v17=4) -> v19:
    """The scaling claim, measured: does EM fall off as the value gets longer?

    Under a soft mixture a value survives only if the gate holds on every token, so EM should decay
    with length. Under the lock the gate is asked once, so length must stop mattering.
    """
    v76 = [v80 for v80 in v72 if v80['n_val_tokens'] and v80['n_val_tokens'] < v27]
    v77 = [v80 for v80 in v72 if v80['n_val_tokens'] and v80['n_val_tokens'] >= v27]
    v78 = v199()
    v79 = v199()
    for v80 in v72:
        if v80['n_val_tokens']:
            v79[v80['n_val_tokens']] += 1
            v78[v80['n_val_tokens']] += v17(v80['em'])
    return {'cut': v27, 'n_short': v294(v76), 'n_long': v294(v77), 'em_short': v109(v391.v204([v80['em'] for v80 in v76])) if v76 else v109('nan'), 'em_long': v109(v391.v204([v80['em'] for v80 in v77])) if v77 else v109('nan'), 'by_n_tokens': {v16(v73): v78[v73] / v79[v73] for v73 in v386(v79)}}

def nce_loss(v28: v200, v81: v195.v201, v82: v195.v201, v83: v195.v201, v84: v109):
    v85 = v299.v202(v28.v300(v81), dim=-1)
    v30 = v85 @ v83.v91() / v84
    v86 = v195.v354(v82, v30, v195.v387(v30, -10000.0)).v203(dim=-1)
    return (v30.v203(dim=-1) - v86).v204()

def fact_batch(v28, v35, v36, v37, v38, v29, v71, v40, v41, v42, v43, *, v87: v18):
    """Teacher-forced CE on the value tokens.

    With `open_only`, steps inside the value use g=1 as a constant: the copy path still gets
    gradient through tau and W_q, but the gate receives none. That is the exact signal that taught
    it to defer to a confident LM mid-word, and it is the only thing removed here.
    """
    v205, v55 = ([], [])
    for v88 in v71:
        v52 = [v126 for v126 in v37.v382(v11.v400(S=v88['S'])).v26 if v126 != v40]
        v206 = [v126 for v126 in v37.v382(' ' + v88['value']).v26 if v126 != v40]
        if not v52 or not v206:
            continue
        v53 = (v52 + v206)[-v388:]
        v207 = v294(v53) - v294(v206)
        v26 = v195.v295([v53], dtype=v195.v350, device=v42)
        v208, v209 = v210(v35, v36, v26, v40)
        for v107, v301 in v186(v206):
            v91 = v207 + v107 - 1
            if v91 < 0 or v91 >= v209.v394(1):
                break
            v212 = v53[:v91 + 1]
            v189 = v209[0, v91]
            v85 = v302(v28, v38, v37, v212, anchor_ids=v52)
            v213 = v29.v117(v85, v43) if v85 is not None else None
            if v213 is None:
                v205.v296(-v195.v236(v299.v383(v189, -1) + 1e-09)[v301])
                continue
            v30, v31 = v213
            v214 = v109(-(v299.v383(v189, -1) * v299.v423(v189, -1)).v353())
            v303, v304 = v305(v28, v29, v30, v31, v212, v41, v42)
            v215 = v28.v306(v208[0, v91], v109(v30.v197()), v109(v30.v204()), v214, v304)
            if v87 and v107 > 0:
                v355 = v195.v389((), device=v42, dtype=v215.v408)
            else:
                v355 = v215
                v55.v296(v109(v215))
            v205.v296(-v385(v189, v355, v303, v304)[v301])
    if not v205:
        return (None, v109('nan'))
    return (v195.v390(v205).v204(), v109(v391.v204(v55)) if v55 else v109('nan'))

def prose_batch(v28, v35, v36, v37, v38, v29, v26, v40, v41, v42, v43, v89, v46=True):
    v208, v209 = v210(v35, v36, v26, v40)
    v205, v55 = ([], [])
    v53 = v26[0].v211()
    v90 = [v91 for v91 in v182(v294(v53) - 1) if v53[v91] != v40 and v53[v91 + 1] != v40]
    if not v90:
        return (None, v109('nan'))
    for v91 in v90[::v197(1, v294(v90) // 8)]:
        v189 = v209[0, v91]
        v212 = v53[:v91 + 1]
        if not v46:
            v205.v296(-v195.v236(v299.v383(v189, -1) + 1e-09)[v53[v91 + 1]])
            v55.v296(0.0)
            continue
        v85 = v302(v28, v38, v37, v212)
        v213 = v29.v117(v85, v43) if v85 is not None else None
        if v213 is None:
            v205.v296(-v195.v236(v299.v383(v189, -1) + 1e-09)[v53[v91 + 1]])
            v55.v296(0.0)
            continue
        v30, v31 = v213
        v214 = v109(-(v299.v383(v189, -1) * v299.v423(v189, -1)).v353())
        v303, v304 = v305(v28, v29, v30, v31, v212, v41, v42)
        v215 = v28.v306(v208[0, v91], v109(v30.v197()), v109(v30.v204()), v214, v304)
        v205.v296(-v385(v189, v215, v303, v304)[v53[v91 + 1]] + v89 * v215)
        v55.v296(v109(v215))
    if not v205:
        return (None, v109('nan'))
    return (v195.v390(v205).v204(), v109(v391.v204(v55)))

def train_glue(v28, v35, v36, v37, v38, v29, v92, v93, v94, v95, v40, v41, v42, *, v96, v43, v89, v97, v98, v99, v100, v87, v101, v102, v103):
    v104 = v195.v307.v216(v28.v308(), lr=0.003, weight_decay=0.01)
    v105 = v29.v83.v109()
    v106 = []
    for v107 in v182(1, v96 + 1):
        v217 = [v92[v101.v392(v294(v92))] for v237 in v182(v292(4, v294(v92)))]
        v309, v310 = v311(v28, v35, v36, v37, v38, v29, v217, v40, v41, v42, v43, open_only=v87)
        v26 = v409.v393(v93, v94, 1, v101, v40, v95).v243(v42)
        v312, v313 = v314(v28, v35, v36, v37, v38, v29, v26, v40, v41, v42, v43, v89)
        v218 = None
        if v97 is not None and v99 > 0:
            v315 = v195.v356(0, v97.v394(0), (v292(64, v97.v394(0)),), device=v42)
            v222 = v299.v410(v98[v315], v105.v394(0)).v18()
            v218 = v99 * v395(v28, v97[v315], v222, v105, v100)
        v219 = [v316 for v316 in (v309, v312, v218) if v316 is not None]
        if not v219:
            continue
        v220 = v219[0]
        for v119 in v219[1:]:
            v220 = v220 + v119
        v104.v317(set_to_none=True)
        v220.v318()
        v195.v396.v357.v319(v28.v308(), 1.0)
        v104.v107()
        if v107 % v197(1, v96 // 5) == 0:
            v106.v296({'step': v107, 'loss_fact': v109(v309) if v309 is not None else None, 'loss_prose': v109(v312) if v312 is not None else None, 'loss_nce': v109(v218) if v218 is not None else None, 'gate_fact': v310, 'gate_prose': v313})
            v236(f"  [{v103}] step {v107}/{v96} fact={(v109(v309) if v309 is not None else v109('nan')):.3f} prose={(v109(v312) if v312 is not None else v109('nan')):.3f} g_fact={v310:.3f} g_prose={v313:.3f} ({v232.v232() - v102:.0f}s)")
    v28.v221()
    return v106

@v195.v68()
def full_bank_read(v28, v38, v37, v29: v183, v71, v40) -> v19:
    """Rank of the gold slot over the WHOLE bank at the cue -- no candidate pool anywhere.

    The closed-pool headline and this number are different questions; 256's exam is saturated here
    (top1 = 1.0), which is precisely why its EM was measuring decode and nothing else.
    """
    v108 = []
    for v88 in v71:
        v52 = [v126 for v126 in v37.v382(v11.v400(S=v88['S'])).v26 if v126 != v40]
        v85 = v302(v28, v38, v37, v52, anchor_ids=v52)
        if v85 is None:
            continue
        v30 = v29.v83 @ v85
        v222 = [v185 for v185, v193 in v186(v29.v194) if v193 == v88['value']]
        if not v222:
            continue
        v223 = v109(v30[v222].v197())
        v108.v296(1 + v17((v30 > v223).v353()))
    if not v108:
        return {'bank_size': v294(v29.v194), 'n': 0}
    return {'bank_size': v294(v29.v194), 'n': v294(v108), 'top1': v109(v391.v204([v80 == 1 for v80 in v108])), 'hit10': v109(v391.v204([v80 <= 10 for v80 in v108])), 'mrr': v109(v391.v204([1.0 / v80 for v80 in v108])), 'median_rank': v109(v391.v358(v108))}

def published_em_256() -> v109 | None:
    if not v5.v320():
        return None
    try:
        v224 = v359.v321(v5.v360(encoding='utf-8'))
        return v109(v224['summary']['em_glue'])
    except v225:
        return None

def main() -> v17:
    v110 = v322.v226()
    v110.v227('--smoke', action='store_true')
    v110.v227('--steps', type=v17, default=0)
    v110.v227('--topk', type=v17, default=8)
    v110.v227('--gate-l1', type=v109, default=0.02)
    v110.v227('--nce-w', type=v109, default=1.0)
    v110.v227('--nce-tau', type=v109, default=0.05)
    v110.v227('--facts', type=v17, default=0)
    v110.v227('--nonsense-facts', type=v17, default=0, help='held-out facts whose value is not a word')
    v110.v227('--distractor-slots', type=v17, default=0)
    v110.v227('--open-thresh', type=v109, default=0.5, help='gate level that opens a span')
    v110.v227('--reopen-margin', type=v109, default=0.1, help='after a span, the gate must fall below open_thresh - margin before it may fire again')
    v110.v227('--max-opens', type=v17, default=1, help="max spans per decode; 1 = exam 'one answer per cue' (not architecture); 0 = unlimited")
    v110.v227('--no-arm-c', action='store_true', help='skip the open-trained arm (half the wall time)')
    v111 = v110.v228()
    v3.v229('', encoding='utf-8')
    v42 = v195.v42('cuda' if v195.v397.v361() else 'cpu')
    v101 = v323.v230(v10)
    v195.v231(v10)
    v102 = v232.v232()
    v96 = v111.v96 or (200 if v111.v235 else 800)
    v112 = v111.v71 or (8 if v111.v235 else 48)
    v113 = v111.v233 or (4 if v111.v235 else 16)
    v114 = v111.v234 or (150 if v111.v235 else 1200)
    v44 = 6 if v111.v235 else 12
    v115 = 4 if v111.v235 else 12
    v116 = 400 if v111.v235 else 6000
    v43 = v111.v117
    v236(f'Stage265 span-lock start {v414.v402(v415.v403).v345()} device={v42} steps={v96} facts={v112} nonsense={v113} distractors={v114} topk={v43} open_thresh={v111.v47}')
    v237, v237, v238, v239 = v240()
    v37 = v324.v241(v16(v362.v325))
    v41 = v37.v242()
    v40 = v37.v326(v327) or 0
    v36 = v398.v363(v37, v238, v40, v41).v243(v42)
    v118 = v7 if v7.v328() else v6
    v35 = v364(v239, v41).v243(v42)
    v35.v244(v195.v365(v118, map_location=v42, weights_only=False)['model'])
    v35.v221()
    for v119 in v35.v245():
        v119.v329(False)
    v120 = v364(v239, v41).v243(v42)
    v120.v244(v195.v365(v6, map_location=v42, weights_only=False)['model'])
    v120.v221()
    for v119 in v120.v245():
        v119.v329(False)
    v121 = v246(v120, v238, v42)
    v236(f'  trunk={v118.v24} (frozen) fp_version={v344()}')
    with v9.v290('r', encoding='utf-8', errors='ignore') as v88:
        v247 = v88.v330(1000000 if v111.v235 else 6000000)
    v122 = v34(v19.v331((v21.v370(1) for v21 in v369.v333(v247) if v294(v21.v370(1)) >= 5)))
    v101.v248(v122)
    v123 = [v366.v192() for v366 in v247.v399('\n') if v294(v366.v192()) >= 60][:v116]
    v124 = [v32 for v32 in v367(v188(v122), v101, v112 + v113 + 60) if v294(v32) >= 5]
    v124 = v34(v19.v331(v124))
    v125 = [v32 for v32 in v367(v188(v122) | v188(v124), v101, v113 + 40) if v294(v32) >= 6]
    v125 = [v32 for v32 in v19.v331(v125) if v32 not in v124][:v113]
    if v294(v124) < v112 + v294(v125):
        raise v286(f'not enough distinct subjects: {v294(v124)} < {v112 + v294(v125)}')
    v71 = []
    for v126 in v182(v112):
        v71.v296({'S': v124[v126], 'value': v122[v126], 'sent': v13.v400(S=v124[v126], V=v122[v126]), 'glue_train': v126 % 2 == 0, 'kind': 'wiki'})
    for v185, v249 in v186(v125):
        v250 = v124[v112 + v185]
        v71.v296({'S': v250, 'value': v249, 'sent': v13.v400(S=v250, V=v249), 'glue_train': False, 'kind': 'nonsense'})
    v92 = [v88 for v88 in v71 if v88['glue_train']]
    v127 = [v88 for v88 in v71 if not v88['glue_train'] and v88['kind'] == 'wiki']
    v128 = [v88 for v88 in v71 if v88['kind'] == 'nonsense']
    v129 = v127 + v128
    v236(f'  facts: fit={v294(v92)} held_out_wiki={v294(v127)} held_out_nonsense={v294(v128)}')
    v251, v252 = ([], [])
    v253, v254 = ([], [])
    for v88 in v71:
        v255 = v121.v368([v88['S']])[0]
        v256 = v121.v332(v88['sent'], exclude=v88['value'])
        v251.v296(v299.v202(v255 + v256, dim=-1) if v256 is not None else v255)
        v252.v296(v88['value'])
    v130 = v188(v252)
    for v131 in v123:
        if v294(v252) >= v294(v71) + v114:
            break
        for v21 in v369.v333(v131):
            v214 = v21.v370(1)
            if v294(v214) < 5 or v214 in v130:
                continue
            v371, v372 = (v197(0, v21.v418() - 120), v292(v294(v131), v21.v419() + 120))
            v256 = v121.v332(v131[v371:v372], exclude=v214)
            if v256 is None:
                continue
            v334 = [v32 for v32 in v420.v411(v131[v371:v21.v418()]) if v32 != v214]
            if not v334:
                continue
            v251.v296(v299.v202(v121.v368([v334[-1]])[0] + v256, dim=-1))
            v335 = v121.v332(v131[v371:v21.v418()])
            if v335 is not None:
                v253.v296(v299.v202(v121.v368([v334[-1]])[0] + v335, dim=-1))
                v254.v296(v294(v252))
            v252.v296(v214)
            v130.v373(v214)
            if v294(v252) >= v294(v71) + v114:
                break
    v29 = v183(v195.v390(v251, 0).v243(v42), v252, v37, v40)
    v236(f'  tape slots={v294(v252)} ({v294(v71)} planted + {v294(v252) - v294(v71)} wiki noise)')
    v132 = v199((v294(v29.v293[v252.v412(v88['value'])]) for v88 in v129))
    v236(f'  held-out value lengths (BPE tokens): {v19(v386(v132.v413()))}')
    v133 = '\n'.v257(v123 + [v15] * v292(v294(v71), v294(v123) // 4))
    v93, v94 = v336.v258(v133, v37, v40, max_lines=v116 + 64, min_line_len=20)
    v134 = v294(v94) - 1
    v135 = v34(v182(v197(1, v134 - v197(2, v134 // 20)), v134))
    v95 = v34(v182(0, v135[0]))
    v136 = v337.v259(v93, v94, v135, v40, v115, v10 + 5)
    v97 = v195.v390(v253).v243(v42).v109() if v253 else None
    v98 = v195.v295(v254, device=v42) if v254 else None
    v236(f'  W_q training pairs={(0 if v97 is None else v97.v394(0))}')
    v137 = 2 * (v35.v374.v338 // 2)
    v138 = v337.v260(v35, v136, v36, v40, v42)

    def run_exam(v28, v261, v262, v45, v51=None):
        return v264(v28, v35, v36, v37, v121, v261, v262, v40, v41, v42, v43, v44, locked=v45, open_thresh=v111.v47, reopen_margin=v111.v48, max_opens=v111.v49, trace=v51)
    v139 = v200(v137, v42)
    v140 = v263(v139, v35, v36, v37, v121, v29, v92, v93, v94, v95, v40, v41, v42, steps=v96, k=v43, gate_l1=v111.v89, nce_q=v97, nce_slot=v98, nce_w=v111.v99, nce_tau=v111.v100, open_only=False, rng=v101, t0=v102, tag='soft')
    v141 = v264(v139, v35, v36, v37, v121, v29, v129, v40, v41, v42, v43, v44, locked=False, use_glue=False)
    v142: v34 = []
    v143 = v265(v139, v29, v129, locked=False, trace=v142)
    v144: v34 = []
    v145 = v265(v139, v29, v129, locked=True, trace=v144)
    v236(f"arm A (soft train, soft decode)  EM={v143['em']:.3f} em3={v143['em3']:.3f} gate={v143['gate_mean']:.3f}\narm B (soft train, LOCKED decode) em_span={v145['em_span']:.3f} em_text={v145['em_text']:.3f} glue_bpe={v145['glue_bpe_rate']:.3f} verbatim={v145['verbatim']:.3f} open_rec={v145['open_recall']:.3f} restart={v145['restart_rate']:.3f}")
    v146 = None
    v147 = []
    v148 = None
    v149: v34 = []
    if not v111.v266:
        v148 = v200(v137, v42)
        v147 = v263(v148, v35, v36, v37, v121, v29, v92, v93, v94, v95, v40, v41, v42, steps=v96, k=v43, gate_l1=v111.v89, nce_q=v97, nce_slot=v98, nce_w=v111.v99, nce_tau=v111.v100, open_only=True, rng=v101, t0=v102, tag='open')
        v149 = []
        v146 = v265(v148, v29, v129, locked=True, trace=v149)
        v236(f"arm C (open train, LOCKED decode) em_span={v146['em_span']:.3f} em_text={v146['em_text']:.3f} glue_bpe={v146['glue_bpe_rate']:.3f} verbatim={v146['verbatim']:.3f} open_rec={v146['open_recall']:.3f} restart={v146['restart_rate']:.3f}")
    v150 = v146 if v146 is not None and v146['em'] >= v145['em'] else v145
    v151 = v148 if v150 is v146 else v139
    v152 = 'C_open_locked' if v150 is v146 else 'B_soft_locked'
    v153 = v265(v151, v29.v339(v10 + 1), v129, locked=True)
    v154 = v265(v151, v29.v340(), v129, locked=True)
    v267, v268 = ([], [])
    for v88 in v129:
        v269 = v29.v341()
        v269.v342(v88['value'])
        v267.v296(v265(v151, v269, [v88], locked=True)['em'])
        v270 = [v343 for v343 in v129 if v343 is not v88]
        if v270:
            v268.v296(v265(v151, v269, v270, locked=True)['em'])
    v155 = v109(v391.v204(v267)) if v267 else v109('nan')
    v156 = v109(v391.v204(v268)) if v268 else v109('nan')
    with v195.v68():
        v272, v273, v274 = ([], [], [])
        v271 = v323.v230(v10 + 99)
        for v237 in v182(12):
            v26 = v409.v393(v93, v94, 1, v271, v40, v135).v243(v42)
            v375, v306 = v314(v151, v35, v36, v37, v121, v29, v26, v40, v41, v42, v43, 0.0, True)
            v376, v237 = v314(v151, v35, v36, v37, v121, v29, v26, v40, v41, v42, v43, 0.0, False)
            if v375 is not None and v376 is not None:
                v273.v296(v109(v375))
                v274.v296(v109(v376))
            if not v401.v378(v306):
                v272.v296(v306)
    v157 = v109(v391.v204(v272)) if v272 else v109('nan')
    v158 = v109(v391.v204(v273)) if v273 else v109('nan')
    v159 = v109(v391.v204(v274)) if v274 else v109('nan')

    def split_em(v275):
        v32 = [v80 for v80 in v275['rows'] if v80['gold'] in {v88['value'] for v88 in v127}]
        v276 = [v80 for v80 in v275['rows'] if v80['gold'] in {v88['value'] for v88 in v128}]
        return (v109(v391.v204([v80['em'] for v80 in v32])) if v32 else v109('nan'), v109(v391.v204([v80['em'] for v80 in v276])) if v276 else v109('nan'))
    v277, v278 = v279(v143)
    v280, v281 = v279(v150)
    v160 = v282(v143['rows'])
    v161 = v282(v150['rows'])
    v162 = v283(v151, v121, v37, v29, v129, v40)
    v236(f"length: soft short={v160['em_short']:.3f} long={v160['em_long']:.3f} | locked short={v161['em_short']:.3f} long={v161['em_long']:.3f}\nprior:  soft wiki={v277:.3f} nonsense={v278:.3f} | locked wiki={v280:.3f} nonsense={v281:.3f}\nfull bank: top1={v162.v377('top1')} mrr={v162.v377('mrr')} bank={v162.v377('bank_size')}")
    v163 = v284()
    v164 = v143['em'] > v141['em'] + 0.2 and v162.v377('top1', 0.0) >= 0.95 and (not v401.v378(v143['gate_mean'])) and (v143['gate_mean'] >= 0.5)
    v165 = v150['verbatim'] >= 0.95 * v150['open_recall'] and v150['open_recall'] > 0
    v166 = v150['em'] >= v143['em'] + 0.1
    v167 = not v401.v378(v161['em_long']) and (not v401.v378(v161['em_short'])) and (v161['em_long'] >= v161['em_short'] - 0.1)
    v168 = not v401.v378(v280) and (not v401.v378(v281)) and (v379(v280 - v281) <= 0.1)
    v169 = v150['restart_rate'] <= 0.05
    v170 = v153['em'] <= v197(0.1, v150['em'] - 0.4)
    v171 = v154['em'] <= 0.1
    v172 = v150['em'] >= 0.4 and v155 <= 0.1 and (v156 >= 0.7 * v150['em'])
    v173 = not v401.v378(v158) and (not v401.v378(v159)) and (v158 <= v159 + 0.05)
    v174 = not v401.v378(v150['gate_mean']) and (not v401.v378(v157)) and (v150['gate_mean'] >= v157 + 0.2)
    v90 = v164 and v170 and v171 and v173
    v175 = v90 and v165 and v166 and v169
    if v175 and v167 and v168 and v172 and v174:
        v285 = 'SPAN_LOCK_OK'
    elif v175:
        v285 = 'SPAN_LOCK_PARTIAL'
    elif not v90:
        v285 = 'SPAN_LOCK_INVALID'
    else:
        v285 = 'SPAN_LOCK_NO'

    def strip(v275):
        return {v380: v381 for v380, v381 in v275.v413() if v380 != 'rows'} if v275 else None
    v33 = {'stage': 265, 'overall': v285, 'trunk': v118.v24, 'fp_version': v344(), 'topk': v43, 'steps': v96, 'open_thresh': v111.v47, 'max_new': v44, 'tape_slots': v294(v252), 'n_fit': v294(v92), 'n_eval_wiki': v294(v127), 'n_eval_nonsense': v294(v128), 'best_arm': v152, 'gates': {'G_soft_reproduces_256': v164, 'G_span_verbatim': v165, 'G_locked_beats_soft': v166, 'G_length_flat': v167, 'G_prior_invariant': v168, 'G_no_restart': v169, 'G_tape_causal': v170, 'G_no_param_leak': v171, 'G_slot_delete_clean': v172, 'G_lang_intact': v173, 'G_gate_selective': v174}, 'arms': {'A_soft_soft': v192(v143), 'B_soft_locked': v192(v145), 'C_open_locked': v192(v146), 'head_only': v192(v141)}, 'controls': {'em_shuffled_tape': v153['em'], 'em_empty_tape': v154['em'], 'em_target_after_delete': v155, 'em_retained_after_delete': v156, 'prose_ce_glue_on': v158, 'prose_ce_glue_off': v159, 'gate_mean_prose': v157, 'hold_ce_base': v138}, 'length': {'soft': v160, 'locked': v161}, 'prior_split': {'soft_wiki': v277, 'soft_nonsense': v278, 'locked_wiki': v280, 'locked_nonsense': v281}, 'full_bank_at_cue': v162, 'em_256_published': v163, 'curve': {'soft': v140, 'open': v147}, 'note': 'Span-lock: the gate decides only WHERE a value starts; once open, tape.tok_ids is emitted verbatim and neither the gate nor the LM is consulted until the span ends. Training stays a soft mixture (257: hard-commit-only training collapses the stop gate); open-train additionally withholds gradient from the gate on in-span steps. G_span_verbatim is an assertion: an opened span is bit-identical to the slot. Headline EM is em_span (first word of the emitted span). em_text scores the full max_new continuation alongside — when the LM BPE-glues onto the value (Whammy+n -> Whammyn) em_span stays high and em_text drops; glue_bpe_rate counts that gap. The end-of-value boundary is NOT solved: stop-at-span and em_span only defer it via the one-answer exam. Same defect family as restart; on a tape where generation continues after the value, it returns at full strength. G_length_flat and G_prior_invariant are the scaling claims.', 'deferred': {'end_of_value_boundary': 'Not solved. em_span / max_opens=1 hide LM subword glue and multi-value continuation. Next line: an explicit end-of-value signal (or refuse to hand the wheel back to LM without a boundary), not another exam-only stop.'}, 'timestamp': v414.v402(v415.v403).v345(), 'wall_s': v232.v232() - v102}
    v0.v178(parents=True, exist_ok=True)
    v1.v229(v359.v346(v33, indent=2), encoding='utf-8')
    v4.v229(v359.v346({'soft': v142[:12], 'locked': v144[:12], 'open_locked': v149[:12], 'C_verbatim_em_mismatch': [{'S': v80['S'], 'gold': v80['gold'], 'got': v80['got'], 'em_first_word': v80.v377('em_first_word'), 'gold_decode': v80.v377('gold_decode'), 'em': v80['em'], 'em3': v80['em3'], 'verbatim': v80['verbatim'], 'opened_correct': v80['opened_correct'], 'opened_at': v80['opened_at'], 'opened_value': v80['opened_value'], 'truncated': v80['truncated'], 'n_val_tokens': v80['n_val_tokens'], 'opens': v80.v377('opens')} for v80 in (v146['rows'] if v146 is not None else []) if v18(v80.v377('verbatim')) != v18(v80.v377('em'))], 'B_verbatim_em_mismatch': [{'S': v80['S'], 'gold': v80['gold'], 'got': v80['got'], 'em_first_word': v80.v377('em_first_word'), 'gold_decode': v80.v377('gold_decode'), 'em': v80['em'], 'verbatim': v80['verbatim'], 'opened_correct': v80['opened_correct'], 'opened_at': v80['opened_at'], 'truncated': v80['truncated'], 'n_val_tokens': v80['n_val_tokens']} for v80 in v145['rows'] if v18(v80.v377('verbatim')) != v18(v80.v377('em'))]}, indent=2), encoding='utf-8')
    v2.v229(f"# Stage 265 span-lock\n\n**{v285}** trunk={v118.v24} slots={v294(v252)} eval={v294(v129)} best={v152}\n\n- EM headline (=em_span): head_only **{v141['em']:.3f}** | soft **{v143['em']:.3f}** | locked **{v150.v377('em_span', v150['em']):.3f}** (em_text **{v150.v377('em_text', v109('nan')):.3f}**, glue_bpe **{v150.v377('glue_bpe_rate', v109('nan')):.3f}**)\n- verbatim spans **{v150['verbatim']:.3f}**, open recall {v150['open_recall']:.3f}, restart {v150['restart_rate']:.3f} (undefined for the soft arm: it opens no spans)\n- length (short/long): soft {v160['em_short']:.2f}/{v160['em_long']:.2f}, locked {v161['em_short']:.2f}/{v161['em_long']:.2f}\n- prior (wiki/nonsense): soft {v277:.2f}/{v278:.2f}, locked {v280:.2f}/{v281:.2f}\n- causal: shuffled {v153['em']:.3f}, empty {v154['em']:.3f}, delete {v150['em']:.2f}->{v155:.2f} (retained {v156:.2f})\n- full bank @ cue: top1 {v162.v377('top1')} mrr {v162.v377('mrr')} over {v162.v377('bank_size')} slots\n- deferred: end-of-value boundary not solved — em_span only defers LM BPE-glue\n", encoding='utf-8')
    v236(v359.v346({'overall': v285, 'gates': v33['gates'], 'arms': v33['arms']}, indent=2))
    if not v111.v235 and v151 is not None:
        v8.v289.v178(exist_ok=True)
        v195.v347({'W_q_glue': v151.v300.v404(), 'gate': v151.v416.v404(), 'log_tau': v151.v424.v421().v405(), 'stage': 265, 'arm': v152, 'fp_version': v344()}, v8)
    return 0
if v176 == '__main__':
    raise v286(v348())