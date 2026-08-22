"""
Stage 282 — One mind: typed silence, an editable query, and verification by the return path.

278 gave the controller a value baseline, a BC anchor that survives RL, an exhaustive teacher and
a margin counted in votes; 279 gave the tape a write decision; 280 put the two on raw text and
281 fixed what an assertion is. What is left is the policy itself, and three things it cannot
currently express - each of them lexical, each judged by the corpus rather than by a label.

  A. SILENCE IS TYPED.
     One STOP served two situations that have opposite continuations: the witnesses contradict
     each other, and nothing was found at all. STOP_CONFLICT and STOP_UNKNOWN cost nothing - the
     features that separate them are already in the state - and abstention stops being a scalar
     and becomes a diagnosis, which is separately checkable: UNKNOWN must land where the votes
     were silent, CONFLICT where support was equal. If the two are used interchangeably that
     shows up as a failed gate rather than as a good-looking abstention rate.

  B. THE QUERY IS EDITABLE.
     ASK_Q always reissued the same cue, so a bad retrieve list left the mind a choice between
     reading rubbish and giving up. 261 established that the query must be WORDS, and a set of
     words can be edited: DROP_i removes one, ADD_i takes one from the passage just read.
     Retrieval stops being a fixed function applied to the policy and becomes part of it.

     A and B compose. A typed silence is a reason: UNKNOWN means reformulate, CONFLICT means
     stop. Without the type there is no way to tell "ask differently" from "there is no answer".

  C. VERIFICATION BY THE RETURN PATH.
     Counting witnesses is the only check the mind had. There is a stronger one and it is also
     lexical: take the value just read and ask the tape about IT. Sleaford -> Lincolnshire, then
     ask Lincolnshire and see whether the tape leads back to Sleaford. A plausible-but-wrong
     value usually has no return path; a right one usually does. ASK_VALUE_i is a PROBE - it
     restores the candidate list afterwards, so it checks without navigating away, and the
     ANSWER indices stay meaningful.

Nothing here is trained to address anything: no query vector (261 measured that it loses), no
embedding of the question, no reading of item['truth'] or item['kind'] anywhere including the
teacher. The tape comes from 280 unchanged, so a difference against 280 is a difference from
these three actions.

  python _stage282_mind.py --smoke
  python _stage282_mind.py --bc-episodes 4000 --rl-episodes 3000 --min-mentions 2
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
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
import _stage271_controller as s271
import _stage280_raw_exam as s280
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _inprint_glue import hidden_and_logits
from _tape_index import context_words
v0 = v11('results')
v1 = v11('checkpoints/stage191_p1_curve.pt')
v2 = v11('checkpoints/stage253_joint_l02.pt')
v3 = v11('data/_wikitext103_train.txt')
v4 = 282
v5 = v12.v5
v6 = v0 / '_stage282_log.txt'
v7 = 10

def log(v13: v15) -> None:
    v14 = v13 if v13.v231('\n') else v13 + '\n'
    try:
        v232(v14, end='', flush=True)
    except v112:
        v232(v14.v371('ascii', 'replace').v334('ascii'), end='', flush=True)
    v6.v233.v113(parents=True, exist_ok=True)
    with v6.v234('a', encoding='utf-8') as v114:
        v114.v235(v14)

class Acts:
    """Nine families, all lexical. Nothing here indexes an embedding."""

    def __init__(v115, v62: v10, v78: v10):
        v115.v62, v115.v78 = (v62, v78)
        v115.v116 = 0
        v115.v117 = 1
        v115.v118 = 2
        v115.v119 = v115.v118 + v62
        v115.v120 = v115.v119 + v62
        v115.v121 = v115.v120 + v62
        v115.v122 = v115.v121 + v78
        v115.v123 = v115.v122 + v78
        v115.v124 = v115.v123 + 1
        v115.v125 = v115.v124 + 1

    def name(v115, v126: v10) -> v15:
        for v236, v70 in ((v115.v118, 'READ'), (v115.v119, 'ANSWER'), (v115.v120, 'ASK_VALUE'), (v115.v121, 'DROP'), (v115.v122, 'ADD')):
            v237 = v115.v78 if v70 in ('DROP', 'ADD') else v115.v62
            if v236 <= v126 < v236 + v237:
                return f'{v70}_{v126 - v236}'
        return {v115.v116: 'ASK_Q', v115.v117: 'ASK_READ', v115.v123: 'STOP_CONFLICT', v115.v124: 'STOP_UNKNOWN'}.v159(v126, f'?{v126}')

class Policy(v16.v8):
    """Global head for the query and the two silences; per-candidate head for the rest.

    Candidate positions get no global logit - 274 measured what happens otherwise: the positional
    head learns that ANSWER_0 is right often enough and drowns the per-candidate signal.
    """

    def __init__(v115, v91: v10, v25: v187, v24):
        v335().v238()
        v115.v25, v115.v239 = (v25, v25.v125)
        v127 = v91 + v7
        v115.v114 = v16.v336(v16.v358(v127, 128), v16.v359(), v16.v358(128, v25.v125)).v195(v24)
        v115.v128 = v16.v336(v16.v358(v127 + 4, 64), v16.v359(), v16.v358(64, 3)).v195(v24)
        v115.v129 = v16.v336(v16.v358(v127 + 2, 32), v16.v359(), v16.v358(32, 2)).v195(v24)
        v115.v130 = v16.v336(v16.v358(v127, 128), v16.v359(), v16.v358(128, 1)).v195(v24)
        for v13 in (v115.v114, v115.v128, v115.v129, v115.v130):
            v16.v337.v295(v13[-1].v296)
            v16.v337.v295(v13[-1].v297)
        v115.v131 = None

    def forward(v115, v28, v36, v37, v132=None, v133=None):
        v134 = v246.v240([v28, v36], dim=-1)
        if v115.v131 is not None:
            v115.v131.v256(v115.v130(v134).v338(-1))
        v135 = v115.v114(v134)
        v126 = v115.v25
        v136 = v246.v241(v135)
        for v39 in (v126.v116, v126.v117, v126.v123, v126.v124):
            v136 = v136.v298(0, v246.v141([v39], device=v135.v24), v135[v39].v339(1))
        if v132 is not None and v132.v299():
            v125 = v132.v300(0)
            v138 = v115.v128(v246.v240([v134.v374(0).v372(v125, -1), v132], dim=-1))
            for v252, v301 in v145((v126.v118, v126.v119, v126.v120)):
                v47 = v246.v340(v301, v301 + v125, device=v135.v24)
                v136 = v136.v341(0, v47, v138[:, v252])
        if v133 is not None and v133.v299():
            v125 = v133.v300(0)
            v242 = v115.v129(v246.v240([v134.v374(0).v372(v125, -1), v133], dim=-1))
            for v252, v301 in v145((v126.v121, v126.v122)):
                v47 = v246.v340(v301, v301 + v125, device=v135.v24)
                v136 = v136.v341(0, v47, v242[:, v252])
        return v136.v243(~v37, -1000000000.0)

def build_state(v17, v18, v19, v20, v21, v22, v23, v24, v25, v26):
    v27 = [v39 for v39 in v20.v371(v22['transcript']).v27 if v39 != v23][-v245.v342:] if v244(v245, 'MAX_ARCS') else [v39 for v39 in v20.v371(v22['transcript']).v27 if v39 != v23][-256:]
    if not v27:
        return None
    v28, v64 = v137(v18, v19, v246.v141([v27], dtype=v246.v302, device=v24), v23)
    v28 = v28[0, -1]
    if v9:
        v28 = v28[:0]
    v41, v138 = (v22['cands'], v22['sc'])
    v29 = [v138.v159(v144, 0.0) for v144 in v41]
    v30 = v152(v29) if v29 else 0.0
    v31 = v303(v29, reverse=True)[1] if v247(v29) > 1 else 0.0
    v32 = v139(v22['opened'])
    v33 = v32.v140(2)
    v34 = v33[0][1] if v33 else 0
    v35 = v33[1][1] if v247(v33) > 1 else 0
    v36 = v246.v141([v30, v30 - v31, v247(v41) / v152(1, v25.v62), v22['n_reads'] / v26, v254(v143(v22['last_words'])), v34 / v152(1, v247(v22['opened']) or 1), v254(v146(v34 - v35, 3)), v247(v22['qwords']) / v152(1, v25.v78), v22['n_edits'] / v152(1, v25.v78), v254(v30 <= 0.0)], device=v24, dtype=v28.v304 if v28.v299() else v246.v305)
    v37 = v246.v142(v25.v125, dtype=v246.v143, device=v24)
    v37[v25.v116] = True
    v37[v25.v117] = v143(v22['last_words'])
    v37[v25.v123] = True
    v37[v25.v124] = True
    for v39, v144 in v145(v41[:v25.v62]):
        v37[v25.v118 + v39] = v144 not in v22['seen']
        v37[v25.v119 + v39] = True
        v37[v25.v120 + v39] = v22['probes'] < v22['max_probes']
    v38 = v146(v247(v22['qwords']), v25.v78)
    for v39 in v147(v38):
        v37[v25.v121 + v39] = v247(v22['qwords']) > 1 and v22['n_edits'] < v25.v78
    for v39 in v147(v146(v247(v22['addable']), v25.v78)):
        v37[v25.v122 + v39] = v22['n_edits'] < v25.v78
    v40 = None
    if v41:
        v148 = [v21['tape'].v306[v144] for v144 in v41]
        v149 = v139(v22['opened'])
        v150 = v152(v29) if v29 and v152(v29) > 0 else 1.0
        v151 = v152(1, v247(v22['opened']))
        v40 = v246.v141([[v29[v39] / v150, v149.v159(v148[v39], 0) / v151, v254(v41[v39] in v22['seen']), v146(v22['ret_ok'].v159(v41[v39], -1.0), 2.0)] for v39 in v147(v247(v41))], device=v24, dtype=v36.v304)
    v42 = None
    v43 = v152(v38, v146(v247(v22['addable']), v25.v78))
    if v43:
        v153 = []
        for v39 in v147(v43):
            v248 = v22['qwords'][v39] if v39 < v38 else None
            v249 = v22['addable'][v39] if v39 < v247(v22['addable']) else None
            v153.v256([v21['idf'].v159(v248, 0.0) if v248 else 0.0, v21['idf'].v159(v249, 0.0) if v249 else 0.0])
        v42 = v246.v141(v153, device=v24, dtype=v36.v304)
    return (v17(v28, v36, v37, v40, v42), v37)
v9 = False

def teacher(v21, v22, v25, v44, v26, v45):
    """Executable, label-free, and it demonstrates every new action.

    An action the teacher never takes is an action BC never shows the policy, so each of the
    three additions has a rule here that any reader could run by hand.
    """
    v41 = v22['cands']
    if not v41:
        if not v22['asked']:
            return v25.v116
        if v247(v22['qwords']) > 1 and v22['n_edits'] < v25.v78:
            v236 = v146(v147(v146(v247(v22['qwords']), v25.v78)), key=lambda v39: v21['idf'].v159(v22['qwords'][v39], 0.0))
            return v25.v121 + v236
        if v22['addable'] and v22['n_edits'] < v25.v78:
            v250 = v152(v147(v146(v247(v22['addable']), v25.v78)), key=lambda v39: v21['idf'].v159(v22['addable'][v39], 0.0))
            return v25.v122 + v250
        return v25.v124
    v46 = [v39 for v39, v144 in v145(v41[:v25.v62]) if v144 not in v22['seen']]
    if v46 and v22['n_reads'] < v45 and (v22['n_reads'] + 3 <= v26):
        return v25.v118 + v46[0]
    v32 = v139(v22['opened'])
    v33 = v32.v140(2)
    v34 = v33[0][1] if v33 else 0
    v31 = v33[1][1] if v247(v33) > 1 else 0
    if v34 == 0:
        return v25.v124

    def slot_of(v154):
        return v251((v39 for v39, v144 in v145(v41[:v25.v62]) if v21['tape'].v306[v144] == v154), None)
    if v34 == v31 and (not v22.v159('tie_probe')):
        return v25.v123
    if v34 == v31:
        for v154 in (v33[0][0], v33[1][0]):
            v252 = v158(v154)
            if v252 is not None and v22['ret_ok'].v159(v41[v252], -1.0) < 0 and (v22['probes'] < v22['max_probes']) and (v22['n_reads'] + 2 <= v26):
                return v25.v120 + v252
        v155 = [(v154, v158(v154)) for v154 in (v33[0][0], v33[1][0])]
        v155 = [(v252, v22['ret_ok'].v159(v41[v252], -1.0)) for v64, v252 in v155 if v252 is not None]
        v156 = [v252 for v252, v125 in v155 if v125 >= 2]
        v157 = [v252 for v252, v125 in v155 if v125 > 0]
        if v247(v156) == 1 and v247(v157) == 1:
            return v25.v119 + v156[0]
        return v25.v123
    v47 = v158(v33[0][0])
    if v47 is None:
        return v25.v124
    v48 = v22['ret_ok'].v159(v41[v47], -1.0)
    if v48 < 0 and v22['probes'] < v22['max_probes'] and (v22['n_reads'] + 2 <= v26):
        return v25.v120 + v47
    if v48 == 0.0 and v34 - v31 < 2:
        return v25.v124
    return v25.v119 + v47

def rollout(v17, v18, v19, v20, v21, v44, v23, v24, v25, *, v26, v45, v49, v50, v51, v52, v53, v54, v55, v56=2, v57=False, v58=False, v59=True, v60=False, v61=0.0):
    v62 = v25.v62
    v22 = {'transcript': v245.v307.v253(S=v44.v159('query') or v44['S']), 'qwords': v308(v245.v307.v253(S=v44.v159('query') or v44['S']))[:v25.v78], 'cands': [], 'sc': {}, 'seen': v170(), 'opened': [], 'last_words': [], 'addable': [], 'n_reads': 0, 'n_edits': 0, 'probes': 0, 'max_probes': v56, 'asked': False, 'ret_ok': {}, 'tie_probe': v57}
    v160, v161, v162, v163 = ([], [], [], [])
    v164, v165, v166 = (None, None, True)
    v167, v168, v169 = (v254('nan'), v254('nan'), 0)
    v63 = v170(v44['slots'])

    def do_ask(v171):
        v144, v138, v172 = v12.v255(v21, v171, v62, v52, v44, v55, v53, v54)
        v22['cands'], v22['sc'], v22['asked'] = (v144, v138, True)
        return v172
    for v64 in v147(v26):
        if v60:
            v126 = v309(v21, v22, v25, v44, v26, v45)
        else:
            v110 = v310(v17, v18, v19, v20, v21, v22, v23, v24, v25, v26)
            if v110 is None:
                break
            v136, v64 = v110
            if v58:
                v126 = v309(v21, v22, v25, v44, v26, v45)
                if not v246.v373(v136[v126]) or v136[v126] < -100000000.0:
                    break
                v160.v256(v328.v360(v136.v374(0), v246.v141([v126], device=v24)))
            else:
                v311 = v246.v361.v343(logits=v136)
                v126 = v10(v136.v375()) if v59 else v10(v311.v376())
                v161.v256(v311.v362(v246.v141(v126, device=v24)))
                v162.v256(v311.v363())
                if v61 > 0.0:
                    v344 = v309(v21, v22, v25, v44, v26, v45)
                    if v246.v373(v136[v344]) and v136[v344] > -100000000.0:
                        v160.v256(v328.v360(v136.v374(0), v246.v141([v344], device=v24)))
        v163.v256(v25.v312(v126))
        if v126 == v25.v116:
            v169 += v10(v345(v22['qwords']))
        elif v126 == v25.v117:
            v169 += v10(v345(v22['last_words'] or v22['qwords']))
        elif v25.v120 <= v126 < v25.v120 + v62:
            v39 = v126 - v25.v120
            if v39 >= v247(v22['cands']):
                break
            v346 = v22['cands'][v39]
            v364, v365 = (v22['cands'], v22['sc'])
            v347 = v308(v21['tape'].v306[v346]) or [v21['tape'].v306[v346]]
            v366, v64, v64 = v12.v255(v21, v347, v62, v52, None, False, v53, v54, index='probe')
            v348 = v21['tape'].v306[v346].v367()
            v22['ret_ok'][v346] = v254(v313((1 for v144 in v366 if v144 != v346 and v44['S'] in v21['texts_lc'][v144] and (v348 in v21['texts_lc'][v144]))))
            v22['cands'], v22['sc'] = (v364, v365)
            v22['probes'] += 1
        elif v25.v121 <= v126 < v25.v121 + v25.v78:
            v39 = v126 - v25.v121
            if v39 >= v247(v22['qwords']) or v247(v22['qwords']) <= 1:
                break
            v22['qwords'] = v22['qwords'][:v39] + v22['qwords'][v39 + 1:]
            v22['n_edits'] += 1
        elif v25.v122 <= v126 < v25.v122 + v25.v78:
            v39 = v126 - v25.v122
            if v39 >= v247(v22['addable']):
                break
            v78 = v22['addable'][v39]
            if v78 not in v22['qwords']:
                v22['qwords'] = (v22['qwords'] + [v78])[:v25.v78]
            v22['n_edits'] += 1
        elif v126 in (v25.v123, v25.v124):
            v165 = 'conflict' if v126 == v25.v123 else 'unknown'
            v166 = False
            break
        elif v25.v118 <= v126 < v25.v118 + v62:
            v39 = v126 - v25.v118
            if v39 >= v247(v22['cands']):
                break
            v346 = v22['cands'][v39]
            v22['transcript'] = (v22['transcript'] + ' | ' + v21['texts'][v346])[-2000:]
            v22['last_words'] = v308(v21['texts'][v346], exclude=v21['tape'].v306[v346])
            v22['addable'] = [v78 for v78 in v22['last_words'] if v78 not in v22['qwords']][:v25.v78]
            v22['seen'].v382(v346)
            v22['opened'].v256(v21['tape'].v306[v346])
            v22['n_reads'] += 1
        else:
            v39 = v126 - v25.v119
            if v39 >= v247(v22['cands']):
                break
            v164 = v21['tape'].v306[v22['cands'][v39]]
            v166 = False
            break
        if v22['cands'] and v332.v288(v167):
            v257 = v313((1 for v144 in v22['cands'] if v144 in v63))
            v167 = v257 / v247(v22['cands'])
            v168 = v257 / v152(1, v247(v63))
    v65 = v164 is None
    if v65:
        v173 = 0
        v66 = 0.0 if v166 else v51
    else:
        v173 = v10(v44['truth'] is not None and v164 == v44['truth'])
        v66 = 1.0 if v173 else -v50
    v66 -= v49 * v22['n_reads']
    return {'loss': v246.v278(v160).v314() if v160 else v246.v142((), device=v24), 'logps': v161, 'entropy': v162, 'reward': v66, 'correct': v173, 'abstained': v65, 'stop_kind': v165, 'stalled': v143(v166 and v65), 'n_reads': v22['n_reads'], 'n_edits': v22['n_edits'], 'probes': v22['probes'], 'ret_ok': v152(v22['ret_ok'].v306()) if v22['ret_ok'] else v254('nan'), 'trace': v163, 'kind': v44['kind'], 'hops': v169, 'answer_is_slot': v164 is None or v164 in v170(v21['tape'].v306), 'retrieval_precision': v167, 'witness_recall': v168, 'words_silent': v143(v22['sc']) is False or v152(v22['sc'].v306(), default=0.0) <= 0.0}

def main() -> v10:
    v67 = v258.v174()
    v67.v175('--smoke', action='store_true')
    v67.v175('--bc-episodes', type=v10, default=0)
    v67.v175('--rl-episodes', type=v10, default=0)
    v67.v175('--tape-period', type=v10, default=0)
    v67.v175('--addresses', type=v10, default=0)
    v67.v175('--min-mentions', type=v10, default=2)
    v67.v175('--min-per-family', type=v10, default=8)
    v67.v175('--address-tau', type=v254, default=0.9)
    v67.v175('--address-overlap', type=v10, default=2)
    v67.v175('--soft-match', type=v254, default=0.0)
    v67.v175('--topk', type=v10, default=7)
    v67.v175('--query-words', type=v10, default=6, help='editable query slots')
    v67.v175('--max-probes', type=v10, default=2)
    v67.v175('--max-steps', type=v10, default=14)
    v67.v175('--max-reads', type=v10, default=7)
    v67.v175('--hop', choices=('none', 'fp'), default='fp')
    v67.v175('--hop-min', type=v254, default=1.0)
    v67.v175('--k-gap', type=v254, default=0.35)
    v67.v175('--read-cost', type=v254, default=0.02)
    v67.v175('--wrong-cost', type=v254, default=1.0)
    v67.v175('--abstain-reward', type=v254, default=0.75)
    v67.v175('--entropy-bonus', type=v254, default=0.01)
    v67.v175('--lr-policy', type=v254, default=0.001)
    v67.v175('--lr-value', type=v254, default=0.003)
    v67.v175('--lr-upper', type=v254, default=3e-05)
    v67.v175('--value-coef', type=v254, default=0.5)
    v67.v175('--bc-anchor', type=v254, default=0.5)
    v67.v175('--no-hidden', action='store_true')
    v67.v175('--no-edit', action='store_true', help='ablation: drop B')
    v67.v175('--no-probe', action='store_true', help='ablation: drop C')
    v67.v175('--tie-probe', action='store_true', help='ablation: let the return path overturn a tie (two witnesses)')
    v67.v175('--subject-filter', choices=('off', 'on'), default='on')
    v67.v175('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v67.v175('--frozen-trunk', action='store_true')
    v67.v175('--run-tag', type=v15, default='')
    v68 = v67.v176()
    global NO_HIDDEN, LOG_PATH
    v9 = v68.v69
    v70 = v68.v226 and f'_{v68.v226}' or ''
    v70 += '_nohid' if v68.v69 else ''
    v70 += '_noedit' if v68.v177 else ''
    v70 += '_noprobe' if v68.v178 else ''
    v70 += '_tieprobe' if v68.v57 else ''
    v6 = v0 / f'_stage282_log{v70}.txt'
    v6.v233.v113(parents=True, exist_ok=True)
    v6.v179('', encoding='utf-8')
    v24 = v246.v24('cuda' if v246.v349.v315() else 'cpu')
    v71 = v259.v180(v4)
    v246.v181(v4)
    v72 = v182.v182()
    v73 = v68.v183 or (400 if v68.v227 else 4000)
    v74 = v152(0, v68.v184)
    v75 = v68.v75 or (50 if v68.v227 else 200)
    v76 = v68.v185 or (60 if v68.v227 else 400)
    v62 = v68.v77
    v78 = 0 if v68.v177 else v68.v186
    v25 = v187(v62, v78)
    v79 = 'none' if v68.v188 else 'upper'
    v189(f'Stage282 mind start {v369.v356(v370.v357).v292()} device={v24} actions={v25.v125} k={v62} w={v78} probes={(0 if v68.v178 else v68.v56)} bc={v73} rl={v74}')
    v64, v64, v190, v191 = v192()
    v20 = v260.v193(v15(v316.v261))
    v80 = v20.v194()
    v23 = v20.v262(v263) or 0
    v19 = v350.v317(v20, v190, v23, v80).v195(v24)
    v81 = v2 if v2.v264() else v1
    v18 = v318(v191, v80).v195(v24)
    v18.v196(v246.v319(v81, map_location=v24, weights_only=False)['model'])
    v265.v197(v18, v79)
    v82 = v245.v198(v18)
    v83 = v318(v191, v80).v195(v24)
    v83.v196(v246.v319(v1, map_location=v24, weights_only=False)['model'])
    v83.v199()
    for v84 in v83.v200():
        v84.v266(False)
    v85 = v201(v83, v190, v24)
    with v3.v234('r', encoding='utf-8', errors='ignore') as v114:
        v202 = v114.v267(4000000 if v68.v227 else 30000000)
    v86 = [v269.v268() for v269 in v202.v320('\n') if 80 <= v247(v269.v268()) <= 400]
    v87 = v10(0.7 * v247(v86))
    v88 = v86[:v87][:3000 if v68.v227 else 25000]
    v89 = v86[v87:][:1500 if v68.v227 else 12000]

    def new_pack(v203, v204):
        return v12.v270(v204, bank=v85, tok=v20, pad_id=v23, device=v24, rng=v203, n_addr=v76, min_mentions=v68.v228, tau=v68.v321, overlap=v68.v322, soft_match=v68.v323, min_per_family=v68.v229, addr_key=v68.v324)
    v21 = v205(v71, v88)
    v90 = v139((v39['kind'] for v39 in v21['items']))
    v189(f"  tape: {v21['n_addresses']} addresses, {v21['n_slots']} slots | items {v333.v293(v208(v90))} ({v182.v182() - v72:.0f}s)")
    if v247(v21['items']) < 8:
        v189('  too few items')
        return 1
    v91 = 0 if v68.v69 else 2 * (v18.v351.v325 // 2)
    v17 = v206(v91, v25, v24)
    v92 = [v84 for v84 in v18.v200() if v84.v271]
    v93 = v246.v272.v207([{'params': [v84 for v378, v84 in v17.v379() if not v378.v380('v.')], 'lr': v68.v352}, {'params': v281(v17.v130.v200()), 'lr': v68.v353}] + ([{'params': v92, 'lr': v68.v368}] if v92 else []), weight_decay=0.01)
    v94 = v208(max_steps=v68.v26, max_reads=v68.v45, read_cost=v68.v49, wrong_cost=v68.v50, abstain_reward=v68.v51, hop=v68.v52, hop_min=v68.v53, k_gap=v68.v54, subject_filter=v68.v55 == 'on', max_probes=0 if v68.v178 else v68.v56, tie_probe=v68.v57)
    v209, v210 = ([], [])
    v17.v211()
    v18.v211(v79 != 'none')
    for v95 in v147(1, v73 + 1):
        if (v95 - 1) % v75 == 0 and v95 > 1:
            v21 = v205(v71, v88)
        v44 = v21['items'][v71.v326(v247(v21['items']))]
        v110 = v273(v17, v18, v19, v20, v21, v44, v23, v24, v25, bc=True, **v94)
        v93.v274(set_to_none=True)
        v110['loss'].v275()
        v246.v16.v327.v276(v281(v17.v200()) + v92, 1.0)
        v93.v277()
        if v95 % v152(1, v73 // 8) == 0:
            v209.v256({'phase': 'bc', 'episode': v95, 'loss': v254(v110['loss']), 'kind': v110['kind'], 'trace': v110['trace']})
            v189(f"  bc {v95}/{v73} loss={v254(v110['loss']):.4f} [{v110['kind']}] {v110['trace']}")
    for v95 in v147(1, v74 + 1):
        if (v95 - 1) % v75 == 0 and v95 > 1:
            v21 = v205(v71, v88)
        v44 = v21['items'][v71.v326(v247(v21['items']))]
        v17.v131 = []
        v110 = v273(v17, v18, v19, v20, v21, v44, v23, v24, v25, greedy=False, bc_anchor=v68.v61, **v94)
        v148, v17.v131 = (v17.v131, None)
        if not v110['logps']:
            continue
        v212 = v110['reward']
        v213 = v246.v278(v148[:v247(v110['logps'])])
        v214 = v328.v279(v213, v246.v329(v213, v212))
        v210.v256(v254(v214))
        v215 = v246.v278(v110['entropy']).v313() if v110['entropy'] else v246.v142((), device=v24)
        v216 = -((v212 - v213).v381() * v246.v278(v110['logps'])).v313() + v68.v354 * v214 - v68.v330 * v215
        if v68.v61 > 0.0 and v110['loss'].v271:
            v216 = v216 + v68.v61 * v110['loss']
        v93.v274(set_to_none=True)
        v216.v275()
        v246.v16.v327.v276(v281(v17.v200()) + v92, 1.0)
        v93.v277()
        if v95 % v152(1, v74 // 8) == 0:
            v209.v256({'phase': 'rl', 'episode': v95, 'v_mse': v254(v377.v314(v210[-200:])), 'kind': v110['kind'], 'trace': v110['trace']})
            v189(f"  rl {v95}/{v74} v_mse={v377.v314(v210[-200:]):.3f} [{v110['kind']}] {v110['trace']}")
    v17.v199()
    v18.v199()
    v96 = v245.v198(v18)

    @v246.v222()
    def evaluate(v84):
        v217 = {v114: v280(v281) for v114 in v5}
        v218 = {v114: v280(v281) for v114 in v5}
        v219 = v280(v281)
        v220 = []
        for v221 in v84['items']:
            v282 = v273(v17, v18, v19, v20, v84, v221, v23, v24, v25, **v94)
            v283 = v273(v17, v18, v19, v20, v84, v221, v23, v24, v25, teacher_only=True, **v94)
            v114 = v221['kind']
            for v284 in ('correct', 'n_reads', 'reward', 'n_edits', 'probes'):
                v217[v114][v284].v256(v282[v284])
            v217[v114]['abstain'].v256(v10(v282['abstained']))
            if not v332.v288(v282['retrieval_precision']):
                v217[v114]['prec'].v256(v282['retrieval_precision'])
                v217[v114]['rec'].v256(v282['witness_recall'])
            v217[v114]['conflict'].v256(v10(v282['stop_kind'] == 'conflict'))
            v217[v114]['unknown'].v256(v10(v282['stop_kind'] == 'unknown'))
            v218[v114]['correct'].v256(v283['correct'])
            v218[v114]['abstain'].v256(v10(v283['abstained']))
            v218[v114]['reward'].v256(v283['reward'])
            v219['slot'].v256(v10(v282['answer_is_slot']))
            v219['stall'].v256(v10(v282['stalled']))
            v219['silent'].v256(v10(v282['words_silent']))
            v219['hops'].v256(v282['hops'])
            if v282['stop_kind'] == 'unknown':
                v219['unknown_when_silent'].v256(v10(v282['words_silent']))
            if v282['stop_kind'] == 'conflict':
                v219['conflict_when_tie'].v256(v10(v114 == 'tie'))
            if not v332.v288(v282['ret_ok']):
                v219['probe_hit'].v256(v254(v282['ret_ok'] > 0))
                if not v282['abstained']:
                    v219['acc_when_probe_hit' if v282['ret_ok'] > 0 else 'acc_when_probe_miss'].v256(v282['correct'])
            if v247(v220) < 24:
                v220.v256({'kind': v114, 'S': v221['S'], 'trace': v282['trace'], 'correct': v282['correct'], 'stop': v282['stop_kind'], 'edits': v282['n_edits'], 'probes': v282['probes']})
        v13 = lambda v331: v254(v377.v314(v331)) if v331 else v254('nan')
        v110 = {'answer_is_slot': v13(v219['slot']), 'stall_rate': v13(v219['stall']), 'words_silent_rate': v13(v219['silent']), 'hops_per_episode': v13(v219['hops']), 'unknown_when_silent': v13(v219['unknown_when_silent']), 'conflict_when_tie': v13(v219['conflict_when_tie']), 'probe_hit_rate': v13(v219['probe_hit']), 'acc_when_probe_hit': v13(v219['acc_when_probe_hit']), 'acc_when_probe_miss': v13(v219['acc_when_probe_miss']), 'reward_total': v13([v203 for v114 in v5 for v203 in v217[v114]['reward']]), 'teacher_reward_total': v13([v203 for v114 in v5 for v203 in v218[v114]['reward']]), 'retrieval_precision': v13([v134 for v114 in v5 for v134 in v217[v114]['prec']]), 'witness_recall': v13([v134 for v114 in v5 for v134 in v217[v114]['rec']]), 'n_items': v247(v84['items']), 'traces': v220}
        v285, v286 = (0, 0)
        for v114 in v5:
            v287 = v313((1 for v126 in v217[v114]['abstain'] if not v126))
            v285 += v313(v217[v114]['correct'])
            v286 += v287
            v110[v114] = {'n': v247(v217[v114]['abstain']), 'coverage': 1.0 - v13(v217[v114]['abstain']), 'acc_answered': v313(v217[v114]['correct']) / v287 if v287 else v254('nan'), 'abstain': v13(v217[v114]['abstain']), 'stop_conflict': v13(v217[v114]['conflict']), 'stop_unknown': v13(v217[v114]['unknown']), 'mean_reads': v13(v217[v114]['reads']) if v217[v114]['reads'] else v13(v217[v114]['n_reads']), 'mean_edits': v13(v217[v114]['n_edits']), 'mean_probes': v13(v217[v114]['probes']), 'reward': v13(v217[v114]['reward']), 'teacher_abstain': v13(v218[v114]['abstain']), 'teacher_acc_all': v13(v218[v114]['correct'])}
        v110['coverage_all'] = v286 / v152(1, v247(v84['items']))
        v110['acc_answered_all'] = v285 / v152(1, v286)
        v110['teacher_coverage_all'] = v313((v247(v218[v114]['abstain']) - v313(v218[v114]['abstain']) for v114 in v5)) / v152(1, v247(v84['items']))
        return v110
    v97 = v223(v21)
    v98 = v205(v259.v180(v4 + 99), v89)
    v99 = v223(v98)
    v189(f"  HELD-OUT {v333.v293({v289: v290 for v289, v290 in v99.v355() if v289 != 'traces'})}")
    v100 = v99['teacher_reward_total']
    v101 = v82 == v96
    v102 = v99['answer_is_slot'] >= 0.99
    v103 = v224((v99[v114]['n'] >= 4 for v114 in v5))
    v104 = v100 >= 0.5 * v68.v51 and 1.0 - v99['clean']['teacher_abstain'] >= 0.5
    v105 = v99['reward_total'] >= v100 - 0.1 and v99['coverage_all'] >= 0.5 * v99['teacher_coverage_all']
    v106 = v99['conflict_when_tie'] >= 0.6 and (v332.v288(v99['unknown_when_silent']) or v99['unknown_when_silent'] >= 0.5)
    v107 = v99['tie']['stop_conflict'] >= 0.5
    v108 = v68.v178 or v332.v288(v99['acc_when_probe_miss']) or v99['acc_when_probe_hit'] > v99['acc_when_probe_miss']
    v109 = v99['acc_answered_all'] >= 0.6
    if not (v101 and v102 and v103):
        v225 = 'MIND_INVALID'
    elif not v104:
        v225 = 'TEACHER_UNUSABLE'
    elif v105 and v106 and v107 and v109:
        v225 = 'MIND_OK'
    elif v106 or v107:
        v225 = 'MIND_PARTIAL'
    else:
        v225 = 'MIND_NO'
    v110 = {'stage': 282, 'overall': v225, 'actions': v25.v125, 'topk': v62, 'query_words': v78, 'max_probes': v94['max_probes'], 'no_hidden': v68.v69, 'no_edit': v68.v177, 'no_probe': v68.v178, 'tie_probe': v68.v57, 'run_tag': v68.v226, 'smoke': v68.v227, 'seed': v4, 'bc_episodes': v73, 'rl_episodes': v74, 'min_mentions': v68.v228, 'min_per_family': v68.v229, 'reward': {'correct': 1.0, 'wrong': -v68.v50, 'abstain': v68.v51, 'stall': 0.0, 'read': -v68.v49}, 'teacher_ceiling_reward': v100, 'gates': {'G_arc_enc_frozen': v101, 'G_answer_is_slot': v102, 'G_all_families_present': v103, 'G_teacher_usable': v104, 'G_reaches_teacher': v105, 'G_silence_is_typed': v106, 'G_conflict_on_tie': v107, 'G_probe_filters_wrong_answers': v108, 'G_acc_when_answering': v109}, 'train_tape': {v289: v290 for v289, v290 in v97.v355() if v289 != 'traces'}, 'held_out': v99, 'curve': v209, 'arc_enc_hash_before': v82, 'arc_enc_hash_after': v96, 'fp_version': v245.v291(), 'reference_280': {'teacher_ceiling': 0.36666666666666653, 'policy_q8': 0.4666666666666666}, 'note': 'One mind over the 280 tape, with three things the policy could not express. Silence is typed - STOP_CONFLICT where support is equal, STOP_UNKNOWN where nothing was found - so abstention is a diagnosis and G_silence_is_typed checks the diagnosis against the situation rather than counting refusals. The query is an editable set of words, since 261 established the query must be words, so DROP_i and ADD_i make retrieval part of the policy instead of a fixed function applied to it; a typed silence is what tells the two apart, UNKNOWN meaning reformulate and CONFLICT meaning stop. And ASK_VALUE_i verifies by the return path: ask the tape about the value before answering with it, as a probe that restores the candidate list, so a plausible-but-wrong value with no way back is caught by something other than a vote count. The teacher demonstrates all three and reads no label. --no-edit and --no-probe are the ablations.', 'timestamp': v369.v356(v370.v357).v292(), 'wall_s': v182.v182() - v72}
    v0.v113(parents=True, exist_ok=True)
    (v0 / f'stage282_decision{v70}.json').v179(v333.v293(v110, indent=2), encoding='utf-8')
    v189(v333.v293({'overall': v225, 'gates': v110['gates']}, indent=2))
    return 0
if v111 == '__main__':
    raise v230(v294())