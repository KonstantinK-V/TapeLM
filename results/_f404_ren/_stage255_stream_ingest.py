"""
Stage 255 тАФ Stream-ingest engine: chunked training with domain switching, bounded RAM.

North star: never hold the corpus in memory, never re-read it, switch domains mid-stream,
run for days on a small GPU, resume after a kill.

Loop per chunk:
    lines = reader.next_chunk()            # only this chunk is resident
    tape.append(gate(entities(lines)))     # knowledge grows, weights untouched
    trunk = joint_train(trunk, lines + reservoir_replay)   # CE + lam*CPC, ~1 epoch
    del lines                              # dropped; reservoir keeps a bounded sample

Domain switch is declared by --schedule (dom:chunks,...). Keys are always written with the
FROZEN canonical arc_enc. Queries use a trainable QueryAdapter (W_q) so understanding can
align reads to a growing bank without rewriting slot keys.

Probes (vs consumed tokens, not steps): exam next_tok, held-out CE for every domain seen
(carved once, never trained), probe-fact recall as the bank fills, uniformity, throughput.

  python _stage255_stream_ingest.py --smoke
  python _stage255_stream_ingest.py --schedule wiki:12 --chunk-lines 25000 --run-tag wiki12
  python _stage255_stream_ingest.py --schedule wiki:12 --lambda-admit --lambda-admit-alpha 0.015 --run-tag wiki12_lam
"""
from __future__ import annotations
import argparse
import json
import math
import os
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
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import slot_query_words
from _tapelm_ext import DomainAdapter
v0 = v16('results')
v1 = v0 / 'stream255'
v2 = v0 / 'stage255_decision.json'
v3 = v0 / 'stage255_mini.md'
v4 = v0 / '_stage255_log.txt'
v5 = v16('checkpoints/stage191_p1_curve.pt')
v6 = 255
v7 = v6 + 9000
v8 = 0.2

def effective_lambda(v17: v9, v18: v15, v19: v9, v20: v30, v21: v15) -> v9:
    """Lower CPC when ingest load is high; n_admitted normalized by entity_cap so lambda tracks fill rate."""
    if not v20 or v18 <= 0 or v21 <= 0:
        return v17
    v22 = v9(v18) / v9(v21)
    return v17 / (1.0 + v19 * v22)
v10 = {'wiki': v16('data/_wikitext103_train.txt'), 'med': v16('data/_stage254_med.txt'), 'news': v16('data/_stage254_news.txt'), 'stories': v16('data/_tinystories_raw_100k.txt')}
v11 = 40

def log(v23: v129) -> None:
    v24 = v23 if v23.v260('\n') else v23 + '\n'
    try:
        v261(v24, end='', flush=True)
    except v130:
        v261(v24.v404('ascii', 'replace').v378('ascii'), end='', flush=True)
    v4.v262.v131(parents=True, exist_ok=True)
    with v4.v263('a', encoding='utf-8') as v132:
        v132.v264(v24)

def safe_torch_save(v25, v26: v16) -> None:
    v26.v262.v131(parents=True, exist_ok=True)
    v27 = v26.v133(v26.v265 + '.tmp')
    v149.v134(v25, v27)
    v266.v135(v27, v26)

class SegmentReader:
    """Walks a domain schedule, yielding chunks of lines; only the file handle is resident."""

    def __init__(v136, v72: v64[v34[v129, v15]], v73: v15, v137: v14 | None=None):
        v136.v72 = v72
        v136.v73 = v73
        v136.v137 = v137 or {}
        v136.v138 = 0
        v136.v139 = 0

    def state(v136) -> v14:
        return {'positions': v136.v137, 'seg_i': v136.v138, 'chunk_in_seg': v136.v139}

    def load_state(v136, v140: v14) -> None:
        v136.v137 = v140.v251('positions', {})
        v136.v138 = v140.v251('seg_i', 0)
        v136.v139 = v140.v251('chunk_in_seg', 0)

    def next_chunk(v136) -> v34[v129, v64[v129]] | None:
        while v136.v138 < v268(v136.v72):
            v43, v337 = v136.v72[v136.v138]
            if v136.v139 >= v337:
                v136.v138 += 1
                v136.v139 = 0
                continue
            v26 = v10[v43]
            v267 = v136.v137.v251(v43, 0)
            v31: v64[v129] = []
            with v26.v263('r', encoding='utf-8', errors='ignore') as v132:
                v132.v379(v267)
                while v268(v31) < v136.v73:
                    v380 = v132.v395()
                    if not v380:
                        break
                    v61 = v380.v295()
                    if v268(v61) >= v11:
                        v31.v269(v61)
                v136.v137[v43] = v132.v381()
            if not v31:
                v136.v137[v43] = 0
                v136.v138 += 1
                v136.v139 = 0
                continue
            v136.v139 += 1
            return (v43, v31)
        return None

class Tape:
    """Append-only canonical slot bank. fp16 on CPU; matmul in chunks so VRAM stays flat."""

    def __init__(v136, v141: v15, v70):
        v136.v141 = v141
        v136.v70 = v70
        v136.v142: v64[v149.v29] = []
        v136.v143: v149.v29 | None = None
        v136.v144: v64[v129] = []
        v136.v145: v64[v14] = []
        v136.v146: v177[v129] = v177()
        v136.v147: v64[v64[v129]] = []
        v136.v148 = None

    def _sync_postings(v136) -> None:
        from _inprint_glue import SlotPostings
        if v268(v136.v147) == v268(v136.v144) and v136.v147:
            v136.v148 = v382.v338(v136.v147, v149.v70('cpu'))
        else:
            v136.v148 = None

    @v28
    def postings(v136):
        return v136.v148

    def __len__(v136) -> v15:
        return v268(v136.v144)

    @v28
    def K(v136) -> v149.v29:
        if v136.v143 is None:
            v136.v143 = v149.v341(v136.v142, 0) if v136.v142 else v149.v340(0, v136.v141, dtype=v149.v273)
        return v136.v143

    def has_value(v136, v150: v129) -> v30:
        return v150 in v136.v146

    def append(v136, v151: v149.v29, v144: v64[v129], v145: v64[v14], v152: v64[v64[v129]] | None=None) -> v15:
        if not v144:
            return 0
        v136.v142.v269(v151.v383().v212('cpu', v149.v273))
        v136.v143 = None
        v136.v144.v270(v144)
        v136.v145.v270(v145)
        v136.v146.v271(v144)
        if v152 is not None:
            v136.v147.v270(v152)
        v136.v272()
        return v268(v144)

    def scores(v136, v153: v149.v29, v154: v15=200000) -> v149.v29:
        v155 = v136.v155
        if v155.v339() == 0:
            return v149.v340(0)
        v156 = v153.v383().v212('cpu', v149.v273)
        v62 = []
        for v51 in v185(0, v268(v136.v144), v154):
            v62.v269((v155[v51:v51 + v154] @ v156).v9())
        return v149.v341(v62) if v62 else v149.v340(0)

    def max_score_for(v136, v153: v149.v29, v150: v129) -> v9:
        v157 = [v274 for v274, v310 in v188(v136.v144) if v310 == v150]
        if not v157:
            return -1.0
        return v9(v136.v405(v153)[v157].v333())

    def nbytes(v136) -> v15:
        return v136.v155.v339() * 2

    def save(v136, v26: v16) -> None:
        v149.v134({'K': v136.v155, 'values': v136.v144, 'meta': v136.v145}, v26)

    def load(v136, v26: v16) -> None:
        v158 = v149.v22(v26, map_location='cpu', weights_only=False)
        v136.v142 = [v158['K']] if v158['K'].v339() else []
        v136.v143 = None
        v136.v144, v136.v145 = (v158['values'], v158['meta'])
        v136.v146 = v177(v136.v144)

class Reservoir:
    """Uniform sample of documents seen so far; bounded RAM, survives chunk deletion."""

    def __init__(v136, v40: v15, v159: v15):
        v136.v40 = v40
        v136.v89: v64[v278.v165] = []
        v136.v160 = 0
        v136.v55 = v281.v173(v159)

    def offer(v136, v161: v278.v165) -> None:
        v136.v160 += 1
        if v268(v136.v89) < v136.v40:
            v136.v89.v269(v161)
        else:
            v274 = v136.v55.v342(v136.v160)
            if v274 < v136.v40:
                v136.v89[v274] = v161

    def as_flat(v136) -> v34[v278.v165, v278.v165] | None:
        if not v136.v89:
            return None
        v162 = [0]
        for v158 in v136.v89:
            v162.v269(v162[-1] + v268(v158))
        return (v278.v279(v136.v89), v278.v280(v162, dtype=v278.v344))

    def save(v136, v26: v16) -> None:
        v278.v275(v26, n_seen=v136.v160, cap=v136.v40, items=v278.v384(v136.v89, dtype=v305))

    def load(v136, v26: v16) -> None:
        v158 = v278.v22(v26, allow_pickle=True)
        v136.v160 = v15(v158['n_seen'])
        v136.v40 = v15(v158['cap'])
        v136.v89 = v64(v158['items'])

def chunk_to_flat(v31: v64[v129], v32: v163, v33: v15) -> v34[v278.v165, v278.v165]:
    return v276.v164('\n'.v277(v31), v32, v33, max_lines=v268(v31) + 8, min_line_len=20)

def merge_flats(v35, v36):
    """Concatenate two (flat, off) corpora into one doc-id space."""
    v166, v167 = v35
    v168, v169 = v36
    v37 = v64(v167) + [v15(v167[-1]) + v15(v343) for v343 in v169[1:]]
    return (v278.v279([v166, v168]), v278.v280(v37, dtype=v278.v344))
v12 = v170.v38('\\b([A-Z][a-z]{2,})\\b')
v13 = 40000

def ingest_entities(v39: v171, v31: v64[v129], v40: v15, v41: v9, v42: v172, v43: v129, v44: v15, v45: v281.v173):
    """Real entities from the chunk -> canonical keys, novelty-gated so the tape does not bloat.

    Keys must use the SAME convention as probe facts (subject anchor + context). A context-only
    key is a generic direction that outscores anchored keys for any query and blinds the bank.
    """
    v151, v174, v145, v175, v176 = ([], [], [], [], [])
    v46 = v177()
    v47 = v64(v185(v268(v31)))
    v45.v178(v47)
    for v48 in v47:
        v179 = v31[v48]
        for v23 in v345.v282(v179):
            v283 = v23.v346(1)
            if v268(v283) < 5 or v283 in v46:
                continue
            v347, v348 = (v333(0, v23.v406() - 120), v332(v268(v179), v23.v407() + 120))
            v192 = v39.v291(v179[v347:v348], exclude=v283)
            if v192 is None:
                continue
            v284 = [v290 for v290 in v12.v396(v179[v347:v23.v406()]) if v290 != v283]
            if not v284:
                v284 = [v290 for v290 in v12.v396(v179[v347:v348]) if v290 != v283]
            if not v284:
                continue
            v285 = v284[-1]
            v46.v349(v283)
            v286 = v39.v350([v285])[0]
            v151.v269(v397.v385(v286 + v192, dim=-1))
            v287 = v39.v291(v179[v347:v23.v406()])
            v175.v269(v397.v385(v286 + v287, dim=-1) if v287 is not None else None)
            v176.v269(v351(v179[v347:v348], exclude=v283))
            v174.v269(v283)
            v145.v269({'domain': v43, 'chunk': v44, 'kind': 'entity', 'anchor': v285})
            if v268(v151) >= v40:
                break
        if v268(v151) >= v40:
            break
    if not v151:
        return (0, 0, [])
    v49 = v149.v180(v151, 0)
    v52, v181, v182, v183, v184 = ([], [], [], [], [])
    v50 = 0
    for v51 in v185(v268(v174)):
        if v42.v288(v174[v51]):
            v50 += 1
            continue
        if v52:
            v289 = v9((v149.v180(v52, 0) @ v49[v51]).v333())
            if v289 > v41:
                v50 += 1
                continue
        v52.v269(v49[v51])
        v181.v269(v174[v51])
        v182.v269(v145[v51])
        v183.v269(v176[v51])
        if v175[v51] is not None:
            v184.v269({'q': v175[v51].v383().v212('cpu', v149.v273), 'value': v174[v51]})
    if v52:
        v42.v269(v149.v180(v52, 0), v181, v182, ctxw=v183)
    return (v268(v181), v50, v184)

def make_probe_facts(v39: v171, v53: v64[v129], v54: v15, v43: v129, v55: v281.v173):
    """Controlled facts written to the tape only. Half fit W_q, half are held out for recall."""
    v56 = [v290 for v290 in v386(v177(v53), v55, v54 + 20) if v268(v290) >= 5][:v54]
    v58, v151, v186 = ([], [], [])
    for v51, v187 in v188(v56):
        v189 = v53[v55.v342(v268(v53))]
        v190 = f'{v187} was appointed director of {v189} in the {v43} chronicle of 1987 .'
        v132 = {'S': v187, 'value': v189, 'sent': v190, 'domain': v43, 'fid': f'{v43}_probe_{v51}', 'wq_train': v51 % 2 == 0}
        v191 = v39.v350([v187])[0]
        v192 = v39.v291(v190, exclude=v189)
        v151.v269(v397.v385(v191 + v192, dim=-1) if v192 is not None else v191)
        v186.v269(v351(v190, exclude=v189))
        v58.v269(v132)
    return (v58, v149.v180(v151, 0) if v151 else v149.v340(0, 256), v186)

def probe_bank_metrics(v57, v42: v172, v58, v59, v60=None) -> v14:
    return v292.v193(v58, v59, v57, v42.v155, v42.v144, v7, W_bwd=v60, postings=v42.v293)

def parse_schedule(v61: v129) -> v64[v34[v129, v15]]:
    v62 = []
    for v63 in v61.v194(','):
        v43, v206, v54 = v63.v294(':')
        v43 = v43.v295()
        if v43 not in v10:
            raise v259(f'unknown domain {v43}; known: {v64(v10)}')
        v62.v269((v43, v15(v54 or 1)))
    return v62

def main() -> v15:
    v65 = v296.v195()
    v65.v196('--smoke', action='store_true')
    v65.v196('--schedule', type=v129, default='wiki:6,med:4,news:4')
    v65.v196('--chunk-lines', type=v15, default=0)
    v65.v196('--epochs-per-chunk', type=v9, default=1.0)
    v65.v196('--reservoir', type=v15, default=0)
    v65.v196('--replay-frac', type=v9, default=0.2)
    v65.v196('--arc', choices=['frozen', 'adapt'], default='frozen')
    v65.v196('--entity-cap', type=v15, default=0, help='entities ingested per chunk')
    v65.v196('--novelty', type=v9, default=0.97)
    v65.v196('--resume', action='store_true')
    v65.v196('--ckpt-every', type=v15, default=2)
    v65.v196('--lambda-base', type=v9, default=v8, help='CPC weight when dynamic=off or n_admit=0')
    v65.v196('--lambda-admit', action='store_true', help='lambda_eff = lambda_base / (1 + alpha * n_admitted_this_chunk)')
    v65.v196('--lambda-admit-alpha', type=v9, default=0.35, help='at full entity_cap load: lambda*=1/(1+alpha)')
    v65.v196('--run-tag', type=v129, default='', help='subdir under results/stream255/ for ckpts')
    v65.v196('--no-query-train', action='store_true', help='disable W_q contrastive training (ablation)')
    v65.v196('--query-steps', type=v15, default=0, help='W_q steps per chunk (0=auto)')
    v66 = v65.v197()
    v67 = v66.v198 or ('smoke' if v66.v203 else 'default')
    global RUN
    v1 = v0 / 'stream255' / v67
    v1.v131(parents=True, exist_ok=True)
    v68 = v66.v69
    if not v66.v199:
        v4.v258('', encoding='utf-8')
    v70 = v149.v70('cuda' if v149.v387.v352() else 'cpu')
    v55 = v281.v173(v6)
    v149.v200(v6)
    v71 = v201.v201()
    v72 = v202('wiki:2,med:2' if v66.v203 else v66.v72)
    v73 = v66.v73 or (400 if v66.v203 else 25000)
    v74 = v66.v90 or (200 if v66.v203 else 4000)
    v21 = v66.v21 or (40 if v66.v203 else 400)
    v75 = 12 if v66.v203 else 32
    v76 = 40 if v66.v203 else 120
    v77 = 4 if v66.v203 else 12
    v78 = 30 if v66.v203 else 250
    v79 = 30 if v66.v203 else 350
    v80 = 40 if v66.v203 else 200
    v81 = v66.v81 or (25 if v66.v203 else 150)
    v82 = not v66.v204
    v205(f'Stage255 stream start {v402.v393(v403.v394).v334()} device={v70} schedule={v72} chunk_lines={v73} arc={v66.v255} resume={v66.v199} run={v67} lambda_base={v68} lambda_admit={v66.v253} alpha={v66.v254} query_train={v82} query_steps={v81}')
    v206, v206, v207, v208 = v209()
    v32 = v163.v210(v129(v353.v297))
    v83 = v32.v211()
    v33 = v32.v298(v299) or 0
    v84 = v388.v354(v32, v207, v33, v83).v212(v70)
    v85 = v355(v208, v83).v212(v70)
    v85.v213(v149.v22(v5, map_location=v70, weights_only=False)['model'])
    v85.v214()
    for v86 in v85.v215():
        v86.v300(False)
    v39 = v171(v85, v207, v70)
    with v10['wiki'].v263('r', encoding='utf-8', errors='ignore') as v132:
        v216 = v132.v301(1000000 if v66.v203 else 4000000)
    v53 = v64(v14.v302((v23.v346(1) for v23 in v345.v282(v216) if v268(v23.v346(1)) >= 5)))
    v55.v178(v53)
    v87 = v64(v14.v302((v290 for v290 in v170.v396('[A-Za-z][a-z]{2,}', v216) if v268(v290) <= 14)))[:v80]
    v88 = v303.v217(v39, v87)
    v89 = v304.v218(v76)
    v42 = v172(256, v70)
    v90 = v219(v74, v6)
    v91 = v220(v72, v73)
    v92 = v85
    v93: v14[v129, v64] = {}
    v94: v14[v129, v64] = {}
    v95: v14[v129, v305] = {}
    v96: v14[v129, v171] = {}
    v97: v64[v14] = []
    v98 = 0
    v99 = 0
    v44 = 0
    v100: v14[v129, v9] = {}
    v101 = v304.v221(v85, v84, v33, v89, v70)
    v102 = v292.v306(v70) if v82 else None
    v103: v64[v14] = []
    v104 = v1 / 'state.json'
    if v66.v199 and v104.v307():
        v140 = v356.v308(v104.v357(encoding='utf-8'))
        v91.v309(v140['reader'])
        v98 = v140.v251('tokens_train', v140.v251('tokens_seen', 0))
        v99 = v140.v251('tokens_unique', 0)
        v44, v97 = (v140['chunk_i'], v140['history'])
        v93 = {v191: v310 for v191, v310 in v140['probe_facts'].v89()}
        v100 = v140.v251('baseline_hold_ce', {})
        v101 = v140.v251('baseline_exam', v101)
        v42.v22(v1 / 'tape.pt')
        v222 = v1 / 'reservoir.npz'
        if v222.v307():
            v90.v22(v222)
        v92 = v355(v208, v83).v212(v70)
        v92.v213(v149.v22(v1 / 'trunk.pt', map_location=v70, weights_only=False)['model'])
        v92.v214()
        for v86 in v92.v215():
            v86.v300(False)
        v223 = v149.v22(v1 / 'holdouts.pt', map_location='cpu', weights_only=False)
        v94 = {v191: [v358 for v358 in v310] for v191, v310 in v223.v89()}
        v224 = v1 / 'w_era.pt'
        if v224.v307() and v66.v255 == 'adapt':
            v311 = v149.v22(v224, map_location=v70, weights_only=False)
            for v43, v359 in v311.v89():
                v60 = v408(256).v212(v70)
                v60.v213(v359)
                v95[v43] = v60
                v360 = v1 / f'shift_{v43}.pt'
                if v360.v307():
                    v389 = v355(v208, v83).v212(v70)
                    v389.v213(v149.v22(v360, map_location=v70, weights_only=False)['model'])
                    v389.v214()
                    for v86 in v389.v215():
                        v86.v300(False)
                    v96[v43] = v171(v389, v207, v70)
        v225 = v1 / 'query_adapter.pt'
        if v102 is not None and v225.v307():
            v312 = v149.v22(v225, map_location=v70, weights_only=False)
            v102.v213(v312['W_q_stream'] if v409(v312, v14) and 'W_q_stream' in v312 else v312)
        v226 = v1 / 'qpairs.pt'
        if v226.v307():
            v158 = v149.v22(v226, map_location='cpu', weights_only=False)
            v103 = [{'q': v153, 'value': v310} for v153, v310 in v398(v158['q'], v158['values'])]
        v205(f'resumed at chunk {v44}, unique_tok={v99}, train_tok={v98}, tape {v268(v42)}')
    v105 = v1 / 'STOP'
    while True:
        if v105.v307():
            v205('STOP file present тАФ halting cleanly')
            break
        v227 = v91.v313()
        if v227 is None:
            v205('schedule exhausted')
            break
        v43, v31 = v227
        v44 += 1
        v238, v239 = v314(v31, v32, v33)
        v228 = v268(v239) - 1
        if v228 < 8:
            v205(f'chunk {v44} ({v43}) too small ({v228} docs) тАФ skipped')
            continue
        v229 = v43 not in v94
        if v229:
            v315 = v333(2, v15(v228 * 0.05))
            v223 = []
            for v158 in v185(v228 - v315, v228):
                v361 = v238[v239[v158]:v239[v158 + 1]][:v399]
                v247 = v278.v390((1, v399), v33, v278.v344)
                v247[0, :v268(v361)] = v361
                v223.v269(v149.v400(v247))
            v94[v43] = v223[:v77] if v268(v223) >= v77 else v223
            v100[v43] = v371.v362(v85, v94[v43], v84, v33, v70)
            v316 = v64(v185(0, v228 - v315))
            v317, v363, v364 = v365(v39, v53, v75, v43, v55)
            v93[v43] = v317
            v42.v269(v363, [v132['value'] for v132 in v317], [{'domain': v43, 'kind': 'probe'} for v206 in v317], ctxw=v364)
            v205(f'  [{v43}] holdout={v268(v94[v43])} probe_facts={v268(v317)} P1_hold_ce={v100[v43]:.3f}')
        else:
            v316 = v64(v185(0, v228))
        v230 = v281.v173(v6 + v44 * 7919)
        v318, v319, v320 = v321(v39, v31, v21, v66.v41, v42, v43, v44, v230)
        v103.v270(v320)
        if v268(v103) > v13:
            v103 = v281.v173(v6 + v44).v366(v103, v13)
        v231 = v318 + (v268(v93.v251(v43, [])) if v229 else 0)
        v232 = v322(v68, v231, v66.v254, v66.v253, v21)
        if v66.v255 == 'adapt' and v43 not in v95:
            v323 = v303.v367(v85, v238, v239, v84, v33, v70, v78, v6 + v44)
            v324 = v171(v323, v207, v70)
            v60, v368 = v303.v369(v408(256).v212(v70), v303.v217(v324, v87), v88, v55, v79, v70)
            v95[v43], v96[v43] = (v60, v324)
            v149.v134({'model': v323.v401()}, v1 / f'shift_{v43}.pt')
            v205(f'  [{v43}] W era align={v368:.3f}')
        v240, v241 = (v238, v239)
        v233 = v90.v325()
        if v233 is not None and v66.v257 > 0:
            v240, v241 = v370((v238, v239), v233)
            v326 = v268(v233[1]) - 1
            v327 = v332(v326, v15(v268(v316) * v66.v257 / v333(1e-06, 1 - v66.v257)))
            v328 = v64(v185(v228, v228 + v326))
            v316 = v316 + v281.v173(v6 + v44).v366(v328, v327)
        v234 = v15(v66.v256 * v15(v239[-1]))
        v235 = [v36 for v158 in v94 for v36 in v94[v158]]
        v92, v145 = v371.v329(v92, v240, v241, v84, v33, v70, v234, v232, v6 + 1000 + v44, f'c{v44}:{v43}', v316, v235, v89[:24], early_stop=False, n_probes=2)
        v99 += v15(v239[-1])
        v98 += v145['tokens_ce']
        v236 = v9('nan')
        if v102 is not None and v103:
            v236 = v292.v372(v102, v103, v42.v155, v42.v144, v70, v81, v6 + v44)
        v237 = v64(v185(v228))
        if v268(v237) > 800:
            v330 = v333(1, v268(v237) // 800)
            v237 = v237[::v330]
        for v158 in v237:
            v90.v373(v238[v239[v158]:v239[v158 + 1]].v391())
        del v31, v238, v239, v240, v241
        v242 = v64(v14.v302([v132['value'] for v331 in v93.v144() for v132 in v331] + v42.v144))
        v243 = v102 if v102 is not None else v95.v251(v43)
        v244 = {v158: v331 for v158, v331 in v93.v89() if v331}
        v245 = {v158: v374(v96.v251(v158, v39), v42, v244[v158], v242, W=None) for v158 in v244}
        v246 = {v158: v374(v96.v251(v158, v39), v42, v244[v158], v242, W=v243) for v158 in v244}
        v247 = {'chunk': v44, 'domain': v43, 'tokens_unique': v99, 'tokens_train': v98, 'tape_slots': v268(v42), 'tape_mb': v42.v392() / 1000000.0, 'entities_added': v318, 'entities_dropped': v319, 'n_admit_lambda': v231, 'lambda_eff': v232, 'query_loss': v236, 'exam_next_tok': v304.v221(v92, v84, v33, v89, v70), 'hold_ce': {v158: v371.v362(v92, v94[v158], v84, v33, v70) for v158 in v94}, 'probe_bank_frozen': v245, 'probe_bank': v246, 'wall_s': v201.v201() - v71}
        v97.v269(v247)
        v248 = ' '.v277((f"{v158}:ce={v247['hold_ce'][v158]:.2f}" + (f"/top1={v246[v158]['top1']:.2f}(f={v245[v158]['top1']:.2f})/mrr={v246[v158]['mrr']:.2f}" if v158 in v246 else '') for v158 in v94))
        v205(f"  chunk {v44} [{v43}] uniq_tok={v99} train_tok={v98} slots={v268(v42)} (+{v318}/-{v319}) lam={v232:.4f} (n_admit={v231}) q_loss={v236:.3f} exam={v247['exam_next_tok']:.3f} {v248} tape={v247['tape_mb']:.1f}MB ({v247['wall_s']:.0f}s)")
        if v44 % v66.v375 == 0:
            v376({'model': v92.v401()}, v1 / 'trunk.pt')
            v42.v134(v1 / 'tape.pt')
            v90.v134(v1 / 'reservoir.npz')
            v376({v191: v310 for v191, v310 in v94.v89()}, v1 / 'holdouts.pt')
            if v95:
                v376({v191: v310.v401() for v191, v310 in v95.v89()}, v1 / 'w_era.pt')
            if v102 is not None:
                v376({'W_q_stream': v102.v401(), 'W_query': v102.v401()}, v1 / 'query_adapter.pt')
            if v103:
                v376({'q': v149.v180([v86['q'] for v86 in v103]), 'values': [v86['value'] for v86 in v103]}, v1 / 'qpairs.pt')
            v104.v258(v356.v335({'reader': v91.v410(), 'tokens_unique': v99, 'tokens_train': v98, 'chunk_i': v44, 'history': v97, 'probe_facts': v93, 'baseline_hold_ce': v100, 'baseline_exam': v101}, indent=2), encoding='utf-8')
    if not v97:
        v205('no chunks processed')
        return 1
    v249, v250 = (v97[0], v97[-1])
    v106 = v64(v250['hold_ce'].v151())
    v107 = {}
    for v108 in v97:
        for v158, v310 in v108['hold_ce'].v89():
            v107.v377(v158, v310)
    v109 = {v158: v250['hold_ce'][v158] - v107[v158] for v158 in v106}
    v110 = {v158: v250['hold_ce'][v158] - v100.v251(v158, v107[v158]) for v158 in v106}
    v111 = [(v108['tape_slots'], v332((v23['top1'] for v23 in v108['probe_bank'].v144()))) for v108 in v97 if v108.v251('probe_bank')]
    v112 = [(v108['tape_slots'], v332((v23['top1'] for v23 in v108.v251('probe_bank_frozen', {}).v144()))) for v108 in v97 if v108.v251('probe_bank_frozen')]
    v113 = [(v108['tape_slots'], v332((v23['mrr'] for v23 in v108['probe_bank'].v144()))) for v108 in v97 if v108.v251('probe_bank')]
    v114 = v250.v251('probe_bank', {})
    v115 = v250.v251('probe_bank_frozen', {})
    v116 = v332((v310['top1'] for v310 in v114.v144())) if v114 else 0.0
    v117 = v332((v310['mrr'] for v310 in v114.v144())) if v114 else 0.0
    v118 = v332((v310['top1'] for v310 in v115.v144())) if v115 else v116
    v119 = v268(v97) >= 2
    v120 = v333(v110.v144()) <= 0.15
    v121 = v333(v109.v144()) <= 0.15
    v122 = v250['exam_next_tok'] >= v101 - 0.01
    v123 = v116 >= v118 + 0.02 or v117 >= 0.1
    v124 = v116 >= 0.1 if v82 else v118 >= 0.04
    v125 = v250['tape_mb'] < 2000
    if v119 and v120 and v122 and v123 and v124:
        v252 = 'STREAM_INGEST_OK'
    elif v119 and v120 and v122 and (v123 or v124):
        v252 = 'STREAM_INGEST_PARTIAL'
    else:
        v252 = 'STREAM_INGEST_NO'
    v62 = {'stage': 255, 'overall': v252, 'lambda_base': v68, 'lambda_admit': v66.v253, 'lambda_admit_alpha': v66.v254, 'run_tag': v67, 'chunk_lines': v73, 'entity_cap': v21, 'query_train': v82, 'query_steps': v81, 'arc_mode': v66.v255, 'epochs_per_chunk': v66.v256, 'replay_frac': v66.v257, 'reservoir_cap': v74, 'gates': {'G_streamed': v119, 'G_no_forget_vs_P1': v120, 'G_peak_hold_regress': v121, 'G_understanding_holds': v122, 'G_recall_query_beats_frozen': v123, 'G_recall_adapt_top1_floor': v124, 'G_tape_bounded': v125}, 'summary': {'chunks': v268(v97), 'tokens_unique': v250.v251('tokens_unique', 0), 'tokens_train': v250.v251('tokens_train', 0), 'tape_slots': v250['tape_slots'], 'tape_mb': v250['tape_mb'], 'baseline_exam': v101, 'exam_first_chunk': v97[0]['exam_next_tok'], 'exam_last': v250['exam_next_tok'], 'forget_hold_ce_vs_first_chunk': v109, 'forget_hold_ce_vs_P1': v110, 'baseline_hold_ce': v100, 'recall_top1_vs_bank': v111, 'recall_top1_frozen_vs_bank': v112, 'recall_mrr_vs_bank': v113, 'recall_final_top1': v116, 'recall_final_top1_frozen': v118, 'recall_final_mrr': v117}, 'history': v97, 'note': 'Canonical frozen keys on tape; trainable W_q (QueryAdapter) aligns queries to keys — understanding→read, not re-indexing. W_q trains on ingested-entity contrastive pairs per chunk; recall gates score held-out probe facts only. Hold CE vs P1 is primary no-forget gate. Recall gates use W_q-adapted top1/MRR, not frozen-query-only.', 'timestamp': v402.v393(v403.v394).v334(), 'wall_s': v201.v201() - v71}
    v126 = v0 / (f'stage255_decision_{v67}.json' if v67 != 'default' else 'stage255_decision.json')
    v127 = v0 / (f'stage255_mini_{v67}.md' if v67 != 'default' else 'stage255_mini.md')
    v126.v258(v356.v335(v62, indent=2), encoding='utf-8')
    v127.v258(f"# Stage 255 stream ingest\n\n**{v252}** chunks={v268(v97)} uniq_tok={v250.v251('tokens_unique', 0)} train_tok={v250.v251('tokens_train', 0)} slots={v250['tape_slots']} ({v250['tape_mb']:.1f} MB)\n\n- exam P1={v101:.3f} last={v250['exam_next_tok']:.3f}\n- forget vs P1: {v356.v335({v191: v411(v310, 3) for v191, v310 in v110.v89()})}\n- recall top1 vs bank: {v111}\n- recall mrr vs bank: {v113}\n", encoding='utf-8')
    v205(v356.v335({'overall': v252, 'chunks': v268(v97), 'slots': v250['tape_slots']}, indent=2))
    return 0
if v128 == '__main__':
    raise v259(v336())