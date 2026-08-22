"""
Stage 269 — 268's claim, measured on an exam that has room to fail.

268 came back MIND_LEARNS_TAPE_PARTIAL with EM 1.000 on the last training tape and 1.000 on a
tape never seen. G_novel_tape was true, but it compared two ceilings: retrieval on the planted
exam is saturated (256/263 spent months on exactly this), span-lock then makes EM follow
retrieval, and a gate that cannot fail proves nothing. 269 keeps 268's training verbatim and
changes only what is measured.

Every tape now carries a second population of slots that is never trained on:

    planted facts   cue template + value, half fit / half held out   -> TRAINS the procedure
    open entities   key written from real sentence A, question is
                    the prefix of a different real sentence B        -> SCORES it
    distractors     wiki noise

The open half is 261/264/267's exam, where zero-train word votes reach top1 0.246 on 4352 slots.
There is headroom by construction, and `G_headroom` asserts it rather than assuming it: if the
trained query lands above 0.90 on the training tape, the verdict is NOVEL_TAPE_SATURATED and the
run says so instead of claiming transfer.

Two comparisons matter, and 268 had neither:

  G_novel_tape   the same procedure on a tape whose open entities were never used in any rebuild.
                 Tapes are rebuilt every ~200 steps, so nothing factual survives; if this holds,
                 what transferred is procedure.

  G_beats_votes  the trained query against zero-train postings on the identical items. 266 showed
                 a learned query vector losing to plain words 0.062 vs 0.199. If the unfrozen mind
                 still loses, "the mind learned to use the tape" is not the right sentence.

268's G_beats_frozen_mind is dropped. It ran the trained glue against the untouched trunk and read
0.000 — but 265 got 0.975 from that same frozen trunk, so what it measured was the glue having
co-adapted to a drifting trunk, not the mind's contribution. The honest control is a paired run
with --frozen-baseline: identical budget, identical exam, set_train_mode("none"), its own glue.

  python _stage269_open_novel_tape.py --smoke
  python _stage269_open_novel_tape.py                     # night, upper trains
  python _stage269_open_novel_tape.py --frozen-baseline   # the paired control
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
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
from _stage261_nl_query import WORD_RE, collect, ctx_words, jaccard
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, SlotBias, TapeView, ctx_query
from _tape_index import context_words
v0 = v14('results')
v1 = v14('checkpoints/stage191_p1_curve.pt')
v2 = v14('checkpoints/stage253_joint_l02.pt')
v3 = v14('data/_wikitext103_train.txt')
v4 = 269
v5 = v6
v7 = v8
v9 = v15.v9

def paths(v16: v116):
    v17 = '_frozen' if v16 else ''
    return (v0 / f'stage269_decision{v17}.json', v0 / f'stage269_mini{v17}.md', v0 / f'_stage269_log{v17}.txt')
v10 = v0 / '_stage269_log.txt'

def log(v18: v11) -> None:
    v19 = v18 if v18.v210('\n') else v18 + '\n'
    try:
        v211(v19, end='', flush=True)
    except v117:
        v211(v19.v327('ascii', 'replace').v304('ascii'), end='', flush=True)
    v10.v212.v118(parents=True, exist_ok=True)
    with v10.v213('a', encoding='utf-8') as v43:
        v43.v214(v19)

def fp_version() -> v11:
    v20 = v119(v120, 'canonical_fp_version', None)
    if v121(v20):
        try:
            return v11(v20())
        except v215:
            pass
    return v1.v21

def arc_enc_hash(v22: v122) -> v11:
    v23 = v216.v123()
    for v124, v125 in v126(v22.v322.v305().v58()):
        v23.v217(v125.v340().v339().v335().v323().v282())
    return v23.v127()

def build_tape(*, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36) -> v12:
    """One tape. `used` carries across rebuilds so no planted value and no open entity repeats.

    268 silently reset its pool when it ran dry, which would have let values repeat between tapes
    and quietly broken the whole "nothing factual survives a rebuild" claim. Here exhaustion is an
    error, and the count that would have hidden it is reported.
    """
    v37 = [v128 for v128 in v29 if v128 not in v32 and v235(v128) >= 5]
    v28.v129(v37)
    v38 = [v128 for v128 in v283(v144(v32) | v144(v37), v28, v33 + v34 + 80) if v235(v128) >= 5 and v128 not in v32]
    v38 = v130(v12.v218(v38))
    v39 = [v128 for v128 in v283(v144(v32) | v144(v37) | v144(v38), v28, v34 + 40) if v235(v128) >= 6 and v128 not in v38 and (v128 not in v32)]
    v39 = v130(v12.v218(v39))[:v34]
    v40 = [v45 for v45 in v126(v31) if v45 not in v32]
    v28.v129(v40)
    if v235(v38) < v33 + v235(v39) or v235(v37) < v33 or v235(v40) < v35:
        raise v219(f'pool exhausted: subs={v235(v38)} avail={v235(v37)} open={v235(v40)} (need facts={v33} nonsense={v34} open={v35}) — raise the corpus, do not recycle: repeated values would break the novel-tape claim')
    v41 = []
    for v42 in v131(v33):
        v41.v220({'S': v38[v42], 'value': v37[v42], 'sent': v7.v306(S=v38[v42], V=v37[v42]), 'glue_train': v42 % 2 == 0, 'kind': 'wiki'})
        v32.v221(v37[v42])
        v32.v221(v38[v42])
    for v132, v133 in v134(v39):
        v135 = v38[v33 + v132]
        v41.v220({'S': v135, 'value': v133, 'sent': v7.v306(S=v135, V=v133), 'glue_train': False, 'kind': 'nonsense'})
        v32.v221(v133)
        v32.v221(v135)
    v136, v137, v138 = ([], [], [])
    for v43 in v41:
        v139 = v24.v284([v43['S']])[0]
        v140 = v24.v222(v43['sent'], exclude=v43['value'])
        v136.v220(v309.v285(v139 + v140, dim=-1) if v140 is not None else v139)
        v137.v220(v43['value'])
        v138.v220(v43['sent'])
    v44 = []
    for v45 in v40:
        if v235(v44) >= v35:
            break
        v141 = v31[v45]
        v223, v224 = (v141[0], v141[1])
        v142 = v223['line'][v277(0, v223['start'] - 140):v307(v235(v223['line']), v223['end'] + 140)]
        v143 = v224['line'][v277(0, v224['start'] - 200):v224['start']].v225()
        if v235(v324.v308(v143)) < 4:
            continue
        v140 = v24.v222(v142, exclude=v45)
        if v140 is None:
            continue
        v136.v220(v309.v285(v24.v284([v223['anchor']])[0] + v140, dim=-1))
        v44.v220({'ent': v45, 'slot': v235(v137), 'qtext': v143, 'wctx': v142, 'qwords': v233(v143, exclude=v45), 'overlap': v310(v325(v142, v45), v325(v143, v45))})
        v137.v220(v45)
        v138.v220(v142)
        v32.v221(v45)
    v46 = v144(v137)
    v145, v146 = ([], [])
    for v47 in v30:
        if v235(v137) >= v235(v41) + v235(v44) + v36:
            break
        for v18 in v286.v226(v47):
            v227 = v18.v287(1)
            if v235(v227) < 5 or v227 in v46:
                continue
            v288, v289 = (v277(0, v18.v332() - 120), v307(v235(v47), v18.v333() + 120))
            v140 = v24.v222(v47[v288:v289], exclude=v227)
            if v140 is None:
                continue
            v228 = [v128 for v128 in v334.v308(v47[v288:v18.v332()]) if v128 != v227]
            if not v228:
                continue
            v136.v220(v309.v285(v24.v284([v228[-1]])[0] + v140, dim=-1))
            v229 = v24.v222(v47[v288:v18.v332()])
            if v229 is not None:
                v145.v220(v309.v285(v24.v284([v228[-1]])[0] + v229, dim=-1))
                v146.v220(v235(v137))
            v137.v220(v227)
            v138.v220(v47[v288:v289])
            v46.v221(v227)
            if v235(v137) >= v235(v41) + v235(v44) + v36:
                break
    v48: v12[v11, v130[v13]] = v147(v130)
    for v148, (v230, v231) in v134(v232(v137, v138)):
        for v128 in v233(v231, exclude=v230):
            v48[v128].v220(v148)
    v49 = {v128: 1.0 / v311.v175(2.0 + v235(v48[v128])) for v128 in v48}
    return {'tape': v234(v156.v326(v136, 0).v181(v27), v137, v25, v26), 'fit_facts': [v43 for v43 in v41 if v43['glue_train']], 'eval_facts': [v43 for v43 in v41 if not v43['glue_train']], 'open_items': v44, 'postings': v48, 'idf': v49, 'nce_q': v156.v326(v145).v181(v27).v160() if v145 else None, 'nce_slot': v156.v290(v146, device=v27) if v146 else None, 'n_slots': v235(v137)}

def rank_stats(v50, v51, v52) -> v12:
    v53 = v236.v149(v50, dtype=v236.v237)
    v54 = v236.v149(v51)
    v55 = v53 == 1

    def _m(v150, v151):
        return v160(v151[v150].v243()) if v150.v291() else v160('nan')
    return {'top1': v160(v55.v243()), 'mrr': v160(v236.v243(1.0 / v53)), 'median_rank': v160(v236.v275(v53)), 'top1_low_overlap': v238(v54, v55), 'top1_high_overlap': v238(~v54, v55), 'n': v235(v50), 'n_slots': v52}

@v156.v61()
def score_trained_query(v56, v24, v25, v57, v58, v26, v59) -> v12:
    """Rank of the gold slot under the query the mind actually forms. No candidate pool."""
    v50, v51 = ([], [])
    for v60 in v58:
        v152 = [v42 for v42 in v25.v327(v60['qtext']).v152 if v42 != v26]
        v153 = v239(v56, v24, v25, v152)
        if v153 is None:
            v50.v220(v57.v240.v241(0))
            v51.v220(v60['overlap'] <= v59)
            continue
        v154 = v57.v240 @ v153
        v50.v220(1 + v13((v154 > v154[v60['slot']]).v185()))
        v51.v220(v60['overlap'] <= v59)
    return v155(v50, v51, v57.v240.v241(0))

def score_votes(v58, v48, v49, v52, v59) -> v12:
    """Zero-train postings on the identical items — the bar the mind has to clear (266)."""
    v50, v51, v157 = ([], [], [])
    for v60 in v58:
        v158: v12[v13, v160] = v147(v160)
        for v128 in v60['qwords']:
            for v148 in v48.v242(v128, ()):
                v158[v148] += v49.v242(v128, 0.0)
        v159 = v158.v242(v60['slot'], 0.0)
        v50.v220(v52 if v159 <= 0.0 else 1 + v185((1 for v230 in v158.v338() if v230 > v159)))
        v51.v220(v60['overlap'] <= v59)
        v157.v220(v159 <= 0.0)
    v62 = v155(v50, v51, v52)
    v62['tie_at_zero_frac'] = v160(v236.v243(v157))
    return v62

def main() -> v13:
    v63 = v244.v161()
    v63.v162('--smoke', action='store_true')
    v63.v162('--steps', type=v13, default=0)
    v63.v162('--tape-period', type=v13, default=0)
    v63.v162('--topk', type=v13, default=8)
    v63.v162('--gate-l1', type=v160, default=0.02)
    v63.v162('--nce-w', type=v160, default=1.0)
    v63.v162('--nce-tau', type=v160, default=0.05)
    v63.v162('--facts', type=v13, default=0)
    v63.v162('--nonsense-facts', type=v13, default=0)
    v63.v162('--open-items', type=v13, default=0, help='never-trained cross-mention slots')
    v63.v162('--distractor-slots', type=v13, default=0)
    v63.v162('--lr-glue', type=v160, default=0.003)
    v63.v162('--lr-upper', type=v160, default=3e-05)
    v63.v162('--frozen-baseline', action='store_true', help='paired control: identical budget, trunk NOT unfrozen, its own glue')
    v64 = v63.v163()
    global LOG_PATH
    v164, v165, v10 = v166(v64.v167)
    v10.v212.v118(parents=True, exist_ok=True)
    v10.v168('', encoding='utf-8')
    v27 = v156.v27('cuda' if v156.v312.v292() else 'cpu')
    v28 = v245.v169(v4)
    v156.v170(v4)
    v65 = v171.v171()
    v66 = v64.v66 or (400 if v64.v174 else 8000)
    v67 = v64.v67 or (100 if v64.v174 else 200)
    v33 = v64.v41 or (8 if v64.v174 else 48)
    v34 = v64.v172 or (4 if v64.v174 else 16)
    v35 = v64.v44 or (20 if v64.v174 else 120)
    v36 = v64.v173 or (150 if v64.v174 else 1200)
    v68 = 6 if v64.v174 else 12
    v69 = 4 if v64.v174 else 12
    v70 = 1500 if v64.v174 else 12000
    v71 = v64.v72
    v73 = 'none' if v64.v167 else 'upper'
    v175(f'Stage269 open-novel-tape start {v330.v320(v331.v321).v279()} device={v27} steps={v66} tape_period={v67} open={v35} dist={v36} trunk_mode={v73}')
    v124, v124, v176, v177 = v178()
    v25 = v246.v179(v11(v293.v247))
    v74 = v25.v180()
    v26 = v25.v248(v249) or 0
    v75 = v313.v294(v25, v176, v26, v74).v181(v27)
    v76 = v2 if v2.v250() else v1
    v22 = v122(v177, v74).v181(v27)
    v22.v182(v156.v295(v76, map_location=v27, weights_only=False)['model'])
    v251.v183(v22, v73)
    v77 = v184(v22)
    v78 = v185((v80.v296() for v80 in v22.v187() if v80.v259))
    v175(f'  trunk={v76.v21} mode={v73} trainable_trunk_params={v78} arc_enc hash0={v77[:16]}…')
    v79 = v122(v177, v74).v181(v27)
    v79.v182(v156.v295(v1, map_location=v27, weights_only=False)['model'])
    v79.v186()
    for v80 in v79.v187():
        v80.v252(False)
    v24 = v188(v79, v176, v27)
    v175(f'  fp_version={v278()}')
    with v3.v213('r', encoding='utf-8', errors='ignore') as v43:
        v189 = v43.v253(2000000 if v64.v174 else 16000000)
    v29 = v130(v12.v218((v18.v287(1) for v18 in v286.v226(v189) if v235(v18.v287(1)) >= 5)))
    v28.v129(v29)
    v30 = [v297.v225() for v297 in v189.v314('\n') if 60 <= v235(v297.v225()) <= 400][:v70]
    v31 = v190(v30, v24)
    v175(f'  entity pool={v235(v29)} lines={v235(v30)} multi_mention={v235(v31)} (need >= {v35 * (v66 // v67 + 2)} across rebuilds)')
    v81 = '\n'.v191(v30 + [v9] * 32)
    v192, v193 = v251.v194(v81, v25, v26, max_lines=v70 + 64, min_line_len=20)
    v82 = v235(v193) - 1
    v83 = v130(v131(v277(1, v82 - v277(2, v82 // 20)), v82))
    v84 = v130(v131(0, v83[0]))
    v85 = v254.v195(v192, v193, v83, v26, v69, v4 + 5)
    v86 = v254.v196(v22, v85, v75, v26, v27)
    v175(f'  hold CE base={v86:.4f}')
    v87 = 2 * (v22.v298.v255 // 2)
    v56 = v197(v87, v27)
    v88 = v156.v256.v198(v56.v257(), lr=v64.v258, weight_decay=0.01)
    v89 = [v80 for v80 in v22.v187() if v80.v259]
    v90 = v156.v256.v198(v89, lr=v64.v299, weight_decay=0.01) if v89 else None
    v32: v144[v11] = v144()
    v91 = None
    v92 = 0
    v93 = []
    for v94 in v131(1, v66 + 1):
        if v91 is None or (v94 - 1) % v67 == 0:
            v91 = v205(bank_can=v24, tok=v25, pad_id=v26, device=v27, rng=v28, values_pool=v29, lines=v30, cands=v31, used=v32, n_facts=v33, n_nonsense=v34, n_open=v35, n_dist=v36)
            v92 += 1
            v175(f"  tape#{v92} @step {v94}: slots={v91['n_slots']} fit={v235(v91['fit_facts'])} open={v235(v91['open_items'])} used={v235(v32)}")
        if v73 != 'none':
            v251.v183(v22, v73)
        v57 = v91['tape']
        v199 = [v91['fit_facts'][v28.v315(v235(v91['fit_facts']))] for v124 in v131(v307(4, v235(v91['fit_facts'])))]
        v260, v261 = v15.v262(v56, v22, v75, v25, v24, v57, v199, v26, v74, v27, v71, open_only=True)
        v152 = v328.v316(v192, v193, 1, v28, v26, v84).v181(v27)
        v263, v264 = v15.v265(v56, v22, v75, v25, v24, v57, v152, v26, v74, v27, v71, v64.v266)
        v200 = None
        if v91['nce_q'] is not None and v64.v300 > 0:
            v267 = v57.v240.v160()
            v268 = v156.v301(0, v91['nce_q'].v241(0), (v307(64, v91['nce_q'].v241(0)),), device=v27)
            v269 = v309.v329(v91['nce_slot'][v268], v267.v241(0)).v116()
            v200 = v64.v300 * v15.v317(v56, v91['nce_q'][v268], v269, v267, v64.v318)
        v201 = [v270 for v270 in (v260, v263, v200) if v270 is not None]
        if not v201:
            continue
        v202 = v201[0]
        for v80 in v201[1:]:
            v202 = v202 + v80
        v88.v271(set_to_none=True)
        if v90 is not None:
            v90.v271(set_to_none=True)
        v202.v272()
        v156.v319.v302.v273(v130(v56.v257()) + v89, 1.0)
        v88.v94()
        if v90 is not None:
            v90.v94()
        if v94 % v277(1, v66 // 10) == 0 or v94 == v66:
            v22.v186()
            v274 = v160(v236.v275([v60['overlap'] for v60 in v91['open_items']]))
            with v156.v61():
                v153 = v203(v56, v24, v25, v57, v91['open_items'], v26, v274)
            v93.v220({'step': v94, 'tape': v92, 'open_top1': v153['top1'], 'open_median_rank': v153['median_rank'], 'loss_fact': v160(v260) if v260 is not None else None, 'gate_fact': v261, 'gate_prose': v264})
            v175(f"  step {v94}/{v66} tape#{v92} open_top1={v153['top1']:.3f} median={v153['median_rank']:.0f} ({v171.v171() - v65:.0f}s)")
            if v73 != 'none':
                v251.v183(v22, v73)
    v56.v186()
    v22.v186()
    v95 = v184(v22)
    v96 = v77 == v95
    v97 = v160(v236.v275([v60['overlap'] for v60 in v91['open_items']]))
    v98 = v203(v56, v24, v25, v91['tape'], v91['open_items'], v26, v97)
    v99 = v204(v91['open_items'], v91['postings'], v91['idf'], v91['n_slots'], v97)
    v100 = v205(bank_can=v24, tok=v25, pad_id=v26, device=v27, rng=v245.v169(v4 + 99), values_pool=v29, lines=v30, cands=v31, used=v32, n_facts=v33, n_nonsense=v34, n_open=v35, n_dist=v36)
    v101 = v160(v236.v275([v60['overlap'] for v60 in v100['open_items']]))
    v102 = v203(v56, v24, v25, v100['tape'], v100['open_items'], v26, v101)
    v103 = v204(v100['open_items'], v100['postings'], v100['idf'], v100['n_slots'], v101)
    v175(f"  TRAIN tape open: query top1={v98['top1']:.3f} votes top1={v99['top1']:.3f}")
    v175(f"  NOVEL tape open: query top1={v102['top1']:.3f} votes top1={v103['top1']:.3f} median={v102['median_rank']:.0f} slots={v100['n_slots']}")
    v104 = v15.v206(v56, v22, v75, v25, v24, v100['tape'], v100['eval_facts'], v26, v74, v27, v71, v68, locked=True)
    v105 = v15.v206(v56, v22, v75, v25, v24, v100['tape'].v276(), v100['eval_facts'], v26, v74, v27, v71, v68, locked=True)
    v106 = v100['tape'].v207(v4 + 1)
    v107 = v203(v56, v24, v25, v106, v100['open_items'], v26, v101)
    v108 = v254.v196(v22, v85, v75, v26, v27)
    v175(f"  planted EM={v104['em']:.3f} empty={v105['em']:.3f} shuffled_open_top1={v107['top1']:.3f} hold {v86:.4f}->{v108:.4f}")
    v109 = v98['top1'] < 0.9
    v110 = v116(v109 and v102['top1'] >= v98['top1'] - 0.05)
    v111 = v116(v102['top1'] >= v103['top1'] + 0.03)
    v112 = v105['em'] <= 0.1
    v113 = v107['top1'] <= v277(0.05, v102['top1'] - 0.1)
    v114 = v108 <= v86 + 0.05
    if not v109:
        v208 = 'NOVEL_TAPE_SATURATED'
    elif v110 and v111 and v96 and v112 and v113:
        v208 = 'OPEN_NOVEL_TAPE_OK'
    elif v110 and v96 and v112 and v113:
        v208 = 'OPEN_NOVEL_TAPE_PARTIAL'
    else:
        v208 = 'OPEN_NOVEL_TAPE_NO'
    v62 = {'stage': 269, 'overall': v208, 'frozen_baseline': v64.v167, 'trunk_mode': v73, 'trainable_trunk_params': v78, 'smoke': v64.v174, 'seed': v4, 'trunk': v76.v21, 'fp_version': v278(), 'steps': v66, 'tape_period': v67, 'n_tapes': v92, 'n_open_per_tape': v35, 'distractor_slots': v36, 'used_pool_final': v235(v32), 'gates': {'G_headroom': v109, 'G_novel_tape': v110, 'G_beats_votes': v111, 'G_arc_enc_frozen': v96, 'G_no_param_leak': v112, 'G_tape_causal': v113, 'G_lang_intact': v114}, 'open_exam': {'train_tape_query': v98, 'train_tape_votes': v99, 'novel_tape_query': v102, 'novel_tape_votes': v103, 'shuffled_keys_query': v107, 'delta_novel_minus_train': v102['top1'] - v98['top1'], 'delta_query_minus_votes': v102['top1'] - v103['top1']}, 'planted_half': {'em': v104['em'], 'verbatim': v104['verbatim'], 'open_recall': v104['open_recall'], 'em_empty_tape': v105['em']}, 'controls': {'hold_ce_base': v86, 'hold_ce_after': v108, 'arc_enc_hash_before': v77, 'arc_enc_hash_after': v95}, 'curve': v93, 'note': "268 read EM 1.000 on both tapes — two ceilings, so its G_novel_tape could not fail. Here the scored half is 261/264/267's cross-mention exam, never trained on, and G_headroom asserts the room to fail instead of assuming it. G_beats_votes is the question 266 left open: does an unfrozen mind beat zero-train postings on the same items. 268's G_beats_frozen_mind is gone — it compared a co-adapted glue against an untouched trunk and read 0.000 where 265 got 0.975; the paired --frozen-baseline run is the honest control. Tapes rebuild every tape_period steps and the pool never recycles, so nothing factual survives a rebuild.", 'timestamp': v330.v320(v331.v321).v279(), 'wall_s': v171.v171() - v65}
    v0.v118(parents=True, exist_ok=True)
    v164.v168(v303.v280(v62, indent=2), encoding='utf-8')
    v165.v168(f"# Stage 269 open novel tape{(' (frozen baseline)' if v64.v167 else '')}\n\n**{v208}** · mode={v73} · tapes={v92} · open={v35}/tape · slots≈{v100['n_slots']}{(' · SMOKE' if v64.v174 else '')}\n\n| exam (open half) | top1 | median rank |\n|---|---:|---:|\n| train tape, trained query | {v98['top1']:.3f} | {v98['median_rank']:.0f} |\n| **novel tape, trained query** | **{v102['top1']:.3f}** | {v102['median_rank']:.0f} |\n| novel tape, zero-train votes | {v103['top1']:.3f} | {v103['median_rank']:.0f} |\n| novel tape, shuffled keys | {v107['top1']:.3f} | {v107['median_rank']:.0f} |\n\n## Gates (read G_headroom first)\n\n" + ''.v191((f'- {v336}: **{v337}**\n' for v336, v337 in v62['gates'].v58())) + f"\n- planted half EM {v104['em']:.3f}, empty tape {v105['em']:.3f}\n- hold CE {v86:.3f} → {v108:.3f}\n", encoding='utf-8')
    v175(v303.v280({'overall': v208, 'gates': v62['gates'], 'open': v62['open_exam']['delta_query_minus_votes']}, indent=2))
    return 0
if v115 == '__main__':
    raise v209(v281())