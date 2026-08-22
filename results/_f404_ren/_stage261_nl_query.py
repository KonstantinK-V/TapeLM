"""
Stage 261 — Can a natural question drive the retrieval, with no cue template anywhere?

256/257 used hand-written cues. 258 removed the cue's lexical overlap but still drew wording
from a fixed per-relation dictionary, so W_sem could learn "this template -> that relation".
Both sides of the exam were still authored. This stage authors neither.

The fact is written from one REAL wikitext sentence mentioning entity E. The question is a
DIFFERENT real wikitext sentence mentioning the same E, truncated just before it. Both
contexts are natural prose, written by different people about the same thing, and nothing in
between was designed by us:

    slot   key = norm( fp(anchor_A) + ctx_fp(sentence A, exclude=E) ),  value = E
    query  from sentence B, prefix up to where E begins        gold = that slot

The discriminator, and the reason this stage exists: LEXICAL OVERLAP between the two contexts.
Bag-of-spellings retrieval works when A and B happen to share words. Report accuracy split by
overlap quartile — if it only works in the high-overlap half, the query is still spelling
matching and NL_QUERY_LEXICAL_ONLY is the verdict, not a win.

Channels compared on identical queries:
    fp-only          W_q(anchor fp + ctx_fp)              = the 256 path
    fp + semantic    blend with W_sem(h_t), as in 258     = trunk understanding in the query

The bank also has to contain entities that are NOT on the exam. The first run built it from the
exam entities alone - 53 slots for 26 fit and 27 eval items - so InfoNCE could satisfy itself by
learning "point at one of these 26", drove the loss to 0.007 by step 40, and then sent eval
queries to the same places: fp+sem 0.037 against fp-only 0.148, below even the shuffled control.
Wiki noise slots make that shortcut worthless.

Keys canonical frozen fp; P1 and trunk frozen; only W_q, W_sem and the blend train. Entities
used for fitting and for evaluation are disjoint.

  python _stage261_nl_query.py [--smoke] [--no-gpt-control]
  python _stage261_nl_query.py --recipe fix1p   # strong fp floor + alpha cap + max-score eval
  python _stage261_nl_query.py --recipe fix1q   # fp-only W_q pretrain, freeze W_q, then sem mixer
  python _stage261_nl_query.py --recipe tape_rerank   # fp top-k, trunk reranks (no alpha)
  python _stage261_nl_query.py --recipe tape_dualkey  # fp + sem keys, read = max (no alpha)
  python _stage261_nl_query.py --recipe tape_symkey   # h_t in key and query symmetrically
  python _stage261_nl_query.py --recipe tape_rerank_val  # fp top-k, rerank reads entity value (fp)
  python _stage261_nl_query.py --recipe tape_qkey     # write-side key = predicted question h
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage260f_open_gate as s260f
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, hidden_and_logits
from _stage262_trunk_swap import ExternalTrunk
v0 = v20('results')
v1 = v0 / 'stage261_decision.json'
v2 = v0 / 'stage261_mini.md'
v3 = v0 / '_stage261_log.txt'
v4 = v20('checkpoints/stage191_p1_curve.pt')
v5 = v20('checkpoints/stage253_joint_l02.pt')
v6 = v20('checkpoints/stage261_nl_query.pt')
v7 = v20('data/_wikitext103_train.txt')
v8 = v153.v21('[A-Za-z][a-z]{2,}')
v9 = 261

def log(v22: v154) -> None:
    v23 = v22 if v22.v279('\n') else v22 + '\n'
    try:
        v280(v23, end='', flush=True)
    except v155:
        v280(v23.v452('ascii', 'replace').v417('ascii'), end='', flush=True)
    v3.v281.v156(parents=True, exist_ok=True)
    with v3.v282('a', encoding='utf-8') as v157:
        v157.v197(v23)
v10 = ('baseline', 'fix1', 'fix1m', 'fix1p', 'fix1q', 'fix2', 'fix3', 'all', 'tape_rerank', 'tape_dualkey', 'tape_symkey', 'tape_rerank_val', 'tape_qkey')
v11 = 60
v12 = {'tape_rerank': 'rerank', 'tape_dualkey': 'dualkey', 'tape_symkey': 'symkey', 'tape_rerank_val': 'rerank_val', 'tape_qkey': 'qkey'}

def recipe_flags(v24: v154) -> v13:
    v24 = v24 if v24 in v10 else 'baseline'
    v25 = v12.v158(v24)
    return {'name': v24, 'tape_mode': v25, 'fp_floor': v24 in ('fix1', 'fix1m', 'fix1p', 'fix1q', 'fix2', 'fix3', 'all'), 'fp_floor_strong': v24 in ('fix1p', 'fix1q'), 'rrf': v24 in ('fix2', 'fix3', 'all'), 'feat_gate': v24 in ('fix3', 'all'), 'hygiene': v24 == 'all', 'alpha_cap': 0.35 if v24 in ('fix1m', 'fix1p', 'fix1q') else None if v24 != 'all' else 0.3, 'score_max_fusion': v24 in ('fix1m', 'fix1p', 'fix1q'), 'wq_fp_pretrain_steps': 800 if v24 == 'fix1q' else 0, 'freeze_wq': v24 == 'fix1q'}

def alpha_warmup_cap(v26: v19, v27: v19, v28: v14=0.3) -> v14:
    if v27 <= 0:
        return 1.0
    return v28 * v168(1.0, v26 / v27)

def fp_floor_loss(v29: v36.v15, v30: v36.v15, v31: v36.v15, v32: v36.v15, v33: v14) -> v36.v15:
    """Blend must not score gold below fp-only on the same batch."""
    v34 = (v29 * v31[v32]).v159(dim=-1)
    v35 = (v30 * v31[v32]).v159(dim=-1)
    return v372.v365(v34 - v35).v160()

def fp_floor_loss_strong(v29: v36.v15, v30: v36.v15, v31: v36.v15, v32: v36.v15, v33: v14, *, v37: v14=8.0, v38: v14=4.0) -> v36.v15:
    """Gold score floor + when fp argmax is gold, blend must not promote a distractor above gold."""
    v34 = v29 @ v31.v307() / v33
    v35 = v30 @ v31.v307() / v33
    v39 = v32.v161(1)
    v40 = v34.v366(1, v39).v162(1)
    v41 = v35.v366(1, v39).v162(1)
    v42 = v372.v365(v40 - v41).v160()
    v43 = v34.v163(dim=-1)
    v44 = v43 == v32
    if v44.v164():
        v165 = v35.v257(dim=-1).v85
        v166 = v372.v365(v165[v44] - v41[v44]).v160()
    else:
        v166 = v34.v283(())
    return v37 * v42 + v38 * v166

def rrf_scores(v34: v36.v15, v45: v36.v15, v46: v19=v11) -> v36.v15:
    """Reciprocal rank fusion; sem cannot demote fp-only winner outside top-k rerank pool."""
    v47 = v34.v167(-1)
    v46 = v168(v46, v47)
    v48 = v36.v284(v34, v46, dim=-1).v49
    v50 = v34.v169()
    for v51 in v170(v34.v167(0)):
        for v171 in v48[v51]:
            v285 = 1 + v19((v34[v51] > v34[v51, v171]).v159())
            v286 = 1 + v19((v45[v51] > v45[v51, v171]).v159())
            v50[v51, v171] = 1.0 / (v11 + v285) + 1.0 / (v11 + v286)
    return v50

class SemQuery(v52.v16):
    """Trunk state -> key space; blend gate from fp conf (2) or 260f retrieval feats (5)."""

    def __init__(v172, v173: v19, v58, *, v174: v18=False, v175: v18=False):
        v418().v287()
        v172.v174 = v174
        v172.v175 = v175
        v172.v176 = v52.v419(v173, 256).v236(v58)
        if not v175:
            v288 = v254(v423.v420) if v174 else v173 + 2
            v172.v289 = v52.v446(v52.v419(v288, 64), v52.v462(), v52.v419(64, 1)).v236(v58)
            v52.v421.v367(v172.v289[-1].v368)
            v52.v421.v369(v172.v289[-1].v370, -2.0)
        if v174:
            v172.v371('mu', v36.v422(v254(v423.v420), device=v58))
            v172.v371('sd', v36.v258(v254(v423.v420), device=v58))

    def q(v172, v177):
        return v372.v290(v172.v176(v177), dim=-1)

    def fit_feat_norm(v172, v178: v216[v36.v15]) -> None:
        if not v178 or not v172.v174:
            return
        v179 = v36.v209(v178)
        v172.v373.v291(v179.v160(0))
        v172.v374.v291(v179.v447(0).v375(0.001))

    def blend_input(v172, v177, v29: v36.v15, v31: v36.v15, v33: v14):
        v83 = v29 @ v31.v307()
        v180 = v83 / v33
        if v172.v174:
            v292 = v423.v376(v83[0], v180[0])
            v293 = (v292 - v172.v373) / v172.v374
            return v293.v161(0)
        v181 = v177 if v177.v359() == 1 else v177.v377(-1)
        v182 = v448(v29, v31).v377(-1)[:2]
        return v36.v424([v181, v182], dim=-1).v161(0)

    def a(v172, v177, v29: v36.v15, v31: v36.v15, v33: v14=0.05):
        if v172.v175:
            return v29.v283(())
        if v29.v359() == 1:
            v29 = v29.v161(0)
        if v177.v359() == 1:
            v177 = v177.v161(0)
        v183 = []
        for v184 in v170(v29.v167(0)):
            v294 = v172.v378(v177[v184], v29[v184:v184 + 1], v31, v33)
            v183.v306(v36.v463(v172.v289(v294)).v162(-1))
        return v36.v209(v183)

class RerankHead(v52.v16):
    """Trunk read step: score fp top-k slot keys with query hidden state (no vector blend)."""

    def __init__(v172, v173: v19, v58):
        v418().v287()
        v172.v176 = v52.v419(v173, 256).v236(v58)

    def scores(v172, v185: v36.v15, v186: v36.v15) -> v36.v15:
        v187 = v372.v290(v172.v176(v185 if v185.v359() == 1 else v185.v377(-1)), dim=-1)
        if v186.v359() == 1:
            return v186 @ v187
        return v186 @ v187

def train_wq_fp_only(v53, v54, v55, v31, v56, v57: v19, v33: v14, v58, *, v59: v19) -> None:
    v60 = v36.v295.v188(v53.v239(), lr=0.002, weight_decay=0.01)
    for v26 in v170(1, v57 + 1):
        v189 = v36.v379(v56)[0]
        v190 = v189[v36.v380(0, v189.v300(), (v168(32, v189.v300()),))]
        v29 = v372.v290(v53(v54[v190]), dim=-1)
        v191 = v372.v296(v29 @ v31.v307() / v33, v55[v190])
        v60.v297(set_to_none=True)
        v191.v298()
        v36.v52.v381.v299(v53.v239(), 1.0)
        v60.v26()
        if v26 == v57 or v26 % v257(1, v59) == 0:
            v230(f'  wq-fp {v26}/{v57} loss={v14(v191):.3f}')

def _fp_topk_cands(v34: v36.v15, v46: v19) -> v36.v15:
    v46 = v168(v46, v34.v300())
    return v36.v284(v34, v46).v49

def _gold_in_fp_topk(v34: v36.v15, v32: v19, v46: v19) -> v62[v18, v36.v15]:
    v61 = v192(v34, v46)
    return (v19(v32) in v61.v301(), v61)

def train_tape_rerank(v53, v63, v64, v31, v65, v55, v54, v56, v57, v33, v66, v58, *, v67: v36.v15 | None=None) -> v13:
    """Rerank only within fp top-k; never inject gold (avoids low-fp outlier leak)."""
    v60 = v36.v295.v188(v216(v63.v239()), lr=0.002, weight_decay=0.01)
    v68 = v67 if v67 is not None else v31
    v193, v194 = (0, 0)
    for v26 in v170(1, v57 + 1):
        v189 = v36.v379(v56)[0]
        v190 = v189[v36.v380(0, v189.v300(), (v168(24, v189.v300()),))]
        v195 = []
        for v196 in v190.v301():
            v29 = v372.v290(v53(v54[v196:v196 + 1]), dim=-1)[0]
            v34 = v31 @ v29
            v382, v61 = v383(v34, v19(v55[v196]), v66)
            if not v382:
                v193 += 1
                continue
            v194 += 1
            v180 = v63.v425(v65[v196], v68[v61]) / v33
            v302 = (v61 == v55[v196]).v464(as_tuple=True)[0].v162()
            v195.v306(v372.v296(v180.v161(0), v302.v161(0)))
        if not v195:
            continue
        v191 = v36.v209(v195).v160()
        v60.v297(set_to_none=True)
        v191.v298()
        v60.v26()
        if v26 == 40 or v26 % v257(1, v57 // 5) == 0:
            v230(f'  rerank {v26}/{v57} loss={v14(v191):.3f} train={v194} skip={v193}')
    return {'rerank_train_steps': v194, 'rerank_skip_gold_not_in_topk': v193}

class QKeyTape(v52.v16):
    """Write-side key = address in question space; read-side query from actual question h_t."""

    def __init__(v172, v173: v19, v58):
        v418().v287()
        v172.v197 = v52.v419(v173, 256).v236(v58)
        v172.v68 = v52.v419(v173, 256).v236(v58)

    def keys(v172, v198: v36.v15) -> v36.v15:
        return v372.v290(v172.v197(v198), dim=-1)

    def query(v172, v199: v36.v15) -> v36.v15:
        if v199.v359() == 1:
            return v372.v290(v172.v68(v199), dim=-1)
        return v372.v290(v172.v68(v199), dim=-1)

def train_tape_qkey(v69: v200, v70, v71, v56, v57, v33, v58):
    v54, v65, v55, v201 = v71
    v60 = v36.v295.v188(v69.v239(), lr=0.002, weight_decay=0.01)
    for v26 in v170(1, v57 + 1):
        v202 = v69.v244(v70)
        v189 = v36.v379(v56)[0]
        v190 = v189[v36.v380(0, v189.v300(), (v168(32, v189.v300()),))]
        v187 = v69.v303(v65[v190])
        v191 = v372.v296(v187 @ v202.v307() / v33, v55[v190])
        v60.v297(set_to_none=True)
        v191.v298()
        v60.v26()
        if v26 == 40 or v26 % v257(1, v57 // 5) == 0:
            v230(f'  qkey {v26}/{v57} loss={v14(v191):.3f}')

def train_tape_dualkey(v53, v72, v70, v71, v56, v57, v33, v58):
    v54, v65, v55, v203 = v71
    v60 = v36.v295.v188(v216(v53.v239()) + v216(v72.v239()), lr=0.002, weight_decay=0.01)
    for v26 in v170(1, v57 + 1):
        v189 = v36.v379(v56)[0]
        v190 = v189[v36.v380(0, v189.v300(), (v168(32, v189.v300()),))]
        v204 = v72.v187(v70)
        v29 = v372.v290(v53(v54[v190]), dim=-1)
        v205 = v72.v187(v65[v190])
        v206 = v36.v304(v29 @ v203.v307(), v205 @ v204.v307())
        v191 = v372.v296(v206 / v33, v55[v190])
        v60.v297(set_to_none=True)
        v191.v298()
        v60.v26()
        if v26 == 40 or v26 % v257(1, v57 // 5) == 0:
            v230(f'  dualkey {v26}/{v57} loss={v14(v191):.3f}')

def train_tape_symkey(v53, v72, v70, v71, v56, v57, v33, v58):
    v54, v65, v55, v203 = v71
    v60 = v36.v295.v188(v216(v53.v239()) + v216(v72.v239()), lr=0.002, weight_decay=0.01)
    for v26 in v170(1, v57 + 1):
        v189 = v36.v379(v56)[0]
        v190 = v189[v36.v380(0, v189.v300(), (v168(32, v189.v300()),))]
        v207 = v372.v290(v203 + v72.v187(v70), dim=-1)
        v187 = v372.v290(v53(v54[v190]) + v72.v187(v65[v190]), dim=-1)
        v191 = v372.v296(v187 @ v207.v307() / v33, v55[v190])
        v60.v297(set_to_none=True)
        v191.v298()
        v60.v26()
        if v26 == 40 or v26 % v257(1, v57 // 5) == 0:
            v230(f'  symkey {v26}/{v57} loss={v14(v191):.3f}')

@v36.v82()
def acc_20way_batch(v53, v72, v73, v31, v74, v75, v76: v18, v77: v14, *, v78: v18, v33: v14) -> v14:
    v79 = v305.v208(v9 + 99)
    v80 = []
    for v81 in v74:
        v29 = v372.v290(v53(v81['raw'].v161(0)), dim=-1)
        if v76 and v72 is not None:
            v90 = v14(v72.v90(v81['h'], v29[0], v31, v33).v377(-1)[0])
            v205 = v372.v290(v72.v187(v81['h'].v161(0)), dim=-1).v377(-1)
            if v78:
                v34 = (v31 @ v29[0]).v161(0)
                v45 = (v31 @ v205).v161(0)
                v83 = v449(v34, v45)[0]
            else:
                v187 = v372.v290((1 - v90) * v29[0] + v90 * v205, dim=-1)
                v83 = v31 @ v187
        else:
            v83 = v31 @ v29[0]
        v189 = [v171 for v171 in v79.v450(v170(v31.v167(0)), v168(20 * 3, v31.v167(0))) if v171 != v81['slot']][:19]
        v80.v306(v19(v407((v14(v83[v81['slot']]) >= v14(v83[v171]) for v171 in v189))))
    return v14(v397.v160(v80)) if v80 else 0.0

def fp_conf(v29, v31):
    v83 = v29 @ v31.v307()
    v84 = v36.v284(v83, v168(2, v83.v167(-1)), dim=-1).v85
    if v84.v167(-1) < 2:
        return v36.v209([v84[..., 0], v84[..., 0]], dim=-1)
    return v36.v209([v84[..., 0], v84[..., 0] - v84[..., 1]], dim=-1)

def ctx_words(v86: v154, v87: v154 | None=None) -> v17:
    return {v309.v308() for v309 in v8.v212(v86) if v309 != v87}

def entity_in_query(v88: v154, v89: v154) -> v18:
    """True if gold entity string appears in the natural query prefix (before truncation at E)."""
    return v153.v310(f'\\b{v153.v451(v88)}\\b', v89, v153.v311) is not None

def jaccard(v90: v17, v51: v17) -> v14:
    return v254(v90 & v51) / v257(1, v254(v90 | v51))

def fp_raw(v73: v210, v86: v154, v91: v18=True):
    """256 recipe is anchor fp + context — shared anchor in 258; 261 write/ask anchors differ."""
    v92 = v73.v211(v86)
    if v92 is None:
        return None
    if not v91:
        return v92
    v93 = v312.v212(v86)
    return v372.v290(v73.v430([v93[-1]])[0] + v92, dim=-1) if v93 else v92

@v36.v82()
def trunk_state(v94, v95, v96, v97, v58, v86):
    v98 = [v184 for v184 in v96.v452(v86).v98 if v184 != v97][-v384:]
    if not v98:
        return None
    v177, v213 = v214(v94, v95, v36.v256([v98], device=v58), v97)
    return v177[0, -1].v385().v14()

def collect(v99, v73, v100=2):
    """Entities appearing in at least two different real sentences: one writes the slot, the
    other asks the question. Neither sentence was authored by us."""
    v101 = v215(v216)
    for v102 in v99:
        for v22 in v386.v313(v102):
            v119 = v22.v387(1)
            if v254(v119) < 5:
                continue
            v314 = v257(0, v22.v453() - 140)
            v93 = [v309 for v309 in v312.v212(v102[v314:v22.v453()]) if v309 != v119]
            if v93 and v254(v101[v119]) < 4:
                v101[v119].v306({'line': v102, 'start': v22.v453(), 'end': v22.v465(), 'anchor': v93[-1]})
    return {v119: v217 for v119, v217 in v101.v74() if v254(v217) >= v100}

def main() -> v19:
    v103 = v315.v218()
    v103.v219('--smoke', action='store_true')
    v103.v219('--steps', type=v19, default=0)
    v103.v219('--tau', type=v14, default=0.05)
    v103.v219('--entities', type=v19, default=0)
    v103.v219('--distractor-slots', type=v19, default=0, help='wiki entities added to the bank that no query ever asks for')
    v103.v219('--no-anchor', action='store_true', help='ctx only on keys and queries (write/ask anchors are different entities)')
    v103.v219('--query-names-entity', action='store_true', help='variant-2 ceiling: append gold entity to query fp+trunk (names what to retrieve)')
    v103.v219('--model', type=v154, default='', help='external frozen causal LM for h_t (262 ExternalTrunk); empty = curve trunk')
    v103.v219('--no-gpt-control', action='store_true')
    v103.v219('--recipe', type=v154, default='baseline', choices=v10)
    v103.v219('--rerank-k', type=v19, default=32, help='fp pool size for tape_rerank read step')
    v104 = v103.v220()
    v105 = v221(v104.v222)
    v3.v223('', encoding='utf-8')
    v58 = v36.v58('cuda' if v36.v426.v388() else 'cpu')
    v106 = v305.v208(v9)
    v36.v224(v9)
    v107 = v225.v225()
    v91 = not v104.v226
    v108 = v104.v108
    v57 = v104.v57 or (200 if v104.v229 else 800)
    v109 = v104.v227 or (60 if v104.v229 else 400)
    v110 = v104.v228 or (400 if v104.v229 else 4000)
    v111 = 3000 if v104.v229 else 25000
    v230(f"Stage261 nl query start {v460.v444(v461.v445).v360()} device={v58} steps={v57} recipe={v105['name']} anchor={v91} query_names_entity={v108} trunk={v104.v94 or 'curve'}")
    v213, v213, v231, v232 = v233()
    v96 = v316.v234(v154(v389.v317))
    v112 = v96.v235()
    v97 = v96.v318(v319) or 0
    v95 = v427.v390(v96, v231, v97, v112).v236(v58)
    v113 = v5 if v5.v320() else v4
    v94 = v391(v232, v112).v236(v58)
    v94.v237(v36.v392(v113, map_location=v58, weights_only=False)['model'])
    v94.v238()
    for v114 in v94.v239():
        v114.v321(False)
    v115 = v391(v232, v112).v236(v58)
    v115.v237(v36.v392(v4, map_location=v58, weights_only=False)['model'])
    v115.v238()
    for v114 in v115.v239():
        v114.v321(False)
    v73 = v210(v115, v231, v58)
    v116: v240 | None = None
    if v104.v94:
        try:
            v116 = v240(v104.v94, v58)
            v230(f'  external trunk: {v104.v94} hidden={v116.v359} (query h_t only; fp keys unchanged)')
        except v322 as e:
            v230(f'  could not load {v104.v94}: {v472(v119).v152}: {v119}')
            return 1

    def query_h(v86: v154):
        if v116 is not None:
            return v116.v393(v86)
        return v323(v94, v95, v96, v97, v58, v86)
    with v7.v282('r', encoding='utf-8', errors='ignore') as v157:
        v241 = v157.v68(3000000 if v104.v229 else 20000000)
    v99 = [v394.v325() for v394 in v241.v428('\n') if 80 <= v254(v394.v325()) <= 400][:v111]
    v117 = v242(v99, v73)
    v118 = v324(v117)[:v109]
    v106.v243(v118)
    v230(f'  entities with >=2 natural mentions: {v254(v117)} (using {v254(v118)})')
    if v254(v118) < 16:
        v230('  not enough multi-mention entities')
        return 1
    v244, v245, v74, v246 = ([], [], [], [])
    for v119 in v118:
        v247 = v117[v119]
        v90, v51 = (v247[0], v247[1])
        v248 = v90['line'][v257(0, v90['start'] - 140):v168(v254(v90['line']), v90['end'] + 140)]
        v46 = v73.v211(v248, exclude=v119)
        if v46 is None:
            continue
        v249 = v323(v94, v95, v96, v97, v58, v248)
        v89 = v51['line'][v257(0, v51['start'] - 200):v51['start']].v325()
        if v254(v8.v212(v89)) < 4:
            continue
        v250 = v326(v119, v89)
        v251 = f'{v89} {v119}' if v108 else v89
        v252 = v327(v73, v251, v91)
        v253 = v328(v251)
        if v252 is None or v253 is None or v249 is None:
            continue
        v244.v306(v372.v290(v73.v430([v90['anchor']])[0] + v46, dim=-1) if v91 else v46)
        v246.v306(v249)
        v74.v306({'ent': v119, 'slot': v254(v245), 'qtext': v89, 'q_use': v251, 'raw': v252, 'h': v253, 'h_write': v249, 'ent_in_query': v250, 'overlap': v429(v454(v248, v119), v454(v89, v119))})
        v245.v306(v119)
    if v254(v74) < 16:
        v230('  not enough usable (write, ask) pairs')
        return 1
    v120 = v254(v244)
    v121 = {v81['ent'] for v81 in v74}
    for v102 in v99:
        if v254(v244) >= v120 + v110:
            break
        for v22 in v386.v313(v102):
            v119 = v22.v387(1)
            if v254(v119) < 5 or v119 in v121:
                continue
            v314, v395 = (v257(0, v22.v453() - 140), v168(v254(v102), v22.v465() + 140))
            v92 = v73.v211(v102[v314:v395], exclude=v119)
            if v92 is None:
                continue
            v93 = [v309 for v309 in v312.v212(v102[v314:v22.v453()]) if v309 != v119]
            if v91 and (not v93):
                continue
            v249 = v323(v94, v95, v96, v97, v58, v102[v314:v395])
            if v249 is None:
                continue
            v244.v306(v372.v290(v73.v430([v93[-1]])[0] + v92, dim=-1) if v91 else v92)
            v246.v306(v249)
            v245.v306(v119)
            v121.v396(v119)
            if v254(v244) >= v120 + v110:
                break
    v31 = v36.v209(v244, 0).v236(v58).v14()
    v122 = v372.v290(v73.v430(v245), dim=-1).v236(v58)
    v230(f'  bank: {v120} exam slots + {v254(v244) - v120} wiki noise = {v254(v244)}')
    v123 = v254(v74) // 2
    v64, v255 = (v74[:v123], v74[v123:])
    v77 = v14(v397.v329([v81['overlap'] for v81 in v255]))
    v230(f'  exam_slots={v120} fit={v254(v64)} eval={v254(v255)} | overlap median={v77:.3f}')
    v70 = v36.v209(v246, 0).v236(v58).v14()
    v54 = v36.v209([v81['raw'] for v81 in v64]).v236(v58).v14()
    v65 = v36.v209([v81['h'] for v81 in v64]).v236(v58).v14()
    v55 = v36.v256([v81['slot'] for v81 in v64], device=v58)
    v124 = v257(4, v254(v64) // 5)
    v125 = v216(v170(v254(v64) - v124, v254(v64)))
    v56 = v36.v258(v254(v64), dtype=v36.v18)
    v56[v125] = False
    v126 = [v64[v184] for v184 in v125]
    v25 = v105.v158('tape_mode')
    v127: v13 = {}
    v66 = v257(8, v104.v66)
    v128 = v57 if v25 else v257(400, v57 // 2)
    v129 = v57
    v130 = v57
    v53 = v330.v259(v58)
    v131 = v330.v259(v58)
    v72 = None
    v63 = None
    v69 = None
    if v25:
        v230(f'  tape mode={v25} fp_steps={v128} tape_steps={v129} rerank_k={v66}')
        v260 = v19(v64[0]['h'].v300())
        v71 = (v54, v65, v55, v31)
        if v25 != 'qkey':
            v398(v53, v54, v55, v31, v56, v128, v104.v33, v58, log_every=v257(1, v128 // 4))
            v131.v237(v53.v401())
        else:
            v398(v53, v54, v55, v31, v56, v128, v104.v33, v58, log_every=v257(1, v128 // 4))
            v131.v237(v53.v401())
        if v25 in ('rerank', 'rerank_val'):
            v63 = v399(v260, v58)
            for v114 in v53.v239():
                v114.v321(False)
            v331 = v122 if v25 == 'rerank_val' else v31
            v127 = v400(v53, v63, v64, v31, v65, v55, v54, v56, v129, v104.v33, v66, v58, read_vecs=v331)
            v63.v238()
        elif v25 == 'qkey':
            v69 = v200(v260, v58)
            v431(v69, v70, v71, v56, v129, v104.v33, v58)
            v69.v238()
        else:
            v72 = v341(v260, v58, key_only=True)
            if v25 == 'dualkey':
                v455(v53, v72, v70, v71, v56, v129, v104.v33, v58)
            else:
                v456(v53, v72, v70, v71, v56, v129, v104.v33, v58)
            v72.v238()

        @v36.v82()
        def score_fp(v332, v333=v31, v334: v19=20, v335=v53):
            v79 = v305.v208(v9 + 5)
            v354, v314, v395, v355 = ([], [], [], [])
            for v81 in v332:
                v29 = v372.v290(v335(v81['raw'].v161(0)), dim=-1)[0]
                v83 = v333 @ v29
                v273 = 1 + v19((v83 > v83[v81['slot']]).v159())
                v354.v306(v273)
                (v395 if v81['overlap'] > v77 else v314).v306(v19(v273 == 1))
                v189 = [v171 for v171 in v79.v450(v170(v333.v167(0)), v168(v334 * 3, v333.v167(0))) if v171 != v81['slot']][:v334 - 1]
                v355.v306(v19(v407((v14(v83[v81['slot']]) >= v14(v83[v171]) for v171 in v189))))
            v273 = v397.v356(v354, dtype=v397.v413)
            return {'top1': v14(v397.v160(v273 == 1)), 'mrr': v14(v397.v160(1.0 / v273)), 'median_rank': v14(v397.v329(v273)), 'top1_low_overlap': v14(v397.v160(v314)) if v314 else v14('nan'), 'top1_high_overlap': v14(v397.v160(v395)) if v395 else v14('nan'), 'alpha': 0.0, 'n': v254(v354), f'acc_{v334}way': v14(v397.v160(v355)) if v355 else v14('nan'), f'chance_{v334}way': 1.0 / v334}

        @v36.v82()
        def score_tape(v332, v333=v31, v336=v70, v337=v122, v334: v19=20):
            v79 = v305.v208(v9 + 5)
            v354, v314, v395, v355 = ([], [], [], [])
            v338 = -10000.0
            v331 = v337 if v25 == 'rerank_val' else v333
            for v81 in v332:
                v29 = v372.v290(v53(v81['raw'].v161(0)), dim=-1)[0]
                v32 = v19(v81['slot'])
                if v25 in ('rerank', 'rerank_val'):
                    v34 = v333 @ v29
                    v382, v61 = v383(v34, v32, v66)
                    if v382:
                        v83 = v36.v466((v333.v167(0),), v338, device=v333.v58, dtype=v34.v471)
                        v83[v61] = v63.v425(v81['h'], v331[v61])
                    else:
                        v83 = v34
                elif v25 == 'qkey':
                    v202 = v69.v244(v336)
                    v83 = v202 @ v69.v303(v81['h'])
                elif v25 == 'dualkey':
                    v467 = v72.v187(v336)
                    v205 = v72.v187(v81['h'].v161(0))[0]
                    v83 = v36.v304(v333 @ v29, v467 @ v205)
                else:
                    v207 = v372.v290(v333 + v72.v187(v336), dim=-1)
                    v187 = v372.v290(v53(v81['raw'].v161(0)) + v72.v187(v81['h'].v161(0)), dim=-1)[0]
                    v83 = v207 @ v187
                v273 = 1 + v19((v83 > v83[v32]).v159())
                v354.v306(v273)
                (v395 if v81['overlap'] > v77 else v314).v306(v19(v273 == 1))
                v189 = [v171 for v171 in v79.v450(v170(v333.v167(0)), v168(v334 * 3, v333.v167(0))) if v171 != v32][:v334 - 1]
                if v25 in ('rerank', 'rerank_val'):
                    v355.v306(v19(v407((v14(v83[v32]) >= v14(v83[v171]) for v171 in v189))))
                elif v25 == 'qkey':
                    v355.v306(v19(v407((v14(v83[v32]) >= v14(v83[v171]) for v171 in v189))))
                else:
                    v355.v306(v19(v407((v14(v83[v32]) >= v14(v83[v171]) for v171 in v189))))
            v273 = v397.v356(v354, dtype=v397.v413)
            return {'top1': v14(v397.v160(v273 == 1)), 'mrr': v14(v397.v160(1.0 / v273)), 'median_rank': v14(v397.v329(v273)), 'top1_low_overlap': v14(v397.v160(v314)) if v314 else v14('nan'), 'top1_high_overlap': v14(v397.v160(v395)) if v395 else v14('nan'), 'alpha': 0.0, 'n': v254(v354), f'acc_{v334}way': v14(v397.v160(v355)) if v355 else v14('nan'), f'chance_{v334}way': 1.0 / v334}
        v261 = v339(v255)
        v262 = v340(v255)
        v263 = v339(v255, Wq=v131)
        v264 = v36.v350(v31.v167(0), generator=v36.v470().v224(v9 + 1)).v236(v31.v58)
        v265 = v31[v264]
        v266 = v70[v264] if v25 in ('dualkey', 'symkey', 'qkey') else v70
        v267 = v340(v255, Kmat=v265, H_w=v266)
        v230(f'fp-only: {v416.v362(v261)}')
        v230(f'tape ({v25}): {v416.v362(v262)}')
        v230(f"shuffled keys: top1={v267['top1']:.3f}")
    else:
        v72 = v341(v19(v64[0]['h'].v300()), v58, feat_gate=v105['feat_gate'])
        v53 = v330.v259(v58)
        v131 = v330.v259(v58)
        v131.v237(v53.v401())
        for v114 in v131.v239():
            v114.v321(False)
        if v105['feat_gate']:
            v342 = []
            v343 = v330.v259(v58)
            with v36.v82():
                for v81 in v64:
                    v432 = v372.v290(v343(v81['raw'].v161(0)), dim=-1)
                    v83 = v432 @ v31.v307()
                    v342.v306(v423.v376(v83[0], v83[0] / v104.v33))
            v72.v402(v342)
        v268 = v19(v105.v158('wq_fp_pretrain_steps') or 0)
        if v268 > 0:
            v344 = v36.v295.v188(v53.v239(), lr=0.002, weight_decay=0.01)
            for v345 in v170(1, v268 + 1):
                v189 = v36.v379(v56)[0]
                v190 = v189[v36.v380(0, v189.v300(), (v168(32, v189.v300()),))]
                v29 = v372.v290(v53(v54[v190]), dim=-1)
                v403 = v372.v296(v29 @ v31.v307() / v104.v33, v55[v190])
                v344.v297(set_to_none=True)
                v403.v298()
                v36.v52.v381.v299(v53.v239(), 1.0)
                v344.v26()
                if v345 == v268 or v345 % v257(1, v268 // 4) == 0:
                    v230(f'  wq-pretrain {v345}/{v268} loss={v14(v403):.3f}')
            v131.v237(v53.v401())
            if v105['freeze_wq']:
                for v114 in v53.v239():
                    v114.v321(False)
        v269 = [] if v105['freeze_wq'] else v216(v53.v239())
        v60 = v36.v295.v188(v216(v72.v239()) + v269, lr=0.002, weight_decay=0.01)
        v270 = 200 if v105['hygiene'] else 0
        v346, v271 = (v57, v57)
        for v26 in v170(1, v57 + 1):
            v189 = v36.v379(v56)[0]
            v190 = v189[v36.v380(0, v189.v300(), (v168(32, v189.v300()),))]
            v29 = v372.v290(v53(v54[v190]), dim=-1)
            v347 = v72.v90(v65[v190], v29, v31, v104.v33)
            v90 = v347.v377(-1, 1)
            if v105['hygiene']:
                v404 = v433(v26, v270, max_cap=v14(v105['alpha_cap'] or 0.3))
                v90 = (v90 * v404).v434(0.0, v404)
            elif v105['alpha_cap'] is not None:
                v90 = v90.v434(0.0, v14(v105['alpha_cap']))
            v205 = v72.v187(v65[v190])
            v187 = v372.v290((1 - v90) * v29 + v90 * v205, dim=-1)
            v180 = v187 @ v31.v307() / v104.v33
            v191 = v372.v296(v180, v55[v190])
            if v105['fp_floor_strong']:
                v191 = v191 + v457(v29, v187, v31, v55[v190], v104.v33)
                if not v105['freeze_wq']:
                    with v36.v82():
                        v458 = v372.v290(v131(v54[v190]), dim=-1)
                    v191 = v191 + 0.5 * v372.v296(v458 @ v31.v307() / v104.v33, v55[v190])
            elif v105['fp_floor']:
                v191 = v191 + v468(v29, v187, v31, v55[v190], v104.v33)
            v60.v297(set_to_none=True)
            v191.v298()
            v36.v52.v381.v299(v216(v72.v239()) + v269, 1.0)
            v60.v26()
            if v105['hygiene'] and v26 >= 80 and (v26 % 40 == 0):
                v72.v238()
                v137 = v435(v53, None, v73, v31, v126, {}, False, v77, rrf=False, tau=v104.v33)
                v405 = v435(v53, v72, v73, v31, v126, {}, True, v77, rrf=v105['rrf'], tau=v104.v33)
                v72.v436()
                if v405 + 1e-06 < v137:
                    v271 = v26
                    v230(f'  early-stop @ {v26}: holdout 20-way fp={v137:.3f} sem={v405:.3f}')
                    break
                v346 = v26
            if v26 == 40 or v26 % v257(1, v57 // 5) == 0:
                v230(f'  step {v26}/{v57} loss={v14(v191):.3f} a={v14(v90.v160()):.3f}')
        v72.v238()
        v130 = v271

        def _alpha_eval(v348: v36.v15) -> v14:
            v90 = v14(v348.v377(-1)[0])
            if v105['alpha_cap'] is not None:
                return v168(v90, v14(v105['alpha_cap']))
            return v90

        @v36.v82()
        def score(v332, v76, v333=v31, v334: v19=20, v335=v53):
            v79 = v305.v208(v9 + 5)
            v354, v183, v314, v395, v355 = ([], [], [], [], [])
            for v81 in v332:
                v29 = v372.v290(v335(v81['raw'].v161(0)), dim=-1)[0]
                if v76:
                    v437 = v459(v72.v90(v81['h'], v29, v333, v104.v33))
                    v183.v306(v437)
                    v205 = v372.v290(v72.v187(v81['h'].v161(0)), dim=-1).v377(-1)
                    v438 = v333 @ v29
                    if v105['rrf']:
                        v83 = v449(v438.v161(0), (v333 @ v205).v161(0))[0]
                    elif v105['score_max_fusion']:
                        v469 = v372.v290((1 - v437) * v29 + v437 * v205, dim=-1)
                        v83 = v36.v304(v438, v333 @ v469)
                    else:
                        v187 = v372.v290((1 - v437) * v29 + v437 * v205, dim=-1)
                        v83 = v333 @ v187
                else:
                    v83 = v333 @ v29
                v273 = 1 + v19((v83 > v83[v81['slot']]).v159())
                v354.v306(v273)
                (v395 if v81['overlap'] > v77 else v314).v306(v19(v273 == 1))
                v189 = [v171 for v171 in v79.v450(v170(v333.v167(0)), v168(v334 * 3, v333.v167(0))) if v171 != v81['slot']][:v334 - 1]
                v355.v306(v19(v407((v14(v83[v81['slot']]) >= v14(v83[v171]) for v171 in v189))))
            v273 = v397.v356(v354, dtype=v397.v413)
            return {'top1': v14(v397.v160(v273 == 1)), 'mrr': v14(v397.v160(1.0 / v273)), 'median_rank': v14(v397.v329(v273)), 'top1_low_overlap': v14(v397.v160(v314)) if v314 else v14('nan'), 'top1_high_overlap': v14(v397.v160(v395)) if v395 else v14('nan'), 'alpha': v14(v397.v160(v183)) if v183 else 0.0, 'n': v254(v354), f'acc_{v334}way': v14(v397.v160(v355)) if v355 else v14('nan'), f'chance_{v334}way': 1.0 / v334}
        v261, v262 = (v349(v255, False), v349(v255, True))
        v263 = v349(v255, False, Wq=v131)
        v264 = v36.v350(v31.v167(0), generator=v36.v470().v224(v9 + 1))
        v267 = v349(v255, True, Kmat=v31[v264.v236(v31.v58)])
        v230(f'fp-only: {v416.v362(v261)}')
        v230(f'fp+sem : {v416.v362(v262)}')
        v230(f"shuffled keys: top1={v267['top1']:.3f}")
    v132 = None
    if not v104.v351 and (not v25):
        try:
            v352 = v330.v406(v58)
            v353 = []
            for v81 in v74:
                v119 = v330.v439(v352, v96, v97, v58, [v184 for v184 in v96.v452(v81['qtext']).v98 if v184 != v97])
                v353.v306(None if v119 is None else v119.v385().v14())
            if v407((v294 is not None for v294 in v353)):
                for v81, v294 in v440(v74, v353):
                    v81['h_gpt'] = v294
                v408 = v341(v19(v353[0].v300()), v58, feat_gate=v105['feat_gate'])
                v409 = v330.v259(v58)
                v410 = v36.v295.v188(v216(v408.v239()) + v216(v409.v239()), lr=0.002)
                v411 = v36.v209([v81['h_gpt'] for v81 in v64]).v236(v58).v14()
                for v412 in v170(v130):
                    v190 = v36.v380(0, v54.v167(0), (v168(32, v54.v167(0)),), device=v58)
                    v432 = v372.v290(v409(v54[v190]), dim=-1)
                    v90 = v408.v90(v411[v190], v432, v31, v104.v33).v161(-1)
                    v187 = v372.v290((1 - v90) * v432 + v90 * v408.v187(v411[v190]), dim=-1)
                    v441 = v372.v296(v187 @ v31.v307() / v104.v33, v55[v190])
                    if v105['fp_floor']:
                        v441 = v441 + v468(v432, v187, v31, v55[v190], v104.v33)
                    v410.v297(set_to_none=True)
                    v441.v298()
                    v410.v26()
                v408.v238()
                with v36.v82():
                    v354 = []
                    for v81 in v255:
                        v432 = v372.v290(v409(v81['raw'].v161(0)), dim=-1)[0]
                        v90 = v408.v90(v81['h_gpt'], v432, v31, v104.v33)
                        v437 = v14(v90 if v90.v300() == 1 else v90.v160())
                        v205 = v408.v187(v81['h_gpt'].v161(0))[0]
                        if v105['rrf']:
                            v34 = (v31 @ v432).v161(0)
                            v45 = (v31 @ v205).v161(0)
                            v206 = v449(v34, v45)[0]
                        else:
                            v187 = v372.v290((1 - v437) * v432 + v437 * v205, dim=-1)
                            v206 = v31 @ v187
                        v354.v306(1 + v19((v206 > v206[v81['slot']]).v159()))
                    v442 = v397.v356(v354, dtype=v397.v413)
                    v132 = {'top1': v14(v397.v160(v442 == 1)), 'mrr': v14(v397.v160(1.0 / v442))}
                v230(f'gpt2+sem: {v416.v362(v132)}')
        except v322 as e:
            v230(f'  gpt control unavailable: {v472(v119).v152}: {v119}')

    @v36.v82()
    def fp_subset_metrics(v272):
        if not v272:
            return {'n': 0, 'top1': v14('nan'), 'acc_20way': v14('nan')}
        v79 = v305.v208(v9 + 7)
        v354, v355 = ([], [])
        for v81 in v272:
            v29 = v372.v290(v53(v81['raw'].v161(0)), dim=-1)[0]
            v83 = v31 @ v29
            v32 = v19(v81['slot'])
            v354.v306(1 + v19((v83 > v83[v32]).v159()))
            v189 = [v171 for v171 in v79.v450(v170(v31.v167(0)), v168(20 * 3, v31.v167(0))) if v171 != v32][:19]
            v355.v306(v19(v407((v14(v83[v32]) >= v14(v83[v171]) for v171 in v189))))
        v273 = v397.v356(v354, dtype=v397.v413)
        return {'n': v254(v272), 'top1': v14(v397.v160(v273 == 1)), 'acc_20way': v14(v397.v160(v355)) if v355 else v14('nan')}
    v133 = [v81 for v81 in v255 if v81.v158('ent_in_query')]
    v134 = [v81 for v81 in v255 if not v81.v158('ent_in_query')]
    v135 = {'query_names_entity': v108, 'eval_n': v254(v255), 'ent_in_natural_query_n': v254(v133), 'fp_ent_absent': v357(v134), 'fp_ent_present_leak': v357(v133)}
    v230(f'  query entity diag: {v416.v362(v135)}')
    v136 = 1.0 / v254(v245)
    v137 = v261.v158('acc_20way', 0.0)
    v138 = v263.v158('acc_20way', v137)
    v139 = v262.v158('acc_20way', 0.0)
    v140 = v267.v158('acc_20way', 0.0)
    v141 = v262['top1'] >= 0.3
    v142 = v262['top1'] >= v261['top1'] + 0.1
    v143 = not v443.v414(v262['top1_low_overlap']) and v262['top1_low_overlap'] >= 0.25
    v144 = not v443.v414(v262['top1_low_overlap']) and (not v443.v414(v262['top1_high_overlap'])) and (v262['top1_low_overlap'] > 0.0) and (v262['top1_low_overlap'] >= 0.6 * v262['top1_high_overlap'])
    v145 = v137 >= 0.05 + 1.0 / 20
    v146 = v139 >= 0.05 + 1.0 / 20
    v147 = v139 < v137 - 0.03
    v148 = v139 >= v137 * 0.9 and v137 >= 0.12
    v149 = v267['top1'] <= v257(0.05, v136 * 3)
    v150 = v140 >= 0.045 and v140 <= 0.085
    v151 = v132 is not None and v132['top1'] <= v257(0.02, v261['top1'])
    if v141 and v142 and v149 and v144 and v143:
        v274 = 'NL_QUERY_OK'
    elif v141 and v149 and (not v144):
        v274 = 'NL_QUERY_LEXICAL_ONLY'
    elif v141 and v149:
        v274 = 'NL_QUERY_PARTIAL'
    elif v148 and v145 and v149 and v150:
        v274 = 'NL_QUERY_MIXER_OK'
    elif v145 and v149 and v150 and v147:
        v274 = 'NL_QUERY_NWAY_FP_ONLY'
    elif v146 and v149 and v150 and (not v147) and (v137 >= 0.12):
        v274 = 'NL_QUERY_NWAY_ONLY'
    elif v132 is not None and v151:
        v274 = 'NL_QUERY_NO_AT_SCALE'
    elif v132 is None and v261['top1'] <= 0.05 and (v262['top1'] <= 0.05):
        v274 = 'NL_QUERY_NO_AT_SCALE'
    else:
        v274 = 'NL_QUERY_NO'
    v50 = {'stage': 261, 'overall': v274, 'recipe': v105['name'], 'recipe_flags': v105, 'use_anchor': v91, 'query_names_entity': v108, 'query_entity_diag': v135, 'trunk': v113.v24, 'fp_version': v330.v358(), 'external_model': v104.v94 or None, 'external_hidden': v116.v359 if v116 is not None else None, 'steps': v57, 'steps_done': v130, 'slots': v254(v245), 'exam_slots': v120, 'noise_slots': v254(v245) - v120, 'n_fit': v254(v64), 'n_eval': v254(v255), 'chance': v136, 'overlap_median': v77, **({'rerank_stats': v127} if v127 else {}), 'gates': {'G_works': v141, 'G_beats_fp_only': v142, 'G_low_overlap_works': v143, 'G_not_lexical': v144, 'G_tape_causal': v149, 'G_tape_causal_20way': v150, 'G_signal_fp_20way': v145, 'G_signal_sem_20way': v146, 'G_sem_harms_fp': v147, 'G_sem_neutral_vs_fp': v148}, 'read': {'acc_20way_fp': v137, 'acc_20way_fp_init_Wq': v138, 'acc_20way_fp_trained_Wq': v261.v158('acc_20way', 0.0), 'acc_20way_sem': v139, 'acc_20way_shuffled': v140, 'sem_over_fp_20way': v139 / v257(v137, 1e-09), 'blend_alpha_eval': v262.v158('alpha', 0.0), 'full_bank_top1_fp': v261.v158('top1'), 'full_bank_top1_sem': v262.v158('top1'), 'full_bank_median_rank_fp': v261.v158('median_rank'), 'full_bank_median_rank_sem': v262.v158('median_rank')}, 'summary': {'fp_only': v261, 'fp_only_frozen_Wq': v263, 'fp_plus_sem': v262, 'shuffled_keys': v267, 'gpt_control': v132}, 'gpt_parity': v18(v151) if v132 is not None else None, 'note': 'Full: 353 exam + 4000 wiki noise. Headline NL_QUERY_NO_AT_SCALE (GPT top1 0). acc_20way: fp-only beats chance (~4.4x); fp+sem at high alpha can harm fp — see results/stage261_close.md. G_not_lexical requires top1_low_overlap > 0.', 'timestamp': v460.v444(v461.v445).v360(), 'wall_s': v225.v225() - v107}
    if v108 and v105['name'] == 'baseline':
        v275 = v0 / 'stage261_decision_query_names_entity.json'
        v276 = v0 / 'stage261_mini_query_names_entity.md'
    elif v104.v94 and v105['name'] == 'baseline':
        v361 = v104.v94.v415('/', '_')
        v275 = v0 / f'stage261_decision_{v361}.json'
        v276 = v0 / f'stage261_mini_{v361}.md'
    elif not v91 and v105['name'] == 'baseline':
        v275 = v0 / 'stage261_decision_no_anchor.json'
        v276 = v0 / 'stage261_mini_no_anchor.md'
    else:
        v275 = v1 if v105['name'] == 'baseline' else v0 / f"stage261_decision_{v105['name']}.json"
        v276 = v2 if v105['name'] == 'baseline' else v0 / f"stage261_mini_{v105['name']}.md"
    v275.v223(v416.v362(v50, indent=2), encoding='utf-8')
    v276.v223(f"# Stage 261 natural-question retrieval ({v105['name']})\n\n**{v274}** slots={v254(v245)} eval={v254(v255)} chance={v136:.4f}\n\n- top1: fp-only **{v261['top1']:.3f}** -> fp+sem **{v262['top1']:.3f}** (shuffled {v267['top1']:.3f})\n- by overlap: low **{v262['top1_low_overlap']:.3f}** vs high **{v262['top1_high_overlap']:.3f}** (median {v77:.3f})\n- 20-way (chance 0.05): fp-only **{v261.v158('acc_20way', v14('nan')):.3f}** (init Wq {v263.v158('acc_20way', v14('nan')):.3f}) -> fp+sem **{v262.v158('acc_20way', v14('nan')):.3f}** (shuffled {v267.v158('acc_20way', v14('nan')):.3f})\n- mrr {v262['mrr']:.3f}, median rank {v262['median_rank']:.0f}, blend a {v262['alpha']:.3f}\n" + (f"- matched GPT-2: top1 {v132['top1']:.3f}\n" if v132 else '- matched GPT-2: not run\n'), encoding='utf-8')
    v230(v416.v362({'overall': v274, 'gates': v50['gates']}, indent=2))
    if not v104.v229:
        v6.v281.v156(exist_ok=True)
        v277 = {'W_q': v53.v401(), 'stage': 261, 'recipe': v105['name']}
        if v72 is not None:
            v277['sem'] = v72.v401()
        if v63 is not None:
            v277['rerank'] = v63.v401()
        if v69 is not None:
            v277['qkey'] = v69.v401()
        v36.v363(v277, v6)
    return 0
if v152 == '__main__':
    raise v278(v364())