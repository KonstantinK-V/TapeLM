"""Inprint slot-bias glue — inference API (stage 256). Trunk frozen; W_q + gate + copy mixture."""
from __future__ import annotations
import math
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, SelfModelXL
from _stage194_fp_fact_memory import FpBank
from _retrieval_modes import vote_scores
v0 = '{S} was appointed director of'
v1 = '{S} was appointed director of {V} in the regional chronicle of 1987 .'
v2 = v85.v13('\\b([A-Z][a-z]{2,})\\b')
from _tape_index import VOTES_AUTO_MIN_SLOTS, DEFAULT_RETRIEVE_TOPK
v3 = 'auto'
v4 = v11('checkpoints/stage256_slot_bias.pt')
v5 = v11('checkpoints/stage253_joint_l02.pt')
v6 = v11('checkpoints/stage191_p1_curve.pt')
from _tape_index import context_words

class RetrieveStats:
    """Counts glue retrieval steps by effective backend (decode/train diagnostics)."""
    v14 = ('votes', 'cosine', 'miss')

    def __init__(v86) -> None:
        v86.v87 = 0
        v86.v88 = 0
        v86.v89 = 0

    def record(v86, v32: v7, v41: v207 | None) -> None:
        if v32 == 'votes':
            v86.v87 += 1
        elif v32 == 'cosine':
            v86.v88 += 1
        if v41 is None:
            v86.v89 += 1

    def to_dict(v86) -> v8:
        v90 = v86.v87 + v86.v88
        return {'votes_steps': v86.v87, 'cosine_steps': v86.v88, 'miss': v86.v89, 'total_glue_steps': v90}

def slot_query_words(v15: v7, v16: v7 | None=None) -> v17[v7]:
    return v91(v15, exclude=v16)

class SlotPostings:
    """Word -> slot postings with idf weights (zero-train retrieval index)."""

    def __init__(v86, v92: v17[v17[v7]], v47: v55.v47):
        v86.v93: v8[v7, v17[v43]] = v160(v17)
        for v120, v161 in v130(v92):
            for v48 in v161:
                v86.v93[v48].v169(v120)
        v86.v94 = {v48: 1.0 / v238.v134(2.0 + v162(v100)) for v48, v100 in v86.v93.v227()}
        v86.v47 = v47
        v86.v95 = v162(v92)

    @v18
    def from_ctxw(v96, v92: v17[v17[v7]], v47: v55.v47) -> v19:
        return v96(v92, v47)

    def topk(v86, v97: v17[v7], v38: v43, v98: v55.v10 | None=None):
        v99 = v163(v97, v86.v93, v86.v94)
        if v98 is not None:
            v99 = {v120: v100 for v120, v100 in v99.v227() if v120 < v98.v249() and v12(v98[v120])}
        if not v99:
            return None
        v45 = v208(v99, key=lambda v120: -v99[v120])[:v38]
        v100 = v55.v164([v99[v120] for v120 in v45], dtype=v55.v209, device=v86.v47)
        v100 = v100 / v100.v213().v210(1e-06)
        return (v100, v55.v164(v45, dtype=v55.v222, device=v86.v47))

def resolve_retrieve_mode(v20: v7, v21: v43) -> v7:
    if v20 != 'auto':
        return v20
    return 'votes' if v21 >= v165 else 'cosine'

@v55.v35()
def full_bank_cue_summary(v22: v7, v23: v72 | None, v24: v101, v25: v102, v26: v103, v27: v17[v8], v28: v43, *, v29: v7=v0) -> v8:
    """Gold slot rank over all live tape slots at the decode cue (open bank)."""
    v30: v17[v43] = []
    v31 = v43(v26.v98.v211()) if v26.v98 is not None else v162(v26.v67)
    v32 = v104(v22, v31)
    for v33 in v27:
        v37 = [v153 for v153 in v25.v235(v29.v200(S=v33['S'])).v58 if v153 != v28]
        v84 = [v120 for v120, v100 in v130(v26.v67) if v100 == v33['value']]
        if not v84:
            continue
        v105 = v32 == 'votes' and v26.v93 is not None
        if v105:
            v97 = v212(v25.v139(v37))
            v99 = v163(v97, v26.v93.v93, v26.v93.v94)
            v166 = v213((v99.v239(v120, 0.0) for v120 in v84), default=0.0)
            v167 = 1 + v211((1 for v100 in v99.v67() if v100 > v166))
        else:
            if v23 is None:
                continue
            v40 = v107(v23, v24, v25, v37, anchor_ids=v37)
            if v40 is None:
                continue
            v44 = v26.v118 @ v40
            if v26.v98 is not None:
                v44 = v44.v186(~v26.v98, v159('-inf'))
            v168 = v159(v44[v84].v213())
            v167 = 1 + v43((v44 > v168).v211().v240())
        v30.v169(v167)
    if not v30:
        return {'full_bank_top1': v159('nan'), 'full_bank_mrr': v159('nan'), 'full_bank_median_rank': v159('nan'), 'n': 0}
    v34 = v170.v106(v30, dtype=v170.v171)
    return {'full_bank_top1': v159(v170.v214(v34 == 1)), 'full_bank_mrr': v159(v170.v214(1.0 / v34)), 'full_bank_median_rank': v159(v170.v215(v34)), 'n': v162(v30)}

def retrieve_topk(v20: v7, v23: v72 | None, v24: v101, v25: v102, v26: v103, v36: v17[v43], v37: v17[v43] | None, v38: v43, v39: v172 | None=None):
    """Unified retrieval: auto picks votes at scale, cosine on small banks."""
    v31 = v43(v26.v98.v211()) if v26.v98 is not None else v162(v26.v67)
    v32 = v104(v20, v31)
    if v32 == 'votes':
        if v26.v93 is None:
            v32 = 'cosine'
        else:
            v97 = v212(v25.v139(v36[-60:]))
            v41 = v26.v93.v173(v97, v38, v26.v98)
            if v39 is not None:
                v39.v174('votes', v41)
            return v41
    if v23 is None:
        if v39 is not None:
            v39.v174('cosine', None)
        return None
    v40 = v107(v23, v24, v25, v36, anchor_ids=v37)
    v41 = v26.v173(v40, v38) if v40 is not None else None
    if v39 is not None:
        v39.v174('cosine', v41)
    return v41

class SlotBias(v42.v9):
    """Retrieved slots -> copy distribution, mixed into LM logits by a per-step gate."""

    def __init__(v86, v108: v43, v47):
        v228().v175()
        v86.v109 = v216.v176(v47)
        v86.v110 = v42.v229(v42.v241(v108 + 4, 64), v42.v242(), v42.v241(64, 1)).v177(v47)
        v42.v217.v178(v86.v110[-1].v179)
        v42.v217.v180(v86.v110[-1].v181, -2.0)
        v86.v111 = v42.v182(v55.v164(-1.5, device=v47))

    def trainable(v86):
        return v17(v86.v109.v243()) + v17(v86.v110.v243()) + [v86.v111]

    def weights(v86, v44: v55.v10) -> v55.v10:
        return v197.v133(v44 / v55.v247(v86.v111).v230(0.001, 10.0), dim=-1)

    def g(v86, v112, v113: v159, v114: v159, v115: v159, v116) -> v55.v10:
        v50 = v116 if v55.v218(v116) else v55.v164(v116, device=v112.v47)
        v117 = v55.v183([v55.v164(v113, device=v112.v47, dtype=v112.v194), v55.v164(v114, device=v112.v47, dtype=v112.v194), v55.v164(v115, device=v112.v47, dtype=v112.v194), v50.v177(v112.v194).v220(())])
        return v55.v231(v86.v110(v55.v138([v112, v117], dim=-1))).v184(-1)

class TapeView:
    """Read-only slot bank for glue decode."""

    def __init__(v86, v118: v55.v10, v67: v17[v7], v25: v102, v28: v43, v92: v17[v17[v7]] | None=None):
        v86.v118 = v118
        v86.v67 = v67
        v86.v119 = [[v153 for v153 in v25.v235(' ' + v100).v58 if v153 != v28] for v100 in v67]
        v86.v98 = v55.v185(v162(v67), dtype=v55.v12, device=v118.v47)
        v86.v92 = [v17(v48) for v48 in v92] if v92 is not None else None
        v86.v93 = v19.v190(v86.v92, v118.v47) if v86.v92 else None

    def n_live(v86) -> v43:
        return v43(v86.v98.v211())

    def topk(v86, v40: v55.v10, v38: v43):
        if v40 is None or not v12(v86.v98.v244()):
            return None
        v44 = v86.v118 @ v40
        v44 = v44.v186(~v86.v98, -10000.0)
        v38 = v187(v38, v43(v86.v98.v211()))
        v100, v45 = v55.v173(v44, v38)
        return (v100, v45)

    def copy(v86) -> 'TapeView':
        v90 = v103.v188(v103)
        v90.v118, v90.v67, v90.v119 = (v86.v118, v86.v67, v86.v119)
        v90.v98 = v86.v98.v189()
        v90.v92 = [v17(v48) for v48 in v86.v92] if v86.v92 is not None else None
        v90.v93 = v19.v190(v90.v92, v86.v118.v47) if v90.v92 is not None else None
        return v90

    def reindex(v86, v120: v43, v121: v17[v7]) -> None:
        """Replace slot j's write-context words and rebuild postings."""
        if v86.v92 is None:
            raise v219('tape has no ctxw/postings index')
        if v120 < 0 or v120 >= v162(v86.v92):
            raise v219(f'slot index {v120} out of range')
        v86.v92 = [v17(v48) for v48 in v86.v92]
        v86.v92[v120] = v17(v121)
        v86.v93 = v19.v190(v86.v92, v86.v118.v47)

    def drop_value(v86, v122: v7) -> v43:
        v95 = 0
        for v120, v100 in v130(v86.v67):
            if v100 == v122 and v86.v98[v120]:
                v86.v98[v120] = False
                v95 += 1
        return v95

    def shuffled(v86, v123: v43) -> 'TapeView':
        v90 = v86.v191()
        v52 = v55.v232(device='cpu').v192(v123)
        v124 = v55.v233(v86.v118.v245(0), generator=v52).v177(v86.v118.v47)
        v90.v118 = v86.v118[v124]
        return v90

    def emptied(v86) -> 'TapeView':
        v90 = v86.v191()
        v90.v98 = v55.v193(v90.v98)
        return v90

    def with_value(v86, v125: v7, v126: v7, v25: v102, v28: v43, *, v121: v17[v7] | None=None) -> 'TapeView':
        """Update a fact: same slot, same KEY, new value — zero gradient steps.

        Keys are written as norm(fp(anchor) + ctx_fp(sentence, exclude=value)), so the value
        never enters its own key; replacing it leaves the key bit-identical and this stays a
        fact update rather than a re-index. Context changes must use ``reindex()``, not this
        method — silent postings drift is not allowed.
        """
        if v121 is not None:
            raise v219('write-context change requires TapeView.reindex(j, new_ctx_words); with_value only replaces the value string')
        v90 = v86.v191()
        v90.v67 = v17(v86.v67)
        v90.v119 = v17(v86.v119)
        v58 = [v153 for v153 in v25.v235(' ' + v126).v58 if v153 != v28]
        for v120, v100 in v130(v86.v67):
            if v100 == v125:
                v90.v67[v120] = v126
                v90.v119[v120] = v58
        return v90

def copy_dist(v23, v26, v44, v45, v36, v46, v47):
    v48 = v23.v127(v44)
    v49 = v55.v128(v46, device=v47, dtype=v48.v194)
    v50 = v55.v128((), device=v47, dtype=v48.v194)
    for v129, v120 in v130(v45.v195()):
        v58 = v26.v119[v120]
        if not v58:
            continue
        v131 = 0
        for v132 in v154(v187(v162(v58), v162(v36)), 0, -1):
            if v36[-v132:] == v58[:v132]:
                v131 = v132
                break
        if v131 >= v162(v58):
            continue
        v49 = v49.v196(0, v55.v164([v58[v131]], device=v47), v48[v129].v220(1))
        v50 = v50 + v48[v129]
    if v159(v50) > 1e-06:
        v49 = v49 / v50
    return (v49, v50)

def mix_logprob(v51: v55.v10, v52: v55.v10, v53: v55.v10, v50) -> v55.v10:
    v54 = v197.v133(v51, dim=-1)
    if v53 is None or v159(v50) <= 1e-06:
        return v55.v134(v54 + 1e-09)
    return v55.v134((1.0 - v52) * v54 + v52 * v53 + 1e-09)

def hidden_and_logits(v56: v135, v57, v58: v55.v10, v28: v43):
    v59 = v58 == v28
    v60 = v56.v136(v57[v58], v58)
    v61 = v56.v61(v60, pad_mask=v59)
    v137, v80, v80 = v56.v137(v60, v59)
    v62 = v55.v138([v61, v137], dim=-1)
    return (v62, v56.v198(v62))

def raw_query(v24: v101, v25: v102, v58: v17[v43], v63: v17[v43] | None=None):
    v15 = v25.v139(v58[-40:])
    v64 = v24.v140(v15)
    if v64 is None:
        return None
    v65 = v2.v141(v25.v139(v63) if v63 is not None else v15)
    if v65:
        v64 = v197.v199(v24.v221([v65[-1]])[0] + v64, dim=-1)
    return v64

def ctx_query(v23, v24, v25, v58, v63=None):
    v40 = v142(v24, v25, v58, v63)
    if v40 is None:
        return None
    return v197.v199(v23.v109(v40.v234(0)), dim=-1)[0]

def build_planted_keys(v24: v101, v66: v17[v7], v67: v17[v7], v68: v7=v1):
    v143, v144, v145 = ([], [], [])
    for v146, v147 in v148(v66, v67):
        v149 = v68.v200(S=v146, V=v147)
        v150 = v24.v221([v146])[0]
        v64 = v24.v140(v149, exclude=v147)
        if v64 is None:
            continue
        v143.v169(v197.v199(v150 + v64, dim=-1))
        v144.v169(v146)
        v145.v169(v147)
    if not v143:
        return (v55.v128(0, 256), [], [])
    return (v55.v183(v143, 0), v144, v145)

def load_glue(v56: v135, v47, v69: v11 | None=None) -> v72 | None:
    v49 = v69 or v4
    if not v49.v201():
        return None
    v70 = 2 * (v56.v198.v202 // 2)
    v23 = v72(v70, v47)
    v71 = v55.v151(v49, map_location=v47, weights_only=False)
    with v55.v35():
        v23.v109.v203(v71['W_q'])
        v23.v110.v203(v71['gate'])
        v23.v111.v204(v71['log_tau'].v177(v47))
    v23.v152()
    return v23

def trunk_ckpt_path() -> v11:
    return v5 if v5.v201() else v6

@v55.v35()
def free_decode_value(v23, v56, v57, v25, v24, v26: v103, v73: v7, v28: v43, v74: v43, v47, *, v75: v7=v0, v38: v43=8, v76: v43=6, v77: v12=True) -> v82[v7, v159]:
    v37 = [v153 for v153 in v25.v235(v75.v200(S=v73)).v58 if v153 != v28]
    v78 = v17(v37)
    v79 = []
    for v80 in v154(v76):
        v58 = v55.v164([v78[-v248:]], dtype=v55.v222, device=v47)
        v62, v205 = v206(v56, v57, v58, v28)
        v155 = v205[0, -1]
        v156 = v55.v134(v197.v133(v155, -1) + 1e-09)
        if v77 and v23 is not None:
            v41 = v223(v3, v23, v24, v25, v26, v78, v37, v38)
            if v41 is not None:
                v44, v45 = v41
                v224 = v159(-(v197.v133(v155, -1) * v197.v250(v155, -1)).v211())
                v53, v50 = v236(v23, v26, v44, v45, v78, v74, v47)
                v225 = v23.v52(v62[0, -1], v159(v44.v213()), v159(v44.v214()), v224, v50)
                v156 = v237(v155, v225, v53, v50)
                v79.v169(v159(v225))
        v157 = v43(v156.v226())
        v78.v169(v157)
    v15 = v25.v139(v78[v162(v37):]).v158()
    v81 = v159(v211(v79) / v162(v79)) if v79 else v159('nan')
    return (v15, v81)

def value_exact_match(v83: v7, v84: v7) -> v12:
    if not v83:
        return False
    return v83.v158().v246(' ')[0].v158(' .,;:') == v84