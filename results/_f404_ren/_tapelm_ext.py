"""Shared helpers for TapeLM extension stages 214+ (zero-train policies, adapters).

Canonical memory + decode (227 / 228c):
  - Write slot keys in **canonical** fp (frozen P1 `arc_enc`).
  - Read under domain shift: **qmap** query with `W_bwd` (domain → canonical).
  - Official decode: **4-way slot retrieve** → score candidates with `cos(fp(c), fp(retrieved))`.
  Do **not** use global argmax retrieve + fp (228b anti-pattern).
"""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank
if v0:
    pass
v1 = v11('checkpoints/w_registry')
v2 = v1 / 'w_registry.json'
v3 = v4

class RecencyFpBank(v5):
    """ctx_fp with exponential decay by word index distance to entity (zero-train)."""

    def __init__(v100, v12, v101, v16, v102: v8=0.0, v103: v27=True):
        v191().v153(v12, v101, v16)
        v100.v102 = v102
        v100.v103 = v103

    @v30.v23()
    def ctx_fp(v100, v104: v10, v71: v10 | None=None) -> v30.v7 | None:
        if v100.v102 <= 0.0:
            return v191().v135(v104, exclude=v71)
        v105 = [v108 for v108 in v3.v192(v104) if v108 != v71]
        if v156(v105) < 3:
            return None
        v106 = None
        for v93 in v154(v156(v105) - 1, -1, -1):
            if v195.v193(v105[v93]) and v105[v93] != v71:
                v106 = v93
                break
        if v106 is None:
            v106 = v156(v105) - 1 if v100.v103 else 0
        v65 = v100.v51(v105)
        v107 = []
        for v93 in v154(v156(v105)):
            v49 = v151(v93 - v106)
            v107.v180(v187.v165(-v100.v102 * v49))
        v108 = v30.v109(v107, device=v65.v16, dtype=v65.v183)
        v108 = v108 / v108.v185().v184(min=1e-09)
        return v158.v114((v65 * v108.v161(-1)).v185(0), dim=-1)

@v30.v23()
def slow_endpoint_vec(v12, v13, v14, v15: v66[v89], v16: v30.v16) -> v30.v7 | None:
    """Last slow-channel state for a BPE token-id context window (frozen P1)."""
    from _stage191_night import MAX_ARCS
    v17 = v15[-v186:]
    if not v17:
        return None
    v18 = v30.v109([v17], dtype=v30.v155, device=v16)
    v19 = v18 == v14
    v20 = v12.v110(v13[v18], ids=v18)
    v111, v112, v112 = v12.v111(v20, v19)
    v21 = (~v19[0]).v113(as_tuple=False)
    if v156(v21) == 0:
        return None
    v22 = v89(v21[-1].v157())
    return v158.v114(v111[0, v22], dim=-1)

class DomainAdapter(v24.v6):
    """Learnable warp in fp-space: fp' = normalize(W @ fp)."""

    def __init__(v100, v49: v89=256):
        v191().v153()
        v100.v108 = v24.v159(v49, v49, bias=False)

    def forward(v100, v51: v30.v7) -> v30.v7:
        return v158.v114(v100.v108(v51), dim=-1)

    def map_raw(v100, v51: v30.v7) -> v30.v7:
        """Linear part before normalize (for slot keys)."""
        return v100.v108(v51)

class BottleneckRemap(v24.v6):
    """Tiny d -> r -> d remap (fewer params than full 256x256)."""

    def __init__(v100, v49: v89=256, v115: v89=32):
        v191().v153()
        v100.v116 = v24.v159(v49, v115, bias=False)
        v100.v117 = v24.v159(v115, v49, bias=False)

    def forward(v100, v51: v30.v7) -> v30.v7:
        return v158.v114(v100.v117(v100.v116(v51)), dim=-1)

    def map_raw(v100, v51: v30.v7) -> v30.v7:
        return v100.v117(v100.v116(v51))

class WFamilyPolicy:
    """
    L2 migration policy: Identity vs registry[family] vs learn-new.
    Thresholds seeded by stage 224 (knobs, not laws).
    """

    def __init__(v100, v38: v25[v10, v9] | None=None, v118: v8=0.85, v119: v8=0.65):
        v100.v38 = v25(v38 or {})
        v100.v118 = v118
        v100.v119 = v119

    def decide(v100, v120: v8, v121: v10 | None) -> v25:
        if v120 > v100.v118:
            return {'action': 'identity', 'family': v121, 'mean_cos': v120}
        if v121 and v121 in v100.v38:
            return {'action': 'use_registry', 'family': v121, 'mean_cos': v120}
        if v120 < v100.v119:
            return {'action': 'learn_family_W', 'family': v121 or 'outlier', 'mean_cos': v120}
        return {'action': 'learn_or_attach_family', 'family': v121 or 'prose', 'mean_cos': v120}

    def get(v100, v121: v10) -> v9 | None:
        return v100.v38.v128(v121)

    def set(v100, v121: v10, v50: v9) -> None:
        v100.v38[v121] = v50

    @v26
    def should_fork(v122: v8, v123: v8, v124: v8=0.05) -> v27:
        """True if existing family W is not good enough for a new corpus."""
        return v122 - v123 >= v124

def fp_bind(v28: v30.v7, v29: v30.v7) -> v30.v7:
    """Circular-convolution-style bind in fp-space (elementwise product, normalized)."""
    if v28.v160() == 1:
        v28 = v28.v161(0)
    if v29.v160() == 1:
        v29 = v29.v161(0)
    return v158.v114(v28 * v29, dim=-1)

def weighted_slot_sims(v31: v30.v7, v32: v66[v89], v33: v66[v10], v34: v10, v35: v8, v36: v8=0.25) -> v30.v7:
    """L3 read: age decay × penalty when slot `w_version` ≠ active registry era."""
    v37 = v31.v125()
    for v126, (v162, v163) in v127(v164(v32, v33)):
        v108 = v187.v165(-v162 / v182(v35, 1e-06))
        if v163 != v34:
            v108 *= v36
        v37[v126] = v37[v126] * v108
    return v37

def pick_w_bwd_for_era(v38: v25[v10, v9], v39: v10, v40: v10='prose_bwd') -> v9 | None:
    """Temporal W: map logical era label → persisted adapter key (e.g. prose_v2_bwd)."""
    v41 = f'{v39}_bwd' if not v39.v188('_bwd') else v39
    if v41 in v38:
        return v38[v41]
    return v38.v128(v40)

@v30.v23()
def mean_core_cos(v42: v5, v43: v5, v44: v66[v10]) -> v8:
    v45 = v42.v51(v44)
    v46 = v43.v51(v44)
    return v8((v45 * v46).v185(-1).v166())

def compose_w_bwd(v47: v9, v48: v9) -> v9:
    """Compose qmap adapters: ``normalize(W_outer @ W_inner @ q)`` (227 qmap chain)."""
    v49 = v47.v108.v167.v129[0]
    v50 = v9(v49).v130(v47.v108.v167.v16)
    with v30.v23():
        v50.v108.v167.v168(v47.v108.v167 @ v48.v108.v167)
    v50.v131()
    return v50

def lexicon_nearest(v51: v30.v7, v52: v30.v7) -> v30.v7:
    """Snap batch of fp vectors to nearest row in lex_fps."""
    v53 = (v51 @ v52.v189).v132(dim=-1)
    return v52[v53]

def apply_qmap(v54: v9, v55: v30.v7) -> v30.v7:
    """Map domain query fp → canonical key space (227 P_qmap)."""
    if v55.v160() == 1:
        return v158.v114(v54.v169(v55.v161(0)), dim=-1)[0]
    return v158.v114(v54.v169(v55), dim=-1)

def slot_retrieve_4way(v56: v30.v7, v57: v66[v10], v58: v30.v7, v59: v66[v10]) -> v10:
    """Among `candidates`, pick value with best max slot-key cosine to `qq` (227 exam protocol)."""
    v133, v61 = (-1.0, v59[0])
    for v60 in v59:
        v134 = [v93 for v93, v194 in v127(v57) if v194 == v60]
        if not v134:
            v170 = -1.0
        else:
            v170 = v8((v56[v134] @ v58).v182())
        if v170 > v133:
            v133, v61 = (v170, v60)
    return v61

def slot_retrieve_global(v56: v30.v7, v57: v66[v10], v58: v30.v7) -> v10:
    """Global argmax over all slots — 228b broken protocol; contrast only."""
    return v57[v89((v56 @ v58).v132())]

@v30.v23()
def fp_cos_scores(v62: v5, v63: v10, v59: v66[v10]) -> v66[v8]:
    """cos(fp(word), fp(c)) for each candidate c."""
    v64 = v62.v51([v63])[0]
    v65 = v62.v51(v59)
    return [v8((v65[v93] * v64).v185()) for v93 in v154(v156(v59))]

@v30.v23()
def fp_decode_pick_retrieved_4way(v67: v5, v68: v30.v7, v57: v66[v10], v54: v9, v69: v5, v70: v10, v71: v10 | None, v59: v66[v10]) -> v75[v10, v10]:
    """
    Official TapeLM memory decode API (228c).

    Returns (retrieved_slot_value, chosen_candidate).
    """
    v55 = v69.v135(v70, exclude=v71)
    if v55 is None:
        raise v171('ctx_fp returned None for decode context')
    v58 = v136(v54, v55)
    v72 = v137(v68, v57, v58, v59)
    v73 = v138(v67, v72, v59)
    v74 = v59[v89(v182(v154(v156(v73)), key=lambda v93: v73[v93]))]
    return (v72, v74)

def save_w_family(v76: v11, v50: v9, v77: v25) -> None:
    v76.v172.v139(parents=True, exist_ok=True)
    v78 = {'meta': v77, 'state_dict': v50.v173()}
    v30.v140(v78, v76)

def load_w_family(v76: v11, v16: v30.v16 | v10='cpu') -> v75[v9, v25]:
    v79 = v30.v141(v76, map_location=v16, weights_only=False)
    v50 = v9(256).v130(v16)
    v50.v142(v79['state_dict'])
    v50.v131()
    return (v50, v25(v79.v128('meta') or {}))

def load_w_registry(v80: v11 | None=None, v16: v30.v16 | v10='cpu') -> v75[v25[v10, v9], v25]:
    """
    Load manifest + all family adapters listed in `w_registry.json`.

    Expects keys like `prose_bwd`, `code_bwd` (qmap) and optional `*_fwd` (keylift).
    """
    v81 = v80 or v1
    v82 = v81 / 'w_registry.json'
    if not v82.v174():
        raise v175(f'Missing {v82}; run artifact/scripts/export_w_registry.py')
    v83 = v176.v143(v82.v177(encoding='utf-8'))
    v84: v25[v10, v9] = {}
    for v121, v144 in v83.v128('families', {}).v145():
        for v178, v179 in v144.v128('files', {}).v145():
            v41 = f'{v121}_{v178}'
            v76 = v81 / v179
            v50, v112 = v190(v76, v16)
            v84[v41] = v50
    return (v84, v83)

@v90(frozen=True)
class AnnotatedSlotHit:
    v85: v10
    v86: v8
    v87: v10
    v88: v89

def subject_slot_hits(v56: v30.v7, v57: v66[v10], v58: v30.v7, v91: v66[v89], v77: v66[v25]) -> v66[v94]:
    """Cosine scores for a subject's slot rows, highest first."""
    v92: v66[v94] = []
    for v93 in v91:
        v146 = v77[v93]
        v92.v180(v94(value=v57[v93], score=v8(v56[v93] @ v58), provenance=v10(v146.v128('provenance', 'unknown')), year=v89(v146.v128('year', 0))))
    v92.v147(key=lambda v152: v152.v86, reverse=True)
    return v92

def _query_wants_revision(v95: v10) -> v27 | None:
    v55 = v95.v148()
    if v181.v149('\\b1999\\b|revision|later claim|updated record', v55):
        return True
    if v181.v149('\\b1987\\b|original record|official records|as filed', v55):
        return False
    return None

def resolve_slot_contradiction(v92: v66[v94], v95: v10, v96: v10='composite', v97: v8=0.05) -> v10:
    """
    Pick one value among contradictory slot hits (229 upper layer).

    Policies: ``argmax``, ``recency``, ``query_cue``, ``composite`` (default).
    """
    if not v92:
        raise v171('no slot hits to resolve')
    if v156(v92) == 1:
        return v92[0].v85
    if v96 == 'argmax':
        return v92[0].v85
    if v96 == 'recency':
        return v182(v92, key=lambda v152: v152.v88).v85
    v98 = v150(v95)
    if v96 == 'query_cue':
        if v98 is True:
            for v152 in v92:
                if v152.v87 == 'revision':
                    return v152.v85
        if v98 is False:
            for v152 in v92:
                if v152.v87 == 'official':
                    return v152.v85
        return v182(v92, key=lambda v152: v152.v88).v85
    v99 = v151(v92[0].v86 - v92[1].v86)
    if v99 >= v97:
        return v92[0].v85
    if v98 is True:
        for v152 in v92:
            if v152.v87 == 'revision':
                return v152.v85
    if v98 is False:
        for v152 in v92:
            if v152.v87 == 'official':
                return v152.v85
    return v182(v92, key=lambda v152: v152.v88).v85