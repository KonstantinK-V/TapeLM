"""
Stage 180 — Dual-channel curve: fast ink + slow instance.

Architecture (not just loss):
  FAST  — local ink dynamics over BPE arcs (can follow the suffix).
  SLOW  — gated cumulative write budget (write-once-ish): early arcs consume
          capacity; suffix cannot fully rewind the slow state.

Readout for gates = concat(fast_last, slow_last) (and slow-only probe).

Losses:
  - fast: weak next + far (ink)
  - slow: past-bag from slow state; recover random instance written only early
  - retention on slow endpoints (same last piece, different prefixes)
  - light combine consistency

NO text CE. Reuses Stage177 ByteLevel BPE.

  python _stage180_dual_channel.py
  python _stage180_dual_channel.py --steps 10000
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage178_curve_retention as s178
import _stage179_curve_harden_B as s179
v0 = v26('results')
v1 = v26('checkpoints')
v2 = v0 / '_stage180_log.txt'
v3 = v0 / 'stage180_decision.json'
v4 = v0 / 'stage180_mini.md'
v5 = v1 / 'stage180_dual_channel.pt'
v6 = v27.v6
v7 = v0 / 'plan_curve_dynamics.md'
v8 = 180
v9 = 128
v10 = 128
v11 = v27.v11
v12 = 12
v13 = 6
v14 = 4
v15 = 0.0003
v16 = 1500
v17 = 10000
v18 = 0.4
v19 = 1.0
v20 = 1.0
v21 = 1.2
v22 = 0.1

def log(v28: v81) -> None:
    v29 = v28 if v28.v163('\n') else v28 + '\n'
    try:
        v164(v29, end='', flush=True)
    except v82:
        v164(v29.v255('ascii', 'replace').v236('ascii'), end='', flush=True)
    v2.v165.v83(parents=True, exist_ok=True)
    with v2.v166('a', encoding='utf-8') as v84:
        v84.v90(v29)

def write_json(v30: v26, v31: v85) -> None:
    v30.v165.v83(parents=True, exist_ok=True)
    v30.v86(v219.v167(v31, indent=2, ensure_ascii=False), encoding='utf-8')

class SlowWriter(v32.v23):
    """Cumulative write-budget memory: early writes spend capacity; late can't overwrite."""

    def __init__(v87, v88: v25, v89: v25=v10):
        v237().v168()
        v87.v90 = v32.v169(v32.v170(v88 + v89, v89), v32.v220(), v32.v170(v89, v89))
        v87.v91 = v32.v170(v88 + v89, 1)
        v87.v92 = v32.v171(v89)
        v87.v93 = v32.v172(v36.v221(v89))

    def forward(v87, v94: v36.v24, v34: v36.v24) -> v36.v24:
        v42, v48, v126 = v94.v95
        v41 = v94.v41
        v96 = v87.v93.v223(0).v238(v42, -1).v173()
        v97 = v36.v174(v42, 1, device=v41)
        v98 = []
        for v51 in v114(v48):
            v175 = v94[:, v51]
            v176 = v36.v222([v175, v96], dim=-1)
            v177 = v36.v239(v87.v91(v176)) * v97
            v178 = v87.v90(v176)
            v179 = (~v34[:, v51]).v121().v223(-1)
            v177 = v177 * v179
            v96 = v87.v92(v96 + v177 * v178)
            v97 = (v97 - v177).v224(min=0.0)
            v98.v120(v96)
        return v36.v180(v98, dim=1)

class DualChannel(v32.v23):

    def __init__(v87, v99: v25):
        v237().v168()
        v87.v100 = v27.v181(v99, d=v9)
        v87.v101 = v27.v182(d=v9)
        v87.v96 = v183(v9, v10)
        v87.v102 = v32.v169(v32.v170(v9, v9), v32.v220(), v32.v170(v9, v9))
        v87.v103 = v32.v169(v32.v170(v9, v9), v32.v220(), v32.v170(v9, v9))
        v87.v104 = v32.v169(v32.v170(v10, v9), v32.v220(), v32.v170(v9, v9))
        v87.v105 = v32.v169(v32.v170(v10, v10), v32.v220(), v32.v170(v10, v10))

    def encode_arcs(v87, v40: v36.v24) -> v36.v24:
        return v87.v100(v40)

    def forward_channels(v87, v40: v36.v24, v34: v36.v24, v106: v36.v24 | None=None):
        """
        inst_prefix: [B,D] added only into early arc embeddings (handwriting/instance cue),
        slow must carry it; fast may ignore.
        """
        v94 = v87.v184(v40)
        if v106 is not None:
            v42, v48, v126 = v94.v95
            v185 = (~v34).v190(dim=1).v224(min=2)
            v186 = (v185.v121() * 0.5).v187().v224(min=1)
            v94 = v94.v173()
            for v117 in v114(v42):
                v94[v117, :v25(v186[v117])] = v94[v117, :v25(v186[v117])] + 0.4 * v106[v117]
        v101 = v87.v101(v94, pad_mask=v34)
        v96 = v87.v96(v94, v34)
        return (v94, v101, v96)

def last_state(v33: v36.v24, v34: v36.v24) -> v36.v24:
    v35 = (~v34).v190(dim=1).v224(min=1).v187() - 1
    return v33[v36.v225(v33.v109(0), device=v33.v41), v35]

def cos_match(v37, v38) -> v36.v24:
    return (1.0 - v188.v131(v37, v38.v194(), dim=-1)).v107()

def train_step_loss(v39: v108, v40, v34, v41):
    v42 = v40.v109(0)
    v43 = v188.v110(v36.v189(v42, v10, device=v41), dim=-1)
    v44 = v43 if v10 == v9 else v43[:, :v9]
    v94, v101, v96 = v39.v111(v40, v34, inst_prefix=v44)
    v45 = {}
    v46 = []
    v47 = ~v34[:, :-1] & ~v34[:, 1:]
    if v47.v190() > 0:
        v112 = v39.v102(v101[:, :-1])
        v46.v120(v18 * v226(v112[v47], v94[:, 1:][v47]))
        v45['cos_next'] = v121(v188.v131(v112[v47], v94[:, 1:][v47].v194(), dim=-1).v107())
    if v94.v109(1) > v14 + 1:
        v113 = ~v34[:, :-v14] & ~v34[:, v14:]
        if v113.v190() > 0:
            v191 = v39.v103(v101[:, :-v14])
            v46.v120(v18 * v226(v191[v113], v94[:, v14:][v113]))
            v45['cos_far'] = v121(v188.v131(v191[v113], v94[:, v14:][v113].v194(), dim=-1).v107())
    v48 = v94.v109(1)
    v49 = []
    v50 = []
    for v51 in v114(3, v48):
        v115 = ~v34[:, v51]
        if v115.v190() < 1:
            continue
        v116 = []
        for v117 in v114(v42):
            if v34[v117, v51]:
                v116.v120(v36.v221(v9, device=v41))
                continue
            v192 = v94[v117, :v51]
            v193 = ~v34[v117, :v51]
            v116.v120(v192[v193].v107(0) if v193.v250() else v36.v221(v9, device=v41))
        v118 = v36.v180(v116, 0)
        v37 = v39.v104(v96[:, v51])
        v49.v120(v226(v37[v115], v118[v115]))
        v50.v120(v121(v188.v131(v37[v115], v118[v115].v194(), dim=-1).v107()))
    if v49:
        v46.v120(v19 * v36.v180(v49).v107())
        v45['cos_past_slow'] = v121(v240.v107(v50))
    v52 = v119(v96, v34)
    v53 = v39.v105(v52)
    v46.v120(v20 * v226(v53, v43))
    v45['cos_inst_slow'] = v121(v188.v131(v53, v43, dim=-1).v107())
    v54 = v119(v101, v34)
    if v52.v109(-1) == v54.v109(-1):
        v57 = v188.v131(v52, v54.v194(), dim=-1).v241().v107()
        v46.v120(v22 * v57)
        v45['slow_fast_sim'] = v121(v57.v194())
    v55 = v190(v46) if v46 else v101.v190() * 0.0
    v45['loss'] = v121(v55.v194())
    return (v55, v45, v101, v96)

def retention_slow(v39: v108, v56):
    v122, v123, v124, v125 = v56
    v126, v126, v127 = v39.v111(v122, v124, inst_prefix=None)
    v126, v126, v128 = v39.v111(v123, v125, inst_prefix=None)
    v129, v130 = (v119(v127, v124), v119(v128, v125))
    v57 = v188.v131(v129, v130, dim=-1)
    v58 = v188.v242(v57 - 0.4).v107() + 0.2 * v57.v107()
    return (v21 * v58, {'ret_slow_cos': v121(v57.v107().v194())})

class DualGateWrap(v32.v23):
    """Expose combined (or slow) states for A/B gates expecting forward_states(char_ids)."""

    def __init__(v87, v132: v108, v63: v81='combined'):
        v237().v168()
        v87.v132 = v132
        v87.v63 = v63

    def forward_states(v87, v40, v133=None):
        if v133 is None:
            v133 = v36.v221(v40.v109(0), v40.v109(1), dtype=v36.v243, device=v40.v41)
        v94, v101, v96 = v87.v132.v111(v40, v133, inst_prefix=None)
        if v87.v63 == 'slow':
            return v96
        if v87.v63 == 'fast':
            return v101
        return 0.5 * v101 + 0.5 * v96

def gate_A_modes(v39, v59, v60, v41, v61):
    v62 = {}
    for v63 in ('combined', 'slow', 'fast'):
        v134 = v244(v39, mode=v63).v147(v41)
        v62[v63] = v27.v195(v134, v59, v60, v41, v61, n_pairs=60)
    return v62

def gate_B_modes(v39, v64, v60, v41, v61):
    v62 = {}
    for v63 in ('combined', 'slow', 'fast'):
        v134 = v244(v39, mode=v63).v147(v41)
        v62[v63] = v227.v196(v134, v64, v60, v41, v61)
    return v62

def main() -> v25:
    v65 = v197.v135()
    v65.v136('--steps', type=v25, default=v17)
    v65.v136('--device', default='cuda' if v36.v251.v245() else 'cpu')
    v66 = v65.v137()
    v0.v83(parents=True, exist_ok=True)
    v1.v83(parents=True, exist_ok=True)
    v2.v86('', encoding='utf-8')
    v138(f'Stage180 start {v253.v248(v254.v249).v216()}')
    v138('Dual-channel: FAST ink Transformer + SLOW write-budget memory')
    v138(f'plan={v7}')
    if not v6.v198():
        raise v199(v6)
    v64 = v200.v139(v81(v6))
    v67 = v201.v140(max_chars=20000000)
    v68 = v141(v228(v67) | {' '})
    v69 = ['<pad>'] + v68
    v60 = {v142: v202 + 1 for v202, v142 in v229(v68)}
    v59 = v27.v143(v64, v67)
    v70 = v59[v25(0.8 * v246(v59)):] or v59[-100:]
    v71 = v59[:v25(0.8 * v246(v59))] or v59
    v72 = v203.v144(v71)
    v138(f'docs={v246(v59)} same-last={v246(v72)} V={v64.v247()} d={v9}')
    v41 = v36.v41(v66.v41)
    v36.v145(v8)
    v204.v146(v8)
    v39 = v108(v246(v69)).v147(v41)
    v73 = v36.v205.v148(v39.v206(), lr=v15, weight_decay=0.0001)
    v61 = v204.v149(v8)
    v39.v150()
    v74 = v151(v39, v70, v60, v41, v204.v149(v8))
    v75 = v152(v39, v64, v60, v41, v204.v149(v8 + 1))
    for v63 in ('combined', 'slow', 'fast'):
        v207, v117 = (v74[v63], v75[v63])
        v138(f"  init [{v63}] A_same={v207['mean_cos_same_last_piece']:.3f}→{v207['verdict']} | B para={v117['mean_cos_paraphrase']:.3f} hard={v117['mean_cos_hard_spelling']:.3f}→{v117['verdict']}")
    v153, v154 = (v74, v75)
    v76 = None
    v39.v71()
    for v77 in v114(1, v66.v208 + 1):
        v175, v34 = v27.v209(v71, v60, v12, v61, v41)
        v55, v210, v126, v126 = v211(v39, v175, v34, v41)
        v155 = v203.v212(v72, v60, v13, v61, v41)
        if v155 is not None:
            v230, v231 = v232(v39, v155)
            v55 = v55 + v230
            v210.v233(v231)
        v73.v213(set_to_none=True)
        v55.v214()
        v32.v234.v215(v39.v206(), 1.0)
        v73.v77()
        v76 = v210['loss'] if v76 is None else 0.95 * v76 + 0.05 * v210['loss']
        if v77 % v16 == 0 or v77 == v66.v208:
            v39.v150()
            v153 = v151(v39, v70, v60, v41, v204.v149(v8 + v77))
            v154 = v152(v39, v64, v60, v41, v204.v149(v8 + v77 + 3))
            v156, v157 = (v153['combined'], v154['combined'])
            v158, v159 = (v153['slow'], v154['slow'])
            v138(f"  step {v77}: loss~{v76:.3f} inst={v210.v256('cos_inst_slow', 0):.3f} past={v210.v256('cos_past_slow', 0):.3f} ret={v210.v256('ret_slow_cos', 0):.3f} sf_sim={v210.v256('slow_fast_sim', 0):.3f} | A_comb={v156['mean_cos_same_last_piece']:.3f} A_slow={v158['mean_cos_same_last_piece']:.3f} | B_comb {v157['verdict']} para={v157['mean_cos_paraphrase']:.3f} hard={v157['mean_cos_hard_spelling']:.3f} | B_slow {v159['verdict']}")
            v39.v71()
            v36.v235({'model': v39.v252(), 'stoi': v60, 'step': v77, 'A': v153, 'B': v154}, v5)
    v156, v157 = (v153['combined'], v154['combined'])
    v158, v159 = (v153['slow'], v154['slow'])
    v78 = v156['mean_cos_same_last_piece'] < 0.9
    if v78 and 'PASS' in v157['verdict']:
        v160 = 'DUAL_A_YES_B_YES'
    elif v78 and 'WEAK' in v157['verdict']:
        v160 = 'DUAL_A_YES_B_WEAK'
    elif v78:
        v160 = 'DUAL_A_YES_B_FAIL'
    else:
        v160 = 'DUAL_A_FAIL'
    v79 = {'slow_A_same': v158['mean_cos_same_last_piece'], 'fast_A_same': v153['fast']['mean_cos_same_last_piece'], 'slow_better_than_fast_for_A': v158['mean_cos_same_last_piece'] + 0.05 < v153['fast']['mean_cos_same_last_piece']}
    v62 = {'timestamp': v253.v248(v254.v249).v216(), 'protocol': 'dual_channel_180', 'overall': v160, 'architecture': {'fast': 'causal Transformer over BPE arc ink', 'slow': "cumulative write-budget memory (early spends, late can't fully overwrite)", 'readout': 'combined = 0.5 fast + 0.5 slow; also report slow/fast alone'}, 'arch_check': v79, 'final_A': v153, 'final_B': v154, 'init_A': v74, 'init_B': v75, 'note': 'Architectural anti-coil: slow channel write-once-ish. B may still need semantic pressure.', 'next': 'If slow A << fast A: architecture works for retention. If B still form>>meaning: put semantic/contrastive load on SLOW only. Do not soak fast next-local.'}
    v161(v3, v62)
    v4.v86('\n'.v217(['# Stage180 — dual channel', '', f'**Overall:** `{v160}`', '', f"- A combined same={v156['mean_cos_same_last_piece']:.3f}; slow={v158['mean_cos_same_last_piece']:.3f}; fast={v153['fast']['mean_cos_same_last_piece']:.3f}", f"- B combined: {v157['verdict']} para={v157['mean_cos_paraphrase']:.3f} hard={v157['mean_cos_hard_spelling']:.3f}", f"- arch: slow_better_A={v79['slow_better_than_fast_for_A']}", f"- {v62['next']}", '']), encoding='utf-8')
    v138(f'[180] {v160}')
    return 0
if v80 == '__main__':
    raise v162(v218())