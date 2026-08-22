"""
Stage 289c — the unprompted audit: the mind publishes where it is competent, before being asked.

Every stage so far answers a question someone posed. This one asks the mind to sweep its own
tape and say, unprompted, which regions of it it can be trusted on. That is not a new
capability bolted on - it is the only honest use of the confidence the earlier stages already
produce, and it is what turns a set of gate booleans into something a person can act on.

The claim is narrow and checkable: **the mind's stated confidence must be an empirical
frequency.** If it publishes "0.8 on this region", it must be right about 80% of the time
there. That is calibration, and it is measurable without any authored label, because the truth
of every question in the sweep is already free from the tape.

What makes this an audit rather than a report card:

  - The regions are not hand-drawn. They are cut by properties the mind can see WITHOUT the
    answer: how many mentions the address has, and whether they agree. Cutting by anything that
    needs the truth would make the map a restatement of the score. Rare context words and
    sibling relations are two more honest cuts and are deliberately held back - every extra cut
    splits the denominators, and a region under MIN_ANSWERED states nothing.
  - The mind ranks regions by its own confidence and the examiner checks that ranking against
    realised accuracy. A map that is right on average but orders the regions wrongly is
    useless: you would trust the wrong half of your own tape.
  - Refusal is a legitimate cell. A region the mind declines is scored on whether declining was
    right - measured as the accuracy it WOULD have had there, which the tape knows.

Gates:
  G_calibrated             stated confidence tracks realised accuracy (ECE under a bound that
                           is derived from the bin counts, not chosen)
  G_ranks_regions          Spearman between stated confidence and realised accuracy > 0
  G_refusal_is_informed    the regions it declines really are the ones it would fail on
  G_map_transfers          the ordering learned on the train tape survives on the held-out tape

  python _stage289c_audit.py --smoke
  python _stage289c_audit.py --train-steps 6000
  python _stage289c_audit.py --train-steps 6000 --holdout address
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage271_controller as s271
import _stage279_write_decision as s279
import _stage280_raw_exam as s280
import _stage286_evidence as s286
import _stage289_derivation as s289
import _stage289a_presupposition as s289a
from _tape_speed import CachedBank, install_assertion_cache, install_fast_fp_addresses
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('data/_wikitext103_train.txt')
v3 = 2893
v4 = v0 / '_stage289c_log.txt'

def log(v9: v5) -> None:
    v10 = v9 if v9.v172('\n') else v9 + '\n'
    try:
        v173(v10, end='', flush=True)
    except v80:
        v173(v10.v269('ascii', 'replace').v246('ascii'), end='', flush=True)
    v4.v174.v81(parents=True, exist_ok=True)
    with v4.v175('a', encoding='utf-8') as v82:
        v82.v176(v10)

def region_of(v11, v12) -> v5:
    """Cut the tape by what is visible WITHOUT the answer.

    Two coordinates, both readable from the question and the evidence alone: how deep the
    address is, and whether its mentions agree. Nothing that needs the truth may enter, or the
    map becomes a restatement of the score - 286's exam tautology in the shape of a chart. Rare
    words and sibling relations belong here too and are deliberately left out for now: each new
    cut splits the denominators, and a region under MIN_ANSWERED states nothing.
    """
    v13 = v83(v12['slots'])
    v14 = v12['vals']
    v15 = v264(v14).v220(1)[0][1]
    v16 = 'thin' if v13 <= 2 else 'mid' if v13 <= 4 else 'deep'
    v17 = 'unanimous' if v15 == v13 else 'majority' if v15 > v13 / 2 else 'split'
    return f"{v12['verb']}|{v16}|{v17}"

def spearman(v18, v19) -> v6:
    v13 = v83(v18)
    if v13 < 3:
        return v6('nan')

    def rank(v84):
        v85 = v101(v145(v13), key=lambda v87: v84[v87])
        v86 = [0.0] * v13
        v87 = 0
        while v87 < v13:
            v177 = v87
            while v177 + 1 < v13 and v84[v85[v177 + 1]] == v84[v85[v87]]:
                v177 += 1
            v9 = (v87 + v177) / 2.0
            for v104 in v145(v87, v177 + 1):
                v86[v85[v104]] = v9
            v87 = v177 + 1
        return v86
    v88, v89 = (v178(v18), v178(v19))
    v90, v91 = (v92(v88) / v13, v92(v89) / v13)
    v20 = v92(((v84 - v90) * (v247 - v91) for v84, v247 in v248(v88, v89)))
    v21 = v179.v93(v92(((v84 - v90) ** 2 for v84 in v88)))
    v22 = v179.v93(v92(((v247 - v91) ** 2 for v247 in v89)))
    return v20 / (v21 * v22) if v21 > 0 and v22 > 0 else v6('nan')

def ece(v23, v24: v7=10):
    """Expected calibration error, with the bound it must clear DERIVED rather than chosen.

    Each bin's realised accuracy is a binomial mean, so its own sampling noise is
    sqrt(acc(1-acc)/n). Summing that across bins, weighted the same way ECE is, gives the ECE a
    perfectly calibrated mind would still show on this many questions. The gate is ECE below
    that, so the threshold comes from the data and not from taste.
    """
    v25 = v94(v95)
    for v96, v97 in v23:
        v25[v258(v24 - 1, v7(v96 * v24))].v180((v96, v97))
    v13 = v83(v23)
    if not v13:
        return (v6('nan'), v6('nan'), [])
    v98, v99, v100 = (0.0, 0.0, [])
    for v19 in v101(v25):
        v102 = [v181 for v181, v117 in v25[v19]]
        v103 = [v182 for v117, v182 in v25[v19]]
        v104 = v83(v103)
        v183, v184 = (v92(v103) / v104, v92(v102) / v104)
        v98 += v104 / v13 * v221(v183 - v184)
        v99 += v104 / v13 * v179.v93(v256(v183 * (1 - v183), 1e-09) / v104)
        v100.v180({'bin': v19, 'n': v104, 'stated': v184, 'realised': v183})
    return (v98, v99, v100)

def main() -> v7:
    v26 = v185.v105()
    v26.v106('--smoke', action='store_true')
    v26.v106('--train-steps', type=v7, default=0)
    v26.v106('--tape-period', type=v7, default=50)
    v26.v106('--addresses', type=v7, default=0)
    v26.v106('--min-mentions', type=v7, default=2)
    v26.v106('--address-tau', type=v6, default=0.9)
    v26.v106('--address-overlap', type=v7, default=2)
    v26.v106('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v26.v106('--lr', type=v6, default=0.001)
    v26.v106('--holdout', choices=('corpus', 'address'), default='corpus')
    v26.v106('--no-scan-cache', action='store_true', help='disable the exact corpus-scan memo (use to verify it changes nothing)')
    v26.v106('--no-fast-grouping', action='store_true', help='disable the batched single-link grouping (use to verify it changes nothing)')
    v26.v106('--wiki-bytes', type=v7, default=0)
    v26.v106('--train-lines', type=v7, default=0)
    v26.v106('--eval-lines', type=v7, default=0)
    v26.v106('--refuse-at', type=v6, default=0.0, help="0 means derive the threshold: refuse where stated confidence is below the point at which answering stops beating the region's own floor")
    v26.v106('--edge-channels', type=v5, default='same,cos,rare', help="match 289's edge ablation")
    v26.v106('--no-ladder', action='store_true', help="match 289's ablation: audit a mind trained without the ladder")
    v26.v106('--run-tag', type=v5, default='')
    v27 = v26.v107()
    v108.v28 = not v27.v109
    v108.v29 = {v181.v186() for v181 in v27.v249.v222(',') if v181.v186()}
    global LOG_PATH
    v30 = v27.v170 and f'_{v27.v170}' or ''
    v30 += '_addrholdout' if v27.v134 == 'address' else ''
    v4 = v0 / f'_stage289c_log{v30}.txt'
    v4.v174.v81(parents=True, exist_ok=True)
    v4.v110('', encoding='utf-8')
    v31 = v187.v31('cuda' if v187.v250.v223() else 'cpu')
    v32 = v188.v111(v3)
    v187.v112(v3)
    v33 = v113.v113()
    v34 = v27.v114 or (600 if v27.v169 else 6000)
    v35 = v27.v115 or (300 if v27.v169 else 400)
    v116(f'Stage289c audit start {v267.v262(v268.v263).v217()} device={v31}')
    v117, v117, v118, v119 = v120()
    v36 = v189.v121(v5(v224.v190))
    v37 = v36.v191(v192) or 0
    v38 = v225(v119, v36.v251()).v122(v31)
    v38.v123(v187.v226(v1, map_location=v31, weights_only=False)['model'])
    v38.v124()
    for v11 in v38.v125():
        v11.v193(False)
    v39 = v126(v194(v38, v118, v31))
    v40 = v195.v127(v38)
    v41: v42 = {}
    v43 = v128.v44

    def _cached_common(v129, v130: v7=3):
        v104 = (v227(v129), v83(v129), v130)
        if v104 not in v41:
            v41[v104] = v43(v129, v130)
        return v41[v104]
    v128.v44 = v45
    if not v27.v131:
        v196(v128)
    if not v27.v132:
        v197(v128)
    with v2.v175('r', encoding='utf-8', errors='ignore') as v82:
        v133 = v82.v198(v27.v228 or (4000000 if v27.v169 else 30000000))
    v46 = [v199.v186() for v199 in v133.v222('\n') if 80 <= v83(v199.v186()) <= 400]
    v47 = v7(0.7 * v83(v46))
    v48 = v46[:v47][:v27.v48 or (3000 if v27.v169 else 25000)]
    v49 = v46[v47:][:v27.v49 or (1500 if v27.v169 else 12000)]
    if v27.v134 == 'address':
        v49 = v48

    def side(v135: v5) -> v7:
        return v7(v272.v270(v274.v273(v135).v269('utf-8')).v252(), 16) & 1

    def new_pack(v86, v129, v136):
        v11 = v229.v200(v129, bank=v39, tok=v36, pad_id=v37, device=v31, rng=v86, n_addr=v35, min_mentions=v27.v230, tau=v27.v231, overlap=v27.v232, soft_match=0.0, min_per_family=8, addr_key=v27.v233)
        if v27.v134 == 'address':
            v11 = v42(v11)
            v11['items'] = [v234 for v234 in v11['items'] if v271(v234['address']) == v136]
        return v11
    v50 = v108.v137(v31)
    v51 = v187.v201.v138(v50.v125(), lr=v27.v202, weight_decay=0.01)
    v52 = v7(v92((v84.v253() for v84 in v50.v125())))
    v53 = v139(v32, v48, 0)
    v54 = v108.v140(v53, v32)
    v55 = v139(v188.v111(v3 + 99), v49, 1)
    v56 = v108.v140(v55, v188.v111(v3 + 7))

    def by_verb(v141):
        v142 = v94(v95)
        for v12 in v141:
            v142[v12['verb']].v180(v12)
        return v142
    v57 = v143(v54)
    v58 = [v144 for v144 in v108.v203 if v57.v235(v144)]
    v116(f"  tape: {v53['n_addresses']} addresses | questions {v245.v218({v104: v83(v144) for v104, v144 in v57.v242()})} | params {v52}")
    for v59 in v145(1, v34 + 1):
        if (v59 - 1) % v27.v254 == 0 and v59 > 1:
            v53 = v139(v32, v48, 0)
            v54 = v108.v140(v53, v32)
            v57 = v143(v54)
            v58 = [v144 for v144 in v108.v203 if v57.v235(v144)]
        if not v58:
            v116('  empty tape after resample')
            return 1
        v144 = v58[v32.v236(v83(v58))]
        v12 = v57[v144][v32.v236(v83(v57[v144]))]
        v146 = v108.v204(v50, v53, v12, v31, v39)
        v51.v205(set_to_none=True)
        v146.v206()
        v187.v255.v237.v207(v50.v125(), 1.0)
        v51.v59()
        if v59 % v256(1, v34 // 8) == 0:
            v116(f'  step {v59}/{v34} loss={v6(v146):.4f}')
    v50.v124()
    v60 = v195.v127(v38)

    @v187.v148()
    def sweep(v11, v141):
        """Walk the whole tape unprompted and record, per question, what the mind would say,
        how sure it is, and whether it is right. Nothing is asked of it - it audits itself."""
        v147 = []
        for v12 in v141:
            v238, v239, v240 = v108.v241(v50, v11, v12, v31, v39)
            v147.v180({'region': v265(v11, v12), 'verb': v12['verb'], 'conf': v238, 'hit': v7(v239 == v240), 'n_choices': v108.v266(v12)})
        return v147

    def summarise(v147):
        v149 = v94(v95)
        for v86 in v147:
            v149[v86['region']].v180(v86)
        v78 = {}
        for v208, v209 in v101(v149.v242()):
            if v83(v209) < v257.v243:
                continue
            v78[v208] = {'n': v83(v209), 'stated_confidence': v6(v260.v244([v86['conf'] for v86 in v209])), 'realised_accuracy': v6(v260.v244([v86['hit'] for v86 in v209])), 'random_floor': v6(v260.v244([1.0 / v86['n_choices'] for v86 in v209]))}
        return v78
    v150, v151 = (v210(v53, v54), v210(v55, v56))
    v152, v153 = (v211(v150), v211(v151))
    v116(f'  MAP_HELD {v245.v218(v153)}')
    v154, v155, v25 = v156([(v86['conf'], v86['hit']) for v86 in v151])

    def _temp_scale(v157, v158):
        v212, v213 = (1.0, v6('inf'))
        for v159 in [0.05 * v104 for v104 in v145(1, 101)]:
            v199 = 0.0
            for v86 in v157:
                v12 = v258(v256(v86['conf'] ** (1.0 / v159), 1e-06), 1 - 1e-06)
                v199 -= v179.v116(v12) if v86['hit'] else v179.v116(1 - v12)
            if v199 < v213:
                v212, v213 = (v159, v199)
        return (v212, [(v258(v256(v86['conf'] ** (1.0 / v212), 1e-06), 1 - 1e-06), v86['hit']) for v86 in v158])
    v160, v161 = v162(v150, v151)
    v163, v164, v165 = v156(v161)
    v61 = v101(v153)
    v62 = v166([v153[v86]['stated_confidence'] for v86 in v61], [v153[v86]['realised_accuracy'] for v86 in v61])
    v63 = [v86 for v86 in v61 if v86 in v152]
    v64 = v166([v152[v86]['stated_confidence'] for v86 in v63], [v153[v86]['realised_accuracy'] for v86 in v63])
    v65 = v27.v66
    if v65 <= 0.0:
        v167 = v101({v259(v86['conf'], 3) for v86 in v151})
        v65 = 0.0
        for v159 in v167:
            v214 = [v86 for v86 in v151 if v86['conf'] >= v159]
            if v83(v214) < v257.v243:
                break
            v183 = v260.v244([v86['hit'] for v86 in v214])
            v215 = v260.v244([1.0 / v86['n_choices'] for v86 in v214])
            if v183 > v215:
                v65 = v159
                break
    v67 = [v86 for v86 in v151 if v86['conf'] >= v65]
    v68 = [v86 for v86 in v151 if v86['conf'] < v65]
    v69 = {'threshold': v65, 'answered_n': v83(v67), 'refused_n': v83(v68), 'answered_accuracy': v6(v260.v244([v86['hit'] for v86 in v67])) if v67 else v6('nan'), 'refused_would_have_been': v6(v260.v244([v86['hit'] for v86 in v68])) if v68 else v6('nan'), 'vacuous': not v68}
    v70 = v40 == v60
    v71 = v83(v153) >= 3 and v83(v151) >= 4 * v257.v243
    v72 = v168(not v179.v261(v154) and v154 <= v155)
    v73 = v168(not v179.v261(v163) and v163 <= v164)
    v74 = v168(not v179.v261(v62) and v62 > 0)
    v75 = v168(v69['refused_n'] >= v257.v243 and v69['answered_n'] >= v257.v243 and (v69['refused_would_have_been'] < v69['answered_accuracy']))
    v76 = v168(not v179.v261(v64) and v64 > 0)
    v77 = 'NO_TASK' if not (v71 and v70) else 'AUDIT_OK' if v72 and v74 and v75 and v76 else 'AUDIT_PARTIAL' if v74 or v72 else 'AUDIT_NO'
    v78 = {'stage': '289c', 'overall': v77, 'seed': v3, 'smoke': v27.v169, 'holdout': v27.v134, 'run_tag': v27.v170, 'train_steps': v34, 'params': v52, 'gates': {'G_arc_enc_frozen': v70, 'G_task_exists': v71, 'G_calibrated': v72, 'G_calibrated_after_train_temperature': v73, 'G_ranks_regions': v74, 'G_refusal_is_informed': v75, 'G_map_transfers': v76}, 'competence_map_held': v153, 'competence_map_train': v152, 'calibration': {'ece': v154, 'ece_sampling_noise': v155, 'bins': v25}, 'calibration_after_train_temperature': {'temperature': v160, 'ece': v163, 'ece_sampling_noise': v164, 'bins': v165, 'note': 'one parameter, fitted on the TRAIN sweep only, never on the held-out rows. It can rescale confidence but cannot invent an ordering, so G_ranks_regions remains the primary claim and this says whether the remaining error was a scale problem or a knowledge problem'}, 'spearman_conf_vs_accuracy': v62, 'spearman_train_map_vs_held_accuracy': v64, 'refusal': v69, 'region_note': "regions are cut by depth and agreement only - both readable without the answer. Cutting by anything that needs the truth would make the map a restatement of the score, which is 286's exam tautology wearing a chart", 'calibration_note': "the ECE bound is DERIVED: each bin's realised accuracy is a binomial mean whose own noise is sqrt(acc(1-acc)/n), weighted the same way ECE is. A perfectly calibrated mind still shows that much on this many questions, so the gate is ECE under its own noise floor and no threshold was chosen by taste", 'arc_enc_hash_before': v40, 'arc_enc_hash_after': v60, 'fp_version': v195.v216(), 'note': "The unprompted audit. The mind sweeps its own tape and publishes where it can be trusted, before anyone asks. The claim is narrow and checkable: stated confidence must be an empirical frequency - 0.8 must mean right eight times in ten - and the ordering of regions must be right, because a map that is accurate on average but orders regions wrongly makes you trust the wrong half of your tape. Refusal is a cell in the map, not an absence: the tape knows what the refused questions would have scored, so declining is checkable too. No new capability and no new model - 289's mind and 289's questions, read through the confidence they already produce, because a competence map that needed its own model would be a second mind grading the first.", 'timestamp': v267.v262(v268.v263).v217(), 'wall_s': v113.v113() - v33}
    v0.v81(parents=True, exist_ok=True)
    (v0 / f'stage289c_decision{v30}.json').v110(v245.v218(v78, indent=2), encoding='utf-8')
    v116(v245.v218({'overall': v77, 'gates': v78['gates'], 'ece': v154, 'noise': v155, 'rho': v62, 'refusal': v69}, indent=2))
    return 0
if v79 == '__main__':
    raise v171(v219())