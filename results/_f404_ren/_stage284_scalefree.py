"""
Stage 284 — Addressing without a constant to tune.

283 said the tape survives growth, and it said so while every decision in the addressing was
made by a number chosen by hand: merge at cosine 0.90, read seven slots, demand two shared
words, hop when the top score falls under 1.0. Those numbers were picked on a tape of 530
slots. A system that needs them re-picked at 10^5 has not scaled, it has been re-fitted, and
the difference is invisible as long as the ladder is allowed a fresh configuration per rung.

So this stage removes the constants rather than retuning them, using the rule the mind already
lives by. 278's teacher never read an absolute score - it compared the leader with the runner
up - and a comparison does not care how full the space is. The same idea, three times:

  MERGE. Not "cosine clears tau" but "each is the other's nearest": two mentions join only if
  they are MUTUALLY nearest, in both channels. Mutual nearest neighbour has no threshold at
  all, and it cannot chain a crowd together the way single-link at a fixed cosine does, because
  a mention with a closer neighbour elsewhere simply is not mutual with this one.

  SHARED WORDS. Not "at least two" but "at least one that is not ordinary": the shared word has
  to carry more idf than the median word of the tape, which is a quantity the corpus computes
  about itself rather than a number brought from outside.

  READING. Not "the top seven" but "everything before the biggest drop": the candidate list is
  cut at the largest gap between consecutive scores, so a well supported address gives up more
  slots than a thin one, and neither is capped by a constant that has to grow with the tape.

The gate that matters is not any single number. It is that ONE configuration clears every rung:
G_one_config_all_rungs fails if a metric passes at the top of the ladder while failing lower
down, which is exactly what per-rung tuning buys and what 283 could not see, since it read its
gates off the last rung alone.

  python _stage284_scalefree.py --smoke
  python _stage284_scalefree.py --rungs 4M 30M 120M
  python _stage284_scalefree.py --rungs 4M 30M 120M --rule fixed   # the arm 283 measured
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
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage271_controller as s271
import _stage279_write_decision as s279
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
v0 = v9('results')
v1 = v9('checkpoints/stage191_p1_curve.pt')
v2 = v9('data/_wikitext103_train.txt')
v3 = 284
v4 = ('clean', 'decidable', 'tie')
v5 = v0 / '_stage284_log.txt'

def log(v10: v78) -> None:
    v11 = v10 if v10.v156('\n') else v10 + '\n'
    try:
        v157(v11, end='', flush=True)
    except v79:
        v157(v11.v256('ascii', 'replace').v240('ascii'), end='', flush=True)
    v5.v22.v80(parents=True, exist_ok=True)
    with v5.v158('a', encoding='utf-8') as v81:
        v81.v159(v11)

def chars_of(v12: v78) -> v6:
    v13 = {'K': 1000, 'M': 1000000, 'G': 1000000000}.v82(v12[-1].v160(), 1)
    return v6(v7(v12[:-1]) * v13) if v13 > 1 else v6(v12)

def mutual_groups(v14: v161.v83, v15: v161.v83, v16, v17, v18: v6=512):
    """Group mentions that are each other's nearest, in both channels and in the words.

    No cosine threshold anywhere. Two mentions are joined when neither has a closer partner -
    which is a statement about the ranking, so it means the same thing on a tape of fifty slots
    and on one of fifty thousand. The pairs are then closed transitively, but only through
    mutual pairs: a mention dragged toward a crowd loses its mutuality to every member of it
    and stays where it is, which is the failure single-link-at-tau could not avoid.
    """
    v19 = v14.v84(0)
    if v19 == 0:
        return []
    v20 = v161.v85(v19, dtype=v161.v162)
    for v21 in v86(0, v19, v18):
        v87 = v98(v21 + v18, v19)
        v88 = v161.v163(v14[v21:v87] @ v14.v215, v15[v21:v87] @ v15.v215)
        v88[v161.v241(v87 - v21, device=v14.v63), v161.v241(v21, v87, device=v14.v63)] = -2
        v20[v21:v87] = v88.v242(dim=1).v164()
    v22 = v28(v86(v19))

    def find(v89):
        while v22[v89] != v89:
            v22[v89] = v22[v22[v89]]
            v89 = v22[v89]
        return v89
    for v21 in v86(v19):
        v87 = v6(v20[v21])
        if v6(v20[v87]) == v21 and v16[v21] & v16[v87] & v17:
            v46, v216 = (v243(v21), v243(v87))
            if v46 != v216:
                v22[v46] = v216
    v23: v8[v6, v28[v6]] = v90(v28)
    for v21 in v86(v19):
        v23[v243(v21)].v165(v21)
    return v28(v23.v166())

def tau_groups(v14: v161.v83, v15: v161.v83, v16, v24: v7, v25: v6):
    """279's rule, kept runnable so the constants can be measured rather than argued about."""
    v26: v28[v28[v6]] = []
    for v21 in v86(v14.v84(0)):
        v20, v167 = (-1, v24)
        for v50, v168 in v118(v26):
            v169 = v161.v217(v168, device=v14.v63)
            v88 = v161.v163(v14[v169] @ v14[v21], v15[v169] @ v15[v21]).v218()
            for v87, v219 in v118(v88):
                if v219 >= v167 and v119(v16[v21] & v16[v168[v87]]) >= v25:
                    v20, v167 = (v50, v219)
        if v20 < 0:
            v26.v165([v21])
        else:
            v26[v20].v165(v21)
    return v26

def elbow(v27: v28[v220[v6, v7]]) -> v28[v6]:
    """Everything above the biggest drop. A rank rule, so no k and no floor.

    A fixed k hands the reader whatever fills the list and caps a well attested address at the
    same size as a thin one; 283 measured that cap biting - 12% of addresses had more mentions
    than the reader could hold. The largest gap between consecutive scores is where the list
    stops being about this address, and it is computed from the scores themselves.
    """
    if v119(v27) <= 1:
        return [v38 for v38, v101 in v27]
    v91, v92 = (v119(v27), -1.0)
    for v21 in v86(1, v119(v27)):
        v93 = v27[v21 - 1][1] - v27[v21][1]
        if v93 > v92:
            v92, v91 = (v93, v21)
    return [v38 for v38, v101 in v27[:v91]]

def retrieve(v29, v16, v30: v78, v31: v6):
    v94, v95 = v170.v96(v16, v29['postings'], v29['idf'], v221(v31, 64) if v30 == 'margin' else v31)
    if not v94:
        return ([], {})
    v27 = v97(((v38, v95.v82(v38, 0.0)) for v38 in v94), key=lambda v117: -v117[1])
    v32 = v171(v27) if v30 == 'margin' else [v38 for v38, v101 in v27[:v31]]
    return (v32, {v38: v95.v82(v38, 0.0) for v38 in v32})

def return_path(v29, v33, v34) -> v7:
    v16 = v172(v34) or [v34]
    v35 = v98((v29['postings_probe'].v82(v109, ()) for v109 in v16), key=v119, default=())
    v36 = v34.v99()
    v37 = 0
    for v38 in v35:
        if v33 in v29['texts_lc'][v38] and v36 in v29['texts_lc'][v38]:
            v37 += 1
            if v37 >= 2:
                return 1.0
    return 0.0

def build(v39, v40, v41, v42, v43, v30, v24, v44):
    v45 = v173.v100(v39)
    v47, v101 = v173.v102(v39, v41, v42, v43, 'anchor_rel', common=v45)
    if not v47:
        return None
    v103, v104, v16 = ([], [], [])
    for v46 in v47:
        v105 = v46['address'].v222('|')[0]
        v38 = v40.v174(v46['ctx'], exclude=v46['value'])
        v106 = v40.v223([v105])[0]
        v103.v165(v175.v107(v106, dim=-1))
        v104.v165(v175.v107(v38, dim=-1) if v38 is not None else v175.v107(v106, dim=-1))
        v16.v165({v109.v99() for v109 in v173.v260.v257(v46['ctx']) if v109.v99() not in v173.v258} - {v46['value'].v99()})
    v14 = v175.v107(v161.v244(v103).v7(), dim=-1)
    v15 = v175.v107(v161.v244(v104).v7(), dim=-1)
    v48 = v108((v109 for v224 in v16 for v109 in v224))
    v49 = v7(v245.v225(v28(v48.v166()))) if v48 else 1.0
    v17 = {v109 for v109, v19 in v48.v54() if v19 <= v49}
    v23 = v176(v14, v15, v16, v17) if v30 == 'margin' else v177(v14, v15, v16, v24, v44)
    v23 = [v50 for v50 in v23 if v119(v50) >= v43]
    if not v23:
        return None
    v110, v111, v112 = ([], [], [])
    for v50 in v23:
        v113 = []
        for v21 in v50:
            v113.v165(v119(v110))
            v110.v165(v47[v21]['value'])
            v111.v165(v47[v21]['ctx'])
        v112.v165(v113)
    v114, v115 = (v90(v28), v90(v28))
    for v116, v117 in v118(v111):
        for v109 in v172(v117, exclude=v110[v116]):
            v114[v109].v165(v116)
        for v109 in v172(v117):
            v115[v109].v165(v116)
    v51 = v119(v110)
    v52 = {v109: v226.v149(v221(2.0, v51 / v221(1, v119(v114[v109])))) for v109 in v114}
    v53 = {v109: v226.v149(v221(2.0, v51 / v221(1, v119(v115[v109])))) for v109 in v115}
    v54 = []
    for v120, v50 in v118(v23):
        v113 = v112[v120]
        v121 = v108((v110[v21] for v21 in v113))
        v122 = v121.v178(2)
        v179, v180 = (v122[0][1], v122[1][1] if v119(v122) > 1 else 0)
        v181, v182 = ('clean', v122[0][0]) if v119(v121) == 1 else ('tie', None) if v179 == v180 else ('decidable', v122[0][0])
        v105 = v47[v50[0]]['address'].v222(':', 1)[-1].v222('|')[0]
        v123 = (v47[v50[0]]['address'].v222('|', 1) + [''])[1]
        v54.v165({'S': v105, 'query': (v105 + ' ' + v123).v236(), 'truth': v182, 'slots': v113, 'kind': v181})
    return {'items': v54, 'texts_lc': [v117.v99() for v117 in v111], 'postings': v114, 'idf': v52, 'postings_probe': v115, 'idf_probe': v53, 'n_addresses': v119(v23), 'n_slots': v51, 'values': v110}

def measure(v29, v41, v30, v31) -> v8:
    v54, v110 = (v29['items'], v29['values'])
    v55 = v97(v183(v110))
    v124, v125, v126, v127, v128 = ([], [], [], [], [])
    v129, v130 = ([], [])
    for v56 in v54:
        v94, v101 = v184(v29, v172(v170.v253.v246(S=v56['query'])), v30, v31)
        v131 = v183(v56['slots'])
        if v94:
            v185 = v148((1 for v38 in v94 if v38 in v131))
            v124.v165(v185 / v119(v94))
            v125.v165(v185 / v221(1, v119(v131)))
        v126.v165(v119(v94))
        v128.v165(v6(v119(v56['slots']) <= v119(v94)) if v94 else 0)
        v127.v165(v148((1 for v38 in v56['slots'] if v56['S'] not in v29['texts_lc'][v38])) / v221(1, v119(v56['slots'])))
        if v56['truth'] is not None:
            v129.v165(v247(v29, v56['S'], v56['truth']))
            v186 = [v227 for v227 in v55 if v227 != v56['truth']]
            if v186:
                v130.v165(v247(v29, v56['S'], v41.v259(v186)))
    v57 = v108((v56['kind'] for v56 in v54))
    v10 = lambda v187: v7(v245.v248(v187)) if v119(v187) else v7('nan')
    return {'n_addresses': v29['n_addresses'], 'n_slots': v29['n_slots'], 'n_items': v119(v54), 'slots_per_address': v29['n_slots'] / v221(1, v29['n_addresses']), 'families_natural': {v81: v57.v82(v81, 0) / v221(1, v119(v54)) for v81 in v4}, 'retrieval_precision': v10(v124), 'witness_recall': v10(v125), 'mean_candidates': v10(v126), 'reader_covers_address': v10(v128), 'foreign_member_rate': v10(v127), 'return_path_true': v10(v129), 'return_path_other': v10(v130), 'return_path_separation': v10(v129) - v10(v130)}

def rung_gates(v58, v59) -> v8:
    return {'precision': v58['retrieval_precision'] >= v59.v188, 'return_path': v58['return_path_separation'] >= v59.v189, 'addresses_distinct': v58['foreign_member_rate'] <= v59.v190, 'reader_covers': v58['reader_covers_address'] >= v59.v191, 'ties_exist': v58['families_natural']['tie'] > 0.0}

def main() -> v6:
    v60 = v192.v132()
    v60.v133('--smoke', action='store_true')
    v60.v133('--rungs', nargs='*', default=[])
    v60.v133('--corpus', type=v9, default=v2)
    v60.v133('--rule', choices=('margin', 'fixed'), default='margin')
    v60.v133('--addresses', type=v6, default=4000)
    v60.v133('--min-mentions', type=v6, default=2)
    v60.v133('--topk', type=v6, default=7, help='only read under --rule fixed')
    v60.v133('--address-tau', type=v7, default=0.9, help='only under --rule fixed')
    v60.v133('--address-overlap', type=v6, default=2, help='only under --rule fixed')
    v60.v133('--min-precision', type=v7, default=0.6)
    v60.v133('--min-separation', type=v7, default=0.3)
    v60.v133('--max-foreign', type=v7, default=0.1)
    v60.v133('--min-covers', type=v7, default=0.9)
    v60.v133('--run-tag', type=v78, default='')
    v59 = v60.v134()
    global LOG_PATH
    v61 = v59.v154 and f'_{v59.v154}' or ''
    v61 += '' if v59.v30 == 'margin' else '_fixed'
    v5 = v0 / f'_stage284_log{v61}.txt'
    v5.v22.v80(parents=True, exist_ok=True)
    v5.v135('', encoding='utf-8')
    v62 = [v228(v229) for v229 in v59.v62] or ([2000000, 4000000] if v59.v153 else [4000000, 30000000, 120000000])
    v62.v136()
    v63 = v161.v63('cuda' if v161.v249.v230() else 'cpu')
    v64 = v137.v137()
    v101, v101, v138, v139 = v140()
    v65 = v193.v141(v78(v231.v194))
    v66 = v65.v195(v196) or 0
    v197.v142(v65, v138, v66, v65.v198())
    v67 = v232(v139, v65.v198()).v143(v63)
    v67.v144(v161.v233(v1, map_location=v63, weights_only=False)['model'])
    v67.v145()
    for v68 in v67.v146():
        v68.v199(False)
    v40 = v147(v67, v138, v63)
    v69 = v148((v68.v234() for v68 in v67.v146()))
    v149(f'Stage284 scalefree start {v254.v251(v255.v252).v212()} rule={v59.v30} rungs={v62} mind_params={v69}')
    if not v59.v210.v200():
        v149(f'  corpus not found: {v59.v210}')
        return 1
    with v59.v210.v158('r', encoding='utf-8', errors='ignore') as v81:
        v150 = v81.v201(v221(v62))
    v70 = []
    for v71 in v62:
        v151 = v137.v137()
        v41 = v235.v202(v3)
        v39 = [v237.v236() for v237 in v150[:v71].v222('\n') if 80 <= v119(v237.v236()) <= 400]
        v29 = v203(v39, v40, v41, v59.v204, v59.v43, v59.v30, v59.v205, v59.v206)
        if v29 is None or not v29['items']:
            v149(f'  rung {v71} produced nothing, skipped')
            continue
        v58 = v207(v29, v41, v59.v30, v59.v208)
        v58.v209({'chars': v71, 'build_s': v137.v137() - v151, 'mind_params': v69, 'gates': v250(v58, v59)})
        v70.v165(v58)
        v149(f"  rung {v71 / 1000000.0:.0f}M -> {v58['n_addresses']} addr / {v58['n_slots']} slots | precision {v58['retrieval_precision']:.3f} | return sep {v58['return_path_separation']:+.3f} | foreign {v58['foreign_member_rate']:.3f} | covers {v58['reader_covers_address']:.3f} | cands {v58['mean_candidates']:.2f} | ties {v58['families_natural']['tie']:.3f} ({v58['build_s']:.0f}s)")
    if not v70:
        v149('  nothing measured')
        return 1
    v72 = [v152(v58['gates'].v166()) for v58 in v70]
    v73 = v152(v72) and v119(v70) > 1
    v74 = v152((v58['mind_params'] == v69 for v58 in v70))
    v75 = 'SCALEFREE_OK' if v73 and v74 else 'SCALEFREE_PARTIAL' if v238(v72) else 'SCALEFREE_NO'
    v76 = {'stage': 284, 'overall': v75, 'rule': v59.v30, 'seed': v3, 'smoke': v59.v153, 'corpus': v78(v59.v210), 'run_tag': v59.v154, 'mind_params': v69, 'constants_in_use': {'none': True} if v59.v30 == 'margin' else {'tau': v59.v205, 'overlap': v59.v206, 'topk': v59.v208}, 'thresholds': {'min_precision': v59.v188, 'min_separation': v59.v189, 'max_foreign': v59.v190, 'min_covers': v59.v191}, 'gates': {'G_one_config_all_rungs': v73, 'G_mind_does_not_grow': v74}, 'per_rung_pass': v72, 'slopes': {v106: [v58[v106] for v58 in v70] for v106 in ('n_slots', 'n_addresses', 'retrieval_precision', 'witness_recall', 'return_path_separation', 'foreign_member_rate', 'reader_covers_address', 'mean_candidates')}, 'tie_share': [v58['families_natural']['tie'] for v58 in v70], 'rungs': v70, 'fp_version': v170.v211(), 'reference_283': {'rule': 'fixed', 'foreign_member_rate_top': 0.00860178354884333, 'precision_top': 0.6201471466579791, 'k_covers_top': 0.9486887115165337}, 'note': "283 said the tape survives growth while every decision in the addressing was made by a number picked on a tape of 530 slots: merge at cosine 0.90, read seven, demand two shared words. A system that needs those re-picked at 10^5 has not scaled, it has been re-fitted, and a ladder allowed a fresh configuration per rung cannot tell the two apart. So the constants are removed rather than retuned, using the rule 278's teacher already lived by - compare the leader with the runner up, never read an absolute score. Mentions merge when they are MUTUALLY nearest in both channels, agreeing on a word rarer than the tape's median; the reader stops at the largest gap between consecutive scores instead of at a fixed k. The gate is that one configuration clears every rung, which is what per-rung tuning buys and what reading the gates off the last rung alone would hide. --rule fixed is the arm 283 measured, kept runnable so the constants can be compared rather than argued.", 'timestamp': v254.v251(v255.v252).v212(), 'wall_s': v137.v137() - v64}
    v0.v80(parents=True, exist_ok=True)
    (v0 / f'stage284_decision{v61}.json').v135(v239.v213(v76, indent=2), encoding='utf-8')
    v149(v239.v213({'overall': v75, 'gates': v76['gates'], 'per_rung_pass': v72, 'slopes': v76['slopes']}, indent=2))
    return 0
if v77 == '__main__':
    raise v155(v214())