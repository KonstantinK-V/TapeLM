"""
Stage 283 — Does the tape survive its own growth?

280 reached RAW_EXAM_OK on 187 addresses over 530 slots, and every number in it is a statement
about an index of 530 entries. The concept's own invariant is that the mind does not grow with
the knowledge, so the only way that invariant can fail is on the tape side: addressing, not
parameters. This measures the tape side alone - no policy, no BC, no RL, nothing trained - so a
ladder that would take days with a mind attached takes minutes.

Four things are expected to degrade, and each has a number here rather than an opinion.

  RETRIEVAL PRECISION. Votes over an inverted index return k slots whatever the index holds.
  With 530 slots a wrong slot is a coincidence; with 10^5 it is a population. If precision
  falls with N, every downstream number in 280 is a number about small tapes.

  THE RETURN PATH. 282's probe asks whether some other mention carries the subject and the
  value together. Accidental co-occurrence of two strings grows with the corpus - the 4MB smoke
  already showed it, where anchors like "september" made every value look corroborated. So the
  check is run twice at each rung: on the value the corpus actually settled on, and on a value
  taken from elsewhere in the tape. Only the DISTANCE between those two rates says whether the
  check still discriminates.

  ADDRESS COLLISIONS. An address is norm(fp(anchor) + ctx_fp(context)) and two addresses are
  merged when they pass tau with enough shared words. Distinct entities crowd as N grows, so the
  nearest-other-address cosine is reported as a distribution, not a mean.

  THE NATURAL FAMILY MIX. 280 quotas the families with --min-per-family so that abstention can
  be measured at all. That quota is a property of the exam, not of the corpus. Here the mix is
  left alone, which is the only way to see what a tie rate really is.

Nothing is trained and nothing is claimed about accuracy: an exam needs a mind, and this is a
measurement of what the mind would be handed.

  python _stage283_scale.py --smoke
  python _stage283_scale.py --rungs 4M:400 30M:400 120M:2000 400M:10000
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
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage271_controller as s271
import _stage280_raw_exam as s280
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('data/_wikitext103_train.txt')
v3 = 283
v4 = v0 / '_stage283_log.txt'

def log(v9: v68) -> None:
    v10 = v9 if v9.v132('\n') else v9 + '\n'
    try:
        v133(v10, end='', flush=True)
    except v69:
        v133(v10.v224('ascii', 'replace').v209('ascii'), end='', flush=True)
    v4.v134.v70(parents=True, exist_ok=True)
    with v4.v135('a', encoding='utf-8') as v71:
        v71.v136(v10)

def parse_rungs(v11: v14[v68]) -> v14[v74[v7, v7]]:
    """'30M:400' -> (30_000_000 chars, 400 addresses). Both have to move: the corpus caps how
    many addresses exist and n_addr caps how many are taken, so raising one alone measures the
    other's ceiling rather than scale."""
    v12 = []
    for v13 in v11:
        v118, v104, v137 = v13.v138(':')
        v72 = {'K': 1000, 'M': 1000000, 'G': 1000000000}.v139(v118[-1].v188(), 1)
        v17 = v7(v149(v118[:-1]) * v72) if v72 > 1 else v7(v118)
        v12.v140((v17, v7(v137)))
    return v73(v12)

def nearest_other(v15: v141.v75, v16: v7=512) -> v20.v5:
    """Cosine to the closest OTHER address. Chunked, because the pairwise matrix is the one
    thing here that grows quadratically and the whole point is to run at N where that bites."""
    v17 = v15.v76(0)
    v18 = v141.v77(v17, dtype=v141.v142, device=v15.v50)
    for v19 in v78(0, v17, v16):
        v79 = v143(v19 + v16, v17)
        v80 = v15[v19:v79] @ v15.v144
        v80[v141.v210(v79 - v19, device=v15.v50), v141.v210(v19, v79, device=v15.v50)] = -2
        v18[v19:v79] = v80.v189(dim=1).v38
    return v18.v190().v81()

def nearest_other_two(v21, v22, v23: v141.v75, v16: v7=256):
    """The same closest-other-member number, scored the way --addr-key two scores it.

    Written because the first version of this measurement read the summed key whatever rule was
    in force, so an arm that changes the rule would have moved nothing and the run would have
    proved only that the metric was blind.
    """
    v17 = v21.v76(0)
    v24 = v7(v23.v189()) + 1 if v17 else 0
    v18 = v141.v82((v24,), -2.0, device=v21.v50)
    for v19 in v78(0, v17, v16):
        v79 = v143(v19 + v16, v17)
        v80 = v141.v145(v21[v19:v79] @ v21.v144, v22[v19:v79] @ v22.v144)
        v80[v23[v19:v79].v211(1) == v23.v211(0)] = -2
        v18.v146(0, v23[v19:v79], v80.v189(dim=1).v38, reduce='amax')
    return v18.v190().v81()

def nearest_other_member(v15: v141.v75, v23: v141.v75, v16: v7=256):
    """Closest member of a DIFFERENT address, per address. The set rule's own crowding.

    Averaging members hides how they sit: two addresses whose means are far apart can still
    have one mention each that all but coincide, and under MaxSim that pair is what decides a
    hop. So the mean-key number stays as the old arm's and this one measures the rule in use.
    """
    v17 = v15.v76(0)
    v24 = v7(v23.v189()) + 1 if v17 else 0
    v18 = v141.v82((v24,), -2.0, device=v15.v50)
    for v19 in v78(0, v17, v16):
        v79 = v143(v19 + v16, v17)
        v80 = v15[v19:v79] @ v15.v144
        v80[v23[v19:v79].v211(1) == v23.v211(0)] = -2
        v18.v146(0, v23[v19:v79], v80.v189(dim=1).v38, reduce='amax')
    return v18.v190().v81()

def q(v25, *v26):
    v27 = v20.v83([v147 for v147 in v25 if not v225.v220(v147)], dtype=v20.v148)
    if v27.v76 == 0:
        return {f'p{v56}': v149('nan') for v56 in v26}
    return {f'p{v7(v56)}': v149(v20.v191(v27, v56)) for v56 in v26}

def measure(v28, v29, v30, v31, v32, v33, v34, v35, v36) -> v6:
    v37 = v28['items']
    v38 = v28['tape'].v38
    v39 = v73(v150(v38))
    v84, v85, v86, v87 = ([], [], [], [])
    v88, v89 = ([], [])
    v90, v91 = ([], [])
    for v40 in v37:
        v90.v140(v116((1 for v216 in v40['slots'] if v40['S'] not in v28['texts_lc'][v216])) / v189(1, v161(v40['slots'])))
        v91.v140(v7(v161(v40['slots']) <= v31))
        v92 = v151(v207.v212.v192(S=v40.v139('query') or v40['S']))
        v94, v152, v104 = v193.v153(v28, v92, v31, v32, v40, v35, v33, v34)
        v93 = v150(v40['slots'])
        if v94:
            v154 = v116((1 for v216 in v94 if v216 in v93))
            v84.v140(v154 / v161(v94))
            v85.v140(v154 / v189(1, v161(v93)))
        v86.v140(v161(v94))
        v87.v140(v7(not v152 or v189(v152.v38(), default=0.0) <= 0.0))
        if v40['truth'] is not None:
            v88.v140(v193.v213(v28, v40, v40['truth']))
            v155 = [v194 for v194 in v39 if v194 != v40['truth']]
            if v155:
                v89.v140(v193.v213(v28, v40, v30.v226(v155)))
    v41 = v156(v28['addr_keys']) if v28['addr_keys'] is not None else v20.v157([v149('nan')])
    if v28.v139('slot_keys') is not None:
        v95 = v141.v158([v28['slot_addr'][v19] for v19 in v28['slot_keys_slot']], device=v28['slot_keys'].v50)
        v96 = v159(v28['slot_keys'], v95)
        v97 = v195(v28['anc_keys'], v28['ctx_keys'], v95) if v28.v139('anc_keys') is not None else v20.v157([v149('nan')])
    else:
        v96 = v97 = v20.v157([v149('nan')])
    v42 = v73({v40['S'] for v40 in v37})
    v43 = v141.v214.v196.v160(v29.v221(v42).v149(), dim=-1) if v42 else None
    v44 = v156(v43.v111(v28['addr_keys'].v50)) if v43 is not None and v43.v76(0) > 1 else v20.v157([v149('nan')])
    v45 = v98((v40['kind'] for v40 in v37))
    v9 = lambda v25: v149(v20.v198(v25)) if v161(v25) else v149('nan')
    return {'n_addresses': v28['n_addresses'], 'n_slots': v28['n_slots'], 'n_items': v161(v37), 'slots_per_address': v28['n_slots'] / v189(1, v28['n_addresses']), 'write_actions': v28['write_actions'], 'families_natural': {v71: v45.v139(v71, 0) / v189(1, v161(v37)) for v71 in ('clean', 'decidable', 'tie')}, 'retrieval_precision': v9(v84), 'witness_recall': v9(v85), 'precision_q': v162(v84, 5, 50, 95), 'mean_candidates': v9(v86), 'words_silent_rate': v9(v87), 'foreign_member_rate': v9(v90), 'k_covers_address': v9(v91), 'return_path_true': v9(v88), 'return_path_other': v9(v89), 'return_path_separation': v9(v88) - v9(v89), 'nearest_other_q': v162(v41, 50, 95, 99), 'nearest_other_max': v149(v20.v197(v41)), 'address_crowding': v149(v20.v198(v41 >= v36)), 'member_crowding_q': v162(v96, 50, 95, 99), 'member_crowding': v149(v20.v198(v96 >= v36)), 'two_channel_crowding': v149(v20.v198(v97 >= v36)), 'two_channel_q': v162(v97, 50, 95, 99), 'distinct_anchors': v161(v42), 'anchor_only_q': v162(v44, 50, 95, 99), 'anchor_only_crowding': v149(v20.v198(v44 >= v36))}

def main() -> v7:
    v46 = v163.v99()
    v46.v100('--smoke', action='store_true')
    v46.v100('--rungs', nargs='*', default=[], help='chars:addresses, e.g. 30M:400 120M:2000')
    v46.v100('--corpus', type=v8, default=v2)
    v46.v100('--min-mentions', type=v7, default=2)
    v46.v100('--address-tau', type=v149, default=0.9)
    v46.v100('--address-overlap', type=v7, default=2)
    v46.v100('--soft-match', type=v149, default=0.0)
    v46.v100('--topk', type=v7, default=7)
    v46.v100('--hop', choices=('none', 'fp'), default='fp')
    v46.v100('--hop-min', type=v149, default=1.0)
    v46.v100('--k-gap', type=v149, default=0.35)
    v46.v100('--subject-filter', choices=('off', 'on'), default='on')
    v46.v100('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v46.v100('--min-precision', type=v149, default=0.6)
    v46.v100('--min-separation', type=v149, default=0.3)
    v46.v100('--max-crowding', type=v149, default=0.2)
    v46.v100('--max-foreign', type=v149, default=0.1)
    v46.v100('--run-tag', type=v68, default='')
    v47 = v46.v101()
    global LOG_PATH
    v48 = v47.v129 and f'_{v47.v129}' or ''
    v4 = v0 / f'_stage283_log{v48}.txt'
    v4.v134.v70(parents=True, exist_ok=True)
    v4.v102('', encoding='utf-8')
    v49 = v164(v47.v49) if v47.v49 else [(2000000, 100), (4000000, 200)] if v47.v128 else [(4000000, 400), (30000000, 400), (120000000, 2000)]
    v50 = v141.v50('cuda' if v141.v215.v199() else 'cpu')
    v51 = v103.v103()
    v104, v104, v105, v106 = v107()
    v52 = v165.v108(v68(v200.v166))
    v53 = v52.v109()
    v54 = v52.v167(v168) or 0
    v169.v110(v52, v105, v54, v53)
    v55 = v201(v106, v53).v111(v50)
    v55.v112(v141.v202(v1, map_location=v50, weights_only=False)['model'])
    v55.v113()
    for v56 in v55.v114():
        v56.v170(False)
    v29 = v115(v55, v105, v50)
    v57 = v116((v56.v203() for v56 in v55.v114()))
    v117(f'Stage283 scale start {v222.v218(v223.v219).v185()} device={v50} rungs={v49} mind_params={v57}')
    if not v47.v179.v171():
        v117(f'  corpus not found: {v47.v179}')
        return 1
    with v47.v179.v135('r', encoding='utf-8', errors='ignore') as v71:
        v82 = v71.v172(v189((v216 for v216, v104 in v49)))
    v117(f'  corpus {v161(v82) / 1000000.0:.1f}M chars read ({v103.v103() - v51:.0f}s)')
    v58 = []
    for v118, v119 in v49:
        v120 = v103.v103()
        v30 = v204.v173(v3)
        v121 = [v206.v205() for v206 in v82[:v118].v217('\n') if 80 <= v161(v206.v205()) <= 400]
        v28 = v193.v174(v121, bank=v29, tok=v52, pad_id=v54, device=v50, rng=v30, n_addr=v119, min_mentions=v47.v182, tau=v47.v177, overlap=v47.v180, soft_match=v47.v181, min_per_family=0, addr_key=v47.v130)
        v122 = v103.v103() - v120
        if not v28['items']:
            v117(f'  rung {v118}:{v119} produced no items, skipped')
            continue
        v123 = v175(v28, v29, v30, v47.v176, v47.v32, v47.v33, v47.v34, v47.v35 == 'on', v47.v177)
        v123.v178({'chars': v118, 'n_addr_requested': v119, 'build_s': v122, 'mind_params': v57, 'lines': v161(v121)})
        v58.v140(v123)
        v117(f"  rung {v118 / 1000000.0:.0f}M:{v119} -> {v123['n_addresses']} addr / {v123['n_slots']} slots | precision {v123['retrieval_precision']:.3f} | return true {v123['return_path_true']:.3f} other {v123['return_path_other']:.3f} (sep {v123['return_path_separation']:+.3f}) | crowding {v123['address_crowding']:.3f} (members {v123['member_crowding']:.3f} two {v123['two_channel_crowding']:.3f} anchors alone {v123['anchor_only_crowding']:.3f}) foreign {v123['foreign_member_rate']:.3f} | k covers {v123['k_covers_address']:.3f} | ties {v123['families_natural']['tie']:.3f} ({v122:.0f}s)")
    if not v58:
        v117('  nothing measured')
        return 1
    v59 = v58[-1]
    v60 = v58[0]
    v61 = v59['retrieval_precision'] >= v47.v124
    v62 = v59['return_path_separation'] >= v47.v125
    v63 = v59['foreign_member_rate'] <= v47.v126
    v64 = v59['families_natural']['tie'] > 0.0
    v65 = v127((v123['mind_params'] == v57 for v123 in v58))
    v66 = 'SCALE_OK' if v127((v61, v62, v63, v64)) else 'SCALE_PARTIAL' if v61 and v64 else 'SCALE_NO'
    v12 = {'stage': 283, 'overall': v66, 'seed': v3, 'smoke': v47.v128, 'corpus': v68(v47.v179), 'run_tag': v47.v129, 'addr_key': v47.v130, 'address': {'tau': v47.v177, 'overlap': v47.v180, 'soft_match': v47.v181, 'min_mentions': v47.v182}, 'retrieval': {'topk': v47.v176, 'hop': v47.v32, 'hop_min': v47.v33, 'k_gap': v47.v34, 'subject_filter': v47.v35}, 'thresholds': {'min_precision': v47.v124, 'min_separation': v47.v125, 'max_crowding': v47.v183, 'max_foreign': v47.v126}, 'mind_params': v57, 'gates': {'G_precision_holds': v61, 'G_return_path_separates': v62, 'G_addresses_stay_distinct': v63, 'G_k_covers_the_address': v59['k_covers_address'] >= 0.9, 'G_ties_exist_naturally': v64, 'G_mind_does_not_grow': v65}, 'slopes': {'slots': [v123['n_slots'] for v123 in v58], 'precision': [v123['retrieval_precision'] for v123 in v58], 'return_path_separation': [v123['return_path_separation'] for v123 in v58], 'address_crowding': [v123['address_crowding'] for v123 in v58], 'member_crowding': [v123['member_crowding'] for v123 in v58], 'two_channel_crowding': [v123['two_channel_crowding'] for v123 in v58], 'foreign_member_rate': [v123['foreign_member_rate'] for v123 in v58], 'distinct_anchors': [v123['distinct_anchors'] for v123 in v58], 'anchor_only_crowding': [v123['anchor_only_crowding'] for v123 in v58], 'anchor_growth_exponent': v225.v117(v189(1, v59['distinct_anchors']) / v189(1, v60['distinct_anchors'])) / v225.v117(v59['chars'] / v189(1, v60['chars'])) if v161(v58) > 1 and v59['chars'] != v60['chars'] else v149('nan'), 'k_covers_address': [v123['k_covers_address'] for v123 in v58], 'tie_share': [v123['families_natural']['tie'] for v123 in v58], 'slots_growth': v59['n_slots'] / v189(1, v60['n_slots'])}, 'rungs': v58, 'fp_version': v207.v184(), 'reference_280': {'n_addresses': 187, 'n_slots': 530, 'retrieval_precision_train': 0.8031194295900178, 'held_out_reward': 0.7043859649122806}, 'note': "The tape side of scale, with no mind attached. 280's numbers describe an index of 530 slots, and the concept's invariant is that the mind does not grow with the knowledge - so if anything breaks with N it breaks in the addressing. Four things are measured at each rung and none of them needs training: how much of a retrieved list belongs to the address it was retrieved for; whether the return path still tells a settled value from a value taken elsewhere in the tape, which is the check meant to replace G_answer_is_slot once an answer stops being a slot index; how close the nearest other address key gets, since distinct entities crowd as the tape fills; and what the family mix is when nothing quotas it, because 280's tie rate is an artefact of --min-per-family and not a fact about text. Slopes are reported beside the gates: a metric that clears its floor while falling steeply is the case that decides whether this scales or merely has not broken yet.", 'timestamp': v222.v218(v223.v219).v185(), 'wall_s': v103.v103() - v51}
    v0.v70(parents=True, exist_ok=True)
    (v0 / f'stage283_decision{v48}.json').v102(v208.v186(v12, indent=2), encoding='utf-8')
    v117(v208.v186({'overall': v66, 'gates': v12['gates'], 'slopes': v12['slopes']}, indent=2))
    return 0
if v67 == '__main__':
    raise v131(v187())