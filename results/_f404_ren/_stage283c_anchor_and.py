"""
Stage 283c — Anchor-block write with AND + relation overlap.

283b scoped soft merge to one anchor and killed cross-entity foreign (0.0, flat), but OR
linking on full context glued distinct relations of the same name → tie ≈ 0.70.

283c keeps the hard block. Inside the block the edge is conjunction: relation-tail overlap
>= min_overlap AND two-channel cos >= tau=0.90. Same numbers as write2; no tau sweep. Overlap
reads the address relation (`anchor|rel`), not ambient context English.

  python _stage283c_anchor_and.py --smoke
  python _stage283c_anchor_and.py --rungs 4M:400 30M:400 120M:2000
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage283_scale as s283
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
v0 = v7('results')
v1 = v7('checkpoints/stage191_p1_curve.pt')
v2 = v7('data/_wikitext103_train.txt')
v3 = 283
v4 = v0 / '_stage283c_log.txt'
v5 = 'anchor_block_and'

def log(v8: v33) -> None:
    v9 = v8 if v8.v72('\n') else v8 + '\n'
    try:
        v73(v9, end='', flush=True)
    except v34:
        v73(v9.v131('ascii', 'replace').v122('ascii'), end='', flush=True)
    v4.v74.v35(parents=True, exist_ok=True)
    with v4.v75('a', encoding='utf-8') as v36:
        v36.v76(v9)

def main() -> v6:
    v10 = v77.v37()
    v10.v38('--smoke', action='store_true')
    v10.v38('--rungs', nargs='*', default=[])
    v10.v38('--corpus', type=v7, default=v2)
    v10.v38('--min-mentions', type=v6, default=2)
    v10.v38('--address-tau', type=v78, default=0.9, help='fixed; 283c does not sweep this')
    v10.v38('--address-overlap', type=v6, default=2)
    v10.v38('--soft-match', type=v78, default=0.0)
    v10.v38('--topk', type=v6, default=7)
    v10.v38('--hop', choices=('none', 'fp'), default='fp')
    v10.v38('--hop-min', type=v78, default=1.0)
    v10.v38('--k-gap', type=v78, default=0.35)
    v10.v38('--subject-filter', choices=('off', 'on'), default='on')
    v10.v38('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v10.v38('--min-precision', type=v78, default=0.6)
    v10.v38('--min-separation', type=v78, default=0.3)
    v10.v38('--max-crowding', type=v78, default=0.2)
    v10.v38('--max-foreign', type=v78, default=0.1)
    v11 = v10.v39()
    global LOG_PATH
    v4 = v0 / '_stage283c_log.txt'
    v4.v74.v35(parents=True, exist_ok=True)
    v4.v40('', encoding='utf-8')
    v12 = v109.v79(v11.v12) if v11.v12 else [(2000000, 100), (4000000, 200)] if v11.v69 else [(4000000, 400), (30000000, 400), (120000000, 2000)]
    v13 = v80.v13('cuda' if v80.v123.v110() else 'cpu')
    v14 = v41.v41()
    v42, v42, v43, v44 = v45()
    v15 = v81.v46(v33(v111.v82))
    v16 = v15.v47()
    v17 = v15.v83(v84) or 0
    v85.v48(v15, v43, v17, v16)
    v18 = v112(v44, v16).v49(v13)
    v18.v50(v80.v113(v1, map_location=v13, weights_only=False)['model'])
    v18.v51()
    for v19 in v18.v52():
        v19.v86(False)
    v20 = v53(v18, v43, v13)
    v21 = v54((v19.v114() for v19 in v18.v52()))
    v55(f'Stage283c anchor-block AND start {v129.v127(v130.v128).v106()} device={v13} rungs={v12} mind_params={v21} merge_mode={v5}')
    if not v11.v99.v87():
        v55(f'  corpus not found: {v11.v99}')
        return 1
    with v11.v99.v75('r', encoding='utf-8', errors='ignore') as v36:
        v56 = v36.v88(v115((v124 for v124, v42 in v12)))
    v55(f'  corpus {v126(v56) / 1000000.0:.1f}M chars read ({v41.v41() - v14:.0f}s)')
    v22 = []
    for v57, v58 in v12:
        v59 = v41.v41()
        v60 = v116.v89(v3)
        v61 = [v118.v117() for v118 in v56[:v57].v125('\n') if 80 <= v126(v118.v117()) <= 400]
        v62 = v119.v90(v61, bank=v20, tok=v15, pad_id=v17, device=v13, rng=v60, n_addr=v58, min_mentions=v11.v102, tau=v11.v96, overlap=v11.v100, soft_match=v11.v101, min_per_family=0, addr_key=v11.v70, merge_mode=v5)
        v63 = v41.v41() - v59
        if not v62['items']:
            v55(f'  rung {v57}:{v58} produced no items, skipped')
            continue
        v64 = v109.v91(v62, v20, v60, v11.v92, v11.v93, v11.v94, v11.v95, v11.v103 == 'on', v11.v96)
        v64.v97({'chars': v57, 'n_addr_requested': v58, 'build_s': v63, 'mind_params': v21, 'lines': v126(v61), 'merge_mode': v5})
        v22.v98(v64)
        v55(f"  rung {v57 / 1000000.0:.0f}M:{v58} -> {v64['n_addresses']} addr / {v64['n_slots']} slots | precision {v64['retrieval_precision']:.3f} | return true {v64['return_path_true']:.3f} other {v64['return_path_other']:.3f} (sep {v64['return_path_separation']:+.3f}) | crowding {v64['address_crowding']:.3f} (members {v64['member_crowding']:.3f} two {v64['two_channel_crowding']:.3f} anchors alone {v64['anchor_only_crowding']:.3f}) foreign {v64['foreign_member_rate']:.3f} | k covers {v64['k_covers_address']:.3f} | ties {v64['families_natural']['tie']:.3f} clean {v64['families_natural']['clean']:.3f} ({v63:.0f}s)")
    if not v22:
        v55('  nothing measured')
        return 1
    v23 = v22[-1]
    v24 = v22[0]
    v25 = v23['retrieval_precision'] >= v11.v65
    v26 = v23['return_path_separation'] >= v11.v66
    v27 = v23['foreign_member_rate'] <= v11.v67
    v28 = v23['families_natural']['tie'] > 0.0
    v29 = v68((v64['mind_params'] == v21 for v64 in v22))
    v30 = 'SCALE_OK' if v68((v25, v26, v27, v28)) else 'SCALE_PARTIAL' if v25 and v28 else 'SCALE_NO'
    v31 = {'stage': '283c', 'overall': v30, 'seed': v3, 'smoke': v11.v69, 'corpus': v33(v11.v99), 'merge_mode': v5, 'link': 'and', 'overlap_from': 'rel', 'addr_key': v11.v70, 'address': {'tau': v11.v96, 'overlap': v11.v100, 'soft_match': v11.v101, 'min_mentions': v11.v102}, 'retrieval': {'topk': v11.v92, 'hop': v11.v93, 'hop_min': v11.v94, 'k_gap': v11.v95, 'subject_filter': v11.v103}, 'thresholds': {'min_precision': v11.v65, 'min_separation': v11.v66, 'max_crowding': v11.v104, 'max_foreign': v11.v67}, 'mind_params': v21, 'gates': {'G_precision_holds': v25, 'G_return_path_separates': v26, 'G_addresses_stay_distinct': v27, 'G_k_covers_the_address': v23['k_covers_address'] >= 0.9, 'G_ties_exist_naturally': v28, 'G_mind_does_not_grow': v29}, 'slopes': {'slots': [v64['n_slots'] for v64 in v22], 'precision': [v64['retrieval_precision'] for v64 in v22], 'return_path_separation': [v64['return_path_separation'] for v64 in v22], 'address_crowding': [v64['address_crowding'] for v64 in v22], 'member_crowding': [v64['member_crowding'] for v64 in v22], 'two_channel_crowding': [v64['two_channel_crowding'] for v64 in v22], 'foreign_member_rate': [v64['foreign_member_rate'] for v64 in v22], 'distinct_anchors': [v64['distinct_anchors'] for v64 in v22], 'anchor_only_crowding': [v64['anchor_only_crowding'] for v64 in v22], 'anchor_growth_exponent': v132.v55(v115(1, v23['distinct_anchors']) / v115(1, v24['distinct_anchors'])) / v132.v55(v23['chars'] / v115(1, v24['chars'])) if v126(v22) > 1 and v23['chars'] != v24['chars'] else v78('nan'), 'k_covers_address': [v64['k_covers_address'] for v64 in v22], 'tie_share': [v64['families_natural']['tie'] for v64 in v22], 'clean_share': [v64['families_natural']['clean'] for v64 in v22], 'slots_growth': v23['n_slots'] / v115(1, v24['n_slots'])}, 'rungs': v22, 'fp_version': v120.v105(), 'reference_283b': {'n_addresses': 2000, 'n_slots': 11074, 'foreign_member_rate': 0.0, 'tie': 0.6955, 'clean': 0.126}, 'reference_write2': {'n_addresses': 877, 'n_slots': 2765, 'foreign_member_rate': 0.00860178354884333, 'tie': 0.34891676168757124, 'clean': 0.5245153933865451}, 'reference_geom': {'n_addresses': 936, 'n_slots': 4730, 'foreign_member_rate': 0.05161396505477364}, 'note': "Letter step 283c: hard anchor block as in 283b; inside the block, edge = relation-tail overlap AND two-channel cos>=tau. Fixes 283b's tie explosion from OR+ctx overlap without retuning tau. Pass if foreign stays flat/near-zero, tie falls toward write2, slots stay above write2.", 'timestamp': v129.v127(v130.v128).v106(), 'wall_s': v41.v41() - v14}
    v0.v35(parents=True, exist_ok=True)
    (v0 / 'stage283c_decision.json').v40(v121.v107(v31, indent=2), encoding='utf-8')
    v55(v121.v107({'overall': v30, 'gates': v31['gates'], 'slopes': v31['slopes']}, indent=2))
    return 0
if v32 == '__main__':
    raise v71(v108())