"""
Stage 230 — Resolution policy on contradictory slots (229 follow-up).

229 showed multi-hit + small gaps; raw argmax always picked the first-written value.
This stage tests **upper-layer** policies (provenance / recency / query cue / composite).

  python _stage230_slot_resolution.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import resolve_slot_contradiction, subject_slot_hits
v0 = v10('results')
v1 = v0 / 'stage230_decision.json'
v2 = v0 / 'stage230_mini.md'
v3 = v10('checkpoints/stage191_p1_curve.pt')
v4 = v10('data/_wikitext103_train.txt')
v5 = 230
v6 = 'In the report {S} was linked to the organization.'
v7 = 'Per the 1987 official records, {S} was linked to the organization.'
v8 = 'Per the 1999 revision, {S} was linked to the organization.'

def main() -> v9:
    v11 = v77.v39()
    v11.v40('--smoke', action='store_true')
    v12 = v11.v41()
    v13 = v78.v13('cuda' if v78.v117.v101() else 'cpu')
    v14 = 8 if v12.v42 else 30
    v15 = v79.v43(v5)
    v44, v45, v46, v47 = v48()
    v16 = v80.v49(v81(v102.v82))
    v17 = v103(v47, v16.v118()).v50(v13)
    v17.v51(v78.v104(v3, map_location=v13, weights_only=False)['model'])
    v17.v52()
    v18 = v53(v17, v46, v13)
    with v4.v83('r', encoding='utf-8', errors='ignore') as v54:
        v55 = v84(v119.v105((v127.v126(1) for v127 in v131.v130(v54.v132(4000000)) if v86(v127.v126(1)) >= 5)))
    v19 = v85(v106(v55), v15, v14 + 5)[:v14]
    v20 = v55[:v14]
    v21 = v55[v14:2 * v14]
    if v86(v21) < v14:
        v21 = v84(v107(v55[:v14]))
    v56, v57, v58, v29 = ([], [], [], [])
    for v59, v60, v61 in v62(v19, v20, v21):
        v63 = f'Official records state {v59} was director of {v60} in 1987 .'
        v64 = f'Later revision claims {v59} was director of {v61} in 1999 .'
        v65 = v18.v108([v59])[0]
        v66 = v18.v87(v63, exclude=v60)
        v67 = v18.v87(v64, exclude=v61)
        if v66 is None or v67 is None:
            continue
        v68 = v86(v56)
        v56.v88(v120.v109(v65 + v66, dim=-1))
        v57.v88(v60)
        v58.v88({'provenance': 'official', 'year': 1987, 'subject': v59})
        v69 = v86(v56)
        v56.v88(v120.v109(v65 + v67, dim=-1))
        v57.v88(v61)
        v58.v88({'provenance': 'revision', 'year': 1999, 'subject': v59})
        v29.v88((v59, v60, v61, [v68, v69]))
    v22 = v78.v70(v56, 0)
    v23 = ['argmax', 'recency', 'query_cue', 'composite']
    v24 = [('neutral', v6, lambda v60, v61: v61), ('official_cue', v7, lambda v60, v61: v60), ('revision_cue', v8, lambda v60, v61: v61)]
    v25 = {v71: {v72: 0 for v72, v110, v110 in v24} for v71 in v23}
    v26 = {v72: 0 for v72, v110, v110 in v24}
    v27 = 0
    v28 = 0
    for v59, v60, v61, v73 in v29:
        for v89, v90, v91 in v24:
            v92 = v90.v111(S=v59)
            v93 = v18.v87(v92, exclude=None)
            if v93 is None:
                continue
            v94 = v112(v22, v57, v93, v73, v58)
            v95 = [v121.v113 for v121 in v94[:2]]
            v27 += v9(v60 in v95 and v61 in v95)
            v96 = v91(v60, v61)
            for v71 in v23:
                v114 = v122(v94, v92, policy=v71)
                v25[v71][v89] += v9(v114 == v96)
            v26[v89] += 1
            v28 += 1
    v30 = {v71: {v72: v25[v71][v72] / v97(1, v26[v72]) for v72 in v26} for v71 in v23}
    v31 = {v71: v115(v30[v71].v123()) / v97(1, v86(v30[v71])) for v71 in v23}
    v32 = v27 / v97(1, v28)
    v33 = v30['query_cue']['official_cue'] >= 0.85 and v30['query_cue']['revision_cue'] >= 0.85
    v34 = v31['composite'] >= v31['argmax'] + 0.1
    v35 = v30['composite']['neutral'] >= 0.7
    v36 = 'RESOLUTION_POLICY_OK' if v33 and v34 and v35 else 'RESOLUTION_POLICY_PARTIAL' if v33 or v34 else 'RESOLUTION_POLICY_NO'
    v37 = {'stage': 230, 'overall': v36, 'gates': {'G_query_cue_cued_ge_0p85': v33, 'G_composite_beats_argmax_macro': v34, 'G_composite_neutral_ge_0p70': v35}, 'rate_both_values_in_top2': v32, 'accuracy_by_policy_suite': v30, 'macro_accuracy': v31, 'argmax_bias_note': '229: argmax always preferred first-written official slot', 'timestamp': v128.v124(v129.v125).v98()}
    v1.v74(v116.v99(v37, indent=2), encoding='utf-8')
    v2.v74(f"# Stage 230 resolution\n\n**{v36}** macro composite={v31['composite']:.3f} argmax={v31['argmax']:.3f} cue={v30['query_cue']['revision_cue']:.3f}\n", encoding='utf-8')
    v75(v116.v99(v37, indent=2))
    return 0
if v38 == '__main__':
    raise v76(v100())