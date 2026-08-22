"""
Stage 174 — Early context/meaning falsify on frozen curve (no dyn retrain).

A) same letter-suffix, different prefix → does z still differ?
B) paraphrase pairs vs random pairs → is z closer for same meaning?
C) sentence-order shuffle (local orthography kept, discourse broken) → does z/dyn care?

  python _stage174_curve_context_falsify.py
"""
from __future__ import annotations
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import _stage170_curve_dynamics as s170
import _stage172_curve_scale as s172
v0 = v16('results')
v1 = v16('checkpoints/stage170_curve.pt')
v2 = v16('checkpoints/stage172_curve.pt')
v3 = v0 / '_stage174_log.txt'
v4 = v0 / 'stage174_context_falsify_decision.json'
v5 = v0 / 'stage174_context_falsify_mini.md'
v6 = 174
v7 = 24
v8 = 96
v9 = 120
v10 = 120

def log(v17: v59) -> None:
    v18 = v17 if v17.v134('\n') else v17 + '\n'
    try:
        v135(v18, end='', flush=True)
    except v83:
        v135(v18.v140('ascii', 'replace').v173('ascii'), end='', flush=True)
    v3.v136.v84(parents=True, exist_ok=True)
    with v3.v137('a', encoding='utf-8') as v85:
        v85.v138(v18)

def to_ids(v19: v59, v20: v13) -> v21.v11:
    return v21.v86([v20.v163(v70, 0) for v70 in v19], dtype=v21.v139)

@v21.v25()
def encode_z(v22, v19: v59, v20: v13, v23) -> v21.v11:
    v24 = v179(v19, v20).v164(0).v87(v23)
    return v22.v140(v24)[0]

def z_summary(v26: v21.v11) -> v21.v11:
    """Endpoint + mean pool — fixed vector for comparisons."""
    return v141.v88(v21.v142([v26[-1], v26.v150(0)], dim=0), dim=0)

def cos(v27: v21.v11, v28: v21.v11) -> v12:
    return v12(v141.v174(v27.v164(0), v28.v164(0)).v143())

def mine_same_suffix_pairs(v19: v59, v29: v144.v89):
    """Windows ending with identical last SUFFIX_LEN chars, different earlier PREFIX_LEN."""
    v30 = v8 + v7
    v31 = v90(v58)
    v32 = 17
    for v33 in v91(0, v147(v19) - v30 - 1, v32):
        v92 = v19[v33:v33 + v30]
        v93 = v92[-v7:]
        v94 = v92[:-v7]
        v31[v93].v145((v33, v94, v92))
    v34 = []
    for v93, v95 in v31.v95():
        if v147(v95) < 2:
            continue
        v96 = {}
        for v33, v94, v92 in v95:
            if v94 not in v96:
                v96[v94] = v92
            if v147(v96) >= 2:
                break
        if v147(v96) < 2:
            continue
        v97 = v58(v96.v165())
        v34.v145((v97[0], v97[1], v93))
        if v147(v34) >= v9 * 2:
            break
    v29.v98(v34)
    return v34[:v9]

def mine_diff_suffix_pairs(v19: v59, v29: v144.v89, v35: v15):
    v30 = v8 + v7
    v34 = []
    for v36 in v91(v35 * 3):
        v33 = v29.v146(0, v147(v19) - v30 - 1)
        v99 = v29.v146(0, v147(v19) - v30 - 1)
        v27, v28 = (v19[v33:v33 + v30], v19[v99:v99 + v30])
        if v27[-v7:] == v28[-v7:]:
            continue
        v34.v145((v27, v28))
        if v147(v34) >= v35:
            break
    return v34

@v21.v25()
def test_A(v22, v19, v20, v23, v29) -> v13:
    v100('### A) same letter-suffix, different prefix')
    v37 = v101(v19, v29)
    v38 = v102(v19, v29, v10)
    v100(f'  pairs same_suffix={v147(v37)} diff_suffix={v147(v38)}')
    v103, v104 = ([], [])
    for v27, v28, v93 in v37:
        v106, v107 = (v166(v22, v27, v20, v23), v166(v22, v28, v20, v23))
        v103.v145(v167(v141.v88(v106[-1], dim=0), v141.v88(v107[-1], dim=0)))
    for v27, v28 in v38:
        v106, v107 = (v166(v22, v27, v20, v23), v166(v22, v28, v20, v23))
        v104.v145(v167(v141.v88(v106[-1], dim=0), v141.v88(v107[-1], dim=0)))
    v39 = []
    for v27, v28, v93 in v37[:80]:
        v106, v107 = (v166(v22, v27, v20, v23), v166(v22, v28, v20, v23))
        v33 = v8 - 1
        v39.v145(v167(v141.v88(v106[v33], dim=0), v141.v88(v107[v33], dim=0)))
    v40 = v12(v168.v150(v103)) if v103 else 0.0
    v41 = v12(v168.v150(v104)) if v104 else 0.0
    v42 = v12(v168.v150(v39)) if v39 else 0.0
    v43 = v40 - v41 > 0.15 and v40 - v42 > 0.1
    v44 = v40 - v41 < 0.05
    if v43:
        v105 = 'A_FAIL_CONTEXT_WIPED_BY_SUFFIX'
    elif v44:
        v105 = 'A_PASS_PREFIX_STILL_VISIBLE'
    else:
        v105 = 'A_WEAK_PARTIAL_TRACE'
    v45 = {'verdict': v105, 'n_same': v147(v103), 'n_diff': v147(v104), 'mean_cos_endpoint_same_suffix': v40, 'mean_cos_endpoint_diff_suffix': v41, 'mean_cos_at_prefix_end_same_suf_pairs': v42, 'delta_same_minus_diff': v40 - v41}
    v100(f'  endpoint cos same_suf={v40:.3f} diff_suf={v41:.3f} at_prefix_end={v42:.3f} → {v105}')
    return v45
v14 = [('The cat sat on the mat.', 'A cat was sitting on the mat.'), ('She quickly opened the door.', 'She opened the door quickly.'), ('He bought a new car yesterday.', 'Yesterday he purchased a new automobile.'), ('The weather is very cold today.', 'It is extremely chilly outside today.'), ('Children are playing in the park.', 'Kids are playing at the park.'), ('I need to finish this work soon.', 'I must complete this task shortly.'), ('The book was written by a famous author.', 'A famous writer wrote the book.'), ('They arrived at the station early.', 'They got to the station early.'), ('Water boils at one hundred degrees.', 'Water boils at 100 degrees.'), ('The dog chased the ball across the yard.', 'Across the yard the dog ran after the ball.'), ('Please close the window.', 'Could you shut the window?'), ('The train leaves at noon.', 'The train departs at midday.'), ('He is afraid of spiders.', 'Spiders scare him.'), ('She teaches mathematics at school.', 'She is a math teacher at the school.'), ('The film was long and boring.', 'The movie was lengthy and dull.'), ('We should start the meeting now.', "Let's begin the meeting now."), ('The river flows into the sea.', 'The river runs into the ocean.'), ('His answer was completely wrong.', 'His reply was totally incorrect.'), ('The store opens at nine.', 'The shop opens at 9.'), ('Birds fly south in winter.', 'In winter birds migrate south.'), ('The bridge connects the two cities.', 'The two cities are linked by the bridge.'), ('She drank a cup of tea.', 'She had a cup of tea.'), ('The problem is difficult to solve.', 'Solving the problem is hard.'), ('He forgot his keys at home.', 'He left his keys at home.'), ('The sun rises in the east.', 'In the east the sun comes up.')]

@v21.v25()
def test_B(v22, v20, v23, v29) -> v13:
    v100('### B) paraphrase proximity vs random pairs')
    v46 = []
    for v27, v28 in v14:
        v106 = v148(v166(v22, v27, v20, v23))
        v107 = v148(v166(v22, v28, v20, v23))
        v46.v145((v106, v107, v167(v106, v107)))
    v47 = [v70 for v36, v36, v70 in v46]
    v48 = [v106 for v106, v107, v36 in v46] + [v107 for v106, v107, v36 in v46]
    v49 = []
    for v36 in v91(v147(v47) * 4):
        v33, v99 = v29.v149(v91(v147(v48)), 2)
        v49.v145(v167(v48[v33], v48[v99]))
    v50 = [('The cat sat on the mat.', 'The car sat on the mat.'), ('She opened the door quickly.', 'She opened the book quickly.'), ('He bought a new car yesterday.', 'He bought a new cat yesterday.'), ('The weather is very cold today.', 'The weather is very warm today.'), ('Children are playing in the park.', 'Children are studying in the park.'), ('The train leaves at noon.', 'The plane leaves at noon.'), ('Water boils at one hundred degrees.', 'Oil boils at one hundred degrees.'), ('She teaches mathematics at school.', 'She teaches history at school.')]
    v51 = []
    for v27, v28 in v50:
        v51.v145(v167(v148(v166(v22, v27, v20, v23)), v148(v166(v22, v28, v20, v23))))
    v52 = v12(v168.v150(v47))
    v53 = v12(v168.v150(v49))
    v54 = v12(v168.v150(v51))
    v55 = v52 - v53
    v56 = v52 - v54
    if v55 > 0.05 and v56 > 0.03:
        v105 = 'B_PASS_MEANING_STRUCTURE'
    elif v55 > 0.03 and v56 <= 0.02:
        v105 = 'B_FAIL_FORM_NOT_MEANING'
    elif v55 <= 0.02:
        v105 = 'B_FAIL_NO_PARAPHRASE_CLUSTER'
    else:
        v105 = 'B_WEAK_MIXED'
    v45 = {'verdict': v105, 'mean_cos_paraphrase': v52, 'mean_cos_random': v53, 'mean_cos_hard_spelling': v54, 'lift_vs_random': v55, 'lift_vs_hard': v56, 'n_para': v147(v47)}
    v100(f'  para={v52:.3f} random={v53:.3f} hard_spell={v54:.3f} lift_rand={v55:+.3f} lift_hard={v56:+.3f} → {v105}')
    return v45

def split_sentences(v19: v59) -> v58[v59]:
    v57 = v151.v108('(?<=[.!?])\\s+', v19.v152())
    return [v77 for v77 in v57 if v147(v77) > 20]

@v21.v25()
def test_C(v22, v19, v20, v23, v29) -> v13:
    v100('### C) sentence-order shuffle (discourse break, local orthography kept)')
    v60 = []
    v61 = []
    for v18 in v19.v109():
        v18 = v18.v152()
        if not v18:
            if v61:
                v60.v145(' '.v154(v61))
                v61 = []
            continue
        v61.v145(v18)
    if v61:
        v60.v145(' '.v154(v61))
    v62 = []
    for v63 in v60:
        v67 = v153(v63)
        if v147(v67) >= 4:
            v62.v145(v67)
        if v147(v62) >= 80:
            break
    v64 = []
    v65 = []
    v66 = []
    for v67 in v62:
        v110 = ' '.v154(v67)
        v111 = v67[:]
        v29.v98(v111)
        if v111 == v67:
            continue
        v112 = ' '.v154(v111)
        v113 = v110.v108()
        v29.v98(v113)
        v114 = ' '.v154(v113)
        v30 = v155(400, v147(v110), v147(v112), v147(v114))
        v110, v112, v114 = (v110[:v30], v112[:v30], v114[:v30])
        v115 = v148(v166(v22, v110, v20, v23))
        v116 = v148(v166(v22, v112, v20, v23))
        v117 = v148(v166(v22, v114, v20, v23))
        v64.v145(v167(v115, v116))
        v65.v145(v167(v115, v117))
    v68 = v12(v168.v150(v64)) if v64 else 0.0
    v69 = v12(v168.v150(v65)) if v65 else 0.0
    if v68 > 0.9 and v68 - v69 > 0.08:
        v105 = 'C_FAIL_DISCOURSE_BLIND_LOCAL_ORTHO'
    elif v68 < 0.75:
        v105 = 'C_PASS_ORDER_SENSITIVE'
    else:
        v105 = 'C_WEAK_PARTIAL_ORDER'
    v45 = {'verdict': v105, 'n': v147(v64), 'mean_cos_natural_vs_sentence_shuffle': v68, 'mean_cos_natural_vs_word_shuffle': v69, 'gap_sent_minus_word': v68 - v69}
    v100(f'  cos(nat,sent_shuf)={v68:.3f} cos(nat,word_shuf)={v69:.3f} → {v105}')
    return v45

def combine_verdict(v27, v28, v70) -> v73[v59, v59]:
    v71 = v118((1 for v123 in (v27, v28, v70) if 'FAIL' in v123))
    v72 = v118((1 for v123 in (v27, v28, v70) if 'PASS' in v123))
    if v71 >= 2 and v72 == 0:
        return ('CONTEXT_WALL_ON_CURVE', 'Curve holds letter-path form but fails early context/meaning probes — do not scale dyn.')
    if v72 >= 2:
        return ('CONTEXT_SIGNAL_POSSIBLE', 'Enough signal to justify careful dyn/pen work aimed at these probes.')
    return ('CONTEXT_UNCLEAR_MIXED', 'Mixed A/B/C — one more targeted probe before any long soak.')

def main() -> v15:
    v0.v84(parents=True, exist_ok=True)
    v3.v119('', encoding='utf-8')
    v100(f'Stage174 start {v180.v177(v181.v178).v160()}')
    v100('Early context/meaning falsify on frozen pen@170 + dyn@172 (no retrain)')
    v23 = v21.v23('cuda' if v21.v175.v169() else 'cpu')
    v74 = v21.v120(v1, map_location='cpu', weights_only=False)
    v75 = v21.v120(v2, map_location='cpu', weights_only=False)
    v20, v121 = (v74['stoi'], v74['itos'])
    v22 = v176.v170(v147(v121)).v87(v23)
    v22.v122(v75['model'], strict=False)
    v76 = {v156[v147('pen.'):]: v123 for v156, v123 in v74['model'].v95() if v156.v171('pen.')}
    v22.v157.v122(v76, strict=True)
    for v77 in v22.v157.v124():
        v77.v158(False)
    v22.v125()
    v100(f"device={v23} dyn_step={v75.v163('step')}")
    v19 = v159.v126(max_chars=8000000)
    v19 = v19[2000000:6000000]
    v29 = v144.v89(v6)
    v78 = v127(v22, v19, v20, v23, v29)
    v79 = v128(v22, v20, v23, v29)
    v80 = v129(v22, v19, v20, v23, v29)
    v130, v131 = v132(v78['verdict'], v79['verdict'], v80['verdict'])
    v45 = {'timestamp': v180.v177(v181.v178).v160(), 'protocol': 'curve_context_falsify_174', 'overall': v130, 'detail': v131, 'A_same_suffix': v78, 'B_paraphrase': v79, 'C_sentence_shuffle': v80, 'recommendation': 'STOP scaling dyn; redesign pen/object or accept script-engine role' if v130 == 'CONTEXT_WALL_ON_CURVE' else 'Proceed carefully with probes as gates' if v130 == 'CONTEXT_SIGNAL_POSSIBLE' else 'One more falsify before investment'}
    v4.v119(v172.v161(v45, indent=2), encoding='utf-8')
    v81 = [f'overall `{v130}`', v131, f"A: {v78['verdict']} (same_suf_end={v78['mean_cos_endpoint_same_suffix']:.3f} diff={v78['mean_cos_endpoint_diff_suffix']:.3f} pref_end={v78['mean_cos_at_prefix_end_same_suf_pairs']:.3f})", f"B: {v79['verdict']} (para={v79['mean_cos_paraphrase']:.3f} rand={v79['mean_cos_random']:.3f} hard={v79['mean_cos_hard_spelling']:.3f})", f"C: {v80['verdict']} (nat~sent_shuf={v80['mean_cos_natural_vs_sentence_shuffle']:.3f} nat~word_shuf={v80['mean_cos_natural_vs_word_shuffle']:.3f})", f"recommendation: {v45['recommendation']}"]
    v5.v119('\n'.v154(['# Stage174 — context/meaning early falsify', '', f'**Overall:** `{v130}`', ''] + [f'- {v28}' for v28 in v81] + ['']), encoding='utf-8')
    v100(f'[174] {v130}')
    v100(v131)
    v100(v45['recommendation'])
    return 0
if v82 == '__main__':
    raise v133(v162())