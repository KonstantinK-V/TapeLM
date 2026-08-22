"""
Stage 173 — Falsify: language signal vs orthographic trajectory.

Same frozen pen (170) + trained dyn (172). Do NOT retrain.
Eval dynamics on hold windows under corruptions that destroy content
but preserve / destroy form.

  python _stage173_curve_falsify.py
"""
from __future__ import annotations
import json
import random
import string
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
v3 = v0 / '_stage173_log.txt'
v4 = v0 / 'stage173_falsify_decision.json'
v5 = v0 / 'stage173_falsify_mini.md'
v6 = 173
v7 = 96
v8 = (1, 8, 16)

def log(v17: v11) -> None:
    v18 = v17 if v17.v132('\n') else v17 + '\n'
    try:
        v133(v18, end='', flush=True)
    except v73:
        v133(v18.v144('ascii', 'replace').v181('ascii'), end='', flush=True)
    v3.v134.v74(parents=True, exist_ok=True)
    with v3.v135('a', encoding='utf-8') as v75:
        v75.v136(v18)
v9 = v19.v10

def corr_natural(v20: v11, v21: v137.v76) -> v11:
    return v20

def corr_shuffle_all(v20: v11, v21: v137.v76) -> v11:
    v22 = v77(v20)
    v21.v78(v22)
    return ''.v79(v22)

def corr_shuffle_letters_keep_skeleton(v20: v11, v21: v137.v76) -> v11:
    """Keep spaces/punct/digits positions; shuffle only letters among themselves."""
    v22 = v77(v20)
    v23 = [v80 for v80, v26 in v164(v22) if v26.v138()]
    v24 = [v22[v80] for v80 in v23]
    v21.v78(v24)
    for v80, v81 in v82(v23, v24):
        v22[v80] = v81
    return ''.v79(v22)

def corr_random_letters_keep_skeleton(v20: v11, v21: v137.v76) -> v11:
    """Keep whitespace/punct skeleton; replace every letter/digit with random letter."""
    v25 = []
    for v26 in v20:
        if v26.v138():
            v25.v153(v21.v182(v9))
        elif v26.v165():
            v25.v153(v21.v182(v9))
        else:
            v25.v153(v26)
    return ''.v79(v25)

def corr_destroy_spaces(v20: v11, v21: v137.v76) -> v11:
    """Replace spaces with random letters — kill separator turns."""
    return ''.v79((v21.v182(v9) if v26 == ' ' else v26 for v26 in v20))

def corr_only_spaces_punct(v20: v11, v21: v137.v76) -> v11:
    """Letters → 'a'; keep spaces/punct — almost pure skeleton rhythm."""
    return ''.v79(('a' if v26.v183() else v26 for v26 in v20))

def corr_reverse(v20: v11, v21: v137.v76) -> v11:
    return v20[::-1]
v12 = [('natural', v83, 'real text'), ('shuffle_all', v84, 'destroy order entirely'), ('shuffle_letters_keep_skel', v85, 'keep spaces/punct; scramble letters'), ('random_letters_keep_skel', v86, 'same skeleton; random letters'), ('only_skel_flat_letters', v87, "skeleton + all letters='a'"), ('destroy_spaces', v88, 'spaces → random letters'), ('reverse', v89, 'reversed string')]

@v109.v45()
def eval_condition(v27, v28: v48.v14, v29, v30: v15, v31: v15) -> v13:
    """Same metrics spirit as 172 hold, on a full char-id stream."""
    v27.v139.v90()
    v27.v140.v90()
    v21 = v137.v76(v31)
    v32 = v91(v28)
    v33 = v92.v34
    v35 = v92.v36
    v37 = v15(0.85 * v32)
    v38 = v32 - v33 - 2
    if v38 <= v37:
        v37 = v141(0, v15(0.5 * v32))
    v39 = {v44: [] for v44 in v8}
    for v40 in v93(24):
        v20 = v21.v142(0, v141(1, v37 - v33 - 2))
        v94 = v109.v143(v28[v20:v20 + v33][None].v166(v48.v167), device=v29)
        v95 = v27.v144(v94)
        for v44 in v8:
            v39[v44].v153((v95[:, v44:] - v95[:, :-v44]).v145(dim=(0, 1)))
    v41 = {v44: v109.v184(v117, 0).v145(0) for v44, v117 in v39.v168()}
    v42 = {v44: {'cos': [], 'base_mean': [], 'base_copy': []} for v44 in v8}
    v43 = []
    for v40 in v93(v30):
        v20 = v37 + v21.v142(0, v141(1, v38 - v37))
        v20 = v146(v20, v38)
        v94 = v109.v143(v28[v20:v20 + v33][None].v166(v48.v167), device=v29)
        v95 = v27.v144(v94)
        v96 = v95.v147(1)
        v97 = v141(v35, v96 - 1 - v141(v8) - 1)
        v98 = v95[:, v97 + 1 - v35:v97 + 1]
        if v98.v147(1) < v35:
            v148 = v98[:, :1].v169(1, v35 - v98.v147(1), -1)
            v98 = v109.v170([v148, v98], dim=1)
        v99 = v27.v140(v98)
        for v44 in v8:
            if v97 + v44 >= v96:
                continue
            v149 = v95[:, v97 + v44] - v95[:, v97]
            v150 = v99[f'delta_{v44}']
            v151 = v95[:, v97] - v95[:, v141(0, v97 - v44)]
            v42[v44]['cos'].v153(v172(v171.v191(v150, v149, dim=-1).v145()))
            v42[v44]['base_mean'].v153(v172(v171.v191(v41[v44].v192(0), v149, dim=-1).v145()))
            v42[v44]['base_copy'].v153(v172(v171.v191(v151, v149, dim=-1).v145()))
        v100 = 8
        v101 = v171.v152(v95[:, v97 + 1:v97 + 1 + v100].v145(dim=1), dim=-1)
        v102 = v171.v152(v95[:, v35:v35 + v100].v145(dim=1), dim=-1)
        v103 = v99['arc']
        v104 = v146(v103.v147(-1), v101.v147(-1))
        v103, v101, v102 = (v171.v152(v103[:, :v104], dim=-1), v101[:, :v104], v102[:, :v104])
        v43.v153(v172((v171.v191(v103, v101) > v171.v191(v103, v102)).v172().v145()))

    def avg(v105):
        return v173(v105) / v141(v91(v105), 1)
    v25 = {'contrast_pref': v154(v43)}
    for v44 in v8:
        v106 = v154(v42[v44]['cos'])
        v107 = v154(v42[v44]['base_mean'])
        v108 = v154(v42[v44]['base_copy'])
        v25[f'k{v44}'] = {'cos_delta': v106, 'lift_mean': v106 - v107, 'lift_copy': v106 - v108}
    v25['score'] = v25['k1']['cos_delta'] + 0.5 * v25['k16']['cos_delta'] + 0.3 * v25['contrast_pref']
    return v25

def encode_stream(v46: v11, v47: v13) -> v48.v14:
    return v48.v110((v47.v174(v26, 0) for v26 in v46), dtype=v48.v155, count=v91(v46))

def main() -> v15:
    v0.v74(parents=True, exist_ok=True)
    v3.v111('', encoding='utf-8')
    v112(f'Stage173 falsify start {v189.v186(v190.v187).v161()}')
    v112('Q: language vs orthographic trajectory? (no retrain; frozen pen + 172 dyn)')
    if not v1.v175() or not v2.v175():
        v112(f'FATAL missing ckpt pen={v1.v175()} dyn={v2.v175()}')
        return 1
    v29 = v109.v29('cuda' if v109.v185.v176() else 'cpu')
    v49 = v109.v113(v1, map_location='cpu', weights_only=False)
    v50 = v109.v113(v2, map_location='cpu', weights_only=False)
    v47, v114 = (v49['stoi'], v49['itos'])
    v27 = v92.v177(v91(v114)).v115(v29)
    v51 = v50['model']
    v27.v116(v51, strict=False)
    v52 = {v44[v91('pen.'):]: v117 for v44, v117 in v49['model'].v168() if v44.v178('pen.')}
    v27.v139.v116(v52, strict=True)
    for v53 in v27.v139.v118():
        v53.v156(False)
    v27.v139.v90()
    v27.v140.v90()
    v112(f"loaded pen@170 + dyn@172 step={v50.v174('step')} device={v29}")
    v54 = v157.v119(max_chars=5000000)
    v55 = v54[1000000:3000000]
    v112(f'base slice chars={v91(v55)}')
    v56 = v137.v76(v6)
    v57 = {}
    v58 = None
    for v120, v121, v122 in v12:
        v21 = v137.v76(v6 + v188(v120) % 10000)
        v123 = v121(v55, v21)
        v124 = v158(v123, v47)
        v125 = v159(v27, v124, v29, v7, seed=v6 + v91(v120))
        v57[v120] = {'desc': v122, **v125}
        v112(f"  [{v120}] k1={v125['k1']['cos_delta']:.3f} k8={v125['k8']['cos_delta']:.3f} k16={v125['k16']['cos_delta']:.3f} contrast={v125['contrast_pref']:.3f} score={v125['score']:.3f} | {v122}")
        if v120 == 'natural':
            v58 = v125['score']
    v59 = v57['natural']
    v60 = v57['random_letters_keep_skel']
    v61 = v57['only_skel_flat_letters']
    v62 = v57['shuffle_all']
    v63 = v57['destroy_spaces']
    v64 = v57['shuffle_letters_keep_skel']

    def close(v126, v127, v128=0.05):
        return v179(v126 - v127) <= v128
    v65 = v59['k1']['cos_delta']
    v66 = {'gap_shuffle_all': v65 - v62['k1']['cos_delta'], 'gap_random_letters_skel': v65 - v60['k1']['cos_delta'], 'gap_flat_skel': v65 - v61['k1']['cos_delta'], 'gap_destroy_spaces': v65 - v63['k1']['cos_delta'], 'gap_shuffle_letters': v65 - v64['k1']['cos_delta']}
    v67 = v160(v65, v60['k1']['cos_delta'], 0.07) or v160(v65, v61['k1']['cos_delta'], 0.07)
    v68 = v66['gap_shuffle_all'] > 0.15
    v69 = v66['gap_destroy_spaces'] > 0.1
    v70 = v66['gap_random_letters_skel'] > 0.1 and v66['gap_shuffle_letters'] > 0.08
    if v67 and v69 and (not v70):
        v129 = 'MOSTLY_ORTHOGRAPHIC_SKELETON'
        v130 = 'Dynamics nearly as good on skeleton/random letters; spaces matter; letter identity weak.'
    elif v70 and v68:
        v129 = 'LETTER_SEQUENCE_SENSITIVE'
        v130 = 'Real letter order helps beyond skeleton — still may be orthographic n-grams, not semantics.'
    elif v67 and (not v69):
        v129 = 'GENERIC_SMOOTH_TRAJECTORY'
        v130 = 'Even skeleton/space ablations stay high — likely easy local curve predictability.'
    else:
        v129 = 'MIXED_FORM_SIGNAL'
        v130 = 'Partial drops under corruptions; form-heavy with some letter-path sensitivity.'
    v25 = {'timestamp': v189.v186(v190.v187).v161(), 'protocol': 'curve_falsify_ortho_vs_language', 'verdict_reading': v129, 'detail': v130, 'gaps_k1_vs_natural': v66, 'conditions': v57, 'natural_k1': v65, 'flags': {'skel_close_to_natural': v67, 'shuffle_kills': v68, 'spaces_matter': v69, 'letters_matter': v70}, 'caveat': 'LETTER_SEQUENCE_SENSITIVE ≠ language understanding. It only means char-order of real text affects Δ predictability under this pen.', 'next': 'If MOSTLY_ORTHOGRAPHIC_*: need non-char pen or semantic probes. If LETTER_SEQUENCE_SENSITIVE: still need meaning falsify (paraphrase/cross-lingual) later.'}
    v4.v111(v180.v162(v25, indent=2), encoding='utf-8')
    v71 = [f'reading `{v129}`', v130, f"natural k1={v65:.3f} | random_skel k1={v60['k1']['cos_delta']:.3f} | flat_skel k1={v61['k1']['cos_delta']:.3f}", f"shuffle_all k1={v62['k1']['cos_delta']:.3f} | destroy_spaces k1={v63['k1']['cos_delta']:.3f}", f"gaps={{{', '.v79((f'{v44}={v117:+.3f}' for v44, v117 in v66.v168()))}}}", v25['caveat']]
    v5.v111('\n'.v79(['# Stage173 — orthography vs language falsify', '', f'**Reading:** `{v129}`', ''] + [f'- {v127}' for v127 in v71] + ['']), encoding='utf-8')
    v112(f'[173] {v129}')
    v112(v130)
    return 0
if v72 == '__main__':
    raise v131(v163())