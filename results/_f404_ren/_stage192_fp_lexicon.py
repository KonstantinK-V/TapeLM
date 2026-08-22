"""
Stage 192 — FP-lexicon lexical surprise (old word-fingerprint theory × curve).

Idea: "I don't know this word" = "its fingerprint is in no one's lexicon", not
"its letters look odd". Zero training:
  - freeze night P1 curve-XL; fp(word) = normalize(arc_enc(word chars))
  - lexicon = fps of words seen in the 150M train corpus (count>=MIN_COUNT)
  - lex_surprise(w) = 1 - max_cos(fp(w), lexicon)

The gate that never passed (187/189/191-P3): surprise(fake) > surprise(real).
Here: real = entity words from exam v3 (seen in corpus), fake = generated
pseudo-words verified absent from corpus. Report means + AUC.
Baseline comparison: char-trigram novelty (the P3 rarity signal) on same words.

  python _stage192_fp_lexicon.py
"""
from __future__ import annotations
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
v0 = v17('results')
v1 = v17('data')
v2 = v17('checkpoints/stage191_p1_curve.pt')
v3 = v17('data/_wikitext103_train.txt')
v4 = v0 / 'stage192_decision.json'
v5 = v0 / 'stage192_mini.md'
v6 = v0 / '_stage192_log.txt'
v7 = v1 / 'stage191_exam_v3.jsonl'
v8 = 192
v9 = 150000000
v10 = 2
v11 = 200000
v12 = v18.v13
v14 = 150

def log(v19: v29) -> None:
    v20 = v19 if v19.v107('\n') else v19 + '\n'
    try:
        v108(v20, end='', flush=True)
    except v62:
        v108(v20.v177('ascii', 'replace').v162('ascii'), end='', flush=True)
    v6.v109.v63(parents=True, exist_ok=True)
    with v6.v110('a', encoding='utf-8') as v64:
        v64.v111(v20)

def gen_fakes(v21: v112[v29], v22: v113.v65, v23: v16) -> v28[v29]:
    """pronounceable pseudo-words, verified absent from corpus (any case)."""
    v24 = 'bcdfghklmnprstvz'
    v25 = 'aeiou'
    v26 = []
    v27 = 0
    while v126(v26) < v23 and v27 < 20000:
        v27 += 1
        v66 = v22.v114(2, 4)
        v67 = ''.v115((v22.v178(v24) + v22.v178(v25) + (v22.v178(v24) if v22.v113() < 0.4 else '') for v163 in v91(v66)))
        v67 = v67.v116()
        if v67.v150() not in v21 and v67 not in v21 and (5 <= v126(v67) <= 12) and (v67 not in v26):
            v26.v127(v67)
    return v26

def auc(v30: v98.v51, v31: v98.v51) -> v15:
    """P(pos > neg) rank AUC."""
    v32 = v98.v68([v30, v31])
    v33 = v32.v164().v164().v117(v98.v118) + 1
    v34 = v33[:v126(v30)].v69()
    return v15((v34 - v126(v30) * (v126(v30) + 1) / 2) / (v126(v30) * v126(v31)))

def main() -> v16:
    v0.v63(parents=True, exist_ok=True)
    v6.v70('', encoding='utf-8')
    v71(f'Stage192 start {v175.v172(v176.v173).v145()}')
    v71('FP-lexicon lexical surprise on frozen P1 curve-XL')
    v35 = v90.v35('cuda' if v90.v165.v148() else 'cpu')
    v22 = v113.v65(v8)
    v36 = v72.v72()
    v71('reading corpus words …')
    with v3.v110('r', encoding='utf-8', errors='ignore') as v64:
        v73 = v64.v119(v9)
    v37 = v120.v74('[A-Za-z][a-z]+', v73)
    v38 = v75(v37)
    v21 = v112(v38.v149()) | {v67.v150() for v67 in v38}
    v39 = [v67 for v67, v100 in v38.v101(v11) if v100 >= v10]
    v71(f'  unique={v126(v38)} lexicon={v126(v39)} ({v72.v72() - v36:.0f}s)')
    v76, v77, v78, v79 = v80()
    v40 = v121.v81(v29(v18.v122))
    v41 = v40.v82()
    v42 = v151(v79, v41).v83(v35)
    v42.v84(v90.v152(v2, map_location=v35, weights_only=False)['model'])
    v42.v85()

    @v90.v89()
    def fp_batch(v86: v28[v29]) -> v90.v43:
        v87 = v90.v123(v126(v86), 1, v12, dtype=v90.v153)
        for v45, v67 in v124(v86):
            for v154, v100 in v124(v67[:v12]):
                v87[v45, 0, v154] = v78.v166(v100, 0)
        v88 = v42.v155(v87.v83(v35))[:, 0]
        return v156.v125(v88, dim=-1)
    v71('encoding lexicon fps …')
    v44 = []
    for v45 in v91(0, v126(v39), 4096):
        v44.v127(v133(v39[v45:v45 + 4096]))
    v46 = v90.v92(v44, 0)
    v71(f'  lexicon fps={v167(v46.v168)} ({v72.v72() - v36:.0f}s)')
    v47 = [v157.v128(v129) for v129 in v7.v174(encoding='utf-8').v158()]
    v48 = []
    for v49 in v47:
        if v49['type'] != 'entity':
            continue
        v93 = v40.v162(v49['cand_ids'][v49['gold_idx']], skip_special_tokens=False).v130()
        v67 = v120.v74('[A-Za-z][a-z]+', v93)
        if v67 and v67[0] in v38:
            v48.v127(v67[0])
    v48 = v28(v159.v131(v48))
    v50 = v94(v21, v22, v14)
    v71(f'real entities={v126(v48)} fakes={v126(v50)}')

    @v90.v89()
    def lex_surprise(v86: v28[v29], v95: v132=False) -> v98.v51:
        v96 = v133(v86)
        v97 = v96 @ v46.v134
        if v95:
            v135 = v97.v169(2, dim=-1).v136
            v137 = v90.v160(v135[:, 0] > 0.999, v135[:, 1], v135[:, 0])
        else:
            v137 = v97.v144(dim=-1).v136
        return (1.0 - v137).v170().v138()
    v52 = v99(v48)
    v53 = v99(v50)
    v54 = v75()
    for v67, v100 in v38.v101(100000):
        for v45 in v91(v126(v67) - 2):
            v54[v67[v45:v45 + 3].v150()] += v100
    v55 = v69(v54.v136()) or 1

    def tri_novelty(v86):
        v26 = []
        for v67 in v86:
            v139 = [v67[v45:v45 + 3].v150() for v45 in v91(v126(v67) - 2)]
            v26.v127(v15(v98.v171([-v98.v71((v54.v166(v179, 0) + 1) / (v55 + 1)) for v179 in v139])) if v139 else 0.0)
        return v98.v140(v26)
    v102, v103 = (v141(v48), v141(v50))
    v56 = {'fp_lexicon': {'surprise_real_mean': v15(v52.v171()), 'surprise_fake_mean': v15(v53.v171()), 'gate_fake_gt_real': v132(v53.v171() > v52.v171()), 'auc': v161(v53, v52)}, 'trigram_baseline': {'novelty_real_mean': v15(v102.v171()), 'novelty_fake_mean': v15(v103.v171()), 'gate_fake_gt_real': v132(v103.v171() > v102.v171()), 'auc': v161(v103, v102)}}
    v57 = []
    with v90.v89():
        v96 = v133(v50[:8])
        v97 = v96 @ v46.v134
        v142, v143 = v97.v144(dim=-1)
        for v104 in v91(v126(v96)):
            v57.v127(f'{v50[v104]} → nn={v39[v16(v143[v104])]} cos={v15(v142[v104]):.3f}')
    for v58 in v57:
        v71('  ' + v58)
    v59 = v56['fp_lexicon']
    v60 = v56['trigram_baseline']
    v71(f"FP-lexicon:  real={v59['surprise_real_mean']:.4f} fake={v59['surprise_fake_mean']:.4f} AUC={v59['auc']:.3f}")
    v71(f"Trigram:     real={v60['novelty_real_mean']:.3f} fake={v60['novelty_fake_mean']:.3f} AUC={v60['auc']:.3f}")
    if v59['gate_fake_gt_real'] and v59['auc'] >= 0.8:
        v105 = 'FP_LEXICON_SURPRISE_YES'
    elif v59['gate_fake_gt_real']:
        v105 = 'FP_LEXICON_WEAK'
    else:
        v105 = 'FP_LEXICON_NO'
    v26 = {'timestamp': v175.v172(v176.v173).v145(), 'protocol': 'fp_lexicon_192', 'overall': v105, 'results': v56, 'diag_nn': v57, 'lexicon_size': v126(v39), 'n_real': v126(v48), 'n_fake': v126(v50), 'note': 'zero-training probe on frozen 191-P1; fp = normalized arc_enc(word); read-only by construction'}
    v4.v70(v157.v146(v26, indent=2, ensure_ascii=False), encoding='utf-8')
    v5.v70('\n'.v115(['# Stage192 — FP-lexicon lexical surprise', '', f'**Overall:** `{v105}`', '', f"- FP-lexicon: real={v59['surprise_real_mean']:.4f} fake={v59['surprise_fake_mean']:.4f} **AUC={v59['auc']:.3f}**", f"- Trigram baseline: AUC={v60['auc']:.3f}", ''] + [f'- {v58}' for v58 in v57]), encoding='utf-8')
    v71(f'[192] {v105}')
    return 0
if v61 == '__main__':
    raise v106(v147())