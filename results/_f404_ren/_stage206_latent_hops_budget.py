"""
Stage 206 — W5: latent hops as O(1)-encoder vector ops vs O(k) token/decode routes.

Four routes on the SAME planted k-hop chains (k up to 6, beyond window):
  curve_latent : encode the start entity ONCE, then k matvecs in fp-space (no decode, no re-encode)
  rag_text     : GPT index whose values are TEXT -> every hop decodes to a string and re-encodes (k+1 encoder calls)
  rag_vector   : HONESTY FLIP - GPT index whose values are GPT embeddings -> hops are matvecs too (1 encoder call)
  gpt_cot      : vanilla in-context reference (facts beyond window)

Conditions: clean, and noisy (independent char-noise realizations on stored anchors/values and the
query, as in 204) to test whether error COMPOUNDS across hops differently per substrate.

Metrics: accuracy vs k, encoder forward calls per query, wall-clock ms per query.

Gates:
  G_deep      curve_latent acc at k=6 >= 0.70 (chance 0.25)
  G_cheap     curve_latent >= 2x faster than rag_text and uses 1 encoder call vs k+1
  G_flip      does rag_vector tie curve_latent on CLEAN? (if yes -> architectural, report honestly)
  G_noise     curve_latent beats rag_vector at k>=4 under noise by >= 0.10 (compounding advantage)

  python _stage206_latent_hops_budget.py
"""
from __future__ import annotations
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _stage204_noise_robustness import noisy
v0 = v15('results')
v1 = v15('checkpoints/stage191_p1_curve.pt')
v2 = v15('data/_wikitext103_train.txt')
v3 = v0 / 'stage206_decision.json'
v4 = v0 / 'stage206_mini.md'
v5 = v0 / '_stage206_log.txt'
v6 = 206
v7 = 20000000
v8 = 120
v9 = 7
v10 = 3000
v11 = [1, 2, 4, 6]
v12 = 0.15
v13 = 0.25

def log(v16: v53) -> None:
    v17 = v16 if v16.v111('\n') else v16 + '\n'
    try:
        v112(v17, end='', flush=True)
    except v54:
        v112(v17.v195('ascii', 'replace').v177('ascii'), end='', flush=True)
    v5.v113.v55(parents=True, exist_ok=True)
    with v5.v114('a', encoding='utf-8') as v56:
        v56.v115(v17)

def main() -> v14:
    v0.v55(parents=True, exist_ok=True)
    v5.v57('', encoding='utf-8')
    v58(f'Stage206 start {v197.v185(v198.v186).v152()}')
    v58('W5: latent vector hops vs decode/token routes under compute budget')
    v18 = v116.v18('cuda' if v116.v178.v155() else 'cpu')
    v19 = v117.v59(v6)
    v116.v60(v6)
    v20 = v61.v61()
    v62, v63, v64, v65 = v66()
    v21 = v118.v67(v53(v156.v119))
    v22 = v21.v120(v121) or 0
    v23 = v21.v68()
    v24 = v157(v65, v23).v69(v18)
    v24.v70(v116.v158(v1, map_location=v18, weights_only=False)['model'])
    v24.v71()
    for v25 in v24.v72():
        v25.v122(False)
    v26 = v73(v24, v64, v18)
    v27 = v74(v18)
    v58(f'models loaded ({v61.v61() - v20:.0f}s)')
    v28: v75[v53, v116.v123] = {}
    v29 = {'n': 0}

    @v116.v81()
    def gpt_word(v76, v77=True):
        """Cached GPT word embedding — used ONLY for offline index building."""
        if v76 in v28:
            return v28[v76]
        v78 = [v87 for v87 in v21.v195(' ' + v76).v78 if v87 != v22][-v179:]
        v79 = v27.v201(input_ids=v116.v209([v78], device=v18)).v180[0].v124(0)
        v80 = v159.v125(v79, dim=-1)
        v28[v76] = v80
        if v77:
            v29['n'] += 1
        return v80

    @v116.v81()
    def gpt_word_nc(v76):
        """UNCACHED: what a query actually costs at inference (counted)."""
        v29['n'] += 1
        v78 = [v87 for v87 in v21.v195(' ' + v76).v78 if v87 != v22][-v179:]
        v79 = v27.v201(input_ids=v116.v209([v78], device=v18)).v180[0].v124(0)
        return v159.v125(v79, dim=-1)

    @v116.v81()
    def curve_fp_nc(v76):
        """UNCACHED curve arc-encoder call (counted) — the curve's per-query encode cost."""
        v29['n'] += 1
        v82 = v116.v126(1, 1, v156.v127, dtype=v116.v160)
        for v128, v129 in v130(v76[:v156.v127]):
            v82[0, 0, v128] = v64.v161(v129, 0)
        return v159.v125(v24.v189(v82.v69(v18))[:, 0], dim=-1)[0]
    with v2.v114('r', encoding='utf-8', errors='ignore') as v56:
        v83 = v56.v131(v7)
    v30 = v84(v75.v132((v175.v190(1) for v175 in v207.v202(v83) if v182(v175.v190(1)) >= 5)))[:v10 + 10]
    v31 = [v76 for v76 in v162(v181(v30), v19, v8 * v9 + 200) if v182(v76) >= 5]
    v32 = [v31[v87 * v9:(v87 + 1) * v9] for v87 in v138(v8)]
    v32 = [v42 for v42 in v32 if v182(v42) == v9]
    v33 = v84(v75.v132([v76 for v42 in v32 for v76 in v42]))
    v58(f'chains={v182(v32)} depth={v9 - 1} pool={v182(v33)} ({v61.v61() - v20:.0f}s)')

    def build_banks(v85):
        """Slots for all routes. Under noise, stored anchor/value use independent realizations."""
        v86 = v117.v59(v6 + 17 + v14(v85 * 100))
        v133, v134, v135, v136, v137 = ([], [], [], [], [])
        for v42 in v32:
            for v87 in v138(v182(v42) - 1):
                v139 = v164(v42[v87], v85, v86)
                v140 = v164(v42[v87 + 1], v85, v86)
                v133.v165(v26.v88([v139])[0])
                v134.v165(v26.v88([v140])[0])
                v135.v165(v166(v139, count=False))
                v136.v165(v166(v140, count=False))
                v137.v165(v140)
        for v87 in v138(v163(v10, v182(v30) - 1)):
            v139 = v164(v30[v87], v85, v86)
            v140 = v164(v30[v87 + 1], v85, v86)
            v133.v165(v26.v88([v139])[0])
            v134.v165(v26.v88([v140])[0])
            v135.v165(v166(v139, count=False))
            v136.v165(v166(v140, count=False))
            v137.v165(v140)
        return (v116.v89(v133, 0), v116.v89(v134, 0), v116.v89(v135, 0), v116.v89(v136, 0), v137)
    v34 = v26.v88(v33)
    v35 = v116.v89([v166(v76, count=False) for v76 in v33], 0)

    def cands_of(v90, v40):
        v91 = [v141 for v141 in v33 if v141 != v90]
        v40.v142(v91)
        v42 = [v90] + v91[:3]
        v92 = v84(v138(4))
        v40.v142(v92)
        return ([v42[v87] for v87 in v92], v92.v167(0))

    def run_routes(v85):
        v143, v144, v145, v146, v137 = v147(v85)
        v86 = v117.v59(v6 + 41 + v14(v85 * 100))
        v93 = {v42[0]: v164(v42[0], v85, v86) for v42 in v32}
        v94 = {}
        for v95 in ('curve_latent', 'curve_latent_snap', 'rag_vector', 'rag_vector_snap', 'rag_text'):
            v168, v169, v170 = ({}, {}, {})
            for v148 in v11:
                v40 = v117.v59(v6 + 7)
                v41 = 0
                v29['n'] = 0
                if v18.v183 == 'cuda':
                    v116.v178.v191()
                v171 = v61.v61()
                for v42 in v32:
                    v184 = v93[v42[0]]
                    if v95.v192('curve_latent'):
                        v80 = v203(v184)
                        for v193 in v138(v148):
                            v80 = v144[v14((v143 @ v80).v196())]
                            if v95.v111('snap'):
                                v80 = v34[v14((v34 @ v80).v196())]
                        v194 = v34 @ v80
                    elif v95.v192('rag_vector'):
                        v80 = v208(v184)
                        for v193 in v138(v148):
                            v80 = v146[v14((v145 @ v80).v196())]
                            if v95.v111('snap'):
                                v80 = v35[v14((v35 @ v80).v196())]
                        v194 = v35 @ v80
                    else:
                        v204 = v184
                        for v193 in v138(v148):
                            v80 = v208(v204)
                            v204 = v137[v14((v145 @ v80).v196())]
                        v80 = v208(v204)
                        v194 = v35 @ v80
                    v149, v90 = v150(v42[v148], v40)
                    v100 = [v205(v194[v33.v167(v174)]) for v174 in v149]
                    v41 += v14(v14(v206.v196(v100)) == v90)
                if v18.v183 == 'cuda':
                    v116.v178.v191()
                v172 = (v61.v61() - v171) / v182(v32) * 1000.0
                v168[v148] = v41 / v182(v32)
                v169[v148] = v172
                v170[v148] = v29['n'] / v182(v32)
                v58(f"  [{('noisy' if v85 else 'clean')}] {v95} k={v148}: acc={v168[v148]:.3f} {v172:.2f} ms/query enc_calls={v170[v148]:.2f} ({v61.v61() - v20:.0f}s)")
            v94[v95] = {'acc': v168, 'ms': v169, 'enc_calls': v170}
        return v94
    v36 = v96(0.0)
    v37 = v96(v12)
    v38 = ' '.v97((f'{v42[v87]} leads to {v42[v87 + 1]} .' for v42 in v32 for v87 in v138(v182(v42) - 1)))
    v39 = [v87 for v87 in v21.v195(v38).v78 if v87 != v22][-v179 + 8:]
    v40 = v117.v59(v6 + 7)
    v41 = 0
    for v42 in v32[:60]:
        v98 = [v87 for v87 in v21.v195(f' {v42[0]} eventually leads to').v78 if v87 != v22]
        v99 = (v39 + v98)[-v179:]
        v149, v90 = v150(v42[2], v40)
        v100 = [v173(v27, v18, v99, [v87 for v87 in v21.v195(' ' + v174).v78 if v87 != v22]) for v174 in v149]
        v41 += v14(v14(v206.v196(v100)) == v90)
    v43 = v41 / 60
    v58(f'  gpt in-context (k=2, beyond window) acc={v43:.3f}')
    v101, v102, v103 = (v36['curve_latent'], v36['rag_vector'], v36['rag_text'])
    v44 = v36['curve_latent_snap']
    v104, v105 = (v37['curve_latent'], v37['rag_vector'])
    v106, v107 = (v37['curve_latent_snap'], v37['rag_vector_snap'])
    v45 = v101['acc'][6] >= 0.7
    v46 = v103['ms'][6] >= 2 * v101['ms'][6] and v103['enc_calls'][6] >= 2
    v47 = v151(v102['acc'][6] - v101['acc'][6]) <= 0.05
    v48 = v108((v104['acc'][v148] >= v105['acc'][v148] + 0.1 for v148 in (4, 6)))
    v49 = v108((v106['acc'][v148] >= v107['acc'][v148] + 0.1 for v148 in (4, 6)))
    v50 = v106['acc'][6] >= v104['acc'][6] + 0.1
    if v45 and v46 and (v48 or v49):
        v109 = 'LATENT_HOPS_CHEAP_AND_NOISE_ROBUST'
    elif v45 and v46 and v47:
        v109 = 'LATENT_HOPS_CHEAPER_BUT_RAG_VECTOR_TIES'
    else:
        v109 = 'LATENT_HOPS_PARTIAL'
    v51 = {'timestamp': v197.v185(v198.v186).v152(), 'protocol': 'latent_hops_budget_206', 'overall': v109, 'clean': {v148: {v175: {v53(v199): v187 for v199, v187 in v200.v188()} for v175, v200 in v80.v188()} for v148, v80 in v36.v188()}, 'noisy': {v148: {v175: {v53(v199): v187 for v199, v187 in v200.v188()} for v175, v200 in v80.v188()} for v148, v80 in v37.v188()}, 'noise_p': v12, 'gpt_incontext_k2': v43, 'gates': {'g_deep': v45, 'g_cheap': v46, 'g_rag_vector_ties_clean': v47, 'g_noise': v48, 'g_noise_with_snap': v49, 'g_snap_helps_curve': v50}, 'chance': v13, 'note': 'rag_vector is the honesty flip: a vector-valued GPT index also hops with 1 encoder call, so a compute win over rag_text alone is architectural; the substrate claim must come from noise compounding'}
    v3.v57(v176.v153(v51, indent=2, ensure_ascii=False), encoding='utf-8')
    v4.v57('\n'.v97(['# Stage206 — W5 latent hops under compute budget', '', f'**Overall:** `{v109}`', '', '**Clean (compute):**', '', '| k | curve_latent acc/ms/enc | rag_vector acc/ms/enc | rag_text acc/ms/enc |', '|---|-------------------------|-----------------------|---------------------|'] + [f"| {v148} | {v101['acc'][v148]:.3f} / {v101['ms'][v148]:.2f} / {v101['enc_calls'][v148]:.1f} | {v102['acc'][v148]:.3f} / {v102['ms'][v148]:.2f} / {v102['enc_calls'][v148]:.1f} | {v103['acc'][v148]:.3f} / {v103['ms'][v148]:.2f} / {v103['enc_calls'][v148]:.1f} |" for v148 in v11] + ['', f'**Noisy (p={v12}) — compounding across hops, with and without lexicon re-anchoring (snap):**', '', '| k | curve | curve+snap | rag_vector | rag_vector+snap |', '|---|-------|-----------|------------|-----------------|'] + [f"| {v148} | {v104['acc'][v148]:.3f} | **{v106['acc'][v148]:.3f}** | {v105['acc'][v148]:.3f} | {v107['acc'][v148]:.3f} |" for v148 in v11] + ['', f'- vanilla GPT in-context (k=2, beyond window): {v43:.3f} (chance {v13})', f'- gates: deep={v45} cheap={v46} rag_vector_ties_clean={v47} noise={v48} noise_with_snap={v49} snap_helps_curve={v50}']), encoding='utf-8')
    v58(f"[206] {v109} | clean k6 curve={v101['acc'][6]:.2f} ragvec={v102['acc'][6]:.2f} ragtext={v103['acc'][6]:.2f} | noisy k6 curve={v104['acc'][6]:.2f} ragvec={v105['acc'][6]:.2f} | ms k6 curve={v101['ms'][6]:.2f} vs ragtext={v103['ms'][6]:.2f}")
    return 0
if v52 == '__main__':
    raise v110(v154())