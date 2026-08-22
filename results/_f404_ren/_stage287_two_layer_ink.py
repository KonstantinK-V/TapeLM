"""
Stage 287 — Is there a second layer of ink at all, or only the first one averaged?

The proposal is a double tape: a lower ink over letters, and an UPPER ink over chunks of text
carrying something the letters do not have. Before building any of it, one thing has to be
settled, and it can be settled with no training whatsoever.

    Today's upper layer is not a layer. ctx_fp(T) = norm(mean over words of fp(word)).

It is the lower ink, averaged. And that has a consequence nobody has written down: the whole
stack is order-blind TWICE. fp pools characters before the feed-forward, so a word is a
character histogram and anagrams collide bit-exactly (277). ctx_fp then means over words, so a
sentence is a bag of those histograms and any permutation of the words gives a bit-identical
vector. "the dog bit the man" and "the man bit the dog" are the same point on this tape.

So the idea is not wrong, it is pointing at a real hole. The only question worth asking first
is whether an upper ink can be built that is NOT a function of the lower one - because if it
is a function of it, "semantics of the chunk" has nowhere to live, whatever we call it.

There are exactly two things a chunk has that its bag of words does not: ADJACENCY, and
corpus-level co-occurrence. This stage tests the first, because it needs no new machinery:

    lower  L(T) = norm( mean over words w of fp(w) )              - what the tape uses today
    upper  U(T) = norm( mean over adjacent k-word spans s of fp(s) )

U is built from the same frozen encoder and the same characters. It differs only in WHERE the
sum happens: fp of "was born" is the encoder applied to the two words' combined characters,
which is not any function of fp("was") and fp("born"), because the feed-forward sits between
the histogram and the output. Nothing is trained and nothing is tuned.

Two things about U to state before it is measured, so neither can be discovered later as a
surprise. The space is not in the character table and maps to the padding id, so fp("was born")
is fp of the concatenated characters - the span is a combined histogram, which is exactly the
point. And U is invariant to EXACT REVERSAL: "A B C" gives spans {AB, BC} while "C B A" gives
{CB, BA}, and a span's histogram is symmetric, so those multisets are identical. U therefore
separates most permutations but not that one. It is a partial order channel, not a full one,
and the shuffle control below measures how much of the order it actually recovers rather than
assuming.

Four measurements, in the order that can kill the idea fastest:

  1. WIRING PROOF. Shuffle the words inside every text. L must come back BIT-IDENTICAL - if it
     does not, this file is measuring something other than what it claims. U must move. If U
     does not move either, the upper channel has no order content and the idea dies here.

  2. IS IT A SECOND CHANNEL? Spearman rank correlation between cos_L and cos_U over slot pairs.
     At rho = 1 the upper ink is the lower one wearing a hat.

  3. DOES IT CARRY UNIQUE SIGNAL? Take the discrimination the tape actually needs - two
     mentions of the SAME address against two of different addresses - and score three
     channels: L alone, U alone, and U with its linear fit on L removed. Only the third
     answers "unique", and it is gated against its own noise the way 286 gates its AUCs.

  4. NEGATIVE CONTROLS. A residual test that cannot say "no" is not a test, and this project
     has shipped three vacuous gates already. The first attempt used L against ITSELF, which
     read AUC 0.163 at -12 sigma and correctly reported the measurement broken - but the fault
     was in the control, not the estimator. Regressing a channel on itself leaves
     residual = eps*L - delta with eps at machine scale: a perfectly MONOTONE function of L,
     whose AUC is therefore AUC(L) or 1-AUC(L) and can never be 0.5. The control was ill-posed
     by construction. Two well-posed ones replace it: a LABEL PERMUTATION null, which keeps the
     real residual and shuffles the classes, and a REDUNDANT-CHANNEL null, an affine copy of L
     plus noise matched to the observed residual scale. Both must read chance.

What this stage cannot say: whether the unique information is "semantics". It measures
information, not meaning. Unique information is necessary for meaning to be there and does not
establish it - the sufficient test is whether the mind uses it, and 286 already has the socket
(--ctx-channel) for that. Run this first because it is free; wire it only if it passes.

  python _stage287_two_layer_ink.py --smoke
  python _stage287_two_layer_ink.py --span 2
  python _stage287_two_layer_ink.py --span 3
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage286_evidence as s286
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
v0 = v10('results')
v1 = v10('checkpoints/stage191_p1_curve.pt')
v2 = v10('data/_wikitext103_train.txt')
v3 = 287
v4 = v76.v11('[A-Za-z][a-z]{2,}')
v5 = 40
v6 = 1e-06
v7 = v0 / '_stage287_log.txt'

def log(v12: v17) -> None:
    v13 = v12 if v12.v148('\n') else v12 + '\n'
    try:
        v149(v13, end='', flush=True)
    except v77:
        v149(v13.v219('ascii', 'replace').v206('ascii'), end='', flush=True)
    v7.v150.v78(parents=True, exist_ok=True)
    with v7.v151('a', encoding='utf-8') as v79:
        v79.v152(v13)

def words_of(v14: v17, v15: v17 | None=None) -> v16[v17]:
    return [v153 for v153 in v4.v207(v14) if v153 != v15][:v5]

def lower_ink(v18, v19: v16[v17]):
    """What the tape uses today: the mean of the word fingerprints."""
    if v140(v19) < 3:
        return None
    return v154.v80(v18.v220(v19).v8().v81(0), dim=-1)

def upper_ink(v18, v19: v16[v17], v20: v9):
    """The candidate second layer: the mean over ADJACENT spans.

    fp(" ".join(span)) is the frozen encoder applied to the span's combined characters. It is
    not a function of the member words' fingerprints - the feed-forward sits between the
    histogram and the output - so this channel can hold what the averaged one cannot. The
    encoder's own budget is 24 characters per fingerprint, which is why span stays small.
    """
    if v140(v19) < v20 + 1:
        return None
    v21 = [' '.v155(v19[v156:v156 + v20]) for v156 in v141(v140(v19) - v20 + 1)]
    return v154.v80(v18.v220(v21).v8().v81(0), dim=-1)

def spearman(v22: v16[v8], v23: v16[v8]) -> v8:
    """Rank correlation, so a monotone re-encoding of the same channel still reads as 1."""
    if v140(v22) < 3:
        return v8('nan')
    v24, v25 = (v158.v208(v158.v208(v22)).v157(v8), v158.v208(v158.v208(v23)).v157(v8))
    v24 -= v24.v81()
    v25 -= v25.v81()
    v26 = v8(v158.v159.v191(v24) * v158.v159.v191(v25))
    return v8(v24 @ v25 / v26) if v26 > 0 else v8('nan')

def residual_of(v27: v16[v8], v28: v16[v8]) -> v16[v8]:
    """U with its best linear explanation by L removed.

    Fitted on every pair regardless of label, so the label never touches the fit - what is
    removed is "the part of the upper channel that the lower channel already accounts for",
    and what survives is the only thing that could justify a second layer.
    """
    v82, v83 = (v158.v139(v27), v158.v139(v28))
    v29 = v158.v84([v83, v158.v192(v83)], 1)
    v85, *v60 = v158.v159.v86(v29, v82, rcond=None)
    return v16(v82 - v29 @ v85)

def main() -> v9:
    v30 = v160.v87()
    v30.v88('--smoke', action='store_true')
    v30.v88('--span', type=v9, default=2, help="words per upper-ink span. 2 is the smallest that can hold adjacency; above 3 the spans start hitting the encoder's 24-character budget and the channel degrades into a truncation artefact.")
    v30.v88('--addresses', type=v9, default=0)
    v30.v88('--min-mentions', type=v9, default=2)
    v30.v88('--max-pairs', type=v9, default=4000)
    v30.v88('--address-tau', type=v8, default=0.9)
    v30.v88('--address-overlap', type=v9, default=2)
    v30.v88('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v30.v88('--run-tag', type=v17, default='')
    v31 = v30.v89()
    global LOG_PATH
    v32 = v31.v146 and f'_{v31.v146}' or ''
    v32 += f'_span{v31.v20}'
    v7 = v0 / f'_stage287_log{v32}.txt'
    v7.v150.v78(parents=True, exist_ok=True)
    v7.v90('', encoding='utf-8')
    v33 = v161.v33('cuda' if v161.v209.v193() else 'cpu')
    v34 = v162.v91(v3)
    v161.v92(v3)
    v35 = v93.v93()
    v36 = v31.v94 or (60 if v31.v145 else 400)
    v95(f'Stage287 two-layer ink start {v217.v213(v218.v214).v188()} device={v33} span={v31.v20}')
    v60, v60, v96, v97 = v98()
    v37 = v163.v99(v17(v194.v164))
    v38 = v37.v165(v166) or 0
    v39 = v210.v195(v37, v96, v38, v37.v211()).v100(v33)
    v40 = v196(v97, v37.v211()).v100(v33)
    v40.v101(v161.v197(v1, map_location=v33, weights_only=False)['model'])
    v40.v102()
    for v41 in v40.v103():
        v41.v167(False)
    v18 = v104(v40, v96, v33)
    with v2.v151('r', encoding='utf-8', errors='ignore') as v79:
        v105 = v79.v168(4000000 if v31.v145 else 30000000)
    v42 = [v28.v169() for v28 in v105.v198('\n') if 80 <= v140(v28.v169()) <= 400]
    v43 = v42[:v9(0.7 * v140(v42))][:3000 if v31.v145 else 25000]
    v44 = v170.v106(v43, bank=v18, tok=v37, pad_id=v38, device=v33, rng=v34, n_addr=v36, min_mentions=v31.v171, tau=v31.v172, overlap=v31.v173, soft_match=0.0, min_per_family=0, addr_key=v31.v174)
    v107, v108 = (v44['texts'], v44['tape'].v109)
    v95(f"  tape: {v44['n_addresses']} addresses, {v44['n_slots']} slots")
    v83, v82, v110, v111, v112 = ({}, {}, {}, {}, [])
    v45 = v162.v91(v3 + 1)
    for v113, v114 in v115(v107):
        v19 = v175(v114, exclude=v108[v113])
        v176, v177 = (v199(v18, v19), v200(v18, v19, v31.v20))
        if v176 is None or v177 is None:
            continue
        v116 = v19[:]
        v45.v178(v116)
        v83[v113], v82[v113] = (v176, v177)
        v110[v113], v111[v113] = (v199(v18, v116), v200(v18, v116, v31.v20))
        v112.v179(v113)
    v95(f'  usable slots: {v140(v112)}')
    if v140(v112) < 3 * v183.v122:
        v95('  too few usable slots')
        return 1
    v46 = [v8((v83[v118] - v110[v118]).v204().v180()) for v118 in v112]
    v47 = [v8(v82[v118] @ v111[v118]) for v118 in v112]
    v48 = v180(v46) < v6
    v49 = v8(v158.v201(v47)) < 1.0 - v6
    v95(f'  shuffle control: max|dL| = {v180(v46):.3e} (must be ~0), median cos(U, U_shuffled) = {v158.v201(v47):.4f} (must be < 1)')
    v50 = v117(v16)
    for v51 in v44['items']:
        for v118 in v51['slots']:
            if v118 in v83:
                v50[v51['address']].v179(v118)
    v52 = [(v22, v23) for v113 in v50.v109() for v156, v22 in v115(v113) for v23 in v113[v156 + 1:]]
    v53 = v162.v91(v3 + 2)
    v119, v120 = ([], 0)
    while v140(v119) < v140(v52) and v120 < 40 * v180(1, v140(v52)):
        v120 += 1
        v22, v23 = (v53.v202(v112), v53.v202(v112))
        if v22 != v23 and v44['slot_addr'][v22] != v44['slot_addr'][v23]:
            v119.v179((v22, v23))
    if v140(v52) > v31.v121:
        v52 = v53.v181(v52, v31.v121)
        v119 = v53.v181(v119, v31.v121)
    v95(f'  pairs: {v140(v52)} same-address, {v140(v119)} different-address')
    if v182(v140(v52), v140(v119)) < v183.v122:
        v95('  not enough pairs of one kind; raise --addresses')
        return 1

    def cosines(v123):
        return ([v8(v83[v22] @ v83[v23]) for v22, v23 in v123], [v8(v82[v22] @ v82[v23]) for v22, v23 in v123])
    v124, v125 = v126(v52)
    v127, v128 = v126(v119)
    v54 = v129(v124 + v127, v125 + v128)
    v55 = v130(v125 + v128, v124 + v127)
    v131, v132 = (v55[:v140(v125)], v55[v140(v125):])

    def scored(v133, v134):
        v22 = v183.v184(v133, v134)
        return {'auc': v22, 'auc_z': v183.v186(v22, v140(v133), v140(v134))}
    v135, v136, v137 = (v143(v124, v127), v143(v125, v128), v143(v131, v132))
    v56 = v158.v162.v138(v3 + 3)
    v57 = v158.v139(v131 + v132)
    v58 = v140(v131)
    v59 = []
    for v60 in v141(99):
        v142 = v56.v185(v140(v57))
        v59.v179(v183.v184(v16(v57[v142[:v58]]), v16(v57[v142[v58:]])))
    v61 = {'auc': v8(v158.v81(v59)), 'auc_z': v183.v186(v8(v158.v81(v59)), v58, v140(v57) - v58), 'auc_p05': v8(v158.v203(v59, 0.05)), 'auc_p95': v8(v158.v203(v59, 0.95)), 'n_permutations': v140(v59)}
    v62 = v158.v139(v124 + v127)
    v63 = v62 * 2.0 - 0.5 + v56.v187(0.0, v8(v158.v215(v57)) + 1e-12, size=v140(v62))
    v64 = v130(v16(v63), v16(v62))
    v65 = v143(v64[:v58], v64[v58:])
    v66 = {'permutation': v61, 'redundant_channel': v65}
    v95(f'  L {v205.v189(v135)}\n  U {v205.v189(v136)}\n  U|L {v205.v189(v137)}\n  nulls {v205.v189(v66)}')
    v67 = v144(v204(v61['auc_z']) <= 1.645)
    v68 = v144(v48)
    v69 = v144(v49)
    v70 = v144(v204(v54) < 1.0 - v6)
    v71 = v144(not v216.v212(v137['auc_z']) and v137['auc_z'] > 1.645)
    v72 = v144(v136['auc'] > v135['auc'])
    v73 = 'MEASUREMENT_BROKEN' if not (v68 and v67) else 'NO_ORDER_CONTENT' if not v69 else 'SECOND_LAYER_REAL' if v70 and v71 else 'SECOND_LAYER_REDUNDANT'
    v74 = {'stage': 287, 'overall': v73, 'seed': v3, 'smoke': v31.v145, 'span': v31.v20, 'run_tag': v31.v146, 'trained_parameters': 0, 'tape': {'addresses': v44['n_addresses'], 'slots': v44['n_slots'], 'usable': v140(v112)}, 'pairs': {'same_address': v140(v52), 'different_address': v140(v119)}, 'shuffle_control': {'max_abs_delta_lower': v180(v46), 'median_cos_upper_vs_shuffled': v8(v158.v201(v47)), 'mean_cos_upper_vs_shuffled': v8(v158.v81(v47))}, 'spearman_upper_vs_lower': v54, 'order_content': 1.0 - v8(v158.v201(v47)), 'channels': {'lower': v135, 'upper': v136, 'upper_given_lower': v137, 'nulls': v66}, 'gates': {'G_lower_is_order_blind': v68, 'G_residual_test_can_say_no': v67, 'G_upper_sees_word_order': v69, 'G_channels_are_distinct': v70, 'G_upper_carries_unique_signal': v71, 'G_upper_beats_lower_alone': v72}, 'note': "Whether a second layer of ink exists at all, measured before anything is built and with nothing trained. Today's upper layer is not one: ctx_fp is the mean of the word fingerprints, so the stack is order-blind twice - fp pools characters before the feed-forward, and the mean over words then discards word order, which makes 'the dog bit the man' and 'the man bit the dog' the same point. The candidate upper ink is the mean over ADJACENT spans, built from the same frozen encoder: fp of a span is the encoder applied to the span's combined characters and is not any function of the member words' fingerprints, because the feed-forward sits between the histogram and the output. Four measurements in kill-fastest order: shuffling the words must leave the lower ink bit-identical (a wiring proof) and must move the upper one (or there is no order content and the idea is over); the Spearman correlation between the two cosines says whether it is a second channel or the first wearing a hat; the discrimination the tape actually needs - same address against different address - is scored for the lower channel, the upper channel, and the upper with its linear fit on the lower removed, which is the only one that can justify a second layer; and the same residual test is run on a channel redundant BY CONSTRUCTION, the lower against itself, which must come out at chance or the examiner is broken and every number above it is worthless. This measures information, not meaning: unique information is necessary for semantics to live there and does not establish it. The sufficient test is whether the mind uses it, and 286's --ctx-channel is the socket for that - wire it only if this passes.", 'timestamp': v217.v213(v218.v214).v188(), 'wall_s': v93.v93() - v35}
    v0.v78(parents=True, exist_ok=True)
    (v0 / f'stage287_decision{v32}.json').v90(v205.v189(v74, indent=2), encoding='utf-8')
    (v0 / f'stage287_mini{v32}.md').v90(f"# Stage 287 is there a second layer of ink (span {v31.v20})\n\n**{v73}**{(' · SMOKE' if v31.v145 else '')} · trained parameters **0**\n\n| channel | AUC same-vs-different | sigma |\n|---|---:|---:|\n" + ''.v155((f"| {v221} | {v222['auc']:.3f} | {v222['auc_z']:+.2f} |\n" for v221, v222 in (("lower (today's ctx_fp)", v135), ('upper (adjacent spans)', v136), ('upper given lower', v137), ('null: label permutation', v61), ('null: redundant channel', v65)))) + f'\n- shuffling words: lower moves by {v180(v46):.2e} (must be ~0), upper stays at cos {v158.v201(v47):.4f} -> only {1 - v8(v158.v201(v47)):.4f} of it depends on word order\n- Spearman between the two cosines: {v54:.4f}\n\n## Gates\n\n' + ''.v155((f'- {v221}: **{v222}**\n' for v221, v222 in v74['gates'].v223())), encoding='utf-8')
    v95(v205.v189({'overall': v73, 'gates': v74['gates']}, indent=2))
    return 0
if v75 == '__main__':
    raise v147(v190())