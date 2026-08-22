"""
Stage 281 — What counts as an assertion, decided by the corpus rather than by a regex.

280 stopped at a wall its own gate caught: the teacher ceiling on raw text came out at -0.189,
below the 0.750 that unconditional silence scores, and the score-gap cut changed the held-out
numbers by nothing at all - byte for byte the same precision and the same ceiling. The candidates
were not separable by vote score.

The subjects say why. The held-out exam asked about `behind`, `curious`, `experience`, `hot`,
`comedy`, `fantastic`, `united`, `coast`. "The behind was ..." has no answer, so no rule of
aggregation could be right about it. 279's write decision was extracting ADJACENT CAPITALISED
WORDS, not assertions, and every measurement downstream inherited that.

The missing piece is a criterion for "is this an assertion at all", and writing one by hand is
just a bigger regex. So this measures it instead. Every extracted pair carries a FRAME - the
words standing between the anchor and the value, which is the relation the sentence was using -
and a frame can be judged by three statistics of the write journal, none of which reads a label:

  YIELD          a real relation gets independently restated, so its assertions reach CONFIRM.
                 An accidental adjacency is stated once by one source and never corroborated.
                 confirm_rate is that difference, and it is the core of the stage.

  GENERALITY     a relation applies to many subjects. A frame seen with one anchor is that
                 anchor's phrasing, not a relation.

  FUNCTIONALITY  a relation mostly maps one subject to one value. A frame where every anchor
                 carries five different values is an enumeration - "X , the Y ," - and this is
                 how a knowledge base tells a relation from a co-occurrence. Non-obvious and
                 cheap: mean distinct values per anchor, counted.

SKIP is then executable: write everything once, score the frames, drop the assertions whose frame
failed, write again. Nothing is fitted and nothing is labelled.

The fourth thing the frame buys is the one GOAL.md left open. Two assertions can only DISPUTE
each other if they share a frame. "Michael was born in X" and "Michael was appointed Y" are not a
disagreement, they are two facts, and counting them as witnesses of one address is exactly what
made 280's teacher answer everywhere and be wrong. --frame-in-address turns "sources disagree"
and "sources are talking about different things" into different situations mechanically.

The gate is cheap because it needs no training: the teacher runs alone on the rebuilt tape, and
if its ceiling clears half of silence then 280 is worth running again and otherwise it is not.

  python _stage281_frames.py --smoke
  python _stage281_frames.py --smoke --frame-in-address
  python _stage281_frames.py --addresses 400
"""
from __future__ import annotations
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage279_write_decision as s279
import _stage280_raw_exam as s280
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('data/_wikitext103_train.txt')
v3 = 281
v4 = v9.v4
v5 = v0 / '_stage281_log.txt'

def log(v10: v6) -> None:
    v11 = v10 if v10.v130('\n') else v10 + '\n'
    try:
        v131(v11, end='', flush=True)
    except v66:
        v131(v11.v196('ascii', 'replace').v182('ascii'), end='', flush=True)
    v5.v22.v67(parents=True, exist_ok=True)
    with v5.v132('a', encoding='utf-8') as v24:
        v24.v133(v11)

def frame_of(v12: v68) -> v6:
    return (v12['address'].v162('|', 1) + [''])[1].v69()

def score_frames(v13, v14, *, v15=0.0):
    """Three statistics per frame, all counted from the write journal and none from a label."""
    v16 = v70(v71)
    for v12 in v13:
        v16[v192(v12)].v134(v12)
    v17 = {}
    for v72, v73 in v16.v73():
        v74 = v149.v135(v14, v15)
        for v12 in v73:
            v74.v161(v12['address'], v12['value'], v12['source'])
        v75 = v136(1, v153(v73))
        v76 = v70(v79)
        for v12 in v73:
            v76[v12['address'].v162('|', 1)[0]].v137(v12['value'])
        v77 = v78(v193.v183([v153(v197) for v197 in v76.v199()])) if v76 else 0.0
        v17[v72] = {'n': v153(v73), 'confirm_rate': v74.v184[v149.v185] / v75, 'dispute_rate': v74.v184[v149.v186] / v75, 'anchors': v153(v76), 'values_per_anchor': v77, 'empty': v72 == ''}
    return v17

def cluster_frames(v13, v17, *, v18: v78, v19: v7):
    """Two frames are the same relation if they connect the same pairs, not if they share words.

    Paraphrase, jargon and dialect give one relation many surface forms - "was born in", "b.",
    "a native of" - and a criterion that reads the words fragments the statistics across all of
    them and then throws each fragment away for being too small. The extensional test needs no
    word list and no encoder: collect the (subject, value) pairs each frame produces and merge
    frames whose pair sets overlap. Nothing here is English.
    """
    v20 = v70(v79)
    for v12 in v13:
        v80 = v12['address'].v162('|', 1)[0]
        v20[v192(v12)].v137((v80, v12['value']))
    v21 = v81(v20, key=lambda v24: -v153(v20[v24]))
    v22 = {v24: v24 for v24 in v21}

    def find(v82):
        while v22[v82] != v82:
            v22[v82] = v22[v22[v82]]
            v82 = v22[v82]
        return v82
    for v83, v24 in v84(v21):
        for v85 in v21[:v83]:
            v138 = v153(v20[v24] & v20[v85])
            if v138 < v19:
                continue
            if v138 / v136(1, v153(v20[v24] | v20[v85])) >= v18:
                v22[v163(v24)] = v163(v85)
                break
    v23 = v70(v71)
    for v24 in v21:
        v23[v163(v24)].v134(v24)
    v25 = {}
    for v86, v87 in v23.v73():
        v75 = v115((v17[v10]['n'] for v10 in v87))
        v25[v86] = {'members': v87, 'n': v75, 'confirm_rate': v115((v17[v10]['confirm_rate'] * v17[v10]['n'] for v10 in v87)) / v136(1, v75), 'anchors': v153({v43[0] for v10 in v87 for v43 in v20[v10]}), 'values_per_anchor': v78(v193.v183([v17[v10]['values_per_anchor'] for v10 in v87])), 'empty': v125((v17[v10]['empty'] for v10 in v87))}
    return (v25, {v24: v163(v24) for v24 in v21})

def keep_set(v17, *, v26, v27, v28, v29, v30):
    v31 = v79()
    for v72, v88 in v17.v73():
        if v88['empty'] and (not v30):
            continue
        if v88['n'] < v26 or v88['anchors'] < v28:
            continue
        if v88['confirm_rate'] < v27 or v88['values_per_anchor'] > v29:
            continue
        v31.v137(v72)
    return v31

def main() -> v7:
    v32 = v139.v89()
    v32.v90('--smoke', action='store_true')
    v32.v90('--addresses', type=v7, default=0)
    v32.v90('--min-mentions', type=v7, default=3)
    v32.v90('--min-n', type=v7, default=4, help='assertions a frame needs to be judged')
    v32.v90('--min-confirm', type=v78, default=0.15, help='a frame whose assertions are never corroborated is an adjacency')
    v32.v90('--min-anchors', type=v7, default=2, help="a frame seen with one subject is that subject's phrasing")
    v32.v90('--max-values-per-anchor', type=v78, default=2.5, help='a relation is roughly functional; an enumeration is not')
    v32.v90('--cluster-frames', action='store_true', help='merge frames that connect the same (subject, value) pairs before judging them, so one relation stated three ways is judged once. Uses no word list and no encoder.')
    v32.v90('--cluster-jaccard', type=v78, default=0.3)
    v32.v90('--cluster-shared', type=v7, default=2)
    v32.v90('--allow-empty-frame', action='store_true', help='keep pairs where anchor and value are adjacent (apposition)')
    v32.v90('--frame-in-address', action='store_true', help='two assertions may only dispute each other if they share a frame - the difference between sources disagreeing and sources discussing different things')
    v32.v90('--address-tau', type=v78, default=0.9)
    v32.v90('--address-overlap', type=v7, default=2)
    v32.v90('--soft-match', type=v78, default=0.0)
    v32.v90('--topk', type=v7, default=7)
    v32.v90('--max-steps', type=v7, default=10)
    v32.v90('--max-reads', type=v7, default=7)
    v32.v90('--k-gap', type=v78, default=0.35)
    v32.v90('--hop', choices=('none', 'fp'), default='fp')
    v33 = v32.v91()
    v5.v22.v67(parents=True, exist_ok=True)
    v5.v92('', encoding='utf-8')
    v34 = v140.v34('cuda' if v140.v187.v164() else 'cpu')
    v35 = v141.v93(v3)
    v140.v94(v3)
    v36 = v95.v95()
    v37 = v33.v96 or (60 if v33.v127 else 400)
    v97(f'Stage281 frames start {v194.v190(v195.v191).v158()} device={v34} frame_in_address={v33.v128}')
    v98, v98, v99, v100 = v101()
    v38 = v142.v102(v6(v165.v143))
    v39 = v38.v103()
    v40 = v38.v144(v145) or 0
    v41 = v188.v166(v38, v99, v40, v39).v104(v34)
    v42 = v167(v100, v39).v104(v34)
    v42.v105(v140.v168(v1, map_location=v34, weights_only=False)['model'])
    v42.v106()
    for v43 in v42.v107():
        v43.v146(False)
    v14 = v108(v42, v99, v34)
    with v2.v132('r', encoding='utf-8', errors='ignore') as v24:
        v109 = v24.v147(4000000 if v33.v127 else 30000000)
    v44 = [v148.v69() for v148 in v109.v162('\n') if 80 <= v153(v148.v69()) <= 400]
    v45 = v7(0.7 * v153(v44))
    v46 = v44[:v45][:3000 if v33.v127 else 25000]
    v47 = v44[v45:][:1500 if v33.v127 else 12000]
    v48 = v149.v110(v46)
    v13, v98 = v149.v111(v46, v35, v37, v33.v112, 'anchor_rel', common=v48)
    v17 = v113(v13, v14, soft_match=v33.v15)
    v49 = None
    if v33.v50:
        v25, v49 = v50(v13, v17, min_jaccard=v33.v169, min_shared=v33.v170)
        v97(f'  clustering: {v153(v17)} frames -> {v153(v25)} relations')
        v114 = v150(v25, min_n=v33.v26, min_confirm=v33.v27, min_anchors=v33.v28, max_vpa=v33.v156, allow_empty=v33.v157)
        v31 = {v24 for v24, v86 in v49.v73() if v86 in v114}
    else:
        v31 = v150(v17, min_n=v33.v26, min_confirm=v33.v27, min_anchors=v33.v28, max_vpa=v33.v156, allow_empty=v33.v157)
    v51 = v115((v88['n'] for v72, v88 in v17.v73() if v72 in v31))
    v97(f'  pass 1: {v153(v13)} assertions over {v153(v17)} frames -> keep {v153(v31)} frames / {v51} assertions ({v51 / v136(1, v153(v13)):.3f}) ({v95.v95() - v36:.0f}s)')
    v52 = v81(((v88['confirm_rate'], v72, v88) for v72, v88 in v17.v73() if v88['n'] >= v33.v26), reverse=True)
    for v116, v72, v88 in v52[:12]:
        v97(f"    {('KEEP' if v72 in v31 else 'drop')}  '{v72}' n={v88['n']} confirm={v116:.2f} anchors={v88['anchors']} vpa={v88['values_per_anchor']:.2f}")
    if not v31:
        v97('  no frame survived; loosen --min-confirm or --min-n')
        return 1

    def build(v117):
        return v9.v151(v47, bank=v14, tok=v38, pad_id=v40, device=v34, rng=v141.v93(v3 + 9), n_addr=v37, min_mentions=v33.v112, tau=v33.v171, overlap=v33.v172, soft_match=v33.v15, keep_frames=v117)
    v53 = {'before': v152(None), 'after': v152(v31)}

    @v140.v123()
    def ceiling(v118):
        if v153(v118['items']) < 6:
            return {'n_items': v153(v118['items']), 'reward': v78('nan')}
        v119 = {v24: v70(v71) for v24 in v4}
        v120 = v68(k=v33.v173, max_steps=v33.v174, max_reads=v33.v175, read_cost=0.02, wrong_cost=1.0, abstain_reward=0.75, subject_filter=True, hop=v33.v176, hop_min=1.0, k_gap=v33.v177)
        v121 = []
        for v122 in v118['items']:
            v74 = v9.v178(None, v42, v41, v38, v118, v122, v40, v34, teacher_only=True, **v120)
            v119[v122['kind']]['correct'].v134(v74['correct'])
            v119[v122['kind']]['abstain'].v134(v7(v74['abstained']))
            v121.v134(v74['reward'])
        v10 = lambda v179: v78(v193.v183(v179)) if v179 else v78('nan')
        v63 = {'n_items': v153(v118['items']), 'reward': v10(v121), 'addresses': v118['n_addresses'], 'slots': v118['n_slots'], 'families': v68(v189((v83['kind'] for v83 in v118['items'])))}
        for v24 in v4:
            v63[v24] = {'n': v153(v119[v24]['abstain']), 'teacher_acc': v10(v119[v24]['correct']), 'teacher_abstain': v10(v119[v24]['abstain'])}
        return v63
    v54 = v124(v53['before'])
    v55 = v124(v53['after'])
    v97('  ceiling BEFORE frames: ' + v180.v159(v54))
    v97('  ceiling AFTER  frames: ' + v180.v159(v55))
    v56 = 0.75
    v57 = v153(v31) > 0
    v58 = v51 < v153(v13)
    v59 = v125((v17[v72]['values_per_anchor'] <= v33.v156 for v72 in v31))
    v60 = v55['reward'] >= 0.5 * v56
    v61 = v55['reward'] > v54['reward'] + 0.1
    v62 = v55.v154('tie', {}).v154('teacher_abstain', 0.0) >= 0.5
    if not v57:
        v126 = 'NO_FRAME_SURVIVES'
    elif v60 and v62:
        v126 = 'FRAMES_MAKE_THE_EXAM_SOUND'
    elif v61:
        v126 = 'FRAMES_HELP_NOT_ENOUGH'
    else:
        v126 = 'FRAMES_DO_NOT_HELP'
    v63 = {'stage': 281, 'overall': v126, 'trained_parameters': 0, 'smoke': v33.v127, 'seed': v3, 'frame_in_address': v33.v128, 'clustered': v155(v33.v50), 'thresholds': {'min_n': v33.v26, 'min_confirm': v33.v27, 'min_anchors': v33.v28, 'max_values_per_anchor': v33.v156, 'allow_empty_frame': v33.v157}, 'frames': {'total': v153(v17), 'kept': v153(v31), 'assertions_total': v153(v13), 'assertions_kept': v51, 'kept_fraction': v51 / v136(1, v153(v13)), 'top': [{'frame': v72, **v88} for v98, v72, v88 in v52[:20]], 'worst': [{'frame': v72, **v88} for v98, v72, v88 in v52[-10:]]}, 'gates': {'G_frames_survive': v57, 'G_tape_shrinks': v58, 'G_kept_frames_functional': v59, 'G_ceiling_clears_silence': v60, 'G_ceiling_improves': v61, 'G_teacher_abstains_on_tie': v62}, 'ceiling_before': v54, 'ceiling_after': v55, 'reference_280': {'teacher_ceiling_reward': -0.18888888888888897, 'held_out_precision': 0.3412698412698412}, 'note': "280's gate caught a wall its own numbers explain: the held-out exam asked about behind, curious, experience and coast, and no rule of aggregation can be right about a question with no answer. 279 was extracting adjacent capitalised words rather than assertions. Writing a better regex would only move the problem, so the criterion is measured instead. Each pair carries the frame that stood between anchor and value, and a frame is judged by three statistics of the write journal, none of which reads a label: how often its assertions are corroborated by a second source, how many distinct subjects it applies to, and how nearly functional it is - a relation maps one subject to about one value, an enumeration does not. SKIP is then executable. --frame-in-address additionally requires two assertions to share a frame before they may dispute, which is the distinction GOAL.md left open between sources disagreeing and sources discussing different things. The gate needs no training at all: the teacher runs alone on the rebuilt tape, and 280 is worth repeating only if its ceiling clears half the value of unconditional silence.", 'timestamp': v194.v190(v195.v191).v158(), 'wall_s': v95.v95() - v36}
    v0.v67(parents=True, exist_ok=True)
    v64 = '_fia' if v33.v128 else ''
    (v0 / f'stage281_decision{v64}.json').v92(v180.v159(v63, indent=2), encoding='utf-8')
    (v0 / f'stage281_mini{v64}.md').v92(f"# Stage 281 what counts as an assertion\n\n**{v126}**{(' · SMOKE' if v33.v127 else '')} · trained parameters **0**\n\n- frames {v153(v31)}/{v153(v17)} kept, assertions {v51}/{v153(v13)} ({v51 / v136(1, v153(v13)):.1%})\n- teacher ceiling **{v54['reward']:.3f} -> {v55['reward']:.3f}** (silence pays 0.750; 280 measured -0.189)\n- tie abstention by the teacher: {v54.v154('tie', {}).v154('teacher_abstain', v78('nan')):.2f} -> {v55.v154('tie', {}).v154('teacher_abstain', v78('nan')):.2f}\n\n| frame | n | confirm | anchors | values/anchor |\n|---|---:|---:|---:|---:|\n" + ''.v181((f"| `{v72}` | {v88['n']} | {v88['confirm_rate']:.2f} | {v88['anchors']} | {v88['values_per_anchor']:.2f} |\n" for v98, v72, v88 in v52[:10])) + '\n## Gates\n\n' + ''.v181((f'- {v198}: **{v197}**\n' for v198, v197 in v63['gates'].v73())), encoding='utf-8')
    v97(v180.v159({'overall': v126, 'gates': v63['gates']}, indent=2))
    return 0
if v65 == '__main__':
    raise v129(v160())