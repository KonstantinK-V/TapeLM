"""THE FORM CHANNEL. An ending is not material at the hole - it comes from the frame.

WHAT 373 ACTUALLY CLOSED, AND WHAT IT DID NOT. 373 asked whether an unreachable truth can be
made by CONCATENATING WHOLE AVAILABLE VALUES ("wikitext" out of "wiki" and "text") and answered
0.0000 on five corpora. I read that as "production is dead". That reading was wrong, and 373's
own cross-corpus table is the evidence against it:

    en wiki .036   de news .046   fi wiki .049   code idents .164        <- SHARE, piece overlap
    ASSEMBLE 0.0000 everywhere.

SHARE rises strictly with morphology. ASSEMBLE does not move at all. Finnish is agglutinative -
the STEM of the truth is right there and the ENDING is a grammatical item that NEVER stands at
the hole as a whole value. So concatenation was the wrong operation for exactly the corpora
where morphology is strongest. What 373 refuted is composition out of the bag. It never tested
the only composition a language actually performs:

    STEM from the offer (identity - the channel that already reaches 0.65)
        +  ENDING from the frame (form - a channel this project has never had)
        =  A WORD FORM THAT NEVER STOOD AT THIS HOLE.

That is production, it is not chosen from a bag, and no number here has ever measured it.

WHY THE FRAME CAN CARRY THE ENDING AT ALL. The address is (w, left, right) - literally the
tokens around the hole, articles and prepositions included. Phi has never seen the FORM of a
candidate against the FORM of its frame; it sees co-occurrence counts and nothing else. The
inventory of endings is tiny (a few dozen suffixes cover a corpus), so this is a COUNT, it
carries no facts, and it scales to any corpus - the same contract as every other channel.

WHAT IS MEASURED, torch-free. Two disjoint populations, one pass:

  A. SHAPE, on questions the substitution channel already REACHES. The ceiling here is the
     residual hit/reach = 0.88, nothing more - a form cue can only reorder what is offered.
       PRED     is the truth's ending in the frame's predicted top-t, by three predictors:
                  local   the endings of this place's OTHER fillers          (this place)
                  func    the endings pooled over every place whose address carries the SAME
                          FUNCTION WORDS - own place subtracted. THE ARTICLE HYPOTHESIS, and
                          the function-word list is the corpus's F most frequent tokens, counted
                  global  the corpus's top-t endings                          THE NULL
       RERANK   top-1 of the offer, before and after keeping only ending-matching candidates.

  B. PRODUCE, on the subset where every number this project prints is ZERO - truth in neither
     the place's own fillers nor the offer (363's and 373's population, unchanged so the numbers
     are comparable).
       STEMS      {v} and {stem(v)} for every available value v. Both, because stripping is as
                  legitimate as adding: "playing" offered, "play" true.
       ORACLE     truth = stem + e for SOME e in the whole counted inventory.  MATERIAL EXISTS
       PRODUCE    truth = stem + e for e in the frame's predicted top-t.       FRAME FINDS IT
       bare       of those, the ones with an empty ending (pure stripping), reported apart so
                  it cannot silently carry the number.
       NULL       the identical construction for a token from the truth's own FREQUENCY BAND,
                  reusing 373's bands. Endings of common words are everywhere; an unmatched null
                  would hand this a win for free.

  GATES, declared before the run:
    G1  produce_oracle - oracle_null > 0.05      the material for a NEW FORM is present
    G2  produce_func   - func_null   > 0.05      and the frame predicts the ending that builds it
    Both must pass. G1 alone means the pieces exist and nothing points at them; G2 alone cannot
    happen. If G1 fails, the unreachable truths are not inflections of anything available and
    the form channel is empty on this corpus.
    SHAPE is measured and reported but DOES NOT DECIDE THE STEP - it is a policy gain inside a
    population that already works, and no new metric may decide a step.

  Swept, not tuned: --endings 32/64/128, --top-ends 1/4/8, --suffix-max 4. A conclusion that
  holds at one setting is a conclusion about that setting.

    python _audit374_shape.py
    python _audit374_shape.py --endings 128 --top-ends 8 --window-lines 1600
    python _audit374_shape.py --corpus data/_fi_wiki.txt --window-lines 8000
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage374_shape.json')

def ending_inventory(v4, v5, v6):
    """the corpus's endings, by count: rank every suffix of length 1..L by HOW MANY DISTINCT
    TYPES end in it, keep the top k. Types not tokens, so `the` does not make `he` an ending."""
    v7 = v38()
    for v8 in v4:
        for v39 in v112(1, v5 + 1):
            if v120(v8) > v39:
                v7[v8[-v39:]] += 1
    return [v40 for v40, v139 in v7.v128(v6)]

def make_split(v9):
    """one deterministic decomposition per token: the LONGEST known ending it carries."""
    v10 = v41({v120(v40) for v40 in v9}, reverse=True)
    v11 = v42(v9)
    v12 = {}

    def split(v8):
        v43 = v12.v113(v8)
        if v43 is None:
            v43 = (v8, '')
            for v39 in v10:
                if v120(v8) > v39 and v8[-v39:] in v11:
                    v43 = (v8[:-v39], v8[-v39:])
                    break
            v12[v8] = v43
        return v43
    return v13

def main() -> v2:
    v14 = v114.v44()
    v14.v45('--bytes', type=v2, default=30000000)
    v14.v45('--frame-max', type=v2, default=3)
    v14.v45('--min-fillers', type=v2, default=1)
    v14.v45('--lines', type=v2, default=25000)
    v14.v45('--window-lines', type=v2, default=400)
    v14.v45('--topm', type=v2, default=8)
    v14.v45('--endings', type=v2, default=64)
    v14.v45('--suffix-max', type=v2, default=4)
    v14.v45('--top-ends', type=v2, default=4)
    v14.v45('--func', type=v2, default=32)
    v14.v45('--max-questions', type=v2, default=1500)
    v14.v45('--max-shape', type=v2, default=6000)
    v14.v45('--seed', type=v2, default=1337)
    v14.v45('--corpus', default=v140(v0))
    v15 = v14.v46()
    v16 = v3(v15.v102).v141('r', encoding='utf-8', errors='ignore').v47(v15.v48)
    v17 = [v116.v115() for v116 in v16.v13('\n') if v120(v116.v115()) >= 80]
    v18 = v17[:v2(0.7 * v120(v17))][:v15.v18]
    v19 = v117.v49(v15.v50)
    if v15.v51 and v15.v51 < v120(v18):
        v52 = v19.v118(v120(v18) - v15.v51)
        v18 = v18[v52:v52 + v15.v51]
    v53, v54, v55 = v119.v56(v18, v15.v57, v15.v58)
    if not v53:
        v107('no tape')
        return 1
    v20 = [v59 for v59, v142 in v53]
    v21 = [v66(v63) for v143, v63 in v53]
    v60, v61 = ({}, v65(v66))
    for v62, v63 in v64(v21):
        for v34 in v63:
            v60[v34] = v62
            v61[v54[v34]].v122(v34)
    v4 = v41(v61)
    v22 = {v8: v120(v61[v8]) for v8 in v4}
    v23 = {v8: v22[v8].v121() for v8 in v4}
    v24 = v65(v66)
    for v8 in v4:
        v24[v23[v8]].v122(v8)
    v9 = v67(v4, v15.v5, v15.v9)
    v25 = v42(v9) | {''}
    v13 = v68(v9)
    v26 = v38(v54)
    v27 = {v69 for v69, v139 in v26.v128(v15.v27)}
    v70, v71 = ([], [])
    v28 = [v65(v38) for v123 in v112(3)]
    v29 = v38()
    v30 = v38()
    for v62, v63 in v64(v21):
        v72 = v38((v13(v54[v148])[1] for v148 in v63))
        v70.v122(v72)
        v30 += v72
        v124, v125, v126 = v20[v62]
        v73 = v127((v144 for v144 in v125 if v144 in v27))
        v74 = v127((v144 for v144 in v126 if v144 in v27))
        v75 = ((v73, v74), (v73,), (v74,))
        v71.v122(v75)
        for v39 in v112(3):
            v28[v39][v75[v39]] += v72
        v29[v75[0]] += 1
    v31 = {v40 for v40, v139 in v30.v128(v15.v103)}
    v32 = {}

    def offer_of(v62, v76):
        """exactly 363's and 373's offer, returned ORDERED so it can be reranked."""
        v77 = v38((v54[v148] for v148 in v21[v62]))
        v77[v76] -= 1
        if v77[v76] <= 0:
            del v77[v76]
        v78 = v66(v77)[:6]
        if not v78:
            return []
        v79 = v38((v54[v148] for v148 in v21[v62]))
        v80 = v38()
        for v8 in v78:
            v7 = v32.v113(v8)
            if v7 is None:
                v7 = v38()
                for v145 in v61[v8]:
                    for v154 in v21[v60[v145]]:
                        if v54[v154] != v8:
                            v7[v54[v154]] += 1
                v32[v8] = v7
            for v69, v146 in v7.v147():
                v146 -= v79.v113(v69, 0)
                if v146 > 0 and v69 != v8:
                    v80[v69] += v146
        return v80.v128(v15.v104)

    def band_draw(v76, v81):
        v129, v69 = (v66(v24[v23[v76]]), 0)
        while v120(v129) < 16 and v69 < 20:
            v69 += 1
            v129 += v24.v113(v23[v76] - v69, []) + v24.v113(v23[v76] + v69, [])
        for v82 in v112(64):
            v8 = v129[v19.v118(v120(v129))]
            if v8 != v76 and v8 not in v81:
                return v8
        return None

    def buildable(v83, v84, v85):
        """target = stem + e, for an available stem and a PREDICTED ending. Returns
        (built, bare) - bare marks the empty ending, i.e. pure stripping."""
        v86 = v87 = False
        for v88 in v84:
            if not v83.v155(v88):
                continue
            v40 = v83[v120(v88):]
            if v40 in v85:
                v86 = True
                if v40 == '':
                    v87 = True
                else:
                    return (True, False)
        return (v86, v87)
    v33 = [v34 for v63 in v21 for v34 in v63]
    v19.v89(v33)
    v7 = v38()
    for v34 in v33:
        if v7['nB'] >= v15.v130:
            break
        v62 = v60[v34]
        v76 = v54[v34]
        v77 = {v54[v148] for v148 in v21[v62] if v148 != v34}
        if not v77:
            continue
        v7['seen'] += 1
        v90 = v131(v62, v76)
        v80 = {v69 for v69, v139 in v90}
        v91 = v13(v76)[1]
        v92 = v38(v70[v62])
        v92[v91] -= 1
        if v92[v91] <= 0:
            del v92[v91]
        v75 = v71[v62]
        v132, v133 = (v38(), 3)
        for v39 in v112(3):
            v132 = v28[v39][v75[v39]] - v70[v62]
            if v132:
                v133 = v39
                break
        v7[f'lvl{v133}'] += 1
        v7['lvl_n'] += 1
        v93 = {v40 for v40, v139 in v92.v128(v15.v103)} or v31
        v94 = {v40 for v40, v139 in v132.v128(v15.v103)} or v31
        if v76 in v77 or v76 in v80:
            if v7['nA'] >= v15.v149:
                continue
            v7['nA'] += 1
            v7['A_loc'] += v91 in v93
            v7['A_fun'] += v91 in v94
            v7['A_glo'] += v91 in v31
            v7['A_bucket1'] += v29[v75] <= 1
            if v90 and v76 in v80:
                v7['nR'] += 1
                v7['R_base'] += v90[0][0] == v76
                v150 = [v69 for v69, v139 in v90 if v13(v69)[1] in v94]
                v7['R_fun'] += (v150[0] if v150 else v90[0][0]) == v76
                v151 = [v69 for v69, v139 in v90 if v13(v69)[1] in v93]
                v7['R_loc'] += (v151[0] if v151 else v90[0][0]) == v76
            continue
        v95 = (v77 | v80) - {v76}
        if not v95:
            continue
        v7['nB'] += 1
        v84 = v42()
        for v8 in v95:
            v84.v152(v8)
            v84.v152(v13(v8)[0])
        v96, v87 = v134(v76, v84, v25)
        v7['B_oracle'] += v96
        v7['B_oracle_bare'] += v87
        v96, v87 = v134(v76, v84, v94)
        v7['B_fun'] += v96
        v7['B_fun_bare'] += v87
        v96, v123 = v134(v76, v84, v93)
        v7['B_loc'] += v96
        v96, v123 = v134(v76, v84, v31)
        v7['B_glo'] += v96
        v97 = v135(v76, v95)
        if v97 is None:
            continue
        v7['nN'] += 1
        v96, v87 = v134(v97, v84, v25)
        v7['N_oracle'] += v96
        v7['N_oracle_bare'] += v87
        v96, v123 = v134(v97, v84, v94)
        v7['N_fun'] += v96
        v96, v123 = v134(v97, v84, v93)
        v7['N_loc'] += v96
        v96, v123 = v134(v97, v84, v31)
        v7['N_glo'] += v96
    v98, v99, v100, v101 = (v108(1, v7[v6]) for v6 in ('nA', 'nB', 'nN', 'nR'))
    v35 = {'corpus': v15.v102, 'lines': v120(v18), 'places': v120(v21), 'vocab': v120(v4), 'endings': v15.v9, 'suffix_max': v15.v5, 'top_ends': v15.v103, 'func': v15.v27, 'topm': v15.v104, 'min_fillers': v15.v58, 'seen': v7['seen'], 'reachable': v7['nA'], 'unreachable': v7['nB'], 'nulls': v7['nN'], 'buckets': v120(v28[0]), 'singleton_bucket': v7['A_bucket1'] / v98, 'backoff': [v7[f'lvl{v39}'] / v108(1, v7['lvl_n']) for v39 in v112(4)], 'pred_local': v7['A_loc'] / v98, 'pred_func': v7['A_fun'] / v98, 'pred_global': v7['A_glo'] / v98, 'rerank_n': v7['nR'], 'rerank_base': v7['R_base'] / v101, 'rerank_func': v7['R_fun'] / v101, 'rerank_local': v7['R_loc'] / v101, 'produce_oracle': v7['B_oracle'] / v99, 'oracle_null': v7['N_oracle'] / v100, 'produce_func': v7['B_fun'] / v99, 'func_null': v7['N_fun'] / v100, 'produce_local': v7['B_loc'] / v99, 'local_null': v7['N_loc'] / v100, 'produce_global': v7['B_glo'] / v99, 'global_null': v7['N_glo'] / v100, 'oracle_bare': v7['B_oracle_bare'] / v99, 'func_bare': v7['B_fun_bare'] / v99}
    v35['G1'] = v35['produce_oracle'] - v35['oracle_null']
    v35['G2'] = v35['produce_func'] - v35['func_null']
    v35['G2_local'] = v35['produce_local'] - v35['local_null']
    v35['shape_gain'] = v35['pred_func'] - v35['pred_global']
    v35['rerank_gain'] = v35['rerank_func'] - v35['rerank_base']
    v1.v136.v105(parents=True, exist_ok=True)
    v1.v106(v153.v137(v35, indent=1), encoding='utf-8')
    v107(f'tape     {v120(v21)} places, {v120(v4)} values, {v15.v9} endings up to {v15.v5} chars, top-{v15.v103} predicted, {v15.v27} function words')
    v107(f"buckets  {v120(v28[0])} function-word signatures, {v35['singleton_bucket']:.4f} of reachable questions sit in a bucket of one")
    v107(f"backoff  both sides {v35['backoff'][0]:.4f}   left {v35['backoff'][1]:.4f}   right {v35['backoff'][2]:.4f}   nothing, fell to global {v35['backoff'][3]:.4f}")
    v107(f"\nA SHAPE   {v7['nA']} reachable questions - a policy gain, it does NOT decide")
    v107(f"  PRED    true ending predicted: local {v35['pred_local']:.4f}   func {v35['pred_func']:.4f}   global {v35['pred_global']:.4f}   func over global {v35['shape_gain']:+.4f}")
    v107(f"  RERANK  top-1 of the offer ({v7['nR']} q): base {v35['rerank_base']:.4f}   func {v35['rerank_func']:.4f}   local {v35['rerank_local']:.4f}   gain {v35['rerank_gain']:+.4f}")
    v107(f"\nB PRODUCE {v7['nB']} questions reach NEITHER channel; {v7['nN']} band-matched nulls")
    v107(f"  ORACLE  stem available + ANY known ending: {v35['produce_oracle']:.4f}   null {v35['oracle_null']:.4f}   G1 {v35['G1']:+.4f}   (bare {v35['oracle_bare']:.4f})")
    v107(f"  FUNC    stem + ending the ARTICLES predict: {v35['produce_func']:.4f}   null {v35['func_null']:.4f}   G2 {v35['G2']:+.4f}   (bare {v35['func_bare']:.4f})")
    v107(f"  LOCAL   stem + ending this place predicts:  {v35['produce_local']:.4f}   null {v35['local_null']:.4f}   G2l {v35['G2_local']:+.4f}")
    v107(f"  GLOBAL  stem + the corpus's commonest ends: {v35['produce_global']:.4f}   null {v35['global_null']:.4f}")
    v36 = v108((v35['G2'], 'func', v35['produce_func'], v35['func_null']), (v35['G2_local'], 'local', v35['produce_local'], v35['local_null']))
    v109, v110 = (v35['G1'] > 0.05, v36[0] > 0.05)
    if v7['nB'] < 100 or v7['nN'] < 100:
        v107(f"\nVOID, NOT A RESULT. Only {v7['nB']} unreachable questions and {v7['nN']} nulls survived - too few for a 0.05 gate to mean anything. Widen the window (--window-lines) until B has a few hundred, then read it.")
    elif v109 and v110:
        v107(f'\nPRODUCTION IS ALIVE AND THE FRAME AIMS IT. On the holes where every number this project prints is zero, the truth is an AVAILABLE STEM IN A PREDICTED FORM ({v36[1]}) {v36[2]:.4f} of the time against {v36[3]:.4f} for a word of its own frequency. 373 refuted concatenation, not inflection. The value must stop being a whole token: a candidate becomes (stem, ending), the stem comes from the offer that already reaches, the ending from the frame - and the answer is BUILT, not chosen. The fourth trait is a task again, inside one objective.')
    elif v109:
        v107(f"\nTHE MATERIAL IS THERE AND NOTHING POINTS AT IT. The stem is available and a known ending completes it {v35['produce_oracle']:.4f} of the time against {v35['oracle_null']:.4f}, but no frame predictor finds that ending - func {v35['G2']:+.4f}, local {v35['G2_local']:+.4f}. Inflection is real here and the ARTICLES DO NOT CARRY IT at this width; the next question is the predictor, not the atom - a wider address, or the ending as a feature Phi ranks rather than a filter applied before it.")
    else:
        v107(f"\nTHE UNREACHABLE TRUTHS ARE NOT INFLECTIONS OF WHAT IS AVAILABLE. Even knowing the whole ending inventory, the truth is an available stem plus an ending {v35['produce_oracle']:.4f} of the time against {v35['oracle_null']:.4f} for a random word of the same frequency. The stems of the answers are not at the hole either - so it is not the atom and not the form channel: what is missing at these holes is the MATERIAL, and no decomposition of the value creates it.")
    v107(f'\nwritten to {v1}')
    return 0
if v37 == '__main__':
    raise v111(v138())