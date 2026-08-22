"""THE CEILING OF NAMING A PLACE THE WALK CANNOT REACH. Measured before a chooser is built.

WHY THIS AND NOT ANOTHER TERM ON PHI. 389 was voided by its own control (section 31): the raw
score already carries a scale, so the missing thing is not a better-scaled score. The user's own
reading of what is left - "the mind still does not choose WHERE TO LOOK" - is the wall section 29
calls A, an OUTPUT SPACE, approached from the place side instead of the word side. `--moves`
(385/386) chooses among three fixed channels; nothing in this project lets the mind NAME A PLACE
IT WAS NOT OFFERED.

THE STRUCTURAL FACT THIS RESTS ON, and it is 387's own suspect, stated as an opportunity instead
of a complaint. The arm runs `fp=fillers`, so a place's fingerprint IS its bag of fillers and the
cosine between two profiles is ZERO unless they share a filler. `connect` walks places that share
a filler. THEREFORE EVERY CHANNEL THE MIND HAS - step, share, and copy through the line - has the
same non-zero support: places that share a filler with this one. A place that shares no filler is
not ranked low by the current machinery; it is INVISIBLE to it, at any cap and any depth. 387's
gate said so in its own words: if the lanes are redundant "what is needed is a genuinely
different move, one that reaches somewhere the walk cannot".

An address IS such a handle, and it is already written on the tape. `frame_keep` makes a place a
key `(w, left, right)` - two word sequences the corpus wrote. So a place can be named by its
HALVES, and halves compose:

    my place            the |  sat            L0 | R0
    keep my left        the |  ran            L0 | Rj      a place I share no filler with
    keep my right       a   |  sat            Li | R0
    compose             a   |  ran            Li | Rj      a place NOBODY offered me

The output space is then a PRODUCT, |L| x |R|, while the cost of naming a member of it is |L| +
|R|. That is the one shape that does not walk into 347's law - four measurements that a wider
offer costs more than it brings - because the offer stays eight; only the place it is drawn from
is chosen. Whether there is anything to draw is what this file measures, torch-free, BEFORE any
head is built for it.

WHAT IS MEASURED, on the population where reach matters (the truth is not among the question's
own values), with the hidden position excluded from everything:

    std8            the truth in the standing arm's eight (step interleaved with connect)  TODAY
    half_any        the truth anywhere among places sharing a half of my address           IS IT THERE
    half8           the truth in the address lane's eight
    half_only       the truth in the address eight and NOT in the standing eight           WHERE IT DECIDES
    comp8/comp_only the same for COMPOSED addresses - a half of mine or of a walked place,
                    crossed, kept only where the corpus actually wrote that frame
    orc8            the truth in either lane                         A PERFECT CHOOSER OF WHERE
    rand8           the same number of random places                 the floor
    d_half8         a FREQUENCY-MATCHED DECOY through the same lane  is the lane just frequency
    new_share       the share of address neighbours that share NO filler with my place - the
                    part of the reach that no existing channel can see, counted rather than
                    assumed from the cosine argument above

THE LEAK OF SECTION 27 IS THE FIRST THING THIS FILE GUARDS. Both 387 and 388 built the question's
neighbourhood out of `prof[pid]`, WHICH STILL CONTAINS THE HIDDEN TOKEN, and the leak was doing
most of the work (share .611 -> .180). Here every lane takes `qprof` - the profile with the query
row removed - as an argument, and no function reads `T["prof"][pid]` for the question's own place.
`_check390_address.py` re-introduces exactly that substitution and requires the numbers to move.

SAME-LINE PLACES ARE DROPPED FROM EVERY LANE. Frames overlap: `the cat sat on the mat` writes a
hole at `cat` for `the|sat` and for `|sat on` at the same token position, and a neighbouring frame
of the same line is the same words seen twice rather than a second record. `frame_keep` gives each
position exactly one address, so the hidden token itself cannot be re-read - but a place holding
another slot of the same line can still be a window artefact rather than evidence. The drop is
applied to the standing lanes too, so this stays a lane-against-lane comparison at equal footing,
and its direction is conservative: it can only cost a lane, never grant it.

  VOID CHECK, READ BEFORE ANYTHING ELSE. `new_share` and `n_half_new`. If the places that share a
  half are the same places that share a filler - n_half_new near zero - then the address names
  nothing the walk cannot already see, the premise of this step is false on this tape, and it
  closes here without a training run. That is 389's gate 4, kept because it worked.

  GATE. half_only > 0.05 OR comp_only > 0.05, with orc8 - std8 > 0.05, AND the null beaten:
  rand8_only below half of the winning lane's, and d_half8_only < 0.05 (a frequency-matched decoy
  must not ride the same lane). Then the next lever is the address as an output space in the
  stage, at the unchanged cap of eight. If the address lane is empty or is only frequency, the
  handle is closed on this tape and section 29's wall A has to be pushed on the word side.

  Nothing here decides on a new metric: reach@8, only_X and a matched decoy are the quantities
  387 and 388 were specified on.

    python _audit390_address.py
    python _audit390_address.py --window-lines 1600 --places 16
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import _tape_frames as tframes
v0 = v3('data/_wikitext103_train.txt')
v1 = v3('results/_stage390_address.json')

def build_tape(v4, v5=3, v6=1):
    """The frame tape as counts: places, profiles, and the two half indices the address needs."""
    v71, v72, v73 = v148.v74(v4, v5, v6)
    v7 = [v75 for v75, v172 in v71]
    v8 = [v80(v13) for v173, v13 in v71]
    v76, v77 = ({}, v79(v116))
    for v16, v13 in v78(v8):
        for v46 in v13:
            v76[v46] = v16
            v77[v73[v46]].v174(v16)
    v9 = [v85((v72[v46] for v46 in v13)) for v13 in v8]
    v10 = [v175.v151(v157((v90 * v90 for v90 in v81.v182()))) or 1.0 for v81 in v9]
    v11 = v79(v80)
    for v16, v81 in v78(v9):
        for v45 in v81:
            v11[v45].v149(v16)
    v82, v83, v84 = (v79(v80), v79(v80), {})
    for v16, (v112, v100, v101) in v78(v7):
        v82[v100].v149(v16)
        v83[v101].v149(v16)
        v84[v112, v100, v101] = v16
    v12 = v85()
    for v13 in v8:
        for v46 in v13:
            v12[v72[v46]] += 1
    v14 = v79(v80)
    for v45, v86 in v12.v87():
        v14[v86.v110()].v149(v45)
    return {'toks': v72, 'owner': v73, 'addrs': v7, 'places': v8, 'place_of': v76, 'on_line': v77, 'prof': v9, 'norm': v10, 'at_value': v150(v11), 'by_left': v150(v82), 'by_right': v150(v83), 'of_addr': v84, 'freq': v12, 'by_band': v14}

def filler_nbrs(v15, v16, v17, v18=()):
    """Places sharing at least one filler with the QUERY PROFILE, with overlap and dot.

    `qprof` is passed in and `T["prof"][pid]` is never read for the asking place. Section 27:
    reading it puts the hidden token in the search key, so the walk goes to the places holding
    the answer and then scores the answer out of exactly those places.
    """
    v88, v89 = (v85(), v85())
    for v45, v90 in v17.v87():
        for v23 in v15['at_value'].v102(v45, ()):
            if v23 != v16 and v23 not in v18:
                v88[v23] += 1
                v89[v23] += v90 * v15['prof'][v23][v45]
    return (v88, v89)

def walk_order(v15, v16, v17, v19, v18=()):
    """The k nearest places by fingerprint cosine - ONE definition, read by the walk and by the
    composer, so "the places the mind has in hand" cannot drift between the two."""
    v91, v89 = v92(v15, v16, v17, v18)
    v20 = v175.v151(v157((v90 * v90 for v90 in v17.v182()))) or 1.0
    return v97(v89, key=lambda v23: (-(v89[v23] / (v20 * v15['norm'][v23])), v23))[:v19]

def fillers_of(v15, v21, v22):
    """The fillers of a list of places, in place order, deduped, own values excluded. Every lane
    in this file takes its values this way, so a lane is only ever a different SET OF PLACES."""
    v24, v93 = ([], v116(v22))
    for v23 in v21:
        for v45, v152 in v15['prof'][v23].v153():
            if v45 not in v93:
                v93.v174(v45)
                v24.v149(v45)
    return v24

def lane_step(v15, v16, v22, v17, v19, v18=()):
    """The fingerprint walk: fillers of the k nearest places, in place order, uncut."""
    return v94(v15, v118(v15, v16, v17, v19, v18), v22)

def lane_share(v15, v16, v22, v17, v18=(), v25=False):
    """365's connect: a value scored by the summed overlap of the neighbours holding it.

    `norm_by_places` divides by how many places on the whole tape hold the value, which turns the
    sum into a mean overlap. 317, 383 and 387 each found the raw sum rewarding a value for being
    COMMON on a population whose answers are rare; both readings are printed so that is visible
    rather than assumed.
    """
    v88, v95 = v92(v15, v16, v17, v18)
    v26 = v85()
    for v23, v96 in v88.v87():
        for v45 in v15['prof'][v23]:
            if v45 not in v22:
                v26[v45] += v96
    if not v25:
        return [v45 for v45, v180 in v26.v153()]
    return v97(v26, key=lambda v45: (-(v26[v45] / v109(v15['at_value'][v45])), -v26[v45], v45))

def interleave(*v27, v28=None):
    """The stage's merge rule: round-robin, deduped, cut - never appended (347)."""
    v93, v24 = (v116(), [])
    for v29 in v98(*v27):
        for v45 in v29:
            if v45 is not None and v45 not in v93:
                v93.v174(v45)
                v24.v149(v45)
    return v24 if v28 is None else v24[:v28]

def half_nbrs(v15, v16, v18=()):
    """Places sharing a HALF of this address: pid -> how many of the two halves match.

    No filler is consulted, which is the whole point - this is the one relation on the tape that
    is not a function of what stands in the hole.
    """
    v99, v100, v101 = v15['addrs'][v16]
    v30 = v85()
    for v23 in v15['by_left'].v102(v100, ()):
        v30[v23] += 1
    for v23 in v15['by_right'].v102(v101, ()):
        v30[v23] += 1
    v30.v103(v16, None)
    for v23 in v18:
        v30.v103(v23, None)
    return v30

def compose_nbrs(v15, v16, v31, v18=()):
    """Addresses assembled from halves of places already in hand, kept only where the corpus
    wrote that frame.

    THE PRODUCT AT THE COST OF THE SUM. |L| lefts and |R| rights name up to |L|x|R| places while
    the mind emits one of |L| + |R| handles. Widths must match, because `frame_keep`'s key is
    (w, left, right) with |left| = |right| = w - that is the tape's rule, not a filter of mine.

    The places already in hand are excluded, so `comp_only` cannot be won by re-offering the walk.
    """
    v32 = {v15['addrs'][v23][1] for v23 in v31}
    v33 = {v15['addrs'][v23][2] for v23 in v31}
    v24 = v85()
    for v34 in v32:
        for v104 in v33:
            if v109(v34) != v109(v104):
                continue
            v23 = v15['of_addr'].v102((v109(v34), v34, v104))
            if v23 is not None and v23 != v16 and (v23 not in v31) and (v23 not in v18):
                v24[v23] += 1
    return v24

def half_cost(v15, v16, v23, v30):
    """How rare the matched half is - the count that orders the address neighbours.

    A place matched through `the|` is one of thousands; one matched through a half the corpus
    wrote twice is nearly an address on its own. Both are exact class sizes, so the ordering is a
    count and not a weight I chose. Ties fall back to tape order.
    """
    v99, v100, v101 = v15['addrs'][v16]
    v105, v106, v107 = v15['addrs'][v23]
    v35 = []
    if v106 == v100:
        v35.v149(v109(v15['by_left'][v100]))
    if v107 == v101:
        v35.v149(v109(v15['by_right'][v101]))
    return v154(v35) if v35 else v109(v15['places'])

def lane_addr_full(v15, v36, v22, v25=True):
    """Every value standing at an address neighbour, scored by the summed half-match count.

    Ranked the same two ways as `lane_share`, for the same reason and with the same default.
    """
    v26 = v85()
    for v23, v108 in v36.v87():
        for v45 in v15['prof'][v23]:
            if v45 not in v22:
                v26[v45] += v108
    if not v25:
        return [v45 for v45, v180 in v26.v153()]
    return v97(v26, key=lambda v45: (-(v26[v45] / v109(v15['at_value'][v45])), -v26[v45], v45))

def addr_order(v15, v16, v36, v19):
    """The k best-named address neighbours: most halves matched, then the RAREST matched half.

    The uncapped lane is a ceiling and reads every place sharing a half - on a common half that
    is hundreds. This is the affordable form, and both are printed so a ceiling cannot be
    mistaken for a channel.
    """
    return v97(v36, key=lambda v23: (-v36[v23], v184(v15, v16, v23, v36[v23]), v23))[:v19]

def lane_random(v15, v16, v22, v37, v38, v18=()):
    """The floor: the fillers of k random places, taken exactly the way every lane takes them."""
    v39 = v109(v15['places'])
    v21 = [v23 for v23 in (v38.v164(v39) for v43 in v113(v37)) if v23 != v16 and v23 not in v18]
    return v94(v15, v21, v22)

def band_draw(v15, v40, v41, v38):
    """373's frequency-matched decoy, unchanged, so the null is matched the way the others are."""
    v12, v14 = (v15['freq'], v15['by_band'])
    v42 = v12[v40].v110()
    v111, v112 = (v80(v14.v102(v42, ())), 0)
    while v109(v111) < 16 and v112 < 20:
        v112 += 1
        v111 += v14.v102(v42 - v112, [])
        v111 += v14.v102(v42 + v112, [])
    for v43 in v113(64):
        if not v111:
            return None
        v45 = v111[v38.v164(v109(v111))]
        if v45 != v40 and v45 not in v41:
            return v45
    return None

def rank_of(v44, v45):
    """1-based rank, or 0 for absent - so a mean over hits is read on an absolute quantity."""
    for v114, v115 in v78(v44):
        if v115 == v45:
            return v114 + 1
    return 0

def measure(v15, v46, v47, v38):
    """ONE QUESTION, every count it produces. Returns None when the question is out of the
    population (no own values, or the truth is among them - where reach is not the question).

    Split out of the loop so `_check390_address.py` can put ONE designed question through the
    same code the corpus run uses, and assert the numbers it must produce. A checker that
    re-implemented the lanes would be checking itself.
    """
    v72, v73 = (v15['toks'], v15['owner'])
    v16 = v15['place_of'][v46]
    v40 = v72[v46]
    v22 = {v72[v115] for v115 in v15['places'][v16] if v115 != v46}
    if not v22 or v40 in v22:
        return None
    v17 = v85((v72[v115] for v115 in v15['places'][v16] if v115 != v46))
    v18 = v116(v15['on_line'][v73[v46]])
    v18.v117(v16)
    v48 = v118(v15, v16, v17, v47.v8, v18)
    v49 = v94(v15, v48, v22)
    v50 = v119(v15, v16, v22, v17, v18, norm_by_places=False)
    v51 = v120(v49, v50)
    v52 = v51[:v47.v155]
    v53 = v121(v15, v16, v18)
    v122, v95 = v92(v15, v16, v17, v18)
    v54 = v123(v15, v16, v53, v47.v8)
    v55 = v116(v48) | v116(v54) | {v16}
    v56 = v124(v15, v16, v55, v18)
    v57 = v125(v15, v53, v22)
    v58 = v94(v15, v54, v22)[:v47.v155]
    v59 = v94(v15, v123(v15, v16, v56, v47.v8), v22)[:v47.v155]
    v60 = v156(v15, v16, v22, v47.v8, v38, v18)[:v47.v155]
    v126, v127, v128 = (v40 in v52, v40 in v58, v40 in v59)
    v61 = v40 in v60
    v62 = v129(v15, v40, v22 | {v40}, v38)
    v30 = {'n': 1, 'dropped': v109(v18), 'n_half': v109(v53), 'n_half_new': v157((1 for v23 in v53 if v23 not in v122)), 'n_fill': v109(v122), 'both_halves': v157((1 for v108 in v53.v182() if v108 >= 2)), 'n_pool': v109(v55), 'n_comp': v109(v56), 'n_comp_new': v157((1 for v23 in v56 if v23 not in v122)), 'std8': v2(v126), 'half_any': v2(v40 in v57), 'half8': v2(v127), 'comp8': v2(v128), 'rand8': v2(v61), 'half_only': v2(v127 and (not v126)), 'comp_only': v2(v128 and (not v126)), 'rand_only': v2(v61 and (not v126)), 'orc8': v2(v126 or v127 or v128), 'd_n': v2(v62 is not None), 'd_std8': v2(v62 is not None and v62 in v52), 'd_half8': v2(v62 is not None and v62 in v58), 'd_half_only': v2(v62 is not None and v62 in v58 and (v62 not in v52)), 'd_comp_only': v2(v62 is not None and v62 in v59 and (v62 not in v52))}
    v30['_lanes'] = {'std': v52, 'half': v58, 'comp': v59, 'rand': v60}
    v30['_rank_half'] = v158(v57, v40) if v127 else 0
    v30['_rank_std'] = v158(v51, v40) if v126 else 0
    return v30

def run(v15, v47, v38):
    """One pass over the questions. Every count returned; nothing is decided here."""
    v90, v130 = (v85(), {'half': [], 'std': []})
    v63 = [v46 for v13 in v15['places'] for v46 in v13]
    v38.v131(v63)
    for v46 in v63:
        if v90['n'] >= v47.v159:
            break
        v30 = v160(v15, v46, v47, v38)
        if v30 is None:
            continue
        for v19, v45 in v30.v87():
            if not v19.v181('_'):
                v90[v19] += v45
        if v30['_rank_half']:
            v130['half'].v149(v30['_rank_half'])
        if v30['_rank_std']:
            v130['std'].v149(v30['_rank_std'])
    return (v90, v130)

def main() -> v2:
    v64 = v161.v132()
    v64.v133('--bytes', type=v2, default=30000000)
    v64.v133('--frame-max', type=v2, default=3)
    v64.v133('--min-fillers', type=v2, default=1)
    v64.v133('--lines', type=v2, default=25000)
    v64.v133('--window-lines', type=v2, default=400)
    v64.v133('--places', type=v2, default=8)
    v64.v133('--topm', type=v2, default=8)
    v64.v133('--max-questions', type=v2, default=3000)
    v64.v133('--seed', type=v2, default=1337)
    v64.v133('--corpus', default=v169(v0))
    v64.v133('--out', default=v169(v1))
    v47 = v64.v134()
    v65 = v3(v47.v183).v176('r', encoding='utf-8', errors='ignore').v135(v47.v136)
    v66 = [v34.v162() for v34 in v65.v177('\n') if v109(v34.v162()) >= 80]
    v4 = v66[:v2(0.7 * v109(v66))][:v47.v4]
    v38 = v163.v137(v47.v138)
    if v47.v139 and v47.v139 < v109(v4):
        v140 = v38.v164(v109(v4) - v47.v139)
        v4 = v4[v140:v140 + v47.v139]
    v15 = v141(v4, v47.v5, v47.v6)
    if not v15['places']:
        v144('no tape')
        return 1
    v90, v130 = v142(v15, v47, v38)
    v39 = v143(1, v90['n'])
    v67 = v143(1, v90['d_n'])

    def f(v19, v62=v39):
        return v90[v19] / v62
    v68 = {'seed': v47.v138, 'window_lines': v47.v139, 'places_on_tape': v109(v15['places']), 'n': v90['n'], 'same_line_dropped': v86('dropped'), 'n_half': v86('n_half'), 'n_half_new': v86('n_half_new'), 'n_fill': v86('n_fill'), 'new_share': v90['n_half_new'] / v143(1, v90['n_half']), 'both_halves': v86('both_halves'), 'n_comp': v86('n_comp'), 'n_comp_new': v86('n_comp_new'), 'n_pool': v86('n_pool'), 'std8': v86('std8'), 'half8': v86('half8'), 'comp8': v86('comp8'), 'rand8': v86('rand8'), 'half_any': v86('half_any'), 'orc8': v86('orc8'), 'half_only': v86('half_only'), 'comp_only': v86('comp_only'), 'rand_only': v86('rand_only'), 'orc_minus_std': v86('orc8') - v86('std8'), 'd_n': v90['d_n'], 'd_std8': v86('d_std8', v67), 'd_half8': v86('d_half8', v67), 'd_half_only': v86('d_half_only', v67), 'd_comp_only': v86('d_comp_only', v67), 'rank_half_mean': v157(v130['half']) / v109(v130['half']) if v130['half'] else 0.0, 'rank_std_mean': v157(v130['std']) / v109(v130['std']) if v130['std'] else 0.0}
    v144(f"places {v68['places_on_tape']}  questions {v68['n']}  same-line dropped {v68['same_line_dropped']:.2f}/q")
    v144('VOID CHECK  n_half {n_half:.2f}  of which unseen by the walk {n_half_new:.2f} ({new_share:.3f})  filler-nbrs {n_fill:.2f}  both halves {both_halves:.2f}'.v165(**v68))
    v144('            n_comp {n_comp:.2f}  new {n_comp_new:.2f}  from a pool of {n_pool:.1f} places'.v165(**v68))
    v144('REACH@8     std {std8:.4f}  half {half8:.4f}  comp {comp8:.4f}  rand {rand8:.4f}  half_any {half_any:.4f}'.v165(**v68))
    v144('ONLY        half {half_only:.4f}  comp {comp_only:.4f}  rand {rand_only:.4f}  oracle-std {orc_minus_std:+.4f}'.v165(**v68))
    v144('DECOY       std {d_std8:.4f}  half {d_half8:.4f}  half_only {d_half_only:.4f}  comp_only {d_comp_only:.4f}'.v165(**v68))
    v144('RANK        truth in the full address lane {rank_half_mean:.1f}  in the standing lane {rank_std_mean:.1f}'.v165(**v68))
    v24 = v3(v47.v24)
    v24.v166.v145(parents=True, exist_ok=True)
    v69 = v178.v168(v24.v179()) if v24.v167() else {}
    v69[v169(v47.v138)] = v68
    v24.v146(v178.v170(v69, indent=1))
    v144(f'wrote {v24}')
    return 0
if v70 == '__main__':
    raise v147(v171())