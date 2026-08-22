"""The tape written by counting, not by a grammar - 297's frame cutter as a pack.

WHAT A PLACE IS HERE. Not (anchor | relation) pulled out of a sentence by a rule, but a HOLE
whose surroundings recur: take a token, widen the window left and right while the frame is still
something the corpus wrote at least twice, and the widest such frame is the address. What varies
in the hole is the value. No tau, no stopword list, no grammar, and no length chosen by us - the
width is whatever the corpus supports. The 293 audit is why: the parser's relation half is a
function word on this corpus (`canada|and`, `december|the`), and `rare_nonzero_rate` was 0.0.

WHAT THE AUDIT SAID, so the shape of this tape is not a hope: at 600 KB, 7947 addresses and
42266 slots against the parser's ~800 and ~2500, mentions per address 5.3 against 3.1, 35% of
tokens covered, and 22% of rows repeating a filler already at their frame - the CONFIRM supply
that was unmeasurable on the parsed tape. It also killed one of my guesses: width does NOT
predict quality (w=1 averages 4.31 distinct fillers, w=2 only 2.73), so nothing here selects on
width. The quality signal is the number of distinct fillers, which is a count and travels into
the graph as one.

WHY THIS IS A PACK AND NOT A STAGE. The frame tape has the SAME SHAPE as the parsed one -
addresses, slots, values, contexts - so every reader downstream (build_graph, Phi, imports, both
rivals) runs unchanged. Only the writing rule is replaced, which is the whole point: exact
counting in the write path, and the interpolation moved to where the mind lives.
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict
v0: v1 = {}

def frame_keep(v3, v4: v31=12, v5: v31=2):
    """The counted frames of a corpus - everything in the write path that the draw cannot change.

    COUNTED ONCE PER CORPUS, NOT ONCE PER RESAMPLE. The tape is redrawn every 50 steps, 80 times
    in a 4000-step run, and every redraw re-counted every window of the whole corpus to produce
    the same `keep` and then sample it differently. Only the SAMPLE depends on the rng. Memoised
    on the identity of the line list, and the list itself is held in the entry so the id cannot
    be recycled onto a different corpus.

    THE PRUNE IS EXACT, NOT A SHORTCUT. A window of width w+1 contains the width-w window with
    the same hole, so its count can only be lower: a position whose width-w frame occurred once
    cannot have a width-(w+1) frame that occurs twice. Widening only positions still alive turns
    `frame_max` full passes over the corpus into one full pass plus a shrinking tail, and the
    peak allocation from three Counters of million-key tuples - the several gigabytes of private
    working set that grew across the run - never happens. Any key with a true count of 2 or more
    keeps all of its positions alive, so counts on the restricted set equal the global ones for
    every key that survives; keys counted low are exactly the ones being discarded.
    """
    v6 = v32(v3)
    v7 = v0.v33(v6)
    if v7 is not None and v7[0] is v3 and (v7[1] == (v4, v5)):
        return (v7[2], v7[3], v7[4])
    v15, v12 = ([], [])
    for v34, v35 in v36(v3):
        for v37 in v35.v62():
            v15.v64(v37)
            v12.v64(v34)
    v8 = v38(v15)
    if v8 < 3:
        return ([], v15, v12)
    v39, v40 = ({}, v46(v41(1, v8 - 1)))
    for v9 in v41(1, v4 + 1):
        v42 = v45(v46)
        for v43 in v40:
            if v43 - v9 < 0 or v43 + 1 + v9 > v8:
                continue
            v42[v81(v15[v43 - v9:v43]), v81(v15[v43 + 1:v43 + 1 + v9])].v64(v43)
        v44 = []
        for (v69, v70), v63 in v42.v47():
            if v38(v63) >= 2:
                for v43 in v63:
                    v39[v43] = (v9, v69, v70)
                v44.v76(v63)
        v40 = v44
        if not v40:
            break
    v10 = v45(v46)
    for v43, v13 in v39.v47():
        v10[v13].v64(v43)
    v11 = [(v13, v57(v49)) for v13, v49 in v10.v47() if v38({v15[v43] for v43 in v49}) >= v5]
    v11.v48(key=lambda v77: (-v77[0][0], -v38(v77[1]), v77[0][1]))
    while v38(v0) >= 2:
        v0.v65(v74(v78(v0)))
    v0[v6] = (v3, (v4, v5), v11, v15, v12)
    return (v11, v15, v12)
v2: v1 = {}

def _by_line(v11, v12):
    """line -> the (address, position) pairs written on it, built once per corpus.

    Keyed on the identity of `keep`, which frame_keep memoises, so a run that redraws the tape
    eighty times indexes the corpus once. Rebuilding it per draw cost 0.46s on a sixth of the
    pool - the same shape of waste as recounting every window per resample.
    """
    v13 = v32(v11)
    v7 = v2.v33(v13)
    if v7 is not None and v7[0] is v11:
        return v7[1]
    v14 = v45(v46)
    for v30, v49 in v11:
        for v43 in v49:
            v14[v12[v43]].v64((v30, v43))
    while v38(v2) >= 2:
        v2.v65(v74(v78(v2)))
    v2[v13] = (v11, v14)
    return v14

def frame_region(v11, v15, v12, v16, v17, v18, v5=2):
    """The same number of addresses, taken from a CONTIGUOUS stretch of lines.

    WHY THE DRAW IS THE DEFECT. Sampling addresses uniformly takes 3000 of 250195 qualifying
    frames - 1.2% - so two frames of one sentence land on the tape together about once in a
    hundred, and EVERY relation between places is diluted by that ratio before anything reads
    them. The 300 audit measured the dilution rather than the corpus: `line` read 0.041 at 7.7%
    of the pool and 0.0067 at 1.2%, falling with the fraction. On a region tape of the same size
    it reads 0.237, and the questions only adjacency can answer go from 0.1% to 5.0%.

    It also repairs the walk, which was the surprise. Neighbouring places are now neighbours in
    the text rather than accidental look-alikes: the median rank of the truth in the walk's own
    candidate list falls from 21 to 5, reachability at eight candidates rises from 0.162 to
    0.271, and the eight nearest places offer 84 fillers instead of 1015.

    The price is real and is not hidden: mentions per address 7.09 -> 3.75, own 0.450 -> 0.346,
    share 0.596 -> 0.535. A place seen inside one window has been seen fewer times. Mentions
    OUTSIDE the window are dropped, because a tape that is a region cannot quietly borrow
    evidence from the rest of the corpus.
    """
    v19 = v50(v11, v12)
    v20 = v18.v51(v61(1, v16))
    v52, v53, v54 = (v45(v46), v45(v60), 0)
    for v21 in v41(v16):
        for v13, v43 in v19.v33((v20 + v21) % v16, ()):
            v52[v13].v64(v43)
            v66 = v15[v43]
            if v66 not in v53[v13]:
                v53[v13].v72(v66)
                if v38(v53[v13]) == v5:
                    v54 += 1
        if v54 >= v17:
            break
    v22 = [(v13, v57(v66)) for v13, v66 in v52.v47() if v38(v53[v13]) >= v5]
    return v22[:v17] if v17 else v22

def frame_assertions(v3, v4: v31=12, v5: v31=2, v17: v31=0, v18=None, v23: v55='uniform'):
    """Assertions in 279's shape - address, value, ctx, source - cut by recurrence alone.

    Counting and nothing else: `frame_keep` gives each token the widest window the corpus wrote
    twice, and an address survives only if its hole took at least `min_fillers` DIFFERENT values,
    because a frame with one filler is a fixed phrase, not a place where something is written.
    Everything here that is not counting is the draw.
    """
    v11, v15, v12 = v56(v3, v4, v5)
    if not v11:
        return ([], [], 0)
    v24 = v38(v11)
    v25 = v18 or v75.v67(0)
    if v23 == 'region':
        v11 = v68(v11, v15, v12, v38(v3), v17, v25, v5)
    elif v17 and v24 > v17:
        v11 = v25.v23(v11, v17)
    v22, v27 = ([], [])
    for (v9, v69, v70), v49 in v57(v11, key=lambda v77: (v77[0][1], v77[0][2])):
        v58 = f"{' '.v79(v69)}|{' '.v79(v70)}"
        v27.v64(v58)
        for v71, v43 in v36(v49):
            v22.v64({'address': v58, 'value': v15[v43], 'ctx': v3[v12[v43]], 'source': f'frame:{v58}:{v71}', 'pos': v43, 'line': v12[v43]})
    return (v22, v27, v24)

def frame_report(v26, v27):
    v28 = v59((v30['address'] for v30 in v26))
    v29 = v45(v60)
    for v30 in v26:
        v29[v30['address']].v72(v30['value'])
    v8 = v61(1, v38(v27))
    return {'addresses': v38(v27), 'slots': v38(v26), 'mentions_per_address': v38(v26) / v8, 'distinct_fillers_mean': v73((v38(v66) for v66 in v29.v80())) / v8, 'confirm_rows': v73((v28[v30] - v38(v29[v30]) for v30 in v27))}