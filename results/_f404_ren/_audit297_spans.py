"""297: can the tape be cut into slots WITHOUT a parser - and what does it cost?

Three tapes so far were cut by the same rule: extract (anchor | relation | value) from a
sentence, then group by an absolute cosine tau conjoined with a word overlap. The 293 audit
showed what that rule actually produces on wikitext - `canada|and`, `december|the`,
`british|unk` - and `rare_nonzero_rate = 0.0` said the same from the other side: everything two
rows share is ambient English. Every open failure since runs through that cut.

So: cut by the corpus instead of by a grammar. Three rules, all exact, all with NO constant to
choose - which is the whole point, because a threshold at the cutting step is a heuristic at the
place the invariant forbids one:

  words    a slot is a run of non-space characters. Nothing linguistic, nothing tunable.
  repeats  a slot is the LONGEST substring starting here that the corpus writes at least twice.
           Not a chosen length and not a chosen vocabulary size - BPE without the vocabulary
           constant. A piece that occurs once is not knowledge, it is a one-off, and the corpus
           itself says which is which.
  frames   a PLACE is a hole whose left+right frame the corpus writes at least twice, with a
           different filler. Width is the MAXIMUM that still repeats - same idea as repeats, but
           applied to the surroundings of a hole, not to a solid span. The text is punctured,
           not sliced. A frame with several fillers IS an address with several mentions, with no
           tau and no stop-words.

FOUR NUMBERS DECIDE for words/repeats, and none of them needs a model:

  SUPPLY       slots and distinct slots. A tape with no repetition is not a tape, it is text.
  COST         slots per KB, and how many occurrences a type has - that is how many rows an
               import brings, and the graph is quadratic in rows.
  COMPOSITION  the number this audit exists for: what fraction of held-out slots have a type
               the training corpus never wrote. Those cannot be answered by selecting an
               existing row - they must be ASSEMBLED, and 1-NN cannot assemble. If this is
               near zero the new cut changes nothing and we are back to selection; if it is
               large, composition is real and its supply is unlimited.
  CEILING      of the answerable ones, how often the answer is the single most frequent
               continuation - the floor any mind has to clear before it is interesting.

FRAMES add the place-shaped numbers, compared in the face to the parser tape (~800 addresses /
~2500 slots, ~3.1 mentions per address):

  addresses            frames with >= 2 distinct fillers
  mentions_per_address occurrences (rows) per such frame
  corpus_coverage      fraction of token positions that sit in such a hole
  median_frame_width   tokens of context on EACH side at the max width taken

    python _audit297_spans.py [--bytes 200000] [--cap 32] [--frame-max 12]
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
v0 = v4('data/_wikitext103_train.txt')
v1 = v4('results')
v2 = {'addresses': 800, 'slots': 2500, 'mentions_per_address': 3.1}

class SAM:
    """Suffix automaton: every substring of the text, with its occurrence count, in O(n).

    Used for one question only - "is the piece starting here written more than once?" - which is
    exactly what makes `repeats` a rule of the corpus rather than a rule of ours.
    """

    def __init__(v63):
        v63.v64 = [{}]
        v63.v65 = [-1]
        v63.v66 = [0]
        v63.v67 = [0]
        v63.v68 = 0

    def extend(v63, v69):
        v70 = v82(v63.v64)
        v63.v64.v106({})
        v63.v65.v106(-1)
        v63.v66.v106(v63.v66[v63.v68] + 1)
        v63.v67.v106(1)
        v71 = v63.v68
        while v71 != -1 and v69 not in v63.v64[v71]:
            v63.v64[v71][v69] = v70
            v71 = v63.v65[v71]
        if v71 == -1:
            v63.v65[v70] = 0
        else:
            v59 = v63.v64[v71][v69]
            if v63.v66[v71] + 1 == v63.v66[v59]:
                v63.v65[v70] = v59
            else:
                v128 = v82(v63.v64)
                v63.v64.v106(v141(v63.v64[v59]))
                v63.v65.v106(v63.v65[v59])
                v63.v66.v106(v63.v66[v71] + 1)
                v63.v67.v106(0)
                while v71 != -1 and v63.v64[v71].v131(v69) == v59:
                    v63.v64[v71][v69] = v128
                    v71 = v63.v65[v71]
                v63.v65[v59] = v63.v65[v70] = v128
        v63.v68 = v70

    def finish(v63):
        v72 = v79(v83(v82(v63.v64)), key=lambda v31: -v63.v66[v31])
        for v73 in v72:
            v111 = v63.v65[v73]
            if v111 > 0 or v111 == 0:
                if v111 >= 0:
                    v63.v67[v111] += v63.v67[v73]

def cut_words(v5, v6=None, v7=0):
    """A slot is a run of non-space characters. Exact, and there is nothing to tune."""
    v8, v31, v25 = ([], 0, v82(v5))
    while v31 < v25:
        while v31 < v25 and v5[v31].v129():
            v31 += 1
        v74 = v31
        while v74 < v25 and (not v5[v74].v129()):
            v74 += 1
        if v74 > v31:
            v8.v106(v5[v31:v74])
        v31 = v74
    return v8

def cut_repeats(v5, v9, v10):
    """A slot is the longest piece starting here that the corpus writes at least twice.

    `cap` bounds the forward walk. It is a COST bound, not a decision: it can only shorten a
    slot the corpus would have allowed, never invent one, and it is reported so its effect is
    visible rather than assumed.
    """
    v8, v31, v25 = ([], 0, v82(v5))
    while v31 < v25:
        v73, v75 = (0, 0)
        for v46 in v83(v31, v130(v25, v31 + v10)):
            v73 = v9.v64[v73].v131(v5[v46], -1)
            if v73 < 0 or v9.v67[v73] < 2:
                break
            v75 = v46 - v31 + 1
        if v75 == 0:
            v75 = 1
        v8.v106(v5[v31:v31 + v75])
        v31 += v75
    return v8

def stats(v11, v12, v13, v14, v10):
    v76, v77 = (v90(v12), v90(v13))
    v15 = v78(1, v82(v12))
    v16 = v80((v92 for v124, v92 in v76.v94() if v92 >= 2)) / v15
    v17 = v79(v76.v91())
    v18 = v80((1 for v124 in v77 if v124 not in v76))
    v19 = v80((v92 for v124, v92 in v77.v94() if v124 not in v76))
    v20 = v78(1, v82(v13))
    v21 = v76.v139(1)[0][0] if v76 else None
    v22 = v80((v92 for v124, v92 in v77.v94() if v124 == v21)) / v20
    return {'cutter': v11, 'cap': v10 if v11 == 'repeats' else None, 'slots_train': v82(v12), 'distinct_train': v82(v76), 'slots_per_kb': v82(v12) / v78(1.0, v14 / 1024), 'mean_len': v80((v82(v124) for v124 in v12)) / v15, 'recurrence': v16, 'occurrences_median': v17[v82(v17) // 2] if v17 else 0, 'occurrences_p90': v17[v3(0.9 * (v82(v17) - 1))] if v17 else 0, 'must_compose_types': v18 / v78(1, v82(v77)), 'must_compose_occurrences': v19 / v20, 'trivial_ceiling': v22}

def tokenize(v5):
    """Whitespace tokens only - the frame cutter may not smuggle a grammar in through the split."""
    return v81(v5)

def audit_frames(v23, v24):
    """Places = holes whose maximal repeating frame has >= 2 distinct fillers.

    Hole = one token. Frame width w = w tokens on EACH side. For every position take the largest
    w at which that exact (left, right) still occurs somewhere else in the corpus (>= 2 hits);
    keep the frame as an address only when those hits carry at least two different fillers.
    No tau, no stop-list, no chosen width - only counting, with w_max a cost bound like repeats'
    cap (can only shrink, never invent).
    """
    v25 = v82(v23)
    v26 = []
    for v27 in v83(1, v24 + 1):
        v84 = v93(v87)
        for v31 in v83(v27, v25 - v27):
            v112 = v132(v23[v31 - v27:v31])
            v113 = v132(v23[v31 + 1:v31 + 1 + v27])
            v84[v112, v113].v106(v23[v31])
        v26.v106(v84)
    v28 = {}
    v29 = {}
    v30 = 0
    for v31 in v83(v25):
        v85 = None
        for v27 in v83(v130(v24, v31, v25 - v31 - 1), 0, -1):
            v112 = v132(v23[v31 - v27:v31])
            v113 = v132(v23[v31 + 1:v31 + 1 + v27])
            v114 = (v112, v113)
            if v82(v26[v27 - 1][v114]) >= 2:
                v85 = (v27, v114)
                break
        if v85 is None:
            continue
        v27, v114 = v85
        v43 = v26[v27 - 1][v114]
        if v82(v88(v43)) < 2:
            continue
        v86 = (v27, v114)
        if v86 not in v28:
            v28[v86] = v87(v43)
            v29[v86] = v27
        v30 += 1
    v32 = v82(v28)
    v33 = v80((v82(v73) for v73 in v28.v91()))
    v34 = v79((v82(v73) for v73 in v28.v91()))
    v35 = v79((v82(v88(v73)) for v73 in v28.v91()))
    v36 = v87(v29.v91())
    v37 = []
    for (v27, (v112, v113)), v43 in v87(v28.v94())[:5]:
        v37.v106({'w': v27, 'left': v87(v112), 'right': v87(v113), 'fillers': v79(v88(v43))[:8], 'n_fillers': v82(v88(v43)), 'rows': v82(v43)})
    v38 = v39 = 0
    v40 = v88()
    for v27, v84 in v89(v26, start=1):
        for v114, v43 in v84.v94():
            if v82(v43) < 2 or v82(v88(v43)) != 1:
                continue
            v86 = (v27, v114)
            if v86 in v40:
                continue
            v40.v133(v86)
            v38 += 1
            v39 += v82(v43)
    return {'cutter': 'frames', 'frame_max': v24, 'hole': 'one_token', 'tokens': v25, 'addresses': v32, 'slots': v33, 'mentions_per_address_mean': v33 / v32 if v32 else 0.0, 'mentions_per_address_median': v34[v82(v34) // 2] if v34 else 0, 'distinct_fillers_per_address_median': v35[v82(v35) // 2] if v35 else 0, 'corpus_coverage': v30 / v78(1, v25), 'median_frame_width': v79(v36)[v82(v36) // 2] if v36 else 0, 'mean_frame_width': v80(v36) / v82(v36) if v36 else 0.0, 'confirm_only_addresses': v38, 'confirm_only_rows': v39, 'parser_ref': v2, 'vs_parser_addresses': v32 / v2['addresses'] if v32 else 0.0, 'vs_parser_slots': v33 / v2['slots'] if v33 else 0.0, 'sample': v37, '_addr_lists': v28, '_widths': v29}

def frame_quality(v28, v29):
    """Slice the frame tape: filler histogram, CONFIRM/DISPUTE, width vs informativeness.

    Quality of a frame = how many DISTINCT fillers it takes - a count, not a threshold. Narrow
    ambient frames (`the ___ of`) take hundreds; informative ones take few. CONFIRM = extra
    rows of a value already on the address; DISPUTE = a different value at the same frame -
    measurable here because the address is the frame, not a tau-group.
    """
    v41 = v90((v82(v88(v43)) for v43 in v28.v91()))
    v42 = v90()
    for v43 in v28.v91():
        v42[v82(v88(v43))] += v82(v43)
    v33 = v80(v42.v91()) or 1
    v44 = {}
    v45 = 0
    for v46 in v79(v42):
        v45 += v42[v46]
        v44[v115(v46)] = v45 / v33
    v47 = v48 = v49 = 0
    for v43 in v28.v91():
        v92 = v90(v43)
        v47 += v80((v25 - 1 for v25 in v92.v91() if v25 > 1))
        if v82(v92) >= 2:
            v48 += 1
            v49 += v80(v92.v91())
    v32 = v78(1, v82(v28))
    v50 = v80((v82(v140) for v140 in v28.v91())) or 1
    v26 = v93(v87)
    for v86, v43 in v28.v94():
        v26[v29[v86]].v106(v82(v88(v43)))
    v51 = {v115(v27): {'addresses': v82(v134), 'mean_distinct_fillers': v80(v134) / v82(v134), 'median_distinct_fillers': v79(v134)[v82(v134) // 2], 'pct_addresses': v82(v134) / v32} for v27, v134 in v79(v26.v94())}
    return {'cutter': 'frames_quality', 'n_addresses': v82(v28), 'n_slots': v50, 'distinct_fillers_hist': {v115(v46): v41[v46] for v46 in v79(v41)}, 'slots_by_distinct_fillers': {v115(v46): v42[v46] for v46 in v79(v42)}, 'slot_mass_cumulative_le_k_fillers': v44, 'slot_mass_at_2_fillers': v42.v131(2, 0) / v33, 'slot_mass_at_3_to_5': v80((v42.v131(v46, 0) for v46 in v83(3, 6))) / v33, 'slot_mass_at_6_to_20': v80((v42.v131(v46, 0) for v46 in v83(6, 21))) / v33, 'slot_mass_above_20': v80((v73 for v46, v73 in v42.v94() if v46 > 20)) / v33, 'confirm_extra_rows': v47, 'confirm_rate_of_rows': v47 / v50, 'addresses_with_dispute': v48, 'dispute_address_rate': v48 / v32, 'rows_on_dispute_addresses': v49 / v50, 'width_vs_fillers': v51}

def main() -> v3:
    v52 = v116.v95()
    v52.v96('--bytes', type=v3, default=200000)
    v52.v96('--cap', type=v3, default=32)
    v52.v96('--frame-max', type=v3, default=12, help='cost bound on frame half-width in tokens; shrinks only, never invents')
    v52.v96('--skip-spans', action='store_true', help='only run the frames audit (spans already measured)')
    v53 = v52.v97()
    if not v0.v117():
        v100(f'missing {v0}')
        return 1
    v54 = v0.v118(encoding='utf-8', errors='ignore')[:v53.v126 * 2]
    v55 = v82(v54) // 2
    v98, v99 = (v54[:v55], v54[v55:v55 + v53.v126 // 2])
    v100(f'train {v82(v98)} bytes, held {v82(v99)} bytes')
    v8 = []
    if not v53.v101:
        v9 = v119()
        for v69 in v98:
            v9.v135(v69)
        v9.v120()
        v100(f'suffix automaton: {v82(v9.v64)} states')
        for v11, v121 in (('words', v81), ('repeats', v136)):
            v122 = v121(v98, v9, v53.v10)
            v123 = v121(v99, v9, v53.v10)
            if v11 == 'repeats':
                assert ''.v142(v122) == v98 and ''.v142(v123) == v99, 'repeats lost characters'
            v124 = v137(v11, v122, v123, v82(v98), v53.v10)
            v8.v106(v124)
            v100(v138.v125(v124))
            v100('  sample:', v122[200:215])
    v56 = v102(v98)
    v100(f'frames: {v82(v56)} tokens, frame-max {v53.v104}')
    v57 = v103(v56, v53.v104)
    v58 = v57.v105('_addr_lists')
    v36 = v57.v105('_widths')
    v8.v106(v57)
    v100(v138.v125({v46: v73 for v46, v73 in v57.v94() if v46 != 'sample'}))
    v100('  sample frames:', v138.v125(v57['sample'][:3], ensure_ascii=False))
    v59 = v107(v58, v36)
    v8.v106(v59)
    v100(v138.v125(v59, indent=2))
    v1.v108(exist_ok=True)
    v60 = f'_{v53.v126}' if v53.v126 != 200000 else ''
    v61 = v1 / f'stage297_span_audit{v60}.json'
    v61.v109(v138.v125(v8, indent=2), encoding='utf-8')
    v100(f'wrote {v61}')
    return 0
if v62 == '__main__':
    raise v110(v127())