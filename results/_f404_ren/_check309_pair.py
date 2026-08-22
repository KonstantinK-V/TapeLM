"""Does the two-hole verb build honest questions - checked here, where the stage cannot run.

WHY THIS EXISTS. Every fault this project has shipped was wiring or a leak, and both are
invisible until after training. The pair verb adds a new way to leak that none of the earlier
checks can see: two holes on ONE line, so one hole's frame can contain the other's hidden token
and hand over the answer. The rule that closes it is a distance in corpus positions, which is
exact and therefore checkable - so it is checked, on a corpus small enough to verify by hand.

The pure-python half of the verb is lifted out of the stage by AST and run against a synthetic
pack with the walk stubbed. Nothing here needs torch, which is the point.

    python _check309_pair.py
"""
from __future__ import annotations
import ast
import random
from collections import Counter, defaultdict
import _tape_frames as tframes
v0 = '_stage289_derivation.py'
v1 = ('reach_question', 'pair_offer', 'pair_offers', 'pair_question', 'pair_rivals', 'pair_joint_index', 'pair_questions_for', 'reach_line_index', 'outside_mentions')

def lift():
    """The verb's pure-python functions, executed with the walk stubbed out."""
    v3 = v77(v0, encoding='utf-8').v27()
    v4 = v52.v31(v3)
    v5 = {'Counter': v32, 'defaultdict': v33, 'object': v34}
    for v6 in v4.v7:
        if v78(v6, v52.v79) and v80((v78(v94, v52.v99) and v94.v105.v100() for v94 in v6.v95)):
            try:
                v84(v89(v52.v96([v6], []), '<c>', 'exec'), v5)
            except v81:
                pass
    for v6 in v4.v7:
        if v78(v6, v52.v82) and v6.v83 in v1:
            v84(v89(v52.v96([v6], []), '<f>', 'exec'), v5)
    v8 = [v35 for v35 in v1 if v35 not in v5]
    if v8:
        raise v51(f'could not lift {v8}')
    return v5

class Tape:

    def __init__(v36, v11):
        v36.v11 = v11

def build_pack(v9, v10, v5):
    """A pack in the stage's shape, with only what the pair verb reads."""
    v37, v38, v39 = v53.v40(v9, v10, 2, 0, v85.v54(0))
    v11 = [v55['value'] for v55 in v37]
    v12 = [v55['address'] for v55 in v37]
    v13 = {'tape': v56(v11), 'straddr': v12, 'line': [v55['line'] for v55 in v37], 'pos': [v55['pos'] for v55 in v37]}
    v14 = v33(v41)
    for v42, v43 in v44(v12):
        v14[v43].v57(v42)
    v13['items'] = [{'address': v43, 'S': v43, 'slots': v58} for v43, v58 in v14.v15() if v87(v58) >= 2]
    v15 = v13['items']
    v16 = [[(v74, [v90 for v90 in v59['slots'] if v13['tape'].v11[v90] == v74], v86) for v74, v86 in v32((v13['tape'].v11[v90] for v90 in v59['slots'])).v15()] for v59 in v15]
    v17 = {'items': v15, 'fills': v16, 'of': {v59['address']: v60 for v60, v59 in v44(v15)}}
    v5['reach_index'] = lambda v61: v17
    v5['reach_candidates'] = lambda v61, v62: {'cands': [v74 for v101, v59 in v44(v15) if v59['address'] != v62['address'] for v74, v102, v103 in v16[v101]][:8]}
    return v13

def main() -> v2:
    v5 = v45()
    v10 = 3
    v5['FRAME_MAX'] = v10
    v9 = ['the capital of france is paris and the capital of spain is madrid today', 'the capital of spain is madrid and the capital of italy is rome today', 'the capital of italy is rome and the capital of france is paris today', 'a red car drove past the old grey house near the small river bank there', 'a blue car drove past the new grey house near the small river bank there', 'a green car drove past the old white house near the large river bank there']
    v13 = v46(v9, v10, v5)
    v18 = v5['pair_questions_for'](v13, v85.v54(1337))
    for v19 in v18:
        v5['pair_offers'](v13, v19)
    v20 = True
    v47(f"tape   {v87(v13['straddr'])} slots, {v87(v13['items'])} places, {v87(v18)} pair questions")
    if not v18:
        v47('no questions built - the check cannot say anything')
        return 1
    v21 = [v19 for v19 in v18 if v91(v13['pos'][v19['holes'][0]['slot']] - v13['pos'][v19['holes'][1]['slot']]) <= v10]
    v47(f"leak   holes closer than frame_max: {v87(v21)} -> {('OK' if not v21 else 'BROKEN')}")
    v20 &= not v21
    v22 = []
    for v19 in v18:
        v55, v63 = v19['holes']
        for v64, v65 in ((v55, v63), (v63, v55)):
            if v65['truth'] in v64['address'].v104('|', ' ').v92():
                v22.v57((v64['address'], v65['truth']))
    v47(f"leak   truth of one hole inside the other's address: {v87(v22)} -> {('OK' if not v22 else 'BROKEN')}")
    v20 &= not v22
    v23 = [v19 for v19 in v18 for v66 in v19['holes'] if v66['slot'] in v66['rows']]
    v47(f"hide   a hole carrying its own row as evidence: {v87(v23)} -> {('OK' if not v23 else 'BROKEN')}")
    v20 &= not v23
    v24 = []
    for v19 in v18:
        for v67, v66 in v44(v19['holes']):
            if v19['slots'][v19['query_rows'][v67]] != v66['slot']:
                v24.v57(v19)
        if v87(v19['slots']) != v87(v19['vals']) or v87(v97(v19['query_rows'])) != 2:
            v24.v57(v19)
    v47(f"shape  worlds whose query rows do not point at the holes: {v87(v24)} -> {('OK' if not v24 else 'BROKEN')}")
    v20 &= not v24
    v25 = v48((1 for v19 in v18 for v66 in v19['holes'] if v98((v74 in v66['own'] for v74 in v66['offer'])) and v98((v74 not in v66['own'] for v74 in v66['offer']))))
    v26 = v48((v87(v19['holes']) for v19 in v18))
    v47(f'offer  holes offering both own and walked values: {v25}/{v26}')
    v20 &= v25 > 0
    v5['pair_joint_index'](v13)
    v27 = 0
    for v19 in v18:
        v68, v69, v70, v71, v72 = v5['pair_rivals'](v13, v19)
        v49 = v73((v66['truth'] for v66 in v19['holes']))
        v55, v63 = v19['holes']
        v50 = v48((1 for v93 in v97(v13['line']) if v93 != v19['line'] and {(v13['straddr'][v90], v13['tape'].v11[v90]) for v90 in v106(v87(v13['straddr'])) if v13['line'][v90] == v93} >= {(v55['address'], v49[0]), (v63['address'], v49[1])}))
        if v50 == 0 and v69 == v49:
            v27 += 1
        if (v50 > 0) != v88(v70):
            v27 += 1
    v47(f"joint  rival reading the hidden line, or miscounting it: {v27} -> {('OK' if not v27 else 'BROKEN')}")
    v20 &= not v27
    v28 = 0
    for v19 in v18:
        v26 = {v66['slot'] for v66 in v19['holes']}
        for v74, v75 in v19['_pair_ev'].v15():
            if v26 & v97(v75) or v97(v75) & v97(v19['slots']):
                v28 += 1
        if not v78(v19['_pair_b'], v2) or v19['_pair_b'] < 0:
            v28 += 1
    v47(f"evid   evidence rows leaking a hidden or evidence slot: {v28} -> {('OK' if not v28 else 'BROKEN')}")
    v20 &= not v28
    v29 = v18[0]
    v47(f"\nexample  line {v29['line']}: [{v29['holes'][0]['address']}] = {v29['holes'][0]['truth']} + [{v29['holes'][1]['address']}] = {v29['holes'][1]['truth']}")
    v47(f"         offers {v29['holes'][0]['offer']} x {v29['holes'][1]['offer']}, world {v87(v29['slots'])} rows")
    v47('\nPAIR OK' if v20 else '\nPAIR BROKEN')
    return 0 if v20 else 1
if v30 == '__main__':
    raise v51(v76())