"""297/298: write the tape by frames - exact count, no tau.

A place is a hole whose left+right frame the corpus writes at least twice. Width is the maximum
that still repeats. Fillers are literal. No parser, no stop-words, no merge threshold.

Interpolation (cosine) does NOT live here: each frame is its own address. Frame-frame cosine is
handed to Phi as the cos edge at read time, so "is this the same place" is the mind's job.

    pack_from_frames(...) -> same schema as pack_from_corpus, plus:
      frame_mode, frame_nfill[slot], frame_nfill_max, frame_fps[slot]
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from _inprint_glue import TapeView
from _tape_index import context_words
v0 = v44.v3('^=+\\s*.*?\\s*=+$')
v1 = v44.v3('\\s*@-@\\s*')

def hygiene_text(v4: v2) -> v2:
    """Drop heading lines and BPE joint marks; collapse <unk>. Not a grammar."""
    v5 = []
    for v6 in v4.v45():
        v46 = v6.v90()
        if not v46 or v0.v112(v46):
            continue
        v46 = v1.v126('-', v46).v91('<unk>', ' ')
        if v46.v90():
            v5.v95(v46)
    return '\n'.v47(v5)

def tokenize(v4: v2) -> v7[v2]:
    return [v48 for v48 in v4.v113() if v48]

def _frame_key(v8, v9, v10) -> v2:
    return ' '.v47(v8[v9 - v10:v9]) + '\x00' + ' '.v47(v8[v9 + 1:v9 + 1 + v10])

def _repeating_at_width(v8, v10: v49) -> v14[v2]:
    """Keys whose frame appears at least twice. Hapaxes never enter a list - that was the OOM."""
    v11 = v50(v8)
    v12: v51[v2, v49] = {}
    v13: v14[v2] = v14()
    for v9 in v52(v10, v11 - v10):
        v53 = v92(v8, v9, v10)
        if v53 in v13:
            continue
        if v53 in v12:
            del v12[v53]
            v13.v114(v53)
        else:
            v12[v53] = v9
    return v13

def pack_from_frames(v15, *, v16, v17, v18, v19, v20, v21, v22, v23: v49=12, **v24):
    """Build a pack whose addresses ARE frames. Signature mirrors pack_from_corpus extras."""
    if v54(v15, v2):
        v8 = v93(v115(v15))
    else:
        v8 = []
        for v6 in v15:
            v48 = v115(v6)
            if v48:
                v8.v127(v93(v48))
    v11 = v50(v8)
    if v11 < 3:
        return v94(v16, v17, v18, v19)
    v25: v7[v14[v2]] = []
    for v10 in v52(1, v23 + 1):
        v25.v95(v116(v8, v10))
    v26: v51[v2, v7[v49]] = v55(v7)
    v27 = {}
    for v9 in v52(v11):
        v56 = None
        for v10 in v52(v117(v23, v9, v11 - v9 - 1), 0, -1):
            v53 = v92(v8, v9, v10)
            if v53 in v25[v10 - 1]:
                v41 = v128(v8[v9 - v10:v9])
                v42 = v128(v8[v9 + 1:v9 + 1 + v10])
                v56 = (v10, v41, v42)
                break
        if v56 is None:
            continue
        v10, v41, v42 = v56
        v32 = v96(v10, v41, v42)
        v26[v32].v95(v9)
        v27[v32] = (v10, v41, v42)
    del v25
    v28 = [v57 for v57, v58 in v26.v39() if v50(v58) >= v22]

    def _rank(v57: v2):
        v58 = v26[v57]
        v53 = v50({v8[v9] for v9 in v58})
        v59 = v50(v58)
        v60 = 0 if 2 <= v53 <= 20 else 1 if v53 == 1 else 2
        return (v60, 0 if v59 <= 128 else 1, -v117(v59, 64), v57)
    v28.v61(key=v97)
    v28 = v28[:v109(1, v21)]
    v26 = {v57: v26[v57] for v57 in v28}
    if not v28:
        return v94(v16, v17, v18, v19)
    v29 = v49(v24.v98('frame_row_cap', 48))
    v62, v63, v64, v65, v66, v67 = ([], [], [], [], [], [])
    v68, v69 = ([], [])
    v30 = {}
    v31 = []
    for v32 in v28:
        v10, v41, v42 = v27[v32]
        v70 = v26[v32]
        v71 = v50({v8[v9] for v9 in v70})
        if v50(v70) > v29:
            v99 = v50(v70) / v29
            v70 = [v70[v49(v129 * v99)] for v129 in v52(v29)]
        v72 = v16.v130(v7(v41)).v100()
        v73 = v16.v130(v7(v42)).v100()
        v74 = v118.v101(v118.v101(v72, dim=-1).v131(0) + v118.v101(v73, dim=-1).v131(0), dim=-1)
        for v9 in v70:
            v102 = v8[v9]
            v103 = ' '.v47(v41) + ' ___ ' + ' '.v47(v42)
            v104 = v16.v119(v103, exclude=v102)
            v53 = v74
            v62.v95(v53)
            v63.v95(v74)
            v64.v95(v118.v101(v104, dim=-1) if v104 is not None else v74)
            v65.v95(v102)
            v66.v95(v103)
            v67.v95(v32)
            v68.v95(v71)
            v69.v95(v74)
            v30[v9] = v50(v65) - 1
            v31.v95(v50(v65) - 1)
    v33 = {v57: [v30[v9] for v9 in v26[v57] if v9 in v30] for v57 in v28}
    v34: v51[v2, v7[v49]] = v55(v7)
    v35: v51[v2, v7[v49]] = v55(v7)
    for v75, v48 in v76(v66):
        for v10 in v105(v48, exclude=v65[v75]):
            v34[v10].v95(v75)
        for v10 in v105(v48):
            v35[v10].v95(v75)
    v36 = v50(v65)
    v37 = {v10: v120.v106(v109(2.0, v36 / v109(1, v50(v34[v10])))) for v10 in v34}
    v38 = {v10: v120.v106(v109(2.0, v36 / v109(1, v50(v35[v10])))) for v10 in v35}
    v39 = []
    for v32 in v28:
        v77 = v33.v98(v32, [])
        if v50(v77) < 2:
            continue
        v78 = v107((v65[v9] for v9 in v77))
        v79 = v78.v108(2)
        v80 = v79[0][1]
        v81 = v79[1][1] if v50(v79) > 1 else 0
        if v50(v78) == 1:
            v121, v122 = ('clean', v79[0][0])
        elif v80 == v81:
            v121, v122 = ('tie', None)
        else:
            v121, v122 = ('decidable', v79[0][0])
        v10, v41, v42 = v27[v32]
        v82 = v32
        v83 = ' '.v47(v41) + ' ' + ' '.v47(v42)
        v39.v95({'S': v82, 'query': v83, 'truth': v122, 'slots': v77, 'kind': v121, 'address': v32, 'frame_w': v10, 'frame_nfill': v50(v78)})
    v20.v84(v39)
    v85, v86, v87 = ([], [], [0] * v50(v65))
    for v88, v32 in v76(v28):
        v77 = v33.v98(v32, [])
        if v77:
            v85.v95(v118.v101(v111.v132([v62[v9] for v9 in v77]).v131(0), dim=-1))
            v86.v95(v77)
            for v9 in v77:
                v87[v9] = v50(v86) - 1
    v40 = v109(v68) if v68 else 1
    return {'tape': v110(v111.v132(v62, 0).v123(v19), v65, v17, v18), 'texts': v66, 'items': v39, 'postings': v34, 'idf': v37, 'straddr': v67, 'postings_probe': v35, 'idf_probe': v38, 'texts_lc': [v48.v124() for v48 in v66], 'addr_keys': v118.v101(v111.v132(v85).v100(), dim=-1).v123(v19) if v85 else None, 'addr_slots': v86, 'slot_addr': v87, 'addr_key': 'set', 'slot_keys': v118.v101(v111.v132([v62[v9] for v77 in v86 for v9 in v77]).v100(), dim=-1).v123(v19) if v86 else None, 'slot_keys_slot': [v9 for v77 in v86 for v9 in v77], 'anc_keys': v111.v132([v63[v9] for v77 in v86 for v9 in v77]).v100().v123(v19) if v86 else None, 'ctx_keys': v111.v132([v64[v9] for v77 in v86 for v9 in v77]).v100().v123(v19) if v86 else None, 'bank': v16, 'write_actions': {'FRAME': v36, 'CONFIRM': v125((1 for v104 in v107(v67).v133() if v104 > 1))}, 'n_addresses': v50(v28), 'n_slots': v36, 'frame_mode': True, 'frame_nfill': v68, 'frame_nfill_max': v40, 'frame_fps': v69}

def _addr_key(v10, v41, v42) -> v2:
    return f"f{v10}:{' '.v47(v41)}|{' '.v47(v42)}"

def _empty_pack(v16, v17, v18, v19):
    v43 = v111.v89(1, 32, device=v19)
    return {'tape': v110(v43, ['?'], v17, v18), 'texts': ['?'], 'items': [], 'postings': {}, 'idf': {}, 'straddr': ['?'], 'postings_probe': {}, 'idf_probe': {}, 'texts_lc': ['?'], 'addr_keys': None, 'addr_slots': [], 'slot_addr': [0], 'addr_key': 'set', 'slot_keys': None, 'slot_keys_slot': [], 'anc_keys': None, 'ctx_keys': None, 'bank': v16, 'write_actions': {}, 'n_addresses': 0, 'n_slots': 0, 'frame_mode': True, 'frame_nfill': [1], 'frame_nfill_max': 1, 'frame_fps': [v43[0]]}