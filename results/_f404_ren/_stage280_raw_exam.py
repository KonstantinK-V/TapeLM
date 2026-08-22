"""
Stage 280 — The whole thing, on text nobody wrote for it.

Every controller stage from 271 to 278 ran on a tape built from WITNESS_TMPL. The templates were
never the point; they existed because a controlled disagreement was needed and raw text does not
hand one over on request. 279 removed that need: writing as a WRITE / CONFIRM / DISPUTE decision
against an address produces witnesses and a support count from the corpus itself.

So this joins the two halves and asks the question GOAL.md left open. Everything that survived
its own measurement is here and nothing else:

    the tape        279's write decision, fp addressing with a votes check, common-noun anchors
                    excluded, values filtered - no template anywhere
    the families    NOT declared. An address with one value is clean, an address whose leader
                    beats the runner-up is decidable, an address where the top two tie is a tie.
                    Which questions exist is a property of the corpus.
    retrieval       word votes with classic idf, and where the votes go silent the fp address
                    proposes and a read verifies - the one job 277 left the ink
    the mind        278's policy: value baseline, BC anchor kept on through RL, the exhaustive
                    teacher, margin counted in votes, wrong -1.0 against abstain +0.75
    the exam        a disjoint slice of the corpus, so the tape at evaluation was never trained on

Two things will be worse here than on templates, and both are the point of running it.

Retrieval will fall silent. 264 measured that 49% of open-bank queries score zero on the gold,
and templates hid that because every witness shared the subject. --hop fp is the arm that says
whether the address can recover what the words missed.

And the teacher may simply be wrong. On manufactured disagreement the majority IS the truth by
construction; on wikitext a majority can be a popular error, and three sentences that mention
one name are often not three claims about one fact. G_teacher_ceiling therefore stops being a
pass/fail gate and becomes a measurement: teacher_acc_all on the corpus verdict is the ceiling
this exam actually has, and the policy is scored against that rather than against 1.0.

  python _stage280_raw_exam.py --smoke
  python _stage280_raw_exam.py --bc-episodes 4000 --rl-episodes 3000 --hop fp
  python _stage280_raw_exam.py --bc-episodes 4000 --rl-episodes 3000 --hop fp --no-hidden
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
import _stage274_truthfree_oracle as s274
import _stage278_value_baseline as s278
import _stage279_write_decision as s279
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _inprint_glue import TapeView
from _tape_index import context_words
v0 = v10('results')
v1 = v10('checkpoints/stage191_p1_curve.pt')
v2 = v10('checkpoints/stage253_joint_l02.pt')
v3 = v10('checkpoints/stage280_raw_exam.pt')
v4 = v10('data/_wikitext103_train.txt')
v5 = 280
v6 = ('clean', 'decidable', 'tie')
v7 = v0 / '_stage280_log.txt'

def log(v11: v134) -> None:
    v12 = v11 if v11.v253('\n') else v11 + '\n'
    try:
        v254(v12, end='', flush=True)
    except v135:
        v254(v12.v394('ascii', 'replace').v367('ascii'), end='', flush=True)
    v7.v255.v136(parents=True, exist_ok=True)
    with v7.v256('a', encoding='utf-8') as v137:
        v137.v257(v12)

def pack_from_corpus(v13, *, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24='classic', v25=0, v26=None, v27=0, v28='set', v29=None, v30=True):
    """Write the corpus with 279's decision, then read the result as an exam.

    Nothing here declares a family or a truth. An address carries whatever the corpus said at it,
    the leader is the corpus verdict, and a tie means the corpus never settled - which is exactly
    the abstention case 278 learned, arrived at from data rather than from a template.
    """
    if v29 is not None:
        v33 = v151(v29)
    else:
        v113 = v262.v258(v13)
        v33, v78 = v262.v259(v13, v18, v19, v20, 'anchor_rel', common=v113)
    if v26 is not None and v29 is None:
        v33 = [v32 for v32 in v33 if (v32['address'].v327('|', 1) + [''])[1] in v26]
    if v260(v21) and v29 is None:
        v21 = v21(v33, v14, v22, v20, v28)
    if v30:
        v33, v40 = v262.v261(v33, v14, v21, v22, v20, addr_key=v28)
    else:
        v138 = {}
        for v32 in v33:
            v138.v326(v32['address'], 0)
            v32['straddr'] = v32['address']
        v40 = v151(v138)
    v31 = v262.v139(v14, v23)
    for v32 in v33:
        v31.v263(v32['address'], v32['value'], v32['source'])
    v140, v141, v142, v143, v144, v145 = ([], [], [], [], [], v150(v151))
    v34 = []
    v146, v147 = ([], [])
    for v32 in v33:
        v34.v264(v32.v266('straddr', v32['address']))
        v146.v264(v32.v266('pos', -1))
        v147.v264(v32.v266('line', -1))
        v148 = v32['address'].v327(':', 1)[-1].v327('|')[0]
        v85 = v14.v265(v32['ctx'], exclude=v32['value'])
        v43 = v14.v328([v148])[0]
        v140.v264(v368.v329(v43 + v85 if v85 is not None else v43, dim=-1))
        v141.v264(v368.v329(v43, dim=-1))
        v142.v264(v368.v329(v85, dim=-1) if v85 is not None else v368.v329(v43, dim=-1))
        v145[v32['address']].v264(v156(v143))
        v143.v264(v32['value'])
        v144.v264(v32['ctx'])
    v35: v149[v134, v151[v9]] = v150(v151)
    v36: v149[v134, v151[v9]] = v150(v151)
    for v152, v153 in v154(v144):
        for v155 in v187(v153, exclude=v143[v152]):
            v35[v155].v264(v152)
        for v155 in v187(v153):
            v36[v155].v264(v152)
    v37 = v156(v143)
    if v24 == 'classic':
        v157 = {v155: v369.v218(v177(2.0, v37 / v177(1, v156(v35[v155])))) for v155 in v35}
        v158 = {v155: v369.v218(v177(2.0, v37 / v177(1, v156(v36[v155])))) for v155 in v36}
    else:
        v157 = {v155: 1.0 / v369.v218(2.0 + v156(v35[v155])) for v155 in v35}
        v158 = {v155: 1.0 / v369.v218(2.0 + v156(v36[v155])) for v155 in v36}
    v38 = []
    for v39 in v40:
        v159 = v145.v266(v39, [])
        if not v159:
            continue
        v160 = v233((v143[v184] for v184 in v159))
        v161 = v160.v267(2)
        v162 = v161[0][1]
        v163 = v161[1][1] if v156(v161) > 1 else 0
        if v156(v160) == 1:
            v330, v331 = ('clean', v161[0][0])
        elif v162 == v163:
            v330, v331 = ('tie', None)
        else:
            v330, v331 = ('decidable', v161[0][0])
        v164 = v39.v327(':', 1)[-1]
        v148, v268 = (v164.v327('|', 1) + [''])[:2]
        v165 = (v148 + ' ' + v268).v269()
        v38.v264({'S': v148, 'query': v165, 'truth': v331, 'slots': v159, 'kind': v330, 'address': v39})
    v18.v166(v38)
    if v27:
        v167 = v150(v151)
        for v168 in v38:
            v167[v168['kind']].v264(v168)
        v270, v271 = ([], [])
        for v137 in ('clean', 'decidable', 'tie'):
            v270 += v167[v137][:v27]
            v271 += v167[v137][v27:]
        v38 = v270 + v271
    if v25:
        v38 = v38[:v25]
    v169, v170, v171 = ([], [], [0] * v156(v143))
    for v172, v39 in v154(v40):
        v159 = v145.v266(v39, [])
        if v159:
            v169.v264(v368.v329(v281.v357([v140[v184] for v184 in v159]).v342(0), dim=-1))
            v170.v264(v159)
            for v184 in v159:
                v171[v184] = v156(v170) - 1
    return {'tape': v272(v281.v357(v140, 0).v224(v17), v143, v15, v16), 'texts': v144, 'items': v38, 'postings': v35, 'idf': v157, 'straddr': v34, 'pos': v146, 'line': v147, 'postings_probe': v36, 'idf_probe': v158, 'texts_lc': [v153.v204() for v153 in v144], 'addr_keys': v368.v329(v281.v357(v169).v8(), dim=-1).v224(v17) if v169 else None, 'addr_slots': v170, 'slot_addr': v171, 'addr_key': v28, 'slot_keys': v368.v329(v281.v357([v140[v184] for v159 in v170 for v184 in v159]).v8(), dim=-1).v224(v17) if v170 else None, 'slot_keys_slot': [v184 for v159 in v170 for v184 in v159], 'anc_keys': v281.v357([v141[v184] for v159 in v170 for v184 in v159]).v8().v224(v17) if v170 else None, 'ctx_keys': v281.v357([v142[v184] for v159 in v170 for v184 in v159]).v8().v224(v17) if v170 else None, 'bank': v14, 'write_actions': v149(v31.v273), 'n_addresses': v156(v40), 'n_slots': v37}

def retrieve(v41, v42, v43, v44: v134, v45=None, v46=False, v47=0.0, v48: v8=0.0, v49: v134='main'):
    """Votes decide. The address proposes, and only where the votes had little to say.

    The first trigger was "no candidates", which never fired: votes always return k slots, they
    are simply the wrong ones. Silence on raw text is not an empty list, it is a weak one - so
    the hop fires on a low top score or a short list, and its slots are APPENDED behind the
    votes rather than replacing them. The exact channel keeps the front of the ranking.
    """
    if v49 == 'main':
        v175, v176 = (v41['postings'], v41['idf'])
    else:
        if 'postings_probe' not in v41:
            raise v332('pack has no probe index: _stage280_raw_exam.py is out of date')
        v175, v176 = (v41['postings_probe'], v41['idf_probe'])
    v72, v173 = v274.v174(v42, v175, v176, v43)
    v50 = v177((v173.v266(v85, 0.0) for v85 in v72), default=0.0)
    v51 = False
    if v44 == 'fp' and v41['addr_keys'] is not None and v42 and (v156(v72) < v43 or v50 < v47):
        v178 = v41['bank'].v328(v42).v8()
        v178 = v368.v329(v368.v329(v178, dim=-1).v342(0), dim=-1).v224(v41['addr_keys'].v17)
        v98 = v41.v266('addr_key', 'mean')
        if v98 == 'two' and v41.v266('anc_keys') is not None:
            v275 = v9(v281.v395(v41['anc_keys'] @ v178, v41['ctx_keys'] @ v178).v370())
            v181 = v41['slot_addr'][v41['slot_keys_slot'][v275]]
        elif v98 == 'set' and v41.v266('slot_keys') is not None:
            v275 = v9((v41['slot_keys'] @ v178).v370())
            v181 = v41['slot_addr'][v41['slot_keys_slot'][v275]]
        else:
            v181 = v9((v41['addr_keys'] @ v178).v370())
        v179 = [v85 for v85 in v41['addr_slots'][v181] if v85 not in v72]
        if v179:
            v276 = v203((v173.v266(v85, 0.0) for v85 in v72), default=1.0)
            for v85 in v179:
                v173[v85] = v276 * 0.5
            v72 = (v72 + v179)[:v43]
            v51 = True
    if v46 and v45 is not None and v72:
        v180 = [v85 for v85 in v72 if v45['S'] in v41['texts_lc'][v85]]
        v72 = v180 or v72
    if v48 > 0.0 and v72:
        v181 = v177((v173.v266(v85, 0.0) for v85 in v72))
        if v181 > 0:
            v72 = [v85 for v85 in v72 if v173.v266(v85, 0.0) >= v48 * v181] or v72[:1]
    return (v72, {v85: v173.v266(v85, 0.0) for v85 in v72}, v51)

def rollout(v52, v53, v54, v15, v41, v45, v16, v17, *, v43, v55, v56, v57, v58, v59, v46, v44, v47=0.0, v48=0.0, v60=False, v61=True, v62=False, v63=0.0, v64=False, v65=None):
    """278's rollout with one line changed: retrieval may take the ink's hop when words fail."""
    v66 = v65 or v209.v182
    v67 = v41['tape']
    v183.v68 = {v184: v185 for v184, v185 in v154(v67.v333)}
    v69 = v274.v277.v186(S=v45.v266('query') or v45['S'])
    v70 = v187(v69)
    v71 = v69
    v72: v151[v9] = []
    v73: v151[v134] = []
    v74: v188[v9] = v188()
    v75: v151[v134] = []
    v189, v190, v191, v192 = ([], [], [], [])
    v193, v80, v194, v195 = (0, None, False, True)
    v196, v197, v198 = (v8('nan'), v8('nan'), 0)
    v199, v200 = (None, 0)
    v76 = v188(v45['slots'])
    v77 = v149(max_steps=v55, max_reads=v56, k=v43)
    for v78 in v201(v55):
        if v62:
            v32 = v66(cands=v72, seen_reads=v74, opened_values=v75, n_reads=v193, cand_scores=v41.v266('_sc'), **v77)
        else:
            v278 = v209.v334(v52, v53, v54, v15, v41, v71, v72, v74, v75, v73, v193, v16, v17, v43, v55)
            if v278 is None:
                break
            v335, v78 = v278
            if v60:
                v32 = v66(cands=v72, seen_reads=v74, opened_values=v75, n_reads=v193, cand_scores=v41.v266('_sc'), **v77)
                if not v281.v396(v335[v32]) or v335[v32] < -100000000.0:
                    break
                v189.v264(v368.v385(v335.v397(0), v281.v398([v32], device=v17)))
            else:
                v336 = v281.v386.v371(logits=v335)
                v32 = v9(v335.v370()) if v61 else v9(v336.v399())
                v190.v264(v336.v387(v281.v398(v32, device=v17)))
                v191.v264(v336.v388())
                if v63 > 0.0:
                    v372 = v66(cands=v72, seen_reads=v74, opened_values=v75, n_reads=v193, cand_scores=v41.v266('_sc'), **v77)
                    if v281.v396(v335[v372]) and v335[v372] > -100000000.0:
                        v189.v264(v368.v385(v335.v397(0), v281.v398([v372], device=v17)))
        v192.v264(v274.v373(v43)[v32])
        if v32 in (v274.v337, v274.v338):
            v42 = v70 if v32 == v274.v337 else v73
            v72, v173, v339 = v340(v41, v42, v43, v44, v45, v46, v47, v48)
            v198 += v9(v339)
            if v199 is None:
                v199 = v339 or not v72
            v41['_sc'] = v173
            if v72 and v369.v374(v196):
                v341 = v359((1 for v85 in v72 if v85 in v76))
                v196 = v341 / v156(v72)
                v197 = v341 / v177(1, v156(v76))
                v200 = v156(v72)
        elif v32 == 2 + 2 * v43:
            v194, v195 = (True, False)
            break
        elif v32 < 2 + v43:
            v184 = v32 - 2
            if v184 >= v156(v72):
                break
            v375 = v72[v184]
            v71 = (v71 + ' | ' + v41['texts'][v375])[-2000:]
            v73 = v187(v41['texts'][v375], exclude=v67.v333[v375])
            v74.v389(v375)
            v75.v264(v67.v333[v375])
            v193 += 1
        else:
            v184 = v32 - 2 - v43
            if v184 >= v156(v72):
                break
            v80 = v67.v333[v72[v184]]
            v195 = False
            break
    if v194 or v80 is None:
        v202, v194 = (0, True)
        v79 = 0.0 if v195 else v59
    else:
        v202 = v9(v45['truth'] is not None and v80 == v45['truth'])
        v79 = 1.0 if v202 else -v58
    v79 -= v57 * v193
    return {'loss': v281.v357(v189).v342() if v189 else v281.v343((), device=v17), 'logps': v190, 'entropy': v191, 'reward': v79, 'correct': v202, 'abstained': v194, 'n_reads': v193, 'trace': v192, 'kind': v45['kind'], 'answer_is_slot': v80 is None or v80 in v188(v67.v333), 'retrieval_precision': v196, 'witness_recall': v197, 'hops': v198, 'words_silent': v279(v199), 'n_cands': v200, 'stalled': v279(v195 and v194), 'return_path': v344(v41, v45, v80) if v64 else v8('nan')}

def return_path(v41, v45, v80) -> v8:
    """Does the tape lead from the answer back to the subject? An observer, never a vote.

    282 gave the policy this as an action and it did not pay for its place: on the same tape the
    mind that never probes scores 0.704 held-out against 0.594 for the mind that does. But the
    question the probe asks outlives the exam that rejected it. Today an answer is a slot index,
    so G_answer_is_slot proves the answer came off the tape; when the answer stops being an
    index there is no such proof, and the only label-free replacement is this - ask the tape
    about the value and see whether some OTHER mention carries the subject and the value
    together. So it is measured here on every answer, costs the policy nothing, and is reported
    beside accuracy, where a reader can see whether it separates right answers from wrong ones
    before anything is built on it.
    """
    if v80 is None or 'postings_probe' not in v41:
        return v8('nan')
    v42 = v187(v80) or [v80]
    v81 = v41['postings_probe']
    v82 = v203((v81.v266(v155, ()) for v155 in v42), key=v156, default=())
    v83 = v80.v204()
    v84 = 0
    for v85 in v82:
        if v45['S'] in v41['texts_lc'][v85] and v83 in v41['texts_lc'][v85]:
            v84 += 1
            if v84 >= 2:
                return 1.0
    return 0.0

def main() -> v9:
    v86 = v280.v205()
    v86.v206('--smoke', action='store_true')
    v86.v206('--bc-episodes', type=v9, default=0)
    v86.v206('--rl-episodes', type=v9, default=0)
    v86.v206('--tape-period', type=v9, default=0)
    v86.v206('--addresses', type=v9, default=0)
    v86.v206('--min-mentions', type=v9, default=3)
    v86.v206('--max-items', type=v9, default=0)
    v86.v206('--min-per-family', type=v9, default=8, help='put this many of each family at the front of the exam. Noun-phrase anchors made addresses specific enough that exact ties are rare - the held-out tape had two - and abstention cannot be measured on two items. 0 leaves the natural frequency.')
    v86.v206('--address-tau', type=v8, default=0.9)
    v86.v206('--address-overlap', type=v9, default=2)
    v86.v206('--soft-match', type=v8, default=0.0)
    v86.v206('--addr-key', choices=('two', 'set', 'mean'), default='set', help='how a query is matched to an address. two scores identity and situation separately and takes the minimum; set is MaxSim over the member keys; mean is the averaged key every number before 283 came from. new_pack reads this - without the flag the run died in argparse.')
    v86.v206('--run-tag', type=v134, default='', help='prefix for the log, checkpoint and decision filenames, so two arms do not overwrite each other')
    v86.v206('--hop', choices=('none', 'fp'), default='fp')
    v86.v206('--k-gap', type=v8, default=0.35, metavar='F', help='drop retrieved slots scoring below F x the top score. A fixed k fills the list with foreign slots on raw text and the teacher aggregates them; 0 restores the fixed-k behaviour.')
    v86.v206('--hop-min', type=v8, default=1.0, help='fire the address hop when the best vote score falls below this. The first version fired on an empty candidate list, which never happens - votes always return k slots, just the wrong ones.')
    v86.v206('--topk', type=v9, default=7)
    v86.v206('--max-steps', type=v9, default=10)
    v86.v206('--max-reads', type=v9, default=7)
    v86.v206('--read-cost', type=v8, default=0.02)
    v86.v206('--wrong-cost', type=v8, default=1.0)
    v86.v206('--abstain-reward', type=v8, default=0.75)
    v86.v206('--entropy-bonus', type=v8, default=0.01)
    v86.v206('--lr-policy', type=v8, default=0.001)
    v86.v206('--lr-value', type=v8, default=0.003)
    v86.v206('--lr-upper', type=v8, default=3e-05)
    v86.v206('--value-coef', type=v8, default=0.5)
    v86.v206('--bc-anchor', type=v8, default=0.5)
    v86.v206('--no-hidden', action='store_true')
    v86.v206('--no-value-head', action='store_true')
    v86.v206('--subject-filter', choices=('off', 'on'), default='on', help="drop retrieved slots that do not mention the asked subject. Computable from the cue, and on raw text the teacher's majority rule is unsound without it - the smoke read two unrelated slots for every relevant one and the teacher stopped abstaining entirely.")
    v86.v206('--frozen-trunk', action='store_true')
    v87 = v86.v207()
    v88 = v87.v46 == 'on'
    v89 = not v87.v208
    v209.v90 = v87.v91
    global LOG_PATH
    v92 = f'_{v87.v44}' + ('_nohid' if v87.v91 else '') + (f'_{v87.v251}' if v87.v251 else '')
    v7 = v0 / f'_stage280_log{v92}.txt'
    v7.v255.v136(parents=True, exist_ok=True)
    v7.v210('', encoding='utf-8')
    v17 = v281.v17('cuda' if v281.v376.v345() else 'cpu')
    v18 = v282.v211(v5)
    v281.v212(v5)
    v93 = v213.v213()
    v94 = v87.v214 or (400 if v87.v250 else 4000)
    v95 = v177(0, v87.v215)
    v96 = v87.v96 or (50 if v87.v250 else 200)
    v19 = v87.v216 or (60 if v87.v250 else 400)
    v43 = v87.v97
    v98 = 'none' if v87.v217 else 'upper'
    v218(f'Stage280 raw exam start {v392.v383(v393.v384).v323()} device={v17} hop={v87.v44} no_hidden={v87.v91} bc={v94} rl={v95} k={v43} mode={v98}')
    v78, v78, v219, v220 = v221()
    v15 = v283.v222(v134(v346.v284))
    v99 = v15.v223()
    v16 = v15.v285(v286) or 0
    v54 = v377.v347(v15, v219, v16, v99).v224(v17)
    v100 = v2 if v2.v287() else v1
    v53 = v348(v220, v99).v224(v17)
    v53.v225(v281.v349(v100, map_location=v17, weights_only=False)['model'])
    if v98 == 'none':
        v53.v227()
        for v103 in v53.v228():
            v103.v289(False)
    else:
        v350.v288(v53, v98)
    v101 = v274.v226(v53)
    v102 = v348(v220, v99).v224(v17)
    v102.v225(v281.v349(v1, map_location=v17, weights_only=False)['model'])
    v102.v227()
    for v103 in v102.v228():
        v103.v289(False)
    v14 = v229(v102, v219, v17)
    with v4.v256('r', encoding='utf-8', errors='ignore') as v137:
        v230 = v137.v290(4000000 if v87.v250 else 30000000)
    v104 = [v291.v269() for v291 in v230.v327('\n') if 80 <= v156(v291.v269()) <= 400]
    v105 = v9(0.7 * v156(v104))
    v106 = v104[:v105][:3000 if v87.v250 else 25000]
    v107 = v104[v105:][:1500 if v87.v250 else 12000]
    v218(f'  corpus split: {v156(v106)} train lines / {v156(v107)} held-out lines')

    def new_pack(v231, v13):
        return v292(v13, bank=v14, tok=v15, pad_id=v16, device=v17, rng=v231, n_addr=v19, min_mentions=v87.v20, tau=v87.v318, overlap=v87.v319, soft_match=v87.v23, max_items=v87.v25, min_per_family=v87.v27, addr_key=v87.v28)
    v41 = v232(v18, v106)
    v108 = v233((v184['kind'] for v184 in v41['items']))
    v218(f"  tape: {v41['n_addresses']} addresses, {v41['n_slots']} slots, write {v365.v324(v41['write_actions'])} | items {v365.v324(v149(v108))} ({v213.v213() - v93:.0f}s)")
    if v156(v41['items']) < 8 or v108['tie'] == 0:
        v218('  corpus produced too few items or no natural tie; raise --addresses')
        return 1
    v109 = 0 if v87.v91 else 2 * (v53.v378.v351 // 2)
    v52 = v209.v293(v109 + v209.v352, v43, v17) if v89 else v183.v294(v109 + v209.v352, v43, v17)
    v110 = [v103 for v103 in v53.v228() if v103.v295]
    v111 = [{'params': [v103 for v390, v103 in v52.v391() if not v390.v400('v.')], 'lr': v87.v296}]
    if v89:
        v111.v264({'params': v151(v52.v185.v228()), 'lr': v87.v353})
    if v110:
        v111.v264({'params': v110, 'lr': v87.v354})
    v112 = v281.v297.v234(v111, weight_decay=0.01)
    v113 = v149(k=v43, max_steps=v87.v55, max_reads=v87.v56, read_cost=v87.v57, wrong_cost=v87.v58, abstain_reward=v87.v59, subject_filter=v88, hop=v87.v44, hop_min=v87.v47, k_gap=v87.v48)
    v235, v236, v237 = (0.0, [], [])
    v52.v238()
    v53.v238(v98 != 'none')
    for v114 in v201(1, v94 + 1):
        if (v114 - 1) % v96 == 0 and v114 > 1:
            v41 = v232(v18, v106)
        v45 = v41['items'][v18.v355(v156(v41['items']))]
        v132 = v298(v52, v53, v54, v15, v41, v45, v16, v17, bc=True, **v113)
        v112.v299(set_to_none=True)
        v132['loss'].v300()
        v281.v379.v356.v301(v151(v52.v228()) + v110, 1.0)
        v112.v302()
        if v114 % v177(1, v94 // 8) == 0:
            v236.v264({'phase': 'bc', 'episode': v114, 'loss': v8(v132['loss']), 'kind': v132['kind'], 'trace': v132['trace']})
            v218(f"  bc {v114}/{v94} loss={v8(v132['loss']):.4f} [{v132['kind']}] {v132['trace']}")
    for v114 in v201(1, v95 + 1):
        if (v114 - 1) % v96 == 0 and v114 > 1:
            v41 = v232(v18, v106)
        v45 = v41['items'][v18.v355(v156(v41['items']))]
        if v89:
            v52.v303 = []
        v132 = v298(v52, v53, v54, v15, v41, v45, v16, v17, greedy=False, bc_anchor=v87.v63, **v113)
        v143 = v52.v303 if v89 else None
        if v89:
            v52.v303 = None
        if not v132['logps']:
            continue
        v239 = v132['reward']
        if v89 and v143:
            v304 = v281.v357(v143[:v156(v132['logps'])])
            v305 = v368.v358(v304, v281.v380(v304, v239))
            v237.v264(v8(v305))
            v241 = -((v239 - v304).v402() * v281.v357(v132['logps'])).v359() + v87.v381 * v305
        else:
            v235 = 0.99 * v235 + 0.01 * v239
            v241 = -(v239 - v235) * v281.v357(v132['logps']).v359()
        v240 = v281.v357(v132['entropy']).v359() if v132['entropy'] else v281.v343((), device=v17)
        v241 = v241 - v87.v360 * v240
        if v87.v63 > 0.0 and v132['loss'].v295:
            v241 = v241 + v87.v63 * v132['loss']
        v112.v299(set_to_none=True)
        v241.v300()
        v281.v379.v356.v301(v151(v52.v228()) + v110, 1.0)
        v112.v302()
        if v114 % v177(1, v95 // 8) == 0:
            v236.v264({'phase': 'rl', 'episode': v114, 'v_mse': v8(v401.v342(v237[-200:])) if v237 else None, 'kind': v132['kind'], 'trace': v132['trace']})
            v218(f"  rl {v114}/{v95} v_mse={(v401.v342(v237[-200:]) if v237 else v8('nan')):.3f} [{v132['kind']}] {v132['trace']}")
    v52.v227()
    v53.v227()
    v115 = v274.v226(v53)

    @v281.v244()
    def evaluate(v103):
        v242 = {v137: v150(v151) for v137 in v6}
        v243 = {v137: v150(v151) for v137 in v6}
        v306, v307, v198, v308, v309, v310 = ([], [], [], [], [], [])
        v311, v312, v313 = ([], [], [])
        for v168 in v103['items']:
            v314 = v298(v52, v53, v54, v15, v103, v168, v16, v17, diag=True, **v113)
            v153 = v298(v52, v53, v54, v15, v103, v168, v16, v17, teacher_only=True, **v113)
            v137 = v168['kind']
            v242[v137]['correct'].v264(v314['correct'])
            v242[v137]['abstain'].v264(v9(v314['abstained']))
            v242[v137]['reads'].v264(v314['n_reads'])
            v242[v137]['reward'].v264(v314['reward'])
            if not v369.v374(v314['retrieval_precision']):
                v242[v137]['prec'].v264(v314['retrieval_precision'])
                v242[v137]['rec'].v264(v314['witness_recall'])
            v243[v137]['correct'].v264(v153['correct'])
            v243[v137]['abstain'].v264(v9(v153['abstained']))
            v243[v137]['reward'].v264(v153['reward'])
            v306.v264(v9(v314['answer_is_slot']))
            v198.v264(v314['hops'])
            v308.v264(v9(v314['words_silent']))
            v309.v264(v314['n_cands'])
            v310.v264(v9(v314['stalled']))
            if not v369.v374(v314['return_path']):
                v311.v264(v314['return_path'])
                (v312 if v314['correct'] else v313).v264(v314['return_path'])
            if v156(v307) < 24:
                v307.v264({'kind': v137, 'S': v168['S'], 'trace': v314['trace'], 'correct': v314['correct'], 'abstained': v314['abstained'], 'hops': v314['hops'], 'stalled': v314['stalled']})
        v11 = lambda v361: v8(v401.v342(v361)) if v361 else v8('nan')
        v132 = {'answer_is_slot': v11(v306), 'traces': v307, 'reward_total': v11([v231 for v137 in v6 for v231 in v242[v137]['reward']]), 'teacher_reward_total': v11([v231 for v137 in v6 for v231 in v243[v137]['reward']]), 'retrieval_precision': v11([v382 for v137 in v6 for v382 in v242[v137]['prec']]), 'witness_recall': v11([v382 for v137 in v6 for v382 in v242[v137]['rec']]), 'words_silent_rate': v11(v308), 'hops_per_episode': v11(v198), 'mean_candidates': v11(v309), 'stall_rate': v11(v310), 'return_path_rate': v11(v311), 'return_path_when_correct': v11(v312), 'return_path_when_wrong': v11(v313), 'n_items': v156(v103['items'])}
        v315, v316 = (0, 0)
        for v137 in v6:
            v317 = v359((1 for v32 in v242[v137]['abstain'] if not v32))
            v315 += v359(v242[v137]['correct'])
            v316 += v317
            v132[v137] = {'n': v156(v242[v137]['abstain']), 'coverage': 1.0 - v11(v242[v137]['abstain']), 'acc_answered': v359(v242[v137]['correct']) / v317 if v317 else v8('nan'), 'abstain': v11(v242[v137]['abstain']), 'mean_reads': v11(v242[v137]['reads']), 'reward': v11(v242[v137]['reward']), 'precision': v11(v242[v137]['prec']), 'recall': v11(v242[v137]['rec']), 'teacher_abstain': v11(v243[v137]['abstain']), 'teacher_acc_all': v11(v243[v137]['correct'])}
        v132['coverage_all'] = v316 / v177(1, v156(v103['items']))
        v132['acc_answered_all'] = v315 / v177(1, v316)
        return v132
    v116 = v245(v41)
    v117 = v232(v282.v211(v5 + 99), v107)
    v218(f"  held-out tape: {v117['n_addresses']} addresses, {v117['n_slots']} slots, items {v365.v324(v149(v233((v184['kind'] for v184 in v117['items']))))}")
    v118 = v245(v117)
    v218(f"  HELD-OUT {v365.v324({v320: v321 for v320, v321 in v118.v38() if v320 != 'traces'})}")
    v119 = v118['teacher_reward_total']
    v120 = v87.v59
    v121 = v119 >= 0.5 * v120
    v122 = v101 == v115
    v123 = v118['answer_is_slot'] >= 0.99
    v124 = v246((v118[v137]['n'] >= 4 for v137 in v6))
    v125 = v118['reward_total'] >= v119 - 0.1
    v126 = v118['clean']['abstain'] <= 0.25 and v118['decidable']['abstain'] <= 0.4
    v127 = v118['tie']['abstain'] >= 0.6
    v128 = v118['acc_answered_all'] >= 0.6
    v129 = v118['reward_total'] >= v116['reward_total'] - 0.15
    v130 = v87.v44 == 'none' or v118['words_silent_rate'] < 0.05 or v118['hops_per_episode'] > 0.0
    if not (v122 and v123 and v124):
        v247 = 'RAW_EXAM_INVALID'
    elif not v121:
        v247 = 'TEACHER_UNUSABLE_ON_RAW'
    elif v125 and v127 and v126 and v129:
        v247 = 'RAW_EXAM_OK'
    elif v125 and v129:
        v247 = 'RAW_EXAM_REACHES_CEILING'
    elif v127 or v126:
        v247 = 'RAW_EXAM_PARTIAL'
    else:
        v247 = 'RAW_EXAM_NO'
    v131 = v3.v248(f'{v3.v362}{v92}{v3.v363}')
    v281.v249({'policy': v52.v364(), 'model': v53.v364(), 'stage': 280, 'hop': v87.v44, 'no_hidden': v87.v91, 'min_per_family': v87.v27, 'arc_enc_hash': v115}, v131)
    v132 = {'stage': 280, 'overall': v247, 'hop': v87.v44, 'no_hidden': v87.v91, 'value_head': v89, 'subject_filter': v87.v46, 'smoke': v87.v250, 'seed': v5, 'bc_episodes': v94, 'rl_episodes': v95, 'topk': v43, 'min_per_family': v87.v27, 'run_tag': v87.v251, 'checkpoint': v134(v131), 'address': {'tau': v87.v318, 'overlap': v87.v319, 'soft_match': v87.v23, 'min_mentions': v87.v20}, 'reward': {'correct': 1.0, 'wrong': -v87.v58, 'abstain': v87.v59, 'read': -v87.v57}, 'train_tape_shape': {'addresses': v41['n_addresses'], 'slots': v41['n_slots'], 'write_actions': v41['write_actions'], 'families': v149(v108)}, 'held_out_tape_shape': {'addresses': v117['n_addresses'], 'slots': v117['n_slots'], 'write_actions': v117['write_actions'], 'families': v149(v233((v184['kind'] for v184 in v117['items'])))}, 'teacher_ceiling_reward': v119, 'gates': {'G_arc_enc_frozen': v122, 'G_answer_is_slot': v123, 'G_all_families_present': v124, 'G_teacher_usable': v121, 'G_reaches_teacher': v125, 'G_answers_when_decidable': v126, 'G_abstain_on_tie': v127, 'G_acc_when_answering': v128, 'G_generalises_to_held_out': v129, 'G_hop_covers_silence': v130}, 'train_tape': {v320: v321 for v320, v321 in v116.v38() if v320 != 'traces'}, 'held_out': v118, 'fp_version': v274.v322(), 'arc_enc_hash_before': v101, 'arc_enc_hash_after': v115, 'curve': v236, 'reference_278_templates': {'reward_total': 0.7875, 'teacher': 0.85, 'acc_answered_all': 1.0, 'tie_abstain': 1.0}, 'note': "Everything that survived its own measurement, on text nobody wrote for it. The tape is 279's write decision - fp addressing checked by shared words, common-noun anchors excluded - and the families are not declared: an address with one value is clean, a leader beating the runner-up is decidable, a tie at the top is a tie, and which questions exist is a property of the corpus. The mind is 278's policy with the value baseline, the BC anchor kept on through RL and the exhaustive teacher. Two things are expected to be worse than on templates and both are why this runs: words go silent where a template guaranteed the subject was shared, which --hop fp answers or fails to; and the teacher can simply be wrong, because on wikitext a majority may be a popular error, so teacher_reward_total is reported as the ceiling this exam has and the policy is scored against that rather than against 1.0. The held-out tape is built from a disjoint 30% of the corpus.", 'timestamp': v392.v383(v393.v384).v323(), 'wall_s': v213.v213() - v93}
    v0.v136(parents=True, exist_ok=True)
    (v0 / f'stage280_decision{v92}.json').v210(v365.v324(v132, indent=2), encoding='utf-8')
    (v0 / f'stage280_mini{v92}.md').v210(f"# Stage 280 the exam on raw text (hop {v87.v44}{(', no hidden' if v87.v91 else '')})\n\n**{v247}**{(' · SMOKE' if v87.v250 else '')} · teacher ceiling **{v119:.3f}**\n\n| family (held out) | n | coverage | acc answered | abstain | teacher acc | reads |\n|---|---:|---:|---:|---:|---:|---:|\n" + ''.v366((f"| {v137} | {v118[v137]['n']} | {v118[v137]['coverage']:.2f} | {v118[v137]['acc_answered']:.2f} | {v118[v137]['abstain']:.2f} | {v118[v137]['teacher_acc_all']:.2f} | {v118[v137]['mean_reads']:.1f} |\n" for v137 in v6)) + f"\n- policy {v118['reward_total']:.3f} vs teacher {v119:.3f}; coverage {v118['coverage_all']:.2f} at accuracy {v118['acc_answered_all']:.2f}\n- words silent on {v118['words_silent_rate']:.2f} of episodes, {v118['hops_per_episode']:.2f} ink hops per episode\n- retrieval precision {v118['retrieval_precision']:.2f}, recall {v118['witness_recall']:.2f}\n\n## Gates\n\n" + ''.v366((f'- {v320}: **{v321}**\n' for v320, v321 in v132['gates'].v38())), encoding='utf-8')
    v218(v365.v324({'overall': v247, 'gates': v132['gates']}, indent=2))
    return 0
if v133 == '__main__':
    raise v252(v325())