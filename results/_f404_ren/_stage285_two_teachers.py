"""
Stage 285 — Two teachers, so that imitation stops being an answer.

The loop up to here was circular and worth saying plainly: a heuristic that does not understand
the question is called the teacher, the mind is trained to resemble it, and success is measured
as resemblance. G_reaches_teacher says so literally. The one thing that was never imitation is
the reward - it comes from the tape's own verdict, not from the teacher - which is why the
policy ends up above its teacher at all, 0.704 against 0.625 on the baseline.

This stage removes the escape route. There are two judges now, both reading the same tape,
neither reading a label:

  VOTES counts witnesses. The value the address says most often wins; equal support abstains.
  This is 278's teacher, unchanged.

  RETURN counts corroboration. A value wins by how many OTHER mentions carry it together with
  the subject, and one is not enough. It never looks at how often the address said something.

Where they agree there is a demonstration and BC uses it. Where they disagree there is no
target at all - two different actions cannot both be copied - and the episode is left to the
reward. So the contested items are the ones no amount of resemblance can solve.

The bar follows from that, and it is not "beat two teachers". Both are readings of one tape and
the tape is the truth here, so where they differ at least one is misreading it. The mind's job
is to be right more often than either reading alone:

  G_arbitration: accuracy on contested items above BOTH teachers' accuracy on the same items.

Everything else - the tape, the retrieval, the reward, the value baseline - is the 280 baseline
untouched, so a difference here is a difference from having two judges.

  python _stage285_two_teachers.py --smoke
  python _stage285_two_teachers.py --bc-episodes 4000 --rl-episodes 3000 --min-mentions 2
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
import _stage280_raw_exam as s280
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words
v0 = v8('results')
v1 = v8('checkpoints/stage191_p1_curve.pt')
v2 = v8('checkpoints/stage253_joint_l02.pt')
v3 = v8('data/_wikitext103_train.txt')
v4 = 285
v5 = v9.v5
v6 = v0 / '_stage285_log.txt'

def log(v10: v69) -> None:
    v11 = v10 if v10.v147('\n') else v10 + '\n'
    try:
        v148(v11, end='', flush=True)
    except v70:
        v148(v11.v253('ascii', 'replace').v231('ascii'), end='', flush=True)
    v6.v149.v71(parents=True, exist_ok=True)
    with v6.v150('a', encoding='utf-8') as v72:
        v72.v151(v11)

def support(v12, v13: v69, v14: v69) -> v7:
    """How many mentions carry the subject and this value together. No ranking, no labels."""
    v15 = v152(v14) or [v14]
    v16 = v73((v12['postings_probe'].v205(v206, ()) for v206 in v15), key=v153, default=())
    v17 = v14.v74()
    return v75((1 for v207 in v16 if v13 in v12['texts_lc'][v207] and v17 in v12['texts_lc'][v207]))

def return_teacher(v12, v18):
    """Decides by corroboration and never counts votes.

    This is deliberately blind to the thing the other judge is made of. It cannot see that an
    address said something four times; it can only see how many independent mentions tie a value
    to the subject. Two judges that share a criterion would agree everywhere and there would be
    nothing to arbitrate.
    """

    def teach(*, v76, v77, v78, v79, v80, v81, v82, v40):
        if not v76:
            return v171.v154
        v83 = [v155 for v155, v207 in v232(v76[:v40]) if v207 not in v77]
        if v83 and v79 < v82 and (v79 + 3 <= v81):
            return 2 + v83[0]
        if not v78:
            return 2 + 2 * v40
        v84 = v156(((v17, v245(v12, v18['S'], v17)) for v17 in v246(v78)), key=lambda v247: -v247[1])
        v157, v158 = v84[0]
        v85 = v84[1][1] if v153(v84) > 1 else 0
        if v158 < 2 or v158 == v85:
            return 2 + 2 * v40
        v86 = v159((v155 for v155, v207 in v232(v76[:v40]) if v12['tape'].v249[v207] == v157), None)
        return 2 + 2 * v40 if v86 is None else 2 + v40 + v86
    return v19

def run(v20, v21, v22, v23, v12, v18, v24, v25, v26, **v27):
    return v9.v87(v20, v21, v22, v23, v12, v18, v24, v25, **v26, **v27)

def verdicts(v20, v21, v22, v23, v12, v18, v24, v25, v26):
    """What each judge would answer, run as episodes so both pay the same reading costs."""
    v17 = v88(v20, v21, v22, v23, v12, v18, v24, v25, v26, teacher_only=True)
    v28 = v88(v20, v21, v22, v23, v12, v18, v24, v25, v26, teacher_only=True, teacher_fn=v208(v12, v18))
    return (v17, v28)

def main() -> v7:
    v29 = v160.v89()
    v29.v90('--smoke', action='store_true')
    v29.v90('--bc-episodes', type=v7, default=0)
    v29.v90('--rl-episodes', type=v7, default=0)
    v29.v90('--tape-period', type=v7, default=0)
    v29.v90('--addresses', type=v7, default=0)
    v29.v90('--min-mentions', type=v7, default=2)
    v29.v90('--min-per-family', type=v7, default=8)
    v29.v90('--address-tau', type=v161, default=0.9)
    v29.v90('--address-overlap', type=v7, default=2)
    v29.v90('--soft-match', type=v161, default=0.0)
    v29.v90('--addr-key', choices=('two', 'set', 'mean'), default='two')
    v29.v90('--topk', type=v7, default=7)
    v29.v90('--max-steps', type=v7, default=14)
    v29.v90('--max-reads', type=v7, default=7)
    v29.v90('--hop', choices=('none', 'fp'), default='fp')
    v29.v90('--hop-min', type=v161, default=1.0)
    v29.v90('--k-gap', type=v161, default=0.35)
    v29.v90('--read-cost', type=v161, default=0.02)
    v29.v90('--wrong-cost', type=v161, default=1.0)
    v29.v90('--abstain-reward', type=v161, default=0.75)
    v29.v90('--entropy-bonus', type=v161, default=0.01)
    v29.v90('--lr-policy', type=v161, default=0.001)
    v29.v90('--lr-value', type=v161, default=0.003)
    v29.v90('--lr-upper', type=v161, default=3e-05)
    v29.v90('--value-coef', type=v161, default=0.5)
    v29.v90('--bc-anchor', type=v161, default=0.5)
    v29.v90('--subject-filter', choices=('off', 'on'), default='on')
    v29.v90('--no-hidden', action='store_true')
    v29.v90('--one-teacher', action='store_true', help="ablation: 280's single judge, BC everywhere")
    v29.v90('--frozen-trunk', action='store_true')
    v29.v90('--run-tag', type=v69, default='')
    v30 = v29.v91()
    global LOG_PATH
    v92.v31 = v30.v32
    v33 = v30.v142 and f'_{v30.v142}' or ''
    v33 += '_one' if v30.v93 else ''
    v6 = v0 / f'_stage285_log{v33}.txt'
    v6.v149.v71(parents=True, exist_ok=True)
    v6.v94('', encoding='utf-8')
    v25 = v162.v25('cuda' if v162.v233.v209() else 'cpu')
    v34 = v163.v95(v4)
    v162.v96(v4)
    v35 = v97.v97()
    v36 = v30.v98 or (400 if v30.v141 else 4000)
    v37 = v99(0, v30.v100)
    v38 = v30.v38 or (50 if v30.v141 else 200)
    v39 = v30.v101 or (60 if v30.v141 else 400)
    v40 = v30.v41
    v102(f'Stage285 two teachers start {v251.v243(v252.v244).v202()} device={v25} k={v40} bc={v36} rl={v37} one_teacher={v30.v93}')
    v103, v103, v104, v105 = v106()
    v23 = v164.v107(v69(v210.v165))
    v42 = v23.v108()
    v24 = v23.v166(v167) or 0
    v22 = v234.v211(v23, v104, v24, v42).v109(v25)
    v43 = v2 if v2.v168() else v1
    v21 = v212(v105, v42).v109(v25)
    v21.v110(v162.v213(v43, map_location=v25, weights_only=False)['model'])
    v169.v111(v21, 'none' if v30.v170 else 'upper')
    v44 = v171.v112(v21)
    v45 = v212(v105, v42).v109(v25)
    v45.v110(v162.v213(v1, map_location=v25, weights_only=False)['model'])
    v45.v113()
    for v46 in v45.v114():
        v46.v172(False)
    v47 = v115(v45, v104, v25)
    with v3.v150('r', encoding='utf-8', errors='ignore') as v72:
        v116 = v72.v173(4000000 if v30.v141 else 30000000)
    v48 = [v175.v174() for v175 in v116.v214('\n') if 80 <= v153(v175.v174()) <= 400]
    v49 = v7(0.7 * v153(v48))
    v50 = v48[:v49][:3000 if v30.v141 else 25000]
    v51 = v48[v49:][:1500 if v30.v141 else 12000]

    def new_pack(v28, v117):
        return v9.v176(v117, bank=v47, tok=v23, pad_id=v24, device=v25, rng=v28, n_addr=v39, min_mentions=v30.v144, tau=v30.v215, overlap=v30.v216, soft_match=v30.v217, min_per_family=v30.v145, addr_key=v30.v143)
    v12 = v118(v34, v50)
    if v153(v12['items']) < 8:
        v102('  too few items')
        return 1
    v102(f"  tape: {v12['n_addresses']} addresses, {v12['n_slots']} slots | items {v230.v203(v121(v254((v155['kind'] for v155 in v12['items']))))}")
    v52 = 0 if v30.v32 else 2 * (v21.v235.v218 // 2)
    v20 = v92.v119(v52 + v92.v177, v40, v25)
    v53 = [v46 for v46 in v21.v114() if v46.v178]
    v54 = v162.v179.v120([{'params': [v46 for v256, v46 in v20.v257() if not v256.v258('v.')], 'lr': v30.v236}, {'params': v228(v20.v17.v114()), 'lr': v30.v237}] + ([{'params': v53, 'lr': v30.v248}] if v53 else []), weight_decay=0.01)
    v26 = v121(k=v40, max_steps=v30.v81, max_reads=v30.v82, read_cost=v30.v180, wrong_cost=v30.v181, abstain_reward=v30.v182, hop=v30.v183, hop_min=v30.v184, k_gap=v30.v185, subject_filter=v30.v219 == 'on')

    def contested(v46):
        v67 = {}
        for v122 in v46['items']:
            v17, v28 = v220(v20, v21, v22, v23, v46, v122, v24, v25, v26)
            v67[v238(v122)] = v17['correct'] != v28['correct'] or v17['abstained'] != v28['abstained']
        return v67
    with v162.v139():
        v123 = v186(v12)
    v102(f"  contested on train tape: {v75(v123.v249())}/{v153(v12['items'])}")
    v124, v125 = ([], [])
    v20.v126()
    v21.v126(not v30.v170)
    for v55 in v127(1, v36 + 1):
        if (v55 - 1) % v38 == 0 and v55 > 1:
            v12 = v118(v34, v50)
            with v162.v139():
                v123 = v186(v12)
        v18 = v12['items'][v34.v221(v153(v12['items']))]
        if not v30.v93 and v123.v205(v238(v18)):
            continue
        v67 = v88(v20, v21, v22, v23, v12, v18, v24, v25, v26, bc=True)
        v54.v187(set_to_none=True)
        v67['loss'].v188()
        v162.v239.v222.v189(v228(v20.v114()) + v53, 1.0)
        v54.v190()
        if v55 % v99(1, v36 // 8) == 0:
            v124.v195({'phase': 'bc', 'episode': v55, 'loss': v161(v67['loss']), 'kind': v67['kind'], 'trace': v67['trace']})
            v102(f"  bc {v55}/{v36} loss={v161(v67['loss']):.4f} [{v67['kind']}] {v67['trace']}")
    for v55 in v127(1, v37 + 1):
        if (v55 - 1) % v38 == 0 and v55 > 1:
            v12 = v118(v34, v50)
            with v162.v139():
                v123 = v186(v12)
        v18 = v12['items'][v34.v221(v153(v12['items']))]
        v20.v128 = []
        v129 = 0.0 if not v30.v93 and v123.v205(v238(v18)) else v30.v191
        v67 = v88(v20, v21, v22, v23, v12, v18, v24, v25, v26, greedy=False, bc_anchor=v129)
        v192, v20.v128 = (v20.v128, None)
        if not v67['logps']:
            continue
        v130 = v67['reward']
        v131 = v162.v193(v192[:v153(v67['logps'])])
        v132 = v223.v194(v131, v162.v224(v131, v130))
        v125.v195(v161(v132))
        v133 = v162.v193(v67['entropy']).v75() if v67['entropy'] else v162.v225((), device=v25)
        v134 = -((v130 - v131).v259() * v162.v193(v67['logps'])).v75() + v30.v240 * v132 - v30.v226 * v133
        if v129 > 0.0 and v67['loss'].v178:
            v134 = v134 + v129 * v67['loss']
        v54.v187(set_to_none=True)
        v134.v188()
        v162.v239.v222.v189(v228(v20.v114()) + v53, 1.0)
        v54.v190()
        if v55 % v99(1, v37 // 8) == 0:
            v124.v195({'phase': 'rl', 'episode': v55, 'v_mse': v161(v255.v250(v125[-200:])), 'kind': v67['kind'], 'trace': v67['trace']})
            v102(f"  rl {v55}/{v37} v_mse={v255.v250(v125[-200:]):.3f} [{v67['kind']}] {v67['trace']}")
    v20.v113()
    v21.v113()
    v56 = v171.v112(v21)

    @v162.v139()
    def evaluate(v46):
        v135 = {'agree': v227(v228), 'disagree': v227(v228)}
        v136 = {v72: v227(v228) for v72 in v5}
        v137 = []
        for v122 in v46['items']:
            v196 = v88(v20, v21, v22, v23, v46, v122, v24, v25, v26)
            v17, v28 = v220(v20, v21, v22, v23, v46, v122, v24, v25, v26)
            v197 = 'disagree' if v17['correct'] != v28['correct'] or v17['abstained'] != v28['abstained'] else 'agree'
            v198 = v135[v197]
            v198['policy_correct'].v195(v196['correct'])
            v198['policy_abstain'].v195(v7(v196['abstained']))
            v198['policy_reward'].v195(v196['reward'])
            v198['votes_correct'].v195(v17['correct'])
            v198['return_correct'].v195(v28['correct'])
            v198['votes_reward'].v195(v17['reward'])
            v198['return_reward'].v195(v28['reward'])
            v72 = v122['kind']
            v136[v72]['correct'].v195(v196['correct'])
            v136[v72]['abstain'].v195(v7(v196['abstained']))
            v136[v72]['reward'].v195(v196['reward'])
            if v153(v137) < 20:
                v137.v195({'kind': v72, 'S': v122['S'], 'bucket': v197, 'trace': v196['trace'], 'correct': v196['correct'], 'votes': v17['correct'], 'return': v28['correct']})
        v10 = lambda v229: v161(v255.v250(v229)) if v153(v229) else v161('nan')
        v67 = {'n_items': v153(v46['items']), 'traces': v137, 'contested_rate': v153(v135['disagree']['policy_correct']) / v99(1, v153(v46['items'])), 'reward_total': v10([v241 for v138 in v135.v249() for v241 in v138['policy_reward']]), 'votes_reward_total': v10([v241 for v138 in v135.v249() for v241 in v138['votes_reward']]), 'return_reward_total': v10([v241 for v138 in v135.v249() for v241 in v138['return_reward']])}
        for v138 in ('agree', 'disagree'):
            v198 = v135[v138]
            v67[v138] = {'n': v153(v198['policy_correct']), 'policy_acc': v10(v198['policy_correct']), 'votes_acc': v10(v198['votes_correct']), 'return_acc': v10(v198['return_correct']), 'policy_abstain': v10(v198['policy_abstain']), 'policy_reward': v10(v198['policy_reward'])}
        for v72 in v5:
            v67[v72] = {'n': v153(v136[v72]['abstain']), 'coverage': 1.0 - v10(v136[v72]['abstain']), 'abstain': v10(v136[v72]['abstain']), 'reward': v10(v136[v72]['reward']), 'correct': v10(v136[v72]['correct'])}
        return v67
    v57 = v140(v12)
    v58 = v118(v163.v95(v4 + 99), v51)
    v59 = v140(v58)
    v102(f"  HELD-OUT {v230.v203({v199: v200 for v199, v200 in v59.v242() if v199 != 'traces'})}")
    v60 = v59['disagree']
    v61 = v44 == v56
    v62 = v60['n'] >= 5
    v63 = v62 and v60['policy_acc'] > v60['votes_acc'] and (v60['policy_acc'] > v60['return_acc'])
    v64 = v59['agree']['policy_acc'] >= 0.6
    v65 = v59['reward_total'] >= 0.704 - 0.05
    v66 = 'ARBITRATION_OK' if v61 and v63 and v64 else 'ARBITRATION_NO' if v62 else 'NO_DISAGREEMENT'
    v67 = {'stage': 285, 'overall': v66, 'seed': v4, 'smoke': v30.v141, 'one_teacher': v30.v93, 'run_tag': v30.v142, 'addr_key': v30.v143, 'bc_episodes': v36, 'rl_episodes': v37, 'topk': v40, 'min_mentions': v30.v144, 'min_per_family': v30.v145, 'reward': {'correct': 1.0, 'wrong': -v30.v181, 'abstain': v30.v182, 'read': -v30.v180}, 'gates': {'G_arc_enc_frozen': v61, 'G_teachers_disagree': v62, 'G_arbitration': v63, 'G_agreement_kept': v64, 'G_holds_baseline': v65}, 'train_tape': {v199: v200 for v199, v200 in v57.v242() if v199 != 'traces'}, 'held_out': v59, 'curve': v124, 'arc_enc_hash_before': v44, 'arc_enc_hash_after': v56, 'fp_version': v171.v201(), 'reference_280_baseline': {'held_out_reward': 0.704, 'acc_answered_all': 0.9, 'teacher_ceiling': 0.625}, 'note': "Two judges over the 280 tape, neither reading a label. Votes counts witnesses and abstains on equal support; return counts how many other mentions carry a value together with the subject and abstains under two. They share no criterion on purpose - judges made of the same quantity would agree everywhere and leave nothing to arbitrate. Where they agree BC takes the demonstration; where they differ there is no target to copy and the episode is left to the reward, so the contested items are exactly the ones resemblance cannot solve. The bar is not to beat two teachers: both read one tape and the tape is the truth here, so where they differ at least one is misreading it, and the mind has to be right more often than either reading alone. --one-teacher restores 280's single judge with BC everywhere, which is the arm the 0.704 baseline came from.", 'timestamp': v251.v243(v252.v244).v202(), 'wall_s': v97.v97() - v35}
    v0.v71(parents=True, exist_ok=True)
    (v0 / f'stage285_decision{v33}.json').v94(v230.v203(v67, indent=2), encoding='utf-8')
    v102(v230.v203({'overall': v66, 'gates': v67['gates'], 'disagree': v59['disagree']}, indent=2))
    return 0
if v68 == '__main__':
    raise v146(v204())