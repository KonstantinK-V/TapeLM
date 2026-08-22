"""Twenty lines out of a two-megabyte report - the numbers 299 is actually read by.

WHY THIS EXISTS. The stage writes everything it measured, which is right: a report that keeps
only what looked interesting on the day cannot be re-read when the question changes. But the
file is megabytes, and pasting it into a conversation burns the budget on braces. This prints
the void conditions first and the claim second, in the order HANDOFF 0 says to read them, and
nothing else.

    python _read299.py out/_stage289_decision_299e_both.json [more.json ...]
    python _read299.py out/_stage289_decision_299e_s*.json --held      seeds of one arm

POOLING SEEDS. Given several files it adds the walk-only counts and prints one McNemar over the
total - the honest way to buy power on a subset of ~104 questions per run: more seeds, same
mechanism, counts added, not a rerun kept because it read better. It pools only files that agree
on fingerprint, lookahead, refusal, import budget, places and candidate cap; mixed arms are
printed side by side and refused, because summing two different configurations produces a
real-looking z for a comparison nobody made. `--held` drops the train control.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
v0 = ('reachable_rate', 'reachable_random', 'reachable_wide', 'own_hit_rate', 'ceiling', 'walk_only_rate', 'step_rate')
v1 = ('hit_of_walk_only', 'rival_of_walk_only', 'hit_rate', 'own_rival_hit_rate', 'rival_hit_rate', 'hit_of_own', 'own_rival_of_own', 'steps_on_walk_only')
v2 = ('cand_places', 'world_rows_own', 'world_rows_candidate', 'world_rows_expand', 'world_rows_expand_when_stepped', 'world_rows_expand_when_not', 'step_vs_size_r')

def counts(v4):
    """The walk-only contest as integers. Rates rounded to four places hide the denominator,
    and on 104 questions the denominator is the whole story."""
    v5 = v4.v32('walk_only_paired')
    if v5:
        return (v5['mind_only'], v5['rival_only'], v5['n'])
    v6 = v3(v48(v4['walk_only_rate'] * v4['n']))
    return (v3(v48(v4['hit_of_walk_only'] * v6)), v3(v48(v4['rival_of_walk_only'] * v6)), v6)

def z_of(v7, v8):
    return (v7 - v8) / v75.v64(v7 + v8) if v7 + v8 else v49('nan')

def one(v9, v10):
    v11 = v50.v33(v51(v9, encoding='utf-8'))
    v12 = v11.v32('reach')
    if not v12 or not v12.v32(v10):
        return None
    v4 = v12[v10]
    v13 = (v12.v32('frame_fp'), v12.v32('lookahead'), v12.v32('no_refuse'), v11.v32('import_k'), v12.v32('places'), v12.v32('cands_cap'), v12.v32('speak_batch'), v12.v32('speak_weight'), v12.v32('calib_batch'), v12.v32('calib_weight'))
    v34(f"\n{v9}  [{v10}]  {v11['wall_s']:.0f}s  fp={v12.v32('frame_fp')} lookahead={v12.v32('lookahead')} import={v12.v32('import')} no_refuse={v12.v32('no_refuse')} seed={v11['seed']}" + (f"  retain={v11['retain']}/{v11.v32('retain_by')}" if v11.v32('retain') else '') + (f"  SPEAK-BATCH={v12['speak_batch']}x{v12.v32('speak_weight')}" if v12.v32('speak_batch') else '') + (f"  CALIB={v12['calib_batch']}x{v12.v32('calib_weight')}" if v12.v32('calib_batch') else '') + f"  min_fillers={v11.v32('min_fillers')}" + ('  CONNECT' if v11.v32('connect') else '') + (f"  COPY d={v11.v32('copy_d')}" + ('-BACKFILL' if v11.v32('copy_backfill') else '') if v11.v32('copy') else '') + ('  CHANNEL' if v11.v32('reach_channel') else '') + (f"  MOVES[{','.v65(v11.v32('move_set') or [])}]" if v11.v32('moves') else '') + ('  OWN-IN-OFFER' if v11.v32('own_in_offer') else '') + ('  OWN-IMPORT' if v11.v32('own_import') else ''))
    v34(f"  tape   resample_overlap {v11['resample']['mean_overlap']:.3f}   rare_nnz {v11['rare_nonzero_rate']:.5f}   params {v11['params']}")
    v34(f"  cos    mean {v11['cos_mean']:.3f} std {v11['cos_std']:.3f}  (dominated by same-address pairs - not a channel test)")
    v34('  void   ' + '  '.v65((f"{v79.v92('_rate', '')} {v4[v79]:.4f}" for v79 in v0 if v79 in v4)))
    v34('  size   ' + '  '.v65((f"{v79.v92('world_rows_', '')} {v4[v79]:.2f}" for v79 in v2 if v79 in v4 and v4[v79] == v4[v79])))
    v34('  claim  ' + '  '.v65((f'{v79} {v4[v79]:.4f}' for v79 in v1 if v79 in v4 and v4[v79] == v4[v79])))
    if v4.v32('move_share'):
        v35 = v4.v32('move_hit') or {}
        v34('  MOVES  ' + '  '.v65((f'{v52} {v94:.4f}' + (f'/{v35[v52]:.4f}' if v52 in v35 else '') for v52, v94 in v4['move_share'].v82())) + '   (share/hit)')
    v14 = v4.v32('line_only_paired')
    if v14 and v14['n']:
        v34(f"  LINE-ONLY  mind {v14['mind_only']} / rival {v14['rival_only']} of {v14['n']}   z {v14['mcnemar_z']:+.2f}   line_reach {v4.v32('line_reach_rate', v49('nan')):.4f}  step_line {v4.v32('step_line_rate', v49('nan')):.4f}")
    v15 = v4.v32('own_paired')
    if v15 and v15['n']:
        v34(f"  CONFIRM    mind {v15['mind_only']} / rival {v15['rival_only']} of {v15['n']}   z {v15['mcnemar_z']:+.2f}{('   UNDERPOWERED' if v75.v64(v15['mind_only'] + v15['rival_only']) <= 1.645 else '')}")
    v16 = v4.v32('router')
    if v16 and v16['mind_enrichment'] == v16['mind_enrichment']:
        v34(f"  ROUTER     stepped {v16['n_stepped']}  mind {v16['mind_enrichment']:.2f}x  counting {v16['count_enrichment']:.2f}x  |own| stepped {v16['n_own_when_stepped']:.2f} vs {v16['n_own_when_not']:.2f}")
        if 'share_enrichment' in v16:
            v34(f"  ROUTER share  {v16['share_enrichment']:.2f}x   top_share stepped {v16['top_share_when_stepped']:.3f} vs {v16['top_share_when_not']:.3f}" + (f"   ties {v4['count_rival_ties']:.2f}" if 'count_rival_ties' in v4 else ''))
    v5 = v4.v32('walk_only_pick') or {}
    if v5.v32('n'):
        v34(f"  PICK       stepped {v5['n']} of walk-only   mind {v5['mind']} ({v5['hit_rate']:.4f}) / rival {v5['rival']} ({v5['rival_rate']:.4f})   paired {v5['mind_only']}/{v5['rival_only']}   z {v5['mcnemar_z']:+.2f}   arrive {v4.v32('walk_only_arrive', v49('nan')):.4f}")
        if 'count_rival' in v5:
            v34(f"  PICK vs COUNT   count-rival {v5['count_rival']} ({v5['count_rival_rate']:.4f})   paired {v5['vs_count_mind_only']}/{v5['vs_count_rival_only']}   z {v5['vs_count_z']:+.2f}")
    if 'deep_rate' in v4:
        v34(f"  DEPTH      deep {v4['deep_rate']:.4f}   hit_of_deep {v4.v32('hit_of_deep', v49('nan')):.4f}   hit_of_depth1 {v4.v32('hit_of_depth1', v49('nan')):.4f}")
    v17 = v4.v32('bisect') or {}
    if v17.v32('n'):
        v34(f"  BISECT     {v17['bisect_right']}/{v17['n']} vs flat {v17['flat_right']}/{v17['n']}   paired {v17['bisect_only']}/{v17['flat_only']}   z {v17['mcnemar_z']:+.2f}   splits {v17['splits_mean']:.1f}")
    v18 = v4.v32('other_mind')
    if v18:
        for v36 in ('all', 'walk_only', 'confirm'):
            v52 = v18[v36]
            if not v52['n']:
                continue
            v34(f"  OTHER {v36:<10} this {v52['this']} / other {v52['other']} of {v52['n']}   paired {v52['this_only']}/{v52['other_only']}   z {v52['mcnemar_z']:+.2f}" + ('   IDENTICAL - right on exactly the same questions' if v52.v32('identical') else '   UNDERPOWERED' if v52['underpowered'] else ''))
        v34(f"  OTHER step   this {v18['step_rate']:.4f} vs other {v18['other_step_rate']:.4f}")
    v19 = v4.v32('question_rank') or {}
    for v20 in ('answerable', 'ceiling', 'right'):
        v37 = v19.v32(v20)
        if not v37:
            continue

        def a(v46):
            return v37[v46]['auc']

        def pk(v46):
            return v37[v46]['prec'][-1]
        v34(f"  RANK {v20:<10} base {v37['base_rate']:.4f}   AUC mind {v43('mind_margin'):.4f} (score {v43('mind_score'):.4f})  |own| {v43('count_n_own'):.4f}  share {v43('count_top_share'):.4f}   p@{v37['k'][-1]} mind {v87('mind_margin'):.4f} vs {v88(v87('count_n_own'), v87('count_top_share')):.4f}")
        v38 = v43('mind_score') - 0.5
        v34(f"       GAUGE    raw-score AUC {v43('mind_score'):.4f}  ({v38:+.4f} vs a coin)   margin {v43('mind_margin'):.4f}   rivals {v43('count_n_own'):.4f}/{v43('count_top_share'):.4f}   {('FREE' if v93(v38) < 0.03 else 'TIED')}")
    v21 = v4.v32('margin_by_stage')
    if v21:
        v34(f"  MARGIN     right {v21['by_right']['right']:.3f} vs wrong {v21['by_right']['wrong']:.3f}   |   stayed {v21['stayed']:.3f} (n {v21['n_stayed']}) vs stepped {v21['stepped']:.3f} (n {v21['n_stepped']})")
        v34(f"             within-stage: stayed {v21['stayed_right']:.3f}/{v21['stayed_wrong']:.3f}   stepped {v21['stepped_right']:.3f}/{v21['stepped_wrong']:.3f}   (right/wrong)")
    for v39, v40 in (('GATE', v4.v32('gate')), ('GATE-WO', v4.v32('gate_walk_only'))):
        if not v40:
            continue
        v34(f"  {v39:<9}  n {v40['n']}  ungated hit {v40['ungated_hit_rate']:.4f}  payoff {v40['ungated_payoff']:+.4f}   ALWAYS-SILENT PAYS {v40.v32('always_silent', v49('nan')):+.4f}")
        for v41 in v40['fractions']:
            v11 = v40[f'{v41:.2f}']
            v52, v43, v66 = (v11['mind'], v11['count_n_own'], v11['count_top_share'])
            v53 = v11.v32('random') or {}
            v54 = v11.v32('composition') or {}
            v34(f"    top {v41:>4.0%} k={v11['k']:<5} mind {v52['precision']:.4f} ({v52['yield']} right, gain {v52.v32('gain', v49('nan')):+.4f})   |own| {v43['precision']:.4f} ({v43['yield']})   share {v66['precision']:.4f} ({v66['yield']})   rand {v53.v32('precision', v49('nan')):.4f} ({v53.v32('yield', 0)})   z vs |own| {v11['vs_count_n_own']['mcnemar_z']:+.2f}  vs share {v11['vs_count_top_share']['mcnemar_z']:+.2f}" + (f"   [kept {v54.v32('confirm', 0)}c/{v54.v32('walk_only', 0)}w, right {v54.v32('right_confirm', 0)}c/{v54.v32('right_walk_only', 0)}w]" if v54 and v39 == 'GATE' else ''))
    v7, v8, v6 = v42(v4)
    v34(f"  WALK-ONLY  mind {v7} / rival {v8} of {v6}   z {v76(v7, v8):+.2f}{('   UNDERPOWERED' if v75.v64(v7 + v8) <= 1.645 else '')}")
    return (v7, v8, v6, v13)

def main(v22) -> v3:
    v23 = [v43 for v43 in v22 if not v43.v77('--')]
    if not v23:
        v34(v55)
        return 1
    v44, v45 = (v56(), [])
    for v24 in v23:
        v46 = v89(v24).v67.v57('_')
        if v46 in v44:
            continue
        v44.v58(v46)
        v45.v59(v24)
    if v60(v45) != v60(v23):
        v34(f'note: dropped {v60(v23) - v60(v45)} duplicate report name(s) - out/ and results/ hold the same file')
        v23 = v45
    v25 = ['held_out'] if '--held' in v22 else ['held_out', 'train_control']
    v26 = {v43: [0, 0, 0] for v43 in v25}
    v27 = {v43: [0, 0, 0] for v43 in v25}
    v28 = {v43: {'all': [0, 0, 0], 'walk_only': [0, 0, 0]} for v43 in v25}
    v29 = {v43: {} for v43 in v25}
    v30 = {v43: v56() for v43 in v25}
    for v24 in v23:
        for v43 in v25:
            v61 = v68(v24, v43)
            if v61:
                for v69 in v78(3):
                    v26[v43][v69] += v61[v69]
                v30[v43].v58(v61[3])
                v70 = v50.v33(v51(v24, encoding='utf-8'))['reach'][v43]
                v15 = v70.v32('own_paired')
                if v15:
                    v27[v43][0] += v15['mind_only']
                    v27[v43][1] += v15['rival_only']
                    v27[v43][2] += v15['n']
                v71 = v70.v32('gate_walk_only')
                if v71:
                    for v41 in v71['fractions']:
                        v90 = v29[v43].v91(v41, [0, 0, 0, 0])
                        v90[0] += v71[f'{v41:.2f}']['k']
                        v90[1] += v71[f'{v41:.2f}']['mind']['yield']
                        v90[2] += v71[f'{v41:.2f}']['count_top_share']['yield']
                        v90[3] += (v71[f'{v41:.2f}'].v32('random') or {}).v32('yield', 0)
                v18 = v70.v32('other_mind')
                if v18:
                    for v36 in ('all', 'walk_only'):
                        v28[v43][v36][0] += v18[v36]['this_only']
                        v28[v43][v36][1] += v18[v36]['other_only']
                        v28[v43][v36][2] += v18[v36]['n']
    if v60(v23) > 1:
        for v43 in v25:
            v7, v8, v6 = v26[v43]
            if v60(v30[v43]) > 1:
                v34(f'\nNOT POOLED ({v43}): {v60(v30[v43])} different arms among these files. Pool seeds of ONE configuration; arms are compared, not summed.')
                continue
            v34(f"\nPOOLED {v43} over {v60(v23)} runs: mind {v7} / rival {v8} of {v6}   z {v76(v7, v8):+.2f}{('   UNDERPOWERED' if v75.v64(v7 + v8) <= 1.645 else '')}")
            v72, v73, v62 = v27[v43]
            if v62:
                v34(f'POOLED CONFIRM {v43} over {v60(v23)} runs: mind {v72} / rival {v73} of {v62}   z {v76(v72, v73):+.2f}')
            for v41, (v79, v35, v80, v81) in v74(v29[v43].v82()):
                if not v79:
                    continue
                v34(f'POOLED GATE-WO {v43} top {v41:>4.0%}: k {v79}   mind {v35} ({v35 / v79:.4f})   share {v80} ({v80 / v79:.4f})   random {v81} ({v81 / v79:.4f})')
            for v36 in ('all', 'walk_only'):
                v83, v84, v85 = v28[v43][v36]
                if not v85:
                    continue
                v34(f"POOLED OTHER {v36} {v43} over {v60(v23)} runs: this {v83} / other {v84}   z {v76(v83, v84):+.2f}   ({v83 + v84} discordant of {v85}){('   TOO FEW DISCORDANT TO READ' if v83 + v84 < 10 else '')}")
    return 0
if v31 == '__main__':
    raise v47(v63(v86.v22[1:]))