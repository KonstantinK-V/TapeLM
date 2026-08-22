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
VOID = ('reachable_rate', 'reachable_random', 'reachable_wide', 'own_hit_rate', 'ceiling', 'walk_only_rate', 'step_rate')
CLAIM = ('hit_of_walk_only', 'rival_of_walk_only', 'hit_rate', 'own_rival_hit_rate', 'rival_hit_rate', 'hit_of_own', 'own_rival_of_own', 'steps_on_walk_only')
SIZE = ('cand_places', 'world_rows_own', 'world_rows_candidate', 'world_rows_expand', 'world_rows_expand_when_stepped', 'world_rows_expand_when_not', 'step_vs_size_r')

def counts(r):
    """The walk-only contest as integers. Rates rounded to four places hide the denominator,
    and on 104 questions the denominator is the whole story."""
    wp = r.get('walk_only_paired')
    if wp:
        return (wp['mind_only'], wp['rival_only'], wp['n'])
    n = int(round(r['walk_only_rate'] * r['n']))
    return (int(round(r['hit_of_walk_only'] * n)), int(round(r['rival_of_walk_only'] * n)), n)

def z_of(b, c):
    return (b - c) / math.sqrt(b + c) if b + c else float('nan')

def one(path, arm):
    d = json.load(open(path, encoding='utf-8'))
    rc = d.get('reach')
    if not rc or not rc.get(arm):
        return None
    r = rc[arm]
    cfg = (rc.get('frame_fp'), rc.get('lookahead'), rc.get('no_refuse'), d.get('import_k'), rc.get('places'), rc.get('cands_cap'), rc.get('speak_batch'), rc.get('speak_weight'), rc.get('calib_batch'), rc.get('calib_weight'))
    print(f"\n{path}  [{arm}]  {d['wall_s']:.0f}s  fp={rc.get('frame_fp')} lookahead={rc.get('lookahead')} import={rc.get('import')} no_refuse={rc.get('no_refuse')} seed={d['seed']}" + (f"  retain={d['retain']}/{d.get('retain_by')}" if d.get('retain') else '') + (f"  SPEAK-BATCH={rc['speak_batch']}x{rc.get('speak_weight')}" if rc.get('speak_batch') else '') + (f"  CALIB={rc['calib_batch']}x{rc.get('calib_weight')}" if rc.get('calib_batch') else '') + f"  min_fillers={d.get('min_fillers')}" + ('  CONNECT' if d.get('connect') else '') + (f"  COPY d={d.get('copy_d')}" + ('-BACKFILL' if d.get('copy_backfill') else '') if d.get('copy') else '') + ('  CHANNEL' if d.get('reach_channel') else '') + (f"  MOVES[{','.join(d.get('move_set') or [])}]" if d.get('moves') else '') + ('  OWN-IN-OFFER' if d.get('own_in_offer') else '') + ('  OWN-IMPORT' if d.get('own_import') else ''))
    print(f"  tape   resample_overlap {d['resample']['mean_overlap']:.3f}   rare_nnz {d['rare_nonzero_rate']:.5f}   params {d['params']}")
    print(f"  cos    mean {d['cos_mean']:.3f} std {d['cos_std']:.3f}  (dominated by same-address pairs - not a channel test)")
    print('  void   ' + '  '.join((f"{k.replace('_rate', '')} {r[k]:.4f}" for k in VOID if k in r)))
    print('  size   ' + '  '.join((f"{k.replace('world_rows_', '')} {r[k]:.2f}" for k in SIZE if k in r and r[k] == r[k])))
    print('  claim  ' + '  '.join((f'{k} {r[k]:.4f}' for k in CLAIM if k in r and r[k] == r[k])))
    if r.get('move_share'):
        mh = r.get('move_hit') or {}
        print('  MOVES  ' + '  '.join((f'{m} {v:.4f}' + (f'/{mh[m]:.4f}' if m in mh else '') for m, v in r['move_share'].items())) + '   (share/hit)')
    lp = r.get('line_only_paired')
    if lp and lp['n']:
        print(f"  LINE-ONLY  mind {lp['mind_only']} / rival {lp['rival_only']} of {lp['n']}   z {lp['mcnemar_z']:+.2f}   line_reach {r.get('line_reach_rate', float('nan')):.4f}  step_line {r.get('step_line_rate', float('nan')):.4f}")
    op = r.get('own_paired')
    if op and op['n']:
        print(f"  CONFIRM    mind {op['mind_only']} / rival {op['rival_only']} of {op['n']}   z {op['mcnemar_z']:+.2f}{('   UNDERPOWERED' if math.sqrt(op['mind_only'] + op['rival_only']) <= 1.645 else '')}")
    rt = r.get('router')
    if rt and rt['mind_enrichment'] == rt['mind_enrichment']:
        print(f"  ROUTER     stepped {rt['n_stepped']}  mind {rt['mind_enrichment']:.2f}x  counting {rt['count_enrichment']:.2f}x  |own| stepped {rt['n_own_when_stepped']:.2f} vs {rt['n_own_when_not']:.2f}")
        if 'share_enrichment' in rt:
            print(f"  ROUTER share  {rt['share_enrichment']:.2f}x   top_share stepped {rt['top_share_when_stepped']:.3f} vs {rt['top_share_when_not']:.3f}" + (f"   ties {r['count_rival_ties']:.2f}" if 'count_rival_ties' in r else ''))
    wp = r.get('walk_only_pick') or {}
    if wp.get('n'):
        print(f"  PICK       stepped {wp['n']} of walk-only   mind {wp['mind']} ({wp['hit_rate']:.4f}) / rival {wp['rival']} ({wp['rival_rate']:.4f})   paired {wp['mind_only']}/{wp['rival_only']}   z {wp['mcnemar_z']:+.2f}   arrive {r.get('walk_only_arrive', float('nan')):.4f}")
        if 'count_rival' in wp:
            print(f"  PICK vs COUNT   count-rival {wp['count_rival']} ({wp['count_rival_rate']:.4f})   paired {wp['vs_count_mind_only']}/{wp['vs_count_rival_only']}   z {wp['vs_count_z']:+.2f}")
    if 'deep_rate' in r:
        print(f"  DEPTH      deep {r['deep_rate']:.4f}   hit_of_deep {r.get('hit_of_deep', float('nan')):.4f}   hit_of_depth1 {r.get('hit_of_depth1', float('nan')):.4f}")
    bi = r.get('bisect') or {}
    if bi.get('n'):
        print(f"  BISECT     {bi['bisect_right']}/{bi['n']} vs flat {bi['flat_right']}/{bi['n']}   paired {bi['bisect_only']}/{bi['flat_only']}   z {bi['mcnemar_z']:+.2f}   splits {bi['splits_mean']:.1f}")
    om = r.get('other_mind')
    if om:
        for sub in ('all', 'walk_only', 'confirm'):
            m = om[sub]
            if not m['n']:
                continue
            print(f"  OTHER {sub:<10} this {m['this']} / other {m['other']} of {m['n']}   paired {m['this_only']}/{m['other_only']}   z {m['mcnemar_z']:+.2f}" + ('   IDENTICAL - right on exactly the same questions' if m.get('identical') else '   UNDERPOWERED' if m['underpowered'] else ''))
        print(f"  OTHER step   this {om['step_rate']:.4f} vs other {om['other_step_rate']:.4f}")
    qr = r.get('question_rank') or {}
    for tgt in ('answerable', 'ceiling', 'right'):
        t = qr.get(tgt)
        if not t:
            continue

        def a(nm):
            return t[nm]['auc']

        def pk(nm):
            return t[nm]['prec'][-1]
        print(f"  RANK {tgt:<10} base {t['base_rate']:.4f}   AUC mind {a('mind_margin'):.4f} (score {a('mind_score'):.4f})  |own| {a('count_n_own'):.4f}  share {a('count_top_share'):.4f}   p@{t['k'][-1]} mind {pk('mind_margin'):.4f} vs {max(pk('count_n_own'), pk('count_top_share')):.4f}")
        gap = a('mind_score') - 0.5
        print(f"       GAUGE    raw-score AUC {a('mind_score'):.4f}  ({gap:+.4f} vs a coin)   margin {a('mind_margin'):.4f}   rivals {a('count_n_own'):.4f}/{a('count_top_share'):.4f}   {('FREE' if abs(gap) < 0.03 else 'TIED')}")
    mb = r.get('margin_by_stage')
    if mb:
        print(f"  MARGIN     right {mb['by_right']['right']:.3f} vs wrong {mb['by_right']['wrong']:.3f}   |   stayed {mb['stayed']:.3f} (n {mb['n_stayed']}) vs stepped {mb['stepped']:.3f} (n {mb['n_stepped']})")
        print(f"             within-stage: stayed {mb['stayed_right']:.3f}/{mb['stayed_wrong']:.3f}   stepped {mb['stepped_right']:.3f}/{mb['stepped_wrong']:.3f}   (right/wrong)")
    for gname, g in (('GATE', r.get('gate')), ('GATE-WO', r.get('gate_walk_only'))):
        if not g:
            continue
        print(f"  {gname:<9}  n {g['n']}  ungated hit {g['ungated_hit_rate']:.4f}  payoff {g['ungated_payoff']:+.4f}   ALWAYS-SILENT PAYS {g.get('always_silent', float('nan')):+.4f}")
        for fr in g['fractions']:
            d = g[f'{fr:.2f}']
            m, a, s = (d['mind'], d['count_n_own'], d['count_top_share'])
            r0 = d.get('random') or {}
            co = d.get('composition') or {}
            print(f"    top {fr:>4.0%} k={d['k']:<5} mind {m['precision']:.4f} ({m['yield']} right, gain {m.get('gain', float('nan')):+.4f})   |own| {a['precision']:.4f} ({a['yield']})   share {s['precision']:.4f} ({s['yield']})   rand {r0.get('precision', float('nan')):.4f} ({r0.get('yield', 0)})   z vs |own| {d['vs_count_n_own']['mcnemar_z']:+.2f}  vs share {d['vs_count_top_share']['mcnemar_z']:+.2f}" + (f"   [kept {co.get('confirm', 0)}c/{co.get('walk_only', 0)}w, right {co.get('right_confirm', 0)}c/{co.get('right_walk_only', 0)}w]" if co and gname == 'GATE' else ''))
    b, c, n = counts(r)
    print(f"  WALK-ONLY  mind {b} / rival {c} of {n}   z {z_of(b, c):+.2f}{('   UNDERPOWERED' if math.sqrt(b + c) <= 1.645 else '')}")
    return (b, c, n, cfg)

def main(argv) -> int:
    files = [a for a in argv if not a.startswith('--')]
    if not files:
        print(__doc__)
        return 1
    seen, uniq = (set(), [])
    for f in files:
        nm = Path(f).name.lstrip('_')
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append(f)
    if len(uniq) != len(files):
        print(f'note: dropped {len(files) - len(uniq)} duplicate report name(s) - out/ and results/ hold the same file')
        files = uniq
    arms = ['held_out'] if '--held' in argv else ['held_out', 'train_control']
    tot = {a: [0, 0, 0] for a in arms}
    con = {a: [0, 0, 0] for a in arms}
    oth = {a: {'all': [0, 0, 0], 'walk_only': [0, 0, 0]} for a in arms}
    gate = {a: {} for a in arms}
    cfgs = {a: set() for a in arms}
    for f in files:
        for a in arms:
            got = one(f, a)
            if got:
                for i in range(3):
                    tot[a][i] += got[i]
                cfgs[a].add(got[3])
                r_a = json.load(open(f, encoding='utf-8'))['reach'][a]
                op = r_a.get('own_paired')
                if op:
                    con[a][0] += op['mind_only']
                    con[a][1] += op['rival_only']
                    con[a][2] += op['n']
                gw = r_a.get('gate_walk_only')
                if gw:
                    for fr in gw['fractions']:
                        e = gate[a].setdefault(fr, [0, 0, 0, 0])
                        e[0] += gw[f'{fr:.2f}']['k']
                        e[1] += gw[f'{fr:.2f}']['mind']['yield']
                        e[2] += gw[f'{fr:.2f}']['count_top_share']['yield']
                        e[3] += (gw[f'{fr:.2f}'].get('random') or {}).get('yield', 0)
                om = r_a.get('other_mind')
                if om:
                    for sub in ('all', 'walk_only'):
                        oth[a][sub][0] += om[sub]['this_only']
                        oth[a][sub][1] += om[sub]['other_only']
                        oth[a][sub][2] += om[sub]['n']
    if len(files) > 1:
        for a in arms:
            b, c, n = tot[a]
            if len(cfgs[a]) > 1:
                print(f'\nNOT POOLED ({a}): {len(cfgs[a])} different arms among these files. Pool seeds of ONE configuration; arms are compared, not summed.')
                continue
            print(f"\nPOOLED {a} over {len(files)} runs: mind {b} / rival {c} of {n}   z {z_of(b, c):+.2f}{('   UNDERPOWERED' if math.sqrt(b + c) <= 1.645 else '')}")
            cb, cc, cn = con[a]
            if cn:
                print(f'POOLED CONFIRM {a} over {len(files)} runs: mind {cb} / rival {cc} of {cn}   z {z_of(cb, cc):+.2f}')
            for fr, (k, mh, sh, rh) in sorted(gate[a].items()):
                if not k:
                    continue
                print(f'POOLED GATE-WO {a} top {fr:>4.0%}: k {k}   mind {mh} ({mh / k:.4f})   share {sh} ({sh / k:.4f})   random {rh} ({rh / k:.4f})')
            for sub in ('all', 'walk_only'):
                ob, oc, on = oth[a][sub]
                if not on:
                    continue
                print(f"POOLED OTHER {sub} {a} over {len(files)} runs: this {ob} / other {oc}   z {z_of(ob, oc):+.2f}   ({ob + oc} discordant of {on}){('   TOO FEW DISCORDANT TO READ' if ob + oc < 10 else '')}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))