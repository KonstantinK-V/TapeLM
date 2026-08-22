"""A TRUTH SIGNAL THAT IS NOT "THE TOKEN THAT STOOD THERE". The missing criterion.

WHY THIS IS THE BLOCKER. Four traits of a real mind were named at step 0. Three were tested and
closed. The fourth - PRODUCE WHAT WAS NEVER OFFERED - was never tested at all, and the reason is
not the substrate: IT IS THE EXAM. Every reward in this project is "say the token that was in
the hole". A produced value that is not that token scores zero whether it is brilliant or
gibberish, so generation cannot fail here - IT CANNOT BE MEASURED. No architecture fixes that.

WHAT A CRITERION WOULD HAVE TO DO. Score a candidate WITHOUT knowing the answer, and keep
working for candidates THE OFFER NEVER PROPOSED - otherwise it is just the offer's own count
under another name, and a count cannot score a value it has never seen at such a place.

THE CANDIDATE CRITERION, and it is a count, not a heuristic: A VALUE IS WORTH SAYING IF SAYING
IT CONNECTS. Write c into this hole and the tape gains a mention of c at this place; c is then
offered wherever this place's fillers are a lens. So

    U(c) = how many positions elsewhere hold c AND sit at a place that already shares a filler
           with this one.

Nothing about the truth enters it. It is defined for ANY value in the vocabulary, including
values the offer never proposed - which is the whole point.

WHERE IT IS TESTED, and this is chosen to be the hard place: THE HOLES NEITHER CHANNEL CAN
REACH. Not in the walk's offer, not among the place's own fillers. That is precisely the subset
where a mind would have to PRODUCE, and where every number in this project is currently zero.
The truth is dropped into a pool of random vocabulary and U has to find it.

    AUC_U      the truth against `--pool` random values, scored by U
    AUC_FREQ   the same by raw corpus frequency        THE RIVAL - the dumb label-free score
    AUC_NULL   U computed against a RANDOM OTHER PLACE's fillers - same shape, wrong place.
               If this is not near 0.5 the signal is about the vocabulary, not about the hole.

  GATE  AUC_U - AUC_FREQ > 0.05 AND AUC_U - AUC_NULL > 0.05.
        Then there is a label-free criterion that reaches past the offer, generation becomes a
        measurable task for the first time, and the fourth trait is open.
        If AUC_U ~ AUC_FREQ, "useful" means "common" and we have learned that the criterion is
        the same count we already had. If AUC_U ~ 0.5, the fourth trait is closed for want of a
        criterion, not for want of a mind - and that is worth knowing exactly as much.

    python _audit363_useful.py
    python _audit363_useful.py --min-fillers 1 --pool 128
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage363_useful.json')

def auc(pos, negs):
    """one positive against many: the fraction of distractors it outranks, ties half."""
    if not negs:
        return 0.5
    win = sum((1.0 for x in negs if pos > x)) + 0.5 * sum((1.0 for x in negs if pos == x))
    return win / len(negs)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--pool', type=int, default=64, help='random distractors per question')
    ap.add_argument('--match-freq', action='store_true', help="draw distractors from the truth's own frequency band (same power of two, widened until the band holds enough). The first run showed the uniform pool hands FREQ an AUC of 0.70 for free - random vocabulary is mostly hapaxes, so 'commoner than noise' already finds the truth. Matching deletes that channel BY CONSTRUCTION: what survives is the part of U that is about THIS hole, which is the only part that could ever score generation")
    ap.add_argument('--cap', type=int, default=500, help='positions read per value, bounded so one frequent token cannot cost a minute. Reported.')
    ap.add_argument('--neigh-cap', type=int, default=4000, help='neighbourhood places scanned when U is used FORWARD as a third channel. Reported; raise it if the top quartile looks truncated')
    ap.add_argument('--max-questions', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--corpus', default=str(WIKI))
    args = ap.parse_args()
    text = Path(args.corpus).open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    keep, toks, _owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if not keep:
        print('no tape')
        return 1
    places = [list(ps) for _a, ps in keep]
    place_of, where = ({}, defaultdict(list))
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
            where[toks[s]].append(s)
    fillers = [set((toks[s] for s in ps)) for ps in places]
    places_of_value = defaultdict(set)
    for pid, f in enumerate(fillers):
        for v in f:
            places_of_value[v].add(pid)
    vocab = sorted(where)
    freq = {v: len(where[v]) for v in vocab}
    band_of = {v: freq[v].bit_length() for v in vocab}
    by_band = defaultdict(list)
    for v in vocab:
        by_band[band_of[v]].append(v)
    cooc_cache = {}

    def offer_of(pid, truth):
        """today's substitution offer, exactly as every audit builds it"""
        own = Counter((toks[x] for x in places[pid]))
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        lens = list(own)[:6]
        if not lens:
            return set()
        ban = Counter((toks[x] for x in places[pid]))
        off = Counter()
        for v in lens:
            c = cooc_cache.get(v)
            if c is None:
                c = Counter()
                for s2 in where[v]:
                    for s3 in places[place_of[s2]]:
                        if toks[s3] != v:
                            c[toks[s3]] += 1
                cooc_cache[v] = c
            for w, n in c.items():
                n -= ban.get(w, 0)
                if n > 0 and w != v:
                    off[w] += n
        return {w for w, _n in off.most_common(args.topm)}

    def useful(c, neigh, home_pid):
        """positions holding c that sit at a place already sharing a filler with this one.
        The question's OWN place never counts - writing an answer cannot corroborate itself.

        Returns BOTH counts. Positions is the raw one; DISTINCT PLACES is the same measure with
        the frequency channel largely removed by construction - a value repeated nine times at
        one place connects to ONE place, and connecting broadly is what "useful" was supposed
        to mean. The matched run showed FREQ still at 0.56-0.58, so the coarse bit_length bands
        do not delete frequency entirely; this deletes what they left."""
        n, seen = (0, set())
        for s2 in where[c][:args.cap]:
            pid2 = place_of[s2]
            if pid2 != home_pid and pid2 in neigh:
                n += 1
                seen.add(pid2)
        return (n, len(seen))

    def draw_pool(truth, others):
        if args.match_freq:
            src, w = (list(by_band[band_of[truth]]), 0)
            while len(src) < args.pool + 8 and w < 20:
                w += 1
                src += by_band.get(band_of[truth] - w, []) + by_band.get(band_of[truth] + w, [])
        else:
            src = vocab
        out = []
        while len(out) < args.pool:
            v = src[rng.randrange(len(src))]
            if v != truth and v not in others:
                out.append(v)
        return out
    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    c = Counter()
    au, af, an, ap_, anp, aw = ([], [], [], [], [], [])
    rows = []
    for s in qs:
        if c['n'] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        others = {toks[x] for x in places[pid] if x != s}
        c['seen'] += 1
        off_here = offer_of(pid, truth)
        if truth in others or truth in off_here:
            continue
        c['n'] += 1
        neigh = set()
        for v in others:
            neigh |= places_of_value[v]
        neigh.discard(pid)
        if not neigh:
            c['no_neighbourhood'] += 1
            continue
        opid = rng.randrange(len(places))
        for _t in range(4):
            if opid != pid and len(fillers[opid]) == len(others):
                break
            opid = rng.randrange(len(places))
        nneigh = set()
        for v in fillers[opid]:
            nneigh |= places_of_value[v]
        nneigh.discard(pid)
        pool = draw_pool(truth, others)
        nl = sorted(neigh)
        if len(nl) > args.neigh_cap:
            nl = [nl[i] for i in range(0, len(nl), max(1, len(nl) // args.neigh_cap))]
            c['neigh_capped'] += 1
        upl_all, wpl_all, spl_all = (Counter(), Counter(), Counter())
        for pid2 in nl:
            ov = len(fillers[pid2] & others)
            for v in fillers[pid2]:
                upl_all[v] += 1
                wpl_all[v] += ov
                if ov >= 2:
                    spl_all[v] += 1
        for v in list(others) + list(off_here):
            upl_all.pop(v, None)
            wpl_all.pop(v, None)
            spl_all.pop(v, None)
        c['u_reach'] += truth in {w for w, _n in upl_all.most_common(args.topm)}
        c['uw_reach'] += truth in {w for w, _n in wpl_all.most_common(args.topm)}
        c['us_reach'] += truth in {w for w, _n in spl_all.most_common(args.topm)}
        c['us_offer'] += len(spl_all)
        c['u_offer'] += len(upl_all)
        aw.append(auc(wpl_all.get(truth, 0), [wpl_all.get(v, 0) for v in pool]))
        if len(neigh) <= args.neigh_cap:
            want = sum((1 for pid2 in neigh if truth in fillers[pid2]))
            if upl_all[truth] != want:
                raise SystemExit('forward U != neighbourhood place-count of the truth')
        ut, upl = useful(truth, neigh, pid)
        pu = [useful(v, neigh, pid) for v in pool]
        nt, npl = useful(truth, nneigh, pid)
        pn = [useful(v, nneigh, pid) for v in pool]
        au.append(auc(ut, [x[0] for x in pu]))
        ap_.append(auc(upl, [x[1] for x in pu]))
        af.append(auc(freq[truth], [freq[v] for v in pool]))
        an.append(auc(nt, [x[0] for x in pn]))
        anp.append(auc(npl, [x[1] for x in pn]))
        c['truth_nonzero'] += ut > 0
        c['pool_all_zero'] += all((x[0] == 0 for x in pu))
        c['null_all_zero'] += nt == 0 and all((x[0] == 0 for x in pn))
        c['p1'] += all((ut > x[0] for x in pu))
        rank = 1 + sum((1 for x in pu if x[0] >= ut))
        c['p5'] += rank <= 5
        rows.append((len(neigh), ap_[-1], 1.0 if rank <= 5 else 0.0, 1.0 if upl > 0 else 0.0))
    n = max(1, len(au))
    rep = {'lines': len(lines), 'places': len(places), 'vocab': len(vocab), 'seen': c['seen'], 'unreachable_questions': c['n'], 'scored': len(au), 'no_neighbourhood': c['no_neighbourhood'], 'pool': args.pool, 'cap': args.cap, 'min_fillers': args.min_fillers, 'match_freq': bool(args.match_freq), 'auc_useful': sum(au) / n, 'auc_freq': sum(af) / n, 'auc_null': sum(an) / n, 'auc_places': sum(ap_) / n, 'auc_places_null': sum(anp) / n, 'truth_nonzero': c['truth_nonzero'] / n, 'pool_all_zero': c['pool_all_zero'] / n, 'null_all_zero': c['null_all_zero'] / n, 'p_at_1': c['p1'] / n, 'p_at_5': c['p5'] / n, 'chance_p_at_5': 5.0 / (args.pool + 1), 'chance_p_at_1': 1.0 / (args.pool + 1), 'window_lines': args.window_lines}
    rows.sort(key=lambda r: r[0])
    q = max(1, len(rows) // 4)

    def split(ch):
        d = [r for r in ch if r[3] > 0]
        s_ = [r for r in ch if r[3] == 0]
        return (sum((r[1] for r in d)) / len(d) if d else float('nan'), sum((r[1] for r in s_)) / len(s_) if s_ else float('nan'))
    _syn = [(0, 0.9, 0.0, 1.0), (0, 0.9, 0.0, 1.0), (0, 0.4, 0.0, 0.0), (0, 0.4, 0.0, 0.0)]
    _ad, _as = split(_syn)
    _p = sum((r[3] for r in _syn)) / len(_syn)
    _mean = sum((r[1] for r in _syn)) / len(_syn)
    if abs(_p * _ad + (1.0 - _p) * _as - _mean) > 1e-12:
        raise SystemExit('recombination identity failed on the synthetic case')
    rep['by_neighbourhood'] = [{'quartile': i + 1, 'n': len(ch), 'neigh_lo': ch[0][0], 'neigh_hi': ch[-1][0], 'auc_places': sum((r[1] for r in ch)) / len(ch), 'p_at_5': sum((r[2] for r in ch)) / len(ch), 'truth_nonzero': sum((r[3] for r in ch)) / len(ch), 'auc_when_defined': split(ch)[0], 'auc_when_silent': split(ch)[1]} for i, ch in enumerate([rows[0:q], rows[q:2 * q], rows[2 * q:3 * q], rows[3 * q:]]) if ch]
    ov_d, ov_s = split(rows)
    rep['auc_when_defined'] = ov_d
    rep['auc_when_silent'] = ov_s
    rep['u_reach'] = c['u_reach'] / n
    rep['uw_reach'] = c['uw_reach'] / n
    rep['us_reach'] = c['us_reach'] / n
    rep['us_offer'] = c['us_offer'] / n
    rep['auc_weighted'] = sum(aw) / max(1, len(aw))
    rep['u_offer'] = c['u_offer'] / n
    rep['neigh_capped'] = c['neigh_capped'] / n
    rep['neigh_cap'] = args.neigh_cap
    rep['over_freq'] = rep['auc_useful'] - rep['auc_freq']
    rep['over_null'] = rep['auc_useful'] - rep['auc_null']
    rep['places_over_null'] = rep['auc_places'] - rep['auc_places_null']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f'tape     {len(places)} places, {len(vocab)} values, min_fillers {args.min_fillers}   window {len(lines)}   cap {args.cap}')
    print(f"SUBSET   {c['n']} of {c['seen']} questions reach NEITHER channel; {len(au)} scored, {c['no_neighbourhood']} had no neighbourhood")
    print(f"USEFUL   AUC {rep['auc_useful']:.4f}   (the truth against {args.pool} random values)")
    print(f"FREQ     AUC {rep['auc_freq']:.4f}   over freq {rep['over_freq']:+.4f}")
    print(f"NULL     AUC {rep['auc_null']:.4f}   over null {rep['over_null']:+.4f}   (a random other place's neighbourhood)")
    print(f"PLACES   AUC {rep['auc_places']:.4f}   null {rep['auc_places_null']:.4f}   over null {rep['places_over_null']:+.4f}   (distinct places, not positions)")
    print(f"ALIVE?   truth scores nonzero {rep['truth_nonzero']:.4f}   whole pool zero {rep['pool_all_zero']:.4f}   null all zero {rep['null_all_zero']:.4f}")
    print(f"USABLE   p@1 {rep['p_at_1']:.4f} (chance {rep['chance_p_at_1']:.4f})   p@5 {rep['p_at_5']:.4f} (chance {rep['chance_p_at_5']:.4f})   - what a REWARD would have to live on")
    print(f"DEFINED  AUC|defined {rep['auc_when_defined']:.4f}   AUC|silent {rep['auc_when_silent']:.4f}   (label-conditioned diagnostic, never a gate)")
    ch8 = args.topm / max(1.0, rep['u_offer'])
    print(f"CHANNEL  U forward: reach {rep['u_reach']:.4f}   overlap-weighted {rep['uw_reach']:.4f}   of the UNREACHABLE holes at top-{args.topm}")
    print(f"         out of {rep['u_offer']:.0f} candidate values - chance {ch8:.4f}, so {rep['u_reach'] / max(ch8, 1e-09):.0f}x and {rep['uw_reach'] / max(ch8, 1e-09):.0f}x   ({rep['neigh_capped']:.2f} capped at {args.neigh_cap})")
    chs = args.topm / max(1.0, rep['us_offer'])
    print(f"         STRICT (neighbours sharing >= 2 fillers): reach {rep['us_reach']:.4f} out of {rep['us_offer']:.0f} values - chance {chs:.4f}, {rep['us_reach'] / max(chs, 1e-09):.0f}x")
    print(f"         AUC by overlap weight {rep['auc_weighted']:.4f} against plain {rep['auc_places']:.4f}")
    print('BY NEIGHBOURHOOD (split on a number known before the answer):')
    for b in rep['by_neighbourhood']:
        print(f"   q{b['quartile']}  places {b['neigh_lo']:>5}-{b['neigh_hi']:<5} n {b['n']:>4}   AUC {b['auc_places']:.4f}   p@5 {b['p_at_5']:.4f}   defined {b['truth_nonzero']:.4f}   AUC|defined {b['auc_when_defined']:.4f}   AUC|silent {b['auc_when_silent']:.4f}")
    dfn = [b['auc_when_defined'] for b in rep['by_neighbourhood']]
    if dfn and max(dfn) - min(dfn) < 0.08:
        print('   -> AUC|defined is FLAT across quartiles: the trend is COVERAGE, not discrimination. The criterion is as sharp on a thin hole as on a thick one, it is simply SILENT more often - and silence is fixable with density, sharpness would not be. (label-conditioned diagnostic, never a gate)')
    if args.match_freq and abs(rep['auc_freq'] - 0.5) > 0.08:
        print('\nMATCHING FAILED - FREQ is still ' + f"{rep['auc_freq']:.4f}" + ' on a pool meant to erase it. Widen the bands or shrink the pool before reading anything.')
    elif args.match_freq:
        print('\nRead BY NEIGHBOURHOOD: rich-and-sharp can be a task on that share; flat 2x-over-chance closes the fourth trait for want of a measure.')
    elif rep['over_freq'] > 0.05 and rep['over_null'] > 0.05:
        print(f"\nTHERE IS A CRITERION. On the holes NEITHER channel reaches - where every number this project has printed is zero - a label-free count finds the truth in a pool of {args.pool} at AUC {rep['auc_useful']:.4f}, beating raw frequency and its own wrong-place null. Generation can be SCORED, which is the thing it has always lacked, and the fourth trait becomes a task instead of a wish.")
    elif rep['over_null'] > 0.05:
        print(f"\n'USEFUL' MEANS 'COMMON'. The signal is real against its null but does not beat raw frequency ({rep['auc_freq']:.4f}). It is the count we already had wearing a new name, and it cannot tell a good novel answer from a frequent one.")
    else:
        print('\nNO CRITERION. Connecting to the neighbourhood says nothing about being right. The fourth trait is closed FOR WANT OF A MEASURE, not for want of a mind: on these holes there is no label-free way to tell a produced answer from noise, so no reward can be written and no architecture can be blamed.')
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())