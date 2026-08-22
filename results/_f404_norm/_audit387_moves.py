"""THE CEILING OF CHOOSING A MOVE. Measured before a better chooser is built.

385 and 386 both failed their gate, and both failed the same way: the more the mind used a move
other than `step`, the worse the arm. 386 added `move_hit` and it named the fault precisely - on
three seeds of four the hit of `share` was BELOW the hit of `step`, so the mind was picking the
second move on the questions where that move is worse. The probe is one row, and one row is
evidently not enough to tell which lane will pay.

BUT THE LOSS IS NOT THE PRICE OF COMMITTING. The merged offer is the two lanes interleaved and
cut at eight, so `merged` is a SUBSET of `step_top8 | share_top8`. A perfect move chooser
therefore reaches at least as much as the interleave, always, by construction - while the
measured reach FELL on three seeds of four (.6525 -> .5943). Every point of that gap is chooser
error, not the cost of choosing. So there is headroom above the interleave, and the question is
how much.

    reach_step      the truth in the fingerprint walk's eight
    reach_share     the truth in the connect lane's eight
    reach_merged    the truth in the interleaved eight            TODAY
    reach_oracle    the truth in EITHER lane                      A PERFECT CHOOSER
    only_share      the truth in the share lane and NOT in step   WHERE THE MOVE DECIDES
    overlap         mean Jaccard of the two eights                ARE THEY EVEN DIFFERENT
    reach_random    K random places                               the floor

THE SUSPECT, DECLARED BEFORE THE RUN. The arm runs with `fp=fillers`, so a place's fingerprint
IS its bag of fillers - and the cosine between two filler profiles is ZERO unless they share a
filler. `connect` walks places that share a filler, weighted by how many. So both lanes draw
from THE SAME NEIGHBOUR SET and differ only in how they RANK it: cosine of the profile against
count of the overlap. If that is right, `only_share` is small, the two moves are one move with
two orderings, and no chooser can pay - the fix would be a move that reaches somewhere the walk
cannot, not a better way of picking between two views of the same neighbourhood.

  GATE  oracle - merged > 0.05 AND only_share > 0.05.
        Then a perfect choice is worth having and the next lever is the CHOOSER - the one-row
        probe is the thing to replace. If only_share is small the lanes are redundant, the move
        as an output space is closed on this tape, and what is needed is a genuinely different
        move.

  Measured on the population where reach matters: the truth is NOT among the question's own
  values. Own values are excluded from both lanes, the question's own place from both
  neighbourhoods, and the hidden position from everything.

    python _audit387_moves.py
    python _audit387_moves.py --window-lines 1600 --places 16
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
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage387_moves.json')
CUTS = (8, 16, 32, 64, 128)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--places', type=int, default=8)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--max-questions', type=int, default=3000)
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
    place_of = {}
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
    prof = [Counter((toks[s] for s in ps)) for ps in places]
    norm = [math.sqrt(sum((c * c for c in pr.values()))) or 1.0 for pr in prof]
    at_value = defaultdict(list)
    for pid, pr in enumerate(prof):
        for v in pr:
            at_value[v].append(pid)

    def neighbours(pid, qprof):
        """every place sharing at least one filler, with the shared-filler COUNT.

        The cosine of two filler profiles is zero unless they share a filler, so this is not a
        shortcut for the walk - it is the walk's entire non-zero support, and the connect
        channel's neighbourhood is the same set. That identity is the thing being measured.

        THE QUERY'S PROFILE IS PASSED IN, AND THE FIRST VERSION OF THIS FUNCTION READ
        `prof[pid]` - WHICH STILL CONTAINS THE HIDDEN TOKEN. The truth was therefore one of the
        keys its own search was run from: every place holding it got overlap credit, and the
        lane then accumulated score for it out of those places. The stage never did this -
        `reach_connect` builds its lens from `slots[:query_row]`, the hidden row excluded - so
        the audit was measuring a channel the arm does not have. Every share number printed
        before this fix is inflated by it.
        """
        ov, dot = (Counter(), Counter())
        for v, c in qprof.items():
            for j in at_value[v]:
                if j != pid:
                    ov[j] += 1
                    dot[j] += c * prof[j][v]
        return (ov, dot)

    def lane_step(pid, own, qprof):
        """the walk's lane IN FULL - every filler of the K nearest places, in place order.

        Returned uncut so the RANK of the truth can be read. The eight the arm actually offers
        is out[:topm]; anything past that is what the cap is throwing away.
        """
        _ov, dot = neighbours(pid, qprof)
        qn = math.sqrt(sum((c * c for c in qprof.values()))) or 1.0
        order = sorted(dot, key=lambda j: (-(dot[j] / (qn * norm[j])), j))[:args.places]
        out, seen = ([], set(own))
        for j in order:
            for v, _c in prof[j].most_common():
                if v not in seen:
                    seen.add(v)
                    out.append(v)
        return out

    def lane_share(pid, own, qprof, norm_by_places=False):
        """connect's lane in full, ranked two ways.

        Note the ASYMMETRY, which is not a bug and is worth having written down: `step` sees the
        fillers of K places, while `share` scores every place that shares a filler at all. The
        two moves were never the same size of read.

        SUM AGAINST SHARE, AND THIS PROJECT HAS MADE THE SAME MISTAKE TWICE ALREADY. 365's rule
        is score(v) = the SUM of the overlaps of every neighbouring place holding v, so a value
        standing at many places accumulates score for being COMMON. The depth block says what
        that costs: the truth is in this lane .78-.88 of the time and sits at mean rank 67 (315
        at w1600), and on this population the truth is RARE by construction - the question is
        here precisely because its answer is not among its own values. 317 found exactly this in
        cons_resolve, measured the raw-count rival at 2/69 = 0.029 against a one-place rule's
        0.222, and fixed it by dividing; 383 found it again in the count rival. Third appearance.

        Dividing by HOW MANY PLACES hold v turns the sum into the MEAN OVERLAP of the places
        where v stands - "on average, how related to my place are the places this value lives
        at". Each place contributes exactly once to both halves, so the ratio is two exact
        counts and nothing is fitted.

        THE CONTROL THIS NEEDS, and it is not optional: dividing by frequency PREFERS RARE
        VALUES, and the truth here is rare. A frequency-matched decoy is ranked the same two ways
        so that "the truth moved up" can be told apart from "everything rare moved up".
        """
        ov, _dot = neighbours(pid, qprof)
        score = Counter()
        for j, o in ov.items():
            for v in prof[j]:
                if v not in own:
                    score[v] += o
        if not norm_by_places:
            return [v for v, _n in score.most_common()]
        return sorted(score, key=lambda v: (-(score[v] / len(at_value[v])), -score[v], v))

    def lane_random(pid, own, k):
        out, seen = ([], set(own))
        pool = len(places)
        for _t in range(args.places):
            j = rng.randrange(pool)
            if j == pid:
                continue
            for v, _c in prof[j].most_common():
                if v not in seen:
                    seen.add(v)
                    out.append(v)
                    if len(out) >= k:
                        return out
        return out
    freq = Counter()
    for ps in places:
        for s in ps:
            freq[toks[s]] += 1
    by_band = defaultdict(list)
    for v, f in freq.items():
        by_band[f.bit_length()].append(v)

    def band_draw(truth, banned):
        src, w = (list(by_band[freq[truth].bit_length()]), 0)
        while len(src) < 16 and w < 20:
            w += 1
            src += by_band.get(freq[truth].bit_length() - w, [])
            src += by_band.get(freq[truth].bit_length() + w, [])
        for _t in range(64):
            v = src[rng.randrange(len(src))]
            if v != truth and v not in banned:
                return v
        return None
    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    c, jac = (Counter(), [])
    for s in qs:
        if c['n'] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in places[pid] if x != s}
        if not own or truth in own:
            continue
        c['n'] += 1
        qprof = Counter((toks[x] for x in places[pid] if x != s))
        st_full = lane_step(pid, own, qprof)
        sh_full = lane_share(pid, own, qprof)
        sn_full = lane_share(pid, own, qprof, norm_by_places=True)
        decoy = band_draw(truth, own)
        c['len_step'] += len(st_full)
        c['len_share'] += len(sh_full)
        for nm, full in (('st', st_full), ('sh', sh_full), ('sn', sn_full)):
            r = full.index(truth) + 1 if truth in full else 0
            if r:
                c[f'in_{nm}'] += 1
                c[f'rank_{nm}'] += r
                for cut in CUTS:
                    c[f'{nm}@{cut}'] += r <= cut
        if decoy is not None:
            c['dn'] += 1
            for nm, full in (('dsh', sh_full), ('dsn', sn_full)):
                r = full.index(decoy) + 1 if decoy in full else 0
                if r:
                    for cut in CUTS:
                        c[f'{nm}@{cut}'] += r <= cut
        u = {v: i for i, v in enumerate(st_full)}
        for i, v in enumerate(sh_full):
            u[v] = min(u.get(v, i), i)
        if truth in u:
            for cut in CUTS:
                c[f'un@{cut}'] += u[truth] + 1 <= cut
        st, sh = (st_full[:args.topm], sh_full[:args.topm])
        merged, seen2 = ([], set())
        for tup in zip_longest(st, sh):
            for e in tup:
                if e is not None and e not in seen2:
                    seen2.add(e)
                    merged.append(e)
        merged = merged[:args.topm]
        S, H, M = (set(st), set(sh), set(merged))
        c['step'] += truth in S
        c['share'] += truth in H
        c['merged'] += truth in M
        c['oracle'] += truth in S | H
        c['only_step'] += truth in S and truth not in H
        c['only_share'] += truth in H and truth not in S
        c['both'] += truth in S and truth in H
        c['random'] += truth in set(lane_random(pid, own, args.topm))
        if S or H:
            jac.append(len(S & H) / max(1, len(S | H)))
    n = max(1, c['n'])
    g = {k: c[k] / n for k in ('step', 'share', 'merged', 'oracle', 'only_step', 'only_share', 'both', 'random')}
    rep = {'corpus': args.corpus, 'lines': len(lines), 'places': len(places), 'questions': c['n'], 'reach_k': args.places, 'topm': args.topm, 'min_fillers': args.min_fillers, **{f'reach_{k}': v for k, v in g.items()}, 'lane_overlap': sum(jac) / max(1, len(jac))}
    rep['headroom'] = g['oracle'] - g['merged']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape     {len(places)} places, {c['n']} questions where the truth is NOT among the question's own values, {args.places} places walked, top-{args.topm}")
    print(f"LANES    step {g['step']:.4f}   share {g['share']:.4f}   random {g['random']:.4f}   mean Jaccard of the two eights {rep['lane_overlap']:.4f}")
    print(f"WHO      both {g['both']:.4f}   only step {g['only_step']:.4f}   only share {g['only_share']:.4f}")
    print(f"CEILING  merged {g['merged']:.4f} (today)   oracle {g['oracle']:.4f} (a perfect move)   headroom {rep['headroom']:+.4f}")
    print(f"DEPTH    lane sizes: step {c['len_step'] / n:.1f}   share {c['len_share'] / n:.1f}   (step sees {args.places} places, share every place sharing a filler)")
    for nm, lab in (('st', 'step '), ('sh', 'share'), ('sn', 'sh/pl'), ('un', 'union')):
        row = '  '.join((f"@{cut} {c[f'{nm}@{cut}'] / n:.4f}" for cut in CUTS))
        extra = ''
        if nm != 'un':
            got = max(1, c[f'in_{nm}'])
            extra = f"   present {c[f'in_{nm}'] / n:.4f}   mean rank when present {c[f'rank_{nm}'] / got:.1f}"
        print(f'  {lab}  {row}{extra}')
        for cut in CUTS:
            rep[f'{nm}_at_{cut}'] = c[f'{nm}@{cut}'] / n
    rep['present_step'], rep['present_share'] = (c['in_st'] / n, c['in_sh'] / n)
    rep['cut_cost'] = rep['un_at_128'] - g['merged']
    nd = max(1, c['dn'])
    for cut in CUTS:
        rep[f'decoy_sum_at_{cut}'] = c[f'dsh@{cut}'] / nd
        rep[f'decoy_norm_at_{cut}'] = c[f'dsn@{cut}'] / nd
    rep['rerank_truth'] = rep['sn_at_8'] - rep['sh_at_8']
    rep['rerank_decoy'] = rep['decoy_norm_at_8'] - rep['decoy_sum_at_8']
    rep['rerank_net'] = rep['rerank_truth'] - rep['rerank_decoy']
    print(f'  decoy  ' + '  '.join((f"@{cut} {rep[f'decoy_sum_at_{cut}']:.4f}->{rep[f'decoy_norm_at_{cut}']:.4f}" for cut in CUTS)) + f"   ({c['dn']} frequency twins, sum -> per-place)")
    print(f"RERANK   share@8 {rep['sh_at_8']:.4f} -> {rep['sn_at_8']:.4f} ({rep['rerank_truth']:+.4f})   decoy {rep['rerank_decoy']:+.4f}   net {rep['rerank_net']:+.4f}")
    if c['n'] < 300:
        print(f"\nVOID, NOT A RESULT. Only {c['n']} questions had the truth outside their own values - too few for a 0.05 gate. Widen --window-lines and read it again.")
    elif rep['headroom'] > 0.05 and g['only_share'] > 0.05:
        print(f"\nCHOOSING IS WORTH SOMETHING AND THE CHOOSER IS THE FAULT. A perfect move reaches {g['oracle']:.4f} against the interleave's {g['merged']:.4f}, and {g['only_share']:.4f} of these questions are reachable ONLY through the second lane. 386's mind gave that back and more by picking on one probe row, so the next lever is the CHOOSER, not the move set: something that reads more of a lane than its first row before committing to it.")
    elif g['only_share'] > 0.05:
        print(f"\nTHE LANES DIFFER AND THE INTERLEAVE ALREADY TAKES BOTH. {g['only_share']:.4f} of these questions are reachable through share alone, so the second move is a real route - but the interleave already reaches {g['merged']:.4f} against a perfect move's {g['oracle']:.4f}, headroom {rep['headroom']:+.4f}. CHOOSING CANNOT ADD WHAT POOLING ALREADY HAS: committing to one lane can only lose the other, which is precisely what 385 and 386 measured. A move pays only where the cap forces the interleave to drop something the chooser would have kept.")
    elif rep['headroom'] > 0.05:
        print(f"\nHEADROOM WITHOUT A REASON TO MOVE. The oracle beats the interleave by {rep['headroom']:+.4f}, but only {g['only_share']:.4f} of questions need the second lane - the gain is the CAP, not the channel: both lanes hold the truth deeper than eight and the interleave cuts one of them short. That is an offer question, and 347 has answered offer questions four times.")
    else:
        print(f"\nTHE TWO LANES ARE ONE LANE WITH TWO ORDERINGS. A perfect move choice is worth {rep['headroom']:+.4f} over the interleave, only {g['only_share']:.4f} of questions are reachable through share alone, at a mean overlap of {rep['lane_overlap']:.4f}. With `fp=fillers` the walk's cosine is non-zero only between places that SHARE A FILLER, which is exactly connect's neighbourhood - so the two moves read the same places in a different order. No chooser can pay here. What is needed is a move that reaches where the walk cannot, and neither of these two is one.")
    if rep['rerank_net'] > 0.05:
        print(f"\nTHE LANE WAS BADLY ORDERED, AND THAT IS A FREE WIN. Ranking connect by the MEAN overlap of the places a value stands at, instead of the SUM, moves the truth into the offered eight {rep['rerank_truth']:+.4f} of the time while lifting a frequency twin only {rep['rerank_decoy']:+.4f} - so this is finding answers, not preferring rare words. Same eight candidates, same cap, same channel: 365's rule has been summing where it should have been dividing, the third time this project has made that exact mistake after 317 and 383.")
    elif rep['rerank_truth'] > 0.05:
        print(f"\nRARITY, NOT RELATEDNESS. The reranking lifts the truth {rep['rerank_truth']:+.4f} into the eight and lifts a frequency-matched decoy {rep['rerank_decoy']:+.4f} - within {abs(rep['rerank_net']):.4f} of the same. Dividing by how many places hold a value simply prefers rare values, and on this population the truth is rare by construction. The gain is the null's, not the channel's.")
    else:
        print(f"\nTHE ORDERING IS NOT THE FAULT. Mean overlap instead of sum moves the truth {rep['rerank_truth']:+.4f} into the eight. The truth sits deep in this lane for some other reason than frequency, and the next question is what the top of the lane is actually full of.")
    if rep['cut_cost'] > 0.05:
        print(f"\nTHE CUT BINDS, NOT THE MATERIAL. Read to depth 128 the two lanes together hold the truth {rep['un_at_128']:.4f} of the time against the offered {g['merged']:.4f} - {rep['cut_cost']:+.4f} is sitting below rank eight. The question stops being WHERE TO LOOK and becomes HOW MANY the mind can weigh, which 368 tested on a merged offer and never on the depth of one lane.")
    else:
        print(f"\nTHE MATERIAL BINDS, NOT THE CUT. Reading to depth 128 adds only {rep['cut_cost']:+.4f} over the offered eight, so the truths the arm misses are not sitting further down the lane - they are not in it at any depth. That is the same wall 373, 375 and 376 each reached from a different side, and no arrangement of the offer moves it.")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())