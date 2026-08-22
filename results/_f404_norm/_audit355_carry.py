"""IS THERE ANYTHING TO CARRY BETWEEN QUESTIONS THAT IS NOT A FACT?

354 CLOSED THE OBVIOUS KIND OF MEMORY. A previous answer used as a lens buys nothing: pooled it
LOSES (-0.038), and chosen perfectly by an oracle it gains +0.0025 / -0.0004 against a null that
writes WRONG answers back. A trained level cannot beat its own oracle, so no amount of
architecture makes remembered CONTENT pay. That is the substrate's answer and it is final.

BUT CONTENT IS NOT THE ONLY THING A MIND CARRIES, and this project of all projects should say
so, because its whole thesis is the split: FACTS LIVE ON THE TAPE, POLICY LIVES IN PHI. 354
measured a memory of facts. A memory of READING - "in this situation the tape is thin, do not
speak" - holds no facts, breaks no invariant, and has never been measured.

WHAT WOULD MAKE IT REAL. Difficulty must be a property of the SITUATION, not only of the
question. And it must be a property Phi CANNOT ALREADY SEE in the question in front of it -
otherwise a carried state is redundant and a level added for it is dead weight.

    lift        p(reach | the previous question in this situation reached)
                  minus p(reach | it did not)
    null        the same, with the "previous" question taken from ANOTHER situation
    WITHIN      the same, computed inside strata of the CURRENT question's own difficulty
                (bins of the top-1 minus top-2 count margin of its own offer - a count, no
                heuristic). THIS IS THE NUMBER THAT DECIDES: it is the part of the situation
                that the question in front of Phi does not already announce.

  GATE  within-stratum lift > 0.05 with the cross-situation null near zero. Then a carried
        state is worth a level, and it is a level that holds no facts.
  If WITHIN is flat while raw lift is not, situations do differ but the current question
  already says so, and the state is redundant - build nothing.

    python _audit355_carry.py --session 6 --session-lines 40
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage355_carry.json')

def rate(rows, sel):
    hit = [r for r in rows if sel(r)]
    return (sum((r['reach'] for r in hit)) / len(hit), len(hit)) if hit else (0.0, 0)

def lift(rows, key):
    """p(reach | previous reached) - p(reach | previous did not), and the two supports."""
    a, na = rate(rows, lambda r: r[key] == 1)
    b, nb = rate(rows, lambda r: r[key] == 0)
    return (a - b, a, na, b, nb)

def within(rows, key, bins):
    """the same difference computed INSIDE strata of the current question's own margin, then
    pooled by stratum size. A situation effect that survives this is one the question in front
    of Phi does not already announce."""
    ms = sorted((r['margin'] for r in rows))
    if not ms:
        return 0.0
    cuts = [ms[int(len(ms) * (i + 1) / bins) - 1] for i in range(bins)]
    tot, wsum = (0.0, 0)
    for i, hi in enumerate(cuts):
        lo = cuts[i - 1] if i else -1e+18
        sub = [r for r in rows if lo < r['margin'] <= hi or (i == 0 and r['margin'] <= hi)]
        d, _a, na, _b, nb = lift(sub, key)
        if na and nb:
            tot += d * (na + nb)
            wsum += na + nb
    return (tot / wsum if wsum else 0.0, wsum)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=2)
    ap.add_argument('--addresses', type=int, default=1500)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--session', type=int, default=6)
    ap.add_argument('--session-lines', type=int, default=40)
    ap.add_argument('--bins', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--sessions', type=int, default=600)
    ap.add_argument('--corpus', default=str(WIKI))
    args = ap.parse_args()
    text = Path(args.corpus).open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.window_lines:
        by_line0 = tframes._by_line(keep, owner)
        start = rng.randrange(max(1, len(lines)))
        acc = defaultdict(list)
        for d in range(args.window_lines):
            for k, i in by_line0.get((start + d) % len(lines), ()):
                acc[k].append(i)
        keep = [(k, sorted(v)) for k, v in acc.items() if len({toks[i] for i in v}) >= args.min_fillers]
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print('no tape')
        return 1
    slots_at = [list(ps) for _k, ps in keep]
    vals_at = [[toks[s] for s in ps] for ps in slots_at]
    P = len(vals_at)
    place_of = {}
    for j, ps in enumerate(slots_at):
        for s in ps:
            place_of[s] = j
    where = defaultdict(list)
    for j, ps in enumerate(slots_at):
        for s in ps:
            where[toks[s]].append(s)
    line_of = {j: min((owner[s] for s in ps)) for j, ps in enumerate(slots_at)}
    places_on = defaultdict(list)
    for j, ln in line_of.items():
        places_on[ln].append(j)
    lns = sorted(places_on)
    cooc = {}

    def co(v):
        c = cooc.get(v)
        if c is None:
            c = Counter()
            for s in where[v]:
                for s2 in slots_at[place_of[s]]:
                    if toks[s2] != v:
                        c[toks[s2]] += 1
            cooc[v] = c
        return c
    sessions = []
    for _ in range(args.sessions):
        if args.session_lines and len(lns) > 1:
            a = rng.randrange(len(lns))
            pool = [j for ln in lns[a:a + args.session_lines] for j in places_on[ln]]
        else:
            pool = list(range(P))
        pool = [j for j in pool if len(vals_at[j]) >= 2]
        if len(pool) < 2:
            continue
        rng.shuffle(pool)
        seq = []
        for j in pool[:args.session]:
            i = rng.randrange(len(vals_at[j]))
            truth = vals_at[j][i]
            own_c = Counter(vals_at[j])
            own_c[truth] -= 1
            if own_c[truth] <= 0:
                del own_c[truth]
            lens = list(own_c)[:6]
            if not lens:
                continue
            mine = set(slots_at[j])
            here = Counter((toks[s] for s in mine))
            off = Counter()
            for v in lens:
                for w, n in co(v).items():
                    if w in here:
                        n -= here[w]
                    if n > 0 and w != v:
                        off[w] += n
            top = off.most_common(args.topm)
            margin = top[0][1] - top[1][1] if len(top) > 1 else top[0][1] if top else 0
            seq.append({'reach': int(truth in {w for w, _n in top}), 'margin': margin})
        if len(seq) >= 2:
            sessions.append(seq)
    rows = []
    for si, seq in enumerate(sessions):
        for k in range(1, len(seq)):
            rows.append({'reach': seq[k]['reach'], 'margin': seq[k]['margin'], 'prev': seq[k - 1]['reach'], 'sid': si, 'pos': k})
    for r in rows:
        for _try in range(8):
            o = rng.randrange(len(sessions))
            if o != r['sid'] and len(sessions[o]) > r['pos']:
                r['prev_null'] = sessions[o][r['pos'] - 1]['reach']
                break
        else:
            r['prev_null'] = r['prev']
    d, a, na, b, nb = lift(rows, 'prev')
    dn, an, nan, bn, nbn = lift(rows, 'prev_null')
    wi, wcov = within(rows, 'prev', args.bins)
    wn, _wncov = within(rows, 'prev_null', args.bins)
    base = sum((r['reach'] for r in rows)) / max(1, len(rows))
    rep = {'places': P, 'sessions': len(sessions), 'pairs': len(rows), 'topm': args.topm, 'session': args.session, 'session_lines': args.session_lines, 'bins': args.bins, 'base': base, 'lift': d, 'p_after_hit': a, 'n_after_hit': na, 'p_after_miss': b, 'n_after_miss': nb, 'lift_null': dn, 'within': wi, 'within_null': wn, 'within_cov': wcov}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f'tape     {P} places, {len(sessions)} situations, {len(rows)} pairs, base reach {base:.4f}')
    print(f'LIFT     after a hit {a:.4f} (n {na})   after a miss {b:.4f} (n {nb})   {d:+.4f}')
    print(f'NULL     previous taken from ANOTHER situation                        {dn:+.4f}')
    print(f"WITHIN   inside strata of the question's OWN margin  {wi:+.4f}   null {wn:+.4f}   (coverage {wcov} of {len(rows)} pairs)")
    if wcov < 0.2 * len(rows):
        print('\nVOID, NOT FLAT. The strata almost never hold both a hit and a miss, so there is no within-stratum comparison to read. Re-run with fewer --bins before believing any WITHIN number.')
    elif wi - abs(wn) > 0.05:
        print(f"\nTHERE IS SOMETHING TO CARRY, AND IT IS NOT A FACT. A situation's difficulty is {wi:+.4f} predictable from the previous question BEYOND what the current question's own margin says. A state that carries it holds no facts and breaks no invariant - it changes HOW the tape is read, which is exactly Phi's half of the split.")
    elif d - abs(dn) > 0.05:
        print(f"\nSITUATIONS DIFFER BUT THE QUESTION ALREADY SAYS SO. Raw lift {d:+.4f} collapses to {wi:+.4f} inside strata of the current question's own margin. A carried state would be redundant with what Phi already sees. Build nothing.")
    else:
        print("\nNOTHING TO CARRY. Answerability is not a property of the situation at all: one question's outcome says nothing about the next, in or out of a situation. Memory is closed on the substrate for CONTENT (354) and for READING (355).")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())