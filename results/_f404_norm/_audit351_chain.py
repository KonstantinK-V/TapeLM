"""THE CEILING OF A CHAIN. Does a second step reach what one step cannot?

WHY THIS AND WHY NOW. Kostya's diagnosis, and it is right: we have laid the parts on the floor
and pressed a pedal that is lying next to the engine. The pipeline IS assembled - text, write
path, tape, question, walk, worlds, Phi, answer, reward, weights - but it is ASSEMBLED FOR ONE
SHOT. There is no loop:

    question k+1 knows nothing of question k          no state
    the answer goes nowhere - never written, never read again
    the reward is terminal and per-question           no consequence propagates
    WE choose the hole, at random                     an exam item, not a situation

AND EVERY CLOSED RESULT WAS CLOSED AS A SINGLE-SHOT OPERATION. Composition, in any mind, is
"answer A, then ask B through A" - we tried it as one joint world and it factorised by identity.
Generation is a partial result recombined with the store, iteratively. Revision literally
requires a second step. We have been diagnosing an engine with its crankshaft on the bench.

THE CLAIM THIS AUDIT TESTS. Generation may not need a new relation on the tape at all. It needs
a SECOND STEP. Substitution can only hand back what already stood where A stands - but a value
reached at step one is NOT at the question's place, so using IT as the next lens reaches things
the question's own paradigm never contained. Two rankings chained produce what neither can.

    step 1   the question's own rows -> what stands with them
    step 2   THOSE values -> what stands with THEM

THIS IS NOT L3. L3 intersected two lenses that were both the question's OWN rows - same
paradigm, narrowed. A chain leaves the paradigm at the first hop, which is the whole point.

AND IT IS NOT NEW EVIDENCE-FREE HOPE. 322 already measured depth on the WALK: reachable 0.54 at
depth 2 against 0.12 at depth 1 - the largest single movement of reach this project has ever
recorded. It was closed because CONFIRM collapsed (42/312) and because an honest rival read
2-3%. BOTH OF THOSE ARE SINGLE-SHOT FRAMING ARTEFACTS: CONFIRM broke because ONE decision had to
serve both "answer at home" and "chase depth", and the rival was weak because a one-hop rival
cannot follow a two-hop path. Neither says the chain does not reach.

WHAT IS MEASURED, all with the question's own place and lines excluded at every hop:

    reach1        the truth in step one's offer                    (today's ceiling)
    reach2        the truth in step two's offer
    chain_only    reached at two hops and NOT at one               THE NUMBER THAT DECIDES
    oracle2       the truth reachable by SOME path of length two   the ceiling of a perfect
                                                                   chooser, which is the job
                                                                   the mind would be given
    paths         how many paths there are to choose between       the size of that job

If chain_only is large and paths is not astronomical, the mind has a well-posed problem it has
never been given, and every single-shot closure has to be re-read as conditional. If chain_only
is small, chaining substitution does not generate either, and the substrate verdict stands.

    python _audit351_chain.py
    python _audit351_chain.py --branch 8 --topm 8      # budget-matched to today's offer
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage351_chain.json')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=2)
    ap.add_argument('--addresses', type=int, default=1500)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8, help='values a lens offers, per hop')
    ap.add_argument('--branch', type=int, default=8, help="how many of step one's values are followed. The chooser's fan-out")
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--max-questions', type=int, default=2000)
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
    qs = [(j, i) for j in range(P) for i in range(len(vals_at[j])) if len(vals_at[j]) >= 2]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]
    c = Counter()
    paths, s1n, s2n = ([], [], [])
    for j, i in qs:
        truth = vals_at[j][i]
        own_c = Counter(vals_at[j])
        own_c[truth] -= 1
        if own_c[truth] <= 0:
            del own_c[truth]
        lens = list(own_c)[:6]
        if not lens:
            continue
        c['n'] += 1
        mine = set(slots_at[j])
        my_places = {j}

        def offer(v):
            """v's co-occurrence with THIS place subtracted - the same exclusion every other
            audit makes, applied at every hop so a chain cannot re-enter the question."""
            out = Counter()
            here = {toks[s]: 0 for s in mine}
            for w, n in co(v).items():
                if w in here:
                    n -= sum((1 for s in mine if toks[s] == w))
                if n > 0 and w != v:
                    out[w] = n
            return out
        s1 = Counter()
        for v in lens:
            s1 += offer(v)
        top1 = [w for w, _n in s1.most_common(args.topm)]
        s1n.append(len(s1))
        reach1 = truth in set(top1)
        full1 = [w for w, _n in s1.most_common(args.branch)]
        s2 = Counter()
        oracle = False
        np_ = 0
        for v1 in full1:
            o2 = offer(v1)
            np_ += 1
            for w, n in o2.most_common(args.topm):
                s2[w] += n
                if w == truth:
                    oracle = True
        top2 = [w for w, _n in s2.most_common(args.topm)]
        s2n.append(len(s2))
        paths.append(np_)
        reach2 = truth in set(top2)
        c['reach1'] += reach1
        c['reach2'] += reach2
        c['union'] += reach1 or reach2
        c['chain_only'] += reach2 and (not reach1)
        c['one_only'] += reach1 and (not reach2)
        c['oracle2'] += oracle
        c['oracle_or_1'] += oracle or reach1
    n = max(1, c['n'])
    rep = {'bytes': args.bytes, 'window_lines': args.window_lines, 'places': P, 'questions': c['n'], 'topm': args.topm, 'branch': args.branch, 'reach1': c['reach1'] / n, 'reach2': c['reach2'] / n, 'union': c['union'] / n, 'chain_only': c['chain_only'] / n, 'one_only': c['one_only'] / n, 'oracle2': c['oracle2'] / n, 'oracle_or_1': c['oracle_or_1'] / n, 'paths': sum(paths) / max(1, len(paths)), 'step1_size': sum(s1n) / max(1, len(s1n)), 'step2_size': sum(s2n) / max(1, len(s2n))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape    {P} places, {c['n']} questions, topm {args.topm}, branch {args.branch}")
    print(f"ONE     reach {rep['reach1']:.4f}   offer {rep['step1_size']:.0f}")
    print(f"TWO     reach {rep['reach2']:.4f}   offer {rep['step2_size']:.0f}   paths {rep['paths']:.1f}")
    print(f"APART   only by TWO {rep['chain_only']:.4f}   only by ONE {rep['one_only']:.4f}   union {rep['union']:.4f}")
    print(f"ORACLE  some 2-hop path lands on the truth {rep['oracle2']:.4f}   with one-hop {rep['oracle_or_1']:.4f}")
    if rep['oracle_or_1'] > rep['reach1'] + 0.05:
        print(f"\nTHE CHAIN REACHES. A perfect chooser over {rep['paths']:.0f} paths would see {rep['oracle_or_1']:.4f} against one step's {rep['reach1']:.4f}. That is a well-posed problem the mind has never been given, and every single-shot closure - composition, generation, revision - has to be re-read as conditional on a loop that does not exist yet.")
    else:
        print('\nCHAINING DOES NOT GENERATE EITHER: a second hop over the same relation reaches no more than the first. Substitution is closed as a substrate, one step or two.')
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())