"""THE TAPE HAS NEVER BEEN ALLOWED TO HOLD A FACT. Two filters, and both were deliberate.

358 FAILED ITS GATE AND TOLD US WHY. Ingestion moved everything in the right direction and
nothing far enough:

    askable   base .2612   null .2627   ingest .2743      by thirds  .2684 .2694 .2849
    reach     base .0384   null .0384   ingest .0434      by thirds  .0417 .0421 .0461
    null and base are FLAT down the document; ingest RISES, on both rows, monotonically.
    gain over null +0.0049, late-early +0.0064.  Gate wanted +0.05.

The mechanism is alive and correctly shaped - a document's own past makes its later holes exist
and reachable, and only its own. It is small because the write path deletes almost all of what
ingestion creates. THE SUSPECT WAS DECLARED BEFORE THE RUN and it is confirmed, but it is
bigger than one filter:

  1. `min_fillers >= 2` DELETES CONSTANT FRAMES. A frame that always holds the same token is
     not substitutable, so it is dropped. Self-reference - an article repeating its subject -
     produces exactly constant frames.
  2. THE EXAM CANNOT ASK ABOUT ONE EVEN IF KEPT. The lens is the place's OTHER fillers and the
     offer excludes the lens value (`w != v`), so for a constant place the truth is excluded
     from its own offer by construction.

TOGETHER THEY DEFINE THE EXAM AS "GUESS THE ALTERNATIVE" AND NEVER "RECALL THE VALUE". Every
number for 350 steps is substitution. But a FACT IS A CONSTANT FRAME - "X was born in Y" holds
Y and only ever Y - and recall is what knowing one means. The project's own thesis says
knowledge is counts and the mind is the decision, so recall being count-trivial is not a
problem: IT IS THE KNOWLEDGE HALF FINALLY HOLDING SOMETHING. The mind's job becomes WHICH
CHANNEL THIS HOLE WANTS, which is a decision with cardinality on both sides - Phi's kind.

WHAT IS MEASURED, torch-free, on a tape built with min_fillers=1 so constant frames survive:

    RECALL       the place's other POSITIONS (this position excluded, the place kept) - what
                 stood here before. Position-level exclusion, not place-level.
    SUBST        today's offer: place-level ban, lens = other distinct fillers, `w != v`.
    ORACLE       the truth in either.                     THE CEILING OF A PERFECT CHANNEL PICK
    RULE         a count, no heuristic: recall if the place's others are all one token.
    NULL         recall from a RANDOM OTHER PLACE of the same size. If recall pays here the
                 tape is not holding anything, it is guessing frequent tokens.

  GATE  oracle - max(always recall, always subst) > 0.05 AND recall - null > 0.05. The first
        says a channel decision exists to be made; the second says recall is real.

    python _audit359_recall.py
    python _audit359_recall.py --min-fillers 2     # the old world, for the contrast
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage359_recall.json')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--max-questions', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--corpus', default=str(WIKI))
    args = ap.parse_args()
    text = Path(args.corpus).open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s:s + args.window_lines]
    keep, toks, _owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if not keep:
        print('no tape')
        return 1
    places = [list(ps) for _a, ps in keep]
    place_of = {}
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
    where = defaultdict(list)
    for ps in places:
        for s in ps:
            where[toks[s]].append(s)
    by_size = defaultdict(list)
    for pid, ps in enumerate(places):
        by_size[len(ps)].append(pid)
    cache = {}

    def co(v):
        c = cache.get(v)
        if c is None:
            c = Counter()
            for s in where[v]:
                for s2 in places[place_of[s]]:
                    if toks[s2] != v:
                        c[toks[s2]] += 1
            cache[v] = c
        return c
    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    qs = qs[:args.max_questions]
    c = Counter()
    for s in qs:
        pid = place_of[s]
        truth = toks[s]
        others = [x for x in places[pid] if x != s]
        c['n'] += 1
        c['const'] += len({toks[x] for x in others}) == 1 if others else 0
        rec = Counter((toks[x] for x in others))
        rtop = rec.most_common(2)
        r_rec = truth in {w for w, _n in rec.most_common(args.topm)}
        pool = by_size.get(len(places[pid]), ())
        r_nul = False
        for _t in range(4):
            if not pool:
                break
            o = pool[rng.randrange(len(pool))]
            if o != pid:
                r_nul = truth in {toks[x] for x in places[o][:args.topm]}
                break
        own = Counter((toks[x] for x in places[pid]))
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        lens = list(own)[:6]
        r_sub = False
        stop, stop_full, off = ([], [], Counter())
        if lens:
            c['subst_askable'] += 1
            ban = Counter((toks[x] for x in places[pid]))
            off = Counter()
            for v in lens:
                for w, n in co(v).items():
                    n -= ban.get(w, 0)
                    if n > 0 and w != v:
                        off[w] += n
            stop_full = off.most_common(args.topm)
            stop = stop_full[:2]
            r_sub = truth in {w for w, _n in stop_full}
        c['recall'] += r_rec
        c['subst'] += r_sub
        c['null'] += r_nul
        c['oracle'] += r_rec or r_sub
        c['both'] += r_rec and r_sub
        c['only_recall'] += r_rec and (not r_sub)
        c['only_subst'] += r_sub and (not r_rec)

        def gap(t):
            if not t:
                return (-1, -1)
            return (t[0][1], t[0][1] - t[1][1] if len(t) > 1 else t[0][1])
        r1, rm = gap(rtop)
        s1, sm = gap(stop)
        c['rule_agree'] += r_rec if others and len({toks[x] for x in others}) == 1 else r_sub
        c['rule_top1'] += r_rec if r1 >= s1 else r_sub
        c['rule_margin'] += r_rec if rm >= sm else r_sub
        c['rule_rep'] += r_rec if r1 >= 2 else r_sub
        merged = rec + (off if lens else Counter())
        c['rival_merge'] += truth in {w for w, _n in merged.most_common(args.topm)}
        inter, ri, si = ([], list(rec.most_common(args.topm)), list(stop_full or ()))
        while len(inter) < args.topm and (ri or si):
            for src in (ri, si):
                if src and len(inter) < args.topm:
                    w = src.pop(0)[0]
                    if w not in inter:
                        inter.append(w)
        c['rival_inter'] += truth in set(inter)
    n = max(1, c['n'])
    g = {k: c[k] / n for k in ('recall', 'subst', 'null', 'oracle', 'both', 'only_recall', 'only_subst', 'const', 'subst_askable', 'rule_agree', 'rule_top1', 'rule_margin', 'rule_rep', 'rival_merge', 'rival_inter')}
    rep = {'lines': len(lines), 'places': len(places), 'questions': c['n'], 'topm': args.topm, 'min_fillers': args.min_fillers, **g}
    rep['best_fixed'] = max(g['recall'], g['subst'])
    rep['best_rival'] = max(rep['best_fixed'], g['rule_agree'], g['rule_top1'], g['rule_margin'], g['rule_rep'], g['rival_merge'], g['rival_inter'])
    rep['decision'] = g['oracle'] - rep['best_fixed']
    rep['headroom'] = g['oracle'] - rep['best_rival']
    rep['recall_over_null'] = g['recall'] - g['null']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape     {len(places)} places, {c['n']} questions, min_fillers {args.min_fillers}, topm {args.topm}")
    print(f"         constant places {g['const']:.4f}   substitution-askable {g['subst_askable']:.4f}")
    print(f"RECALL   {g['recall']:.4f}   null {g['null']:.4f}   over null {rep['recall_over_null']:+.4f}")
    print(f"SUBST    {g['subst']:.4f}   (today's only channel)")
    print(f"ORACLE   {g['oracle']:.4f}   both {g['both']:.4f}   only recall {g['only_recall']:.4f}   only subst {g['only_subst']:.4f}")
    print(f"RIVALS   agree {g['rule_agree']:.4f}   top1 {g['rule_top1']:.4f}   margin {g['rule_margin']:.4f}   repeated {g['rule_rep']:.4f}")
    print(f"NO-PICK  merge {g['rival_merge']:.4f}   interleave {g['rival_inter']:.4f}   (both at the same top-m a chooser gets)")
    print(f"DECISION oracle over the best fixed channel {rep['decision']:+.4f}   over the BEST RIVAL of any kind {rep['headroom']:+.4f}")
    ok = rep['headroom'] > 0.05 and rep['recall_over_null'] > 0.05
    if ok:
        print(f"\nTHERE ARE TWO CHANNELS AND CHOOSING BETWEEN THEM IS WORTH {rep['decision']:+.4f}. The tape can hold a FACT - a constant frame recalled, not an alternative guessed - and the write path has been deleting them since the first commit. Recall is count-trivial and that is correct: it is the knowledge half. The mind's new job is WHICH CHANNEL THIS HOLE WANTS, one decision, one objective, and NO COUNT MAKES IT - the best rival of any kind reads {rep['best_rival']:.4f} against an oracle of {g['oracle']:.4f}.")
    elif rep['recall_over_null'] > 0.05:
        print(f"\nRECALL IS REAL BUT NOT A CHOICE: {g['recall']:.4f} against a null of {g['null']:.4f}, yet the oracle beats the best fixed channel by only {rep['decision']:+.4f}. Add recall to the tape as a channel, give the mind nothing new to decide.")
    else:
        print("\nRECALL IS NOT REAL: reading a place's own past beats a random place of the same size by nothing. Constant frames carry no more than token frequency, the two filters cost nothing, and substitution is the whole substrate after all.")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())