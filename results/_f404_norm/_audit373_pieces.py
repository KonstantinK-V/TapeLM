"""IS THE ANSWER MADE OF MATERIAL THAT IS ALREADY REACHABLE? The atom, never once varied.

THE LAST UNTOUCHED PART OF THE CONSTRUCTION. A value has been a WHOLE WHITESPACE TOKEN from the
first frame commit. Every closure since - the decision (five rearrangements), capacity (two
tests), the relation set (371, 372a, 372b) - was measured with the answer drawn from a finite
bag of whole tokens. While the answer is CHOSEN FROM A BAG, "produce what was never offered" is
not hard, it is ARITHMETICALLY IMPOSSIBLE: you can only pick what is in the bag.

If values were PIECES, a hole could be filled by an assembly that never stood there whole. That
is production, and it is the one route to step 0's fourth trait that the substrate has never
been given. But before any of that is built, one number decides whether the idea is empty:

    ON THE HOLES NOTHING REACHES, IS THE TRUTH MADE OF MATERIAL THAT IS ALREADY IN FRONT OF US?

WHAT IS MEASURED, torch-free, on exactly the subset 363 used - truth not among the place's own
fillers and not in the walk's offer, where every number this project prints is zero:

    material   the values actually available at that hole: its own fillers plus the top-m offer
    COVER      is EVERY piece of the truth present among the pieces of that material
    ASSEMBLE   stricter and far more interpretable: can the truth be formed by CONCATENATING
               whole available values - "wikitext" out of "wiki" and "text"
    NULL       the same two, for a token drawn from the TRUTH'S OWN FREQUENCY BAND. Pieces of
               common words are everywhere, so an unmatched null would hand cover a win for
               free - 363 learned that the hard way and this reuses its bands.
    job        how many available values an assembler would have to consider. The size of the
               problem, as 351 reported `paths`

  GATE  cover - cover_null > 0.05 AND assemble - assemble_null > 0.05.
        Then the material for the unreachable answers is already present and only the ASSEMBLY
        is missing - a piece-level tape has something real to do, and the fourth trait becomes
        a task.
        If cover is high but the null matches it, "made of available pieces" is true of any
        word and means nothing. If assemble is ~0 for both, the answers are not compounds of
        what is reachable, the atom was never the obstacle, and the design is finished.

  `--ngram` is swept, not tuned: 3, 4, 5. A conclusion that only holds at one size is a
  conclusion about that size.

    python _audit373_pieces.py
    python _audit373_pieces.py --ngram 4 --window-lines 1600
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage373_pieces.json')

def pieces(tok, k):
    """character k-grams with boundaries marked, so a prefix and an infix are different
    material. No fitted constant beyond k, which is swept."""
    s = '^' + tok + '$'
    if len(s) <= k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}

def assemble(tok, avail, max_parts=3):
    """can `tok` be spelled by concatenating whole available values? Exact, no scoring, and
    bounded at three parts because a four-part spelling of a wikitext token is a coincidence
    rather than a compound - the bound is reported."""
    n = len(tok)
    reach = {0}
    for _step in range(max_parts):
        nxt = set()
        for i in reach:
            for v in avail:
                if v and tok.startswith(v, i):
                    j = i + len(v)
                    if j == n:
                        return True
                    nxt.add(j)
        if not nxt:
            return False
        reach = nxt
    return False

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--ngram', type=int, default=3)
    ap.add_argument('--max-parts', type=int, default=3)
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
    vocab = sorted(where)
    freq = {v: len(where[v]) for v in vocab}
    band_of = {v: freq[v].bit_length() for v in vocab}
    by_band = defaultdict(list)
    for v in vocab:
        by_band[band_of[v]].append(v)
    cooc = {}

    def offer_of(pid, truth):
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
            c = cooc.get(v)
            if c is None:
                c = Counter()
                for s2 in where[v]:
                    for s3 in places[place_of[s2]]:
                        if toks[s3] != v:
                            c[toks[s3]] += 1
                cooc[v] = c
            for w, n in c.items():
                n -= ban.get(w, 0)
                if n > 0 and w != v:
                    off[w] += n
        return {w for w, _n in off.most_common(args.topm)}

    def band_draw(truth, banned):
        src, w = (list(by_band[band_of[truth]]), 0)
        while len(src) < 16 and w < 20:
            w += 1
            src += by_band.get(band_of[truth] - w, []) + by_band.get(band_of[truth] + w, [])
        for _t in range(64):
            v = src[rng.randrange(len(src))]
            if v != truth and v not in banned:
                return v
        return None
    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    c = Counter()
    cov_t, cov_n, jobs = ([], [], [])
    for s in qs:
        if c['n'] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in places[pid] if x != s}
        if not own:
            continue
        c['seen'] += 1
        off = offer_of(pid, truth)
        if truth in own or truth in off:
            continue
        c['n'] += 1
        material = (own | off) - {truth}
        if not material:
            continue
        avail_p = set()
        for v in material:
            avail_p |= pieces(v, args.ngram)
        jobs.append(len(material))
        tp = pieces(truth, args.ngram)
        cov_t.append(len(tp & avail_p) / len(tp))
        c['cover'] += tp <= avail_p
        c['assemble'] += assemble(truth, material, args.max_parts)
        rv = band_draw(truth, material)
        if rv is None:
            continue
        c['nn'] += 1
        rp = pieces(rv, args.ngram)
        cov_n.append(len(rp & avail_p) / len(rp))
        c['cover_null'] += rp <= avail_p
        c['assemble_null'] += assemble(rv, material, args.max_parts)
    n = max(1, c['n'])
    nn = max(1, c['nn'])
    rep = {'lines': len(lines), 'places': len(places), 'vocab': len(vocab), 'ngram': args.ngram, 'max_parts': args.max_parts, 'topm': args.topm, 'min_fillers': args.min_fillers, 'seen': c['seen'], 'unreachable': c['n'], 'cover': c['cover'] / n, 'cover_null': c['cover_null'] / nn, 'assemble': c['assemble'] / n, 'assemble_null': c['assemble_null'] / nn, 'piece_share_truth': sum(cov_t) / max(1, len(cov_t)), 'piece_share_null': sum(cov_n) / max(1, len(cov_n)), 'job': sum(jobs) / max(1, len(jobs))}
    rep['cover_gain'] = rep['cover'] - rep['cover_null']
    rep['assemble_gain'] = rep['assemble'] - rep['assemble_null']
    rep['share_gain'] = rep['piece_share_truth'] - rep['piece_share_null']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f'tape     {len(places)} places, {len(vocab)} values, {args.ngram}-grams, parts <= {args.max_parts}')
    print(f"SUBSET   {c['n']} of {c['seen']} questions reach NEITHER channel; the material at each is {rep['job']:.1f} values")
    print(f"COVER    every piece present: truth {rep['cover']:.4f}   null {rep['cover_null']:.4f}   gain {rep['cover_gain']:+.4f}")
    print(f"SHARE    fraction of pieces present: truth {rep['piece_share_truth']:.4f}   null {rep['piece_share_null']:.4f}   gain {rep['share_gain']:+.4f}")
    print(f"ASSEMBLE spelled by whole available values: truth {rep['assemble']:.4f}   null {rep['assemble_null']:.4f}   gain {rep['assemble_gain']:+.4f}")
    if rep['cover_gain'] > 0.05 and rep['assemble_gain'] > 0.05:
        print(f"\nTHE MATERIAL IS ALREADY THERE AND ONLY THE ASSEMBLY IS MISSING. On the holes nothing reaches, the truth is spelled out of what is in front of the mind {rep['assemble']:.4f} of the time against {rep['assemble_null']:.4f} for a word of the same frequency. The atom was the obstacle: while a value is a whole token the answer must be CHOSEN, and pieces let it be BUILT. The fourth trait becomes a task over {rep['job']:.0f} available values.")
    elif rep['cover_gain'] > 0.05:
        print(f"\nCOVERED BUT NOT SPELLABLE. Every piece of the truth is present ({rep['cover']:.4f} against {rep['cover_null']:.4f}) yet it cannot be built by concatenating whole available values ({rep['assemble']:.4f} vs {rep['assemble_null']:.4f}). The pieces are there in the way the letters of any word are there - real material, no compositional route from it to the answer.")
    else:
        print("\nTHE ATOM WAS NOT THE OBSTACLE. The unreachable truths are made of no more available material than a random word of the same frequency. Pieces would give the mind nothing to build with, and 'produce what was never offered' does not become possible by making the values smaller.")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())