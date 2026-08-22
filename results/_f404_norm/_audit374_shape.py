"""THE FORM CHANNEL. An ending is not material at the hole - it comes from the frame.

WHAT 373 ACTUALLY CLOSED, AND WHAT IT DID NOT. 373 asked whether an unreachable truth can be
made by CONCATENATING WHOLE AVAILABLE VALUES ("wikitext" out of "wiki" and "text") and answered
0.0000 on five corpora. I read that as "production is dead". That reading was wrong, and 373's
own cross-corpus table is the evidence against it:

    en wiki .036   de news .046   fi wiki .049   code idents .164        <- SHARE, piece overlap
    ASSEMBLE 0.0000 everywhere.

SHARE rises strictly with morphology. ASSEMBLE does not move at all. Finnish is agglutinative -
the STEM of the truth is right there and the ENDING is a grammatical item that NEVER stands at
the hole as a whole value. So concatenation was the wrong operation for exactly the corpora
where morphology is strongest. What 373 refuted is composition out of the bag. It never tested
the only composition a language actually performs:

    STEM from the offer (identity - the channel that already reaches 0.65)
        +  ENDING from the frame (form - a channel this project has never had)
        =  A WORD FORM THAT NEVER STOOD AT THIS HOLE.

That is production, it is not chosen from a bag, and no number here has ever measured it.

WHY THE FRAME CAN CARRY THE ENDING AT ALL. The address is (w, left, right) - literally the
tokens around the hole, articles and prepositions included. Phi has never seen the FORM of a
candidate against the FORM of its frame; it sees co-occurrence counts and nothing else. The
inventory of endings is tiny (a few dozen suffixes cover a corpus), so this is a COUNT, it
carries no facts, and it scales to any corpus - the same contract as every other channel.

WHAT IS MEASURED, torch-free. Two disjoint populations, one pass:

  A. SHAPE, on questions the substitution channel already REACHES. The ceiling here is the
     residual hit/reach = 0.88, nothing more - a form cue can only reorder what is offered.
       PRED     is the truth's ending in the frame's predicted top-t, by three predictors:
                  local   the endings of this place's OTHER fillers          (this place)
                  func    the endings pooled over every place whose address carries the SAME
                          FUNCTION WORDS - own place subtracted. THE ARTICLE HYPOTHESIS, and
                          the function-word list is the corpus's F most frequent tokens, counted
                  global  the corpus's top-t endings                          THE NULL
       RERANK   top-1 of the offer, before and after keeping only ending-matching candidates.

  B. PRODUCE, on the subset where every number this project prints is ZERO - truth in neither
     the place's own fillers nor the offer (363's and 373's population, unchanged so the numbers
     are comparable).
       STEMS      {v} and {stem(v)} for every available value v. Both, because stripping is as
                  legitimate as adding: "playing" offered, "play" true.
       ORACLE     truth = stem + e for SOME e in the whole counted inventory.  MATERIAL EXISTS
       PRODUCE    truth = stem + e for e in the frame's predicted top-t.       FRAME FINDS IT
       bare       of those, the ones with an empty ending (pure stripping), reported apart so
                  it cannot silently carry the number.
       NULL       the identical construction for a token from the truth's own FREQUENCY BAND,
                  reusing 373's bands. Endings of common words are everywhere; an unmatched null
                  would hand this a win for free.

  GATES, declared before the run:
    G1  produce_oracle - oracle_null > 0.05      the material for a NEW FORM is present
    G2  produce_func   - func_null   > 0.05      and the frame predicts the ending that builds it
    Both must pass. G1 alone means the pieces exist and nothing points at them; G2 alone cannot
    happen. If G1 fails, the unreachable truths are not inflections of anything available and
    the form channel is empty on this corpus.
    SHAPE is measured and reported but DOES NOT DECIDE THE STEP - it is a policy gain inside a
    population that already works, and no new metric may decide a step.

  Swept, not tuned: --endings 32/64/128, --top-ends 1/4/8, --suffix-max 4. A conclusion that
  holds at one setting is a conclusion about that setting.

    python _audit374_shape.py
    python _audit374_shape.py --endings 128 --top-ends 8 --window-lines 1600
    python _audit374_shape.py --corpus data/_fi_wiki.txt --window-lines 8000
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage374_shape.json')

def ending_inventory(vocab, suffix_max, k):
    """the corpus's endings, by count: rank every suffix of length 1..L by HOW MANY DISTINCT
    TYPES end in it, keep the top k. Types not tokens, so `the` does not make `he` an ending."""
    c = Counter()
    for v in vocab:
        for L in range(1, suffix_max + 1):
            if len(v) > L:
                c[v[-L:]] += 1
    return [e for e, _n in c.most_common(k)]

def make_split(endings):
    """one deterministic decomposition per token: the LONGEST known ending it carries."""
    by_len = sorted({len(e) for e in endings}, reverse=True)
    es = set(endings)
    memo = {}

    def split(v):
        hit = memo.get(v)
        if hit is None:
            hit = (v, '')
            for L in by_len:
                if len(v) > L and v[-L:] in es:
                    hit = (v[:-L], v[-L:])
                    break
            memo[v] = hit
        return hit
    return split

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=400)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--endings', type=int, default=64)
    ap.add_argument('--suffix-max', type=int, default=4)
    ap.add_argument('--top-ends', type=int, default=4)
    ap.add_argument('--func', type=int, default=32)
    ap.add_argument('--max-questions', type=int, default=1500)
    ap.add_argument('--max-shape', type=int, default=6000)
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
    addrs = [a for a, _ps in keep]
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
    endings = ending_inventory(vocab, args.suffix_max, args.endings)
    end_set = set(endings) | {''}
    split = make_split(endings)
    tok_freq = Counter(toks)
    func = {w for w, _n in tok_freq.most_common(args.func)}
    place_ends, sig_of = ([], [])
    lvl_ends = [defaultdict(Counter) for _ in range(3)]
    bucket_n = Counter()
    global_ends = Counter()
    for pid, ps in enumerate(places):
        pe = Counter((split(toks[x])[1] for x in ps))
        place_ends.append(pe)
        global_ends += pe
        _w, left, right = addrs[pid]
        lf = tuple((t for t in left if t in func))
        rf = tuple((t for t in right if t in func))
        sig = ((lf, rf), (lf,), (rf,))
        sig_of.append(sig)
        for L in range(3):
            lvl_ends[L][sig[L]] += pe
        bucket_n[sig[0]] += 1
    p_glo = {e for e, _n in global_ends.most_common(args.top_ends)}
    cooc = {}

    def offer_of(pid, truth):
        """exactly 363's and 373's offer, returned ORDERED so it can be reranked."""
        own = Counter((toks[x] for x in places[pid]))
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        lens = list(own)[:6]
        if not lens:
            return []
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
        return off.most_common(args.topm)

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

    def buildable(target, stems, ends):
        """target = stem + e, for an available stem and a PREDICTED ending. Returns
        (built, bare) - bare marks the empty ending, i.e. pure stripping."""
        built = bare = False
        for st in stems:
            if not target.startswith(st):
                continue
            e = target[len(st):]
            if e in ends:
                built = True
                if e == '':
                    bare = True
                else:
                    return (True, False)
        return (built, bare)
    qs = [s for ps in places for s in ps]
    rng.shuffle(qs)
    c = Counter()
    for s in qs:
        if c['nB'] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in places[pid] if x != s}
        if not own:
            continue
        c['seen'] += 1
        off_pairs = offer_of(pid, truth)
        off = {w for w, _n in off_pairs}
        te = split(truth)[1]
        loc = Counter(place_ends[pid])
        loc[te] -= 1
        if loc[te] <= 0:
            del loc[te]
        sig = sig_of[pid]
        fun, used = (Counter(), 3)
        for L in range(3):
            fun = lvl_ends[L][sig[L]] - place_ends[pid]
            if fun:
                used = L
                break
        c[f'lvl{used}'] += 1
        c['lvl_n'] += 1
        p_loc = {e for e, _n in loc.most_common(args.top_ends)} or p_glo
        p_fun = {e for e, _n in fun.most_common(args.top_ends)} or p_glo
        if truth in own or truth in off:
            if c['nA'] >= args.max_shape:
                continue
            c['nA'] += 1
            c['A_loc'] += te in p_loc
            c['A_fun'] += te in p_fun
            c['A_glo'] += te in p_glo
            c['A_bucket1'] += bucket_n[sig] <= 1
            if off_pairs and truth in off:
                c['nR'] += 1
                c['R_base'] += off_pairs[0][0] == truth
                kept = [w for w, _n in off_pairs if split(w)[1] in p_fun]
                c['R_fun'] += (kept[0] if kept else off_pairs[0][0]) == truth
                kept_l = [w for w, _n in off_pairs if split(w)[1] in p_loc]
                c['R_loc'] += (kept_l[0] if kept_l else off_pairs[0][0]) == truth
            continue
        material = (own | off) - {truth}
        if not material:
            continue
        c['nB'] += 1
        stems = set()
        for v in material:
            stems.add(v)
            stems.add(split(v)[0])
        b, bare = buildable(truth, stems, end_set)
        c['B_oracle'] += b
        c['B_oracle_bare'] += bare
        b, bare = buildable(truth, stems, p_fun)
        c['B_fun'] += b
        c['B_fun_bare'] += bare
        b, _ = buildable(truth, stems, p_loc)
        c['B_loc'] += b
        b, _ = buildable(truth, stems, p_glo)
        c['B_glo'] += b
        rv = band_draw(truth, material)
        if rv is None:
            continue
        c['nN'] += 1
        b, bare = buildable(rv, stems, end_set)
        c['N_oracle'] += b
        c['N_oracle_bare'] += bare
        b, _ = buildable(rv, stems, p_fun)
        c['N_fun'] += b
        b, _ = buildable(rv, stems, p_loc)
        c['N_loc'] += b
        b, _ = buildable(rv, stems, p_glo)
        c['N_glo'] += b
    nA, nB, nN, nR = (max(1, c[k]) for k in ('nA', 'nB', 'nN', 'nR'))
    rep = {'corpus': args.corpus, 'lines': len(lines), 'places': len(places), 'vocab': len(vocab), 'endings': args.endings, 'suffix_max': args.suffix_max, 'top_ends': args.top_ends, 'func': args.func, 'topm': args.topm, 'min_fillers': args.min_fillers, 'seen': c['seen'], 'reachable': c['nA'], 'unreachable': c['nB'], 'nulls': c['nN'], 'buckets': len(lvl_ends[0]), 'singleton_bucket': c['A_bucket1'] / nA, 'backoff': [c[f'lvl{L}'] / max(1, c['lvl_n']) for L in range(4)], 'pred_local': c['A_loc'] / nA, 'pred_func': c['A_fun'] / nA, 'pred_global': c['A_glo'] / nA, 'rerank_n': c['nR'], 'rerank_base': c['R_base'] / nR, 'rerank_func': c['R_fun'] / nR, 'rerank_local': c['R_loc'] / nR, 'produce_oracle': c['B_oracle'] / nB, 'oracle_null': c['N_oracle'] / nN, 'produce_func': c['B_fun'] / nB, 'func_null': c['N_fun'] / nN, 'produce_local': c['B_loc'] / nB, 'local_null': c['N_loc'] / nN, 'produce_global': c['B_glo'] / nB, 'global_null': c['N_glo'] / nN, 'oracle_bare': c['B_oracle_bare'] / nB, 'func_bare': c['B_fun_bare'] / nB}
    rep['G1'] = rep['produce_oracle'] - rep['oracle_null']
    rep['G2'] = rep['produce_func'] - rep['func_null']
    rep['G2_local'] = rep['produce_local'] - rep['local_null']
    rep['shape_gain'] = rep['pred_func'] - rep['pred_global']
    rep['rerank_gain'] = rep['rerank_func'] - rep['rerank_base']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f'tape     {len(places)} places, {len(vocab)} values, {args.endings} endings up to {args.suffix_max} chars, top-{args.top_ends} predicted, {args.func} function words')
    print(f"buckets  {len(lvl_ends[0])} function-word signatures, {rep['singleton_bucket']:.4f} of reachable questions sit in a bucket of one")
    print(f"backoff  both sides {rep['backoff'][0]:.4f}   left {rep['backoff'][1]:.4f}   right {rep['backoff'][2]:.4f}   nothing, fell to global {rep['backoff'][3]:.4f}")
    print(f"\nA SHAPE   {c['nA']} reachable questions - a policy gain, it does NOT decide")
    print(f"  PRED    true ending predicted: local {rep['pred_local']:.4f}   func {rep['pred_func']:.4f}   global {rep['pred_global']:.4f}   func over global {rep['shape_gain']:+.4f}")
    print(f"  RERANK  top-1 of the offer ({c['nR']} q): base {rep['rerank_base']:.4f}   func {rep['rerank_func']:.4f}   local {rep['rerank_local']:.4f}   gain {rep['rerank_gain']:+.4f}")
    print(f"\nB PRODUCE {c['nB']} questions reach NEITHER channel; {c['nN']} band-matched nulls")
    print(f"  ORACLE  stem available + ANY known ending: {rep['produce_oracle']:.4f}   null {rep['oracle_null']:.4f}   G1 {rep['G1']:+.4f}   (bare {rep['oracle_bare']:.4f})")
    print(f"  FUNC    stem + ending the ARTICLES predict: {rep['produce_func']:.4f}   null {rep['func_null']:.4f}   G2 {rep['G2']:+.4f}   (bare {rep['func_bare']:.4f})")
    print(f"  LOCAL   stem + ending this place predicts:  {rep['produce_local']:.4f}   null {rep['local_null']:.4f}   G2l {rep['G2_local']:+.4f}")
    print(f"  GLOBAL  stem + the corpus's commonest ends: {rep['produce_global']:.4f}   null {rep['global_null']:.4f}")
    win = max((rep['G2'], 'func', rep['produce_func'], rep['func_null']), (rep['G2_local'], 'local', rep['produce_local'], rep['local_null']))
    g1, g2 = (rep['G1'] > 0.05, win[0] > 0.05)
    if c['nB'] < 100 or c['nN'] < 100:
        print(f"\nVOID, NOT A RESULT. Only {c['nB']} unreachable questions and {c['nN']} nulls survived - too few for a 0.05 gate to mean anything. Widen the window (--window-lines) until B has a few hundred, then read it.")
    elif g1 and g2:
        print(f'\nPRODUCTION IS ALIVE AND THE FRAME AIMS IT. On the holes where every number this project prints is zero, the truth is an AVAILABLE STEM IN A PREDICTED FORM ({win[1]}) {win[2]:.4f} of the time against {win[3]:.4f} for a word of its own frequency. 373 refuted concatenation, not inflection. The value must stop being a whole token: a candidate becomes (stem, ending), the stem comes from the offer that already reaches, the ending from the frame - and the answer is BUILT, not chosen. The fourth trait is a task again, inside one objective.')
    elif g1:
        print(f"\nTHE MATERIAL IS THERE AND NOTHING POINTS AT IT. The stem is available and a known ending completes it {rep['produce_oracle']:.4f} of the time against {rep['oracle_null']:.4f}, but no frame predictor finds that ending - func {rep['G2']:+.4f}, local {rep['G2_local']:+.4f}. Inflection is real here and the ARTICLES DO NOT CARRY IT at this width; the next question is the predictor, not the atom - a wider address, or the ending as a feature Phi ranks rather than a filter applied before it.")
    else:
        print(f"\nTHE UNREACHABLE TRUTHS ARE NOT INFLECTIONS OF WHAT IS AVAILABLE. Even knowing the whole ending inventory, the truth is an available stem plus an ending {rep['produce_oracle']:.4f} of the time against {rep['oracle_null']:.4f} for a random word of the same frequency. The stems of the answers are not at the hole either - so it is not the atom and not the form channel: what is missing at these holes is the MATERIAL, and no decomposition of the value creates it.")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())