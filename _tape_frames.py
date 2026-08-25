"""The tape written by counting, not by a grammar - 297's frame cutter as a pack.

WHAT A PLACE IS HERE. Not (anchor | relation) pulled out of a sentence by a rule, but a HOLE
whose surroundings recur: take a token, widen the window left and right while the frame is still
something the corpus wrote at least twice, and the widest such frame is the address. What varies
in the hole is the value. No tau, no stopword list, no grammar, and no length chosen by us - the
width is whatever the corpus supports. The 293 audit is why: the parser's relation half is a
function word on this corpus (`canada|and`, `december|the`), and `rare_nonzero_rate` was 0.0.

WHAT THE AUDIT SAID, so the shape of this tape is not a hope: at 600 KB, 7947 addresses and
42266 slots against the parser's ~800 and ~2500, mentions per address 5.3 against 3.1, 35% of
tokens covered, and 22% of rows repeating a filler already at their frame - the CONFIRM supply
that was unmeasurable on the parsed tape. It also killed one of my guesses: width does NOT
predict quality (w=1 averages 4.31 distinct fillers, w=2 only 2.73), so nothing here selects on
width. The quality signal is the number of distinct fillers, which is a count and travels into
the graph as one.

WHY THIS IS A PACK AND NOT A STAGE. The frame tape has the SAME SHAPE as the parsed one -
addresses, slots, values, contexts - so every reader downstream (build_graph, Phi, imports, both
rivals) runs unchanged. Only the writing rule is replaced, which is the whole point: exact
counting in the write path, and the interpolation moved to where the mind lives.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict


_KEEP_MEMO: dict = {}


def frame_keep(lines, frame_max: int = 12, min_fillers: int = 2):
    """The counted frames of a corpus - everything in the write path that the draw cannot change.

    COUNTED ONCE PER CORPUS, NOT ONCE PER RESAMPLE. The tape is redrawn every 50 steps, 80 times
    in a 4000-step run, and every redraw re-counted every window of the whole corpus to produce
    the same `keep` and then sample it differently. Only the SAMPLE depends on the rng. Memoised
    on the identity of the line list, and the list itself is held in the entry so the id cannot
    be recycled onto a different corpus.

    THE PRUNE IS EXACT, NOT A SHORTCUT. A window of width w+1 contains the width-w window with
    the same hole, so its count can only be lower: a position whose width-w frame occurred once
    cannot have a width-(w+1) frame that occurs twice. Widening only positions still alive turns
    `frame_max` full passes over the corpus into one full pass plus a shrinking tail, and the
    peak allocation from three Counters of million-key tuples - the several gigabytes of private
    working set that grew across the run - never happens. Any key with a true count of 2 or more
    keeps all of its positions alive, so counts on the restricted set equal the global ones for
    every key that survives; keys counted low are exactly the ones being discarded.
    """
    key = id(lines)
    hit = _KEEP_MEMO.get(key)
    if hit is not None and hit[0] is lines and hit[1] == (frame_max, min_fillers):
        return hit[2], hit[3], hit[4]

    toks, owner = [], []
    for li, line in enumerate(lines):
        for t in line.split():
            toks.append(t)
            owner.append(li)
    n = len(toks)
    if n < 3:
        return [], toks, owner

    best_key, alive = {}, list(range(1, n - 1))
    for w in range(1, frame_max + 1):
        pos_of = defaultdict(list)
        for i in alive:
            if i - w < 0 or i + 1 + w > n:
                continue
            # A text line is a record boundary. Flattening is only storage; a
            # counted address may not borrow glue from the previous/next row.
            if owner[i - w] != owner[i] or owner[i + w] != owner[i]:
                continue
            pos_of[(tuple(toks[i - w:i]), tuple(toks[i + 1:i + 1 + w]))].append(i)
        nxt = []
        for (left, right), ps in pos_of.items():
            if len(ps) >= 2:
                for i in ps:
                    best_key[i] = (w, left, right)
                nxt.extend(ps)
        alive = nxt
        if not alive:
            break

    by_addr = defaultdict(list)
    for i, k in best_key.items():
        by_addr[k].append(i)
    keep = [(k, sorted(pos)) for k, pos in by_addr.items()
            if len({toks[i] for i in pos}) >= min_fillers]
    keep.sort(key=lambda kv: (-kv[0][0], -len(kv[1]), kv[0][1]))   # a stable order to sample from
    # two corpora are live in a run - train and held - so the table holds two and never grows
    while len(_KEEP_MEMO) >= 2:
        _KEEP_MEMO.pop(next(iter(_KEEP_MEMO)))
    _KEEP_MEMO[key] = (lines, (frame_max, min_fillers), keep, toks, owner)
    return keep, toks, owner


_LINE_MEMO: dict = {}


def _by_line(keep, owner):
    """line -> the (address, position) pairs written on it, built once per corpus.

    Keyed on the identity of `keep`, which frame_keep memoises, so a run that redraws the tape
    eighty times indexes the corpus once. Rebuilding it per draw cost 0.46s on a sixth of the
    pool - the same shape of waste as recounting every window per resample.
    """
    k = id(keep)
    hit = _LINE_MEMO.get(k)
    if hit is not None and hit[0] is keep:
        return hit[1]
    bl = defaultdict(list)
    for a, pos in keep:
        for i in pos:
            bl[owner[i]].append((a, i))
    while len(_LINE_MEMO) >= 2:          # train and held are both live, as in _KEEP_MEMO
        _LINE_MEMO.pop(next(iter(_LINE_MEMO)))
    _LINE_MEMO[k] = (keep, bl)
    return bl


def frame_region(keep, toks, owner, n_lines, max_addresses, rng, min_fillers=2):
    """The same number of addresses, taken from a CONTIGUOUS stretch of lines.

    WHY THE DRAW IS THE DEFECT. Sampling addresses uniformly takes 3000 of 250195 qualifying
    frames - 1.2% - so two frames of one sentence land on the tape together about once in a
    hundred, and EVERY relation between places is diluted by that ratio before anything reads
    them. The 300 audit measured the dilution rather than the corpus: `line` read 0.041 at 7.7%
    of the pool and 0.0067 at 1.2%, falling with the fraction. On a region tape of the same size
    it reads 0.237, and the questions only adjacency can answer go from 0.1% to 5.0%.

    It also repairs the walk, which was the surprise. Neighbouring places are now neighbours in
    the text rather than accidental look-alikes: the median rank of the truth in the walk's own
    candidate list falls from 21 to 5, reachability at eight candidates rises from 0.162 to
    0.271, and the eight nearest places offer 84 fillers instead of 1015.

    The price is real and is not hidden: mentions per address 7.09 -> 3.75, own 0.450 -> 0.346,
    share 0.596 -> 0.535. A place seen inside one window has been seen fewer times. Mentions
    OUTSIDE the window are dropped, because a tape that is a region cannot quietly borrow
    evidence from the rest of the corpus.
    """
    by_line = _by_line(keep, owner)
    # MIN_FILLERS IS ABOUT DISTINCT VALUES, NOT ABOUT ROWS, and inside a window that is not the
    # same test. The full-corpus keep guarantees two different fillers over ALL of an address's
    # positions; restricted to one stretch of lines it can easily be one filler written twice,
    # which is a fixed phrase and not a place where something is written - the rule this module
    # opens with. Counting rows here would have quietly readmitted them.
    # COUNTED AS THE WINDOW GROWS, not recounted at every line. Rebuilding the set of distinct
    # fillers for every accumulated address once per line is O(window x addresses x mentions):
    # measured at 15s on a sixth of the pool, which is hours across eighty resamples. The count
    # only ever moves when a filler is seen at a place for the FIRST time, so it is maintained.
    start = rng.randrange(max(1, n_lines))
    acc, seen, qualified = defaultdict(list), defaultdict(set), 0
    for d in range(n_lines):
        for k, i in by_line.get((start + d) % n_lines, ()):
            acc[k].append(i)
            v = toks[i]
            if v not in seen[k]:
                seen[k].add(v)
                if len(seen[k]) == min_fillers:
                    qualified += 1
        if qualified >= max_addresses:
            break
    out = [(k, sorted(v)) for k, v in acc.items() if len(seen[k]) >= min_fillers]
    return out[:max_addresses] if max_addresses else out


def frame_assertions(lines, frame_max: int = 12, min_fillers: int = 2,
                     max_addresses: int = 0, rng=None, sample: str = "uniform"):
    """Assertions in 279's shape - address, value, ctx, source - cut by recurrence alone.

    Counting and nothing else: `frame_keep` gives each token the widest window the corpus wrote
    twice, and an address survives only if its hole took at least `min_fillers` DIFFERENT values,
    because a frame with one filler is a fixed phrase, not a place where something is written.
    Everything here that is not counting is the draw.
    """
    keep, toks, owner = frame_keep(lines, frame_max, min_fillers)
    if not keep:
        return [], [], 0
    # SAMPLED, NEVER TRUNCATED. Taking the top N was deterministic over the whole corpus, so
    # every redraw returned the SAME tape - 298 printed resample_overlap 1.0, which retires the
    # one property this project had held onto: a mind cannot memorise a tape it keeps losing.
    # It also made the two slices incomparable, because the widest and densest frames of a
    # different slice are much denser (2.996 mentions per address on train against 11.245 on
    # held). Uniform sampling fixes both, and the pool size is returned so the overlap that
    # results is a number rather than a hope: it is about max_addresses / len(keep).
    pool = len(keep)
    r = rng or random.Random(0)
    if sample == "region":
        keep = frame_region(keep, toks, owner, len(lines), max_addresses, r, min_fillers)
    elif max_addresses and pool > max_addresses:
        keep = r.sample(keep, max_addresses)

    out, addrs = [], []
    for (w, left, right), pos in sorted(keep, key=lambda kv: (kv[0][1], kv[0][2])):
        name = f"{' '.join(left)}|{' '.join(right)}"
        addrs.append(name)
        for j, i in enumerate(pos):
            out.append({"address": name, "value": toks[i],
                        # the whole line is the context, as it was for the parsed tape: the ink
                        # needs a sentence to be worth anything, and ctx_fp excludes the value
                        "ctx": lines[owner[i]], "source": f"frame:{name}:{j}",
                        # WHERE THIS MENTION SITS IN THE CORPUS, and which line it is on.
                        #
                        # Needed because FRAMES OVERLAP. The hole at "cat" in `the cat sat on the
                        # mat` belongs to `the|sat` and to `|sat on` at the SAME token position:
                        # a step to a neighbouring frame of the same line would read the answer
                        # straight off an overlapping window, which is not another record but
                        # the same word seen twice. Only a position can tell those apart.
                        "pos": i, "line": owner[i]})
    return out, addrs, pool


def frame_report(asserts, addrs):
    per = Counter(a["address"] for a in asserts)
    fill = defaultdict(set)
    for a in asserts:
        fill[a["address"]].add(a["value"])
    n = max(1, len(addrs))
    return {"addresses": len(addrs), "slots": len(asserts),
            "mentions_per_address": len(asserts) / n,
            "distinct_fillers_mean": sum(len(v) for v in fill.values()) / n,
            "confirm_rows": sum(per[a] - len(fill[a]) for a in addrs)}
