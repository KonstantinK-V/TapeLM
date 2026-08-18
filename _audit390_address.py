"""THE CEILING OF NAMING A PLACE THE WALK CANNOT REACH. Measured before a chooser is built.

WHY THIS AND NOT ANOTHER TERM ON PHI. 389 was voided by its own control (section 31): the raw
score already carries a scale, so the missing thing is not a better-scaled score. The user's own
reading of what is left - "the mind still does not choose WHERE TO LOOK" - is the wall section 29
calls A, an OUTPUT SPACE, approached from the place side instead of the word side. `--moves`
(385/386) chooses among three fixed channels; nothing in this project lets the mind NAME A PLACE
IT WAS NOT OFFERED.

THE STRUCTURAL FACT THIS RESTS ON, and it is 387's own suspect, stated as an opportunity instead
of a complaint. The arm runs `fp=fillers`, so a place's fingerprint IS its bag of fillers and the
cosine between two profiles is ZERO unless they share a filler. `connect` walks places that share
a filler. THEREFORE EVERY CHANNEL THE MIND HAS - step, share, and copy through the line - has the
same non-zero support: places that share a filler with this one. A place that shares no filler is
not ranked low by the current machinery; it is INVISIBLE to it, at any cap and any depth. 387's
gate said so in its own words: if the lanes are redundant "what is needed is a genuinely
different move, one that reaches somewhere the walk cannot".

An address IS such a handle, and it is already written on the tape. `frame_keep` makes a place a
key `(w, left, right)` - two word sequences the corpus wrote. So a place can be named by its
HALVES, and halves compose:

    my place            the |  sat            L0 | R0
    keep my left        the |  ran            L0 | Rj      a place I share no filler with
    keep my right       a   |  sat            Li | R0
    compose             a   |  ran            Li | Rj      a place NOBODY offered me

The output space is then a PRODUCT, |L| x |R|, while the cost of naming a member of it is |L| +
|R|. That is the one shape that does not walk into 347's law - four measurements that a wider
offer costs more than it brings - because the offer stays eight; only the place it is drawn from
is chosen. Whether there is anything to draw is what this file measures, torch-free, BEFORE any
head is built for it.

WHAT IS MEASURED, on the population where reach matters (the truth is not among the question's
own values), with the hidden position excluded from everything:

    std8            the truth in the standing arm's eight (step interleaved with connect)  TODAY
    half_any        the truth anywhere among places sharing a half of my address           IS IT THERE
    half8           the truth in the address lane's eight
    half_only       the truth in the address eight and NOT in the standing eight           WHERE IT DECIDES
    comp8/comp_only the same for COMPOSED addresses - a half of mine or of a walked place,
                    crossed, kept only where the corpus actually wrote that frame
    orc8            the truth in either lane                         A PERFECT CHOOSER OF WHERE
    rand8           the same number of random places                 the floor
    d_half8         a FREQUENCY-MATCHED DECOY through the same lane  is the lane just frequency
    new_share       the share of address neighbours that share NO filler with my place - the
                    part of the reach that no existing channel can see, counted rather than
                    assumed from the cosine argument above

THE LEAK OF SECTION 27 IS THE FIRST THING THIS FILE GUARDS. Both 387 and 388 built the question's
neighbourhood out of `prof[pid]`, WHICH STILL CONTAINS THE HIDDEN TOKEN, and the leak was doing
most of the work (share .611 -> .180). Here every lane takes `qprof` - the profile with the query
row removed - as an argument, and no function reads `T["prof"][pid]` for the question's own place.
`_check390_address.py` re-introduces exactly that substitution and requires the numbers to move.

SAME-LINE PLACES ARE DROPPED FROM EVERY LANE. Frames overlap: `the cat sat on the mat` writes a
hole at `cat` for `the|sat` and for `|sat on` at the same token position, and a neighbouring frame
of the same line is the same words seen twice rather than a second record. `frame_keep` gives each
position exactly one address, so the hidden token itself cannot be re-read - but a place holding
another slot of the same line can still be a window artefact rather than evidence. The drop is
applied to the standing lanes too, so this stays a lane-against-lane comparison at equal footing,
and its direction is conservative: it can only cost a lane, never grant it.

  VOID CHECK, READ BEFORE ANYTHING ELSE. `new_share` and `n_half_new`. If the places that share a
  half are the same places that share a filler - n_half_new near zero - then the address names
  nothing the walk cannot already see, the premise of this step is false on this tape, and it
  closes here without a training run. That is 389's gate 4, kept because it worked.

  GATE. half_only > 0.05 OR comp_only > 0.05, with orc8 - std8 > 0.05, AND the null beaten:
  rand8_only below half of the winning lane's, and d_half8_only < 0.05 (a frequency-matched decoy
  must not ride the same lane). Then the next lever is the address as an output space in the
  stage, at the unchanged cap of eight. If the address lane is empty or is only frequency, the
  handle is closed on this tape and section 29's wall A has to be pushed on the word side.

  Nothing here decides on a new metric: reach@8, only_X and a matched decoy are the quantities
  387 and 388 were specified on.

    python _audit390_address.py
    python _audit390_address.py --window-lines 1600 --places 16
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

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage390_address.json")


# ---------------------------------------------------------------------------------- the tape

def build_tape(lines, frame_max=3, min_fillers=1):
    """The frame tape as counts: places, profiles, and the two half indices the address needs."""
    keep, toks, owner = tframes.frame_keep(lines, frame_max, min_fillers)
    addrs = [a for a, _ps in keep]
    places = [list(ps) for _a, ps in keep]
    place_of, on_line = {}, defaultdict(set)
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
            on_line[owner[s]].add(pid)
    prof = [Counter(toks[s] for s in ps) for ps in places]
    norm = [math.sqrt(sum(c * c for c in pr.values())) or 1.0 for pr in prof]
    at_value = defaultdict(list)
    for pid, pr in enumerate(prof):
        for v in pr:
            at_value[v].append(pid)
    # THE ADDRESS AS TWO HANDLES. `frame_keep` already wrote them; nothing is parsed here.
    by_left, by_right, of_addr = defaultdict(list), defaultdict(list), {}
    for pid, (w, left, right) in enumerate(addrs):
        by_left[left].append(pid)
        by_right[right].append(pid)
        of_addr[(w, left, right)] = pid          # a bijection: frame_keep groups by this key
    freq = Counter()
    for ps in places:
        for s in ps:
            freq[toks[s]] += 1
    by_band = defaultdict(list)
    for v, f in freq.items():
        by_band[f.bit_length()].append(v)
    return {"toks": toks, "owner": owner, "addrs": addrs, "places": places, "place_of": place_of,
            "on_line": on_line, "prof": prof, "norm": norm, "at_value": dict(at_value),
            "by_left": dict(by_left), "by_right": dict(by_right), "of_addr": of_addr,
            "freq": freq, "by_band": by_band}


# --------------------------------------------------------------- the channels that exist today

def filler_nbrs(T, pid, qprof, drop=()):
    """Places sharing at least one filler with the QUERY PROFILE, with overlap and dot.

    `qprof` is passed in and `T["prof"][pid]` is never read for the asking place. Section 27:
    reading it puts the hidden token in the search key, so the walk goes to the places holding
    the answer and then scores the answer out of exactly those places.
    """
    ov, dot = Counter(), Counter()
    for v, c in qprof.items():
        for j in T["at_value"].get(v, ()):
            if j != pid and j not in drop:
                ov[j] += 1
                dot[j] += c * T["prof"][j][v]
    return ov, dot


def walk_order(T, pid, qprof, k, drop=()):
    """The k nearest places by fingerprint cosine - ONE definition, read by the walk and by the
    composer, so "the places the mind has in hand" cannot drift between the two."""
    _ov, dot = filler_nbrs(T, pid, qprof, drop)
    qn = math.sqrt(sum(c * c for c in qprof.values())) or 1.0
    return sorted(dot, key=lambda j: (-(dot[j] / (qn * T["norm"][j])), j))[:k]


def fillers_of(T, order, own):
    """The fillers of a list of places, in place order, deduped, own values excluded. Every lane
    in this file takes its values this way, so a lane is only ever a different SET OF PLACES."""
    out, seen = [], set(own)
    for j in order:
        for v, _c in T["prof"][j].most_common():
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def lane_step(T, pid, own, qprof, k, drop=()):
    """The fingerprint walk: fillers of the k nearest places, in place order, uncut."""
    return fillers_of(T, walk_order(T, pid, qprof, k, drop), own)


def lane_share(T, pid, own, qprof, drop=(), norm_by_places=False):
    """365's connect: a value scored by the summed overlap of the neighbours holding it.

    `norm_by_places` divides by how many places on the whole tape hold the value, which turns the
    sum into a mean overlap. 317, 383 and 387 each found the raw sum rewarding a value for being
    COMMON on a population whose answers are rare; both readings are printed so that is visible
    rather than assumed.
    """
    ov, _dot = filler_nbrs(T, pid, qprof, drop)
    score = Counter()
    for j, o in ov.items():
        for v in T["prof"][j]:
            if v not in own:
                score[v] += o
    if not norm_by_places:
        return [v for v, _n in score.most_common()]
    return sorted(score, key=lambda v: (-(score[v] / len(T["at_value"][v])), -score[v], v))


def interleave(*lanes, cap=None):
    """The stage's merge rule: round-robin, deduped, cut - never appended (347)."""
    seen, out = set(), []
    for tup in zip_longest(*lanes):
        for v in tup:
            if v is not None and v not in seen:
                seen.add(v)
                out.append(v)
    return out if cap is None else out[:cap]


# ------------------------------------------------------------------ the channel that does not

def half_nbrs(T, pid, drop=()):
    """Places sharing a HALF of this address: pid -> how many of the two halves match.

    No filler is consulted, which is the whole point - this is the one relation on the tape that
    is not a function of what stands in the hole.
    """
    _w, left, right = T["addrs"][pid]
    m = Counter()
    for j in T["by_left"].get(left, ()):
        m[j] += 1
    for j in T["by_right"].get(right, ()):
        m[j] += 1
    m.pop(pid, None)
    for j in drop:
        m.pop(j, None)
    return m


def compose_nbrs(T, pid, seen_pids, drop=()):
    """Addresses assembled from halves of places already in hand, kept only where the corpus
    wrote that frame.

    THE PRODUCT AT THE COST OF THE SUM. |L| lefts and |R| rights name up to |L|x|R| places while
    the mind emits one of |L| + |R| handles. Widths must match, because `frame_keep`'s key is
    (w, left, right) with |left| = |right| = w - that is the tape's rule, not a filter of mine.

    The places already in hand are excluded, so `comp_only` cannot be won by re-offering the walk.
    """
    lefts = {T["addrs"][j][1] for j in seen_pids}
    rights = {T["addrs"][j][2] for j in seen_pids}
    out = Counter()
    for l in lefts:
        for r in rights:
            if len(l) != len(r):
                continue
            j = T["of_addr"].get((len(l), l, r))
            if j is not None and j != pid and j not in seen_pids and j not in drop:
                out[j] += 1
    return out


def half_cost(T, pid, j, m):
    """How rare the matched half is - the count that orders the address neighbours.

    A place matched through `the|` is one of thousands; one matched through a half the corpus
    wrote twice is nearly an address on its own. Both are exact class sizes, so the ordering is a
    count and not a weight I chose. Ties fall back to tape order.
    """
    _w, left, right = T["addrs"][pid]
    _wj, lj, rj = T["addrs"][j]
    sizes = []
    if lj == left:
        sizes.append(len(T["by_left"][left]))
    if rj == right:
        sizes.append(len(T["by_right"][right]))
    return min(sizes) if sizes else len(T["places"])


def lane_addr_full(T, nbrs, own, norm_by_places=True):
    """Every value standing at an address neighbour, scored by the summed half-match count.

    Ranked the same two ways as `lane_share`, for the same reason and with the same default.
    """
    score = Counter()
    for j, mm in nbrs.items():
        for v in T["prof"][j]:
            if v not in own:
                score[v] += mm
    if not norm_by_places:
        return [v for v, _n in score.most_common()]
    return sorted(score, key=lambda v: (-(score[v] / len(T["at_value"][v])), -score[v], v))


def addr_order(T, pid, nbrs, k):
    """The k best-named address neighbours: most halves matched, then the RAREST matched half.

    The uncapped lane is a ceiling and reads every place sharing a half - on a common half that
    is hundreds. This is the affordable form, and both are printed so a ceiling cannot be
    mistaken for a channel.
    """
    return sorted(nbrs, key=lambda j: (-nbrs[j], half_cost(T, pid, j, nbrs[j]), j))[:k]


def lane_random(T, pid, own, k_places, rng, drop=()):
    """The floor: the fillers of k random places, taken exactly the way every lane takes them."""
    n = len(T["places"])
    order = [j for j in (rng.randrange(n) for _t in range(k_places))
             if j != pid and j not in drop]
    return fillers_of(T, order, own)


def band_draw(T, truth, banned, rng):
    """373's frequency-matched decoy, unchanged, so the null is matched the way the others are."""
    freq, by_band = T["freq"], T["by_band"]
    b = freq[truth].bit_length()
    src, w = list(by_band.get(b, ())), 0
    while len(src) < 16 and w < 20:
        w += 1
        src += by_band.get(b - w, [])
        src += by_band.get(b + w, [])
    for _t in range(64):
        if not src:
            return None
        v = src[rng.randrange(len(src))]
        if v != truth and v not in banned:
            return v
    return None


def rank_of(seq, v):
    """1-based rank, or 0 for absent - so a mean over hits is read on an absolute quantity."""
    for i, x in enumerate(seq):
        if x == v:
            return i + 1
    return 0


# ------------------------------------------------------------------------------- the population

def measure(T, s, args, rng):
    """ONE QUESTION, every count it produces. Returns None when the question is out of the
    population (no own values, or the truth is among them - where reach is not the question).

    Split out of the loop so `_check390_address.py` can put ONE designed question through the
    same code the corpus run uses, and assert the numbers it must produce. A checker that
    re-implemented the lanes would be checking itself.
    """
    toks, owner = T["toks"], T["owner"]
    pid = T["place_of"][s]
    truth = toks[s]
    own = {toks[x] for x in T["places"][pid] if x != s}
    if not own or truth in own:
        return None
    # THE QUERY PROFILE, HIDDEN ROW REMOVED. Section 27 in one line.
    qprof = Counter(toks[x] for x in T["places"][pid] if x != s)
    drop = set(T["on_line"][owner[s]])            # window artefacts, dropped from every lane
    drop.discard(pid)

    walked = walk_order(T, pid, qprof, args.places, drop)
    st = fillers_of(T, walked, own)
    sh = lane_share(T, pid, own, qprof, drop, norm_by_places=False)
    std_full = interleave(st, sh)
    std8 = std_full[:args.topm]

    hn = half_nbrs(T, pid, drop)
    fn, _dot = filler_nbrs(T, pid, qprof, drop)

    # THE ALPHABET THE MIND COMPOSES FROM: its own address, the places it walked to, and the
    # places its own halves name. Halves from the address neighbours are what makes this a
    # PRODUCT rather than a second sum - without them every left is L0 and every right is R0.
    near_addr = addr_order(T, pid, hn, args.places)
    pool = set(walked) | set(near_addr) | {pid}
    cn = compose_nbrs(T, pid, pool, drop)

    half_full = lane_addr_full(T, hn, own)
    half8 = fillers_of(T, near_addr, own)[:args.topm]
    comp8 = fillers_of(T, addr_order(T, pid, cn, args.places), own)[:args.topm]
    rnd8 = lane_random(T, pid, own, args.places, rng, drop)[:args.topm]

    in_std, in_half, in_comp = truth in std8, truth in half8, truth in comp8
    in_rand = truth in rnd8
    d = band_draw(T, truth, own | {truth}, rng)
    m = {
        "n": 1, "dropped": len(drop),
        "n_half": len(hn), "n_half_new": sum(1 for j in hn if j not in fn), "n_fill": len(fn),
        # STRUCTURAL, AND IT MUST READ ZERO: `frame_keep`'s key is (w, left, right) with
        # |left| = |right| = w, so two different places cannot share BOTH halves - sharing left
        # forces the same width and sharing right as well forces the same address. The counter
        # stays because the ordering rule reads it, and a non-zero here would mean the address
        # key stopped being a bijection and every count in this file changed meaning.
        "both_halves": sum(1 for mm in hn.values() if mm >= 2),
        "n_pool": len(pool), "n_comp": len(cn),
        "n_comp_new": sum(1 for j in cn if j not in fn),
        "std8": int(in_std), "half_any": int(truth in half_full), "half8": int(in_half),
        "comp8": int(in_comp), "rand8": int(in_rand),
        "half_only": int(in_half and not in_std), "comp_only": int(in_comp and not in_std),
        "rand_only": int(in_rand and not in_std), "orc8": int(in_std or in_half or in_comp),
        "d_n": int(d is not None),
        "d_std8": int(d is not None and d in std8), "d_half8": int(d is not None and d in half8),
        "d_half_only": int(d is not None and d in half8 and d not in std8),
        "d_comp_only": int(d is not None and d in comp8 and d not in std8),
    }
    # the lanes themselves, for the checker: a checker that rebuilt them would be checking
    # itself. Ignored by the aggregator, which skips every key starting with an underscore.
    m["_lanes"] = {"std": std8, "half": half8, "comp": comp8, "rand": rnd8}
    m["_rank_half"] = rank_of(half_full, truth) if in_half else 0
    m["_rank_std"] = rank_of(std_full, truth) if in_std else 0
    return m


def run(T, args, rng):
    """One pass over the questions. Every count returned; nothing is decided here."""
    c, ranks = Counter(), {"half": [], "std": []}
    qs = [s for ps in T["places"] for s in ps]
    rng.shuffle(qs)
    for s in qs:
        if c["n"] >= args.max_questions:
            break
        m = measure(T, s, args, rng)
        if m is None:
            continue
        for k, v in m.items():
            if not k.startswith("_"):
                c[k] += v
        if m["_rank_half"]:
            ranks["half"].append(m["_rank_half"])
        if m["_rank_std"]:
            ranks["std"].append(m["_rank_std"])
    return c, ranks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)      # the stage's REACH_K
    ap.add_argument("--topm", type=int, default=8)        # the stage's REACH_CANDS
    ap.add_argument("--max-questions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]

    T = build_tape(lines, args.frame_max, args.min_fillers)
    if not T["places"]:
        print("no tape")
        return 1
    c, ranks = run(T, args, rng)
    n = max(1, c["n"])
    dn = max(1, c["d_n"])

    def f(k, d=n):
        return c[k] / d

    rep = {
        "seed": args.seed, "window_lines": args.window_lines, "places_on_tape": len(T["places"]),
        "n": c["n"], "same_line_dropped": f("dropped"),
        # THE VOID CHECK - read this line first
        "n_half": f("n_half"), "n_half_new": f("n_half_new"), "n_fill": f("n_fill"),
        "new_share": c["n_half_new"] / max(1, c["n_half"]), "both_halves": f("both_halves"),
        "n_comp": f("n_comp"), "n_comp_new": f("n_comp_new"), "n_pool": f("n_pool"),
        # the gate
        "std8": f("std8"), "half8": f("half8"), "comp8": f("comp8"), "rand8": f("rand8"),
        "half_any": f("half_any"), "orc8": f("orc8"),
        "half_only": f("half_only"), "comp_only": f("comp_only"), "rand_only": f("rand_only"),
        "orc_minus_std": f("orc8") - f("std8"),
        # the null
        "d_n": c["d_n"], "d_std8": f("d_std8", dn), "d_half8": f("d_half8", dn),
        "d_half_only": f("d_half_only", dn), "d_comp_only": f("d_comp_only", dn),
        "rank_half_mean": (sum(ranks["half"]) / len(ranks["half"])) if ranks["half"] else 0.0,
        "rank_std_mean": (sum(ranks["std"]) / len(ranks["std"])) if ranks["std"] else 0.0,
    }
    print(f"places {rep['places_on_tape']}  questions {rep['n']}  "
          f"same-line dropped {rep['same_line_dropped']:.2f}/q")
    print("VOID CHECK  n_half {n_half:.2f}  of which unseen by the walk {n_half_new:.2f} "
          "({new_share:.3f})  filler-nbrs {n_fill:.2f}  both halves {both_halves:.2f}"
          .format(**rep))
    print("            n_comp {n_comp:.2f}  new {n_comp_new:.2f}  from a pool of "
          "{n_pool:.1f} places".format(**rep))
    print("REACH@8     std {std8:.4f}  half {half8:.4f}  comp {comp8:.4f}  rand {rand8:.4f}  "
          "half_any {half_any:.4f}".format(**rep))
    print("ONLY        half {half_only:.4f}  comp {comp_only:.4f}  rand {rand_only:.4f}  "
          "oracle-std {orc_minus_std:+.4f}".format(**rep))
    print("DECOY       std {d_std8:.4f}  half {d_half8:.4f}  half_only {d_half_only:.4f}  "
          "comp_only {d_comp_only:.4f}".format(**rep))
    print("RANK        truth in the full address lane {rank_half_mean:.1f}  "
          "in the standing lane {rank_std_mean:.1f}".format(**rep))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
