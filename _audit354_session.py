"""THE CEILING OF A SESSION. Does answering question k help question k+1?

WHAT 353 SETTLED AND WHAT IT LEFT. Depth is established: two hops inside ONE question, one
price, one head, home kept (353 `--two-way`, max summary). That is a chain ACROSS HOPS. The
loop diagnosis (351) was about a chain ACROSS QUESTIONS, and nothing has touched it:

    question k+1 knows nothing of question k       no state
    the answer goes nowhere                        never written, never read again
    the reward is terminal and per-question        no consequence propagates

324 CLOSED MEMORY AND THE CLOSURE DOES NOT TRANSFER. It measured a PERFECT write-back's
marginal retrieval gain over INDEPENDENT questions - questions drawn at random from the whole
tape, which share nothing, so of course a previous answer bought nothing. It never measured a
SITUATION: several questions out of the same region, where an answer to one is a filler that
the next question's own paradigm does not contain.

WHAT IS MEASURED, torch-free, before anything is built:

    reach_own    the truth in the offer built from the question's OWN rows   (today)
    reach_W      the truth in the offer when the session's PREVIOUS ANSWERS are added as lenses
    chain_only   reached with the session and NOT without it                 THE NUMBER
    reach_R      the same with WRONG answers written back - random fillers from the same places

The budget is matched: the same top-m at the end, so a session cannot win by being offered more.
reach_R IS THE NULL AND IT IS LOAD-BEARING. Adding any lens widens the offer; if wrong answers
buy the same gain, the session is measuring offer size and not memory. The gain that counts is
reach_W - reach_R, not reach_W - reach_own.

W holds the TRUE previous answers - the ceiling of a perfect memory, the job the mind would be
given. If that ceiling is flat, no policy over it can be worth building and 324's closure
stands after all, for a reason 324 never established.

    python _audit354_session.py
    python _audit354_session.py --session 8 --session-lines 40
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage354_session.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--topm", type=int, default=8, help="values the offer keeps. The budget")
    ap.add_argument("--session", type=int, default=6, help="questions in one situation")
    ap.add_argument("--session-lines", type=int, default=40,
                    help="how tight a situation is. 0 = draw from the whole tape, which is "
                         "324's independent-question setting and should reproduce its ~0")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sessions", type=int, default=400)
    ap.add_argument("--corpus", default=str(WIKI))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
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
        keep = [(k, sorted(v)) for k, v in acc.items()
                if len({toks[i] for i in v}) >= args.min_fillers]
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print("no tape")
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
    # which places sit on which line - a SITUATION is a run of lines, not a random draw
    line_of = {}
    for j, ps in enumerate(slots_at):
        line_of[j] = min(owner[s] for s in ps)
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

    c = Counter()
    own_n, w_n = [], []
    for _ in range(args.sessions):
        # ---- draw one situation ---------------------------------------------------------
        if args.session_lines and len(lns) > 1:
            a = rng.randrange(len(lns))
            pool = [j for ln in lns[a:a + args.session_lines] for j in places_on[ln]]
        else:
            pool = list(range(P))
        pool = [j for j in pool if len(vals_at[j]) >= 2]
        if len(pool) < 2:
            continue
        rng.shuffle(pool)
        pool = pool[:args.session]
        qs = [(j, rng.randrange(len(vals_at[j]))) for j in pool]

        W, WR = [], []          # the true answers so far, and the wrong ones (the null)
        for j, i in qs:
            truth = vals_at[j][i]
            own_c = Counter(vals_at[j])
            own_c[truth] -= 1
            if own_c[truth] <= 0:
                del own_c[truth]
            lens = list(own_c)[:6]
            if not lens:
                continue
            mine = set(slots_at[j])

            def offer(vs):
                """the same exclusion every audit makes: this place cannot answer itself,
                at any hop and from any lens."""
                out = Counter()
                here = Counter(toks[s] for s in mine)
                for v in vs:
                    for w, n in co(v).items():
                        if w in here:
                            n -= here[w]
                        if n > 0 and w != v:
                            out[w] += n
                return out

            s_own = offer(lens)
            s_w = offer(lens + [v for v in W if v not in lens])
            s_r = offer(lens + [v for v in WR if v not in lens])
            top = lambda s: {w for w, _n in s.most_common(args.topm)}   # matched budget
            r_own = truth in top(s_own)
            r_w = truth in top(s_w)
            r_r = truth in top(s_r)

            # THE ORACLE, and the first pass did not have it. Pooling every previous answer
            # into one offer measures a memory USED INDISCRIMINATELY, which is not the job this
            # architecture ever gives anything: Phi CHOOSES. The honest ceiling is ONE
            # remembered answer, selected perfectly - and it needs its own null, because with
            # five answers in hand a WRONG one also gets five shots at the top-m.
            o_w = o_r = False
            for w in W:
                if w not in lens and truth in top(offer(lens + [w])):
                    o_w = True
                    break
            for w in WR:
                if w not in lens and truth in top(offer(lens + [w])):
                    o_r = True
                    break

            c["n"] += 1
            c["reach_own"] += r_own
            c["reach_W"] += r_w
            c["reach_R"] += r_r
            c["oracle_W"] += o_w or r_own
            c["oracle_R"] += o_r or r_own
            c["held"] += len(W)
            c["chain_only"] += r_w and not r_own
            c["lost"] += r_own and not r_w
            own_n.append(len(s_own))
            w_n.append(len(s_w))

            W.append(truth)
            others = [v for v in vals_at[j] if v != truth]
            WR.append(rng.choice(others) if others else truth)

    n = max(1, c["n"])
    rep = {
        "places": P, "questions": c["n"], "topm": args.topm, "session": args.session,
        "session_lines": args.session_lines,
        "reach_own": c["reach_own"] / n, "reach_W": c["reach_W"] / n,
        "reach_R": c["reach_R"] / n, "chain_only": c["chain_only"] / n, "lost": c["lost"] / n,
        "offer_own": sum(own_n) / max(1, len(own_n)), "offer_W": sum(w_n) / max(1, len(w_n)),
        "oracle_W": c["oracle_W"] / n, "oracle_R": c["oracle_R"] / n,
        "held": c["held"] / n,
    }
    rep["gain"] = rep["reach_W"] - rep["reach_own"]
    rep["gain_over_null"] = rep["reach_W"] - rep["reach_R"]
    rep["oracle_gain"] = rep["oracle_W"] - rep["oracle_R"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"tape     {P} places, {c['n']} questions, sessions of {args.session} over "
          f"{args.session_lines or 'ALL'} lines, topm {args.topm}")
    print(f"OWN      reach {rep['reach_own']:.4f}   offer {rep['offer_own']:.0f}")
    print(f"SESSION  reach {rep['reach_W']:.4f}   offer {rep['offer_W']:.0f}   "
          f"gain {rep['gain']:+.4f}")
    print(f"NULL     reach {rep['reach_R']:.4f}   (wrong answers written back)   "
          f"gain over null {rep['gain_over_null']:+.4f}")
    print(f"APART    only with the session {rep['chain_only']:.4f}   "
          f"lost to the crowd {rep['lost']:.4f}")
    print(f"ORACLE   ONE answer chosen perfectly {rep['oracle_W']:.4f}   "
          f"null {rep['oracle_R']:.4f}   gain over null {rep['oracle_gain']:+.4f}   "
          f"(choosing among {rep['held']:.1f} held)")
    if rep["oracle_gain"] > 0.05:
        print(f"\nTHE SESSION CARRIES, BUT ONLY UNDER SELECTION. Pooling the answers gains "
              f"{rep['gain_over_null']:+.4f}; CHOOSING ONE gains {rep['oracle_gain']:+.4f} over "
              f"a wrong answer given the same number of shots. That is a well-posed problem - "
              f"pick which remembered answer to read the next question through - and it is the "
              f"kind of problem Phi already solves. 324's closure was scoped to independent "
              f"questions and does not cover it.")
    elif rep["gain_over_null"] > 0.05:
        print(f"\nTHE SESSION CARRIES POOLED but not under selection - unexpected, and the "
              f"audit should be read again before it is believed.")
    else:
        print("\nTHE SESSION DOES NOT CARRY, POOLED OR CHOSEN: a true previous answer buys no "
              "more than a wrong one, at matched budget and given the same number of shots. "
              "The identity of what was answered before carries nothing about the next "
              "question, and 324's closure stands - now for a reason 324 never established.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
