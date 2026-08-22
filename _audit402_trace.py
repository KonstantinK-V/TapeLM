"""THE CEILING OF A CLOSED PATH. Does the trace of the walk say anything the offer does not?

WHAT THIS IS NOT, because three neighbouring operations are already closed:
  * NOT a wider offer. The offer stays eight - 347 measured four times what widening costs, and
    393 measured that committing to one place loses to the merge.
  * NOT a second read that adds candidates. 388's hop2@8 did not beat hop1@8.
  * NOT 387's rerank, which reordered the SAME evidence by a different rule. This brings evidence
    the offer never had: where the walk goes FROM each candidate.

WHAT THE TRACE IS. GPT's understanding does not live in its weights - it lives in the context: the
partial result becomes the next input. This project has no such state at all; every read starts
from zero (30.6). The state that can be carried here without breaking the separation contract is
the PATH, never the values: addresses are structure, values are facts. 351 made the same argument
for "where it stands" - policy state, not a fact.

THE MEASUREMENT. For each candidate c of the question's own eight, step to the place c actually
stands at and walk from THERE. Does that walk come back - is the question's place among the k it
reaches?

    close(c)   the walk from c's place reaches the question's place
    truth      close() on the true candidate
    decoy      close() on a FREQUENCY-MATCHED candidate from the same offer
    random     close() from a random place of the same size          the floor

A path that closes is a consequence of the chain: my place points at c AND c's place points back.
Nothing about the value is carried - only whether the two addresses see each other.

  VOID CHECK, READ FIRST
      `close_rate` overall. If almost every candidate closes, the relation is symmetric by
      construction and carries nothing; if almost none does, there is no signal to rank on.

  GATE
      truth - decoy > 0.05 AND truth - random > 0.05, on 3 of 3 seeds. The decoy is the one that
      matters: it says the truth closes because it is the truth, not because it is frequent.

  If it fails, the trace carries nothing on this tape and "consequence of a chain" is not
  available here - which closes the last untried shape rather than leaving it as a hope.

    python _audit402_trace.py
    python _audit402_trace.py --window-lines 1600
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import _audit390_address as A

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage402_trace.json")


def closes(T, pid, from_pid, hidden, k, drop=()):
    """Does the walk from `from_pid` reach `pid`? The arm's own step - filler overlap, cut at k -
    with ONE correction that decides the whole measurement.

    THE SECTION 27 LEAK, IN THE RETURN DIRECTION. `T["prof"][pid]` still contains the HIDDEN
    token: the question's place holds the truth, that is what makes it the answer. A plain walk
    from the candidate's place would therefore find the question's place THROUGH THE HIDDEN
    TOKEN - the path would close because we hid the answer there, not because the two addresses
    see each other. One mention is subtracted from the target's overlap and from its norm, which
    is the same subtraction `reach_places` already makes on the query side.
    """
    if from_pid is None or from_pid == pid:
        return None
    pf = T["prof"][from_pid]
    dot = Counter()
    for v, cnt in pf.items():
        for j in T["at_value"].get(v, ()):
            if j == from_pid or j in drop:
                continue
            pj = T["prof"][j][v] - (1 if (j == pid and v == hidden) else 0)
            if pj > 0:
                dot[j] += cnt * pj
    qn = math.sqrt(sum(c * c for c in pf.values())) or 1.0

    def norm(j):
        if j != pid:
            return T["norm"][j]
        m = T["prof"][j].get(hidden, 0)
        ss = sum(c * c for c in T["prof"][j].values()) - (2 * m - 1 if m > 0 else 0)
        return math.sqrt(ss) if ss > 0 else 1.0

    order = sorted(dot, key=lambda j: (-(dot[j] / (qn * norm(j))), j))[:k]
    return pid in set(order)


def run(T, args, rng):
    c = Counter()
    toks, place_of, owner = T["toks"], T["place_of"], T["owner"]
    qs = [s for ps in T["places"] for s in ps]
    rng.shuffle(qs)
    for s in qs:
        if c["n"] >= args.max_questions:
            break
        pid = place_of[s]
        truth = toks[s]
        own = {toks[x] for x in T["places"][pid] if x != s}
        if not own or truth in own:
            continue
        qprof = Counter(toks[x] for x in T["places"][pid] if x != s)
        drop = set(T["on_line"][owner[s]])
        drop.discard(pid)
        walked = A.walk_order(T, pid, qprof, args.places, drop)
        # 380'S RULE: A WALKED CANDIDATE KEEPS THE PLACE IT WAS OFFERED FROM. Taking "the first
        # place that holds this value" instead is the exact bug 380 found in `reach_deep`, where
        # a root unrelated to the candidate cost three seeds of hit_of_deep.
        root, cands, seen = {}, [], set(own)
        for j in walked:
            for v, _c in T["prof"][j].most_common():
                if v not in seen:
                    seen.add(v)
                    root[v] = j
                    cands.append(v)
        cands = cands[:args.topm]
        if truth not in cands:
            continue                       # the trace can only rerank what the offer holds
        c["n"] += 1

        def close_of(v):
            return closes(T, pid, root.get(v), truth, args.places, drop)

        got = close_of(truth)
        if got is not None:
            c["truth_n"] += 1
            c["truth"] += int(got)
        d = A.band_draw(T, truth, own | {truth}, rng)
        if d is not None and d in cands:
            g = close_of(d)
            if g is not None:
                c["decoy_n"] += 1
                c["decoy"] += int(g)
        # the floor: a random place of the tape, stepped to and asked the same question
        j = rng.randrange(len(T["places"]))
        if j != pid and j not in drop:
            g = closes(T, pid, j, truth, args.places, drop)
            if g is not None:
                c["rand_n"] += 1
                c["rand"] += int(g)
        for v in cands:
            g = close_of(v)
            if g is not None:
                c["any_n"] += 1
                c["any"] += int(g)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--places", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
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
    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    if not T["places"]:
        print("no tape")
        return 1
    c = run(T, args, rng)

    def rate(k):
        return c[k] / max(1, c[k + "_n"])

    rep = {"seed": args.seed, "n": c["n"], "places": len(T["places"]),
           "close_rate": rate("any"), "truth": rate("truth"), "decoy": rate("decoy"),
           "random": rate("rand"), "truth_n": c["truth_n"], "decoy_n": c["decoy_n"]}
    rep["truth_minus_decoy"] = rep["truth"] - rep["decoy"]
    rep["truth_minus_random"] = rep["truth"] - rep["random"]
    print(f"{rep['places']} places, {rep['n']} questions where the offer holds the truth")
    print(f"VOID CHECK  close_rate {rep['close_rate']:.4f}  <- read first: near 1 or near 0 and "
          f"the relation carries nothing to rank on")
    print(f"CLOSES      truth {rep['truth']:.4f} ({rep['truth_n']})   "
          f"decoy {rep['decoy']:.4f} ({rep['decoy_n']})   random {rep['random']:.4f}")
    print(f"            truth-decoy {rep['truth_minus_decoy']:+.4f}   "
          f"truth-random {rep['truth_minus_random']:+.4f}")
    ok = rep["truth_minus_decoy"] > 0.05 and rep["truth_minus_random"] > 0.05
    rep["gate"] = bool(ok)
    print("\nTHE PATH CLOSES ON THE TRUTH. The trace carries evidence the offer does not, it is "
          "not frequency, and it is structure rather than content - so it can be state without "
          "breaking the separation contract." if ok else
          "\nTHE PATH SAYS NOTHING: " +
          ("the truth closes no more often than a frequency-matched twin. "
           if rep["truth_minus_decoy"] <= 0.05 else "") +
          ("it closes no more often than a random place. " if rep["truth_minus_random"] <= 0.05
           else "") + "The trace is not evidence on this tape.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
