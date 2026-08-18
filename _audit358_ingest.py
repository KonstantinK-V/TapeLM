"""READING IS WRITING. The dependence wikitext actually has, and the project never used.

357 CLOSED RESOLUTION: gain over the permuted null +0.038 thin and +0.0325 thick - under the
gate, SHRINKING with thickness, and against the plain wide tape the thick setting is negative.
One position, one address costs nothing. That is the third audit in a row (354, 355, 357) where
the substrate under INDEPENDENT cloze questions had nothing more to give.

THE DIAGNOSIS, SAID PLAINLY. 354 and 355 did not close memory. They closed memory ON AN EXAM
WHOSE QUESTIONS ARE INDEPENDENT BY CONSTRUCTION - both drawn from a tape that already contained
everything either question needed. No memory can pay on an i.i.d. exam, and that is a property
of the exam, not of minds. The dependence wikitext DOES have, at exactly one scale, is
SELF-REFERENCE WITHIN A DOCUMENT: an article introduces its subject once and then repeats it,
paraphrases it, pronouns it, for forty lines. The first mention is unanswerable from outside;
every later mention is answerable FROM THE ARTICLE'S OWN EARLIER LINES and often from nowhere
else.

AND THE PROJECT HAS NEVER SEEN THIS, because the write path treats the corpus as one finished
bag of lines. The tape is built once; a document is never NEW; nothing is ever ingested. The
loop everyone kept reaching for is not "remember your answers" (354, lost) - it is READING A
DOCUMENT INTO THE WORLD, line by line, where the mind's work on line k is what makes line k+1
askable at all. The answer does not go into a lens set. IT BECOMES WORLD: the line joins the
corpus, frame_keep counts it, new places exist.

WHY THIS EVADES 354'S CLOSURE. There, tape and question shared one world; a remembered answer
was a lens pointing at content the tape already indexed, so a wrong answer of the same shape
bought the same offer. Here the document is NOT in the old world. Its own earlier lines are the
only place its later holes' fillers stand. The null is the same-shaped prefix of a DIFFERENT
document: same added mass, same added places, wrong content. If ingestion is just
offer-widening the null matches it and this closes like everything else.

WHAT IS MEASURED, torch-free, counts only. For each document (a contiguous run of doc-lines
lines from OUTSIDE the old window) and each line k:

    BASE     tape = old world + line k                       the mind never read the document
    INGEST   tape = old world + doc[:k] + line k             it read every line before this one
    NULL     tape = old world + other-doc[:k] + line k       it read the same AMOUNT of noise

    questions = every token position of line k; a position not on a tape counts 0 for that arm,
    so all three arms share one absolute denominator (the 342a lesson).

    reported pooled and BY THIRDS of k - the signature of real ingestion is a gain that GROWS
    down the document, because more of the article's own paradigm exists to stand on.

  GATE  pooled reach_ingest - reach_null > 0.05, with the late third strictly above the early.

    python _audit358_ingest.py
    python _audit358_ingest.py --docs 20 --doc-lines 16     # quick pass
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage358_ingest.json")


def reach_line(corpus, qline, frame_max, min_fillers, topm):
    """corpus is a list of lines whose LAST line is the question line. Every token of that last
    line is a question; the denominator is all of them, askable or not."""
    keep, toks, owner = tframes.frame_keep(corpus, frame_max, min_fillers)
    n_tok = len(corpus[-1].split())
    if not keep:
        return n_tok, 0, 0
    places = [list(ps) for _a, ps in keep]
    place_of = {}
    for pid, ps in enumerate(places):
        for s in ps:
            place_of[s] = pid
    where = defaultdict(list)
    for ps in places:
        for s in ps:
            where[toks[s]].append(s)
    last = len(corpus) - 1
    mine = [s for s in place_of if owner[s] == last]
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

    askable = reached = 0
    for s in mine:
        pid = place_of[s]
        truth = toks[s]
        own = Counter(toks[x] for x in places[pid])
        own[truth] -= 1
        if own[truth] <= 0:
            del own[truth]
        lens = list(own)[:6]
        if not lens:
            continue
        askable += 1
        ban = Counter(toks[x] for x in places[pid])
        off = Counter()
        for v in lens:
            for w, cnt in co(v).items():
                cnt -= ban.get(w, 0)
                if cnt > 0 and w != v:
                    off[w] += cnt
        if truth in {w for w, _n in off.most_common(topm)}:
            reached += 1
    return n_tok, askable, reached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--doc-lines", type=int, default=24)
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--topm", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)

    start = rng.randrange(max(1, len(lines) - args.window_lines))
    old = lines[start:start + args.window_lines]
    rest = lines[:start] + lines[start + args.window_lines:]
    if len(rest) < 3 * args.doc_lines:
        print("not enough lines outside the window")
        return 1

    def draw_doc():
        a = rng.randrange(len(rest) - args.doc_lines)
        return rest[a:a + args.doc_lines], a

    arms = ("base", "ingest", "null")
    tot = {a: Counter() for a in arms}
    thirds = {a: [Counter(), Counter(), Counter()] for a in arms}
    for _d in range(args.docs):
        doc, a0 = draw_doc()
        while True:
            other, a1 = draw_doc()
            if abs(a1 - a0) > 2 * args.doc_lines:
                break
        for k in range(1, args.doc_lines):
            t3 = min(2, 3 * k // args.doc_lines)
            for arm, prefix in (("base", []), ("ingest", doc[:k]), ("null", other[:k])):
                corpus = old + prefix + [doc[k]]
                n_tok, ask, hit = reach_line(corpus, doc[k], args.frame_max,
                                             args.min_fillers, args.topm)
                for c in (tot[arm], thirds[arm][t3]):
                    c["tok"] += n_tok
                    c["ask"] += ask
                    c["hit"] += hit

    def row(c):
        t = max(1, c["tok"])
        return {"positions": c["tok"], "askable": c["ask"] / t, "reach": c["hit"] / t}

    rep = {"window_lines": args.window_lines, "doc_lines": args.doc_lines, "docs": args.docs,
           "topm": args.topm}
    for a in arms:
        rep[a] = row(tot[a])
        rep[a + "_thirds"] = [row(c) for c in thirds[a]]
    rep["gain"] = rep["ingest"]["reach"] - rep["base"]["reach"]
    rep["gain_over_null"] = rep["ingest"]["reach"] - rep["null"]["reach"]
    late = rep["ingest_thirds"][2]["reach"] - rep["null_thirds"][2]["reach"]
    early = rep["ingest_thirds"][0]["reach"] - rep["null_thirds"][0]["reach"]
    rep["late_minus_early"] = late - early
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    print(f"old world {args.window_lines} lines, {args.docs} documents of {args.doc_lines} "
          f"lines, topm {args.topm}")
    for a in arms:
        r = rep[a]
        th = rep[a + "_thirds"]
        print(f"{a.upper():7s}  reach {r['reach']:.4f}   askable {r['askable']:.4f}   "
              f"by thirds " + "  ".join(f"{x['reach']:.4f}" for x in th))
    print(f"GAIN     over base {rep['gain']:+.4f}   over null {rep['gain_over_null']:+.4f}   "
          f"late-early {rep['late_minus_early']:+.4f}")
    if rep["gain_over_null"] > 0.05 and late > early:
        print("\nINGESTION IS REAL. A document's own earlier lines make its later holes "
              "reachable beyond the same mass of foreign text, and the gain grows down the "
              "document. This is the loop: not remembering answers - READING NEW TEXT INTO THE "
              "WORLD, where the reward for line k is that line k+1 becomes askable. Every "
              "closed single-shot result stands; the mind's next job is to choose what to "
              "ingest and when to trust it, and that is a decision, which is Phi's half.")
    else:
        print("\nINGESTION BUYS NOTHING HERE: a document's own past is worth no more than an "
              "equal mass of noise. Wikitext's self-reference either does not survive the "
              "frame filter or is already covered by the old world. The i.i.d. exam was not "
              "hiding a loop, and the substrate verdict is final at this scale.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
