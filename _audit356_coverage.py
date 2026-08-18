"""HOW MUCH OF THE TEXT DOES THE TAPE HAVE A HOLE FOR? The angle nobody has taken.

354 and 355 closed memory - content and reading both - and that closes the loop as a source of
capability. The rule says the VIEW MUST CHANGE, so here is the change, and it is not another
lever on the same picture.

WHAT WE HAVE MEASURED FOR 350 STEPS: accuracy, CONDITIONAL ON THE HOLE BEING ADDRESSABLE. Every
number in this project - reach, hit, pick, gate, the transplant, the invariant - is computed
over questions drawn from `keep`, and `keep` is the set of positions whose frame RECURS and
which have TWO OR MORE distinct fillers. Positions failing either test are dropped in the write
path and have never appeared in any measurement, as question or as answer.

NOBODY HAS EVER STATED WHAT FRACTION OF THE CORPUS THAT IS. If it is small, then every result
holds inside a sliver, and "knowledge separate, decision separate" is proven for the part of
language that happens to be substitutable - which may be exactly the part that carries no facts.

AND THERE IS A RULE IN THE WRITE PATH THAT PHI NEVER GOT A SAY IN. `frame_keep` reads each
position at its WIDEST RECURRING frame. Widest = fewest, most precise fillers. Narrowest =
most positions addressable, vaguest paradigm. That is a real trade with cardinality on both
sides, it was fixed by fiat at the very first commit, and it is the last decision in the
pipeline still made by the tape instead of by the mind. WIDTH IS COVERAGE. This audit measures
what that fiat costs.

WHAT IS COUNTED, no model, no torch:

    addressable    positions whose frame recurs at some width <= frame_max
    on_tape        addressable AND sitting in a place with >= min_fillers distinct fillers,
                   i.e. the positions that can actually be asked about        THE REAL DENOMINATOR
    by width       where the widest recurring frame lands. Width 1 is the cheap majority
    by rarity      P(on_tape | corpus frequency of the token). THE MONEY ROW: a fact is a rare
                   token. If rare tokens are never on the tape, the knowledge half is a lattice
                   of CATEGORIES by construction, and no amount of mind reaches a fact.

    python _audit356_coverage.py
    python _audit356_coverage.py --frame-max 1     # what the narrowest reading would address
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage356_coverage.json")
BANDS = ((1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 10 ** 9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--corpus", default=str(WIKI))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]

    toks, _owner = [], []
    for line in lines:
        for t in line.split():
            toks.append(t)
    n = len(toks)
    freq = Counter(toks)

    # the widening loop of frame_keep, repeated here so the WIDTH of each position is visible.
    # Checked against frame_keep below: if the two disagree the audit is measuring something
    # else and nothing it prints may be believed.
    best_key, alive = {}, list(range(1, n - 1))
    for w in range(1, args.frame_max + 1):
        pos_of = defaultdict(list)
        for i in alive:
            if i - w < 0 or i + 1 + w > n:
                continue
            pos_of[(tuple(toks[i - w:i]), tuple(toks[i + 1:i + 1 + w]))].append(i)
        nxt = []
        for _k, ps in pos_of.items():
            if len(ps) >= 2:
                for i in ps:
                    best_key[i] = (w, _k[0], _k[1])
                nxt.extend(ps)
        alive = nxt
        if not alive:
            break

    by_addr = defaultdict(list)
    for i, k in best_key.items():
        by_addr[k].append(i)
    kept = {k: ps for k, ps in by_addr.items()
            if len({toks[i] for i in ps}) >= args.min_fillers}
    on_tape = {i for ps in kept.values() for i in ps}

    ref, _t, _o = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    agree = len(ref) == len(kept)

    interior = max(1, n - 2)
    wid = Counter(k[0] for k in best_key.values())
    wid_tape = Counter(best_key[i][0] for i in on_tape)
    fill = Counter(len({toks[i] for i in ps}) for ps in kept.values())

    band = {}
    for lo, hi in BANDS:
        tot = sum(1 for i in range(1, n - 1) if lo <= freq[toks[i]] <= hi)
        got = sum(1 for i in on_tape if lo <= freq[toks[i]] <= hi)
        band[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = {
            "positions": tot, "on_tape": got, "share": got / tot if tot else 0.0}

    hap_all = sum(1 for i in range(1, n - 1) if freq[toks[i]] == 1) / interior
    hap_tape = (sum(1 for i in on_tape if freq[toks[i]] == 1) / len(on_tape)) if on_tape else 0.0

    rep = {"tokens": n, "interior": interior, "types": len(freq),
           "frame_max": args.frame_max, "min_fillers": args.min_fillers,
           "addressable": len(best_key), "addressable_share": len(best_key) / interior,
           "places": len(by_addr), "kept_places": len(kept),
           "on_tape": len(on_tape), "on_tape_share": len(on_tape) / interior,
           "width": dict(sorted(wid.items())), "width_on_tape": dict(sorted(wid_tape.items())),
           "fillers_hist": dict(sorted(fill.items())[:8]),
           "by_frequency": band, "hapax_share_all": hap_all, "hapax_share_on_tape": hap_tape,
           "frame_keep_agrees": agree}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    print(f"corpus       {n} tokens, {len(freq)} types, {len(lines)} lines")
    print(f"CHECK        frame_keep agrees on the place count: {agree} "
          f"({len(ref)} vs {len(kept)})")
    print(f"ADDRESSABLE  {len(best_key)} of {interior} interior positions "
          f"{len(best_key) / interior:.4f}   (frame recurs at some width)")
    print(f"ON TAPE      {len(on_tape)} {len(on_tape) / interior:.4f}   "
          f"({len(kept)} places of {len(by_addr)} addresses survive >= "
          f"{args.min_fillers} fillers)")
    print("WIDTH        " + "  ".join(f"w{w}: {c} ({c / max(1, len(best_key)):.3f})"
                                      for w, c in sorted(wid.items())))
    print("FILLERS      " + "  ".join(f"{k}: {v}" for k, v in sorted(fill.items())[:8]))
    print("BY FREQUENCY of the token standing in the hole - P(on tape):")
    for k, v in band.items():
        print(f"   freq {k:>9}  positions {v['positions']:>8}  on tape {v['on_tape']:>8}  "
              f"{v['share']:.4f}")
    print(f"HAPAX        {hap_all:.4f} of all positions, {hap_tape:.4f} of tape positions")

    rare = band.get("1-1", {}).get("share", 0.0)
    common = band.get("101-inf", {}).get("share", 0.0)
    print()
    if rare < 0.5 * common:
        print(f"THE TAPE IS A LATTICE OF CATEGORIES, NOT OF FACTS. A once-seen token stands on "
              f"the tape {rare:.4f} of the time against {common:.4f} for a common one. The "
              f"write path keeps precisely what SUBSTITUTES, and a fact is what does not. Every "
              f"result in this project is accuracy inside that selection, which is honest and "
              f"is not the same claim as 'the knowledge half holds knowledge'.")
    else:
        print(f"RARITY IS NOT THE FILTER: a once-seen token reaches the tape {rare:.4f} against "
              f"{common:.4f} for a common one. The tape's contents are not biased away from "
              f"facts, and the coverage number {len(on_tape) / interior:.4f} is the honest "
              f"denominator for everything measured so far.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
