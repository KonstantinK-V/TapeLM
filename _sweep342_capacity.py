"""342a: sweep the MIND'S SIZE, with the invariant tested at every point.

WHY THIS AND WHY NOW. The corpus has been swept in both of its sizes and is not the lever (335).
The mind's size has never been swept at all: the whole reach era is d=32, 5633 parameters, every
run. `--dim` was varied once, long ago, as a matched-parameter control for the max-pool result -
never as a capability axis.

There is a standing rule in HANDOFF that says not to: "more parameters ... capacity is not the
binding constraint and widening it would be fitting". Its evidence was that a DEGENERATE Phi
loses, which shows capacity above zero is needed, not that 5633 is enough. Two measurements now
point the other way: 321 priced a second objective at 4.0x the route and 341 at 3.90x - two
unrelated terms, the same factor, which is what saturated parameters look like.

WIDENING IS NOT FITTING IF THE INVARIANT IS TESTED AT EVERY POINT, and the invariant was never a
parameter count. "The mind holds no facts" is the sentence "a mind fitted here reads a foreign
tape as well as one fitted there, and dies when the tape is shuffled". So each size gets all
three, together, or the point is not read:

  CAPABILITY  route, PICK vs COUNT, GATE-WO       - does a bigger mind decide better
  INVARIANT   native vs transplanted, paired      - 336, in one run, nothing to align
  NULL        --shuffle-tape                      - 328, the signal must die with the tape

  capability rises AND the transplant stays indistinguishable -> capacity was binding
  capability rises AND the transplant gap opens               -> it is buying MEMORY, and that
                                                                 size is the measured limit
  capability flat                                             -> the standing rule was right

TWO RUNS PER (size, seed) plus one null per size:
  1. wiki, trained, --save-mind          the mind to transplant, and the wiki capability point
  2. news, trained natively, --rival-mind <that mind>    capability AND 336's paired control
  3. news, one seed, --shuffle-tape      the null

DRY RUN BY DEFAULT. This is ~36 runs and the base flags below are reconstructed from report
headers, not from a recorded command line - so it prints every command and executes nothing
until --go. Check the first line against your own arm before spending a night on it.

    python _sweep342_capacity.py                          # print the plan
    python _sweep342_capacity.py --go                     # run it (resumable)
    python _sweep342_capacity.py --dims 32,64 --go        # a cheaper first look
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STAGE = "_stage289_derivation.py"
# WHERE THE REPORTS LAND. The stage writes results/stage289_decision_<tag>.json, but the reports
# read in this project arrive as out/_stage289_decision_<tag>.json - a different directory and a
# leading underscore. Rather than guess which machine this is on, the resume check accepts
# either: a sweep that cannot see its own finished runs would redo a night's work in silence.
RES = (Path("results"), Path("out"))
MINDS = Path("out")
# THE ARM, as run for 336-341 on this machine (not the shorter reconstruction in the patch):
#   frames, fillers, lookahead, no_refuse, gamma 1.0, addresses 1500, region sample, cpu
BASE = ("--reach --tape frames --frame-max 3 --min-mentions 1 "
        "--fp hash --write-fp hash --ink mean --write-ink mean --words ascii "
        "--reach-no-refuse --reach-lookahead --frame-fp fillers "
        "--tape-sample region --objective reward --addresses 1500 --reach-max-q 2000 "
        "--import-k 1 --gamma 1.0 --probe-period 1000 --probe-size 60 --cpu")


def done_already(tag):
    return any((d / f"{pre}stage289_decision_{tag}.json").exists()
               for d in RES for pre in ("", "_"))


def run(cmd, go):
    print("   " + " ".join(cmd[2:]), flush=True)
    if not go:
        return True
    t = time.time()
    r = subprocess.run(cmd)
    print(f"   -> exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="32,64,128,256")
    ap.add_argument("--seeds", default="1337,2890,4711,8642")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--wiki", default="data/_wikitext103_train.txt")
    ap.add_argument("--news", default="data/_stage254_news.txt")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    base = args.base.split()
    print(f"arm: {args.base}\nsizes {dims}   seeds {seeds}   steps {args.steps}")
    print(f"wiki={args.wiki}   news={args.news}")
    print(f"{'PLAN ONLY - add --go to execute' if not args.go else 'RUNNING'}\n")

    done = skipped = failed = 0
    for d in dims:
        for s in seeds:
            mind = MINDS / f"mind342_d{d}_s{s}.pt"
            # THE SEED GOES IN THE TAG, because the stage puts only the tag in the filename -
            # two seeds of one size would otherwise write to one path and the second would
            # silently overwrite the first.
            wtag, ntag = f"342wiki_d{d}_s{s}", f"342news_d{d}_s{s}"
            # 1. the wiki mind. Also the wiki capability point, read for free.
            if done_already(wtag) and mind.exists():
                skipped += 1
            else:
                print(f"[d={d} s={s}] wiki, trained, saving the mind")
                ok = run([sys.executable, STAGE, *base, "--wiki", args.wiki,
                          "--dim", str(d), "--seed", str(s),
                          "--train-steps", str(args.steps),
                          "--save-mind", str(mind), "--run-tag", wtag], args.go)
                done += ok
                failed += not ok
                if args.go and not ok:
                    continue
            # 2. news, trained NATIVELY, with the wiki mind as the paired rival. One run gives
            #    the capability point AND 336's control - the two minds answer the same
            #    question on the same tape, so there is nothing to align and nothing to drift.
            if done_already(ntag):
                skipped += 1
                continue
            print(f"[d={d} s={s}] news, trained natively, wiki mind as rival")
            ok = run([sys.executable, STAGE, *base, "--wiki", args.news,
                      "--dim", str(d), "--seed", str(s),
                      "--train-steps", str(args.steps),
                      "--rival-mind", str(mind), "--run-tag", ntag], args.go)
            done += ok
            failed += not ok
        # 3. THE NULL, once per size. It is a null: if the signal survives a shuffled tape the
        #    point is void whatever else it says, and one seed shows that immediately.
        ztag = f"342null_d{d}_s{seeds[0]}"
        if done_already(ztag):
            skipped += 1
        else:
            print(f"[d={d}] news, SHUFFLED TAPE - the null")
            ok = run([sys.executable, STAGE, *base, "--wiki", args.news,
                      "--dim", str(d), "--seed", str(seeds[0]),
                      "--train-steps", str(args.steps),
                      "--shuffle-tape", "--run-tag", ztag], args.go)
            done += ok
            failed += not ok

    print(f"\n{done} run, {skipped} already present, {failed} failed")
    if args.go:
        print("read it with:  python _read342_capacity.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
