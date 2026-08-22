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
v0 = '_stage289_derivation.py'
v1 = (v5('results'), v5('out'))
v2 = v5('out')
v3 = '--reach --tape frames --frame-max 3 --min-mentions 1 --fp hash --write-fp hash --ink mean --write-ink mean --words ascii --reach-no-refuse --reach-lookahead --frame-fp fillers --tape-sample region --objective reward --addresses 1500 --reach-max-q 2000 --import-k 1 --gamma 1.0 --probe-period 1000 --probe-size 60 --cpu'

def done_already(v6):
    return v21(((v19 / f'{v41}stage289_decision_{v6}.json').v40() for v19 in v1 for v41 in ('', '_')))

def run(v7, v8):
    v22('   ' + ' '.v42(v7[2:]), flush=True)
    if not v8:
        return True
    v9 = v23.v23()
    v10 = v33.v24(v7)
    v22(f'   -> exit {v10.v25} in {v23.v23() - v9:.0f}s', flush=True)
    return v10.v25 == 0

def main() -> v4:
    v11 = v34.v26()
    v11.v27('--dims', default='32,64,128,256')
    v11.v27('--seeds', default='1337,2890,4711,8642')
    v11.v27('--steps', type=v4, default=4000)
    v11.v27('--wiki', default='data/_wikitext103_train.txt')
    v11.v27('--news', default='data/_stage254_news.txt')
    v11.v27('--base', default=v3)
    v11.v27('--go', action='store_true')
    v12 = v11.v28()
    v13 = [v4(v35) for v35 in v12.v13.v29(',')]
    v14 = [v4(v35) for v35 in v12.v14.v29(',')]
    v15 = v12.v15.v29()
    v22(f'arm: {v12.v15}\nsizes {v13}   seeds {v14}   steps {v12.v43}')
    v22(f'wiki={v12.v44}   news={v12.v45}')
    v22(f"{('PLAN ONLY - add --go to execute' if not v12.v8 else 'RUNNING')}\n")
    v16 = v17 = v18 = 0
    for v19 in v13:
        for v30 in v14:
            v36 = v2 / f'mind342_d{v19}_s{v30}.pt'
            v46, v47 = (f'342wiki_d{v19}_s{v30}', f'342news_d{v19}_s{v30}')
            if v38(v46) and v36.v40():
                v17 += 1
            else:
                v22(f'[d={v19} s={v30}] wiki, trained, saving the mind')
                v37 = v24([v49.v48, v0, *v15, '--wiki', v12.v44, '--dim', v50(v19), '--seed', v50(v30), '--train-steps', v50(v12.v43), '--save-mind', v50(v36), '--run-tag', v46], v12.v8)
                v16 += v37
                v18 += not v37
                if v12.v8 and (not v37):
                    continue
            if v38(v47):
                v17 += 1
                continue
            v22(f'[d={v19} s={v30}] news, trained natively, wiki mind as rival')
            v37 = v24([v49.v48, v0, *v15, '--wiki', v12.v45, '--dim', v50(v19), '--seed', v50(v30), '--train-steps', v50(v12.v43), '--rival-mind', v50(v36), '--run-tag', v47], v12.v8)
            v16 += v37
            v18 += not v37
        v31 = f'342null_d{v19}_s{v14[0]}'
        if v38(v31):
            v17 += 1
        else:
            v22(f'[d={v19}] news, SHUFFLED TAPE - the null')
            v37 = v24([v49.v48, v0, *v15, '--wiki', v12.v45, '--dim', v50(v19), '--seed', v50(v14[0]), '--train-steps', v50(v12.v43), '--shuffle-tape', '--run-tag', v31], v12.v8)
            v16 += v37
            v18 += not v37
    v22(f'\n{v16} run, {v17} already present, {v18} failed')
    if v12.v8:
        v22('read it with:  python _read342_capacity.py')
    return 1 if v18 else 0
if v20 == '__main__':
    raise v32(v39())