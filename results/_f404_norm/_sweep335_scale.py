"""How the tape's numbers move with its size - the dimension this project has never varied.

WHY. Everything measured here sits at ONE point: 30 MB, 1500 addresses, a 400-line window. Every
ceiling, every rate, every sentence that begins "the binding constraint is". We have never asked
which of those are asymptotic and which are artefacts of that single point, and three standing
conclusions are exactly the kind that must move with size:

  310  "composition failed and the corpus is at fault"
  324  "memory cannot fill"
  327  "0.807 of the truth is present and we reach 0.217 of it"

THE CURVE THAT DECIDES is the gap `anywhere - union`. If it NARROWS as the tape grows, 0.217 is a
small-scale illness and the cure is more corpus. If it WIDENS, retrieval falls behind knowledge
the faster the more knowledge there is, and an offer of eight enumerated candidates is the wrong
operation no matter how it is tuned - which puts the constraint interface back on the table by
asymptotics rather than by the trigger it was parked on.

Two sweeps, because the tape has two sizes and they are not the same knob:
  --addresses    how many places are kept - the tape's WIDTH at a fixed slice of corpus
  --window-lines how much text the region is drawn from - the tape's DEPTH in the corpus

Both are run through _audit327_presence unchanged, so every point is the same measurement the
single-point number came from. Torch-free; minutes per point.

    python _sweep335_scale.py
    python _sweep335_scale.py --quick        # three points per sweep instead of five
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
OUT = Path('results/_stage335_scale.json')
SRC = Path('results/_stage327_presence.json')
KEYS = ('own', 'walk_k', 'walk_2k', 'shared', 'union_k_shared', 'anywhere', 'present_unreached', 'reach_over_presence', 'places', 'questions')

def one(addresses, window, extra):
    """One point of the sweep: run the audit, read what it wrote."""
    cmd = [sys.executable, '_audit327_presence.py', '--addresses', str(addresses), '--window-lines', str(window)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not SRC.exists():
        print(f'  FAILED addresses={addresses} window={window}\n{r.stdout[-400:]}{r.stderr[-400:]}')
        return None
    d = json.loads(SRC.read_text(encoding='utf-8'))
    return {k: d.get(k) for k in KEYS}

def show(name, knob, rows):
    print(f'\n{name}')
    print(f"  {knob:>8}  {'own':>7} {'walkK':>7} {'union':>7} {'anywhere':>9} {'GAP':>7} {'reached':>8}")
    for v, d in rows:
        if d is None:
            continue
        gap = d['anywhere'] - d['union_k_shared']
        print(f"  {v:>8}  {d['own']:7.4f} {d['walk_k']:7.4f} {d['union_k_shared']:7.4f} {d['anywhere']:9.4f} {gap:7.4f} {d['reach_over_presence']:8.4f}")
    pts = [(v, d['anywhere'] - d['union_k_shared']) for v, d in rows if d]
    if len(pts) >= 2:
        first, last = (pts[0], pts[-1])
        move = last[1] - first[1]
        verdict = 'NARROWS - the offer catches up as the tape grows' if move < -0.01 else 'WIDENS - retrieval falls behind knowledge' if move > 0.01 else 'FLAT - the gap is scale-invariant'
        print(f'  gap {first[1]:.4f} at {first[0]} -> {last[1]:.4f} at {last[0]}   {verdict}')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()
    extra = ['--bytes', str(args.bytes), '--frame-max', str(args.frame_max), '--sample', 'region', '--seed', str(args.seed)]
    addrs = [750, 1500, 3000] if args.quick else [375, 750, 1500, 3000, 6000]
    wins = [200, 400, 800] if args.quick else [100, 200, 400, 800, 1600]
    rep = {'bytes': args.bytes, 'seed': args.seed, 'addresses': {}, 'window_lines': {}}
    rows_a = []
    for a in addrs:
        print(f'addresses={a} ...', flush=True)
        d = one(a, 400, extra)
        rep['addresses'][str(a)] = d
        rows_a.append((a, d))
    rows_w = []
    for w in wins:
        print(f'window={w} ...', flush=True)
        d = one(1500, w, extra)
        rep['window_lines'][str(w)] = d
        rows_w.append((w, d))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    show('WIDTH - how many places are kept (window 400)', 'addresses', rows_a)
    show('DEPTH - how much text the region is drawn from (1500 places)', 'window', rows_w)
    print(f'\nGAP = anywhere - union: what is on the tape and not in front of the mind.')
    print(f'written to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())