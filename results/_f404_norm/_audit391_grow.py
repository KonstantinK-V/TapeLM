"""C: SAME QUESTIONS, TAPE 1x/2x/4x/8x. Requirement 4, measured as a count.

369 scaled `--addresses` (more questions at an unchanged walk). That is not this. Section 29.2
asked for the same Phi, the same question set, the tape cut to 1x/2x/4x/8x, accuracy monotone.

No frozen Phi is on disk (365r3 left metrics, not weights). This file therefore freezes the
STANDING OFFER - walk interleaved with connect, cap 8, qprof, same-line drop - which is what Phi
ranks. If the eight do not improve as the tape grows, Phi has nothing to get smarter with, and
requirement 4 is closed by that measurement rather than by substituting 369.

  ONE RUN. Seed 1337. Nested prefixes of the same wiki window: 400 / 800 / 1600 / 3200 lines.
  Questions are drawn on the 1x prefix and replayed by (line, word-in-line) on every larger tape.

  VOID. If a 1x question's hidden token is not at the same slot on a larger tape, the mapping
  is broken and the run is not C.
  GATE. std8 is non-decreasing on 1x -> 2x -> 4x -> 8x. That monotonicity IS the gate.
        A new metric does not decide it.

    python _audit391_grow.py
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
import _audit390_address as A
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage391_grow.json')
SCALES = (1, 2, 4, 8)

def line_starts(lines):
    off = [0]
    for L in lines:
        off.append(off[-1] + len(L.split()))
    return off

def slot_at(lines, li, wi):
    return line_starts(lines)[li] + wi

def collect_1x(T, lines, args, rng):
    """Questions from the 1x tape, identified independently of place ids."""
    starts = line_starts(lines)
    qs = []
    slots = [s for ps in T['places'] for s in ps]
    rng.shuffle(slots)
    for s in slots:
        if len(qs) >= args.max_questions:
            break
        m = A.measure(T, s, args, rng)
        if m is None:
            continue
        li = T['owner'][s]
        wi = s - starts[li]
        qs.append((li, wi, T['toks'][s]))
    return qs

def replay(T, lines, qs, args, rng):
    """The same questions on a (possibly thicker) tape. Broken maps return None."""
    c, broken = (Counter(), 0)
    for li, wi, hidden in qs:
        if li >= len(lines):
            broken += 1
            continue
        s = slot_at(lines, li, wi)
        if s >= len(T['toks']) or T['toks'][s] != hidden:
            broken += 1
            continue
        m = A.measure(T, s, args, rng)
        if m is None:
            continue
        for k, v in m.items():
            if not k.startswith('_'):
                c[k] += v
    c['broken'] = broken
    return c

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=1)
    ap.add_argument('--base-lines', type=int, default=400)
    ap.add_argument('--places', type=int, default=8)
    ap.add_argument('--topm', type=int, default=8)
    ap.add_argument('--max-questions', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--corpus', default=str(WIKI))
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    text = Path(args.corpus).open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    pool = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    pool = pool[:int(0.7 * len(pool))]
    rng = random.Random(args.seed)
    need = args.base_lines * SCALES[-1]
    if len(pool) < need:
        print('corpus shorter than 8x')
        return 1
    s0 = rng.randrange(len(pool) - need)
    full = pool[s0:s0 + need]
    lines_1x = full[:args.base_lines]
    T1 = A.build_tape(lines_1x, args.frame_max, args.min_fillers)
    qs = collect_1x(T1, lines_1x, args, random.Random(args.seed))
    if not qs:
        print('no 1x questions')
        return 1
    rows = []
    for k in SCALES:
        lines = full[:args.base_lines * k]
        T = A.build_tape(lines, args.frame_max, args.min_fillers)
        c = replay(T, lines, qs, args, random.Random(args.seed))
        n = max(1, c['n'])
        row = {'scale': k, 'lines': len(lines), 'places': len(T['places']), 'n': c['n'], 'broken': c['broken'], 'std8': c['std8'] / n, 'half8': c['half8'] / n, 'half_any': c['half_any'] / n}
        rows.append(row)
        print(f"{k}x  lines {row['lines']}  places {row['places']}  n {row['n']}  broken {row['broken']}  std8 {row['std8']:.4f}  half8 {row['half8']:.4f}  half_any {row['half_any']:.4f}")
    stds = [r['std8'] for r in rows]
    broken = sum((r['broken'] for r in rows))
    mono = all((stds[i] <= stds[i + 1] + 1e-12 for i in range(len(stds) - 1)))
    void = broken > 0
    rep = {'seed': args.seed, 'n_1x': len(qs), 'rows': rows, 'std8': stds, 'monotone': mono, 'void': void, 'broken': broken}
    print('VOID mapping broken' if void else 'GATE PASS  std8 monotone' if mono else 'GATE FAIL  std8 not monotone — requirement 4 closed on the offer')
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'wrote {args.out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())