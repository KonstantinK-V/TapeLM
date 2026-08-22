"""536rnd: jitter control — shuffle cheap_rec[:k] vs teacher marks. Same stream as 536."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import v1_nodes
from _audit528_step import cover, trials
from _audit532_pool import slice_graph
from _audit534_mark import offer

OUT = Path("results/_stage536_rnd.json")
STORIES = "data/_tinystories_train.txt"


def empty_box():
    return dict(n=0, win1=0, lose1=0, neut1=0, sum_d1=0.0)


def rnd_marks(g, by, v, cache, teacher_marks, rng):
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    k = len(teacher_marks)
    if k <= 0 or not rec:
        return []
    rec = list(rec)
    rng.shuffle(rec)
    return rec[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=250)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--lines", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=STORIES)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: args.lines]
    rng = random.Random(args.seed)
    graphs = []
    for _ in range(args.windows):
        g, _nL = slice_graph(pool, args.window_lines, rng, args.frame_max, args.min_fillers)
        if g is not None:
            graphs.append(g)
    if len(graphs) < 4:
        print("too few windows")
        return 1
    n_tr = max(2, int(0.7 * args.windows))
    train_g, test_g = graphs[:n_tr], graphs[n_tr:]
    marks = defaultdict(list)
    seen_m = defaultdict(set)
    n_mark = 0
    for g in train_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        cache = {}
        vs = list(mid) + list(high)
        for v, rest, held, maj in trials(g, by, vs, rng):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            nodes = v1_nodes(g, by, v, cache, k, high_set)
            by[v] = saved
            if not nodes:
                continue
            hop1 = nodes[0]
            seen = {hop1} if hop1 in held else set()
            for c in nodes[1:]:
                if c in held and c not in seen and c != maj:
                    if c not in seen_m[v]:
                        seen_m[v].add(c)
                        marks[v].append(c)
                        n_mark += 1
                    seen.add(c)

    boxes = dict(newnode=empty_box(), reorder=empty_box(), silent=empty_box())
    sum_t = sum_r = 0.0
    n_row = extra_t = extra_r = 0
    rr = random.Random(args.seed + 17)
    jr = random.Random(args.seed + 23)
    for g in test_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        c511, ct, cr = {}, {}, {}
        for v, rest, held, maj in trials(g, by, mid, rr):
            saved = by[v]
            by[v] = rest
            c511.pop(v, None)
            ct.pop(v, None)
            cr.pop(v, None)
            n511 = v1_nodes(g, by, v, c511, k, high_set)
            tm = marks.get(v, ())
            nm_t = offer(g, by, v, ct, k, high_set, tm)
            nm_r = offer(g, by, v, cr, k, high_set, rnd_marks(g, by, v, cr, tm, jr))
            by[v] = saved
            n_row += 1
            d1_t = (cover(nm_t[:1], held) if nm_t else 0.0) - (
                cover(n511[:1], held) if n511 else 0.0)
            d1_r = (cover(nm_r[:1], held) if nm_r else 0.0) - (
                cover(n511[:1], held) if n511 else 0.0)
            sum_t += d1_t
            sum_r += d1_r
            s511, sm = set(n511), set(nm_r)
            newnode = bool(sm - s511)
            changed = nm_r != n511
            if sm - s511:
                extra_r += 1
            if set(nm_t) - s511:
                extra_t += 1
            if not changed:
                b = boxes["silent"]
            elif newnode:
                b = boxes["newnode"]
            else:
                b = boxes["reorder"]
            b["n"] += 1
            d1 = int((nm_r[0] if nm_r else None) in held) - int(
                (n511[0] if n511 else None) in held)
            b["sum_d1"] += d1
            if d1 > 0:
                b["win1"] += 1
            elif d1 < 0:
                b["lose1"] += 1
            else:
                b["neut1"] += 1

    mean_t = sum_t / max(n_row, 1)
    mean_r = sum_r / max(n_row, 1)
    void = n_row < 20
    rec = dict(seed=args.seed, corpus=kind, windows=len(graphs),
               n_mark=n_mark, n=n_row,
               teacher_mean_d1=mean_t, rnd_mean_d1=mean_r,
               extra_teacher=extra_t / max(n_row, 1),
               extra_rnd=extra_r / max(n_row, 1),
               boxes=boxes, void=bool(void))
    print(f"corpus {kind}  windows {len(graphs)}  marks {n_mark}  test {n_row}")
    print(f"TEACHER mean_d1 {mean_t:+.4f}  RND mean_d1 {mean_r:+.4f}  "
          f"ratio {mean_r / mean_t if abs(mean_t) > 1e-9 else 0:.2f}")
    for name in ("newnode", "reorder", "silent"):
        b = boxes[name]
        nn = max(b["n"], 1)
        print(f"RND {name:8} n {b['n']:4}  win {b['win1']} lose {b['lose1']} neut {b['neut1']}"
              f"  sum_d1 {b['sum_d1']:+.1f}  mean {b['sum_d1']/nn:+.4f}")
    print(f"VOID {void}")
    if void:
        print("\nVOID: tiny test.")
    elif abs(mean_r - mean_t) < 0.003:
        print("\nRND ≈ TEACHER: jitter only. 538 illegal.")
    elif abs(mean_r) < abs(mean_t) * 0.5:
        print("\nRND ≪ TEACHER: teacher is real, not shuffle noise.")
    else:
        print("\nMIXED: partial teacher signal. Read baskets before 538.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
