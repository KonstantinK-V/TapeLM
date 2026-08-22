"""537: additive allow. 511 frozen; extras at tail."""
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

OUT = Path("results/_stage537_add.json")
STORIES = "data/_tinystories_train.txt"


def offer_add(g, by, v, cache, k, high_set, marked, n511):
    if v in high_set:
        return list(n511)
    rec_set = {c for c in cheap_rec(g, by, v, cache) if c != v}
    have = set(n511)
    extra = [c for c in marked if c in rec_set and c not in have]
    return list(n511) + extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--window-lines", type=int, default=250)
    ap.add_argument("--windows", type=int, default=16)
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

    n_row = extra_n = hop1_same = 0
    sum_d1 = sum_dg = 0.0
    extra_in_held = 0
    rr = random.Random(args.seed + 17)
    for g in test_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        c511, cm = {}, {}
        for v, rest, held, maj in trials(g, by, mid, rr):
            saved = by[v]
            by[v] = rest
            c511.pop(v, None)
            cm.pop(v, None)
            n511 = v1_nodes(g, by, v, c511, k, high_set)
            nm = offer_add(g, by, v, cm, k, high_set, marks.get(v, ()), n511)
            by[v] = saved
            n_row += 1
            h511 = n511[0] if n511 else None
            hm = nm[0] if nm else None
            hop1_same += int(h511 == hm)
            d1 = cover(nm[:1], held) - cover(n511[:1], held)
            dg = cover(nm, held) - cover(n511, held)
            sum_d1 += d1
            sum_dg += dg
            add = nm[len(n511):]
            if add:
                extra_n += 1
                extra_in_held += int(bool(set(add) & held))

    mean_d1 = sum_d1 / max(n_row, 1)
    mean_dg = sum_dg / max(n_row, 1)
    same = hop1_same / max(n_row, 1)
    void = n_row < 20 or extra_n < 40
    gate = (not void) and (same >= 0.99) and (mean_dg > 0.05)
    rec = dict(seed=args.seed, corpus=kind, windows=len(graphs),
               n_mark=n_mark, n=n_row, extra_n=extra_n,
               extra=extra_n / max(n_row, 1), hop1_same=same,
               mean_d1=mean_d1, mean_dg=mean_dg,
               extra_in_held=extra_in_held,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  W {args.window_lines}  windows {len(graphs)}  "
          f"marks {n_mark}  test {n_row}")
    print(f"extra {extra_n} ({rec['extra']:.3f})  in_held {extra_in_held}  "
          f"hop1 frozen {same:.3f}")
    print(f"Δhop1 {mean_d1:+.4f}  Δallgo {mean_dg:+.4f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: few extra trials.")
    elif same < 0.99:
        print("\nBROKEN: hop1 not frozen.")
    elif gate:
        print("\nGO ADD: extra slot covers held without displacing hop1. 529 kept.")
    else:
        print("\nSTOP: extra at tail does not lift allgo. 536 hop1-gain needed first slot.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[f"{args.seed}_w{args.windows}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
