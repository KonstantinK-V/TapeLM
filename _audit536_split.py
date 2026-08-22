"""536: hop1 split — newnode / reorder / silent. Default stories (534 population)."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import v1_nodes
from _audit528_step import cover, trials
from _audit532_pool import slice_graph
from _audit534_mark import offer

OUT = Path("results/_stage536_split.json")
STORIES = "data/_tinystories_train.txt"


def empty_box():
    return dict(n=0, win1=0, lose1=0, neut1=0, sum_d1=0.0,
                extra_in_held=0, disp_in_held=0)


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

    boxes = dict(newnode=empty_box(), reorder=empty_box(), silent=empty_box())
    extra_n = 0
    sum_d1 = sum_dg = 0.0
    n_row = 0
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
            nm = offer(g, by, v, cm, k, high_set, marks.get(v, ()))
            by[v] = saved
            n_row += 1
            h511 = n511[0] if n511 else None
            hm = nm[0] if nm else None
            d1 = int(hm in held) - int(h511 in held)
            d1_self = (cover(nm[:1], held) if nm else 0.0) - (
                cover(n511[:1], held) if n511 else 0.0)
            dg = cover(nm, held) - cover(n511, held)
            sum_d1 += d1_self
            sum_dg += dg
            s511, sm = set(n511), set(nm)
            newnode = bool(sm - s511)
            changed = nm != n511
            if not changed:
                assert d1_self == 0, "silent offer changed hop1"
                b = boxes["silent"]
            elif newnode:
                b = boxes["newnode"]
                extra_n += 1
                b["extra_in_held"] += int(bool((sm - s511) & held))
                b["disp_in_held"] += int(bool((s511 - sm) & held))
            else:
                b = boxes["reorder"]
            b["n"] += 1
            b["sum_d1"] += d1
            if d1 > 0:
                b["win1"] += 1
            elif d1 < 0:
                b["lose1"] += 1
            else:
                b["neut1"] += 1

    mean_d1 = sum_d1 / max(n_row, 1)
    mean_dg = sum_dg / max(n_row, 1)
    void = n_row < 20 or extra_n < 40
    rec = dict(seed=args.seed, corpus=kind, n_mark=n_mark, n=n_row,
               extra_n=extra_n, extra=extra_n / max(n_row, 1),
               boxes=boxes, mean_d1=mean_d1, mean_dg=mean_dg, void=bool(void))
    print(f"corpus {kind}  marks {n_mark}  test {n_row}  extra {extra_n} "
          f"({rec['extra']:.3f})")
    print(f"SELF  mean_d1 {mean_d1:+.4f}  (534 Δhop1)   mean_dg {mean_dg:+.4f}  (534 Δallgo)")
    for name in ("newnode", "reorder", "silent"):
        b = boxes[name]
        nn = max(b["n"], 1)
        print(f"{name:8} n {b['n']:4}  win {b['win1']} lose {b['lose1']} neut {b['neut1']}"
              f"  sum_d1 {b['sum_d1']:+.1f}  mean {b['sum_d1']/nn:+.4f}")
    print(f"newnode new∈held {boxes['newnode']['extra_in_held']}  "
          f"displaced∈held {boxes['newnode']['disp_in_held']}")
    print(f"VOID {void}")
    if void:
        print("\nVOID: few extra trials. Do not read the split.")
    else:
        n_new, n_re = boxes["newnode"]["n"], boxes["reorder"]["n"]
        s_new, s_re = boxes["newnode"]["sum_d1"], boxes["reorder"]["sum_d1"]
        assert not (n_mark > 0 and n_new == 0), "marks not reaching offer"
        if n_new < 40 or n_re < 40:
            thin = "newnode" if n_new < 40 else "reorder"
            if n_new < 40 and n_re < 40:
                thin = "newnode and reorder"
            print(f"\nTHIN {thin} (newnode {n_new}, reorder {n_re}). No verdict.")
        elif s_new > 0 and s_re < 0:
            print("\nNEWNODE + / REORDER − → additive allow (add, do not displace).")
        elif s_new > 0 and abs(s_new) >= abs(s_re):
            print("\nGAIN IN NEWNODE → density/additive allow, not 535.")
        elif s_re > 0 and abs(s_re) >= abs(s_new):
            print("\nGAIN IN REORDER → 511 hop1 order, extra nodes do not pay. 529 kept.")
        else:
            print("\nSPLIT ACROSS BASKETS. Both halves weak. 529 kept.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
