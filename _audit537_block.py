"""537b: HAVE vs REC-miss. Why extra≈0."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import v1_nodes
from _audit528_step import trials
from _audit532_pool import slice_graph

OUT = Path("results/_stage537_block.json")
STORIES = "data/_tinystories_train.txt"


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

    n_seen = have = recmiss = would = 0
    n_row = 0
    rr = random.Random(args.seed + 17)
    for g in test_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        cache = {}
        for v, rest, held, maj in trials(g, by, mid, rr):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            n511 = v1_nodes(g, by, v, cache, k, high_set)
            rec_set = {c for c in cheap_rec(g, by, v, cache) if c != v}
            have_set = set(n511)
            by[v] = saved
            n_row += 1
            for c in marks.get(v, ()):
                n_seen += 1
                if c in have_set:
                    have += 1
                elif c not in rec_set:
                    recmiss += 1
                else:
                    would += 1

    void = n_seen < 40
    rec = dict(seed=args.seed, corpus=kind, windows=len(graphs),
               n_mark=n_mark, n=n_row, n_seen=n_seen,
               have=have, recmiss=recmiss, would=would,
               p_have=have / max(n_seen, 1),
               p_rec=recmiss / max(n_seen, 1),
               p_add=would / max(n_seen, 1),
               void=bool(void))
    print(f"corpus {kind}  windows {len(graphs)}  marks {n_mark}  "
          f"test {n_row}  mark-hits {n_seen}")
    print(f"HAVE  {have} ({rec['p_have']:.3f})   "
          f"REC-miss {recmiss} ({rec['p_rec']:.3f})   "
          f"ADD {would} ({rec['p_add']:.3f})")
    print(f"VOID {void}")
    if void:
        print("\nVOID: few transferred marks on test v.")
    elif rec["p_have"] > rec["p_rec"]:
        print("\nHAVE blocks: marks already on 511. Next = hop1 teacher, not transfer.")
    elif rec["p_rec"] > rec["p_have"]:
        print("\nREC blocks: marks do not transfer. 538 illegal; fix transfer first.")
    else:
        print("\nSPLIT block. Read both; do not start 538.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
