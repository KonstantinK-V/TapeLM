"""538: the mark is a ROLE, not a word.

537block measured why the 534 teacher does not travel: p_rec ~ 0.63 against
p_have ~ 0.36 - two thirds of the marks are nodes that do not exist in the test
window at all. A mark keyed by a word cannot cross a window, which is 477 said
again inside the teacher.

So nothing here crosses the window but an integer:

    key    (band, peaked, width)   structural, no word in it
    value  a RANK in the window's own rec list

At test the rank is resolved against the TEST window's rec, so the promoted node
is computed locally and never carried. p_rec cannot be a ceiling: a rank always
exists as long as rec is long enough, and that is counted too (`rank_live`).

Read `key_seen` first - it is what replaces 537block's p_have. Then teacher
against random per firing, which is the quantity 536rnd showed to be the
discriminating one (2-14x there). Baskets are read as MEANS, not sums: 536's
"GAIN IN NEWNODE" was a basket-size artifact.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, mentions, pick_corpus, rec_all
from _audit518_reldf import pct_band
from _audit527_learn import v1_nodes
from _audit528_step import cover, trials
from _audit532_pool import slice_graph
from _audit534_mark import offer

OUT = Path("results/_stage538_role.json")
STORIES = "data/_tinystories_train.txt"
RMAX = 8


def role_key(g, by, v, cache, high_set):
    """No word may appear in the key. Band, peak shape, list width only."""
    band = 1 if v in high_set else 0
    ns = sorted((n for _, n in rec_all(g, by, v)), reverse=True)
    peaked = 1 if len(ns) >= 2 and ns[0] > ns[1] else 0
    w = len(cheap_rec(g, by, v, cache))
    width = 0 if w <= 2 else (1 if w <= 5 else 2)
    return (band, peaked, width)


def rec_of(g, by, v, cache):
    return [c for c in cheap_rec(g, by, v, cache) if c != v]


def taught_rank(rec, held, maj):
    """The rank 534 would have marked: first residual held past hop1, not maj."""
    for r in range(1, min(len(rec), RMAX)):
        if rec[r] in held and rec[r] != maj:
            return r
    return None


def empty_box():
    return dict(n=0, win1=0, lose1=0, neut1=0, sum_d1=0.0)


def add_box(b, d1):
    b["n"] += 1
    b["sum_d1"] += d1
    if d1 > 0:
        b["win1"] += 1
    elif d1 < 0:
        b["lose1"] += 1
    else:
        b["neut1"] += 1


def basket(n511, nm):
    if nm == n511:
        return "silent"
    return "newnode" if set(nm) - set(n511) else "reorder"


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

    Q = defaultdict(lambda: defaultdict(int))
    n_taught = 0
    for g in train_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        high_set = set(high)
        cache = {}
        for v, rest, held, maj in trials(g, by, list(mid) + list(high), rng):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            rec = rec_of(g, by, v, cache)
            key = role_key(g, by, v, cache, high_set)
            r = taught_rank(rec, held, maj)
            by[v] = saved
            if r is not None:
                Q[key][r] += 1
                n_taught += 1
    best = {k: max(d.items(), key=lambda kv: (kv[1], -kv[0]))[0] for k, d in Q.items()}

    boxes = {n: empty_box() for n in ("newnode", "reorder", "silent")}
    rboxes = {n: empty_box() for n in ("newnode", "reorder", "silent")}
    n_row = key_seen = rank_live = fire_t = fire_r = 0
    paired_n = paired_d = 0
    sum_t = sum_r = sum_gt = 0.0
    rr = random.Random(args.seed + 17)
    rnd = random.Random(args.seed + 991)
    for g in test_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        c511, cm_t, cm_r, cx = {}, {}, {}, {}
        for v, rest, held, maj in trials(g, by, mid, rr):
            saved = by[v]
            by[v] = rest
            for c in (c511, cm_t, cm_r, cx):
                c.pop(v, None)
            n511 = v1_nodes(g, by, v, c511, k, high_set)
            rec = rec_of(g, by, v, cx)
            key = role_key(g, by, v, cx, high_set)
            r = best.get(key)
            seen_k = r is not None
            live = seen_k and r < len(rec)
            mark_t = [rec[r]] if live else []
            j = rnd.randrange(1, RMAX)
            mark_r = [rec[j]] if j < len(rec) else []
            nm = offer(g, by, v, cm_t, k, high_set, mark_t)
            nx = offer(g, by, v, cm_r, k, high_set, mark_r)
            by[v] = saved
            n_row += 1
            key_seen += int(seen_k)
            rank_live += int(bool(live))
            h5 = n511[0] if n511 else None
            d_t = int((nm[0] if nm else None) in held) - int(h5 in held)
            d_r = int((nx[0] if nx else None) in held) - int(h5 in held)
            sum_t += d_t
            sum_r += d_r
            sum_gt += cover(nm, held) - cover(n511, held)
            add_box(boxes[basket(n511, nm)], d_t)
            add_box(rboxes[basket(n511, nx)], d_r)
            ft = nm != n511
            fr = nx != n511
            fire_t += int(ft)
            fire_r += int(fr)
            if ft and fr:
                paired_n += 1
                paired_d += d_t - d_r

    n_keys = len(Q)
    p_key = key_seen / max(n_row, 1)
    p_live = rank_live / max(n_row, 1)
    mean_t = sum_t / max(n_row, 1)
    mean_r = sum_r / max(n_row, 1)
    shot_t = sum_t / max(fire_t, 1)
    shot_r = sum_r / max(fire_r, 1)
    mass_t = shot_t * fire_t
    mass_r = shot_r * fire_r
    void = n_keys < 3 or n_row < 20 or fire_t < 40
    gate = (not void) and p_key >= 0.9 and paired_n >= 40 and paired_d > 0
    rec_out = dict(seed=args.seed, corpus=kind, windows=args.windows,
                   n_keys=n_keys, n_taught=n_taught, n=n_row,
                   key_seen=p_key, rank_live=p_live,
                   fire_t=fire_t, fire_r=fire_r,
                   paired_n=paired_n, paired_d=paired_d,
                   mean_d1=mean_t, rnd_mean_d1=mean_r,
                   shot_t=shot_t, shot_r=shot_r,
                   mass_t=mass_t, mass_r=mass_r,
                   mean_dg=sum_gt / max(n_row, 1),
                   boxes=boxes, rboxes=rboxes,
                   void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  keys {n_keys}  taught {n_taught}  test {n_row}")
    print(f"KEY   key_seen {p_key:.3f}  (537b p_have 0.36)   rank_live {p_live:.3f}")
    print(f"PAY   teacher fire {fire_t} mass {mass_t:+.1f} shot {shot_t:+.4f}")
    print(f"      random  fire {fire_r} mass {mass_r:+.1f} shot {shot_r:+.4f}")
    print(f"PAIR  n {paired_n}  paired_d {paired_d:+.0f}  "
          f"mean {(paired_d / max(paired_n, 1)):+.4f}")
    for name in ("newnode", "reorder", "silent"):
        b, rb = boxes[name], rboxes[name]
        print(f"{name:8} T n {b['n']:4} mean {b['sum_d1']/max(b['n'],1):+.4f}   "
              f"R n {rb['n']:4} mean {rb['sum_d1']/max(rb['n'],1):+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: fewer than 3 role keys, or the teacher never fired.")
    elif p_key < 0.9:
        print("\nKEY DOES NOT TRAVEL EITHER. The role is too fine; coarsen it.")
    elif paired_n < 40:
        print("\nTHIN PAIR: fewer than 40 co-fired trials. Do not read pay.")
    elif paired_d > 0:
        print("\nGO PAIR: teacher beats random on the same firings. Key travels and pays.")
    else:
        print("\nSTOP: paired_d <= 0. Frequency-only or random wins; fix gate not rank value.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec_out
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
