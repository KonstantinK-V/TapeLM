"""539: rank profile (maj on/off), then role key + order-id value.

Order ids (integers, transferable):
  0 df_up   rare-first (511 default)
  1 n_down  companion count descending
  2 meet_down  meet-score descending
  3 df_down  frequent-first

Teacher residual unchanged (held, not maj). Ceiling: best learned order vs
fixed order 0 on paired co-fired trials (same paired_d as 538).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _audit511_ring import HIGH_DF, cheap_rec, mentions, pick_corpus, rec_all
from _audit518_reldf import pct_band
from _audit527_learn import v1_nodes
from _audit528_step import cover, trials
from _audit532_pool import slice_graph
from _audit534_mark import offer
from _audit538_role import RMAX, role_key

OUT = Path("results/_stage539_order.json")
STORIES = "data/_tinystories_train.txt"
ORDER_NAMES = ("df_up", "n_down", "meet_down", "df_down")
FIXED_ORDER = 0


def meet_score(g, by, v, c, cache):
    s0 = set(cheap_rec(g, by, v, cache))
    s1 = set(cheap_rec(g, by, c, cache))
    m = s0 & s1
    m.discard(v)
    m.discard(c)
    return len(m)


def rec_ordered(g, by, v, cache, oid):
    pairs = rec_all(g, by, v)
    if not pairs:
        return []
    if oid == 0:
        pairs = sorted(pairs, key=lambda cn: g["df"][cn[0]])
    elif oid == 1:
        pairs = sorted(pairs, key=lambda cn: (-cn[1], g["df"][cn[0]]))
    elif oid == 2:
        pairs = sorted(pairs, key=lambda cn: (-meet_score(g, by, v, cn[0], cache),
                                              g["df"][cn[0]]))
    else:
        pairs = sorted(pairs, key=lambda cn: (-g["df"][cn[0]], -cn[1]))
    return [c for c, _ in pairs if g["df"][c] <= HIGH_DF]


def taught_rank(rec, held, maj, skip_maj=True):
    for r in range(1, min(len(rec), RMAX)):
        if rec[r] not in held:
            continue
        if skip_maj and rec[r] == maj:
            continue
        return r
    return None


def peak_ratio(cnt):
    if not cnt:
        return 0.0, 0
    top = cnt.most_common(1)[0]
    total = sum(cnt.values())
    return top[1] / max(total, 1), top[0]


def mode_rank(rank_counts):
    if not rank_counts:
        return 1
    return max(rank_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def mark_for(key, order_q, rank_q, rec, oid):
    if key not in order_q or not order_q[key]:
        return []
    o = oid if oid is not None else max(order_q[key].items(),
                                        key=lambda kv: (kv[1], -kv[0]))[0]
    r = mode_rank(rank_q[key].get(o, {}))
    if r < len(rec):
        return [rec[r]]
    return []


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

    prof_res = Counter()
    prof_maj = Counter()
    order_q = defaultdict(lambda: defaultdict(int))
    rank_q = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
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
            key = role_key(g, by, v, cache, high_set)
            rec0 = rec_ordered(g, by, v, cache, FIXED_ORDER)
            rr = taught_rank(rec0, held, maj, skip_maj=True)
            rm = taught_rank(rec0, held, maj, skip_maj=False)
            if rr is not None:
                prof_res[rr] += 1
            if rm is not None:
                prof_maj[rm] += 1
            for oid in range(4):
                rec = rec_ordered(g, by, v, cache, oid)
                r = taught_rank(rec, held, maj, skip_maj=True)
                if r is not None:
                    order_q[key][oid] += 1
                    rank_q[key][oid][r] += 1
                    n_taught += 1
            by[v] = saved

    pr, tr = peak_ratio(prof_res)
    pm, tm = peak_ratio(prof_maj)

    n_row = key_seen = order_live = fire_t = fire_f = 0
    paired_n = paired_d = 0
    rr = random.Random(args.seed + 17)
    for g in test_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        k = 200.0 / max(g["n"], 1)
        high_set = set(high)
        c511, cm_t, cm_f, cx = {}, {}, {}, {}
        for v, rest, held, maj in trials(g, by, mid, rr):
            saved = by[v]
            by[v] = rest
            for c in (c511, cm_t, cm_f, cx):
                c.pop(v, None)
            n511 = v1_nodes(g, by, v, c511, k, high_set)
            key = role_key(g, by, v, cx, high_set)
            best_oid = None
            if key in order_q and order_q[key]:
                best_oid = max(order_q[key].items(), key=lambda kv: (kv[1], -kv[0]))[0]
            seen_k = best_oid is not None
            rec_best = rec_ordered(g, by, v, cx, best_oid) if seen_k else []
            rec_fix = rec_ordered(g, by, v, cx, FIXED_ORDER)
            mark_t = mark_for(key, order_q, rank_q, rec_best, best_oid)
            mark_f = mark_for(key, order_q, rank_q, rec_fix, FIXED_ORDER)
            live = bool(mark_t)
            nm = offer(g, by, v, cm_t, k, high_set, mark_t)
            nf = offer(g, by, v, cm_f, k, high_set, mark_f)
            by[v] = saved
            n_row += 1
            key_seen += int(seen_k)
            order_live += int(live)
            h5 = n511[0] if n511 else None
            d_t = int((nm[0] if nm else None) in held) - int(h5 in held)
            d_f = int((nf[0] if nf else None) in held) - int(h5 in held)
            ft = nm != n511
            ff = nf != n511
            fire_t += int(ft)
            fire_f += int(ff)
            if ft and ff:
                paired_n += 1
                paired_d += d_t - d_f

    n_keys = len(order_q)
    p_key = key_seen / max(n_row, 1)
    p_live = order_live / max(n_row, 1)
    void = n_keys < 3 or n_row < 20 or fire_t < 40
    gate = (not void) and p_key >= 0.9 and paired_n >= 40 and paired_d > 0
    rec_out = dict(
        seed=args.seed, corpus=kind, windows=args.windows,
        n_keys=n_keys, n_taught=n_taught, n=n_row,
        prof_res=dict(prof_res), prof_maj=dict(prof_maj),
        peak_res=pr, peak_maj=pm, top_res=tr, top_maj=tm,
        key_seen=p_key, order_live=p_live,
        fire_t=fire_t, fire_fix=fire_f,
        paired_n=paired_n, paired_d=paired_d,
        void=bool(void), gate=bool(gate),
    )
    print(f"corpus {kind}  keys {n_keys}  taught {n_taught}  test {n_row}")
    print("PROFILE rank (order=df_up on train)")
    print(f"  residual≠maj  n {sum(prof_res.values())}  peak {pr:.3f} @ r{tr}  {dict(prof_res)}")
    print(f"  with maj      n {sum(prof_maj.values())}  peak {pm:.3f} @ r{tm}  {dict(prof_maj)}")
    if sum(prof_maj.values()) > 0 and pm >= 0.35 and pr < 0.20:
        print("  → maj slice peaked, residual flat: thin arena (436 applies with maj).")
    elif sum(prof_res.values()) > 0 and pr >= 0.25:
        print("  → residual profile peaked: signal exists off maj.")
    else:
        print("  → both flat or sparse: read counts, not peak claim.")
    print(f"KEY   key_seen {p_key:.3f}   order_live {p_live:.3f}")
    print(f"PAIR  vs fixed df_up  n {paired_n}  paired_d {paired_d:+.0f}  "
          f"mean {(paired_d / max(paired_n, 1)):+.4f}")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: few keys or teacher never fired.")
    elif p_key < 0.9:
        print("\nKEY DOES NOT TRAVEL. Coarsen role key.")
    elif paired_n < 40:
        print("\nTHIN PAIR: do not read order pay.")
    elif paired_d > 0:
        print("\nGO ORDER: best order beats fixed df_up on co-fired trials.")
    else:
        print("\nSTOP ORDER: fixed df_up wins or ties on paired trials.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec_out
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
