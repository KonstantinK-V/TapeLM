"""540: cumulative rank cover — budget lever vs flat-top ceiling.

One pass on train profile (df_up list, maj included, ranks from 0):
  P(any of rec[0..m] in held) as m grows.

Interpretation (§65 permutation under fixed allow):
  0.53 → 0.78 → 0.90  ranks quasi-independent; lever = allow (never tried)
  0.53 → 0.55 → 0.56  same node reshuffled; true ceiling on tape
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from _audit511_ring import HIGH_DF, mentions, pick_corpus, rec_all
from _audit518_reldf import pct_band
from _audit527_learn import allow_of
from _audit528_step import trials
from _audit532_pool import slice_graph
from _audit538_role import RMAX

OUT = Path("results/_stage540_budget.json")
STORIES = "data/_tinystories_train.txt"
R_TOP = 6  # report ranks 0..R_TOP-1 and cumulative to m=R_TOP-1


def rec_df_up(g, by, v, cache):
    pairs = sorted(rec_all(g, by, v), key=lambda cn: g["df"][cn[0]])
    return [c for c, _ in pairs if g["df"][c] <= HIGH_DF]


def cumul_profile(recs_held):
    """recs_held: list of (rec, held_set). Returns marg/cum lists length R_TOP."""
    n = len(recs_held)
    if n == 0:
        z = [0.0] * R_TOP
        return z, z, z, z, 0, 0
    marg = [0] * R_TOP
    cum = [0] * R_TOP
    sub = []
    for rec, held in recs_held:
        hit = [False] * R_TOP
        any_so_far = False
        for m in range(R_TOP):
            if m < len(rec) and rec[m] in held:
                hit[m] = True
            any_so_far = any_so_far or hit[m]
            if any_so_far:
                cum[m] += 1
            if m < len(rec):
                marg[m] += int(rec[m] in held)
        if any(i < len(rec) and rec[i] in held for i in range(R_TOP)):
            sub.append((rec, held))
    marg_p = [marg[m] / n for m in range(R_TOP)]
    cum_p = [cum[m] / n for m in range(R_TOP)]
    n_sub = len(sub)
    if n_sub:
        _, cum_c, _ = _cumul_only(sub)
        marg_c = _marg_only(sub, n_sub)
    else:
        cum_c = [0.0] * R_TOP
        marg_c = [0.0] * R_TOP
    return marg_p, cum_p, marg_c, cum_c, n, n_sub


def _marg_only(recs_held, n):
    marg = [0] * R_TOP
    for rec, held in recs_held:
        for m in range(R_TOP):
            if m < len(rec):
                marg[m] += int(rec[m] in held)
    return [marg[m] / n for m in range(R_TOP)]


def _cumul_only(recs_held):
    n = len(recs_held)
    cum = [0] * R_TOP
    for rec, held in recs_held:
        any_so_far = False
        for m in range(R_TOP):
            if m < len(rec) and rec[m] in held:
                any_so_far = True
            if any_so_far:
                cum[m] += 1
    return None, [cum[m] / n for m in range(R_TOP)], n


def fmt_curve(vals):
    return " → ".join(f"{v:.2f}" for v in vals[:3])


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
    train_g = graphs[:n_tr]

    all_rows = []
    mid_rows = []
    allow_mid = Counter()
    for g in train_g:
        by = mentions(g)
        mid, high, _, _ = pct_band(g, by)
        high_set = set(high)
        k = 200.0 / max(g["n"], 1)
        cache = {}
        for v, rest, held, maj in trials(g, by, list(mid) + list(high), rng):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            rec = rec_df_up(g, by, v, cache)
            if rec:
                all_rows.append((rec, held))
                if v in mid:
                    mid_rows.append((rec, held))
                    allow_mid[allow_of(g, v, k, high_set)] += 1
            by[v] = saved

    marg_a, cum_a, marg_ac, cum_ac, n_a, n_ac = cumul_profile(all_rows)
    marg_m, cum_m, marg_mc, cum_mc, n_m, n_mc = cumul_profile(mid_rows)
    gain_a = cum_a[2] - cum_a[0] if n_a else 0.0
    gain_m = cum_m[2] - cum_m[0] if n_m else 0.0
    gain_ac = cum_ac[2] - cum_ac[0] if n_ac else 0.0
    gain_mc = cum_mc[2] - cum_mc[0] if n_mc else 0.0
    flat_marg_m = (max(marg_mc[:3]) - min(marg_mc[:3]) < 0.08) if n_mc else True
    allow1 = allow_mid.get(1, 0) / max(sum(allow_mid.values()), 1)
    void = n_a < 80
    gate = (not void) and flat_marg_m and (gain_mc >= 0.20 or gain_ac >= 0.20)

    print(f"corpus {kind}  train windows {len(train_g)}  trials all {n_a}  mid {n_m}")
    print("CUMUL  all bands  (maj included, ranks from 0, df_up)")
    print(f"  marginal r0..r2  {fmt_curve(marg_a)}")
    print(f"  cumul    m0..m2  {fmt_curve(cum_a)}   Δ02 {gain_a:+.3f}")
    print(f"  |any hit  n {n_ac}  marg {fmt_curve(marg_ac)}  cumul {fmt_curve(cum_ac)}  Δ02 {gain_ac:+.3f}")
    print("CUMUL  mid only")
    print(f"  allow hist (top) {dict(sorted(allow_mid.items())[:6])}…  allow=1 frac {allow1:.3f}")
    print(f"  marginal r0..r2  {fmt_curve(marg_m)}")
    print(f"  cumul    m0..m2  {fmt_curve(cum_m)}   Δ02 {gain_m:+.3f}")
    print(f"  |any hit  n {n_mc}  marg {fmt_curve(marg_mc)}  cumul {fmt_curve(cum_mc)}  Δ02 {gain_mc:+.3f}")
    if flat_marg_m and gain_mc < 0.08 and gain_ac < 0.08:
        print("  → flat top + flat cumul: permutation under fixed allow (§65).")
    elif flat_marg_m and (gain_mc >= 0.20 or gain_ac >= 0.20):
        print("  → flat marginals, rising cumul: budget lever (allow), not order/rank.")
    else:
        print("  → mixed: read both curves.")
    print(f"VOID {void}   GATE {gate}")
    if void:
        print("\nVOID: too few trials.")
    elif gate:
        print("\nGO ALLOW: flat marginals but cumul rises — sweep allow, not mark/order.")
    else:
        print("\nSTOP CEILING: marginals flat and cumul flat — true tape wall.")
    rec_out = dict(
        seed=args.seed, corpus=kind, windows=args.windows,
        n_all=n_a, n_mid=n_m, n_any_all=n_ac, n_any_mid=n_mc,
        marg_all=marg_a, cum_all=cum_a, gain_all=gain_a,
        marg_any_all=marg_ac, cum_any_all=cum_ac, gain_any_all=gain_ac,
        marg_mid=marg_m, cum_mid=cum_m, gain_mid=gain_m,
        marg_any_mid=marg_mc, cum_any_mid=cum_mc, gain_any_mid=gain_mc,
        allow_mid=dict(allow_mid), allow1_frac=allow1,
        void=bool(void), gate=bool(gate),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec_out
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
