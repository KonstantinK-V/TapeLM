"""551 sweep: all exits from 550 filter vs 551 rerank.

One tape pass, many ranking arms. Gates unchanged per arm:
  filter/rn_*  RN-GL or CL-GL > 0.05
  void         n < 40 or split < 0.20

Modes:
  gl          511 order baseline
  filter      550 bundle-only rec (CL)
  rn_mg         551 default (-cnt_m, -cnt_g)
  rn_m           cnt_m only
  rn_stable      cnt_m>0 block, GL order inside
  rn_anchor      anchor=lowest-df env token; mates+counts from anchor
  rn_allenv      mates share full env set
  rn_jacc        rank by env overlap fraction per candidate
  rn_union       rec_gl + bundle-only nodes, then rn_mg
  rn_cond        rn_mg when distinctive else gl
  rn_idf         (-cnt_m/df, -cnt_g)
  rn_boost       (-2*cnt_m - cnt_g)
  rn_peaked      rn_mg only when bundle peaked else gl
  rn_allgo       reranked list, always take [:allow] (549 control)
  rn_distgo      rn_mg only on distinctive slice else gl

    python _audit551_sweep.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import cheap_rec, comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit527_learn import allow_of
from _audit550_bundle import rec_bundle

OUT = Path("results/_stage551_sweep.json")

MODES = (
    "gl", "filter", "rn_mg", "rn_m", "rn_stable", "rn_anchor", "rn_allenv",
    "rn_jacc", "rn_union", "rn_cond", "rn_idf", "rn_boost", "rn_peaked",
    "rn_allgo", "rn_distgo",
)


def rec_from(g, by, v, slots, cache):
    saved = by.get(v, [])
    by[v] = list(slots)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    by[v] = saved
    return rec


def peaked_cnt(rec, cnt):
    if not rec:
        return False
    n0 = cnt.get(rec[0], 0)
    n1 = cnt.get(rec[1], 0) if len(rec) > 1 else 0
    return len(rec) == 1 or (n0 > 0 and n1 < 0.5 * n0)


def mates_any(rest, g, v, env_m):
    return [t for t in rest if set(comps(g, t, v)) & env_m]


def mates_allenv(rest, g, v, env):
    return [t for t in rest if env <= set(comps(g, t, v))]


def mates_anchor(rest, g, v, env_m, df):
    anc = min(env_m, key=lambda t: df[t])
    return [t for t in rest if anc in set(comps(g, t, v))], anc


def build_row_from(g, by, v, probe, rest, held, env, env_m, cache, k, high_set, mid_set, mate_mode):
    if mate_mode == "allenv":
        mates = mates_allenv(rest, g, v, env)
    elif mate_mode == "anchor":
        mates, _anc = mates_anchor(rest, g, v, env_m, g["df"])
    else:
        mates = mates_any(rest, g, v, env_m)
    if len(mates) < 2:
        return None
    rec_gl = rec_from(g, by, v, rest, cache)
    rec_fl = rec_bundle(g, v, mates)
    if not rec_gl:
        return None
    allow = allow_of(g, v, k, high_set)
    if v in high_set:
        allow = 1
    cnt_m = Counter()
    cnt_g = Counter()
    for t in mates:
        cnt_m.update(set(comps(g, t, v)))
    for t in rest:
        cnt_g.update(set(comps(g, t, v)))
    jacc = {}
    for c in set(cnt_g) | set(cnt_m):
        hits = sum(1 for t in mates if c in set(comps(g, t, v)))
        jacc[c] = hits / max(len(mates), 1)
    return dict(
        g=g, held=held, allow=allow, rec_gl=rec_gl, rec_fl=rec_fl,
        cnt_m=cnt_m, cnt_g=cnt_g, jacc=jacc, mates=mates, env=env,
        n_rest=len(rest), n_mates=len(mates),
        split=len(mates) < len(rest),
        distinctive=bool(rec_gl) and rec_gl[0] != held,
    )


def build_row(g, by, v, probe, cache, k, high_set, mid_set, rng, mate_mode="any"):
    sl = list(by[v])
    if len(sl) < 8:
        return None
    rest = [t for t in sl if t != probe]
    if len(rest) < 7:
        return None
    frame = list(comps(g, probe, v))
    if len(frame) < 2:
        return None
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set
    if not env_m:
        env_m = env - high_set
    if not env_m:
        return None
    return build_row_from(g, by, v, probe, rest, held, env, env_m, cache, k,
                          high_set, mid_set, mate_mode)


def rank_gl(row):
    return list(row["rec_gl"])


def rank_filter(row):
    return list(row["rec_fl"])


def rank_rn_mg(row):
    cm, cg = row["cnt_m"], row["cnt_g"]
    return sorted(row["rec_gl"], key=lambda c: (-cm[c], -cg[c]))


def rank_rn_m(row):
    cm = row["cnt_m"]
    return sorted(row["rec_gl"], key=lambda c: (-cm[c], row["rec_gl"].index(c)))


def rank_rn_stable(row):
    cm = row["cnt_m"]
    hi = [c for c in row["rec_gl"] if cm[c] > 0]
    lo = [c for c in row["rec_gl"] if cm[c] <= 0]
    return hi + lo


def rank_rn_jacc(row):
    j = row["jacc"]
    return sorted(row["rec_gl"], key=lambda c: (-j[c], row["rec_gl"].index(c)))


def rank_rn_union(row):
    cm, cg = row["cnt_m"], row["cnt_g"]
    seen = set(row["rec_gl"])
    extra = [c for c in row["rec_fl"] if c not in seen]
    rec = list(row["rec_gl"]) + extra
    return sorted(rec, key=lambda c: (-cm[c], -cg[c], rec.index(c)))


def rank_rn_idf(row):
    g, cm, cg = row["g"], row["cnt_m"], row["cnt_g"]
    return sorted(row["rec_gl"],
                  key=lambda c: (-cm[c] / max(g["df"][c], 1), -cg[c]))


def rank_rn_boost(row):
    cm, cg = row["cnt_m"], row["cnt_g"]
    return sorted(row["rec_gl"], key=lambda c: (-2 * cm[c] - cg[c], row["rec_gl"].index(c)))


def rank_for_mode(row, mode):
    if mode == "gl":
        return rank_gl(row)
    if mode == "filter":
        return rank_filter(row)
    if mode == "rn_mg":
        return rank_rn_mg(row)
    if mode == "rn_m":
        return rank_rn_m(row)
    if mode == "rn_stable":
        return rank_rn_stable(row)
    if mode == "rn_jacc":
        return rank_rn_jacc(row)
    if mode == "rn_union":
        return rank_rn_union(row)
    if mode == "rn_idf":
        return rank_rn_idf(row)
    if mode == "rn_boost":
        return rank_rn_boost(row)
    if mode == "rn_anchor":
        return rank_rn_mg(row)
    if mode == "rn_allenv":
        return rank_rn_mg(row)
    if mode == "rn_cond":
        return rank_rn_mg(row) if row["distinctive"] else rank_gl(row)
    if mode == "rn_peaked":
        r = rank_rn_mg(row)
        return r if peaked_cnt(r, row["cnt_m"]) else rank_gl(row)
    if mode == "rn_allgo":
        return rank_rn_mg(row)
    if mode == "rn_distgo":
        return rank_rn_mg(row) if row["distinctive"] else rank_gl(row)
    raise KeyError(mode)


def hit_row(row, mode):
    rec = rank_for_mode(row, mode)
    allow = row["allow"]
    take = rec[:allow]
    held = row["held"]
    extra = take[1:]
    return dict(
        hit=held in take,
        hit0=bool(take) and take[0] == held,
        n_extra=len(extra),
        n_extra_hit=sum(1 for c in extra if c == held),
        set_diff=set(take) != set(row["rec_gl"][:allow]),
    )


def windows(pool, n_win, L, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= L:
            out.append(pool)
        else:
            s0 = rng.randrange(len(pool) - L + 1)
            out.append(pool[s0:s0 + L])
    return out


def collect_rows(lines, args, rng, k_hold=None):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], None
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    k = 200.0 / max(g["n"], 1) if k_hold is None else k_hold
    cache = {}
    rows = []
    for v in mid:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        for probe in sl[: args.cap_probe]:
            rest = [t for t in sl if t != probe]
            if len(rest) < 7:
                continue
            frame = list(comps(g, probe, v))
            if len(frame) < 2:
                continue
            rng.shuffle(frame)
            held, env = frame[0], set(frame[1:])
            env_m = (env & mid_set) - high_set
            if not env_m:
                env_m = env - high_set
            if not env_m:
                continue
            for mm in ("any", "anchor", "allenv"):
                r = build_row_from(g, by, v, probe, rest, held, env, env_m,
                                   cache, k, high_set, mid_set, mm)
                if r is None:
                    continue
                r["mate_mode"] = mm
                rows.append(r)
    return rows, k


def summarize(rows, mode):
    n = len(rows)
    if n == 0:
        return dict(n=0, void=True, gate=False)
    hits = [hit_row(r, mode) for r in rows]
    gl_rows = [hit_row(r, "gl") for r in rows]
    hit = sum(h["hit"] for h in hits) / n
    hit0 = sum(h["hit0"] for h in hits) / n
    hit_gl = sum(h["hit"] for h in gl_rows) / n
    split = sum(r["split"] for r in rows) / n
    set_d = sum(h["set_diff"] for h in hits) / n
    n_ex = sum(h["n_extra"] for h in hits)
    n_exh = sum(h["n_extra_hit"] for h in hits)
    p_extra = n_exh / max(n_ex, 1)
    void = n < 40 or split < 0.20
    if mode == "gl":
        d = 0.0
    else:
        d = hit - hit_gl
    gate = (not void) and d > 0.05
    dist_ix = [i for i, r in enumerate(rows) if r["distinctive"]]
    nd = len(dist_ix)
    if nd:
        d_dist = (sum(hits[i]["hit"] for i in dist_ix) -
                  sum(gl_rows[i]["hit"] for i in dist_ix)) / nd
    else:
        d_dist = 0.0
    fl_hit = None
    if mode != "filter":
        fl = [hit_row(r, "filter") for r in rows]
        fl_hit = sum(h["hit"] for h in fl) / n
    return dict(n=n, hit=hit, hit0=hit0, hit_gl=hit_gl, d_gl=d, d_dist=d_dist,
                n_dist=nd, split=split, set_diff=set_d, p_extra=p_extra,
                fl_hit=fl_hit, void=bool(void), gate=bool(gate))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=12)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    wins = windows(pool, args.n_win, args.window_lines, rng)
    t0 = time.time()
    print(f"551 sweep  corpus={path}  {kind}  modes={len(modes)}", flush=True)

    rows, k0 = [], None
    for lines in wins:
        rs, k = collect_rows(lines, args, rng, k0)
        if k0 is None:
            k0 = k
        rows.extend(rs)
    print(f"raw rows {len(rows)}  k {k0}", flush=True)

    arms = {}
    for mode in modes:
        s = summarize(rows, mode)
        arms[mode] = s
        tag = "GO" if s.get("gate") else ("VOID" if s.get("void") else "stop")
        print(f"  {mode:12s} hit={s.get('hit', 0):.4f}  d={s.get('d_gl', 0):+.4f}  "
              f"p_ex={s.get('p_extra', 0):.3f}  {tag}", flush=True)

    best = max((m for m in modes if m != "gl"),
               key=lambda m: arms[m].get("d_gl", -99))
    best_d = arms[best].get("d_gl", 0)
    n_go = sum(1 for m in modes if arms[m].get("gate"))
    print(f"best {best} d={best_d:+.4f}  gates_pass={n_go}/{len(modes)}", flush=True)

    rec = dict(seed=args.seed, corpus=kind, k=k0, n_rows=len(rows),
               arms=arms, best=best, best_d_gl=best_d, n_gate=n_go,
               elapsed_s=round(time.time() - t0, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
