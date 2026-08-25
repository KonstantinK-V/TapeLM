"""560: chooser over PLACE / STAR / REFUSE. Counts only.

559 inside: if u_S <= 0.05 on test, VOID — always-557, no mind.

    key = (n_cand 0|1|2+, peaked, width_bin)   no token
    Q[key][P|S|R]  from train gold
    gold: P unique hit, else S unique, else R   (559)

vs always-557 (P iff n_cand==1 else R), vs random, vs null (shuffled gold).

GATE  hit_learn - hit_557 > 0.05  and  hit_learn - hit_rnd > 0.05
VOID  n_test < 40 or u_S <= 0.05

Exam is 557's (held filtered from cand) — not a live policy.

    python _check560_choose.py
    python _audit560_choose.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from _audit511_ring import cheap_rec, comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

OUT = Path("results/_stage560_choose.json")
ACT = ("P", "S", "R")


def rec_from(g, by, v, slots, cache):
    saved = by.get(v, [])
    by[v] = list(slots)
    cache.pop(v, None)
    rec = [c for c in cheap_rec(g, by, v, cache) if c != v]
    by[v] = saved
    return rec


def stand_read(g, by, addr, env_m, held, rng, cap=8):
    sl = [x for x in by.get(addr, [])]
    if len(sl) < 2:
        return False
    rng.shuffle(sl)
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), fr))
    if not scored:
        return False
    scored.sort(key=lambda x: -x[0])
    return held in scored[0][1]


def one_s(g, by, v, s, rest, cache, high_set, mid_set, rng):
    frame = list(comps(g, s, v))
    if len(frame) < 3:
        return None, "frame"
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set
    if not env_m:
        env_m = env - high_set
    if not env_m:
        return None, "env"
    rec_gl = rec_from(g, by, v, rest, cache)
    if not rec_gl:
        return None, "rec"
    jac = []
    for t in rest:
        fr = set(comps(g, t, v))
        ov = len(fr & env_m)
        jac.append((ov / max(len(fr), 1), t, fr))
    if not jac:
        return None, "jac"
    jac.sort(key=lambda x: -x[0])
    peaked = int(len(jac) >= 2 and jac[0][0] - jac[1][0] > 0.05)
    fr_p = jac[0][2]
    if held in fr_p:
        return None, "read_hit"
    cand_p = [c for c in fr_p if c in mid_set and c != v]
    addr_s = rec_gl[0]
    if addr_s == v or addr_s in high_set:
        return None, "no_addr_s"
    hit_s = True if addr_s == held else stand_read(
        g, by, addr_s, env_m, held, rng,
    )
    hit_p = False
    if len(cand_p) == 1:
        hit_p = stand_read(g, by, cand_p[0], env_m, held, rng)
    if hit_p and not hit_s:
        gold = "P"
    elif hit_s and not hit_p:
        gold = "S"
    elif hit_p and hit_s:
        gold = "P"
    else:
        gold = "R"
    nbin = 0 if not cand_p else (1 if len(cand_p) == 1 else 2)
    w = len(by[v])
    wbin = 0 if w < 8 else (1 if w < 20 else 2)
    key = (nbin, peaked, wbin)
    return dict(hit_p=int(hit_p), hit_s=int(hit_s), gold=gold, key=key,
                n_cand=len(cand_p)), None


def one_v(g, by, v, cache, high_set, mid_set, rng, cap):
    sl = list(by[v])
    if len(sl) < 8:
        return [], Counter(slots=1)
    rng.shuffle(sl)
    rows, sk = [], Counter()
    for s in sl[: max(1, min(cap, len(sl)))]:
        rest = [x for x in sl if x != s]
        row, why = one_s(g, by, v, s, rest, cache, high_set, mid_set, rng)
        if row is None:
            sk[why] += 1
        else:
            sk["keep"] += 1
            rows.append(row)
    return rows, sk


def windows(pool, n_win, L, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= L:
            out.append(pool)
        else:
            out.append(pool[rng.randrange(len(pool) - L + 1):][:L])
    return out


def rows_of(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], Counter(nograph=1)
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    cache, rows, sk = {}, [], Counter()
    for v in mid:
        rs, sv = one_v(g, by, v, cache, set(high), set(mid), rng, args.cap_probe)
        rows.extend(rs)
        sk.update(sv)
    return rows, sk


def hit_of(row, a):
    if a == "P":
        return row["hit_p"]
    if a == "S":
        return row["hit_s"]
    return 0


def policy_557(row):
    return "P" if row["n_cand"] == 1 else "R"


def train_q(rows):
    q = defaultdict(lambda: Counter())
    for r in rows:
        q[r["key"]][r["gold"]] += 1
    return q


def pick(q, key, rng):
    bag = q.get(key)
    if not bag:
        return rng.choice(ACT)
    best, sc = [], -1
    for a in ACT:
        v = bag[a]
        if v > sc:
            best, sc = [a], v
        elif v == sc:
            best.append(a)
    return rng.choice(best)


def mean_hit(rows, chooser):
    if not rows:
        return 0.0
    return sum(hit_of(r, chooser(r)) for r in rows) / len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=8)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"560 chooser Q[counts]  corpus={path}  {kind}", flush=True)
    wins = windows(pool, args.n_win, args.window_lines, rng)
    n_tr = max(1, int(0.7 * len(wins)))
    train_rows, test_rows, sk = [], [], Counter()
    for i, lines in enumerate(wins):
        rs, sv = rows_of(lines, args, rng)
        sk.update(sv)
        (train_rows if i < n_tr else test_rows).extend(rs)
    n_te = len(test_rows)
    u_s = (sum(1 for r in test_rows if r["gold"] == "S") / n_te) if n_te else 0.0
    u_p = (sum(1 for r in test_rows if r["gold"] == "P") / n_te) if n_te else 0.0
    print(f"train {len(train_rows)}  test {n_te}  u_P {u_p:.3f}  u_S {u_s:.3f}  "
          f"skip {dict(sk)}", flush=True)
    void = n_te < 40 or u_s <= 0.05
    q = train_q(train_rows)
    rng_te = random.Random(args.seed + 99)
    h_l = mean_hit(test_rows, lambda r: pick(q, r["key"], rng_te))
    h_557 = mean_hit(test_rows, policy_557)
    rng_r = random.Random(args.seed + 7)
    h_rnd = mean_hit(test_rows, lambda r: rng_r.choice(ACT))
    rows_null = []
    golds = [r["gold"] for r in train_rows]
    rng_n = random.Random(args.seed + 3)
    rng_n.shuffle(golds)
    for r, g in zip(train_rows, golds):
        rr = dict(r)
        rr["gold"] = g
        rows_null.append(rr)
    qn = train_q(rows_null)
    rng_c = random.Random(args.seed + 11)
    h_null = mean_hit(test_rows, lambda r: pick(qn, r["key"], rng_c))
    d557, drnd, dnull = h_l - h_557, h_l - h_rnd, h_l - h_null
    gate = (not void) and d557 > 0.05 and drnd > 0.05
    print(f"learn {h_l:.4f}  557 {h_557:.4f}  rnd {h_rnd:.4f}  null {h_null:.4f}")
    print(f"d557 {d557:+.4f}  drnd {drnd:+.4f}  dnull {dnull:+.4f}")
    print(f"n_keys {len(q)}  VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: no STAR-unique mass (or few test). Always-557. No chooser.")
    elif not gate:
        print("\nSTOP: counts do not beat 557. Mind not this key.")
    else:
        print("\nGO CHOOSE. Q[counts] picks STAR/PLACE better than always-557.")
    rec = dict(seed=args.seed, corpus=kind, n_train=len(train_rows), n_test=n_te,
               u_p=u_p, u_s=u_s, n_keys=len(q), skip=dict(sk),
               h_learn=h_l, h_557=h_557, h_rnd=h_rnd, h_null=h_null,
               d557=d557, drnd=drnd, dnull=dnull,
               elapsed_s=round(time.time() - t0, 1),
               void=bool(void), gate=bool(gate))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
