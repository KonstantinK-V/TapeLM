"""615: 614 + extra hops. Oracle pin still. No Q.

614 GO: hop1 of apples finds tree ~19% vs random ~3%.
Question: does hop2/hop3 of that pin add tree that hop1 missed?

Walk unique+PMI extracts, breadth cap B, depth D=3.
Start CTX = held_ctx (apples). Start RAND = random mid.
Hit = ask appears as extract. Residual = Petya SEARCH missed ask (same as 614).

GATE  extra2 = cum2−cum1 > 0.05  AND  extra2 − extra2_rand > 0.05
      (depth pays above a random chain of the same length)
VOID  n_res < 40
DIAG  cum1 vs 614; extra3; bag not gated.

    python _check615_depth.py
    python _audit615_depth.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit615_depth.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit615_depth.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import adjust_frame_stats, prefix_windows
from _audit606_bridge import bands, build_places, extract_at

OUT = Path("results/_stage615_depth.json")
K = 3
B = 6
D = 3


def extracts(pg, word, skip, env_m, mid_set, co, df, n_use):
    rows = []
    for pi in pg["by_place"].get(word, ()):
        if pi == skip:
            continue
        tok, bag, uniq = extract_at(
            pg["places"][pi], word, env_m, mid_set, co, df, n_use,
        )
        rows.append((tok, bag, bool(uniq)))
    return rows


def walk(pg, start, ask, skip, env_m, mid_set, high_set, co, df, n_use, forbid):
    first = [None] * (D + 1)
    cum = [0] * (D + 1)
    if start not in mid_set or start in high_set:
        return first, cum
    seen = set(forbid) | {start}
    level = [start]
    for depth in range(1, D + 1):
        nxt = []
        hit = False
        for word in level:
            rows = extracts(pg, word, skip, env_m, mid_set, co, df, n_use)
            for tok, _bag, _u in rows[:K]:
                if tok is None or tok not in mid_set or tok in high_set:
                    continue
                if tok == ask:
                    hit = True
                    continue
                if tok not in seen:
                    seen.add(tok)
                    nxt.append(tok)
                if len(nxt) >= B:
                    break
            if len(nxt) >= B:
                break
        if hit and first[depth] is None:
            first[depth] = 1
        for d in range(depth, D + 1):
            if hit:
                cum[d] = 1
        level = nxt[:B]
        if not level:
            break
    return first, cum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=80)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=120000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"615 depth  {path}  {kind}  D={D} K={K} B={B}", flush=True)

    n = n_res = 0
    petya = 0
    c_cum = [0] * (D + 1)
    r_cum = [0] * (D + 1)
    c_first = [0] * (D + 1)
    r_first = [0] * (D + 1)
    for lines in windows:
        pg = build_places(lines, args.frame_max, args.min_fillers)
        if pg is None:
            continue
        mid_set, high_set = bands(pg)
        if not mid_set:
            continue
        co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
        places = pg["places"]
        pins = list(mid_set)
        rng.shuffle(pins)
        for pin in pins:
            qids = list(pg["by_place"].get(pin, ()))
            if len(qids) < 3:
                continue
            rng.shuffle(qids)
            for qi in qids[: args.cap_probe]:
                query = places[qi]
                if pin not in set(query["keys"]):
                    continue
                env = set(query["keys"])
                env_m = (env & mid_set) - high_set or (env - high_set)
                if not env_m:
                    continue
                extras = [
                    tok for tok in dict.fromkeys(query["vals"])
                    if tok in mid_set and tok != pin and tok not in env
                    and tok not in high_set
                ]
                if len(extras) < 2:
                    continue
                rng.shuffle(extras)
                held_ctx, held_ask = extras[0], extras[1]
                if held_ctx not in pg["by_place"] or len(pg["by_place"][held_ctx]) < 2:
                    continue
                qtoks = set(query["keys"]) | {held_ctx, held_ask}
                adjust_frame_stats(co, df, qtoks, -1)
                n_use = max(n_fr - 1, 1)
                try:
                    rows_p = extracts(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    if any(tok == held_ask for tok, _b, _u in rows_p[:K]):
                        n += 1
                        petya += 1
                        continue
                    pool_r = [
                        t for t in mid_set
                        if t not in (pin, held_ctx, held_ask) and t not in high_set
                        and len(pg["by_place"].get(t, ())) >= 2
                    ]
                    if not pool_r:
                        continue
                    word_r = rng.choice(pool_r)
                    forbid = {pin, held_ctx, held_ask}
                    _fc, cc = walk(
                        pg, held_ctx, held_ask, qi, env_m, mid_set, high_set,
                        co, df, n_use, forbid,
                    )
                    _fr, rr = walk(
                        pg, word_r, held_ask, qi, env_m, mid_set, high_set,
                        co, df, n_use, forbid | {word_r},
                    )
                finally:
                    adjust_frame_stats(co, df, qtoks, +1)
                n += 1
                n_res += 1
                for d in range(1, D + 1):
                    c_cum[d] += cc[d]
                    r_cum[d] += rr[d]
                    c_first[d] += int(_fc[d] == 1)
                    r_first[d] += int(_fr[d] == 1)

    def r(x, den):
        return x / den if den else 0.0

    void = n_res < 40
    cum_c = [r(c_cum[d], n_res) for d in range(D + 1)]
    cum_r = [r(r_cum[d], n_res) for d in range(D + 1)]
    e2 = cum_c[2] - cum_c[1]
    e2r = cum_r[2] - cum_r[1]
    e3 = cum_c[3] - cum_c[2]
    gate = (not void) and e2 > 0.05 and (e2 - e2r) > 0.05
    print(
        f"n {n}  res {n_res}  PETYA {r(petya, n):.3f}  "
        f"CTX {cum_c[1]:.3f}→{cum_c[2]:.3f}→{cum_c[3]:.3f}  "
        f"RAND {cum_r[1]:.3f}→{cum_r[2]:.3f}→{cum_r[3]:.3f}"
    )
    print(f"extra2 {e2:+.3f}  extra2_rand {e2r:+.3f}  extra3 {e3:+.3f}")
    print(
        f"first  CTX {r(c_first[1], n_res):.3f}/{r(c_first[2], n_res):.3f}/{r(c_first[3], n_res):.3f}  "
        f"RAND {r(r_first[1], n_res):.3f}/{r(r_first[2], n_res):.3f}/{r(r_first[3], n_res):.3f}"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: thin residual.")
    elif gate:
        print("GO DEPTH: hop2 of oracle pin adds tree above a random chain.")
    else:
        print("STOP DEPTH: extra hops ≈ random walk. 614 hop1 stands; do not train depth.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n, n_res=n_res,
        k=K, b=B, d=D, petya=r(petya, n),
        ctx=cum_c[1:], rand=cum_r[1:], extra2=e2, extra2_rand=e2r, extra3=e3,
        first_ctx=[r(c_first[d], n_res) for d in range(1, D + 1)],
        first_rand=[r(r_first[d], n_res) for d in range(1, D + 1)],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
