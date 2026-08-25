"""627: chain of pins = peak extra of ALL cards of current pin.

Not leftover rank. Not held_ask as gate.
W = addresses. Stop on tie / no peak.

hop1 = peak of Petya cards          (618)
hop2 = peak of that extra's cards
RAND = peak of a random extra of the same Petya cards

GATE  P(peak2 | peak1) − P(peak2 | rand extra) > 0.05
VOID  n < 40
Honesty: no peak → refuse, no guess.

    python _check627_chain.py
    python _audit627_chain.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit627_chain.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit627_chain.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import add_pairs, prefix_windows
from _audit606_bridge import bands, build_places, extract_at
from _audit624_pick import hide_two

OUT = Path("results/_stage627_chain.json")
D = 3


def peak_extra(pg, pin, skip, env_m, mid_set, high_set, co, df, n_use, forbid):
    vot = defaultdict(int)
    for pi in pg["by_place"].get(pin, ()):
        if pi == skip:
            continue
        tok, _bag, uniq = extract_at(
            pg["places"][pi], pin, env_m, mid_set, co, df, n_use,
        )
        if not uniq or tok is None:
            continue
        if tok in forbid or tok not in mid_set or tok in high_set:
            continue
        vot[tok] += 1
    if not vot:
        return None, ()
    ranked = sorted(vot, key=lambda t: (-vot[t], t))
    top = vot[ranked[0]]
    if sum(1 for t in vot if vot[t] == top) != 1:
        return None, tuple(ranked)
    return ranked[0], tuple(ranked)


def chain(pg, start, skip, env_m, mid_set, high_set, co, df, n_use):
    w = [start]
    cur = start
    for _ in range(D):
        nxt, _cands = peak_extra(
            pg, cur, skip, env_m, mid_set, high_set, co, df, n_use, set(w),
        )
        if nxt is None:
            break
        w.append(nxt)
        cur = nxt
    return w


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
    print(f"627 chainpin  {path}  {kind}", flush=True)

    n = 0
    n_h1 = n_h2 = n_h3 = 0
    n_r1 = n_r2 = 0
    n_direct = n_compose = 0
    refuse1 = 0
    sum_len = 0
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
                hide_two(co, df, query["keys"], held_ctx, held_ask, -1)
                n_use = max(n_fr - 2, 1)
                try:
                    w = chain(
                        pg, pin, qi, env_m, mid_set, high_set, co, df, n_use,
                    )
                    _pk, cands = peak_extra(
                        pg, pin, qi, env_m, mid_set, high_set, co, df, n_use, {pin},
                    )
                    rnd_h2 = 0
                    rnd_h1 = 0
                    if cands:
                        alt = rng.choice(cands)
                        rnd_h1 = 1
                        wr = chain(
                            pg, alt, qi, env_m, mid_set, high_set, co, df, n_use,
                        )
                        rnd_h2 = int(len(wr) >= 2)
                    n += 1
                    sum_len += len(w)
                    if len(w) < 2:
                        refuse1 += 1
                    else:
                        n_h1 += 1
                        n_h2 += int(len(w) >= 3)
                        n_h3 += int(len(w) >= 4)
                        n_direct += int(w[1] == held_ask)
                        n_compose += int(len(w) >= 3 and w[1] == held_ctx and w[2] == held_ask)
                    n_r1 += rnd_h1
                    n_r2 += rnd_h2
                finally:
                    hide_two(co, df, query["keys"], held_ctx, held_ask, +1)

    def r(x, den):
        return x / den if den else 0.0

    void = n < 40
    p_h2 = r(n_h2, n_h1)
    p_r2 = r(n_r2, n_r1)
    gate = (not void) and (p_h2 - p_r2) > 0.05
    print(f"n {n}  h1 {r(n_h1, n):.3f}  refuse1 {r(refuse1, n):.3f}  mean|W| {r(sum_len, n):.2f}")
    print(f"h2|h1 {p_h2:.3f}  h2|rand {p_r2:.3f}  Δ {p_h2 - p_r2:+.3f}  h3|h1 {r(n_h3, n_h1):.3f}")
    print(f"diag  hop1=ask {r(n_direct, n):.3f}  compose ctx→ask {r(n_compose, n):.3f}  (not gated)")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: no pin scenes.")
    elif gate:
        print("GO CHAIN: peak extra's cards peak again more than a random extra of Petya.")
    else:
        print("STOP CHAIN: second pin is not special. Context chain ≈ random next mid.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n,
        h1=r(n_h1, n), refuse1=r(refuse1, n), mean_w=r(sum_len, n),
        h2_given_h1=p_h2, h2_given_rand=p_r2, h3_given_h1=r(n_h3, n_h1),
        hop1_is_ask=r(n_direct, n), compose=r(n_compose, n),
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
