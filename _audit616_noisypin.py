"""616: 614/615 walk, pin = Petya SEARCH extract (noisy), not oracle apples.

Residual: ask not in Petya first-K extracts (same 614).
Start NOISE = first unique extract of Petya (or REFUSE).
Start RAND  = random mid.
Walk unique+PMI, D=2 (615 hop3 ≈ random).

GATE  noise_d1 − rand_d1 > 0.05
VOID  n_pin < 40  (trials that actually pinned)
DIAG  match = (pin_hat == held_ctx) — 614 recovered only on this slice
      extra2 vs extra2_rand
      refuse rate

If GATE STOP and match-slice still GO: writeback only CONST/peaked extract.
If both STOP: noisy pin is 613. Stay oracle-pin / rare agree.

    python _check616_noisypin.py
    python _audit616_noisypin.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit616_noisypin.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit616_noisypin.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import adjust_frame_stats, prefix_windows
from _audit606_bridge import bands, build_places
from _audit615_depth import extracts, walk

OUT = Path("results/_stage616_noisypin.json")
K = 3


def first_pin(rows, mid_set, high_set, forbid):
    for tok, _bag, _u in rows[:K]:
        if tok is not None and tok in mid_set and tok not in high_set and tok not in forbid:
            return tok
    return None


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
    print(f"616 noisypin  {path}  {kind}  K={K}", flush=True)

    n = n_res = n_pin = n_match = n_ref = 0
    petya = 0
    n_cum = [0] * 3
    r_cum = [0] * 3
    m_cum = [0] * 3
    o_cum = [0] * 3
    n_other = 0
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
                    n += 1
                    n_res += 1
                    hat = first_pin(rows_p, mid_set, high_set, {pin, held_ask})
                    if hat is None:
                        n_ref += 1
                        continue
                    pool_r = [
                        t for t in mid_set
                        if t not in (pin, held_ctx, held_ask, hat) and t not in high_set
                        and len(pg["by_place"].get(t, ())) >= 2
                    ]
                    if not pool_r:
                        continue
                    word_r = rng.choice(pool_r)
                    forbid = {pin, held_ask}
                    _fn, cc = walk(
                        pg, hat, held_ask, qi, env_m, mid_set, high_set,
                        co, df, n_use, forbid | {hat},
                    )
                    _fr, rr = walk(
                        pg, word_r, held_ask, qi, env_m, mid_set, high_set,
                        co, df, n_use, forbid | {word_r},
                    )
                finally:
                    adjust_frame_stats(co, df, qtoks, +1)
                n_pin += 1
                match = hat == held_ctx
                n_match += int(match)
                if not match:
                    n_other += 1
                for d in (1, 2):
                    n_cum[d] += cc[d]
                    r_cum[d] += rr[d]
                    if match:
                        m_cum[d] += cc[d]
                    else:
                        o_cum[d] += cc[d]

    def r(x, den):
        return x / den if den else 0.0

    void = n_pin < 40
    nd1, nd2 = r(n_cum[1], n_pin), r(n_cum[2], n_pin)
    rd1, rd2 = r(r_cum[1], n_pin), r(r_cum[2], n_pin)
    md1 = r(m_cum[1], n_match)
    od1 = r(o_cum[1], n_other)
    e2, e2r = nd2 - nd1, rd2 - rd1
    gate = (not void) and (nd1 - rd1) > 0.05
    print(
        f"n {n}  res {n_res}  pin {n_pin}  refuse {r(n_ref, n_res):.3f}  "
        f"match {r(n_match, n_pin):.3f}"
    )
    print(f"NOISE {nd1:.3f}->{nd2:.3f}  RAND {rd1:.3f}->{rd2:.3f}  d1 {nd1 - rd1:+.3f}")
    print(f"match-slice d1 {md1:.3f}  other-slice d1 {od1:.3f}  extra2 {e2:+.3f} vs {e2r:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: few noisy pins.")
    elif gate:
        print("GO NOISE: Petya extract as pin beats random mid. Writeback can train.")
    else:
        print("STOP NOISE: extract pin ≈ random. Writeback only if extract==ctx (see match-slice).")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n, n_res=n_res, n_pin=n_pin,
        refuse=r(n_ref, n_res), match=r(n_match, n_pin),
        noise=[nd1, nd2], rand=[rd1, rd2], d1=nd1 - rd1,
        match_d1=md1, other_d1=od1, extra2=e2, extra2_rand=e2r,
        petya=r(petya, n),
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
