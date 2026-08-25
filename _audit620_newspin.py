"""620: 618 peaked pin + hop1 on news. No new pin rule.

Honest recalculation (2026-08-25): same held-blind peak + two-record holdout
as 618.

Default corpus: data/_stage254_news.txt (254 news, same as 526/528).
GATE / VOID / WIDE as 618. hop2 not measured (619 STOP).

    python _check620_newspin.py
    python _audit620_newspin.py --seed 1337
    python _audit620_newspin.py --seed 8642
    python _audit620_newspin.py --seed 2890
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
from _audit618_peakpin import peak_pin

OUT = Path("results/_stage620_newspin.json")
NEWS = "data/_stage254_news.txt"
K = 3


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
    ap.add_argument("--corpus", default=NEWS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"620 newspin  {path}  {kind}", flush=True)

    n = n_res = n_peak = n_match = n_ref = 0
    petya = 0
    c1 = r1 = 0
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
                ctx_row = set(query["keys"]) | {held_ctx}
                ask_row = set(query["keys"]) | {held_ask}
                adjust_frame_stats(co, df, ctx_row, -1)
                adjust_frame_stats(co, df, ask_row, -1)
                n_use = max(n_fr - 2, 1)
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
                    hat = peak_pin(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )
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
                    if hat == held_ask:
                        cc = [0, 1]
                    else:
                        _fc, cc = walk(
                            pg, hat, held_ask, qi, env_m, mid_set, high_set,
                            co, df, n_use, {pin, hat},
                        )
                    if word_r == held_ask:
                        rr = [0, 1]
                    else:
                        _fr, rr = walk(
                            pg, word_r, held_ask, qi, env_m, mid_set, high_set,
                            co, df, n_use, {pin, word_r},
                        )
                finally:
                    adjust_frame_stats(co, df, ask_row, +1)
                    adjust_frame_stats(co, df, ctx_row, +1)
                n_peak += 1
                n_match += int(hat == held_ctx)
                c1 += cc[1]
                r1 += rr[1]

    def r(x, den):
        return x / den if den else 0.0

    void = n_peak < 40 or n_res < 80
    pin_rate = r(n_peak, n_res)
    cd1, rd1 = r(c1, n_peak), r(r1, n_peak)
    cover = pin_rate * cd1
    wide = pin_rate > 0.80
    gate = (not void) and (not wide) and (cd1 - rd1) > 0.05
    print(
        f"n {n}  res {n_res}  PEAK {n_peak}  pin_rate {pin_rate:.3f}  "
        f"match {r(n_match, n_peak):.3f}  cover {cover:.3f}"
    )
    print(f"PEAK d1 {cd1:.3f}  RAND {rd1:.3f}  d {cd1 - rd1:+.3f}")
    print(f"VOID {void}  WIDE {wide}  GATE {gate}")
    if void:
        print("VOID: news two-extra residual / peaked pin hungry. Not STOP of 618.")
    elif wide:
        print("STOP WIDE: 616 on news.")
    elif gate:
        print("GO XFER: 618 pin+hop1 lives on news. Freeze v1.")
    else:
        print("STOP XFER: peaked pin is stories-local. Then a new truth signal — on news, not stories.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), wide=bool(wide), gate=bool(gate),
        n=n, n_res=n_res, n_peak=n_peak, pin_rate=pin_rate,
        match=r(n_match, n_peak), d1=cd1, rand=rd1,
        delta=cd1 - rd1, cover=cover, petya=r(petya, n),
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
