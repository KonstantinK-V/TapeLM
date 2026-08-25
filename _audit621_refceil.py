"""621: ceiling on 618-REFUSE only.

Honest recalculation (2026-08-25): held does not filter leftover doors;
ctx/ask rows held out separately. Direct READ if a door equals ask.

Doors = unique extras of other Petya cards (bags do not vote).
Oracle = any leftover door whose hop1 hits held_ask (cap 6, apples included if present).
RAND  = one leftover door.
apples_in = held_ctx still among leftover — then ceiling is 614, job is RANK.

GATE  ora_d1 − rand_d1 > 0.05
VOID  n_ref < 40 or leftover empty on almost all

    python _check621_refceil.py
    python _audit621_refceil.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit621_refceil.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit621_refceil.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import adjust_frame_stats, prefix_windows
from _audit606_bridge import bands, build_places, place_offer
from _audit615_depth import extracts, walk
from _audit618_peakpin import peak_pin

OUT = Path("results/_stage621_refceil.json")
K = 3
CAP = 6


def leftover_doors(pg, pin, skip, env_m, mid_set, high_set, forbid):
    seen = []
    have = set()
    for pi in pg["by_place"].get(pin, ()):
        if pi == skip:
            continue
        _bag, uniq = place_offer(pg["places"][pi], pin, env_m, mid_set)
        for t in uniq:
            if t in high_set or t in forbid or t in have:
                continue
            if t not in mid_set:
                continue
            have.add(t)
            seen.append(t)
    return seen


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
    print(f"621 refceil  {path}  {kind}", flush=True)

    n = n_res = n_pin = n_ref = n_live = 0
    petya = apples = 0
    o1 = r1 = 0
    n_doors = 0
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
                hit_o = hit_r = 0
                live = False
                ain = False
                nd = 0
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
                    if hat is not None:
                        n_pin += 1
                        continue
                    n_ref += 1
                    doors = leftover_doors(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )
                    nd = len(doors)
                    if not doors:
                        continue
                    live = True
                    ain = held_ctx in doors
                    use = list(doors)
                    if ain and held_ctx in use:
                        use.remove(held_ctx)
                        use = [held_ctx] + use
                    use = use[:CAP]
                    for d in use:
                        if d == held_ask:
                            hit_o = 1
                            break
                        _f, cc = walk(
                            pg, d, held_ask, qi, env_m, mid_set, high_set,
                            co, df, n_use, {pin, d},
                        )
                        if cc[1]:
                            hit_o = 1
                            break
                    word_r = rng.choice(doors)
                    if word_r == held_ask:
                        hit_r = 1
                    else:
                        _fr, rr = walk(
                            pg, word_r, held_ask, qi, env_m, mid_set, high_set,
                            co, df, n_use, {pin, word_r},
                        )
                        hit_r = rr[1]
                finally:
                    adjust_frame_stats(co, df, ask_row, +1)
                    adjust_frame_stats(co, df, ctx_row, +1)
                if not live:
                    continue
                n_live += 1
                apples += int(ain)
                n_doors += nd
                o1 += hit_o
                r1 += hit_r

    def r(x, den):
        return x / den if den else 0.0

    void = n_ref < 40 or n_live < 40
    od1, rd1 = r(o1, n_live), r(r1, n_live)
    gate = (not void) and (od1 - rd1) > 0.05
    print(
        f"n {n}  res {n_res}  pin {n_pin}  refuse {n_ref}  live {n_live}  "
        f"apples_in {r(apples, n_live):.3f}  mean_doors {r(n_doors, n_live):.2f}"
    )
    print(f"ORA d1 {od1:.3f}  RAND {rd1:.3f}  d {od1 - rd1:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: refuse leftover empty/hungry.")
    elif gate:
        print("GO CEIL: leftover doors still beat random. 622 RANK/TRUST has a job.")
    else:
        print("STOP CEIL: leftover ≈ random. Freeze REFUSE. No Φ.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n, n_res=n_res,
        n_pin=n_pin, n_ref=n_ref, n_live=n_live,
        apples_in=r(apples, n_live), mean_doors=r(n_doors, n_live),
        ora=od1, rand=rd1, delta=od1 - rd1, petya=r(petya, n),
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
