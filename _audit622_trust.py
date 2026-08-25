"""622: RANK/TRUST leftover doors on 618-REFUSE.

Score(door) = votes + 2 * reciprocal.
reciprocal = 1 if hop1-extracts of the door hit query env (not held_ask).
TRUST if unique argmax, else REFUSE.
Rival: random leftover; majority = most votes (first on tie).

GATE  trust_d1 − rand_d1 > 0.05   on TRUST trials
VOID  n_trust < 40
WIDE  trust_rate > 0.80

    python _check622_trust.py
    python _audit622_trust.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit622_trust.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit622_trust.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import adjust_frame_stats, prefix_windows
from _audit606_bridge import bands, build_places, place_offer
from _audit615_depth import extracts, walk
from _audit618_peakpin import peak_pin
from _audit621_refceil import leftover_doors

OUT = Path("results/_stage622_trust.json")
K = 3
CAP = 6


def votes_of(pg, pin, skip, env_m, mid_set, high_set, forbid):
    c = Counter()
    for pi in pg["by_place"].get(pin, ()):
        if pi == skip:
            continue
        _bag, uniq = place_offer(pg["places"][pi], pin, env_m, mid_set)
        uniq = [t for t in uniq if t not in high_set and t not in forbid]
        if len(uniq) == 1:
            c[uniq[0]] += 1
    return c


def recip(pg, door, skip, env_m, mid_set, co, df, n_use):
    rows = extracts(pg, door, skip, env_m, mid_set, co, df, n_use)
    return int(any(tok in env_m for tok, _b, _u in rows[:K]))


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
    print(f"622 trust  {path}  {kind}", flush=True)

    n = n_res = n_pin = n_ref = n_live = n_tr = n_rf = 0
    petya = apples = maj_hit = 0
    t1 = r1 = m1 = 0
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
                live = False
                pick = None
                word_r = None
                word_m = None
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
                        pg, pin, qi, env_m, mid_set, high_set, {pin, held_ask},
                    )
                    if hat is not None:
                        n_pin += 1
                        continue
                    n_ref += 1
                    doors = leftover_doors(
                        pg, pin, qi, env_m, mid_set, high_set, {pin, held_ask},
                    )
                    if not doors:
                        continue
                    live = True
                    vot = votes_of(
                        pg, pin, qi, env_m, mid_set, high_set, {pin, held_ask},
                    )
                    use = doors[:CAP]
                    scored = []
                    for d in use:
                        rp = recip(pg, d, qi, env_m, mid_set, co, df, n_use)
                        scored.append((vot[d] + 2 * rp, rp, vot[d], d))
                    scored.sort(reverse=True)
                    best = scored[0][0]
                    tops = [d for s, _rp, _v, d in scored if s == best]
                    if len(tops) == 1:
                        pick = tops[0]
                    word_r = rng.choice(doors)
                    if vot:
                        word_m = vot.most_common(1)[0][0]
                        if word_m not in doors:
                            word_m = doors[0]
                    else:
                        word_m = doors[0]
                    forbid = {pin, held_ask}
                    ht = hr = hm = 0
                    if pick is not None:
                        _ft, ct = walk(
                            pg, pick, held_ask, qi, env_m, mid_set, high_set,
                            co, df, n_use, forbid | {pick},
                        )
                        ht = ct[1]
                    _fr, cr = walk(
                        pg, word_r, held_ask, qi, env_m, mid_set, high_set,
                        co, df, n_use, forbid | {word_r},
                    )
                    hr = cr[1]
                    _fm, cm = walk(
                        pg, word_m, held_ask, qi, env_m, mid_set, high_set,
                        co, df, n_use, forbid | {word_m},
                    )
                    hm = cm[1]
                finally:
                    adjust_frame_stats(co, df, qtoks, +1)
                if not live:
                    continue
                n_live += 1
                if pick is None:
                    n_rf += 1
                    continue
                n_tr += 1
                apples += int(pick == held_ctx)
                maj_hit += int(pick == word_m)
                t1 += ht
                r1 += hr
                m1 += hm

    def r(x, den):
        return x / den if den else 0.0

    void = n_tr < 40
    trust_rate = r(n_tr, n_live)
    td1, rd1, md1 = r(t1, n_tr), r(r1, n_tr), r(m1, n_tr)
    wide = trust_rate > 0.80
    gate = (not void) and (not wide) and (td1 - rd1) > 0.05
    print(
        f"n {n}  refuse {n_ref}  live {n_live}  TRUST {n_tr}  "
        f"rate {trust_rate:.3f}  apples {r(apples, n_tr):.3f}  "
        f"=maj {r(maj_hit, n_tr):.3f}"
    )
    print(
        f"TRUST d1 {td1:.3f}  RAND {rd1:.3f}  MAJ {md1:.3f}  "
        f"dr {td1 - rd1:+.3f}  dm {td1 - md1:+.3f}"
    )
    print(f"VOID {void}  WIDE {wide}  GATE {gate}")
    if void:
        print("VOID: unique-argmax rare — score is a tie machine.")
    elif wide:
        print("STOP WIDE: almost always TRUST. Not a refuse policy.")
    elif gate and (td1 - md1) > 0.05:
        print("GO TRUST: reciprocity ranks leftover better than votes and random.")
    elif gate:
        print("GO vs RAND only — not vs majority votes. Weak rank, Φ still early.")
    else:
        print("STOP: leftover rank ≈ random. Freeze 618 REFUSE. No Φ.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), wide=bool(wide), gate=bool(gate),
        n=n, n_res=n_res, n_pin=n_pin, n_ref=n_ref, n_live=n_live,
        n_trust=n_tr, n_refuse2=n_rf, trust_rate=trust_rate,
        apples=r(apples, n_tr), maj_eq=r(maj_hit, n_tr),
        d1=td1, rand=rd1, maj=md1, d_rand=td1 - rd1, d_maj=td1 - md1,
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
