"""625: DECOMP of honest 624 ceiling. No learner.

DIRECT  extract(place) == held_ask
HOPONLY extract ≠ held, hop1 finds held_ask
REACH   any candidate place
COUNT   one place, max count_key
RAND    one place

Two-row hide, n_fr-2, held not in candidate forbid.

HOPONLY live:  h_ora - max(h_count, h_rand) > 0.05
DIRECT  live:  d_ora - max(d_count, d_rand) > 0.05
VOID n < 40

    python _check625_decomp.py
    python _audit625_decomp.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit625_decomp.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit625_decomp.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places, extract_at
from _audit615_depth import extracts, walk
from _audit624_pick import hide_two

OUT = Path("results/_stage625_decomp.json")
K = 3
CAP = 8


def outcome(pg, place, pin, held_ask, qi, env_m, mid_set, high_set, co, df, n_use):
    tok, _bag, uniq = extract_at(place, pin, env_m, mid_set, co, df, n_use)
    if not uniq or tok is None:
        return 0, 0
    if tok == held_ask:
        return 1, 0
    if tok not in mid_set or tok in high_set:
        return 0, 0
    _f, cc = walk(
        pg, tok, held_ask, qi, env_m, mid_set, high_set,
        co, df, n_use, {pin, tok},
    )
    return 0, int(cc[1])


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
    print(f"625 decomp  {path}  {kind}", flush=True)

    n = n_res = n_live = 0
    petya = 0
    d_ora = h_ora = 0
    d_c = h_c = d_r = h_r = 0
    both = 0
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
                live = False
                any_d = any_h = 0
                cd = ch = rd = rh = 0
                try:
                    rows_p = extracts(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    n += 1
                    if any(tok == held_ask for tok, _b, _u in rows_p[:K]):
                        petya += 1
                        continue
                    n_res += 1
                    cids = [pi for pi in pg["by_place"][pin] if pi != qi]
                    if len(cids) < 2:
                        continue
                    rng.shuffle(cids)
                    use = cids[:CAP]
                    scored = []
                    for pi in use:
                        d, h = outcome(
                            pg, places[pi], pin, held_ask, qi,
                            env_m, mid_set, high_set, co, df, n_use,
                        )
                        any_d |= d
                        any_h |= h
                        scored.append((places[pi]["count_key"], d, h, pi))
                    live = True
                    scored.sort(reverse=True)
                    cd, ch = scored[0][1], scored[0][2]
                    pick_r = rng.choice(scored)
                    rd, rh = pick_r[1], pick_r[2]
                finally:
                    hide_two(co, df, query["keys"], held_ctx, held_ask, +1)
                if not live:
                    continue
                n_live += 1
                d_ora += any_d
                h_ora += any_h
                both += int(any_d and any_h)
                d_c += cd
                h_c += ch
                d_r += rd
                h_r += rh

    def r(x, den):
        return x / den if den else 0.0

    void = n_live < 40
    Do, Ho = r(d_ora, n_live), r(h_ora, n_live)
    Dc, Hc = r(d_c, n_live), r(h_c, n_live)
    Dr, Hr = r(d_r, n_live), r(h_r, n_live)
    h_strong = max(Hc, Hr)
    d_strong = max(Dc, Dr)
    h_live = (not void) and (Ho - h_strong) > 0.05
    d_live = (not void) and (Do - d_strong) > 0.05
    print(f"n {n}  res {n_res}  live {n_live}  petya {r(petya, n):.3f}  both {r(both, n_live):.3f}")
    print(f"DIRECT   ora {Do:.3f}  count {Dc:.3f}  rand {Dr:.3f}  d {Do - d_strong:+.3f}  live {d_live}")
    print(f"HOPONLY  ora {Ho:.3f}  count {Hc:.3f}  rand {Hr:.3f}  d {Ho - h_strong:+.3f}  live {h_live}")
    print(f"VOID {void}")
    if void:
        print("VOID: leftover places hungry.")
    elif h_live:
        print("HOPONLY LIVE: context ceiling exists. Full-feedback rank next, not a new net.")
    elif d_live:
        print("HOPONLY DEAD, DIRECT live: 614–623 context not confirmed. Keep SEARCH+REFUSE.")
    else:
        print("BOTH DEAD vs one-shot rivals. Honest leftover has no extra harvest.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), n=n, n_res=n_res, n_live=n_live,
        petya=r(petya, n), both=r(both, n_live),
        d_ora=Do, d_count=Dc, d_rand=Dr, d_live=bool(d_live),
        h_ora=Ho, h_count=Hc, h_rand=Hr, h_live=bool(h_live),
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
