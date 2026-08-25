"""614: CONTEXT-PAY ceiling. Oracle pin = the other extra of the same frame.

Honest recalculation (2026-08-25): two fillers of one address are two records;
ctx and ask rows are removed from co+df separately (n_fr-2). No invented
ctx-ask joint row.

Hop1 (Petya SEARCH) vs next-token: not the exam. Context is hop1 of APPLES
after the pin, looking for TREE that Petya-extracts missed.

Query frame has pin in keys and ≥2 mid extras: ctx, ask.
Residual: ask not in first K extracts of the pin.
CTX    SEARCH places of ctx (oracle pin apples)
RAND   SEARCH places of a random mid word
PETYA  SEARCH pin (should miss ask on residual by construction)

GATE  CTX − RAND > 0.05 on residual
VOID  n_res < 40
DIAG  bag-hit (lottery) printed, not gated — 613 STOP was bag==rand.

If STOP: even true apples do not open tree. Φ / writeback have nothing to learn.
If GO: sequential hop1 (writeback) can pay; then compete with next-token on
the second slot, not the first.

    python _check614_ctx.py
    python _audit614_ctx.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit614_ctx.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit614_ctx.py --seed 2890 --corpus data/_tinystories_train.txt
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

OUT = Path("results/_stage614_ctx.json")
K = 3


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


def hit_extract(rows, ask):
    return any(tok == ask for tok, _bag, _u in rows[:K])


def hit_bag(rows, ask, mid_set):
    for _tok, bag, _u in rows[:K]:
        if ask in bag and ask in mid_set:
            return True
    return False


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
    print(f"614 ctx  {path}  {kind}  windows={len(windows)}  K={K}", flush=True)

    n = n_res = 0
    ctx = rnd = petya = 0
    bag_c = bag_r = 0
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
                ctx_row = set(query["keys"]) | {held_ctx}
                ask_row = set(query["keys"]) | {held_ask}
                adjust_frame_stats(co, df, ctx_row, -1)
                adjust_frame_stats(co, df, ask_row, -1)
                n_use = max(n_fr - 2, 1)
                try:
                    rows_p = extracts(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    rows_c = extracts(
                        pg, held_ctx, qi, env_m, mid_set, co, df, n_use,
                    )
                    pool_r = [
                        t for t in mid_set
                        if t not in (pin, held_ctx, held_ask) and t not in high_set
                        and len(pg["by_place"].get(t, ())) >= 2
                    ]
                    if not pool_r:
                        continue
                    word_r = rng.choice(pool_r)
                    rows_r = extracts(
                        pg, word_r, qi, env_m, mid_set, co, df, n_use,
                    )
                finally:
                    adjust_frame_stats(co, df, ask_row, +1)
                    adjust_frame_stats(co, df, ctx_row, +1)
                n += 1
                if hit_extract(rows_p, held_ask):
                    petya += 1
                    continue
                n_res += 1
                hc = hit_extract(rows_c, held_ask)
                hr = hit_extract(rows_r, held_ask)
                ctx += int(hc)
                rnd += int(hr)
                bag_c += int(hit_bag(rows_c, held_ask, mid_set))
                bag_r += int(hit_bag(rows_r, held_ask, mid_set))

    def r(x, d):
        return x / d if d else 0.0

    void = n_res < 40
    p_c, p_r, p_p = r(ctx, n_res), r(rnd, n_res), r(petya, n)
    d = p_c - p_r
    gate = (not void) and d > 0.05
    print(f"n {n}  res {n_res}  PETYA-hit {p_p:.3f}  CTX {p_c:.3f}  RAND {p_r:.3f}  d {d:+.3f}")
    print(f"bag CTX {r(bag_c, n_res):.3f}  bag RAND {r(bag_r, n_res):.3f}  (diag, 613-class)")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: rare two-extra residual. Hungry, not STOP of hop1.")
    elif gate:
        print("GO CTX: oracle apples open tree above random mid. Writeback can pay.")
    else:
        print("STOP: true pin does not beat random on the second extra. No context to train.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n, n_res=n_res, k=K,
        petya=p_p, ctx=p_c, rand=p_r, d=d,
        bag_ctx=r(bag_c, n_res), bag_rand=r(bag_r, n_res),
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
