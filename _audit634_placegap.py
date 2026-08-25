"""634: conservation of the 625-place offer. Torch-free. No Φ.

633 STOP stands: UNION missed 0.9×623.search by one hit/seed (kept ~.898).
Cause: 623 is leftover door-WORDS + stop-on-held. 634 offer is leftover
ADDRESSES → one extract_633 → hop1 from that extract. Chasing the remaining
623-only mass is a word shortlist. KEEP_623 is diagnostic, not a bar.

GATE, declared before the run (same object as the 633 json):
    VOID  n_live < 40
    D     DIRECT ≥ 0.05
    H     exclusive HOPONLY ≥ 0.05
    P     UNION − peak ≥ 0.05
623, kept, only_623, only_union — printed, not gated.

Same read as 633 (extract_633, first K cards, tape-order PMI ties).
Repeat seed 1337 must reprint the same d/h/u.

    python _check634_placegap.py
    python _audit634_placegap.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit634_placegap.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit634_placegap.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places
from _audit615_depth import K
from _audit618_peakpin import peak_pin
from _audit624_pick import hide_two
from _audit633_gapcon import (
    BAR_D,
    BAR_H,
    BAR_PEAK,
    extracts_633,
    search_623,
    unbundle,
)
from _integrated_contract_v1 import CAP, leftover_records

OUT = Path("results/_stage634_placegap.json")


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
    all_lines = [
        line.strip() for line in text.split("\n")
        if len(line.strip()) >= min_line
    ]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"634 placegap  {path}  {kind}", flush=True)

    n = n_res = n_pin = n_ref = n_live = 0
    n_d = n_h = n_u = n_peak = n_rand = n_623 = 0
    n_only623 = n_onlyu = 0
    n_open623 = 0
    n_cands = 0
    for lines in windows:
        pg = build_places(lines, args.frame_max, args.min_fillers)
        if pg is None:
            continue
        mid_set, high_set = bands(pg)
        if not mid_set:
            continue
        co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
        places = pg["places"]
        pins = sorted(mid_set)
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
                try:
                    n += 1
                    rows_p = extracts_633(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    if any(tok == held_ask for tok, _bag, _uniq in rows_p[:K]):
                        continue
                    n_res += 1
                    hat = peak_pin(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )
                    if hat is not None:
                        n_pin += 1
                        continue
                    n_ref += 1
                    records = leftover_records(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )[:CAP]
                    if not records:
                        continue
                    live = True
                    cands, peak = unbundle(
                        pg, records, pin, qi, env_m, mid_set, high_set,
                        co, df, n_use,
                    )
                    toks = [tok for _kind, tok, _pi in cands]
                    d_hit = any(
                        kind == "direct" and tok == held_ask
                        for kind, tok, _pi in cands
                    )
                    h_hit = (not d_hit) and any(
                        kind == "hop1" and tok == held_ask
                        for kind, tok, _pi in cands
                    )
                    u_hit = int(held_ask in toks)
                    p_hit = int(peak is not None and peak["tok"] == held_ask)
                    r_hit = int(bool(toks) and rng.choice(toks) == held_ask)
                    s_hit, n_op = search_623(
                        pg, pin, qi, held_ask, env_m, mid_set, high_set,
                        co, df, n_use,
                    )
                finally:
                    hide_two(co, df, query["keys"], held_ctx, held_ask, +1)
                if not live:
                    continue
                n_live += 1
                n_d += int(d_hit)
                n_h += int(h_hit)
                n_u += u_hit
                n_peak += p_hit
                n_rand += r_hit
                n_623 += s_hit
                n_only623 += int(bool(s_hit) and not u_hit)
                n_onlyu += int(bool(u_hit) and not s_hit)
                n_open623 += n_op
                n_cands += len(cands)

    def rate(x):
        return x / n_live if n_live else 0.0

    d_ora = rate(n_d)
    h_ora = rate(n_h)
    u_ora = rate(n_u)
    peak_ora = rate(n_peak)
    rand_ora = rate(n_rand)
    s623 = rate(n_623)
    kept = u_ora / s623 if s623 else 0.0
    void = n_live < 40
    # KEEP_623 / BAR_U are not in this expression.
    gate = (
        (not void)
        and d_ora >= BAR_D
        and h_ora >= BAR_H
        and (u_ora - peak_ora) >= BAR_PEAK
    )
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
        void=void, gate=gate,
        n=n, n_res=n_res, n_pin=n_pin, n_ref=n_ref, n_live=n_live,
        n_direct=n_d, n_hoponly=n_h, n_union=n_u,
        n_peak=n_peak, n_rand=n_rand, n_623=n_623,
        n_only623=n_only623, n_onlyu=n_onlyu,
        d_ora=d_ora, h_ora=h_ora, u_ora=u_ora,
        peak_ora=peak_ora, rand_ora=rand_ora, s623=s623,
        kept=kept, only_623=rate(n_only623), only_union=rate(n_onlyu),
        mean_cands=n_cands / n_live if n_live else 0.0,
        mean_open623=n_open623 / n_live if n_live else 0.0,
        bar_d=BAR_D, bar_h=BAR_H, bar_peak=BAR_PEAK,
    )
    print(
        f"live {n_live}  DIRECT {d_ora:.3f}  HOPONLY {h_ora:.3f}  "
        f"UNION {u_ora:.3f}  peak {peak_ora:.3f}  rand {rand_ora:.3f}"
    )
    print(
        f"623 {s623:.3f}  kept {kept:.3f}  "
        f"only_623 {rec['only_623']:.3f}  only_union {rec['only_union']:.3f}  "
        f"UNION-peak {u_ora - peak_ora:+.3f}"
    )
    if void:
        print("VOID: refuse leftover hungry.")
    elif gate:
        print("GO PLACE: address offer keeps DIRECT and HOPONLY; peak is worse.")
    else:
        print("STOP PLACE: 625-address gap did not survive. No Φ.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
