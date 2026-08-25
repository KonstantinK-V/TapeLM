"""633: GAP-CONSERVATION. Torch-free. No Φ.

After 618 REFUSE, unbundle exact places instead of 630 peak-COMMIT:

    W = each leftover door address, resolved once (DIRECT)
        ∪ each exact hop1 address from that resolved door (HOPONLY)

Oracle = any unbundled literal == held_ask.
Peak  = commit_resolved over the same hop1 bag (630 compression).
623   = leftover_doors + walk stop-on-hop1, same trial (paired ceiling).

GATE, declared before the run:
    VOID  n_live < 40
    U     union hit ≥ 0.9 × paired 623.search  and  ≥ 0.17
          (0.17 is the predeclared absolute TinyStories floor)
    D     DIRECT ≥ 0.05  (leftover extra still in the offer)
    H     exclusive HOPONLY ≥ 0.05  (DIRECT-only 632 cannot pass)
    P     union − peak ≥ 0.05  (unbundle beats compression)

Priced-from-623 is not a bar: that file is already ≈ 0 after ~4 opens.

    python _check633_gapcon.py
    python _audit633_gapcon.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit633_gapcon.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit633_gapcon.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places, place_offer
from _audit615_depth import K
from _audit618_peakpin import peak_pin
from _audit621_refceil import leftover_doors
from _audit624_pick import hide_two
from _integrated_contract_v1 import (
    CAP,
    commit_resolved,
    leftover_records,
)

OUT = Path("results/_stage633_gapcon.json")
BAR_U = 0.17
BAR_D = 0.05
BAR_H = 0.05
BAR_PEAK = 0.05
KEEP_623 = 0.90


def extract_633(place, pin, env_m, mid_set, co, df, n_use):
    """Frozen deterministic read; equal PMI follows physical tape-row order."""
    bag, uniq = place_offer(place, pin, env_m, mid_set)
    uniq = list(dict.fromkeys(uniq))
    if not uniq:
        return None, bag, uniq

    def score(tok):
        de = max(df.get(tok, 1), 1)
        acc = 0.0
        for word in sorted(env_m):
            c = co.get((tok, word), 0)
            if c <= 0:
                continue
            dw = max(df.get(word, 1), 1)
            acc += math.log(max((c * n_use) / (de * dw), 1e-9))
        return acc / len(env_m) if env_m else 0.0

    # max is stable: a score tie keeps the first literal written at this place.
    tok = max(uniq, key=score)
    return tok, bag, uniq


def extracts_633(pg, word, skip, env_m, mid_set, co, df, n_use):
    rows = []
    for pi in pg["by_place"].get(word, ()):
        if pi == skip:
            continue
        tok, bag, uniq = extract_633(
            pg["places"][pi], word, env_m, mid_set, co, df, n_use,
        )
        rows.append((tok, bag, bool(uniq)))
    return rows


def hop1_623(
    pg, door, held_ask, skip, env_m, mid_set, high_set, co, df, n_use,
):
    """The depth-1 part of 623 under the same frozen deterministic read."""
    for tok, _bag, _uniq in extracts_633(
        pg, door, skip, env_m, mid_set, co, df, n_use,
    )[:K]:
        if tok is None or tok not in mid_set or tok in high_set:
            continue
        if tok == held_ask:
            return 1
    return 0


def open_hop1_623(
    pg, door, door_support_pi, pin, skip,
    env_m, mid_set, high_set, co, df, n_use,
):
    """The exact first K cards inspected by 623; skip does not buy replacements."""
    observations = []
    scanned = 0
    for pi in pg["by_place"].get(door, ()):
        if pi == skip:
            continue
        if scanned >= K:
            break
        scanned += 1
        place = pg["places"][pi]
        tok, _bag, uniq = extract_633(
            place, door, env_m, mid_set, co, df, n_use,
        )
        if not uniq or tok is None:
            continue
        if tok in {pin, door} or tok not in mid_set or tok in high_set:
            continue
        observations.append(dict(
            tok=tok,
            hop_pi=pi,
            door_support_pi=door_support_pi,
            majority=place["majority"],
            count_key=place["count_key"],
        ))
    return dict(
        door_support_pi=door_support_pi,
        observations=observations,
    )


def unbundle(pg, records, pin, skip, env_m, mid_set, high_set, co, df, n_use):
    """Exact places: door literal + each hop1 observation, not a peak."""
    cands = []
    opened = []
    seen_door_places = set()
    for rec in records:
        door_pi = rec["door_support_pi"]
        if door_pi in seen_door_places:
            continue
        seen_door_places.add(door_pi)
        place = pg["places"][door_pi]
        door, _bag, uniq = extract_633(
            place, pin, env_m, mid_set, co, df, n_use,
        )
        if not uniq or door is None:
            continue
        if door == pin or door not in mid_set or door in high_set:
            continue
        cands.append(("direct", door, door_pi))
        opened.append(open_hop1_623(
            pg, door, door_pi, pin, skip,
            env_m, mid_set, high_set, co, df, n_use,
        ))
        for obs in opened[-1]["observations"]:
            cands.append(("hop1", obs["tok"], obs["hop_pi"]))
    return cands, commit_resolved(opened)


def search_623(pg, pin, skip, held_ask, env_m, mid_set, high_set, co, df, n_use):
    """Paired 623 ceiling: held-blind offer, teacher-only stop."""
    doors = leftover_doors(
        pg, pin, skip, env_m, mid_set, high_set, {pin},
    )[:CAP]
    if not doors:
        return 0, 0
    hit = 0
    opened = 0
    for door in doors:
        opened += 1
        if door == held_ask:
            hit = 1
            break
        if hop1_623(
            pg, door, held_ask, skip, env_m, mid_set, high_set,
            co, df, n_use,
        ):
            hit = 1
            break
    return hit, opened


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
    print(f"633 gapcon  {path}  {kind}", flush=True)

    n = n_res = n_pin = n_ref = n_live = 0
    n_d = n_h = n_u = n_peak = n_rand = n_623 = 0
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
                    # Paired 623 is live here.  An empty exact-place offer is a
                    # UNION miss, never a reason to remove this trial.
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
    need = max(BAR_U, KEEP_623 * s623)
    kept = u_ora / s623 if s623 else 0.0
    short_hits = max(math.ceil(need * n_live - 1e-12) - n_u, 0)
    void = n_live < 40
    gate = (
        (not void)
        and u_ora >= need
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
        d_ora=d_ora, h_ora=h_ora, u_ora=u_ora,
        peak_ora=peak_ora, rand_ora=rand_ora, s623=s623,
        need=need, kept=kept, short_hits=short_hits,
        mean_cands=n_cands / n_live if n_live else 0.0,
        mean_open623=n_open623 / n_live if n_live else 0.0,
        bar_u=BAR_U, bar_d=BAR_D, bar_h=BAR_H,
        bar_peak=BAR_PEAK, keep_623=KEEP_623,
    )
    print(
        f"live {n_live}  DIRECT {d_ora:.3f}  HOPONLY {h_ora:.3f}  "
        f"UNION {u_ora:.3f}  peak {peak_ora:.3f}  rand {rand_ora:.3f}"
    )
    print(
        f"623 {s623:.3f}  need {need:.3f}  "
        f"kept {kept:.3f}  short {short_hits} hit  "
        f"UNION−peak {u_ora - peak_ora:+.3f}  cands {rec['mean_cands']:.1f}"
    )
    if void:
        print("VOID: refuse leftover hungry.")
    elif gate:
        print("GO CONSERVE: unbundled places keep the 623 gap; peak is worse.")
    else:
        print("STOP CONSERVE: exact places miss the predeclared gate. No Φ.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
