"""641: torch-free ceiling on the PMI-silent slice of the 638 offer.

640 STOP 3/3: SHARE-VIA TRUE−SWAP ≈ 0 while oracle−PMI ≈ .15.  That gap is
not a new scorer.  It is mostly trials where PMI unique-max already refused
or already committed the wrong extract.

641 splits those two and gates only the first:

    silent  unique_max(PMI) is None     ← PMI did not decide
    wrong   unique_max(PMI) ≠ held      ← recovering this is 639; diagnostic
    hit     unique_max(PMI) == held     ← excluded

Offer = 640 leftover DIRECT ∪ HOP1 (no SWAP).  Held-blind rule is 618 peak
over leftover extracts (top count >= 2 and strictly > second), not SHARE-VIA.
SHARE is printed, not gated.  Random is the rival.  PMI is silent here, so
it is not a rival.

VOID  n_silent < 80  or  (oracle − random) on silent <= .05
GATE  peak618 − random > .05 on silent, 3/3

GO means Φ may act only while PMI refuses: write peak/refuse, do not rank
extract.  STOP means the .15 room is not a usable silent exam.

    python _check641_silent.py
    python _audit641_silent.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit641_silent.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit641_silent.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places
from _audit615_depth import K
from _audit618_peakpin import peak_pin
from _audit624_pick import hide_two
from _audit633_gapcon import extracts_633
from _audit640_swapceil import (
    BAR,
    contextual_offer,
    pmi_scores,
    unique_max,
)
from _integrated_contract_v1 import CAP, leftover_records

OUT = Path("results/_stage641_silent.json")


def peak618_tok(cands):
    """618 peak on leftover extracts.  Tie or n1 < 2 → refuse."""
    votes = [cand["tok"] for cand in cands]
    if not votes:
        return None
    cnt = Counter(votes)
    if len(cnt) == 1:
        return next(iter(cnt))
    top, n1 = cnt.most_common(1)[0]
    n2 = cnt.most_common(2)[1][1]
    if n1 < 2 or n1 <= n2:
        return None
    return top


def share_pick(pg, cands):
    from _audit640_swapceil import share_via_scores

    idx = unique_max(share_via_scores(pg, cands))
    if idx is None:
        return None
    return cands[idx]["tok"]


def collect_offers(pool, args, rng):
    offers = []
    windows = prefix_windows(pool, args.window_lines, args.n_win)
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
                cands = []
                try:
                    rows_p = extracts_633(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    if any(tok == held_ask for tok, _b, _u in rows_p[:K]):
                        continue
                    hat = peak_pin(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )
                    if hat is not None:
                        continue
                    records = leftover_records(
                        pg, pin, qi, env_m, mid_set, high_set, {pin},
                    )[:CAP]
                    if not records:
                        continue
                    cands = contextual_offer(
                        pg, records, pin, qi, env_m, mid_set, high_set,
                        co, df, n_use,
                    )
                    live = bool(cands)
                finally:
                    hide_two(
                        co, df, query["keys"], held_ctx, held_ask, +1,
                    )
                if not live:
                    continue
                offers.append(dict(
                    pg=pg,
                    cands=cands,
                    qwords=list(query["keys"]),
                    held=held_ask,
                    co=co,
                    df=df,
                    n_use=n_use,
                ))
    return offers


def tally(rows, rng):
    n = max(len(rows), 1)
    peak = share = rnd = ora = 0
    peak_c = share_c = 0
    for row in rows:
        cands, held, pg = row["cands"], row["held"], row["pg"]
        ora += int(any(c["tok"] == held for c in cands))
        rnd += int(rng.choice(cands)["tok"] == held)
        ptok = peak618_tok(cands)
        if ptok is not None:
            peak_c += 1
            peak += int(ptok == held)
        stok = share_pick(pg, cands)
        if stok is not None:
            share_c += 1
            share += int(stok == held)
    return dict(
        n=len(rows),
        oracle=ora / n,
        rand=rnd / n,
        peak=peak / n,
        share=share / n,
        peak_commit=peak_c / n,
        share_commit=share_c / n,
        room=ora / n - rnd / n,
        d_peak=peak / n - rnd / n,
        d_share=share / n - rnd / n,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=120)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=160000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open(
        "r", encoding="utf-8", errors="ignore",
    ).read(args.bytes)
    all_lines = [
        line.strip() for line in text.split("\n")
        if len(line.strip()) >= min_line
    ]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"641 silent  {path}  {kind}", flush=True)

    offers = collect_offers(pool, args, rng)
    silent, wrong, hit = [], [], []
    for row in offers:
        scores = pmi_scores(
            row["cands"], row["qwords"], row["co"], row["df"], row["n_use"],
        )
        pick = unique_max(scores)
        if pick is None:
            silent.append(row)
        elif row["cands"][pick]["tok"] == row["held"]:
            hit.append(row)
        else:
            wrong.append(row)

    rng_e = random.Random(args.seed + 9)
    s_rates = tally(silent, rng_e)
    w_rates = tally(wrong, random.Random(args.seed + 10))
    h_rates = tally(hit, random.Random(args.seed + 11))
    void = s_rates["n"] < 80 or s_rates["room"] <= BAR
    gate = (not void) and s_rates["d_peak"] > BAR
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
        n_offer=len(offers),
        n_silent=s_rates["n"], n_wrong=w_rates["n"], n_hit=h_rates["n"],
        void=void, gate=gate,
        silent=s_rates, wrong=w_rates, hit=h_rates,
        bar=BAR, torch=False, output="exact_place",
        slice="pmi_unique_max_is_none",
        rule="peak618_leftover_extracts",
        share_in_gate=False, wrong_in_gate=False,
    )
    print(
        f"offer {len(offers)}  silent {s_rates['n']}  "
        f"wrong {w_rates['n']}  hit {h_rates['n']}"
    )
    print(
        "SILENT "
        f"ora {s_rates['oracle']:.3f}  rand {s_rates['rand']:.3f}  "
        f"peak {s_rates['peak']:.3f}  share {s_rates['share']:.3f}  "
        f"d_peak {s_rates['d_peak']:+.3f}  room {s_rates['room']:+.3f}  "
        f"peak_c {s_rates['peak_commit']:.3f}"
    )
    print(
        "WRONG  "
        f"ora {w_rates['oracle']:.3f}  rand {w_rates['rand']:.3f}  "
        f"peak {w_rates['peak']:.3f}  d_peak {w_rates['d_peak']:+.3f}  "
        "(diagnostic, not gate)"
    )
    if void:
        print("VOID SILENT: thin slice or no room over random.")
    elif gate:
        print("GO SILENT: 618-peak beats random while PMI refuses.")
    else:
        print("STOP SILENT: PMI-refuse is not a usable exam for phi.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(
        out.read_text(encoding="utf-8"),
    ) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
