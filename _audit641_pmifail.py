"""641: torch-free ceiling on the PMI-fail slice of the 638 offer.

640 froze PMI and killed SHARE-VIA (TRUE~SWAP).  Remaining question: is
oracle - PMI real residual?  Held still among exact places after PMI
unique-max is wrong or refuses?

Slice = live leftover trials where PMI does not uniquely pick held.
No Phi, no torch, no token ids.  SHARE-VIA / ROW-Q / count are diagnostics.

GATE, declared before the run:
  VOID  n_slice < 80
  GATE  oracle_slice - random > .05

Hand-rule deltas vs random (count, ROW-Q, SHARE-VIA) print only.  If GATE
and a hand rule clears .05, that rule is the next constraint to write.
If GATE but no hand rule, residual places exist but this alphabet is blind
- search a new constraint, do not resurrect SHARE-VIA GNN.

    python _check641_pmifail.py
    python _audit641_pmifail.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit641_pmifail.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit641_pmifail.py --seed 2890 --corpus data/_tinystories_train.txt
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
    row_q_scores,
    share_via_scores,
    unique_max,
)
from _integrated_contract_v1 import CAP, leftover_records

OUT = Path("results/_stage641_pmifail.json")


def count_scores(pg, cands):
    return [pg["places"][cand["pi"]]["count_key"] for cand in cands]


def hit_at(cands, scores, held):
    pick = unique_max(scores)
    if pick is None:
        return 0, 0
    return int(cands[pick]["tok"] == held), 1


def collect_slice(pool, args, rng):
    """All live 638 offers; keep only PMI-fail (wrong unique pick or refuse)."""
    slice_trials = []
    n_live = n_oracle = n_pmi_hit = n_refuse = n_wrong = 0
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
                    if not cands:
                        continue
                    live = True
                finally:
                    hide_two(
                        co, df, query["keys"], held_ctx, held_ask, +1,
                    )
                if not live:
                    continue
                n_live += 1
                held_in = any(c["tok"] == held_ask for c in cands)
                n_oracle += int(held_in)
                scores = pmi_scores(cands, query["keys"], co, df, n_use)
                pick = unique_max(scores)
                if pick is not None and cands[pick]["tok"] == held_ask:
                    n_pmi_hit += 1
                    continue
                if pick is None:
                    kind = "refuse"
                    n_refuse += 1
                else:
                    kind = "wrong"
                    n_wrong += 1
                slice_trials.append(dict(
                    pg=pg,
                    cands=cands,
                    qwords=list(query["keys"]),
                    held=held_ask,
                    co=co,
                    df=df,
                    n_use=n_use,
                    fail_kind=kind,
                    held_in=held_in,
                ))
    return slice_trials, dict(
        n_live=n_live,
        n_oracle_all=n_oracle,
        n_pmi_hit=n_pmi_hit,
        n_refuse=n_refuse,
        n_wrong=n_wrong,
        n_slice=len(slice_trials),
    )


def evaluate(slice_trials, rng):
    names = (
        "oracle", "rand", "count", "row_q", "share",
        "oracle_refuse", "oracle_wrong",
        "rand_refuse", "rand_wrong",
    )
    hits = Counter({name: 0 for name in names})
    commits = Counter()
    n_refuse = n_wrong = 0
    for trial in slice_trials:
        cands = trial["cands"]
        held = trial["held"]
        pg = trial["pg"]
        kind = trial["fail_kind"]
        if kind == "refuse":
            n_refuse += 1
        else:
            n_wrong += 1
        hits["oracle"] += int(trial["held_in"])
        hits[f"oracle_{kind}"] += int(trial["held_in"])
        rand_hit = int(rng.choice(cands)["tok"] == held)
        hits["rand"] += rand_hit
        hits[f"rand_{kind}"] += rand_hit

        for name, scores in (
            ("count", count_scores(pg, cands)),
            ("row_q", row_q_scores(pg, cands, trial["qwords"])),
            ("share", share_via_scores(pg, cands)),
        ):
            hit, commit = hit_at(cands, scores, held)
            hits[name] += hit
            commits[name] += commit

    n = max(len(slice_trials), 1)
    rates = {name: hits[name] / n for name in names}
    rates["n"] = len(slice_trials)
    rates["n_refuse"] = n_refuse
    rates["n_wrong"] = n_wrong
    if n_refuse:
        # recompute refuse/wrong denominators honestly
        rates["oracle_refuse"] = hits["oracle_refuse"] / n_refuse
        rates["rand_refuse"] = hits["rand_refuse"] / n_refuse
    else:
        rates["oracle_refuse"] = 0.0
        rates["rand_refuse"] = 0.0
    if n_wrong:
        rates["oracle_wrong"] = hits["oracle_wrong"] / n_wrong
        rates["rand_wrong"] = hits["rand_wrong"] / n_wrong
    else:
        rates["oracle_wrong"] = 0.0
        rates["rand_wrong"] = 0.0
    return rates, dict(commits)


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
    print(f"641 pmifail  {path}  {kind}", flush=True)

    slice_trials, diag = collect_slice(pool, args, rng)
    rates, commits = evaluate(slice_trials, random.Random(args.seed + 9))
    room = rates["oracle"] - rates["rand"]
    void = rates["n"] < 80
    gate = (not void) and room > BAR
    hand = {
        "count": rates["count"] - rates["rand"],
        "row_q": rates["row_q"] - rates["rand"],
        "share": rates["share"] - rates["rand"],
    }
    best_hand = max(hand, key=hand.get)
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
        void=void, gate=gate,
        n_live=diag["n_live"],
        n_oracle_all=diag["n_oracle_all"],
        n_pmi_hit=diag["n_pmi_hit"],
        n_refuse=diag["n_refuse"],
        n_wrong=diag["n_wrong"],
        n_slice=rates["n"],
        oracle=rates["oracle"], rand=rates["rand"],
        count=rates["count"], row_q=rates["row_q"], share=rates["share"],
        room=room, bar=BAR,
        d_count=hand["count"], d_row=hand["row_q"], d_share=hand["share"],
        best_hand=best_hand, best_hand_delta=hand[best_hand],
        oracle_refuse=rates["oracle_refuse"],
        rand_refuse=rates["rand_refuse"],
        oracle_wrong=rates["oracle_wrong"],
        rand_wrong=rates["rand_wrong"],
        n_commit=commits,
        torch=False, output="exact_place",
        slice="pmi_unique_max_misses_held",
    )
    print(
        f"live {diag['n_live']}  pmi_hit {diag['n_pmi_hit']}  "
        f"slice {rates['n']} (refuse {rates['n_refuse']} "
        f"wrong {rates['n_wrong']})"
    )
    print(
        f"oracle {rates['oracle']:.3f}  rand {rates['rand']:.3f}  "
        f"count {rates['count']:.3f}  row_q {rates['row_q']:.3f}  "
        f"share {rates['share']:.3f}  room {room:+.3f}"
    )
    print(
        f"refuse ora/rand {rates['oracle_refuse']:.3f}/"
        f"{rates['rand_refuse']:.3f}  "
        f"wrong ora/rand {rates['oracle_wrong']:.3f}/"
        f"{rates['rand_wrong']:.3f}"
    )
    print(
        f"hand d_count {hand['count']:+.3f}  d_row {hand['row_q']:+.3f}  "
        f"d_share {hand['share']:+.3f}  best {best_hand} "
        f"{hand[best_hand]:+.3f}"
    )
    print(
        "n_commit "
        + " ".join(
            f"{name}={commits.get(name, 0)}"
            for name in ("count", "row_q", "share")
        )
    )
    if void:
        print("VOID PMIFAIL: thin PMI-fail slice.")
    elif gate:
        print("GO PMIFAIL: held still reachable after PMI fails.")
    else:
        print("STOP PMIFAIL: no residual place room over random.")

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
