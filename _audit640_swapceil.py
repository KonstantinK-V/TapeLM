"""640: torch-free TRUE/SWAP ceiling on the 638 exact-place offer.

639 STOP 3/3 closed the residual scorer: without kind-0 PMI atoms, rows/VIA
do not beat frozen PMI.  628 already SWAP'd VIA on a different W (cards of
CURRENT + address kernel) and STOP'd.  640 does not rerun either.

Offer = 638 leftover DIRECT ∪ HOP1 places.  SWAP permutes via_pi among HOP1
only.  Extracts, QUERY, and place ids stay put, so PMI(extract, QUERY) is
identical on the pair — if it moves, VOID (construction bug).

Mind is not in this file.  Tape resolves a tiny constraint alphabet:

    SHARE-VIA  unique place whose address keys meet the VIA place keys
    ROW-Q      unique place whose fillers (minus extract) meet QUERY
    refuse     on a tie or a zero score

GATE, declared before the run:
  VOID  swappable pairs < 80
        or PMI pick differs on TRUE vs SWAP in > 2% of pairs
        or oracle − PMI <= .05
  GATE  SHARE-VIA (TRUE − SWAP) > .05
ROW-Q is diagnostic.  No Φ, no token ids, no CE.

    python _check640_swapceil.py
    python _audit640_swapceil.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit640_swapceil.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit640_swapceil.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
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
from _audit633_gapcon import extract_633, extracts_633, open_hop1_623
from _integrated_contract_v1 import CAP, leftover_records

OUT = Path("results/_stage640_swapceil.json")
BAR = 0.05
PMI_MOVE = 0.02


def contextual_offer(
    pg, records, pin, skip, env_m, mid_set, high_set, co, df, n_use,
):
    """Same 638 leftover places; copied so 640 stays torch-free."""
    candidates = []
    seen_door_places = set()
    for record in records:
        door_pi = record["door_support_pi"]
        if door_pi in seen_door_places:
            continue
        seen_door_places.add(door_pi)
        door_place = pg["places"][door_pi]
        door, _bag, uniq = extract_633(
            door_place, pin, env_m, mid_set, co, df, n_use,
        )
        if not uniq or door is None:
            continue
        if door == pin or door not in mid_set or door in high_set:
            continue
        candidates.append(dict(
            kind="direct",
            tok=door,
            pi=door_pi,
            via_pi=None,
            current=pin,
        ))
        opened_door = open_hop1_623(
            pg, door, door_pi, pin, skip,
            env_m, mid_set, high_set, co, df, n_use,
        )
        for observation in opened_door["observations"]:
            candidates.append(dict(
                kind="hop1",
                tok=observation["tok"],
                pi=observation["hop_pi"],
                via_pi=door_pi,
                current=door,
            ))
    return candidates


def lift_t(left, right, co, df, n_use):
    dl = max(df.get(left, 0), 0)
    dr = max(df.get(right, 0), 0)
    joint = max(co.get((left, right), 0), 0)
    if joint > 0 and dl > 0 and dr > 0:
        lift = (joint * n_use) / (dl * dr)
        return math.tanh(math.log(max(lift, 1e-9)) / 4.0)
    return 0.0


def pmi_scores(cands, qwords, co, df, n_use):
    out = []
    q = sorted(set(qwords))
    for cand in cands:
        if not q:
            out.append(0.0)
            continue
        out.append(
            sum(lift_t(cand["tok"], word, co, df, n_use) for word in q)
            / len(q)
        )
    return out


def unique_max(values):
    if not values:
        return None
    top = max(values)
    winners = [i for i, value in enumerate(values) if value == top]
    return winners[0] if len(winners) == 1 else None


def share_via_scores(pg, cands):
    scores = []
    for cand in cands:
        if cand["via_pi"] is None:
            scores.append(0.0)
            continue
        left = set(pg["places"][cand["pi"]]["keys"])
        right = set(pg["places"][cand["via_pi"]]["keys"])
        scores.append(float(len(left & right)))
    return scores


def row_q_scores(pg, cands, qwords):
    query = set(qwords)
    scores = []
    for cand in cands:
        vals = set(pg["places"][cand["pi"]]["vals"])
        vals.discard(cand["tok"])
        scores.append(float(len(vals & query)))
    return scores


def swap_via(cands, rng):
    """Permute VIA pointers on HOP1.  Extracts and place ids stay."""
    hops = [i for i, cand in enumerate(cands) if cand["via_pi"] is not None]
    vias = [cands[i]["via_pi"] for i in hops]
    if len(set(vias)) < 2:
        return None
    shuffled = list(vias)
    rng.shuffle(shuffled)
    if shuffled == vias:
        shuffled = vias[1:] + vias[:1]
        if shuffled == vias:
            return None
    out = [dict(cand) for cand in cands]
    for index, via in zip(hops, shuffled):
        out[index]["via_pi"] = via
    return out


def hit_at(cands, scores, held):
    pick = unique_max(scores)
    if pick is None:
        return 0, 0
    return int(cands[pick]["tok"] == held), 1


def collect_pairs(pool, args, rng):
    pairs = []
    n_live = n_pos = n_via = 0
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
                swapped = None
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
                    swapped = swap_via(cands, rng)
                finally:
                    hide_two(
                        co, df, query["keys"], held_ctx, held_ask, +1,
                    )
                if not live:
                    continue
                n_live += 1
                n_pos += int(any(c["tok"] == held_ask for c in cands))
                if swapped is None:
                    continue
                n_via += 1
                pairs.append(dict(
                    pg=pg,
                    true=cands,
                    swap=swapped,
                    qwords=list(query["keys"]),
                    held=held_ask,
                    co=co,
                    df=df,
                    n_use=n_use,
                ))
    return pairs, n_live, n_pos, n_via


def evaluate(pairs, rng):
    names = (
        "share_true", "share_swap", "row_true", "row_swap",
        "pmi_true", "pmi_swap", "rand", "oracle",
    )
    hits = Counter({name: 0 for name in names})
    commits = Counter()
    pmi_changed = 0
    for pair in pairs:
        c_true = pair["true"]
        c_swap = pair["swap"]
        held = pair["held"]
        qwords = pair["qwords"]
        pg, co, df, n_use = pair["pg"], pair["co"], pair["df"], pair["n_use"]
        hits["oracle"] += int(any(c["tok"] == held for c in c_true))
        hits["rand"] += int(rng.choice(c_true)["tok"] == held)

        pmi_t = pmi_scores(c_true, qwords, co, df, n_use)
        pmi_s = pmi_scores(c_swap, qwords, co, df, n_use)
        pt, ct = hit_at(c_true, pmi_t, held)
        ps, cs = hit_at(c_swap, pmi_s, held)
        hits["pmi_true"] += pt
        hits["pmi_swap"] += ps
        commits["pmi_true"] += ct
        commits["pmi_swap"] += cs
        pmi_changed += int(unique_max(pmi_t) != unique_max(pmi_s))

        st, cst = hit_at(c_true, share_via_scores(pg, c_true), held)
        ss, css = hit_at(c_swap, share_via_scores(pg, c_swap), held)
        hits["share_true"] += st
        hits["share_swap"] += ss
        commits["share_true"] += cst
        commits["share_swap"] += css

        rt, crt = hit_at(c_true, row_q_scores(pg, c_true, qwords), held)
        rs, crs = hit_at(c_swap, row_q_scores(pg, c_swap, qwords), held)
        hits["row_true"] += rt
        hits["row_swap"] += rs
        commits["row_true"] += crt
        commits["row_swap"] += crs

    n = max(len(pairs), 1)
    rates = {name: hits[name] / n for name in names}
    rates["n"] = len(pairs)
    rates["pmi_changed"] = pmi_changed / n
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
    print(f"640 swapceil  {path}  {kind}", flush=True)

    pairs, n_live, n_pos, n_via = collect_pairs(pool, args, rng)
    rates, commits = evaluate(pairs, random.Random(args.seed + 9))
    d_share = rates["share_true"] - rates["share_swap"]
    d_row = rates["row_true"] - rates["row_swap"]
    d_pmi = abs(rates["pmi_true"] - rates["pmi_swap"])
    room = rates["oracle"] - rates["pmi_true"]
    void = (
        rates["n"] < 80
        or rates["pmi_changed"] > PMI_MOVE
        or room <= BAR
    )
    gate = (not void) and d_share > BAR
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
        n_live=n_live, n_pos=n_pos, n_via=n_via,
        void=void, gate=gate,
        share_true=rates["share_true"], share_swap=rates["share_swap"],
        row_true=rates["row_true"], row_swap=rates["row_swap"],
        pmi_true=rates["pmi_true"], pmi_swap=rates["pmi_swap"],
        rand=rates["rand"], oracle=rates["oracle"],
        d_share=d_share, d_row=d_row, d_pmi=d_pmi,
        pmi_changed=rates["pmi_changed"], room=room,
        n_pairs=rates["n"], bar=BAR, n_commit=commits,
        torch=False, output="exact_place",
        constraint="share_via", swap="permute_via_pi",
    )
    print(
        f"live {n_live}  pos {n_pos}  via-pairs {rates['n']}  "
        f"share T/S {rates['share_true']:.3f}/{rates['share_swap']:.3f}  "
        f"row T/S {rates['row_true']:.3f}/{rates['row_swap']:.3f}  "
        f"pmi T/S {rates['pmi_true']:.3f}/{rates['pmi_swap']:.3f}"
    )
    print(
        f"oracle {rates['oracle']:.3f}  rand {rates['rand']:.3f}  "
        f"room {room:+.3f}  pmi_changed {rates['pmi_changed']:.3f}"
    )
    print(
        f"d_share {d_share:+.3f}  d_row {d_row:+.3f}  |d_pmi| {d_pmi:.3f}"
    )
    print(
        "n_commit "
        + " ".join(
            f"{name}={commits.get(name, 0)}"
            for name in (
                "share_true", "share_swap", "row_true", "pmi_true",
            )
        )
    )
    if void:
        print("VOID SWAPCEIL: thin pairs, PMI moved, or no room over PMI.")
    elif gate:
        print("GO SWAPCEIL: SHARE-VIA constraint is not PMI.")
    else:
        print("STOP SWAPCEIL: SHARE-VIA TRUE-SWAP does not clear .05.")

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
