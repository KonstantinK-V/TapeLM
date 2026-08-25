"""628: same-CURRENT context-swap ceiling.

627 measured whether a second peak exists.  That is path length, not context.
Here query and CURRENT stay fixed; only the exact VIA-address register W is
replaced by a matched history which reached the same CURRENT.

W contains exact place pointers, role tagged as VIA.  Nothing is merged at
write time.  At read time the tape resolves:

    cards(CURRENT) constrained by QUERY and VIA-address geometry

Geometry is a held-blind kernel over exact address keys and visible tape
co-counts.  High-df glue contributes at most one address atom (519 allow=1).
The resolver returns one exact address; a score tie or no VIA evidence means
REFUSE.  The selected card then resolves its unique tape extra.

Arms, on the same query/CURRENT/candidate cards:
  TRUE W    support addresses which actually voted CURRENT
  SWAP W    matched different support addresses which also voted CURRENT
  QUERY     no history, query-address geometry only
  PEAK      current-only 618 vote
  COUNT     max-count exact address
  FREQ      majority filler at TRUE W's selected exact address
  RAND      random exact address
  ORA       any candidate address directly resolves held_ask (ceiling)

Held is used only for reward and diagnostics.  It never constructs W, scores
an address, chooses a swap, or resolves a tie.

GATE:
  TRUE − max(SWAP, QUERY, PEAK, COUNT, FREQ, RAND) > 0.05
  and TRUE/SWAP select different addresses on >= 10% of paired trials.
VOID:
  paired same-CURRENT trials < 40, or ORA has <= .05 room over rivals.

    python _check628_ctxswap.py
    python _audit628_ctxswap.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit628_ctxswap.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit628_ctxswap.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places, extract_at
from _audit624_pick import hide_two

OUT = Path("results/_stage628_ctxswap.json")


def legal_cards(
    pg, pin, skip, env_m, mid_set, high_set, co, df, n_use, forbid,
):
    """Exact cards of pin with one legal unique tape extra."""
    rows = []
    for pi in pg["by_place"].get(pin, ()):
        if pi == skip:
            continue
        place = pg["places"][pi]
        tok, _bag, uniq = extract_at(
            place, pin, env_m, mid_set, co, df, n_use,
        )
        if not uniq or tok is None:
            continue
        if tok in forbid or tok not in mid_set or tok in high_set:
            continue
        rows.append(dict(
            pi=pi,
            tok=tok,
            count_key=place["count_key"],
            majority=place["majority"],
        ))
    return rows


def peak_with_support(rows):
    """618 peak and the exact addresses which voted for it."""
    if not rows:
        return None, ()
    cnt = Counter(row["tok"] for row in rows)
    if len(cnt) == 1:
        top = next(iter(cnt))
    else:
        (top, n1), (_second, n2) = cnt.most_common(2)
        if n1 < 2 or n1 <= n2:
            return None, ()
    support = tuple(row["pi"] for row in rows if row["tok"] == top)
    return top, support


def address_tokens(place, current, high_set):
    """Exact address atoms; high-df glue is capped at one atom."""
    _width, left, right = place["addr"]
    toks = [tok for tok in left + right if tok and tok != current]
    low = list(dict.fromkeys(tok for tok in toks if tok not in high_set))
    high = sorted(set(tok for tok in toks if tok in high_set))
    return tuple(low + high[:1])  # 519: high allow=1


def address_kernel(
    pg, left_pi, right_pi, current, high_set, co, df, n_use,
):
    """Read-time geometry between two exact address pointers."""
    xs = address_tokens(pg["places"][left_pi], current, high_set)
    ys = address_tokens(pg["places"][right_pi], current, high_set)
    if not xs or not ys:
        return 0.0
    score = 0.0
    for x in xs:
        for y in ys:
            if x == y:
                score += 1.0
                continue
            c = co.get((x, y), 0)
            if c <= 0:
                continue
            lift = (c * n_use) / (
                max(df.get(x, 1), 1) * max(df.get(y, 1), 1)
            )
            if lift > 1.0:
                score += min(math.log(lift), 4.0)
    return score / math.sqrt(len(xs) * len(ys))


def unique_best(scored):
    """One exact address or REFUSE.  Never break a score tie by order."""
    if not scored:
        return None
    best = max(score for score, _row in scored)
    if best <= 0.0:
        return None
    winners = [row for score, row in scored if round(score, 12) == round(best, 12)]
    return winners[0] if len(winners) == 1 else None


def resolve_context(
    pg, rows, query_pi, current, via_pis, high_set, co, df, n_use,
):
    """Resolve an exact CURRENT card using QUERY + role-tagged VIA pointers."""
    if not via_pis:
        return None
    scored = []
    for row in rows:
        via = sum(
            address_kernel(
                pg, row["pi"], hpi, current, high_set, co, df, n_use,
            )
            for hpi in via_pis
        ) / math.sqrt(len(via_pis))
        query = address_kernel(
            pg, row["pi"], query_pi, current, high_set, co, df, n_use,
        )
        scored.append((via * (1.0 + query), row))
    return unique_best(scored)


def resolve_query(pg, rows, query_pi, current, high_set, co, df, n_use):
    """No-history control: same cards and query, but no VIA register."""
    return unique_best([
        (
            address_kernel(
                pg, row["pi"], query_pi, current, high_set, co, df, n_use,
            ),
            row,
        )
        for row in rows
    ])


def collect_window(lines, args, seed):
    """Collect first-peak episodes, then pair different W at same CURRENT."""
    rng = random.Random(seed)
    pg = build_places(lines, args.frame_max, args.min_fillers)
    if pg is None:
        return None, [], {}
    mid_set, high_set = bands(pg)
    if not mid_set:
        return None, [], {}
    co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
    places = pg["places"]
    episodes = []
    diag = Counter()

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
            try:
                first_rows = legal_cards(
                    pg, pin, qi, env_m, mid_set, high_set,
                    co, df, n_use, {pin},
                )
                current, support = peak_with_support(first_rows)
                diag["source"] += 1
                if current is None:
                    diag["refuse1"] += 1
                    continue
                diag["peak1"] += 1
                diag["current_is_ctx"] += int(current == held_ctx)
                if current == held_ask:
                    diag["direct"] += 1
                    continue
                rows = legal_cards(
                    pg, current, qi, env_m, mid_set, high_set,
                    co, df, n_use, {pin, current},
                )
                if len(rows) < 2:
                    diag["few_cards"] += 1
                    continue
                episodes.append(dict(
                    pin=pin,
                    qi=qi,
                    current=current,
                    support=support,
                    rows=rows,
                    held_ctx=held_ctx,
                    held_ask=held_ask,
                    hide_keys=tuple(query["keys"]),
                    n_use=n_use,
                    match_sig=(
                        len(support),
                        len(pg["by_place"].get(pin, ())),
                        sum(len(address_tokens(places[pi], current, high_set))
                            for pi in support),
                    ),
                ))
            finally:
                hide_two(co, df, query["keys"], held_ctx, held_ask, +1)

    groups = defaultdict(list)
    for ep in episodes:
        groups[ep["current"]].append(ep)

    paired = []
    for ep in episodes:
        donors = [
            other for other in groups[ep["current"]]
            if other is not ep
            and other["support"] != ep["support"]
            and ep["qi"] not in other["support"]
        ]
        if not donors:
            continue
        different_pin = [other for other in donors if other["pin"] != ep["pin"]]
        if different_pin:
            donors = different_pin

        def distance(other):
            return sum(
                abs(a - b) for a, b in zip(ep["match_sig"], other["match_sig"])
            )

        best_d = min(distance(other) for other in donors)
        best = [other for other in donors if distance(other) == best_d]
        donor = rng.choice(best)
        paired.append((ep, donor["support"]))
    return pg, paired, dict(diag)


def evaluate_window(pg, paired, seed):
    rng = random.Random(seed)
    co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
    _mid_set, high_set = bands(pg)
    c = Counter()
    for ep, swap_support in paired:
        hide_two(
            co, df, ep["hide_keys"], ep["held_ctx"], ep["held_ask"], -1,
        )
        n_use = max(n_fr - 2, 1)
        try:
            true = resolve_context(
                pg, ep["rows"], ep["qi"], ep["current"], ep["support"],
                high_set, co, df, n_use,
            )
            swap = resolve_context(
                pg, ep["rows"], ep["qi"], ep["current"], swap_support,
                high_set, co, df, n_use,
            )
            query = resolve_query(
                pg, ep["rows"], ep["qi"], ep["current"],
                high_set, co, df, n_use,
            )
        finally:
            hide_two(
                co, df, ep["hide_keys"], ep["held_ctx"], ep["held_ask"], +1,
            )

        rows = ep["rows"]
        held = ep["held_ask"]
        peak_tok, _support = peak_with_support(rows)
        count_row = max(rows, key=lambda row: row["count_key"])
        rand_row = rng.choice(rows)

        c["n"] += 1
        c["true"] += int(true is not None and true["tok"] == held)
        c["swap"] += int(swap is not None and swap["tok"] == held)
        c["query"] += int(query is not None and query["tok"] == held)
        c["peak"] += int(peak_tok == held)
        c["count"] += int(count_row["tok"] == held)
        c["rand"] += int(rand_row["tok"] == held)
        c["freq"] += int(true is not None and true["majority"] == held)
        c["ora"] += int(any(row["tok"] == held for row in rows))
        c["true_read"] += int(true is not None)
        c["swap_read"] += int(swap is not None)
        c["query_read"] += int(query is not None)
        true_pi = None if true is None else true["pi"]
        swap_pi = None if swap is None else swap["pi"]
        c["changed"] += int(true_pi != swap_pi)
    return c


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
    t0 = time.time()
    print(f"628 ctxswap  {path}  {kind}", flush=True)

    total = Counter()
    diag = Counter()
    for wi, lines in enumerate(windows):
        pg, paired, one_diag = collect_window(
            lines, args, args.seed + 1009 * wi,
        )
        diag.update(one_diag)
        if pg is None or not paired:
            continue
        total.update(evaluate_window(pg, paired, args.seed + 2003 * wi))

    def rate(name):
        return total[name] / total["n"] if total["n"] else 0.0

    rates = {
        name: rate(name)
        for name in (
            "true", "swap", "query", "peak", "count", "freq", "rand", "ora",
            "true_read", "swap_read", "query_read", "changed",
        )
    }
    strongest = max(
        rates["swap"], rates["query"], rates["peak"], rates["count"],
        rates["freq"], rates["rand"],
    )
    room = rates["ora"] - strongest
    delta = rates["true"] - strongest
    void = total["n"] < 40 or room <= 0.05
    gate = (not void) and rates["changed"] >= 0.10 and delta > 0.05

    source = max(diag["source"], 1)
    print(
        f"source {diag['source']}  peak1 {diag['peak1'] / source:.3f}  "
        f"direct {diag['direct'] / source:.3f}  "
        f"current=ctx {diag['current_is_ctx'] / source:.3f}"
    )
    print(
        f"paired {total['n']}  changed {rates['changed']:.3f}  "
        f"read true/swap/query {rates['true_read']:.3f}/"
        f"{rates['swap_read']:.3f}/{rates['query_read']:.3f}"
    )
    print(
        f"REWARD  true {rates['true']:.3f}  swap {rates['swap']:.3f}  "
        f"query {rates['query']:.3f}  peak {rates['peak']:.3f}"
    )
    print(
        f"RIVALS  count {rates['count']:.3f}  freq {rates['freq']:.3f}  "
        f"rand {rates['rand']:.3f}  ora {rates['ora']:.3f}"
    )
    print(
        f"delta {delta:+.3f}  oracle_room {room:+.3f}  "
        f"VOID {void}  GATE {gate}"
    )
    if void:
        why = "same-CURRENT pairs hungry" if total["n"] < 40 else "no oracle room"
        print(f"VOID CTXSWAP: {why}.")
    elif gate:
        print(
            "GO CTXSWAP: exact VIA history changes the address and improves "
            "record reward. Train a constraint policy next."
        )
    else:
        print(
            "STOP CTXSWAP: this address geometry carries no usable context. "
            "Do not add a recurrent net on it."
        )

    rec = dict(
        seed=args.seed,
        corpus=kind,
        path=str(path),
        n_win=len(windows),
        elapsed_s=round(time.time() - t0, 1),
        n_source=diag["source"],
        n_peak1=diag["peak1"],
        n_direct=diag["direct"],
        n_current_ctx=diag["current_is_ctx"],
        n_pair=total["n"],
        strongest=strongest,
        delta=delta,
        oracle_room=room,
        void=bool(void),
        gate=bool(gate),
        **rates,
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
