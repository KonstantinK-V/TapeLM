"""637: multi-positive listwise rank on the exact-place W from 634.

636 used only the first correct candidate as its class.  When several exact
places resolve to held, CE rewarded one and penalized the other correct places.
637 keeps the same W, held-blind features, 70/30 line split, and 32-wide MLP.

Train only trials with at least one positive in the scored PAD prefix.  The
teacher marks EVERY candidate whose frozen extract equals held.  Loss is

    logsumexp(all place scores) - logsumexp(all correct-place scores)

There is no REFUSE class and no vocabulary output.  Feature normalization is
fit on train candidates only.  Trained and init models start byte-identically.
Eval covers every live trial; empty W or no positive in PAD is a miss.

Rivals: random place, identical untrained init, fixed PMI place, fixed
count-key place, and the majority filler at phi's chosen exact place.

Predeclared gate:
    VOID  eval live < 40 or train positive trials < 40
    phi - strongest(random, init, PMI, count, same-place majority) > .05
    phi - init > .05
Peak compression is printed only.

    python _check637_multipos.py
    python _audit637_multipos.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit637_multipos.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit637_multipos.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places
from _audit615_depth import K
from _audit618_peakpin import peak_pin
from _audit624_pick import hide_two
from _audit633_gapcon import extracts_633, unbundle
from _audit635_placescore import FDIM, PAD, feat_place, with_pmi_rank
from _integrated_contract_v1 import CAP, leftover_records

OUT = Path("results/_stage637_multipos.json")
BAR = 0.05
EPOCHS = 30
LR = 0.01


class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FDIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def logits(self, x, mask):
        score = self.net(x).squeeze(-1)
        return score.masked_fill(~mask, -1e9)


def make_net(seed):
    """Init before construction so trained and init controls are identical."""
    torch.manual_seed(seed)
    return RankNet()


def collect(pool, args, rng):
    trials = []
    n_live = n_pos_all = n_pos_pad = 0
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
                try:
                    rows_p = extracts_633(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    if any(tok == held_ask for tok, _bag, _uniq in rows_p[:K]):
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
                    live = True
                    cands, peak = unbundle(
                        pg, records, pin, qi, env_m, mid_set, high_set,
                        co, df, n_use,
                    )
                    feats = with_pmi_rank([
                        feat_place(
                            kind, tok, pi, pg, env_m, co, df, n_use,
                        )
                        for kind, tok, pi in cands
                    ])
                    correct = [
                        bool(tok == held_ask) for _kind, tok, _pi in cands
                    ]
                    majority_correct = [
                        bool(places[pi]["majority"] == held_ask)
                        for _kind, _tok, pi in cands
                    ]
                    count_keys = [
                        tuple(places[pi]["count_key"])
                        for _kind, _tok, pi in cands
                    ]
                    p_hit = int(
                        peak is not None and peak["tok"] == held_ask
                    )
                finally:
                    hide_two(
                        co, df, query["keys"], held_ctx, held_ask, +1,
                    )
                if not live:
                    continue
                n_live += 1
                n_pos_all += int(any(correct))
                n_pos_pad += int(any(correct[:PAD]))
                trials.append(dict(
                    feats=feats,
                    correct=correct,
                    majority_correct=majority_correct,
                    count_keys=count_keys,
                    p_hit=p_hit,
                ))
    return trials, n_live, n_pos_all, n_pos_pad


def fit_normalizer(train):
    rows = [
        feat
        for trial in train
        for feat in trial["feats"][:PAD]
    ]
    if not rows:
        return torch.zeros(FDIM), torch.ones(FDIM)
    x = torch.tensor(rows, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-4)
    return mean, std


def pack(batch, mean, std):
    bsz = len(batch)
    x = torch.zeros(bsz, PAD, FDIM)
    mask = torch.zeros(bsz, PAD, dtype=torch.bool)
    positive = torch.zeros(bsz, PAD, dtype=torch.bool)
    for bi, trial in enumerate(batch):
        n = min(len(trial["feats"]), PAD)
        if not n:
            continue
        raw = torch.tensor(trial["feats"][:n], dtype=torch.float32)
        x[bi, :n] = (raw - mean) / std
        mask[bi, :n] = True
        positive[bi, :n] = torch.tensor(
            trial["correct"][:n], dtype=torch.bool,
        )
    return x, mask, positive


def multi_positive_loss(logits, positive):
    """Probability mass on any correct exact place, never one chosen label."""
    pos_logits = logits.masked_fill(~positive, -1e9)
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(pos_logits, dim=1)
    ).mean()


def train_net(train_pos, seed, mean, std):
    net = make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    rng = random.Random(seed + 3)
    for _epoch in range(EPOCHS):
        order = list(range(len(train_pos)))
        rng.shuffle(order)
        net.train()
        for start in range(0, len(order), 32):
            batch = [train_pos[i] for i in order[start:start + 32]]
            x, mask, positive = pack(batch, mean, std)
            opt.zero_grad()
            multi_positive_loss(net.logits(x, mask), positive).backward()
            opt.step()
    return net


def unique_max(values):
    if not values:
        return None
    top = max(values)
    hits = [i for i, value in enumerate(values) if value == top]
    return hits[0] if len(hits) == 1 else None


def evaluate(net, init, trials, mean, std, rng):
    counts = dict(
        phi=0, init=0, rand=0, pmi=0, count=0,
        majority=0, peak=0, oracle=0,
    )
    net.eval()
    init.eval()
    with torch.no_grad():
        for trial in trials:
            counts["peak"] += trial["p_hit"]
            n = min(len(trial["feats"]), PAD)
            correct = trial["correct"][:n]
            counts["oracle"] += int(any(correct))
            if not n:
                continue
            counts["rand"] += int(correct[rng.randrange(n)])
            pmi_pick = unique_max(
                [row[5] for row in trial["feats"][:n]],
            )
            if pmi_pick is not None:
                counts["pmi"] += int(correct[pmi_pick])
            count_pick = unique_max(trial["count_keys"][:n])
            if count_pick is not None:
                counts["count"] += int(correct[count_pick])

            x, mask, _positive = pack([trial], mean, std)
            phi_scores = net.logits(x, mask)[0, :n].tolist()
            init_scores = init.logits(x, mask)[0, :n].tolist()
            phi_pick = unique_max(phi_scores)
            init_pick = unique_max(init_scores)
            if phi_pick is not None:
                counts["phi"] += int(correct[phi_pick])
                counts["majority"] += int(
                    trial["majority_correct"][phi_pick]
                )
            if init_pick is not None:
                counts["init"] += int(correct[init_pick])
    den = max(len(trials), 1)
    rates = {key: value / den for key, value in counts.items()}
    rates["n"] = len(trials)
    return rates


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
    text = path.open(
        "r", encoding="utf-8", errors="ignore",
    ).read(args.bytes)
    all_lines = [
        line.strip() for line in text.split("\n")
        if len(line.strip()) >= min_line
    ]
    cut = int(0.7 * len(all_lines))
    t0 = time.time()
    print(f"637 multipos  {path}  {kind}", flush=True)

    train, n_tr, pos_tr_all, pos_tr_pad = collect(
        all_lines[:cut][: args.lines],
        args,
        random.Random(args.seed),
    )
    ev, n_ev, pos_ev_all, pos_ev_pad = collect(
        all_lines[cut:][: args.lines],
        args,
        random.Random(args.seed + 1),
    )
    train_pos = [
        trial for trial in train if any(trial["correct"][:PAD])
    ]
    void = n_ev < 40 or len(train_pos) < 40

    mean, std = fit_normalizer(train)
    init = make_net(args.seed)
    if void:
        rec = dict(
            seed=args.seed, corpus=kind, path=str(path),
            void=True, gate=False,
            n_train=n_tr, n_eval=n_ev,
            pos_train=len(train_pos), pos_eval=pos_ev_pad,
            elapsed_s=round(time.time() - t0, 1),
        )
        print("VOID: no live eval or no train positives.")
    else:
        net = train_net(train_pos, args.seed, mean, std)
        rates = evaluate(
            net, init, ev, mean, std, random.Random(args.seed + 9),
        )
        strongest_name, strongest = max(
            (
                ("rand", rates["rand"]),
                ("init", rates["init"]),
                ("pmi", rates["pmi"]),
                ("count", rates["count"]),
                ("majority", rates["majority"]),
            ),
            key=lambda item: item[1],
        )
        d_strong = rates["phi"] - strongest
        d_init = rates["phi"] - rates["init"]
        gate = d_strong > BAR and d_init > BAR
        positive_counts = [
            sum(trial["correct"][:PAD]) for trial in train_pos
        ]
        rec = dict(
            seed=args.seed, corpus=kind, path=str(path),
            n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
            void=False, gate=gate,
            n_train=n_tr, n_eval=rates["n"],
            pos_train=len(train_pos), pos_train_all=pos_tr_all,
            pos_train_pad=pos_tr_pad,
            pos_eval_all=pos_ev_all, pos_eval_pad=pos_ev_pad,
            mean_positive_places=(
                sum(positive_counts) / len(positive_counts)
                if positive_counts else 0.0
            ),
            phi=rates["phi"], rand=rates["rand"],
            init=rates["init"], pmi=rates["pmi"],
            count=rates["count"], majority=rates["majority"],
            peak=rates["peak"], oracle=rates["oracle"],
            strongest=strongest_name,
            d_strong=d_strong, d_init=d_init, bar=BAR,
            loss="multi_positive_listwise",
            refuse_class=False,
        )
        print(
            f"eval {rates['n']}  pos_train {len(train_pos)}  "
            f"multi_pos {rec['mean_positive_places']:.2f}  "
            f"phi {rates['phi']:.3f}  init {rates['init']:.3f}  "
            f"rand {rates['rand']:.3f}  pmi {rates['pmi']:.3f}  "
            f"count {rates['count']:.3f}  maj {rates['majority']:.3f}"
        )
        print(
            f"oracle {rates['oracle']:.3f}  peak {rates['peak']:.3f}  "
            f"strongest {strongest_name} {strongest:.3f}  "
            f"d_strong {d_strong:+.3f}  d_init {d_init:+.3f}"
        )
        if gate:
            print("GO MULTIPOS: exact-place ranking learned.")
        else:
            print(
                "STOP MULTIPOS: corrected teacher still cannot rank W."
            )

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
