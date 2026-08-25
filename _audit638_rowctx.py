"""638: learned row/context encoder over the exact-place W.

634 proved that exact DIRECT and HOPONLY places retain the answer.  637 closed
the eight-number role/count/mean-PMI summary: even a corrected multi-positive
teacher could not beat init/PMI by .05.

638 keeps the same exact-place output and frozen tape read, but the mind now
sees a variable set of held-blind tape relations for every candidate:

  QUERY <-> candidate output and address keys
  QUERY <-> candidate place filler rows
  VIA   <-> candidate output and address keys
  CURRENT <-> QUERY and candidate output

Each atom contains equality and live tape co/df/lift counts, never a token ID.
DIRECT/HOPONLY candidates retain exact query/candidate/VIA pointers.  A shared
DeepSets encoder learns how to compose row and path evidence, so future W can
append more role-tagged VIA atoms without changing the place output alphabet.
Facts remain literals and counts on each rotating window tape.

Teacher/loss: all candidate places whose frozen extract equals held are
positive; multi-positive listwise loss; no REFUSE and no vocabulary CE.
Train/eval are disjoint 70/30 lines.  Empty W is an eval miss.

Rivals: random exact place, identical init, shuffled-label null, fixed PMI,
fixed count-key, and majority filler at the selected place.

Predeclared gate:
  VOID  eval live < 80, train positive trials < 80, or oracle room <= .05
  phi - strongest rival > .05
  phi - init > .05
Peak and no-VIA ablation are diagnostics, not bars.

    python _check638_rowctx.py
    python _audit638_rowctx.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit638_rowctx.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit638_rowctx.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places
from _audit615_depth import K
from _audit618_peakpin import peak_pin
from _audit624_pick import hide_two
from _audit633_gapcon import (
    extract_633,
    extracts_633,
    open_hop1_623,
)
from _integrated_contract_v1 import CAP, commit_resolved, leftover_records

OUT = Path("results/_stage638_rowctx.json")

PAD = 16
ATOM_CAP = 96
ATOM_TYPES = 7
REL_DIM = 8
ATOM_DIM = ATOM_TYPES + REL_DIM
BASE_DIM = 12
HIDDEN = 32

BAR = 0.05
EPOCHS = 35
LR = 0.003
BATCH = 8


def contextual_offer(
    pg, records, pin, skip, env_m, mid_set, high_set, co, df, n_use,
):
    """634 candidates plus exact VIA provenance; held is unavailable."""
    candidates = []
    opened = []
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
        opened.append(opened_door)
        for observation in opened_door["observations"]:
            candidates.append(dict(
                kind="hop1",
                tok=observation["tok"],
                pi=observation["hop_pi"],
                via_pi=door_pi,
                current=door,
            ))
    return candidates, commit_resolved(opened)


def relation_atom(kind, left, right, co, df, n_use, frequency=0.0):
    """One name-free relation read from the current tape."""
    onehot = [0.0] * ATOM_TYPES
    onehot[kind] = 1.0
    dl = max(df.get(left, 0), 0)
    dr = max(df.get(right, 0), 0)
    joint = max(co.get((left, right), 0), 0)
    log_n = max(math.log1p(n_use), 1.0)
    if joint > 0 and dl > 0 and dr > 0:
        lift = (joint * n_use) / (dl * dr)
        log_lift = math.log(max(lift, 1e-9))
    else:
        log_lift = 0.0
    lift_t = math.tanh(log_lift / 4.0)
    rel = [
        float(left == right),
        math.log1p(joint) / log_n,
        lift_t,
        max(lift_t, 0.0),
        math.log1p(dl) / log_n,
        math.log1p(dr) / log_n,
        min(joint / max(min(dl, dr), 1), 1.0),
        min(max(float(frequency), 0.0), 1.0),
    ]
    return onehot + rel


def candidate_tensors(
    pg, candidate, query_pi, query_words, co, df, n_use, frame_max,
):
    """Base place state plus variable QUERY/rows/VIA relation atoms."""
    place = pg["places"][candidate["pi"]]
    tok = candidate["tok"]
    current = candidate["current"]
    via_pi = candidate["via_pi"]
    keys = list(dict.fromkeys(place["keys"]))
    values = list(place["vals"])
    counts = Counter(values)
    distinct_values = list(dict.fromkeys(values))[:8]
    qwords = sorted(set(query_words))
    via_place = pg["places"][via_pi] if via_pi is not None else None
    via_keys = (
        list(dict.fromkeys(via_place["keys"])) if via_place is not None else []
    )

    atoms = []

    def add_pairs(kind, lefts, rights, cap, frequencies=None):
        added = 0
        for left in lefts:
            freq = 0.0 if frequencies is None else frequencies.get(left, 0.0)
            for right in rights:
                if added >= cap or len(atoms) >= ATOM_CAP:
                    return
                atoms.append(relation_atom(
                    kind, left, right, co, df, n_use, freq,
                ))
                added += 1

    # Preserve each information channel before row expansion consumes capacity.
    add_pairs(0, [tok], qwords, 8)                 # output -> QUERY
    add_pairs(5, [current], qwords, 8)             # CURRENT -> QUERY
    add_pairs(6, [tok], [current], 1)              # output -> CURRENT
    add_pairs(4, [tok], via_keys, 8)               # output -> VIA
    add_pairs(3, keys, via_keys, 18)                # address -> VIA
    add_pairs(1, keys, qwords, 24)                  # address -> QUERY
    value_freq = {
        value: counts[value] / max(len(values), 1)
        for value in distinct_values
    }
    add_pairs(
        2, distinct_values, qwords,
        ATOM_CAP - len(atoms), value_freq,
    )                                               # filler rows -> QUERY

    maj_frac, neg_rows, _neg_keys = place["count_key"]
    width = place["addr"][0]
    via_width = via_place["addr"][0] if via_place is not None else 0
    log_n = max(math.log1p(n_use), 1.0)
    base = [
        float(candidate["kind"] == "direct"),
        float(candidate["kind"] == "hop1"),
        width / max(frame_max, 1),
        len(keys) / max(2 * frame_max, 1),
        math.log1p(len(values)) / 5.0,
        math.log1p(len(counts)) / 4.0,
        float(maj_frac),
        counts.get(tok, 0) / max(-neg_rows, 1),
        math.log1p(max(df.get(tok, 0), 0)) / log_n,
        math.log1p(max(df.get(current, 0), 0)) / log_n,
        float(via_pi is not None),
        via_width / max(frame_max, 1),
    ]
    return base, atoms


def make_trial(
    pg, candidates, peak, query_pi, query_words, held_ask,
    co, df, n_use, frame_max,
):
    """Tensorized exact-place offer; held creates labels only."""
    base = torch.zeros(PAD, BASE_DIM)
    atoms = torch.zeros(PAD, ATOM_CAP, ATOM_DIM)
    candidate_mask = torch.zeros(PAD, dtype=torch.bool)
    atom_mask = torch.zeros(PAD, ATOM_CAP, dtype=torch.bool)
    correct = torch.zeros(PAD, dtype=torch.bool)
    majority_correct = torch.zeros(PAD, dtype=torch.bool)
    pmi_scores = torch.zeros(PAD)
    count_keys = []

    n = min(len(candidates), PAD)
    for ci, candidate in enumerate(candidates[:n]):
        one_base, one_atoms = candidate_tensors(
            pg, candidate, query_pi, query_words,
            co, df, n_use, frame_max,
        )
        base[ci] = torch.tensor(one_base, dtype=torch.float32)
        na = min(len(one_atoms), ATOM_CAP)
        if na:
            atoms[ci, :na] = torch.tensor(
                one_atoms[:na], dtype=torch.float32,
            )
            atom_mask[ci, :na] = True
        candidate_mask[ci] = True
        correct[ci] = candidate["tok"] == held_ask
        place = pg["places"][candidate["pi"]]
        majority_correct[ci] = place["majority"] == held_ask
        count_keys.append(tuple(place["count_key"]))
        # Exact fixed PMI rival used by 635-637.
        query_rel = [
            relation_atom(
                0, candidate["tok"], word, co, df, n_use,
            )[ATOM_TYPES + 2]
            for word in sorted(set(query_words))
        ]
        pmi_scores[ci] = (
            sum(query_rel) / len(query_rel) if query_rel else 0.0
        )
    return dict(
        base=base,
        atoms=atoms,
        candidate_mask=candidate_mask,
        atom_mask=atom_mask,
        correct=correct,
        majority_correct=majority_correct,
        pmi_scores=pmi_scores,
        count_keys=count_keys,
        p_hit=int(peak is not None and peak["tok"] == held_ask),
        n=n,
    )


def collect(pool, args, rng):
    trials = []
    n_live = n_pos = 0
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
                    live = True
                    candidates, peak = contextual_offer(
                        pg, records, pin, qi, env_m, mid_set, high_set,
                        co, df, n_use,
                    )
                    trial = make_trial(
                        pg, candidates, peak, qi, query["keys"], held_ask,
                        co, df, n_use, args.frame_max,
                    )
                finally:
                    hide_two(
                        co, df, query["keys"], held_ctx, held_ask, +1,
                    )
                if not live:
                    continue
                n_live += 1
                n_pos += int(bool(trial["correct"].any()))
                trials.append(trial)
    return trials, n_live, n_pos


def fit_normalizer(train):
    """Streaming train-only moments for base and relation atoms."""
    base_sum = torch.zeros(BASE_DIM)
    base_sq = torch.zeros(BASE_DIM)
    atom_sum = torch.zeros(ATOM_DIM)
    atom_sq = torch.zeros(ATOM_DIM)
    nb = na = 0
    for trial in train:
        bvals = trial["base"][trial["candidate_mask"]]
        avals = trial["atoms"][trial["atom_mask"]]
        if len(bvals):
            base_sum += bvals.sum(dim=0)
            base_sq += (bvals * bvals).sum(dim=0)
            nb += len(bvals)
        if len(avals):
            atom_sum += avals.sum(dim=0)
            atom_sq += (avals * avals).sum(dim=0)
            na += len(avals)

    def moments(total, square, count):
        if not count:
            return torch.zeros_like(total), torch.ones_like(total)
        mean = total / count
        var = (square / count - mean * mean).clamp_min(1e-8)
        return mean, var.sqrt().clamp_min(1e-4)

    return (*moments(base_sum, base_sq, nb), *moments(atom_sum, atom_sq, na))


class RowContextRanker(nn.Module):
    """Shared candidate encoder; output dimension is places, never words."""

    def __init__(self):
        super().__init__()
        self.atom_net = nn.Sequential(
            nn.Linear(ATOM_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.base_net = nn.Sequential(
            nn.Linear(BASE_DIM, HIDDEN),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, base, atoms, atom_mask, candidate_mask):
        ah = self.atom_net(atoms)
        am = atom_mask.unsqueeze(-1)
        mean = (ah * am).sum(dim=2) / am.sum(dim=2).clamp_min(1)
        maximum = ah.masked_fill(~am, -1e9).max(dim=2).values
        empty = ~atom_mask.any(dim=2)
        maximum = maximum.masked_fill(empty.unsqueeze(-1), 0.0)
        bh = self.base_net(base)
        score = self.score(torch.cat([bh, mean, maximum], dim=-1)).squeeze(-1)
        return score.masked_fill(~candidate_mask, -1e9)


def make_net(seed):
    torch.manual_seed(seed)
    return RowContextRanker()


def attach_null_labels(train, seed):
    """Preserve positive count but rotate truth to other W slots."""
    rng = random.Random(seed)
    for trial in train:
        n = trial["n"]
        truth = trial["correct"].clone()
        null = torch.zeros_like(truth)
        if n:
            shift = rng.randrange(1, n) if n > 1 else 0
            null[:n] = torch.roll(truth[:n], shifts=shift)
        trial["null_correct"] = null


def stack_batch(batch, norm, label_key):
    bm, bs, am, ass = norm
    base = torch.stack([trial["base"] for trial in batch])
    atoms = torch.stack([trial["atoms"] for trial in batch])
    candidate_mask = torch.stack([
        trial["candidate_mask"] for trial in batch
    ])
    atom_mask = torch.stack([trial["atom_mask"] for trial in batch])
    positive = torch.stack([trial[label_key] for trial in batch])
    base = (base - bm) / bs
    atoms = (atoms - am) / ass
    base = base.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
    atoms = atoms.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
    return base, atoms, atom_mask, candidate_mask, positive


def multi_positive_loss(logits, positive):
    pos_logits = logits.masked_fill(~positive, -1e9)
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(pos_logits, dim=1)
    ).mean()


def train_net(train, seed, norm, label_key):
    usable = [trial for trial in train if bool(trial[label_key].any())]
    net = make_net(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    rng = random.Random(seed + 17)
    for _epoch in range(EPOCHS):
        order = list(range(len(usable)))
        rng.shuffle(order)
        net.train()
        for start in range(0, len(order), BATCH):
            batch = [usable[i] for i in order[start:start + BATCH]]
            base, atoms, atom_mask, candidate_mask, positive = stack_batch(
                batch, norm, label_key,
            )
            logits = net(base, atoms, atom_mask, candidate_mask)
            loss = multi_positive_loss(logits, positive)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
    return net, len(usable)


def unique_max(values):
    if not values:
        return None
    top = max(values)
    winners = [i for i, value in enumerate(values) if value == top]
    return winners[0] if len(winners) == 1 else None


def model_scores(net, trial, norm, drop_via=False):
    bm, bs, am, ass = norm
    base = trial["base"].clone().unsqueeze(0)
    atoms = trial["atoms"].clone().unsqueeze(0)
    candidate_mask = trial["candidate_mask"].unsqueeze(0)
    atom_mask = trial["atom_mask"].clone().unsqueeze(0)
    if drop_via:
        # Atom types 3..6 and base VIA fields 10..11 are the W/path register.
        via_atom = atoms[..., 3:7].sum(dim=-1) > 0.5
        atom_mask &= ~via_atom
        base[..., 10:12] = 0.0
    base = ((base - bm) / bs).masked_fill(
        ~candidate_mask.unsqueeze(-1), 0.0,
    )
    atoms = ((atoms - am) / ass).masked_fill(
        ~atom_mask.unsqueeze(-1), 0.0,
    )
    with torch.no_grad():
        return net(base, atoms, atom_mask, candidate_mask)[0]


def evaluate(net, init, null, trials, norm, rng):
    keys = (
        "phi", "init", "null", "rand", "pmi", "count",
        "majority", "peak", "oracle", "novia",
    )
    count = Counter({key: 0 for key in keys})
    net.eval()
    init.eval()
    null.eval()
    for trial in trials:
        n = trial["n"]
        truth = trial["correct"][:n].tolist()
        count["oracle"] += int(any(truth))
        count["peak"] += trial["p_hit"]
        if not n:
            continue
        count["rand"] += int(truth[rng.randrange(n)])
        pmi_pick = unique_max(trial["pmi_scores"][:n].tolist())
        if pmi_pick is not None:
            count["pmi"] += int(truth[pmi_pick])
        count_pick = unique_max(trial["count_keys"][:n])
        if count_pick is not None:
            count["count"] += int(truth[count_pick])

        picks = {}
        for name, model in (("phi", net), ("init", init), ("null", null)):
            scores = model_scores(model, trial, norm)[:n].tolist()
            picks[name] = unique_max(scores)
            if picks[name] is not None:
                count[name] += int(truth[picks[name]])
        if picks["phi"] is not None:
            count["majority"] += int(
                trial["majority_correct"][picks["phi"]]
            )
        novia = unique_max(
            model_scores(net, trial, norm, drop_via=True)[:n].tolist(),
        )
        if novia is not None:
            count["novia"] += int(truth[novia])

    den = max(len(trials), 1)
    rates = {key: count[key] / den for key in keys}
    rates["n"] = len(trials)
    return rates


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

    torch.set_num_threads(1)
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
    print(f"638 rowctx  {path}  {kind}", flush=True)

    train, n_tr, pos_tr = collect(
        all_lines[:cut][: args.lines],
        args,
        random.Random(args.seed),
    )
    ev, n_ev, pos_ev = collect(
        all_lines[cut:][: args.lines],
        args,
        random.Random(args.seed + 1),
    )
    attach_null_labels(train, args.seed + 991)
    norm = fit_normalizer(train)
    init = make_net(args.seed)
    net, n_pos_train = train_net(train, args.seed, norm, "correct")
    null, n_null_train = train_net(
        train, args.seed, norm, "null_correct",
    )
    rates = evaluate(
        net, init, null, ev, norm, random.Random(args.seed + 9),
    )
    rival_names = ("rand", "init", "null", "pmi", "count", "majority")
    strongest_name = max(rival_names, key=lambda name: rates[name])
    strongest = rates[strongest_name]
    room = rates["oracle"] - strongest
    void = n_ev < 80 or n_pos_train < 80 or room <= BAR
    d_strong = rates["phi"] - strongest
    d_init = rates["phi"] - rates["init"]
    gate = (not void) and d_strong > BAR and d_init > BAR
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
        void=void, gate=gate,
        n_train=n_tr, n_eval=n_ev,
        pos_train=pos_tr, pos_eval=pos_ev,
        truth_updates=n_pos_train, null_updates=n_null_train,
        phi=rates["phi"], init=rates["init"], null=rates["null"],
        rand=rates["rand"], pmi=rates["pmi"], count=rates["count"],
        majority=rates["majority"], peak=rates["peak"],
        oracle=rates["oracle"], no_via=rates["novia"],
        strongest=strongest_name, room=room,
        d_strong=d_strong, d_init=d_init,
        context_lift=rates["phi"] - rates["novia"],
        bar=BAR, output="exact_place",
        token_ids=False, refuse_class=False,
    )
    print(
        f"eval {n_ev}  pos_train {n_pos_train}  "
        f"phi {rates['phi']:.3f}  init {rates['init']:.3f}  "
        f"null {rates['null']:.3f}  rand {rates['rand']:.3f}  "
        f"pmi {rates['pmi']:.3f}  count {rates['count']:.3f}  "
        f"maj {rates['majority']:.3f}"
    )
    print(
        f"oracle {rates['oracle']:.3f}  peak {rates['peak']:.3f}  "
        f"no_via {rates['novia']:.3f}  strongest "
        f"{strongest_name} {strongest:.3f}  room {room:+.3f}"
    )
    print(
        f"d_strong {d_strong:+.3f}  d_init {d_init:+.3f}  "
        f"context_lift {rec['context_lift']:+.3f}"
    )
    if void:
        print("VOID ROWCTX: thin exam or no oracle room.")
    elif gate:
        print("GO ROWCTX: weights learned exact-place context.")
    else:
        print("STOP ROWCTX: learned row/context encoder did not clear rivals.")

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
