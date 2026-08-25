"""639: frozen-PMI skip plus learned row/VIA residual over exact places.

638 proved that the row/context encoder learns on unseen rows:
phi-init +.066..+.102 and it beats random/null/majority.  It missed the
strongest gate only because the network spent capacity relearning frozen PMI,
ending merely +.024..+.040 above it.

639 keeps the exact same 638 W, 70/30 split, hidden width, and biting rivals.
Before normalization it removes atom kind 0 (extract <-> QUERY lift) from the
residual input because that channel is precisely the frozen skip.  The place
logit is now:

    frozen_PMI(place, QUERY) + learned_row_context_residual(place, QUERY, VIA)

The residual output layer starts at exactly zero, so init == fixed PMI.
All positive/negative place pairs provide full-feedback logistic rank loss;
multi-positive listwise loss remains as a secondary term.  No REFUSE, token
IDs, vocabulary output, answer text, or held input.

Predeclared gate:
  VOID  eval live < 80, train positive trials < 80, or oracle room <= .05
  phi - strongest(random, PMI/init, shuffled null, count, majority) > .05
  phi - init > .05

    python _check639_residualctx.py
    python _audit639_residualctx.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit639_residualctx.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit639_residualctx.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from _audit511_ring import pick_corpus
from _audit638_rowctx import (
    BAR,
    BATCH,
    EPOCHS,
    PAD,
    RowContextRanker,
    attach_null_labels,
    collect,
    fit_normalizer,
    stack_batch,
)

OUT = Path("results/_stage639_residualctx.json")
LR = 0.003
PAIR_MARGIN = 0.10


class ResidualContextRanker(RowContextRanker):
    """Exact-place score with an auditable frozen PMI base."""

    def __init__(self):
        super().__init__()
        # score[-1] is the scalar residual head.  Zero makes init == PMI.
        torch.nn.init.zeros_(self.score[-1].weight)
        torch.nn.init.zeros_(self.score[-1].bias)

    def residual(self, base, atoms, atom_mask, candidate_mask):
        return super().forward(
            base, atoms, atom_mask, candidate_mask,
        )

    def forward(self, base, atoms, atom_mask, candidate_mask, pmi):
        residual = self.residual(base, atoms, atom_mask, candidate_mask)
        score = pmi + residual
        return score.masked_fill(~candidate_mask, -1e9)


def make_net(seed):
    torch.manual_seed(seed)
    return ResidualContextRanker()


def drop_skip_atoms(trials):
    """Remove kind-0 extract/QUERY atoms; PMI survives only in pmi_scores."""
    for trial in trials:
        kind0 = trial["atom_mask"] & (trial["atoms"][..., 0] > 0.5)
        trial["atom_mask"][kind0] = False
        trial["atoms"][kind0] = 0.0


def pairwise_full_feedback(logits, positive, candidate_mask):
    """Every correct-vs-wrong place pair teaches the shared residual."""
    losses = []
    for row, pos, legal in zip(logits, positive, candidate_mask):
        positives = row[pos & legal]
        negatives = row[(~pos) & legal]
        if not len(positives) or not len(negatives):
            continue
        diff = (
            negatives.unsqueeze(1)
            - positives.unsqueeze(0)
            + PAIR_MARGIN
        )
        losses.append(F.softplus(diff).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


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
            pmi = torch.stack([trial["pmi_scores"] for trial in batch])
            logits = net(
                base, atoms, atom_mask, candidate_mask, pmi,
            )
            loss = pairwise_full_feedback(
                logits, positive, candidate_mask,
            ) + 0.25 * multi_positive_loss(logits, positive)
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


def model_scores(net, trial, norm, residual_only=False):
    bm, bs, am, ass = norm
    base = trial["base"].unsqueeze(0)
    atoms = trial["atoms"].unsqueeze(0)
    candidate_mask = trial["candidate_mask"].unsqueeze(0)
    atom_mask = trial["atom_mask"].unsqueeze(0)
    base = ((base - bm) / bs).masked_fill(
        ~candidate_mask.unsqueeze(-1), 0.0,
    )
    atoms = ((atoms - am) / ass).masked_fill(
        ~atom_mask.unsqueeze(-1), 0.0,
    )
    pmi = trial["pmi_scores"].unsqueeze(0)
    with torch.no_grad():
        if residual_only:
            return net.residual(
                base, atoms, atom_mask, candidate_mask,
            )[0]
        return net(base, atoms, atom_mask, candidate_mask, pmi)[0]


def evaluate(net, init, null, trials, norm, rng):
    names = (
        "phi", "residual_only", "init", "null", "rand", "pmi",
        "count", "majority", "peak", "oracle",
    )
    counts = Counter({name: 0 for name in names})
    commits = Counter()
    for trial in trials:
        n = trial["n"]
        truth = trial["correct"][:n].tolist()
        counts["oracle"] += int(any(truth))
        counts["peak"] += trial["p_hit"]
        if not n:
            continue
        counts["rand"] += int(truth[rng.randrange(n)])
        pmi_pick = unique_max(trial["pmi_scores"][:n].tolist())
        if pmi_pick is not None:
            counts["pmi"] += int(truth[pmi_pick])
            commits["pmi"] += 1
        count_pick = unique_max(trial["count_keys"][:n])
        if count_pick is not None:
            counts["count"] += int(truth[count_pick])
            commits["count"] += 1

        picks = {}
        for name, model in (("phi", net), ("init", init), ("null", null)):
            picks[name] = unique_max(
                model_scores(model, trial, norm)[:n].tolist(),
            )
            if picks[name] is not None:
                counts[name] += int(truth[picks[name]])
                commits[name] += 1
        residual_pick = unique_max(
            model_scores(
                net, trial, norm, residual_only=True,
            )[:n].tolist(),
        )
        if residual_pick is not None:
            counts["residual_only"] += int(truth[residual_pick])
            commits["residual_only"] += 1
        if picks["phi"] is not None:
            counts["majority"] += int(
                trial["majority_correct"][picks["phi"]]
            )
    den = max(len(trials), 1)
    rates = {name: counts[name] / den for name in names}
    rates["n"] = len(trials)
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
    print(f"639 residualctx  {path}  {kind}", flush=True)

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
    drop_skip_atoms(train)
    drop_skip_atoms(ev)
    attach_null_labels(train, args.seed + 991)
    norm = fit_normalizer(train)
    init = make_net(args.seed)
    net, truth_updates = train_net(train, args.seed, norm, "correct")
    null, null_updates = train_net(
        train, args.seed, norm, "null_correct",
    )
    rates, commits = evaluate(
        net, init, null, ev, norm, random.Random(args.seed + 9),
    )
    rival_names = ("rand", "init", "null", "pmi", "count", "majority")
    strongest_name = max(rival_names, key=lambda name: rates[name])
    strongest = rates[strongest_name]
    room = rates["oracle"] - strongest
    void = n_ev < 80 or truth_updates < 80 or room <= BAR
    d_strong = rates["phi"] - strongest
    d_init = rates["phi"] - rates["init"]
    gate = (not void) and d_strong > BAR and d_init > BAR
    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
        void=void, gate=gate,
        n_train=n_tr, n_eval=n_ev,
        pos_train=pos_tr, pos_eval=pos_ev,
        truth_updates=truth_updates, null_updates=null_updates,
        phi=rates["phi"], init=rates["init"], null=rates["null"],
        residual_only=rates["residual_only"],
        rand=rates["rand"], pmi=rates["pmi"], count=rates["count"],
        majority=rates["majority"], peak=rates["peak"],
        oracle=rates["oracle"], strongest=strongest_name, room=room,
        d_strong=d_strong, d_init=d_init, bar=BAR,
        output="exact_place", token_ids=False, refuse_class=False,
        base_score="frozen_pmi", learned_score="row_context_residual",
        loss="pairwise_full_feedback_plus_multipos",
        kind0_in_residual=False, n_commit=commits,
    )
    print(
        f"eval {n_ev}  pos_train {truth_updates}  "
        f"phi {rates['phi']:.3f}  residual {rates['residual_only']:.3f}  "
        f"init {rates['init']:.3f}  pmi {rates['pmi']:.3f}  "
        f"null {rates['null']:.3f}  rand {rates['rand']:.3f}  "
        f"count {rates['count']:.3f}  maj {rates['majority']:.3f}"
    )
    print(
        f"oracle {rates['oracle']:.3f}  peak {rates['peak']:.3f}  "
        f"strongest {strongest_name} {strongest:.3f}  room {room:+.3f}"
    )
    print(
        f"d_strong {d_strong:+.3f}  d_init {d_init:+.3f}"
    )
    print(
        "n_commit "
        + " ".join(
            f"{name}={commits.get(name, 0)}"
            for name in ("phi", "residual_only", "init", "pmi", "null")
        )
    )
    if void:
        print("VOID RESIDUALCTX: thin exam or no oracle room.")
    elif gate:
        print("GO RESIDUALCTX: learned context improves frozen PMI.")
    else:
        print("STOP RESIDUALCTX: residual context does not clear PMI/rivals.")

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
