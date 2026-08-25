"""635: score(place) on the 634 W. Learning, not another ceiling.

634 GO: unbundled addresses keep DIRECT and HOPONLY; peak is ~10× worse.
Oracle UNION ≈ 0.26, random ≈ 0.03. Room to rank.

Φ scores each candidate PLACE. Teacher = that place's extract == held_ask.
Held is not a feature. Token identity is not a feature. No next-token CE.

Train on the first 70% of lines, eval on the rest — same file, unseen rows.
Empty W is a miss, not a dropped trial.

GATE, declared before the run:
    VOID  eval n_live < 40  or  train held-in-W < 40
    L     unique_Φ − random > 0.05
    T     unique_Φ − init   > 0.05   (training moved something)
Peak is printed, not a bar (634 already showed it is dead).
623 / KEEP_623 are not in the gate.

    python _check635_placescore.py
    python _audit635_placescore.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit635_placescore.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit635_placescore.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import bands, build_places
from _audit615_depth import K
from _audit618_peakpin import peak_pin
from _audit624_pick import hide_two
from _audit633_gapcon import extracts_633, unbundle
from _integrated_contract_v1 import CAP, leftover_records

OUT = Path("results/_stage635_placescore.json")
FDIM = 8
PAD = 16
BAR = 0.05
EPOCHS = 12
LR = 0.05


def feat_place(kind, tok, pi, pg, env_m, co, df, n_use):
    """Held-blind. No token id. Counts and structure of this address only."""
    place = pg["places"][pi]
    keys = place.get("keys") or ()
    vals = place.get("vals") or ()
    n_keys = max(len(keys), 1)
    # majority is the majority filler string; count_key[0] is its share.
    maj_frac, neg_nvals, _neg_nkeys = place["count_key"]
    pmi = 0.0
    if env_m and tok is not None:
        de = max(df.get(tok, 1), 1)
        acc = 0.0
        for word in sorted(env_m):
            c = co.get((tok, word), 0)
            if c <= 0:
                continue
            dw = max(df.get(word, 1), 1)
            acc += math.log(max((c * n_use) / (de * dw), 1e-9))
        pmi = acc / len(env_m)
    return [
        1.0 if kind == "direct" else 0.0,
        1.0 if kind == "hop1" else 0.0,
        math.log(n_keys),
        float(maj_frac),
        float(-neg_nvals) / n_keys,
        pmi,
        math.log(1.0 + len(vals)),
        0.0,  # filled with rank among this trial's PMI
    ]


def with_pmi_rank(rows):
    order = sorted(range(len(rows)), key=lambda i: -rows[i][5])
    last = max(len(rows) - 1, 1)
    for rank, i in enumerate(order):
        rows[i][7] = rank / last
    return rows


class PlaceScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FDIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.refuse = nn.Parameter(torch.zeros(1))

    def logits(self, x, mask):
        s = self.net(x).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        ref = self.refuse.expand(x.size(0))
        return torch.cat([s, ref.unsqueeze(1)], dim=1)


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
                    toks = [tok for _k, tok, _pi in cands]
                    y = next(
                        (i for i, tok in enumerate(toks) if tok == held_ask),
                        None,
                    )
                    p_hit = int(peak is not None and peak["tok"] == held_ask)
                finally:
                    hide_two(co, df, query["keys"], held_ctx, held_ask, +1)
                if not live:
                    continue
                n_live += 1
                n_pos += int(y is not None)
                trials.append(dict(
                    feats=feats, toks=toks, y=y, p_hit=p_hit,
                ))
    return trials, n_live, n_pos


def pack(batch):
    b = len(batch)
    x = torch.zeros(b, PAD, FDIM)
    mask = torch.zeros(b, PAD, dtype=torch.bool)
    y = torch.full((b,), PAD, dtype=torch.long)
    for i, tr in enumerate(batch):
        n = min(len(tr["feats"]), PAD)
        if n:
            x[i, :n] = torch.tensor(tr["feats"][:n], dtype=torch.float32)
            mask[i, :n] = True
        if tr["y"] is None or tr["y"] >= n:
            y[i] = PAD
        else:
            y[i] = tr["y"]
    return x, mask, y


def unique_rate(net, trials, rng):
    net.eval()
    n = n_phi = n_rand = n_peak = 0
    with torch.no_grad():
        for tr in trials:
            n += 1
            n_peak += tr["p_hit"]
            toks = tr["toks"]
            y = tr["y"]
            n_rand += int(
                bool(toks)
                and rng.choice(range(len(toks))) == (y if y is not None else -1)
            )
            if not tr["feats"]:
                continue
            x, mask, _y = pack([tr])
            log = net.logits(x, mask)[0]
            pick = int(log.argmax().item())
            if pick < len(toks) and y is not None and pick == y:
                n_phi += 1
    return (
        n_phi / n if n else 0.0,
        n_rand / n if n else 0.0,
        n_peak / n if n else 0.0,
        n,
    )


def train_net(train, seed):
    net = PlaceScore()
    torch.manual_seed(seed)
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    opt = torch.optim.SGD(net.parameters(), lr=LR)
    rng = random.Random(seed + 3)
    for _ in range(EPOCHS):
        order = list(range(len(train)))
        rng.shuffle(order)
        net.train()
        for a in range(0, len(order), 32):
            batch = [train[i] for i in order[a:a + 32]]
            x, mask, y = pack(batch)
            opt.zero_grad()
            log = net.logits(x, mask)
            F.cross_entropy(log, y).backward()
            opt.step()
    return net


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
    cut = int(0.7 * len(all_lines))
    train_pool = all_lines[:cut][: args.lines]
    eval_pool = all_lines[cut:][: args.lines]
    rng_tr = random.Random(args.seed)
    rng_ev = random.Random(args.seed + 1)
    t0 = time.time()
    print(f"635 placescore  {path}  {kind}", flush=True)

    train, n_tr, pos_tr = collect(train_pool, args, rng_tr)
    ev, n_ev, pos_ev = collect(eval_pool, args, rng_ev)
    void = n_ev < 40 or pos_tr < 40

    init = PlaceScore()
    torch.manual_seed(args.seed)
    for p in init.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    u_init, r_init, p_init, _ = unique_rate(init, ev, random.Random(args.seed + 7))

    if void or not train:
        rec = dict(
            seed=args.seed, corpus=kind, path=str(path),
            void=True, gate=False, n_train=n_tr, n_eval=n_ev,
            pos_train=pos_tr, pos_eval=pos_ev,
            elapsed_s=round(time.time() - t0, 1),
        )
        print("VOID: no live eval or no train held-in-W.")
    else:
        net = train_net(train, args.seed)
        u_phi, u_rand, u_peak, n_ev2 = unique_rate(
            net, ev, random.Random(args.seed + 9),
        )
        gate = (
            (not void)
            and (u_phi - u_rand) > BAR
            and (u_phi - u_init) > BAR
        )
        rec = dict(
            seed=args.seed, corpus=kind, path=str(path),
            n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
            void=False, gate=gate,
            n_train=n_tr, n_eval=n_ev2, pos_train=pos_tr, pos_eval=pos_ev,
            unique=u_phi, rand=u_rand, peak=u_peak, init=u_init,
            d_rand=u_phi - u_rand, d_init=u_phi - u_init, d_peak=u_phi - u_peak,
            bar=BAR,
        )
        print(
            f"eval {n_ev2}  pos_train {pos_tr}  "
            f"phi {u_phi:.3f}  init {u_init:.3f}  rand {u_rand:.3f}  "
            f"peak {u_peak:.3f}  d_rand {u_phi - u_rand:+.3f}  "
            f"d_init {u_phi - u_init:+.3f}"
        )
        if gate:
            print("GO LEARN: phi ranks unbundled places above random and init.")
        else:
            print(
                "STOP LEARN: scoring places did not beat random/init. "
                "No more phi on this W."
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
