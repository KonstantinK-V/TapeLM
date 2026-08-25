"""636: rank the 634 W without a REFUSE class. Torch.

635 STOP: CE+REFUSE + 72% silence → always refuse (phi≈.002).
634 GO stands: the offer is live. This run changes only the loss.

Train only trials where held ∈ W. Softmax over those places.
Eval on all n_live (empty W = miss). Same split, same unbundle, same feats
as fixed 635. No extra layers.

GATE, declared before the run:
    VOID  eval n_live < 40  or  train pos < 40
    L     unique − random > 0.05
    T     unique − init   > 0.05
Peak printed, not a bar.

    python _check636_posrank.py
    python _audit636_posrank.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit636_posrank.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit636_posrank.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
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
from _audit635_placescore import FDIM, PAD, feat_place, with_pmi_rank
from _integrated_contract_v1 import CAP, leftover_records

OUT = Path("results/_stage636_posrank.json")
BAR = 0.05
EPOCHS = 12
LR = 0.05


class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FDIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def logits(self, x, mask):
        s = self.net(x).squeeze(-1)
        return s.masked_fill(~mask, -1e9)


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
                        feat_place(kind, tok, pi, pg, env_m, co, df, n_use)
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
    y = torch.zeros(b, dtype=torch.long)
    for i, tr in enumerate(batch):
        n = min(len(tr["feats"]), PAD)
        if n:
            x[i, :n] = torch.tensor(tr["feats"][:n], dtype=torch.float32)
            mask[i, :n] = True
        # Train batches are pos-only with y < n; eval may pack miss trials.
        y[i] = 0 if tr["y"] is None else int(tr["y"])
    return x, mask, y


def unique_rate(net, trials, rng):
    net.eval()
    n = n_hit = n_rand = n_peak = 0
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
            pick = int(net.logits(x, mask)[0].argmax().item())
            if y is not None and pick == y:
                n_hit += 1
    return (
        n_hit / n if n else 0.0,
        n_rand / n if n else 0.0,
        n_peak / n if n else 0.0,
        n,
    )


def train_net(pos, seed):
    net = RankNet()
    torch.manual_seed(seed)
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    opt = torch.optim.SGD(net.parameters(), lr=LR)
    rng = random.Random(seed + 3)
    for _ in range(EPOCHS):
        order = list(range(len(pos)))
        rng.shuffle(order)
        net.train()
        for a in range(0, len(order), 32):
            batch = [pos[i] for i in order[a:a + 32]]
            x, mask, y = pack(batch)
            opt.zero_grad()
            F.cross_entropy(net.logits(x, mask), y).backward()
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
    rng_tr = random.Random(args.seed)
    rng_ev = random.Random(args.seed + 1)
    t0 = time.time()
    print(f"636 posrank  {path}  {kind}", flush=True)

    train, n_tr, pos_tr = collect(all_lines[:cut][: args.lines], args, rng_tr)
    ev, n_ev, pos_ev = collect(all_lines[cut:][: args.lines], args, rng_ev)
    # held in W, and the teacher index still fits the scored prefix.
    pos = [
        tr for tr in train
        if tr["y"] is not None and tr["feats"] and tr["y"] < min(len(tr["feats"]), PAD)
    ]
    void = n_ev < 40 or len(pos) < 40

    init = RankNet()
    torch.manual_seed(args.seed)
    for p in init.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    u_init, _r0, _p0, _n0 = unique_rate(init, ev, random.Random(args.seed + 7))

    if void:
        rec = dict(
            seed=args.seed, corpus=kind, path=str(path),
            void=True, gate=False, n_train=n_tr, n_eval=n_ev,
            pos_train=len(pos), pos_eval=pos_ev,
            elapsed_s=round(time.time() - t0, 1),
        )
        print("VOID: no live eval or no train pos.")
    else:
        net = train_net(pos, args.seed)
        u_phi, u_rand, u_peak, n_ev2 = unique_rate(
            net, ev, random.Random(args.seed + 9),
        )
        gate = (u_phi - u_rand) > BAR and (u_phi - u_init) > BAR
        rec = dict(
            seed=args.seed, corpus=kind, path=str(path),
            n_win=args.n_win, elapsed_s=round(time.time() - t0, 1),
            void=False, gate=gate,
            n_train=n_tr, n_eval=n_ev2,
            pos_train=len(pos), pos_eval=pos_ev, pos_all=pos_tr,
            unique=u_phi, rand=u_rand, peak=u_peak, init=u_init,
            d_rand=u_phi - u_rand, d_init=u_phi - u_init, d_peak=u_phi - u_peak,
            bar=BAR, refuse_class=False,
        )
        print(
            f"eval {n_ev2}  pos_train {len(pos)}  "
            f"phi {u_phi:.3f}  init {u_init:.3f}  rand {u_rand:.3f}  "
            f"peak {u_peak:.3f}  d_rand {u_phi - u_rand:+.3f}  "
            f"d_init {u_phi - u_init:+.3f}"
        )
        if gate:
            print("GO RANK: pos-only CE ranks places above random and init.")
        else:
            print(
                "STOP RANK: this place representation does not rank. "
                "Close scoring this W."
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
