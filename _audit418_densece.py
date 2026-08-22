"""418: CE ON 417's DENSE PIN. Softmax over places+REFUSE; y from dense_labels.

Not cut (417cl). Not another 416 expected-reward. Teacher is 417's y.

  loss   -sum_i y_i log softmax(scores)_i     scores = pin logits (places + REFUSE)
  VOID   live teacher rate <= 0.05 on the train window (same as 417)
  GATE   held-out: mind pin-hit > random pin-hit; refuse@df1 > refuse@df>=2
  FAIL   means labels do not fit Phi — not "need cut"
  Lab hole-seeds: only AFTER freeze, as a probe, never in the loss.

    python _check418_densece.py
    python _audit418_densece.py --seed 1337 --steps 3000 --cpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import _audit390_address as A
import _audit417_densepin as D417

OUT = Path("results/_stage418_densece.json")
REFUSE_IX = -1


def hash_fp(parts, d: int) -> torch.Tensor:
    v = torch.zeros(d)
    for i, p in enumerate(parts):
        h = hashlib.sha1(f"{i}:{p}".encode("utf-8")).digest()
        for j in range(d):
            v[j] += ((h[j % 20] / 255.0) * 2.0 - 1.0) * (
                1.0 if (h[(j + 3) % 20] & 1) == 0 else -1.0)
    n = float(v.norm().clamp(min=1e-8))
    return v / n


class PinMind(nn.Module):
    """Scores window → each pin place, plus REFUSE. No word vocab."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.d = d
        self.qw = nn.Linear(d, d)
        self.pw = nn.Linear(d, d)
        self.sc = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.sc[-1].weight)
        nn.init.zeros_(self.sc[-1].bias)

    def scores(self, q, P):
        """q [d], P [K,d] → logits [K+1] (last = REFUSE)."""
        hq = self.qw(q)
        if P.shape[0] == 0:
            z = torch.zeros(self.d, device=q.device, dtype=q.dtype)
            rf = self.sc(torch.cat([hq, self.pw(z)], -1)).squeeze(-1)
            return rf.reshape(1)
        hp = self.pw(P)
        K = P.shape[0]
        qq = hq.unsqueeze(0).expand(K, -1)
        ps = self.sc(torch.cat([qq, hp], -1)).squeeze(-1)
        z = torch.zeros(self.d, device=q.device, dtype=q.dtype)
        rf = self.sc(torch.cat([hq, self.pw(z)], -1)).squeeze(-1)
        return torch.cat([ps, rf.reshape(1)])


def pack_step(T, ix, s, cap, d, device):
    st = D417.step_of(T, ix, s, cap)
    if st is None:
        return None
    # Query key = window keys only (hole already excluded in 417.window_keys).
    q = hash_fp([f"K:{k}" for k in st["keys"]] or ["K:EMPTY"], d).to(device)
    if st["cands"]:
        P = torch.stack([
            hash_fp(D417.fillers_place(T, j, hide=s) or ["EMPTY"], d)
            for j in st["cands"]
        ]).to(device)
    else:
        P = torch.zeros(0, d, device=device)
    y = torch.tensor(st["y"], device=device, dtype=torch.float32)
    return {**st, "q": q, "P": P, "y": y}


def pin_correct(pick, st):
    """pick: index in [0..K] or REFUSE_IX meaning last. Success vs dense teacher support."""
    k = len(st["cands"])
    if pick == REFUSE_IX or pick == k:
        return bool(st["refuse"])
    if pick < 0 or pick >= k:
        return False
    return st["cands"][pick] in st["hits"]


def run_exam(net, T, ix, slots, cap, d, device, rng):
    net.eval()
    tot = {"n": 0, "mind": 0, "rnd": 0}
    df1 = {"n": 0, "refuse": 0}
    df2 = {"n": 0, "refuse": 0}
    with torch.no_grad():
        for s in slots:
            sp = pack_step(T, ix, s, cap, d, device)
            if sp is None:
                continue
            logits = net.scores(sp["q"], sp["P"])
            mind = int(logits.argmax())
            k = logits.numel()
            rnd = rng.randrange(k)
            tot["n"] += 1
            tot["mind"] += int(pin_correct(mind, sp))
            tot["rnd"] += int(pin_correct(rnd, sp))
            refused = int(mind == (k - 1))
            if sp["df"] <= 1:
                df1["n"] += 1
                df1["refuse"] += refused
            else:
                df2["n"] += 1
                df2["refuse"] += refused

    def rate(b, key, nkey="n"):
        return b[key] / b[nkey] if b[nkey] else float("nan")

    mind_hit = rate(tot, "mind")
    rnd_hit = rate(tot, "rnd")
    r1, r2 = rate(df1, "refuse"), rate(df2, "refuse")
    return {
        "n": tot["n"],
        "mind_pin": mind_hit,
        "random_pin": rnd_hit,
        "mind_minus_random": (mind_hit - rnd_hit) if tot["n"] else float("nan"),
        "refuse_df1": r1,
        "refuse_df2": r2,
        "n_df1": df1["n"],
        "n_df2": df2["n"],
        "gate_mind_beats_random": bool(
            tot["n"] and mind_hit == mind_hit and rnd_hit == rnd_hit
            and mind_hit > rnd_hit),
        "gate_refuse_one_gt_ge2": bool(
            df1["n"] and df2["n"] and r1 == r1 and r2 == r2 and r1 > r2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bytes", type=int, default=8_000_000)
    ap.add_argument("--lines", type=int, default=4000)
    ap.add_argument("--window-lines", type=int, default=800)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exam-slots", type=int, default=800)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--save-mind", default=None)
    ap.add_argument("--cpu", action="store_true")
    # Lab hole probe is opt-in and never trains:
    ap.add_argument("--lab-probe", action="store_true",
                    help="after freeze only: print a stub note; never enters the loss")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    rng = random.Random(args.seed)
    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(max(1, len(lines) - args.window_lines))
        lines = lines[s0 : s0 + args.window_lines]
    T = A.build_tape(lines, args.frame_max, args.min_fillers)
    ix = D417.by_ctx_of(T)
    slots = [s for s in T["place_of"]]
    if len(slots) < 50:
        print("tape too small")
        return 1
    rng.shuffle(slots)
    cut = max(1, int(0.8 * len(slots)))
    train_slots, exam_slots = slots[:cut], slots[cut : cut + args.exam_slots]

    # VOID on teacher live rate (sample)
    live = 0
    for s in train_slots[:500]:
        st = D417.step_of(T, ix, s, args.cap)
        if st is not None:
            live += int(not st["refuse"])
    live_rate = live / max(1, min(500, len(train_slots)))
    print(f"places={len(T['places'])} slots={len(slots)} teacher_live={live_rate:.4f}")
    if live_rate <= 0.05:
        print("VOID: teacher almost always refuse — labels have no pin to learn")
        return 1

    net = PinMind(args.dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    losses = []
    for step in range(1, args.steps + 1):
        s = train_slots[rng.randrange(len(train_slots))]
        sp = pack_step(T, ix, s, args.cap, args.dim, device)
        if sp is None:
            continue
        logits = net.scores(sp["q"], sp["P"])
        # CE against dense_labels — never a vocab, never 416's R, never a 289 hole.
        logp = F.log_softmax(logits, 0)
        loss = -(sp["y"] * logp).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % max(1, args.steps // 8) == 0:
            print(f"  step {step}/{args.steps} loss="
                  f"{sum(losses[-200:]) / max(1, len(losses[-200:])):.4f}")

    exam_rng = random.Random(args.seed + 99)
    rep = run_exam(net, T, ix, exam_slots, args.cap, args.dim, device, exam_rng)
    rep.update({
        "seed": args.seed, "steps": args.steps, "teacher_live": live_rate,
        "params": int(sum(p.numel() for p in net.parameters())),
        "n_train": len(train_slots), "n_exam": len(exam_slots),
        "loss": "ce_dense_labels_417",
    })
    print(json.dumps({k: rep[k] for k in (
        "n", "mind_pin", "random_pin", "mind_minus_random",
        "refuse_df1", "refuse_df2",
        "gate_mind_beats_random", "gate_refuse_one_gt_ge2")}, indent=1))
    print("GATE mind>random:", rep["gate_mind_beats_random"],
          "  refuse_df1>df2:", rep["gate_refuse_one_gt_ge2"])
    if not rep["gate_mind_beats_random"]:
        print("FAIL reading: labels did not fit Phi — not 'need cut'")

    if args.save_mind:
        Path(args.save_mind).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "dim": args.dim, "seed": args.seed,
                    "note": "418 CE on 417 dense pins"}, args.save_mind)

    if args.lab_probe:
        # Explicitly not in the loss. Freeze-only щуп placeholder.
        print("LAB PROBE (not in loss): frozen mind ready; hole-seeds not run in this file")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
