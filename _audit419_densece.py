"""419: CE ON 417h's HONEST DENSE PIN. Softmax over places+REFUSE; y from joint labels.

Not 418 (that was CE on 417's one-word OR). Teacher is 417h: overlap>=2, hole out of bag.

  loss   -sum_i y_i log softmax(scores)_i     scores = pin logits (places + REFUSE)
  VOID   live teacher rate <= 0.05 among non-thin steps (|keys|>=min_keys)
  GATE   held-out: mind pin-hit > random pin-hit; refuse@df1 > refuse@df>=2
  READ   mind_live (hit | teacher has pin), always_refuse (rival = always REFUSE),
         refuse_df1 / refuse_df2 — standing metrics, not a new lever
  FAIL   means labels do not fit Phi — not "need cut"
  Lab hole-seeds: only AFTER freeze, as a probe, never in the loss.
  Freeze: out/_mind_419_densece_s*.pt + results/_stage419_densece.json standing.
    python _check419_densece.py
    python _audit419_densece.py --seed 1337 --steps 3000 --cpu
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
import _audit417h_densepin as D417h

OUT = Path("results/_stage419_densece.json")
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


def pack_step(T, bags, s, cap, joint, min_keys, d, device):
    st = D417h.step_of(T, bags, s, cap, joint, min_keys)
    if st is None or st.get("thin"):
        return None
    # Query key = window keys only (hole already excluded in 417h.window_keys).
    q = hash_fp([f"K:{k}" for k in st["keys"]] or ["K:EMPTY"], d).to(device)
    if st["cands"]:
        P = torch.stack([
            hash_fp(D417h.fillers_place(T, j, hide=s) or ["EMPTY"], d)
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


def run_exam(net, T, bags, slots, cap, joint, min_keys, d, device, rng):
    net.eval()
    tot = {"n": 0, "mind": 0, "rnd": 0, "always_refuse": 0}
    live = {"n": 0, "mind": 0}
    df1 = {"n": 0, "refuse": 0}
    df2 = {"n": 0, "refuse": 0}
    with torch.no_grad():
        for s in slots:
            sp = pack_step(T, bags, s, cap, joint, min_keys, d, device)
            if sp is None:
                continue
            logits = net.scores(sp["q"], sp["P"])
            mind = int(logits.argmax())
            k = logits.numel()
            rnd = rng.randrange(k)
            tot["n"] += 1
            tot["mind"] += int(pin_correct(mind, sp))
            tot["rnd"] += int(pin_correct(rnd, sp))
            # Rival: always pick REFUSE — correct iff teacher has no hit.
            tot["always_refuse"] += int(bool(sp["refuse"]))
            if not sp["refuse"]:
                live["n"] += 1
                live["mind"] += int(pin_correct(mind, sp))
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
    mind_live = rate(live, "mind")
    always_refuse = rate(tot, "always_refuse")
    r1, r2 = rate(df1, "refuse"), rate(df2, "refuse")
    return {
        "n": tot["n"],
        "mind_pin": mind_hit,
        "random_pin": rnd_hit,
        "mind_minus_random": (mind_hit - rnd_hit) if tot["n"] else float("nan"),
        "mind_live": mind_live,
        "always_refuse": always_refuse,
        "n_live": live["n"],
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
    ap.add_argument("--joint", type=int, default=2)
    ap.add_argument("--min-keys", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exam-slots", type=int, default=800)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--save-mind", default=None)
    ap.add_argument("--load-mind", default=None,
                    help="freeze/exam-only: load standing mind, skip train")
    ap.add_argument("--cpu", action="store_true")
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
    bags = D417h.place_bags(T)
    slots = [s for s in T["place_of"]]
    if len(slots) < 50:
        print("tape too small")
        return 1
    rng.shuffle(slots)
    cut = max(1, int(0.8 * len(slots)))
    train_slots, exam_slots = slots[:cut], slots[cut : cut + args.exam_slots]

    # VOID on teacher live rate among non-thin (same as 417h)
    live = thin = seen = 0
    for s in train_slots:
        if seen >= 500:
            break
        st = D417h.step_of(T, bags, s, args.cap, args.joint, args.min_keys)
        if st is None:
            continue
        if st.get("thin"):
            thin += 1
            continue
        seen += 1
        live += int(not st["refuse"])
    live_rate = live / max(1, seen)
    print(f"places={len(T['places'])} slots={len(slots)} "
          f"teacher_live={live_rate:.4f} thin_skipped_sample={thin} "
          f"joint={args.joint} min_keys={args.min_keys}")
    if seen == 0 or live_rate <= 0.05:
        print("VOID: joint teacher almost always refuse — labels have no pin to learn")
        return 1

    net = PinMind(args.dim).to(device)
    if args.load_mind:
        ckpt = torch.load(args.load_mind, map_location=device)
        net.load_state_dict(ckpt["state"])
        print(f"loaded standing mind {args.load_mind} (exam only, no train)")
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
        losses = []
        for step in range(1, args.steps + 1):
            s = train_slots[rng.randrange(len(train_slots))]
            sp = pack_step(T, bags, s, args.cap, args.joint, args.min_keys,
                           args.dim, device)
            if sp is None:
                continue
            logits = net.scores(sp["q"], sp["P"])
            # CE against 417h dense_labels — never vocab, never 416 R, never 289 hole.
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
    rep = run_exam(net, T, bags, exam_slots, args.cap, args.joint, args.min_keys,
                   args.dim, device, exam_rng)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    steps_out = args.steps
    if args.load_mind and str(args.seed) in prev:
        steps_out = prev[str(args.seed)].get("steps", args.steps)
    rep.update({
        "seed": args.seed, "steps": steps_out,
        "teacher_live": live_rate,
        "params": int(sum(p.numel() for p in net.parameters())),
        "n_train": len(train_slots), "n_exam": len(exam_slots),
        "joint": args.joint, "min_keys": args.min_keys,
        "loss": "ce_dense_labels_417h",
        "standing": True,
        "freeze": "419_joint_ce",
    })
    print(json.dumps({k: rep[k] for k in (
        "n", "mind_pin", "random_pin", "mind_minus_random",
        "mind_live", "always_refuse", "n_live",
        "refuse_df1", "refuse_df2",
        "gate_mind_beats_random", "gate_refuse_one_gt_ge2")}, indent=1))
    print("GATE mind>random:", rep["gate_mind_beats_random"],
          "  refuse_df1>df2:", rep["gate_refuse_one_gt_ge2"])
    print(f"STANDING  mind_live {rep['mind_live']:.4f}   "
          f"always_refuse {rep['always_refuse']:.4f}   "
          f"refuse_df1 {rep['refuse_df1']:.4f}   refuse_df2 {rep['refuse_df2']:.4f}")
    if not rep["gate_mind_beats_random"]:
        print("FAIL reading: labels did not fit Phi — not 'need cut'")

    if args.save_mind and not args.load_mind:
        Path(args.save_mind).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "dim": args.dim, "seed": args.seed,
                    "note": "419 CE on 417h honest dense pins — standing joint"},
                   args.save_mind)

    if args.lab_probe:
        print("LAB PROBE (not in loss): frozen mind ready; hole-seeds not run in this file")

    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
