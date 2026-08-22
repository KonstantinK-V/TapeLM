"""420: BALANCED CE ON THE SAME 417h y. Refuse mass compressed to live mass.

Same teacher. Features without letters: overlap, bag size, slots, refuse flag.
No place-id in Phi. No fillers_place / token hashes in the scorer.

  w_ref   = n_live / n_ref ; live weight 1
  GATE    (1) mind_live - random_live > 0.05
          (2) mind_pin >= always_refuse
  standing only if both gates. 419 not standing.

    python _check420_balce.py
    python _train420_balce.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import _audit390_address as A
import _audit417h_densepin as D417h

OUT = Path("results/_stage420_balce.json")
NFEAT = 4  # overlap, bag_size, n_slots, refuse_flag
REFUSE_IX = -1


class PinMind(nn.Module):
    """Scores structural pin features only — no letters, no place-id."""

    def __init__(self, d: int = 32, nfeat: int = NFEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nfeat, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def scores(self, F):
        """F [K+1, nfeat] → logits [K+1]."""
        return self.net(F).squeeze(-1)


def overlap_of(bags, keys, token, j):
    return len((bags[j] - {token}) & set(keys))


def feat_row(T, bags, keys, token, j=None, refuse=False):
    if refuse:
        return [0.0, 0.0, 0.0, 1.0]
    ov = overlap_of(bags, keys, token, j)
    return [float(ov), float(len(bags[j])), float(len(T["places"][j])), 0.0]


def pack_step(T, bags, s, cap, joint, min_keys, device):
    st = D417h.step_of(T, bags, s, cap, joint, min_keys)
    if st is None or st.get("thin"):
        return None
    rows = [feat_row(T, bags, st["keys"], st["token"], j=j) for j in st["cands"]]
    rows.append(feat_row(T, bags, st["keys"], st["token"], refuse=True))
    Fmat = torch.tensor(rows, device=device, dtype=torch.float32)
    y = torch.tensor(st["y"], device=device, dtype=torch.float32)
    return {**st, "F": Fmat, "y": y}


def pin_correct(pick, st):
    k = len(st["cands"])
    if pick == REFUSE_IX or pick == k:
        return bool(st["refuse"])
    if pick < 0 or pick >= k:
        return False
    return st["cands"][pick] in st["hits"]


def split_live_refuse(T, bags, slots, cap, joint, min_keys):
    live, refuse = [], []
    for s in slots:
        st = D417h.step_of(T, bags, s, cap, joint, min_keys)
        if st is None or st.get("thin"):
            continue
        (refuse if st["refuse"] else live).append(s)
    return live, refuse


def run_exam(net, T, bags, slots, cap, joint, min_keys, device, rng):
    net.eval()
    tot = {"n": 0, "mind": 0, "always_refuse": 0}
    live = {"n": 0, "mind": 0, "rnd": 0}
    df1 = {"n": 0, "refuse": 0}
    df2 = {"n": 0, "refuse": 0}
    with torch.no_grad():
        for s in slots:
            sp = pack_step(T, bags, s, cap, joint, min_keys, device)
            if sp is None:
                continue
            logits = net.scores(sp["F"])
            mind = int(logits.argmax())
            k = logits.numel()
            rnd = rng.randrange(k)
            tot["n"] += 1
            tot["mind"] += int(pin_correct(mind, sp))
            tot["always_refuse"] += int(bool(sp["refuse"]))
            if not sp["refuse"]:
                live["n"] += 1
                live["mind"] += int(pin_correct(mind, sp))
                live["rnd"] += int(pin_correct(rnd, sp))
            refused = int(mind == (k - 1))
            if sp["df"] <= 1:
                df1["n"] += 1
                df1["refuse"] += refused
            else:
                df2["n"] += 1
                df2["refuse"] += refused

    def rate(b, key, nkey="n"):
        return b[key] / b[nkey] if b[nkey] else float("nan")

    mind_pin = rate(tot, "mind")
    always_refuse = rate(tot, "always_refuse")
    mind_live = rate(live, "mind")
    random_live = rate(live, "rnd")
    dlt = (mind_live - random_live) if live["n"] else float("nan")
    r1, r2 = rate(df1, "refuse"), rate(df2, "refuse")
    gate_pins = bool(
        live["n"] and mind_live == mind_live and random_live == random_live
        and (mind_live - random_live) > 0.05)
    gate_vs_ar = bool(
        tot["n"] and mind_pin == mind_pin and always_refuse == always_refuse
        and mind_pin >= always_refuse)
    return {
        "n": tot["n"],
        "mind_pin": mind_pin,
        "always_refuse": always_refuse,
        "mind_live": mind_live,
        "random_live": random_live,
        "mind_live_minus_random_live": dlt,
        "n_live": live["n"],
        "refuse_df1": r1,
        "refuse_df2": r2,
        "n_df1": df1["n"],
        "n_df2": df2["n"],
        "gate_pins": gate_pins,
        "gate_mind_ge_always_refuse": gate_vs_ar,
        "standing": bool(gate_pins and gate_vs_ar),
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
    ap.add_argument("--cpu", action="store_true")
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

    live_s, refuse_s = split_live_refuse(
        T, bags, train_slots, args.cap, args.joint, args.min_keys)
    n_live, n_ref = len(live_s), len(refuse_s)
    teacher_live = n_live / max(1, n_live + n_ref)
    # Refuse mass compressed to live mass: each refuse step * (n_live/n_ref).
    w_ref = n_live / max(1, n_ref)
    print(f"places={len(T['places'])} slots={len(slots)} "
          f"teacher_live={teacher_live:.4f} n_live={n_live} n_ref={n_ref} "
          f"w_refuse={w_ref:.4f} joint={args.joint}")
    if teacher_live <= 0.05 or n_live == 0 or n_ref == 0:
        print("VOID: not enough live/refuse mass to balance CE")
        return 1

    net = PinMind(args.dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    losses = []
    for step in range(1, args.steps + 1):
        s = train_slots[rng.randrange(len(train_slots))]
        sp = pack_step(T, bags, s, args.cap, args.joint, args.min_keys, device)
        if sp is None:
            continue
        w = w_ref if sp["refuse"] else 1.0
        logits = net.scores(sp["F"])
        logp = F.log_softmax(logits, 0)
        loss = -(sp["y"] * logp).sum() * float(w)
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
                   device, exam_rng)
    rep.update({
        "seed": args.seed, "steps": args.steps, "teacher_live": teacher_live,
        "w_refuse": w_ref, "n_train_live": n_live, "n_train_refuse": n_ref,
        "params": int(sum(p.numel() for p in net.parameters())),
        "n_train": len(train_slots), "n_exam": len(exam_slots),
        "joint": args.joint, "min_keys": args.min_keys,
        "loss": "ce_417h_w_refuse",
        "features": "overlap,bag_size,n_slots,refuse_flag",
        "note": "420; 419 not standing",
    })
    print(json.dumps({k: rep[k] for k in (
        "n", "mind_live", "random_live", "mind_live_minus_random_live",
        "mind_pin", "always_refuse", "w_refuse",
        "refuse_df1", "refuse_df2",
        "gate_pins", "gate_mind_ge_always_refuse", "standing")}, indent=1))
    print("GATE pins:", rep["gate_pins"],
          "  mind>=always_refuse:", rep["gate_mind_ge_always_refuse"],
          "  standing:", rep["standing"])
    if not rep["gate_pins"]:
        print("READ: Phi does not rank joint pins — stop CE on this y")
    elif not rep["gate_mind_ge_always_refuse"]:
        print("READ: pins move, silencer still wins")
    else:
        print("READ: both gates — freeze 420, not 419; then raw tape")

    if args.save_mind:
        Path(args.save_mind).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "dim": args.dim, "seed": args.seed,
                    "note": "420 balce structural feats; 417h y"}, args.save_mind)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
