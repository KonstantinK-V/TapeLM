"""416: STREAM → PLACE → TAPE WORD. The training contract, not the lab hole-exam.

As agreed:

  raw stream, each token a step
  key = window (address halves / neighbours), NOT the token (390 leak discipline)
  mind chooses a PLACE; the tape says a WORD (majority filler of that place)
  reward: said what is in the stream; if the literal appears once on the tape → refuse
  Phi has no vocabulary and no 8-word offer / GATE-WO

This file trains and examines THAT. It does not call reach offers. Our old seeds/exams are
not the teacher.

  VOID  n_cand mean ~ 0 → nowhere to point; arm is empty.
  GATE  (3 seeds, held-out stream)
        ge2: mind_hit > cosine_place_rival
        one: refuse_rate > refuse_rate on ge2

    python _check416_stream.py
    python _audit416_stream.py --seed 1337 --steps 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import _audit390_address as A

WIKI = Path("data/_wikitext103_train.txt")
OUT = Path("results/_stage416_stream.json")
REFUSE = -1  # action index, not a string in Phi


def hash_fp(parts, d: int) -> torch.Tensor:
    """Bag of strings → unit vector. No learned vocab; pure function of the key."""
    v = torch.zeros(d)
    for i, p in enumerate(parts):
        h = hashlib.sha1(f"{i}:{p}".encode("utf-8")).digest()
        for j in range(d):
            v[j] += ((h[j % 20] / 255.0) * 2.0 - 1.0) * (1.0 if (h[(j + 3) % 20] & 1) == 0 else -1.0)
    n = float(v.norm().clamp(min=1e-8))
    return v / n


def window_parts(T, pid) -> list[str]:
    """The key is the address halves, never the hidden token."""
    _w, left, right = T["addrs"][pid]
    return [f"L:{x}" for x in left] + [f"R:{x}" for x in right]


def place_parts(T, pid, hide_slot=None) -> list[str]:
    """Fillers at the place, with the hidden slot removed if it sits here."""
    out = []
    for s in T["places"][pid]:
        if hide_slot is not None and s == hide_slot:
            continue
        out.append(T["toks"][s])
    return out


def resolve(T, pid, hide_slot=None):
    """Tape speaks: majority filler of the place (hidden slot out)."""
    c = Counter(place_parts(T, pid, hide_slot))
    if not c:
        return None
    return c.most_common(1)[0][0]


def cand_places(T, pid, hide_slot, k: int) -> list[int]:
    """Address neighbours (390): share left or right half. Same-line dropped. Not a word offer."""
    _w, left, right = T["addrs"][pid]
    drop = set(T["on_line"][T["owner"][hide_slot]])
    seen, out = set(), []
    for j in list(T["by_left"].get(left, ())) + list(T["by_right"].get(right, ())):
        if j == pid or j in drop or j in seen:
            continue
        seen.add(j)
        out.append(j)
    # Stable order by overlap of halves then id — not by filler overlap (that is the walk).
    def key(j):
        _wj, lj, rj = T["addrs"][j]
        return (-(lj == left) - (rj == right), j)

    out.sort(key=key)
    return out[:k]


class PlaceMind(nn.Module):
    """Scores (window_key, place) → scalar. Output alphabet = places (+ refuse), not words."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.d = d
        self.qw = nn.Linear(d, d)
        self.pw = nn.Linear(d, d)
        self.sc = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.sc[-1].weight)
        nn.init.zeros_(self.sc[-1].bias)

    def score_places(self, q, P):
        # q: [d], P: [K, d]
        hq = self.qw(q)
        hp = self.pw(P)
        K = P.shape[0]
        qq = hq.unsqueeze(0).expand(K, -1)
        return self.sc(torch.cat([qq, hp], -1)).squeeze(-1)

    def refuse_logit(self, q):
        # Refuse is a place-free action: score from the window alone.
        z = torch.zeros(self.d, device=q.device, dtype=q.dtype)
        return self.sc(torch.cat([self.qw(q), self.pw(z)], -1)).squeeze(-1)


def step_pack(T, slot, k, d, device):
    pid = T["place_of"][slot]
    truth = T["toks"][slot]
    freq = T["freq"][truth]
    cands = cand_places(T, pid, slot, k)
    q = hash_fp(window_parts(T, pid), d).to(device)
    if not cands:
        P = torch.zeros(0, d, device=device)
        said = []
    else:
        P = torch.stack([hash_fp(place_parts(T, j, slot), d) for j in cands]).to(device)
        said = [resolve(T, j, slot) for j in cands]
    R = []
    for w in said:
        R.append(1.0 if w == truth else -1.0)
    # refuse: right iff the literal is unique on the tape
    R.append(1.0 if freq == 1 else -1.0)
    return {
        "q": q, "P": P, "cands": cands, "said": said, "R": torch.tensor(R, device=device),
        "truth": truth, "freq": freq, "pid": pid, "slot": slot,
    }


def cosine_rival(q, P):
    if P.shape[0] == 0:
        return REFUSE
    sims = F.normalize(P, dim=-1) @ F.normalize(q, dim=-1)
    return int(sims.argmax())


def act_from_scores(scores, n_places):
    i = int(scores.argmax())
    return REFUSE if i == n_places else i


def run_exam(net, T, slots, k, d, device):
    net.eval()
    ge2 = {"n": 0, "mind": 0, "riv": 0, "refuse": 0}
    one = {"n": 0, "mind": 0, "riv": 0, "refuse": 0}
    with torch.no_grad():
        for slot in slots:
            sp = step_pack(T, slot, k, d, device)
            bucket = one if sp["freq"] == 1 else ge2
            if sp["freq"] < 1:
                continue
            bucket["n"] += 1
            if sp["P"].shape[0] == 0:
                scores = sp["q"].new_tensor([net.refuse_logit(sp["q"]).item()])
                mind_i = REFUSE
                riv_i = REFUSE
            else:
                ps = net.score_places(sp["q"], sp["P"])
                rf = net.refuse_logit(sp["q"])
                scores = torch.cat([ps, rf.reshape(1)])
                mind_i = act_from_scores(scores, sp["P"].shape[0])
                riv_i = cosine_rival(sp["q"], sp["P"])
            mind_say = None if mind_i == REFUSE else sp["said"][mind_i]
            riv_say = None if riv_i == REFUSE else sp["said"][riv_i]
            if mind_i == REFUSE:
                bucket["refuse"] += 1
            if mind_say == sp["truth"] or (mind_i == REFUSE and sp["freq"] == 1):
                bucket["mind"] += 1
            if riv_say == sp["truth"] or (riv_i == REFUSE and sp["freq"] == 1):
                bucket["riv"] += 1

    def rate(b, key):
        return b[key] / b["n"] if b["n"] else float("nan")

    return {
        "ge2": {"n": ge2["n"], "mind_hit": rate(ge2, "mind"), "rival_hit": rate(ge2, "riv"),
                "refuse": rate(ge2, "refuse")},
        "one": {"n": one["n"], "mind_hit": rate(one, "mind"), "rival_hit": rate(one, "riv"),
                "refuse": rate(one, "refuse")},
        "gate_mind_beats_rival": bool(
            ge2["n"] and rate(ge2, "mind") == rate(ge2, "mind")
            and rate(ge2, "riv") == rate(ge2, "riv")
            and rate(ge2, "mind") > rate(ge2, "riv")),
        "gate_refuse_one_gt_ge2": bool(
            one["n"] and ge2["n"]
            and rate(one, "refuse") > rate(ge2, "refuse")),
        "mean_cands": None,
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
    ap.add_argument("--places-k", type=int, default=16)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exam-slots", type=int, default=800)
    ap.add_argument("--corpus", default=str(WIKI))
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
    slots = [s for ps in T["places"] for s in ps]
    if len(slots) < 50:
        print("tape too small")
        return 1
    rng.shuffle(slots)
    cut = max(1, int(0.8 * len(slots)))
    train_slots, exam_slots = slots[:cut], slots[cut:]
    exam_slots = exam_slots[: args.exam_slots]

    # VOID: can we point anywhere?
    sample = train_slots[:200]
    nc = [len(cand_places(T, T["place_of"][s], s, args.places_k)) for s in sample]
    mean_c = sum(nc) / max(1, len(nc))
    print(f"tape places={len(T['places'])} slots={len(slots)} mean_cands={mean_c:.2f}")
    if mean_c < 0.5:
        print("VOID: almost no address neighbours — nowhere for the mind to point")
        return 1

    net = PlaceMind(args.dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    losses = []
    for step in range(1, args.steps + 1):
        slot = train_slots[rng.randrange(len(train_slots))]
        sp = step_pack(T, slot, args.places_k, args.dim, device)
        if sp["P"].shape[0] == 0:
            scores = net.refuse_logit(sp["q"]).reshape(1)
        else:
            scores = torch.cat([net.score_places(sp["q"], sp["P"]),
                                net.refuse_logit(sp["q"]).reshape(1)])
        # expected reward — one scalar over places(+refuse), never over a word list
        loss = -(F.softmax(scores, 0) * sp["R"]).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % max(1, args.steps // 8) == 0:
            print(f"  step {step}/{args.steps} loss={sum(losses[-200:]) / max(1, len(losses[-200:])):.4f}")

    rep = run_exam(net, T, exam_slots, args.places_k, args.dim, device)
    rep["mean_cands"] = mean_c
    rep["seed"] = args.seed
    rep["steps"] = args.steps
    rep["n_train"] = len(train_slots)
    rep["n_exam"] = len(exam_slots)
    rep["params"] = int(sum(p.numel() for p in net.parameters()))
    print(json.dumps(rep, indent=1))
    print(
        "GATE mind>rival ge2:", rep["gate_mind_beats_rival"],
        "  refuse_one>ge2:", rep["gate_refuse_one_gt_ge2"],
    )
    if args.save_mind:
        Path(args.save_mind).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "dim": args.dim, "seed": args.seed,
                    "note": "416 stream place-mind"}, args.save_mind)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
