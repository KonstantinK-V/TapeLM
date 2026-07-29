"""
Stage 171 — Curve dynamics HARDENING (contract).

Frozen pen + multi-step Δ + contrastive arcs.
Separates "we predict Δ" from "we painted an easy curve."

Plan/contract: results/plan_curve_dynamics.md , results/stage170_contract.json

  python _stage171_curve_harden.py
  python _stage171_curve_harden.py --steps 40000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse pen/data helpers from 170
import _stage170_curve_dynamics as s170

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CKPT = ROOT / "checkpoints"
PLAN = RES / "plan_curve_dynamics.md"
CONTRACT = RES / "stage170_contract.json"
LOG = RES / "_stage171_log.txt"
DECISION = RES / "stage171_decision.json"
MINI = RES / "stage171_mini.md"
CKPT_IN = CKPT / "stage170_curve.pt"
CKPT_OUT = CKPT / "stage171_curve.pt"

SEED = 171
D_MODEL = s170.D_MODEL
CTX = s170.CTX
SEQ = 160  # need room for multi-step
MICRO = 24
LR = 3e-4
EVAL_EVERY = 2000
DEFAULT_STEPS = 40_000
K_STEPS = (1, 2, 4, 8)  # multi-step horizons
CONTRAST_TEMP = 0.07


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


class MultiStepDynamics(nn.Module):
    """From past arc, predict Δ over several horizons + arc embedding for contrastive."""

    def __init__(self, d: int = D_MODEL, horizons: tuple[int, ...] = K_STEPS):
        super().__init__()
        self.horizons = horizons
        self.backbone = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
        )
        self.delta_heads = nn.ModuleDict({str(k): nn.Linear(d * 2, d) for k in horizons})
        self.z_heads = nn.ModuleDict({str(k): nn.Linear(d * 2, d) for k in horizons})
        self.arc_proj = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))

    def encode_ctx(self, z_ctx: torch.Tensor) -> torch.Tensor:
        last = z_ctx[:, -1]
        mean = z_ctx.mean(dim=1)
        return self.backbone(torch.cat([last, mean], dim=-1))

    def forward(self, z_ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode_ctx(z_ctx)
        out = {"arc": F.normalize(self.arc_proj(h), dim=-1)}
        for k in self.horizons:
            out[f"delta_{k}"] = self.delta_heads[str(k)](h)
            out[f"z_{k}"] = self.z_heads[str(k)](h)
        return out


class HardenModel(nn.Module):
    def __init__(self, n_char: int):
        super().__init__()
        self.pen = s170.CurvePen(n_char)
        self.dyn = MultiStepDynamics()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pen(x)


def freeze_pen(model: HardenModel) -> None:
    for p in model.pen.parameters():
        p.requires_grad_(False)
    model.pen.eval()


def load_pen_from_170(model: HardenModel, path: Path, device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"need Stage170 ckpt at {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    # 170 saved full CurveModel: pen.* and dyn.*
    sd = blob["model"]
    pen_sd = {k[len("pen.") :]: v for k, v in sd.items() if k.startswith("pen.")}
    missing, unexpected = model.pen.load_state_dict(pen_sd, strict=True)
    return {
        "step170": blob.get("step"),
        "stoi": blob.get("stoi"),
        "itos": blob.get("itos"),
        "missing": list(missing) if missing else [],
        "unexpected": list(unexpected) if unexpected else [],
    }


def ctx_windows(z: torch.Tensor, t0: int, n_pred: int) -> torch.Tensor:
    """z [B,T,d] → windows ending at t0..t0+n_pred-1 → [B,n_pred,CTX,d]."""
    ends = torch.arange(t0, t0 + n_pred, device=z.device)
    idx = ends.unsqueeze(1) - torch.arange(CTX - 1, -1, -1, device=z.device).unsqueeze(0)
    idx = idx.clamp(min=0)
    return z[:, idx]


def harden_loss(model: HardenModel, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Frozen pen → z; train dyn on multi-step Δ/z + contrastive future arcs."""
    with torch.no_grad():
        z = model.encode(x)  # frozen curve ink
    B, T, d = z.shape
    k_max = max(K_STEPS)
    t0 = CTX
    # last index we can predict k_max ahead: T-1-k_max
    last_end = T - 1 - k_max
    if last_end < t0:
        raise RuntimeError("seq too short for multi-step")
    n_pred = last_end - t0 + 1
    z_ctx = ctx_windows(z, t0, n_pred)  # [B,n,CTX,d]
    pred = model.dyn(z_ctx.reshape(B * n_pred, CTX, d))

    loss = 0.0
    stats: dict = {}
    for k in K_STEPS:
        # true z_{t+k} and Δ^{(k)} = z_{t+k}-z_t for ends t = t0..last_end
        z_t = z[:, t0 : t0 + n_pred]
        z_tk = z[:, t0 + k : t0 + n_pred + k]
        delta_k = z_tk - z_t
        z_hat = pred[f"z_{k}"].view(B, n_pred, d)
        d_hat = pred[f"delta_{k}"].view(B, n_pred, d)
        lz = 1.0 - F.cosine_similarity(z_hat, z_tk, dim=-1).mean()
        ld = 1.0 - F.cosine_similarity(d_hat, delta_k, dim=-1).mean()
        lm = F.mse_loss(d_hat, delta_k)
        w = 1.0 / k  # slightly emphasize nearer horizons
        loss = loss + w * (lz + ld + 0.1 * lm)
        stats[f"cos_d_k{k}"] = float(F.cosine_similarity(d_hat, delta_k, dim=-1).mean().detach())
        stats[f"cos_z_k{k}"] = float(F.cosine_similarity(z_hat, z_tk, dim=-1).mean().detach())

    # Contrastive: arc emb from ctx should match true future arc mean, not others in batch
    L = min(8, k_max)
    fut_means = []
    for i in range(n_pred):
        s = t0 + i + 1
        fut_means.append(z[:, s : s + L].mean(dim=1))
    fut = torch.stack(fut_means, dim=1)  # [B,n,d]
    fut = F.normalize(fut, dim=-1)
    arc = pred["arc"].view(B, n_pred, d)
    # InfoNCE within batch*n flattened against wrong futures
    q = arc.reshape(B * n_pred, d)
    pos = fut.reshape(B * n_pred, d)
    logits = q @ pos.t() / CONTRAST_TEMP
    labels = torch.arange(q.size(0), device=q.device)
    loss_c = F.cross_entropy(logits, labels)
    loss = loss + 0.5 * loss_c
    stats["loss_contrast"] = float(loss_c.detach())
    stats["loss"] = float(loss.detach())
    # path energy of frozen curve (monitor collapse of ink)
    energy = (z[:, 1:] - z[:, :-1]).pow(2).mean()
    stats["energy"] = float(energy.detach())
    return loss, stats


@torch.no_grad()
def eval_hold(model: HardenModel, ids: np.ndarray, device, n_windows: int = 80) -> dict:
    model.pen.eval()
    model.dyn.eval()
    rng = random.Random(SEED + 11)
    hold0 = int(0.9 * len(ids))

    # mean multi-step deltas on train-ish windows
    mean_d = {k: [] for k in K_STEPS}
    for _ in range(40):
        s = rng.randint(0, max(1, hold0 - SEQ - 2))
        x = torch.tensor(ids[s : s + SEQ][None].astype(np.int64), device=device)
        z = model.encode(x)
        for k in K_STEPS:
            mean_d[k].append((z[:, k:] - z[:, :-k]).mean(dim=(0, 1)))
    mean_delta = {k: torch.stack(v, 0).mean(0) for k, v in mean_d.items()}

    acc = {k: {"cos": [], "base_mean": [], "base_copy": [], "base_zero": []} for k in K_STEPS}
    contrast_ok = []

    for _ in range(n_windows):
        s = hold0 + rng.randint(0, max(1, len(ids) - hold0 - SEQ - 2))
        x = torch.tensor(ids[s : s + SEQ][None].astype(np.int64), device=device)
        z = model.encode(x)
        T = z.size(1)
        t = T - 1 - max(K_STEPS) - 1
        t = max(t, CTX)
        z_ctx = z[:, t + 1 - CTX : t + 1]
        if z_ctx.size(1) < CTX:
            pad = z_ctx[:, :1].expand(1, CTX - z_ctx.size(1), -1)
            z_ctx = torch.cat([pad, z_ctx], dim=1)
        pred = model.dyn(z_ctx)
        for k in K_STEPS:
            if t + k >= T:
                continue
            delta = z[:, t + k] - z[:, t]
            d_hat = pred[f"delta_{k}"]
            d_copy = z[:, t] - z[:, t - min(k, t)]
            acc[k]["cos"].append(float(F.cosine_similarity(d_hat, delta, dim=-1).mean()))
            acc[k]["base_mean"].append(
                float(F.cosine_similarity(mean_delta[k].unsqueeze(0), delta, dim=-1).mean())
            )
            acc[k]["base_copy"].append(float(F.cosine_similarity(d_copy, delta, dim=-1).mean()))
            acc[k]["base_zero"].append(0.0)

        # tiny contrastive probe: arc closer to true future mean than to random other window
        L = 8
        true_f = F.normalize(z[:, t + 1 : t + 1 + L].mean(dim=1), dim=-1)
        # wrong: from start of sequence
        wrong_f = F.normalize(z[:, CTX : CTX + L].mean(dim=1), dim=-1)
        arc = pred["arc"]
        contrast_ok.append(
            float((F.cosine_similarity(arc, true_f) > F.cosine_similarity(arc, wrong_f)).float().mean())
        )

    def avg(xs):
        return sum(xs) / max(len(xs), 1)

    out: dict = {"contrast_pref": avg(contrast_ok)}
    for k in K_STEPS:
        cos = avg(acc[k]["cos"])
        bm = avg(acc[k]["base_mean"])
        bc = avg(acc[k]["base_copy"])
        out[f"k{k}"] = {
            "cos_delta": cos,
            "base_mean": bm,
            "base_copy": bc,
            "lift_mean": cos - bm,
            "lift_copy": cos - bc,
        }
    # aggregate gate: all horizons beat mean & copy by margin; contrast > 0.55
    out["min_lift_mean"] = min(out[f"k{k}"]["lift_mean"] for k in K_STEPS)
    out["min_lift_copy"] = min(out[f"k{k}"]["lift_copy"] for k in K_STEPS)
    model.dyn.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage171 start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan={PLAN}")
    log(f"contract={CONTRACT}")
    log("HARDEN: frozen pen + multi-step Δ + contrastive arcs; NO text CE")

    if not CKPT_IN.exists():
        log(f"FATAL missing {CKPT_IN}")
        return 1

    text = s170.load_corpus()
    # Prefer charset from 170 ckpt so pen rows align
    blob170 = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    stoi = blob170.get("stoi")
    itos = blob170.get("itos")
    if not stoi or not itos:
        stoi, itos = s170.build_charset(text)
    # map corpus with 170 stoi; unknown → 0
    unk = 0
    ids = np.fromiter((stoi.get(c, unk) for c in text), dtype=np.int32, count=len(text))
    log(f"corpus chars={len(ids)} vocab={len(itos)} pen_from={CKPT_IN.name}")

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    model = HardenModel(len(itos)).to(device)
    meta = load_pen_from_170(model, CKPT_IN, device)
    freeze_pen(model)
    log(f"loaded pen from 170 step={meta.get('step170')}; pen FROZEN")

    opt = torch.optim.AdamW(model.dyn.parameters(), lr=LR, weight_decay=1e-4)
    rng = random.Random(SEED)
    t0 = time.time()
    running = None
    curve = []

    ev0 = eval_hold(model, ids, device)
    log(
        f"  step 0: min_lift_mean={ev0['min_lift_mean']:+.3f} min_lift_copy={ev0['min_lift_copy']:+.3f} "
        f"contrast_pref={ev0['contrast_pref']:.3f} k1_cos={ev0['k1']['cos_delta']:.3f} k8_cos={ev0['k8']['cos_delta']:.3f}"
    )
    curve.append({"step": 0, **ev0})

    model.dyn.train()
    for step in range(1, args.steps + 1):
        x = s170.sample_char_batch(ids, MICRO, SEQ, rng, device)
        loss, st = harden_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.dyn.parameters(), 1.0)
        opt.step()
        running = st["loss"] if running is None else 0.95 * running + 0.05 * st["loss"]

        if step % EVAL_EVERY == 0 or step == args.steps:
            ev = eval_hold(model, ids, device)
            row = {"step": step, **ev, "loss_ema": running, "energy": st.get("energy")}
            curve.append(row)
            log(
                f"  step {step}: loss~{running:.3f} min_lift_mean={ev['min_lift_mean']:+.3f} "
                f"min_lift_copy={ev['min_lift_copy']:+.3f} contrast={ev['contrast_pref']:.3f} "
                f"k1={ev['k1']['cos_delta']:.3f} k4={ev['k4']['cos_delta']:.3f} k8={ev['k8']['cos_delta']:.3f} "
                f"energy={st.get('energy', 0):.4f}"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "step": step,
                    "curve": curve,
                    "pen_frozen": True,
                    "horizons": list(K_STEPS),
                    "from_170": str(CKPT_IN),
                },
                CKPT_OUT,
            )

    wall = (time.time() - t0) / 3600
    final = curve[-1]
    beat_mean = final["min_lift_mean"] > 0.02
    beat_copy = final["min_lift_copy"] > 0.02
    contrast_ok = final["contrast_pref"] > 0.55
    # require near AND far horizon somewhat alive
    far_ok = final["k8"]["lift_mean"] > 0.0 and final["k4"]["lift_mean"] > 0.02

    if beat_mean and beat_copy and contrast_ok and far_ok:
        verdict = "CURVE_DYN_HARDEN_YES"
    elif beat_mean and (contrast_ok or far_ok):
        verdict = "CURVE_DYN_HARDEN_MIXED"
    else:
        verdict = "CURVE_DYN_HARDEN_NULL"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "curve_dynamics_harden_171",
        "plan": str(PLAN),
        "contract": str(CONTRACT),
        "verdict": verdict,
        "wall_hours": wall,
        "steps": args.steps,
        "pen_frozen": True,
        "horizons": list(K_STEPS),
        "from_170": str(CKPT_IN),
        "final": final,
        "curve": curve,
        "gates": {
            "beat_mean": beat_mean,
            "beat_copy": beat_copy,
            "contrast_ok": contrast_ok,
            "far_ok": far_ok,
        },
        "note": "Frozen pen from 170; dynamics-only train; multi-step + contrastive. No text CE.",
        "next": "If YES: scale data / stronger dyn. If MIXED: tighten. If NULL: redesign — do not add CE.",
    }
    write_json(DECISION, out)
    bullets = [
        f"`{verdict}` wall={wall:.2f}h steps={args.steps} pen=FROZEN from 170",
        f"min_lift_mean={final['min_lift_mean']:+.3f} min_lift_copy={final['min_lift_copy']:+.3f} contrast={final['contrast_pref']:.3f}",
        f"k1 cos={final['k1']['cos_delta']:.3f} lift_m={final['k1']['lift_mean']:+.3f}",
        f"k4 cos={final['k4']['cos_delta']:.3f} lift_m={final['k4']['lift_mean']:+.3f}",
        f"k8 cos={final['k8']['cos_delta']:.3f} lift_m={final['k8']['lift_mean']:+.3f}",
        f"gates={out['gates']}",
    ]
    MINI.write_text(
        "\n".join(
            ["# Stage171 — curve harden (frozen pen)", "", f"**Verdict:** `{verdict}`", ""]
            + [f"- {b}" for b in bullets]
            + [""]
        ),
        encoding="utf-8",
    )
    log(f"[171] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
