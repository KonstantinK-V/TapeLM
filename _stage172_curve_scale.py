"""
Stage 172 — Curve dynamics SCALE (contract).

After 171 HARDEN_YES:
  - more char stream
  - stronger dynamics (deeper + ctx attention)
  - longer horizons (incl. k=16)
  - weak decoder probe = readout only (stop-grad on z; NEVER teaches dynamics)

Pen stays FROZEN from 170. No text CE into pen/dyn.

  python _stage172_curve_scale.py
  python _stage172_curve_scale.py --steps 80000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _stage170_curve_dynamics as s170

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CKPT = ROOT / "checkpoints"
PLAN = RES / "plan_curve_dynamics.md"
CONTRACT = RES / "stage170_contract.json"
LOG = RES / "_stage172_log.txt"
DECISION = RES / "stage172_decision.json"
MINI = RES / "stage172_mini.md"
CKPT_PEN = CKPT / "stage170_curve.pt"
CKPT_171 = CKPT / "stage171_curve.pt"
CKPT_OUT = CKPT / "stage172_curve.pt"

SEED = 172
D_MODEL = 128  # stronger than 96
N_PEN_LAYERS = 1  # pen arch must match 170 load → keep 96-d pen, project up
PEN_D = s170.D_MODEL  # 96
CTX = 128
SEQ = 192
MICRO = 16
LR = 2e-4
LR_PROBE = 1e-3
EVAL_EVERY = 4000
DEFAULT_STEPS = 80_000
K_STEPS = (1, 2, 4, 8, 16)
CONTRAST_TEMP = 0.07
CORPUS_CHARS = 80_000_000  # scale vs 20M smoke


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


class CtxAttention(nn.Module):
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # causal mask
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.n1(x + h)
        return self.n2(x + self.ff(x))


class StrongDynamics(nn.Module):
    def __init__(self, d_in: int = PEN_D, d: int = D_MODEL, horizons: tuple[int, ...] = K_STEPS):
        super().__init__()
        self.horizons = horizons
        self.in_proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, CTX, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([CtxAttention(d, n_heads=4) for _ in range(2)])
        self.pool = nn.Sequential(nn.Linear(d * 2, d * 2), nn.GELU(), nn.Linear(d * 2, d * 2))
        self.delta_heads = nn.ModuleDict({str(k): nn.Linear(d * 2, d_in) for k in horizons})
        self.z_heads = nn.ModuleDict({str(k): nn.Linear(d * 2, d_in) for k in horizons})
        self.arc_proj = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d_in))

    def forward(self, z_ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        # z_ctx: [B, CTX, pen_d]
        x = self.in_proj(z_ctx) + self.pos[:, : z_ctx.size(1)]
        for blk in self.blocks:
            x = blk(x)
        last = x[:, -1]
        mean = x.mean(dim=1)
        h = self.pool(torch.cat([last, mean], dim=-1))
        out = {"arc": F.normalize(self.arc_proj(h), dim=-1)}
        for k in self.horizons:
            out[f"delta_{k}"] = self.delta_heads[str(k)](h)
            out[f"z_{k}"] = self.z_heads[str(k)](h)
        return out


class WeakCharProbe(nn.Module):
    """Read-out only: z → next char. Trained with stop-grad on z."""

    def __init__(self, d_in: int, n_char: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, n_char))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.ff(z)


class ScaleModel(nn.Module):
    def __init__(self, n_char: int):
        super().__init__()
        self.pen = s170.CurvePen(n_char, d=PEN_D, n_layers=N_PEN_LAYERS)
        self.dyn = StrongDynamics(d_in=PEN_D, d=D_MODEL, horizons=K_STEPS)
        self.probe = WeakCharProbe(PEN_D, n_char)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pen(x)


def freeze_pen(model: ScaleModel) -> None:
    for p in model.pen.parameters():
        p.requires_grad_(False)
    model.pen.eval()


def load_pen(model: ScaleModel, path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model"]
    pen_sd = {k[len("pen.") :]: v for k, v in sd.items() if k.startswith("pen.")}
    model.pen.load_state_dict(pen_sd, strict=True)
    return {"step": blob.get("step"), "stoi": blob.get("stoi"), "itos": blob.get("itos")}


def ctx_windows(z: torch.Tensor, t0: int, n_pred: int, ctx: int = CTX) -> torch.Tensor:
    ends = torch.arange(t0, t0 + n_pred, device=z.device)
    idx = ends.unsqueeze(1) - torch.arange(ctx - 1, -1, -1, device=z.device).unsqueeze(0)
    idx = idx.clamp(min=0)
    return z[:, idx]


def scale_loss(model: ScaleModel, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with torch.no_grad():
        z = model.encode(x)
    B, T, d = z.shape
    k_max = max(K_STEPS)
    t0 = CTX
    last_end = T - 1 - k_max
    if last_end < t0:
        raise RuntimeError("seq too short")
    n_pred = last_end - t0 + 1
    z_ctx = ctx_windows(z, t0, n_pred)
    pred = model.dyn(z_ctx.reshape(B * n_pred, CTX, d))

    loss_dyn = 0.0
    stats: dict = {}
    for k in K_STEPS:
        z_t = z[:, t0 : t0 + n_pred]
        z_tk = z[:, t0 + k : t0 + n_pred + k]
        delta_k = z_tk - z_t
        z_hat = pred[f"z_{k}"].view(B, n_pred, d)
        d_hat = pred[f"delta_{k}"].view(B, n_pred, d)
        lz = 1.0 - F.cosine_similarity(z_hat, z_tk, dim=-1).mean()
        ld = 1.0 - F.cosine_similarity(d_hat, delta_k, dim=-1).mean()
        lm = F.mse_loss(d_hat, delta_k)
        w = 1.0 / math.sqrt(k)
        loss_dyn = loss_dyn + w * (lz + ld + 0.1 * lm)
        stats[f"cos_d_k{k}"] = float(F.cosine_similarity(d_hat, delta_k, dim=-1).mean().detach())

    L = min(8, k_max)
    fut_means = [z[:, t0 + i + 1 : t0 + i + 1 + L].mean(dim=1) for i in range(n_pred)]
    fut = F.normalize(torch.stack(fut_means, dim=1), dim=-1)
    arc = pred["arc"].view(B, n_pred, -1)
    # arc dim may be pen_d
    if arc.size(-1) != fut.size(-1):
        # project fut if needed — both should be pen_d
        pass
    q = F.normalize(arc, dim=-1).reshape(B * n_pred, -1)
    pos = fut.reshape(B * n_pred, -1)
    # match dims
    if q.size(-1) != pos.size(-1):
        # pad/truncate safer: use min
        m = min(q.size(-1), pos.size(-1))
        q, pos = q[:, :m], pos[:, :m]
    logits = (q @ pos.t()) / CONTRAST_TEMP
    labels = torch.arange(q.size(0), device=q.device)
    loss_c = F.cross_entropy(logits, labels)
    loss_dyn = loss_dyn + 0.5 * loss_c

    # Weak probe: stop-grad z → next char (does not teach pen/dyn)
    z_sg = z.detach()
    logits_ch = model.probe(z_sg[:, :-1])
    loss_probe = F.cross_entropy(logits_ch.reshape(-1, logits_ch.size(-1)), x[:, 1:].reshape(-1))
    with torch.no_grad():
        pred_ch = logits_ch.argmax(-1)
        probe_acc = float((pred_ch == x[:, 1:]).float().mean())

    energy = (z[:, 1:] - z[:, :-1]).pow(2).mean()
    stats.update(
        {
            "loss_dyn": float(loss_dyn.detach()),
            "loss_contrast": float(loss_c.detach()),
            "loss_probe": float(loss_probe.detach()),
            "probe_acc": probe_acc,
            "energy": float(energy.detach()),
        }
    )
    return loss_dyn, loss_probe, stats


@torch.no_grad()
def eval_hold(model: ScaleModel, ids: np.ndarray, device, n_windows: int = 64) -> dict:
    model.pen.eval()
    model.dyn.eval()
    model.probe.eval()
    rng = random.Random(SEED + 3)
    hold0 = int(0.9 * len(ids))

    mean_d = {k: [] for k in K_STEPS}
    for _ in range(32):
        s = rng.randint(0, max(1, hold0 - SEQ - 2))
        x = torch.tensor(ids[s : s + SEQ][None].astype(np.int64), device=device)
        z = model.encode(x)
        for k in K_STEPS:
            mean_d[k].append((z[:, k:] - z[:, :-k]).mean(dim=(0, 1)))
    mean_delta = {k: torch.stack(v, 0).mean(0) for k, v in mean_d.items()}

    acc = {k: {"cos": [], "base_mean": [], "base_copy": []} for k in K_STEPS}
    contrast_ok, probe_accs = [], []

    for _ in range(n_windows):
        s = hold0 + rng.randint(0, max(1, len(ids) - hold0 - SEQ - 2))
        x = torch.tensor(ids[s : s + SEQ][None].astype(np.int64), device=device)
        z = model.encode(x)
        T = z.size(1)
        t = max(CTX, T - 1 - max(K_STEPS) - 1)
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
            d_copy = z[:, t] - z[:, max(0, t - k)]
            acc[k]["cos"].append(float(F.cosine_similarity(d_hat, delta, dim=-1).mean()))
            acc[k]["base_mean"].append(
                float(F.cosine_similarity(mean_delta[k].unsqueeze(0), delta, dim=-1).mean())
            )
            acc[k]["base_copy"].append(float(F.cosine_similarity(d_copy, delta, dim=-1).mean()))

        L = 8
        true_f = F.normalize(z[:, t + 1 : t + 1 + L].mean(dim=1), dim=-1)
        wrong_f = F.normalize(z[:, CTX : CTX + L].mean(dim=1), dim=-1)
        arc = F.normalize(pred["arc"][:, : true_f.size(-1)], dim=-1)
        if arc.size(-1) != true_f.size(-1):
            m = min(arc.size(-1), true_f.size(-1))
            arc, true_f, wrong_f = arc[:, :m], true_f[:, :m], wrong_f[:, :m]
        contrast_ok.append(
            float((F.cosine_similarity(arc, true_f) > F.cosine_similarity(arc, wrong_f)).float().mean())
        )

        logits = model.probe(z[:, :-1])
        probe_accs.append(float((logits.argmax(-1) == x[:, 1:]).float().mean()))

    def avg(xs):
        return sum(xs) / max(len(xs), 1)

    out: dict = {"contrast_pref": avg(contrast_ok), "probe_acc": avg(probe_accs)}
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
    out["min_lift_mean"] = min(out[f"k{k}"]["lift_mean"] for k in K_STEPS)
    out["min_lift_copy"] = min(out[f"k{k}"]["lift_copy"] for k in K_STEPS)
    # near vs far
    out["lift_mean_k1"] = out["k1"]["lift_mean"]
    out["lift_mean_k16"] = out["k16"]["lift_mean"]
    model.dyn.train()
    model.probe.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage172 start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan={PLAN} contract={CONTRACT}")
    log("SCALE: frozen pen + stronger dyn + more data + weak probe readout; probe CE does NOT teach dyn/pen")

    if not (RES / "stage171_decision.json").exists():
        log("FATAL: need stage171_decision.json")
        return 1
    v171 = json.loads((RES / "stage171_decision.json").read_text(encoding="utf-8")).get("verdict")
    if v171 != "CURVE_DYN_HARDEN_YES":
        log(f"WARN: 171 verdict was {v171}, continuing anyway per user request")

    if not CKPT_PEN.exists():
        log(f"FATAL missing pen {CKPT_PEN}")
        return 1

    text = s170.load_corpus(max_chars=CORPUS_CHARS)
    meta = torch.load(CKPT_PEN, map_location="cpu", weights_only=False)
    stoi, itos = meta.get("stoi"), meta.get("itos")
    if not stoi or not itos:
        stoi, itos = s170.build_charset(text)
    ids = np.fromiter((stoi.get(c, 0) for c in text), dtype=np.int32, count=len(text))
    log(f"corpus chars={len(ids)} vocab={len(itos)} d_dyn={D_MODEL} ctx={CTX} seq={SEQ} K={list(K_STEPS)}")

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    model = ScaleModel(len(itos)).to(device)
    pen_meta = load_pen(model, CKPT_PEN)
    freeze_pen(model)
    log(f"pen loaded from 170 step={pen_meta.get('step')}; FROZEN")

    opt_dyn = torch.optim.AdamW(model.dyn.parameters(), lr=LR, weight_decay=1e-4)
    opt_probe = torch.optim.AdamW(model.probe.parameters(), lr=LR_PROBE, weight_decay=0.0)
    rng = random.Random(SEED)
    t0 = time.time()
    running = None
    curve = []

    ev0 = eval_hold(model, ids, device)
    log(
        f"  step 0: min_lift_mean={ev0['min_lift_mean']:+.3f} min_lift_copy={ev0['min_lift_copy']:+.3f} "
        f"contrast={ev0['contrast_pref']:.3f} probe={100*ev0['probe_acc']:.1f}% "
        f"k1={ev0['k1']['cos_delta']:.3f} k16={ev0['k16']['cos_delta']:.3f}"
    )
    curve.append({"step": 0, **ev0})

    model.dyn.train()
    model.probe.train()
    for step in range(1, args.steps + 1):
        x = s170.sample_char_batch(ids, MICRO, SEQ, rng, device)
        loss_dyn, loss_probe, st = scale_loss(model, x)
        opt_dyn.zero_grad(set_to_none=True)
        loss_dyn.backward()
        nn.utils.clip_grad_norm_(model.dyn.parameters(), 1.0)
        opt_dyn.step()
        opt_probe.zero_grad(set_to_none=True)
        loss_probe.backward()
        opt_probe.step()

        running = st["loss_dyn"] if running is None else 0.95 * running + 0.05 * st["loss_dyn"]

        if step % EVAL_EVERY == 0 or step == args.steps:
            ev = eval_hold(model, ids, device)
            row = {"step": step, **ev, "loss_ema": running, "energy": st.get("energy"), "probe_train": st.get("probe_acc")}
            curve.append(row)
            log(
                f"  step {step}: loss_dyn~{running:.3f} min_lift_mean={ev['min_lift_mean']:+.3f} "
                f"min_lift_copy={ev['min_lift_copy']:+.3f} contrast={ev['contrast_pref']:.3f} "
                f"probe={100*ev['probe_acc']:.1f}% k1={ev['k1']['cos_delta']:.3f} "
                f"k8={ev['k8']['cos_delta']:.3f} k16={ev['k16']['cos_delta']:.3f} energy={st.get('energy',0):.3f}"
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
                    "d_model": D_MODEL,
                    "from_pen": str(CKPT_PEN),
                },
                CKPT_OUT,
            )

    wall = (time.time() - t0) / 3600
    final = curve[-1]
    beat_mean = final["min_lift_mean"] > 0.02
    beat_copy = final["min_lift_copy"] > 0.02
    contrast_ok = final["contrast_pref"] > 0.55
    far_ok = final["k16"]["lift_mean"] > 0.0 and final["k8"]["lift_mean"] > 0.02
    # probe is diagnostic only — majority char baseline ~ often high for spaces; just log
    probe_note = final["probe_acc"]

    if beat_mean and beat_copy and contrast_ok and far_ok:
        verdict = "CURVE_DYN_SCALE_YES"
    elif beat_mean and contrast_ok:
        verdict = "CURVE_DYN_SCALE_MIXED"
    else:
        verdict = "CURVE_DYN_SCALE_NULL"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "curve_dynamics_scale_172",
        "plan": str(PLAN),
        "contract": str(CONTRACT),
        "verdict": verdict,
        "wall_hours": wall,
        "steps": args.steps,
        "pen_frozen": True,
        "horizons": list(K_STEPS),
        "arch": {"pen_d": PEN_D, "dyn_d": D_MODEL, "ctx": CTX, "seq": SEQ, "attn_blocks": 2},
        "corpus_chars": int(len(ids)),
        "final": final,
        "curve": curve,
        "gates": {
            "beat_mean": beat_mean,
            "beat_copy": beat_copy,
            "contrast_ok": contrast_ok,
            "far_ok": far_ok,
        },
        "probe_acc_hold": probe_note,
        "probe_role": "readout_only_stopgrad_z_never_teaches_dyn_or_pen",
        "note": "Scale+stronger dyn on frozen 170 pen. Weak char probe is diagnostic only.",
        "next": "If YES: longer soak / domain transfer. Decoder stays weak. Do not revive 169 CE.",
    }
    write_json(DECISION, out)
    bullets = [
        f"`{verdict}` wall={wall:.2f}h steps={args.steps}",
        f"corpus={len(ids)} pen=FROZEN dyn_d={D_MODEL} K={list(K_STEPS)}",
        f"min_lift_mean={final['min_lift_mean']:+.3f} min_lift_copy={final['min_lift_copy']:+.3f} contrast={final['contrast_pref']:.3f}",
        f"k1={final['k1']['cos_delta']:.3f} k8={final['k8']['cos_delta']:.3f} k16={final['k16']['cos_delta']:.3f}",
        f"weak probe hold acc={100*probe_note:.1f}% (readout only, not gate)",
        f"gates={out['gates']}",
    ]
    MINI.write_text(
        "\n".join(
            ["# Stage172 — curve scale", "", f"**Verdict:** `{verdict}`", ""]
            + [f"- {b}" for b in bullets]
            + [""]
        ),
        encoding="utf-8",
    )
    log(f"[172] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
