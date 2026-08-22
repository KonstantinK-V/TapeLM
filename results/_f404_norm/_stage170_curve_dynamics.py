"""
Stage 170 — Curve dynamics smoke.

Contract: text only draws a latent curve; train on curve changes (no char/word CE).
Plan: results/plan_curve_dynamics.md

  python _stage170_curve_dynamics.py
  python _stage170_curve_dynamics.py --steps 30000
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
ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
DATA = ROOT / 'data'
CKPT = ROOT / 'checkpoints'
WIKI = DATA / '_wikitext103_train.txt'
PLAN = RES / 'plan_curve_dynamics.md'
LOG = RES / '_stage170_log.txt'
DECISION = RES / 'stage170_decision.json'
MINI = RES / 'stage170_mini.md'
CKPT_PATH = CKPT / 'stage170_curve.pt'
SEED = 170
D_MODEL = 96
N_LAYERS = 1
CTX = 96
SEQ = 128
MICRO = 32
LR = 0.0003
EVAL_EVERY = 2000
DEFAULT_STEPS = 30000

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def build_charset(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = chars
    return (stoi, itos)

def load_corpus(max_chars: int=20000000) -> str:
    if not WIKI.exists():
        raise FileNotFoundError(f'missing {WIKI} — need local wiki text for pen')
    raw = WIKI.read_text(encoding='utf-8', errors='ignore')
    if len(raw) > max_chars:
        raw = raw[:max_chars]
    return raw

class CurvePen(nn.Module):
    """Char → latent curve z_t (the 'pen'). No char CE anywhere."""

    def __init__(self, n_char: int, d: int=D_MODEL, n_layers: int=N_LAYERS):
        super().__init__()
        self.emb = nn.Embedding(n_char, d)
        self.rnn = nn.GRU(d, d, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        z, _ = self.rnn(h)
        return self.norm(z)

class DynamicsHead(nn.Module):
    """From past curve window, predict next latent and next delta."""

    def __init__(self, d: int=D_MODEL, ctx: int=CTX):
        super().__init__()
        self.ctx = ctx
        self.f = nn.Sequential(nn.Linear(d * 2, d * 2), nn.GELU(), nn.Linear(d * 2, d * 2))

    def forward(self, z_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last = z_ctx[:, -1]
        mean = z_ctx.mean(dim=1)
        h = self.f(torch.cat([last, mean], dim=-1))
        d = last.size(-1)
        z_hat = h[:, :d]
        delta_hat = h[:, d:]
        return (z_hat, delta_hat)

class CurveModel(nn.Module):

    def __init__(self, n_char: int):
        super().__init__()
        self.pen = CurvePen(n_char)
        self.dyn = DynamicsHead()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pen(x)

    def predict_from_prefix(self, z: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = max(0, t + 1 - CTX)
        z_ctx = z[:, start:t + 1]
        if z_ctx.size(1) < CTX:
            pad = z_ctx[:, :1].expand(-1, CTX - z_ctx.size(1), -1)
            z_ctx = torch.cat([pad, z_ctx], dim=1)
        return self.dyn(z_ctx)

def sample_char_batch(ids: np.ndarray, batch: int, seq: int, rng: random.Random, device) -> torch.Tensor:
    n = len(ids)
    max_start = n - seq - 1
    xs = []
    for _ in range(batch):
        s = rng.randint(0, max_start)
        xs.append(ids[s:s + seq].astype(np.int64))
    return torch.tensor(np.stack(xs, 0), dtype=torch.long, device=device)

def dynamics_loss(model: CurveModel, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Teacher path: pen encodes full window (detach optional on early pen — here joint).
    Loss only on latent next-z / delta — never char logits.
    """
    z = model.encode(x)
    B, T, d = z.shape
    t0 = CTX
    if T <= t0 + 2:
        raise RuntimeError('seq too short for ctx')
    n_pred = T - 1 - t0
    ends = torch.arange(t0, T - 1, device=z.device)
    idx = ends.unsqueeze(1) - torch.arange(CTX - 1, -1, -1, device=z.device).unsqueeze(0)
    idx = idx.clamp(min=0)
    z_ctx = z[:, idx]
    z_hat, delta_hat = model.dyn(z_ctx.reshape(B * n_pred, CTX, d))
    z_hat = z_hat.view(B, n_pred, d)
    delta_hat = delta_hat.view(B, n_pred, d)
    z_next = z[:, t0 + 1:T]
    z_cur = z[:, t0:T - 1]
    delta = z_next - z_cur
    loss_z = 1.0 - F.cosine_similarity(z_hat, z_next.detach(), dim=-1).mean()
    loss_d = 1.0 - F.cosine_similarity(delta_hat, delta.detach(), dim=-1).mean()
    loss_mse = F.mse_loss(delta_hat, delta.detach())
    energy = delta.pow(2).mean()
    loss = loss_z + loss_d + 0.1 * loss_mse + 0.01 * F.relu(0.05 - energy)
    stats = {'loss': float(loss.detach()), 'loss_z': float(loss_z.detach()), 'loss_d': float(loss_d.detach()), 'cos_z': float(F.cosine_similarity(z_hat, z_next.detach(), dim=-1).mean().detach()), 'cos_d': float(F.cosine_similarity(delta_hat, delta.detach(), dim=-1).mean().detach()), 'energy': float(energy.detach())}
    return (loss, stats)

@torch.no_grad()
def eval_hold(model: CurveModel, ids: np.ndarray, device, n_windows: int=64) -> dict:
    model.eval()
    rng = random.Random(SEED + 7)
    hold0 = int(0.9 * len(ids))
    cos_z = []
    cos_d = []
    cos_d_zero = []
    cos_d_mean = []
    cos_d_copy = []
    mean_delta_acc = []
    mean_deltas = []
    for _ in range(32):
        s = rng.randint(0, max(1, hold0 - SEQ - 2))
        x = torch.tensor(ids[s:s + SEQ][None].astype(np.int64), device=device)
        z = model.encode(x)
        mean_deltas.append((z[:, 1:] - z[:, :-1]).mean(dim=(0, 1)))
    mean_delta = torch.stack(mean_deltas, 0).mean(0)
    for _ in range(n_windows):
        s = hold0 + rng.randint(0, max(1, len(ids) - hold0 - SEQ - 2))
        x = torch.tensor(ids[s:s + SEQ][None].astype(np.int64), device=device)
        z = model.encode(x)
        B, T, d = z.shape
        t = T - 2
        z_hat, delta_hat = model.predict_from_prefix(z, t)
        z_next = z[:, t + 1]
        z_cur = z[:, t]
        delta = z_next - z_cur
        delta_prev = z[:, t] - z[:, t - 1]
        cos_z.append(float(F.cosine_similarity(z_hat, z_next, dim=-1).mean()))
        cos_d.append(float(F.cosine_similarity(delta_hat, delta, dim=-1).mean()))
        cos_d_zero.append(float(F.cosine_similarity(torch.zeros_like(delta), delta, dim=-1).mean()))
        cos_d_mean.append(float(F.cosine_similarity(mean_delta.unsqueeze(0), delta, dim=-1).mean()))
        cos_d_copy.append(float(F.cosine_similarity(delta_prev, delta, dim=-1).mean()))

    def avg(xs):
        return sum(xs) / max(len(xs), 1)
    out = {'cos_z': avg(cos_z), 'cos_delta': avg(cos_d), 'base_zero_delta': avg(cos_d_zero), 'base_mean_delta': avg(cos_d_mean), 'base_copy_delta': avg(cos_d_copy)}
    out['lift_vs_mean_delta'] = out['cos_delta'] - out['base_mean_delta']
    out['lift_vs_copy_delta'] = out['cos_delta'] - out['base_copy_delta']
    model.train()
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage170 start {datetime.now(timezone.utc).isoformat()}')
    log(f'plan={PLAN}')
    log('contract: text draws curve; loss=latent dynamics only; NO char/word CE')
    text = load_corpus()
    stoi, itos = build_charset(text)
    ids = np.fromiter((stoi[c] for c in text), dtype=np.int32, count=len(text))
    log(f'corpus chars={len(ids)} vocab={len(itos)} file={WIKI.name}')
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = CurveModel(len(itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    rng = random.Random(SEED)
    t0 = time.time()
    running = None
    curve = []
    ev0 = eval_hold(model, ids, device)
    log(f"  step 0: cos_z={ev0['cos_z']:.3f} cos_d={ev0['cos_delta']:.3f} lift_mean={ev0['lift_vs_mean_delta']:+.3f} lift_copy={ev0['lift_vs_copy_delta']:+.3f}")
    curve.append({'step': 0, **ev0})
    model.train()
    for step in range(1, args.steps + 1):
        x = sample_char_batch(ids, MICRO, SEQ, rng, device)
        loss, st = dynamics_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = st['loss'] if running is None else 0.95 * running + 0.05 * st['loss']
        if step % EVAL_EVERY == 0 or step == args.steps:
            ev = eval_hold(model, ids, device)
            curve.append({'step': step, **ev, 'loss_ema': running})
            log(f"  step {step}: loss~{running:.3f} cos_z={ev['cos_z']:.3f} cos_d={ev['cos_delta']:.3f} lift_mean={ev['lift_vs_mean_delta']:+.3f} lift_copy={ev['lift_vs_copy_delta']:+.3f} base_mean={ev['base_mean_delta']:.3f} base_copy={ev['base_copy_delta']:.3f}")
            torch.save({'model': model.state_dict(), 'stoi': stoi, 'itos': itos, 'step': step, 'curve': curve}, CKPT_PATH)
    wall = (time.time() - t0) / 3600
    final = curve[-1]
    beat_mean = final['lift_vs_mean_delta'] > 0.02
    beat_copy = final['lift_vs_copy_delta'] > 0.02
    if beat_mean and beat_copy:
        verdict = 'CURVE_DYN_SMOKE_YES'
    elif beat_mean or beat_copy:
        verdict = 'CURVE_DYN_SMOKE_MIXED'
    else:
        verdict = 'CURVE_DYN_SMOKE_NULL'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'curve_dynamics_smoke', 'plan': str(PLAN), 'verdict': verdict, 'wall_hours': wall, 'steps': args.steps, 'arch': {'d': D_MODEL, 'gru_layers': N_LAYERS, 'ctx': CTX, 'seq': SEQ, 'micro': MICRO}, 'corpus_chars': int(len(ids)), 'char_vocab': len(itos), 'final': final, 'curve': curve, 'note': 'No char/word CE. Gate = latent delta prediction vs mean/copy baselines.', 'next': 'If YES/MIXED: stronger pen / longer soak. If NULL: redesign dynamics loss, not revive 169 CE.', 'frozen_prior': 'stage169 word-CE FROZEN'}
    write_json(DECISION, out)
    bullets = [f'verdict `{verdict}` wall={wall:.2f}h steps={args.steps}', f"final cos_delta={final['cos_delta']:.3f} lift_vs_mean={final['lift_vs_mean_delta']:+.3f} lift_vs_copy={final['lift_vs_copy_delta']:+.3f}", f"baselines mean={final['base_mean_delta']:.3f} copy={final['base_copy_delta']:.3f} zero={final['base_zero_delta']:.3f}", 'loss = latent next-z + delta only (no text CE)', '169 frozen; do not resume word-battery path unless reopened']
    MINI.write_text('\n'.join(['# Stage170 — curve dynamics smoke', '', f'**Verdict:** `{verdict}`', ''] + [f'- {b}' for b in bullets] + ['']), encoding='utf-8')
    log(f'[170] {verdict}')
    return 0 if verdict != 'CURVE_DYN_SMOKE_NULL' else 0
if __name__ == '__main__':
    raise SystemExit(main())