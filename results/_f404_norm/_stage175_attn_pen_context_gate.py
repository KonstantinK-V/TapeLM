"""
Stage 175 — Causal Transformer pen for context (gate A/B).

Replace GRU compression with causal self-attn pen so past arc can remain
visible at the endpoint. Short fit → freeze pen → rerun 174-style A (and B).

Contract: still NO char/word CE as teacher. Loss = latent Δ / next-z / contrastive.

  python _stage175_attn_pen_context_gate.py
  python _stage175_attn_pen_context_gate.py --steps 20000
"""
from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import _stage170_curve_dynamics as s170
RES = Path('results')
CKPT = Path('checkpoints')
LOG = RES / '_stage175_log.txt'
DECISION = RES / 'stage175_decision.json'
MINI = RES / 'stage175_mini.md'
CKPT_OUT = CKPT / 'stage175_attn_pen.pt'
PLAN = RES / 'plan_curve_dynamics.md'
SEED = 175
D = 96
N_LAYERS = 2
N_HEADS = 4
CTX = 128
SEQ = 160
MICRO = 16
LR = 0.0003
EVAL_EVERY = 2000
DEFAULT_STEPS = 20000
K_STEPS = (1, 2, 4, 8)
SUFFIX_LEN = 24
PREFIX_LEN = 96

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

class CausalAttnBlock(nn.Module):

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(0.1))
        self.n2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.n1(x + h)
        return self.n2(x + self.ff(x))

class AttnPen(nn.Module):
    """Char stream → curve z_t with causal attention (context-capable ink)."""

    def __init__(self, n_char: int, d: int=D, n_layers: int=N_LAYERS, max_len: int=512):
        super().__init__()
        self.emb = nn.Embedding(n_char, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([CausalAttnBlock(d, N_HEADS) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if T > self.max_len:
            x = x[:, -self.max_len:]
            T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)

class DynHead(nn.Module):

    def __init__(self, d: int=D, horizons: tuple[int, ...]=K_STEPS):
        super().__init__()
        self.horizons = horizons
        self.f = nn.Sequential(nn.Linear(d * 2, d * 2), nn.GELU(), nn.Linear(d * 2, d * 2))
        self.delta_heads = nn.ModuleDict({str(k): nn.Linear(d * 2, d) for k in horizons})
        self.z_heads = nn.ModuleDict({str(k): nn.Linear(d * 2, d) for k in horizons})

    def forward(self, z_ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        last, mean = (z_ctx[:, -1], z_ctx.mean(1))
        h = self.f(torch.cat([last, mean], -1))
        out = {}
        for k in self.horizons:
            out[f'delta_{k}'] = self.delta_heads[str(k)](h)
            out[f'z_{k}'] = self.z_heads[str(k)](h)
        return out

class Curve175(nn.Module):

    def __init__(self, n_char: int):
        super().__init__()
        self.pen = AttnPen(n_char)
        self.dyn = DynHead()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pen(x)

def ctx_windows(z: torch.Tensor, t0: int, n_pred: int, ctx: int=CTX) -> torch.Tensor:
    ends = torch.arange(t0, t0 + n_pred, device=z.device)
    idx = ends.unsqueeze(1) - torch.arange(ctx - 1, -1, -1, device=z.device).unsqueeze(0)
    return z[:, idx.clamp(min=0)]

def train_loss(model: Curve175, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    z = model.encode(x)
    B, T, d = z.shape
    k_max = max(K_STEPS)
    t0 = CTX
    last_end = T - 1 - k_max
    n_pred = last_end - t0 + 1
    if n_pred < 1:
        raise RuntimeError('seq too short')
    z_ctx = ctx_windows(z, t0, n_pred)
    pred = model.dyn(z_ctx.reshape(B * n_pred, CTX, d))
    loss = 0.0
    stats = {}
    for k in K_STEPS:
        z_t = z[:, t0:t0 + n_pred]
        z_tk = z[:, t0 + k:t0 + n_pred + k]
        delta = z_tk - z_t
        d_hat = pred[f'delta_{k}'].view(B, n_pred, d)
        z_hat = pred[f'z_{k}'].view(B, n_pred, d)
        ld = 1.0 - F.cosine_similarity(d_hat, delta.detach(), dim=-1).mean()
        lz = 1.0 - F.cosine_similarity(z_hat, z_tk.detach(), dim=-1).mean()
        loss = loss + 1.0 / math.sqrt(k) * (ld + lz)
        stats[f'cos_d_k{k}'] = float(F.cosine_similarity(d_hat, delta.detach(), dim=-1).mean().detach())
    energy = (z[:, 1:] - z[:, :-1]).pow(2).mean()
    loss = loss + 0.01 * F.relu(0.05 - energy)
    stats['loss'] = float(loss.detach())
    stats['energy'] = float(energy.detach())
    return (loss, stats)

@torch.no_grad()
def encode_text(model, text: str, stoi: dict, device) -> torch.Tensor:
    ids = torch.tensor([[stoi.get(c, 0) for c in text]], device=device)
    return model.encode(ids)[0]

def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(F.normalize(a, dim=0), F.normalize(b, dim=0), dim=0))

def mine_same_suffix_pairs(text: str, rng: random.Random, n_pairs: int=100):
    L = PREFIX_LEN + SUFFIX_LEN
    by_suf = defaultdict(list)
    for i in range(0, len(text) - L - 1, 17):
        win = text[i:i + L]
        by_suf[win[-SUFFIX_LEN:]].append(win)
    pairs = []
    for suf, wins in by_suf.items():
        prefs = {}
        for w in wins:
            prefs[w[:-SUFFIX_LEN]] = w
            if len(prefs) >= 2:
                break
        if len(prefs) < 2:
            continue
        ws = list(prefs.values())
        pairs.append((ws[0], ws[1]))
        if len(pairs) >= n_pairs * 2:
            break
    rng.shuffle(pairs)
    return pairs[:n_pairs]

def mine_diff_suffix_pairs(text: str, rng: random.Random, n: int):
    L = PREFIX_LEN + SUFFIX_LEN
    out = []
    for _ in range(n * 4):
        i = rng.randint(0, len(text) - L - 1)
        j = rng.randint(0, len(text) - L - 1)
        a, b = (text[i:i + L], text[j:j + L])
        if a[-SUFFIX_LEN:] != b[-SUFFIX_LEN:]:
            out.append((a, b))
        if len(out) >= n:
            break
    return out

@torch.no_grad()
def gate_A(model, text, stoi, device, rng) -> dict:
    same = mine_same_suffix_pairs(text, rng, 100)
    diff = mine_diff_suffix_pairs(text, rng, 100)
    cos_same, cos_diff, cos_pref = ([], [], [])
    for a, b in same:
        za, zb = (encode_text(model, a, stoi, device), encode_text(model, b, stoi, device))
        cos_same.append(cos(za[-1], zb[-1]))
        i = PREFIX_LEN - 1
        cos_pref.append(cos(za[i], zb[i]))
    for a, b in diff:
        za, zb = (encode_text(model, a, stoi, device), encode_text(model, b, stoi, device))
        cos_diff.append(cos(za[-1], zb[-1]))
    m_same = float(np.mean(cos_same))
    m_diff = float(np.mean(cos_diff))
    m_pref = float(np.mean(cos_pref))
    if m_same >= 0.98:
        verdict = 'A_FAIL_STILL_SUFFIX_WIPED'
    elif m_same < 0.9 and m_same - m_diff < 0.35:
        verdict = 'A_PASS_PREFIX_VISIBLE_AT_ENDPOINT'
    else:
        verdict = 'A_WEAK_PARTIAL'
    return {'verdict': verdict, 'mean_cos_endpoint_same_suffix': m_same, 'mean_cos_endpoint_diff_suffix': m_diff, 'mean_cos_at_prefix_end': m_pref, 'n_same': len(cos_same), 'n_diff': len(cos_diff)}
PARAPHRASE_PAIRS = [('The cat sat on the mat.', 'A cat was sitting on the mat.'), ('She quickly opened the door.', 'She opened the door quickly.'), ('He bought a new car yesterday.', 'Yesterday he purchased a new automobile.'), ('The weather is very cold today.', 'It is extremely chilly outside today.'), ('Children are playing in the park.', 'Kids are playing at the park.'), ('I need to finish this work soon.', 'I must complete this task shortly.'), ('Please close the window.', 'Could you shut the window?'), ('The train leaves at noon.', 'The train departs at midday.'), ('He is afraid of spiders.', 'Spiders scare him.'), ('The film was long and boring.', 'The movie was lengthy and dull.'), ('We should start the meeting now.', "Let's begin the meeting now."), ('His answer was completely wrong.', 'His reply was totally incorrect.')]
HARD_PAIRS = [('The cat sat on the mat.', 'The car sat on the mat.'), ('She opened the door quickly.', 'She opened the book quickly.'), ('He bought a new car yesterday.', 'He bought a new cat yesterday.'), ('The weather is very cold today.', 'The weather is very warm today.'), ('The train leaves at noon.', 'The plane leaves at noon.'), ('She teaches mathematics at school.', 'She teaches history at school.')]

@torch.no_grad()
def gate_B(model, stoi, device, rng) -> dict:

    def summ(text):
        z = encode_text(model, text, stoi, device)
        return F.normalize(torch.cat([z[-1], z.mean(0)], 0), dim=0)
    para = [cos(summ(a), summ(b)) for a, b in PARAPHRASE_PAIRS]
    hard = [cos(summ(a), summ(b)) for a, b in HARD_PAIRS]
    flat = []
    for a, b in PARAPHRASE_PAIRS:
        flat.extend([summ(a), summ(b)])
    rand = []
    for _ in range(len(para) * 4):
        i, j = rng.sample(range(len(flat)), 2)
        rand.append(cos(flat[i], flat[j]))
    m_para, m_rand, m_hard = (float(np.mean(para)), float(np.mean(rand)), float(np.mean(hard)))
    lift_r, lift_h = (m_para - m_rand, m_para - m_hard)
    if lift_r > 0.05 and lift_h > 0.03:
        verdict = 'B_PASS_MEANING_STRUCTURE'
    elif lift_h <= 0.0 and lift_r > 0.02:
        verdict = 'B_FAIL_FORM_NOT_MEANING'
    elif lift_r <= 0.02:
        verdict = 'B_FAIL_NO_PARAPHRASE_CLUSTER'
    else:
        verdict = 'B_WEAK_MIXED'
    return {'verdict': verdict, 'mean_cos_paraphrase': m_para, 'mean_cos_random': m_rand, 'mean_cos_hard_spelling': m_hard, 'lift_vs_random': lift_r, 'lift_vs_hard': lift_h}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage175 start {datetime.now(timezone.utc).isoformat()}')
    log('Attn pen (causal Transformer) → fit dynamics → freeze → gates A/B')
    log(f'plan={PLAN}')
    text_full = s170.load_corpus(max_chars=20000000)
    stoi, itos = s170.build_charset(text_full)
    ids = np.fromiter((stoi[c] for c in text_full), dtype=np.int32, count=len(text_full))
    log(f'corpus={len(ids)} vocab={len(itos)} pen=AttnL{N_LAYERS}d{D} seq={SEQ}')
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = Curve175(len(itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    rng = random.Random(SEED)
    hold_text = text_full[int(0.7 * len(text_full)):int(0.7 * len(text_full)) + 2000000]
    A0 = gate_A(model, hold_text, stoi, device, random.Random(SEED))
    log(f"  init A: same={A0['mean_cos_endpoint_same_suffix']:.3f} diff={A0['mean_cos_endpoint_diff_suffix']:.3f} → {A0['verdict']}")
    model.train()
    running = None
    for step in range(1, args.steps + 1):
        x = s170.sample_char_batch(ids, MICRO, SEQ, rng, device)
        loss, st = train_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = st['loss'] if running is None else 0.95 * running + 0.05 * st['loss']
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            A = gate_A(model, hold_text, stoi, device, random.Random(SEED + step))
            log(f"  step {step}: loss~{running:.3f} cos_d1={st.get('cos_d_k1', 0):.3f} A_same={A['mean_cos_endpoint_same_suffix']:.3f} A_diff={A['mean_cos_endpoint_diff_suffix']:.3f} → {A['verdict']}")
            model.train()
    for p in model.pen.parameters():
        p.requires_grad_(False)
    model.pen.eval()
    log('pen FROZEN — final A/B gates')
    Af = gate_A(model, hold_text, stoi, device, random.Random(SEED + 99))
    Bf = gate_B(model, stoi, device, random.Random(SEED + 100))
    log(f"  FINAL A: same={Af['mean_cos_endpoint_same_suffix']:.3f} diff={Af['mean_cos_endpoint_diff_suffix']:.3f} pref={Af['mean_cos_at_prefix_end']:.3f} → {Af['verdict']}")
    log(f"  FINAL B: para={Bf['mean_cos_paraphrase']:.3f} rand={Bf['mean_cos_random']:.3f} hard={Bf['mean_cos_hard_spelling']:.3f} → {Bf['verdict']}")
    if 'PASS' in Af['verdict'] and 'PASS' in Bf['verdict']:
        overall = 'ATTN_PEN_CONTEXT_YES'
    elif 'PASS' in Af['verdict']:
        overall = 'ATTN_PEN_CONTEXT_PARTIAL'
    elif 'FAIL' in Af['verdict']:
        overall = 'ATTN_PEN_CONTEXT_NULL'
    else:
        overall = 'ATTN_PEN_CONTEXT_WEAK'
    torch.save({'model': model.state_dict(), 'stoi': stoi, 'itos': itos, 'step': args.steps, 'pen': 'causal_transformer', 'A': Af, 'B': Bf}, CKPT_OUT)
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'attn_pen_context_gate_175', 'overall': overall, 'steps': args.steps, 'arch': {'pen': 'causal_transformer', 'd': D, 'layers': N_LAYERS, 'heads': N_HEADS}, 'A': Af, 'B': Bf, 'init_A': A0, 'note': 'Compared to GRU pen A_FAIL (same_suf cos=1.0). Gate is endpoint context retention.', 'next': 'If PARTIAL/YES: harden freeze+dyn, extend B. If NULL: try multi-state/memory pen, not deeper same attn only.'}
    write_json(DECISION, out)
    bullets = [f'`{overall}` steps={args.steps}', f"A: {Af['verdict']} same={Af['mean_cos_endpoint_same_suffix']:.3f} diff={Af['mean_cos_endpoint_diff_suffix']:.3f}", f"B: {Bf['verdict']} para={Bf['mean_cos_paraphrase']:.3f} hard={Bf['mean_cos_hard_spelling']:.3f}", f'GRU baseline A was same≈1.0 — attn must beat that wipe', out['next']]
    MINI.write_text('\n'.join(['# Stage175 — attn pen context gate', '', f'**Overall:** `{overall}`', ''] + [f'- {b}' for b in bullets] + ['']), encoding='utf-8')
    log(f'[175] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())