"""
Stage 176 — Curve-as-tokens (arcs).

Split the drawn path into arc-tokens (whitespace/punct segments).
Each arc → one vector (local char encoder).
Causal Transformer over arc sequence predicts next arc / Δ.
NO char/word CE teacher.

Gate A (arc-level): same last arc string, different prefix arcs →
  does final state still wipe? (compare to char-GRU/attn wipe)

  python _stage176_curve_tokens.py
  python _stage176_curve_tokens.py --steps 15000
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import _stage170_curve_dynamics as s170
RES = Path('results')
CKPT_DIR = Path('checkpoints')
LOG = RES / '_stage176_log.txt'
DECISION = RES / 'stage176_decision.json'
MINI = RES / 'stage176_mini.md'
CKPT_OUT = CKPT_DIR / 'stage176_curve_tokens.pt'
PLAN = RES / 'plan_curve_dynamics.md'
SEED = 176
D = 128
N_LAYERS = 4
N_HEADS = 4
MAX_ARCS = 64
MAX_CHARS_PER_ARC = 24
MICRO = 24
LR = 0.0003
EVAL_EVERY = 1500
DEFAULT_STEPS = 15000
ARC_RE = re.compile('\\S+|\\s+')

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

def split_arcs(text: str) -> list[str]:
    """Token-like arcs: non-space chunks; drop pure whitespace (boundary only)."""
    parts = ARC_RE.findall(text)
    return [p for p in parts if p.strip()]

def arcs_to_char_ids(arcs: list[str], stoi: dict, max_chars: int=MAX_CHARS_PER_ARC) -> torch.Tensor:
    """[n_arcs, max_chars] with pad 0; also lengths."""
    rows = []
    for a in arcs:
        ids = [stoi.get(c, 0) for c in a[:max_chars]]
        if len(ids) < max_chars:
            ids = ids + [0] * (max_chars - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)

class ArcEncoder(nn.Module):
    """Local char → one arc vector (mean pool over chars). Ink for one token."""

    def __init__(self, n_char: int, d: int=D):
        super().__init__()
        self.emb = nn.Embedding(n_char, d, padding_idx=0)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(char_ids)
        mask = (char_ids != 0).float().unsqueeze(-1)
        summed = (h * mask).sum(dim=-2)
        denom = mask.sum(dim=-2).clamp(min=1.0)
        pooled = summed / denom
        return self.norm(self.ff(pooled))

class CausalBlock(nn.Module):

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.n1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None=None) -> torch.Tensor:
        T = x.size(1)
        attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.n1(x + h)
        return self.n2(x + self.ff(x))

class ArcTransformer(nn.Module):

    def __init__(self, d: int=D, n_layers: int=N_LAYERS):
        super().__init__()
        self.pos = nn.Embedding(MAX_ARCS, d)
        self.blocks = nn.ModuleList([CausalBlock(d, N_HEADS) for _ in range(n_layers)])

    def forward(self, arc_vecs: torch.Tensor, pad_mask: torch.Tensor | None=None) -> torch.Tensor:
        B, A, d = arc_vecs.shape
        pos = torch.arange(A, device=arc_vecs.device).unsqueeze(0).expand(B, A)
        x = arc_vecs + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        return x

class CurveTokenModel(nn.Module):

    def __init__(self, n_char: int):
        super().__init__()
        self.arc_enc = ArcEncoder(n_char)
        self.tr = ArcTransformer()
        self.pred = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

    def encode_arcs(self, char_ids: torch.Tensor) -> torch.Tensor:
        return self.arc_enc(char_ids)

    def forward_states(self, char_ids: torch.Tensor, pad_mask: torch.Tensor | None=None) -> torch.Tensor:
        arcs = self.encode_arcs(char_ids)
        return self.tr(arcs, pad_mask=pad_mask)

def sample_arc_batch(corpus_arcs: list[list[str]], stoi: dict, batch: int, rng: random.Random, device):
    """Sample contiguous arc windows."""
    xs = []
    masks = []
    for _ in range(batch):
        doc = corpus_arcs[rng.randint(0, len(corpus_arcs) - 1)]
        if len(doc) < 8:
            doc = doc * 4
        max_start = max(0, len(doc) - MAX_ARCS)
        s = rng.randint(0, max_start) if max_start > 0 else 0
        window = doc[s:s + MAX_ARCS]
        pad_n = MAX_ARCS - len(window)
        if pad_n > 0:
            window = window + [''] * pad_n
        char_ids = arcs_to_char_ids(window, stoi)
        xs.append(char_ids)
        mask = torch.tensor([a == '' for a in window], dtype=torch.bool)
        masks.append(mask)
    x = torch.stack(xs, 0).to(device)
    pad = torch.stack(masks, 0).to(device)
    return (x, pad)

def train_loss(model: CurveTokenModel, char_ids: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, dict]:
    with torch.set_grad_enabled(True):
        arc_emb = model.encode_arcs(char_ids)
        states = model.tr(arc_emb, pad_mask=pad)
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        if valid.sum() < 1:
            return (states.sum() * 0.0, {'loss': 0.0, 'cos': 0.0})
        pred = model.pred(states[:, :-1])
        target = arc_emb[:, 1:]
        delta_hat = pred - states[:, :-1]
        delta = arc_emb[:, 1:] - arc_emb[:, :-1]
        cos_n = F.cosine_similarity(pred[valid], target[valid].detach(), dim=-1).mean()
        cos_d = F.cosine_similarity(delta_hat[valid], delta[valid].detach(), dim=-1).mean()
        loss = 1.0 - cos_n + (1.0 - cos_d)
        loss = loss + 0.1 * F.mse_loss(pred[valid], target[valid].detach())
        stats = {'loss': float(loss.detach()), 'cos': float(cos_n.detach()), 'cos_d': float(cos_d.detach())}
        return (loss, stats)

def build_arc_corpus(text: str, max_docs: int=4000) -> list[list[str]]:
    docs = []
    for block in text.split('\n\n'):
        arcs = split_arcs(block)
        if len(arcs) >= 16:
            docs.append(arcs)
        if len(docs) >= max_docs:
            break
    if len(docs) < 50:
        arcs = split_arcs(text)
        for i in range(0, len(arcs) - 64, 32):
            docs.append(arcs[i:i + 128])
            if len(docs) >= max_docs:
                break
    return docs

@torch.no_grad()
def encode_arc_seq(model, arcs: list[str], stoi, device) -> torch.Tensor:
    arcs = arcs[-MAX_ARCS:]
    if not arcs:
        arcs = ['.']
    char_ids = arcs_to_char_ids(arcs, stoi).unsqueeze(0).to(device)
    pad = torch.zeros(1, len(arcs), dtype=torch.bool, device=device)
    states = model.forward_states(char_ids, pad_mask=pad)
    return states[0]

def cos(a, b) -> float:
    return float(F.cosine_similarity(F.normalize(a, dim=0), F.normalize(b, dim=0), dim=0))

def gate_A_arcs(model, docs: list[list[str]], stoi, device, rng, n_pairs: int=80) -> dict:
    """Same last arc string, different prefixes → endpoint state cos."""
    by_last = defaultdict(list)
    for doc in docs:
        if len(doc) < 12:
            continue
        for i in range(8, len(doc)):
            pref = tuple(doc[max(0, i - 24):i])
            last = doc[i]
            by_last[last].append(list(pref) + [last])
    pairs_same = []
    for last, seqs in by_last.items():
        uniq = {}
        for s in seqs:
            key = tuple(s[:-1])
            if key not in uniq:
                uniq[key] = s
            if len(uniq) >= 2:
                break
        if len(uniq) >= 2:
            vals = list(uniq.values())
            pairs_same.append((vals[0], vals[1]))
        if len(pairs_same) >= n_pairs:
            break
    rng.shuffle(pairs_same)
    pairs_same = pairs_same[:n_pairs]
    pairs_diff = []
    flat = [s for seqs in list(by_last.values())[:200] for s in seqs[:3]]
    for _ in range(n_pairs * 3):
        if len(flat) < 2:
            break
        a, b = rng.sample(flat, 2)
        if a[-1] != b[-1]:
            pairs_diff.append((a, b))
        if len(pairs_diff) >= n_pairs:
            break
    cos_same, cos_diff = ([], [])
    for a, b in pairs_same:
        za = encode_arc_seq(model, a, stoi, device)[-1]
        zb = encode_arc_seq(model, b, stoi, device)[-1]
        cos_same.append(cos(za, zb))
    for a, b in pairs_diff:
        za = encode_arc_seq(model, a, stoi, device)[-1]
        zb = encode_arc_seq(model, b, stoi, device)[-1]
        cos_diff.append(cos(za, zb))
    m_same = float(np.mean(cos_same)) if cos_same else 1.0
    m_diff = float(np.mean(cos_diff)) if cos_diff else 0.0
    if m_same >= 0.98:
        verdict = 'A_FAIL_LAST_ARC_WIPES'
    elif m_same < 0.9 and m_same - m_diff < 0.35:
        verdict = 'A_PASS_PREFIX_VISIBLE'
    else:
        verdict = 'A_WEAK_PARTIAL'
    return {'verdict': verdict, 'mean_cos_same_last_arc': m_same, 'mean_cos_diff_last_arc': m_diff, 'n_same': len(cos_same), 'n_diff': len(cos_diff)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage176 start {datetime.now(timezone.utc).isoformat()}')
    log('Curve-as-tokens: arcs (whitespace segments) + causal Transformer next-arc/Δ')
    log(f'plan={PLAN}')
    text = s170.load_corpus(max_chars=20000000)
    chars = sorted(set(text))
    itos = ['<pad>'] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = build_arc_corpus(text)
    log(f'docs={len(docs)} vocab={len(itos)} max_arcs={MAX_ARCS} d={D} layers={N_LAYERS}')
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = CurveTokenModel(len(itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    rng = random.Random(SEED)
    hold_docs = docs[int(0.8 * len(docs)):] or docs[-100:]
    train_docs = docs[:int(0.8 * len(docs))] or docs
    A0 = gate_A_arcs(model, hold_docs, stoi, device, random.Random(SEED))
    log(f"  init A: same={A0['mean_cos_same_last_arc']:.3f} diff={A0['mean_cos_diff_last_arc']:.3f} → {A0['verdict']}")
    model.train()
    running = None
    best_A = A0
    for step in range(1, args.steps + 1):
        x, pad = sample_arc_batch(train_docs, stoi, MICRO, rng, device)
        loss, st = train_loss(model, x, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = st['loss'] if running is None else 0.95 * running + 0.05 * st['loss']
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            A = gate_A_arcs(model, hold_docs, stoi, device, random.Random(SEED + step))
            best_A = A
            log(f"  step {step}: loss~{running:.3f} cos_next={st['cos']:.3f} cos_d={st['cos_d']:.3f} A_same={A['mean_cos_same_last_arc']:.3f} A_diff={A['mean_cos_diff_last_arc']:.3f} → {A['verdict']}")
            model.train()
            torch.save({'model': model.state_dict(), 'stoi': stoi, 'itos': itos, 'step': step, 'A': A}, CKPT_OUT)
    Af = best_A
    if 'PASS' in Af['verdict']:
        overall = 'CURVE_TOKENS_CONTEXT_YES'
    elif 'WEAK' in Af['verdict']:
        overall = 'CURVE_TOKENS_CONTEXT_WEAK'
    else:
        overall = 'CURVE_TOKENS_CONTEXT_NULL'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'curve_as_tokens_176', 'overall': overall, 'steps': args.steps, 'arc_definition': 'whitespace-separated non-space chunks (word/punct-like)', 'unit': 'arc vector from local char mean-pool; sequence model = causal Transformer', 'loss': 'next-arc cosine + arc-Δ cosine (no CE)', 'A': Af, 'init_A': A0, 'note': 'Analog of BPE tokens but continuous arc embeddings on the curve.', 'next': 'If YES/WEAK: add B paraphrase + harden. If NULL: arc unit still local-wipe — try longer memory / retention loss.'}
    write_json(DECISION, out)
    bullets = [f'`{overall}`', f"A: {Af['verdict']} same_last_arc={Af['mean_cos_same_last_arc']:.3f} diff={Af['mean_cos_diff_last_arc']:.3f}", 'arcs = whitespace segments; Transformer predicts next arc/Δ', out['next']]
    MINI.write_text('\n'.join(['# Stage176 — curve as tokens', '', f'**Overall:** `{overall}`', ''] + [f'- {b}' for b in bullets] + ['']), encoding='utf-8')
    log(f'[176] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())