"""
Stage 177 — Curve-BPE (real statistical tokens on the path).

Unlike 176 (whitespace words), this mirrors GPT/BPE:
  - merges by corpus frequency, not meaning / grammar rules
  - frequent forms often = 1 piece; rare/new = several pieces
  - leading space is INSIDE the token (ByteLevel Ġ / decoded space)

Each BPE piece → continuous arc vector (local char pool on the ink of that piece).
Causal Transformer predicts next-arc / Δ. NO BPE-id CE teacher.

Gate A: same last BPE piece string, different prefixes → endpoint wipe?

  python _stage177_curve_bpe.py
  python _stage177_curve_bpe.py --steps 15000 --vocab 4096
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import _stage170_curve_dynamics as s170
RES = Path('results')
CKPT_DIR = Path('checkpoints')
LOG = RES / '_stage177_log.txt'
DECISION = RES / 'stage177_decision.json'
MINI = RES / 'stage177_mini.md'
TOK_PATH = RES / 'stage177_curve_bpe_tokenizer.json'
CKPT_OUT = CKPT_DIR / 'stage177_curve_bpe.pt'
PLAN = RES / 'plan_curve_dynamics.md'
SEED = 177
D = 128
N_LAYERS = 4
N_HEADS = 4
MAX_ARCS = 64
MAX_CHARS_PER_ARC = 24
MICRO = 24
LR = 0.0003
EVAL_EVERY = 1500
DEFAULT_STEPS = 15000
DEFAULT_VOCAB = 4096
PAD_TOKEN = '[PAD]'

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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def train_or_load_bpe(corpus: str, vocab_size: int, force: bool=False) -> Tokenizer:
    """GPT-2-style ByteLevel BPE: space lives inside tokens (Ġ…)."""
    if TOK_PATH.exists() and (not force):
        tok = Tokenizer.from_file(str(TOK_PATH))
        log(f'[bpe] reuse {TOK_PATH.name} V={tok.get_vocab_size()}')
        return tok
    log(f'[bpe] train ByteLevel BPE V={vocab_size} on {len(corpus):,} chars')
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[PAD_TOKEN], show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    lines = [ln for ln in corpus.splitlines() if ln.strip()]
    if len(lines) < 100:
        lines = [corpus[i:i + 256] for i in range(0, min(len(corpus), 5000000), 128)]
    tok.train_from_iterator(lines, trainer=trainer)
    TOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOK_PATH))
    log(f'[bpe] saved {TOK_PATH} V={tok.get_vocab_size()}')
    return tok

def encode_pieces(tok: Tokenizer, text: str) -> list[str]:
    """Return decoded surface strings of BPE pieces (space may be inside)."""
    enc = tok.encode(text)
    pieces = []
    for tid in enc.ids:
        if tid == tok.token_to_id(PAD_TOKEN):
            continue
        s = tok.decode([tid], skip_special_tokens=False)
        if s == '' and tid is not None:
            inv = {v: k for k, v in tok.get_vocab().items()}
            s = inv.get(tid, '')
        if s:
            pieces.append(s)
    return pieces

def bpe_stats(tok: Tokenizer, sample_text: str, n_words: int=2000) -> dict:
    """Show frequent vs rare word fragmentation (BPE property check)."""
    import re
    words = re.findall('\\S+', sample_text[:800000])
    if len(words) > n_words:
        words = words[:n_words]
    counts = Counter(words)
    freq_words = [w for w, _ in counts.most_common(40)]
    rare_words = [w for w, c in counts.most_common() if c == 1][:40]
    if len(rare_words) < 20:
        rare_words = [w for w, c in counts.most_common() if c <= 2][:40]

    def n_pieces(w: str, with_leading_space: bool) -> int:
        s = ' ' + w if with_leading_space else w
        return max(1, len(encode_pieces(tok, s)))
    freq_n = [n_pieces(w, True) for w in freq_words]
    rare_n = [n_pieces(w, True) for w in rare_words] if rare_words else [0]
    examples = []
    for w in freq_words[:5] + rare_words[:5]:
        pcs = encode_pieces(tok, ' ' + w)
        examples.append({'word': w, 'pieces': pcs, 'n': len(pcs)})
    space_in_token = sum((1 for e in examples for p in e['pieces'] if p.startswith(' ') or p.startswith('Ġ') or 'Ġ' in p))
    lead_space = 0
    for w in freq_words[:30]:
        pcs = encode_pieces(tok, ' the ' + w)
        if pcs and (pcs[0].startswith(' ') or any((p.startswith(' ') for p in pcs[:2]))):
            lead_space += 1
    return {'freq_mean_pieces': float(np.mean(freq_n)) if freq_n else 0.0, 'rare_mean_pieces': float(np.mean(rare_n)) if rare_n else 0.0, 'freq_frac_single': float(np.mean([n == 1 for n in freq_n])) if freq_n else 0.0, 'rare_frac_single': float(np.mean([n == 1 for n in rare_n])) if rare_n else 0.0, 'examples': examples[:10], 'note': 'freq should be closer to 1 piece; rare more fragmented; space inside pieces'}

def pieces_to_char_ids(pieces: list[str], stoi: dict, max_chars: int=MAX_CHARS_PER_ARC) -> torch.Tensor:
    rows = []
    for a in pieces:
        ids = [stoi.get(c, 0) for c in a[:max_chars]]
        if len(ids) < max_chars:
            ids = ids + [0] * (max_chars - len(ids))
        rows.append(ids)
    if not rows:
        rows = [[0] * max_chars]
    return torch.tensor(rows, dtype=torch.long)

class ArcEncoder(nn.Module):

    def __init__(self, n_char: int, d: int=D):
        super().__init__()
        self.emb = nn.Embedding(n_char, d, padding_idx=0)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(char_ids)
        mask = (char_ids != 0).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1.0)
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
        B, A, _ = arc_vecs.shape
        pos = torch.arange(A, device=arc_vecs.device).unsqueeze(0).expand(B, A)
        x = arc_vecs + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        return x

class CurveBPEModel(nn.Module):

    def __init__(self, n_char: int):
        super().__init__()
        self.arc_enc = ArcEncoder(n_char)
        self.tr = ArcTransformer()
        self.pred = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

    def encode_arcs(self, char_ids: torch.Tensor) -> torch.Tensor:
        return self.arc_enc(char_ids)

    def forward_states(self, char_ids: torch.Tensor, pad_mask: torch.Tensor | None=None) -> torch.Tensor:
        return self.tr(self.encode_arcs(char_ids), pad_mask=pad_mask)

def build_piece_docs(tok: Tokenizer, text: str, max_docs: int=4000) -> list[list[str]]:
    docs = []
    for block in text.split('\n\n'):
        block = block.strip()
        if len(block) < 40:
            continue
        pcs = encode_pieces(tok, block)
        if len(pcs) >= 16:
            docs.append(pcs)
        if len(docs) >= max_docs:
            break
    if len(docs) < 50:
        pcs = encode_pieces(tok, text[:2000000])
        for i in range(0, max(1, len(pcs) - 64), 48):
            docs.append(pcs[i:i + 128])
            if len(docs) >= max_docs:
                break
    return docs

def sample_batch(docs: list[list[str]], stoi: dict, batch: int, rng: random.Random, device):
    xs, masks = ([], [])
    for _ in range(batch):
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < 8:
            doc = doc * 4
        max_start = max(0, len(doc) - MAX_ARCS)
        s = rng.randint(0, max_start) if max_start > 0 else 0
        window = doc[s:s + MAX_ARCS]
        pad_n = MAX_ARCS - len(window)
        if pad_n > 0:
            window = window + [''] * pad_n
        xs.append(pieces_to_char_ids(window, stoi))
        masks.append(torch.tensor([a == '' for a in window], dtype=torch.bool))
    return (torch.stack(xs, 0).to(device), torch.stack(masks, 0).to(device))

def train_loss(model: CurveBPEModel, char_ids: torch.Tensor, pad: torch.Tensor):
    arc_emb = model.encode_arcs(char_ids)
    states = model.tr(arc_emb, pad_mask=pad)
    valid = ~pad[:, :-1] & ~pad[:, 1:]
    if valid.sum() < 1:
        return (states.sum() * 0.0, {'loss': 0.0, 'cos': 0.0, 'cos_d': 0.0})
    pred = model.pred(states[:, :-1])
    target = arc_emb[:, 1:]
    delta_hat = pred - states[:, :-1]
    delta = arc_emb[:, 1:] - arc_emb[:, :-1]
    cos_n = F.cosine_similarity(pred[valid], target[valid].detach(), dim=-1).mean()
    cos_d = F.cosine_similarity(delta_hat[valid], delta[valid].detach(), dim=-1).mean()
    loss = 1.0 - cos_n + (1.0 - cos_d) + 0.1 * F.mse_loss(pred[valid], target[valid].detach())
    return (loss, {'loss': float(loss.detach()), 'cos': float(cos_n.detach()), 'cos_d': float(cos_d.detach())})

@torch.no_grad()
def encode_seq(model, pieces: list[str], stoi, device) -> torch.Tensor:
    pieces = pieces[-MAX_ARCS:] or ['.']
    char_ids = pieces_to_char_ids(pieces, stoi).unsqueeze(0).to(device)
    pad = torch.zeros(1, len(pieces), dtype=torch.bool, device=device)
    return model.forward_states(char_ids, pad_mask=pad)[0]

def cos(a, b) -> float:
    return float(F.cosine_similarity(F.normalize(a, dim=0), F.normalize(b, dim=0), dim=0))

def gate_A(model, docs: list[list[str]], stoi, device, rng, n_pairs: int=80) -> dict:
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
    for _ in range(n_pairs * 4):
        if len(flat) < 2:
            break
        a, b = rng.sample(flat, 2)
        if a[-1] != b[-1]:
            pairs_diff.append((a, b))
        if len(pairs_diff) >= n_pairs:
            break
    cos_same, cos_diff = ([], [])
    for a, b in pairs_same:
        cos_same.append(cos(encode_seq(model, a, stoi, device)[-1], encode_seq(model, b, stoi, device)[-1]))
    for a, b in pairs_diff:
        cos_diff.append(cos(encode_seq(model, a, stoi, device)[-1], encode_seq(model, b, stoi, device)[-1]))
    m_same = float(np.mean(cos_same)) if cos_same else 1.0
    m_diff = float(np.mean(cos_diff)) if cos_diff else 0.0
    if m_same >= 0.98:
        verdict = 'A_FAIL_LAST_PIECE_WIPES'
    elif m_same < 0.9 and m_same - m_diff < 0.35:
        verdict = 'A_PASS_PREFIX_VISIBLE'
    else:
        verdict = 'A_WEAK_PARTIAL'
    return {'verdict': verdict, 'mean_cos_same_last_piece': m_same, 'mean_cos_diff_last_piece': m_diff, 'n_same': len(cos_same), 'n_diff': len(cos_diff)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--vocab', type=int, default=DEFAULT_VOCAB)
    ap.add_argument('--force-bpe', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage177 start {datetime.now(timezone.utc).isoformat()}')
    log('Curve-BPE: statistical merges + space-in-token; continuous next-arc (no CE)')
    log(f'plan={PLAN}')
    text = s170.load_corpus(max_chars=20000000)
    tok = train_or_load_bpe(text, args.vocab, force=args.force_bpe)
    stats = bpe_stats(tok, text)
    log(f"[bpe] freq_mean_pcs={stats['freq_mean_pieces']:.2f} rare_mean_pcs={stats['rare_mean_pieces']:.2f} freq_single={stats['freq_frac_single']:.2f} rare_single={stats['rare_frac_single']:.2f}")
    for e in stats['examples'][:6]:
        log(f"  ex {e['word']!r} → {e['pieces']!r} (n={e['n']})")
    chars = sorted(set(text) | {' '})
    itos = ['<pad>'] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = build_piece_docs(tok, text)
    log(f'docs={len(docs)} char_vocab={len(itos)} max_arcs={MAX_ARCS} d={D}')
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = CurveBPEModel(len(itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    rng = random.Random(SEED)
    hold_docs = docs[int(0.8 * len(docs)):] or docs[-100:]
    train_docs = docs[:int(0.8 * len(docs))] or docs
    A0 = gate_A(model, hold_docs, stoi, device, random.Random(SEED))
    log(f"  init A: same={A0['mean_cos_same_last_piece']:.3f} diff={A0['mean_cos_diff_last_piece']:.3f} → {A0['verdict']}")
    model.train()
    running = None
    Af = A0
    for step in range(1, args.steps + 1):
        x, pad = sample_batch(train_docs, stoi, MICRO, rng, device)
        loss, st = train_loss(model, x, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = st['loss'] if running is None else 0.95 * running + 0.05 * st['loss']
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            Af = gate_A(model, hold_docs, stoi, device, random.Random(SEED + step))
            log(f"  step {step}: loss~{running:.3f} cos_next={st['cos']:.3f} cos_d={st['cos_d']:.3f} A_same={Af['mean_cos_same_last_piece']:.3f} A_diff={Af['mean_cos_diff_last_piece']:.3f} → {Af['verdict']}")
            model.train()
            torch.save({'model': model.state_dict(), 'stoi': stoi, 'itos': itos, 'step': step, 'A': Af, 'vocab': args.vocab}, CKPT_OUT)
    if 'PASS' in Af['verdict']:
        overall = 'CURVE_BPE_CONTEXT_YES'
    elif 'WEAK' in Af['verdict']:
        overall = 'CURVE_BPE_CONTEXT_WEAK'
    else:
        overall = 'CURVE_BPE_CONTEXT_NULL'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'curve_bpe_177', 'overall': overall, 'practical': 'STILL_LAST_UNIT_WIPE' if Af['mean_cos_same_last_piece'] >= 0.95 else 'PARTIAL' if 'WEAK' in Af['verdict'] else Af['verdict'], 'steps': args.steps, 'bpe': {'style': 'ByteLevel BPE (GPT-2-like); space inside tokens', 'vocab_size': tok.get_vocab_size(), 'tokenizer': str(TOK_PATH), 'stats': {k: v for k, v in stats.items() if k != 'examples'}, 'examples': stats['examples']}, 'vs_176': '176=whitespace words; 177=statistical merges + space-in-token', 'loss': 'next-piece cosine + piece-Delta cosine (no BPE-id CE)', 'A': Af, 'init_A': A0, 'next': 'If still wipe: retention/instance channel. If PASS: gate B paraphrase.'}
    write_json(DECISION, out)
    MINI.write_text('\n'.join(['# Stage177 — curve BPE', '', f'**Overall:** `{overall}`', '', f"- ByteLevel BPE V={tok.get_vocab_size()}; freq_pcs={stats['freq_mean_pieces']:.2f} rare_pcs={stats['rare_mean_pieces']:.2f}", f"- A: {Af['verdict']} same={Af['mean_cos_same_last_piece']:.3f} diff={Af['mean_cos_diff_last_piece']:.3f}", f'- vs 176: statistical merges + space-in-token (not whitespace words)', f"- {out['next']}", '']), encoding='utf-8')
    log(f'[177] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())