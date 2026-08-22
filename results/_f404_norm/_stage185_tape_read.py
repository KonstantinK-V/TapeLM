"""
Stage 185 — Addressable tape vs endpoint (falsify "endpoint = ceiling").

Two matched models, identical budget/seed/data, pure next-piece CE (no hand losses):
  ENDPOINT : logits = head([fast_t ; slow_t])           — memory = one point
  TAPE     : logits = head([fast_t ; read_t]),
             read_t = causal attention of fast_t over slow_1..slow_t — memory = addressable tape

Judge = calibrated Stage184 exam (log-prob). Reference: GPT next_tok=0.758.
Ablation for TAPE: shuffle slow tape along time at eval → accuracy must drop,
else the tape is decorative.

Verdicts:
  TAPE_READ_YES        gain>=+0.03 over endpoint AND shuffle drop>=0.05
  TAPE_GAIN_BUT_DECOR  gain without ablation drop (suspicious)
  TAPE_USED_NO_GAIN    ablation drop but no gain
  ENDPOINT_ENOUGH_HERE neither

  python _stage185_tape_read.py
  python _stage185_tape_read.py --steps 3000
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
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage180_dual_channel as s180
import _stage181_ce_control as s181
RES = Path('results')
DATA = Path('data')
CKPT_DIR = Path('checkpoints')
LOG = RES / '_stage185_log.txt'
DECISION = RES / 'stage185_decision.json'
MINI = RES / 'stage185_mini.md'
EXAM = DATA / 'stage184_exam.jsonl'
DEC184 = RES / 'stage184_decision.json'
TOK_PATH = s177.TOK_PATH
SEED = 185
D = s180.D
D_SLOW = s180.D_SLOW
MAX_ARCS = s177.MAX_ARCS
MAX_CHARS = s177.MAX_CHARS_PER_ARC
MICRO = 16
LR = 0.0003
EVAL_EVERY = 1000
DEFAULT_STEPS = 3000
N_MID_EVAL = 60
PAD = '[PAD]'

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

def build_char_table(tok: Tokenizer, stoi: dict, pad_id: int, V: int) -> torch.Tensor:
    """[V, MAX_CHARS] char-id rows for every BPE id; pad row = zeros."""
    table = torch.zeros(V, MAX_CHARS, dtype=torch.long)
    for tid in range(V):
        if tid == pad_id:
            continue
        piece = tok.decode([tid], skip_special_tokens=False) or ''
        for j, c in enumerate(piece[:MAX_CHARS]):
            table[tid, j] = stoi.get(c, 0)
    return table

class TapeReadModel(nn.Module):
    """mode='endpoint' → head([fast;slow_t]); mode='tape' → head([fast;read over slow_1..t])."""

    def __init__(self, n_char: int, V: int, mode: str):
        super().__init__()
        assert mode in ('endpoint', 'tape')
        self.mode = mode
        self.backbone = s180.DualChannel(n_char)
        if mode == 'tape':
            self.read = nn.MultiheadAttention(D_SLOW, num_heads=4, batch_first=True)
        self.head = nn.Linear(D + D_SLOW, V, bias=False)

    def logits(self, char_ids: torch.Tensor, pad: torch.Tensor, shuffle_tape: bool=False) -> torch.Tensor:
        _, fast, slow = self.backbone.forward_channels(char_ids, pad)
        if self.mode == 'endpoint':
            h = torch.cat([fast, slow], dim=-1)
            return self.head(h)
        kv = slow
        if shuffle_tape:
            T = slow.size(1)
            perm = torch.randperm(T, device=slow.device)
            kv = slow[:, perm]
        T = fast.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=fast.device), diagonal=1)
        read, _ = self.read(fast, kv, kv, attn_mask=causal, key_padding_mask=pad, need_weights=False)
        h = torch.cat([fast, read], dim=-1)
        return self.head(h)

def sample_id_batch(docs, batch, rng, device, pad_id):
    xs = []
    for _ in range(batch):
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < 8:
            doc = doc * 4
        max_start = max(0, len(doc) - MAX_ARCS)
        s = rng.randint(0, max_start) if max_start > 0 else 0
        window = doc[s:s + MAX_ARCS]
        if len(window) < MAX_ARCS:
            window = window + [pad_id] * (MAX_ARCS - len(window))
        xs.append(window)
    return torch.tensor(xs, dtype=torch.long, device=device)

def ce_step(model, ids, char_table, pad_id):
    pad = ids == pad_id
    char_ids = char_table[ids]
    logits = model.logits(char_ids, pad)
    target = ids[:, 1:]
    valid = ~pad[:, :-1] & ~pad[:, 1:]
    ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
    return ce

@torch.no_grad()
def span_logprob(model, char_table, pad_id, ctx_ids, cand_ids, device, shuffle_tape=False) -> float:
    seq = (ctx_ids + cand_ids)[-MAX_ARCS:]
    n_ctx = len(seq) - len(cand_ids)
    x = torch.tensor([seq], dtype=torch.long, device=device)
    pad = x == pad_id
    logits = model.logits(char_table[x], pad, shuffle_tape=shuffle_tape)[0]
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for k, tid in enumerate(cand_ids):
        pos = n_ctx + k - 1
        total += float(logp[pos, tid])
    return total / max(1, len(cand_ids))

@torch.no_grad()
def score_exam(model, char_table, pad_id, items, device, shuffle_tape=False, only_type=None, tag='') -> dict:
    model.eval()
    acc = {}
    for i, it in enumerate(items):
        t = it['type']
        if only_type and t != only_type:
            continue
        scores = [span_logprob(model, char_table, pad_id, it['ctx_ids'], c, device, shuffle_tape) for c in it['cand_ids']]
        pred = int(np.argmax(scores))
        ok, n = acc.get(t, (0, 0))
        acc[t] = (ok + int(pred == it['gold_idx']), n + 1)
    out = {}
    for t, (ok, n) in acc.items():
        out[f'{t}_acc'] = ok / max(1, n)
        out[f'{t}_n'] = n
    return out

def train_variant(mode: str, train_docs, items_mid, items_full, char_table, pad_id, V, n_char, steps, device):
    torch.manual_seed(SEED)
    model = TapeReadModel(n_char, V, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    running = None
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        ids = sample_id_batch(train_docs, MICRO, rng, device, pad_id)
        ce = ce_step(model, ids, char_table, pad_id)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(ce) if running is None else 0.95 * running + 0.05 * float(ce)
        if step % EVAL_EVERY == 0 or step == steps:
            mid = score_exam(model, char_table, pad_id, items_mid, device, only_type='next_tok')
            log(f"  [{mode}] step {step}: ce~{running:.3f} next_tok(mid)={mid.get('next_tok_acc', 0):.3f} ({time.time() - t0:.0f}s)")
            model.train()
    model.eval()
    full = score_exam(model, char_table, pad_id, items_full, device)
    res = {'ce_final': running, **full}
    if mode == 'tape':
        shuf = score_exam(model, char_table, pad_id, items_full, device, shuffle_tape=True, only_type='next_tok')
        res['next_tok_shuffled'] = shuf.get('next_tok_acc', 0.0)
    torch.save({'model': model.state_dict(), 'mode': mode, 'res': res}, CKPT_DIR / f'stage185_{mode}.pt')
    return res

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage185 start {datetime.now(timezone.utc).isoformat()}')
    log('Addressable tape (query read) vs endpoint — matched CE, judge = calibrated 184 exam')
    if not EXAM.exists():
        log('FATAL: run _stage184_exam_logprob.py first (needs data/stage184_exam.jsonl)')
        return 1
    items_full = [json.loads(l) for l in EXAM.read_text(encoding='utf-8').splitlines() if l.strip()]
    items_mid = [it for it in items_full if it['type'] == 'next_tok'][:N_MID_EVAL]
    log(f'exam items={len(items_full)} mid={len(items_mid)}')
    device = torch.device(args.device)
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    text = s170.load_corpus(max_chars=20000000)
    chars = sorted(set(text) | {' '})
    itos = ['<pad>'] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = s181.build_id_docs(tok, text)
    train_docs = docs[:int(0.8 * len(docs))] or docs
    log(f'docs={len(docs)} V={V} n_char={len(itos)}')
    char_table = build_char_table(tok, stoi, pad_id, V).to(device)
    log('char table ready')
    gpt_ref = None
    if DEC184.exists():
        d = json.loads(DEC184.read_text(encoding='utf-8'))
        gpt_ref = d['results']['ce_gpt_181']
    freq = np.ones(V)
    for doc in train_docs:
        for t in doc:
            freq[t] += 1
    logfreq = np.log(freq / freq.sum())
    uni = {'next_tok': [0, 0], 'entity': [0, 0], 'ood': [0, 0]}
    for it in items_full:
        scores = [float(np.mean([logfreq[t] for t in c])) for c in it['cand_ids']]
        ok, n = uni[it['type']]
        uni[it['type']] = (ok + int(int(np.argmax(scores)) == it['gold_idx']), n + 1)
    unigram = {f'{t}_acc': ok / max(1, n) for t, (ok, n) in uni.items()}
    log(f"unigram baseline: next_tok={unigram['next_tok_acc']:.3f} entity={unigram['entity_acc']:.3f}")
    results = {}
    for mode in ('endpoint', 'tape'):
        log(f'train {mode} …')
        results[mode] = train_variant(mode, train_docs, items_mid, items_full, char_table, pad_id, V, len(itos), args.steps, device)
        r = results[mode]
        log(f"  {mode} FINAL: next_tok={r['next_tok_acc']:.3f} entity={r.get('entity_acc', 0):.3f} ood={r.get('ood_acc', 0):.3f}" + (f" shuffled={r['next_tok_shuffled']:.3f}" if 'next_tok_shuffled' in r else ''))
    gain = results['tape']['next_tok_acc'] - results['endpoint']['next_tok_acc']
    drop = results['tape']['next_tok_acc'] - results['tape'].get('next_tok_shuffled', results['tape']['next_tok_acc'])
    if gain >= 0.03 and drop >= 0.05:
        overall = 'TAPE_READ_YES'
    elif gain >= 0.03:
        overall = 'TAPE_GAIN_BUT_DECOR'
    elif drop >= 0.05:
        overall = 'TAPE_USED_NO_GAIN'
    else:
        overall = 'ENDPOINT_ENOUGH_HERE'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'tape_read_vs_endpoint_185', 'overall': overall, 'gain_tape_minus_endpoint': gain, 'shuffle_drop': drop, 'gpt_ref_184': gpt_ref, 'unigram_baseline': unigram, 'steps': args.steps, 'results': results, 'note': 'matched budget/seed/data; pure CE; no retention/hand losses'}
    write_json(DECISION, out)
    lines = ['# Stage185 — addressable tape vs endpoint', '', f'**Overall:** `{overall}`  (gain={gain:+.3f}, shuffle_drop={drop:+.3f})', '', f"GPT ref next_tok: {gpt_ref['next_tok_acc']:.3f}" if gpt_ref else '', f"Unigram (no context): next_tok={unigram['next_tok_acc']:.3f} — context credit is only what's above this"]
    for mode, r in results.items():
        lines.append(f"- `{mode}`: next_tok={r['next_tok_acc']:.3f} entity={r.get('entity_acc', 0):.3f} ood={r.get('ood_acc', 0):.3f}" + (f" shuffled={r['next_tok_shuffled']:.3f}" if 'next_tok_shuffled' in r else ''))
    MINI.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log(f'[185] {overall} gain={gain:+.3f} drop={drop:+.3f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())