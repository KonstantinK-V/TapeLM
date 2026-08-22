"""
Stage 187 — Self-model: predict-own-next-state + surprise-gated writes.

Same clean-CE recipe as 185-endpoint (the healthy baseline), but the slow channel
now has a self-model:
  - pred(slow_t) = expectation of its own next write
  - surprise_t   = 1 - cos(expectation, actual write)
  - write gate  *= sigmoid(k * (surprise - 0.5))  → routine ink barely writes, novelty writes hard

Aux loss = self-prediction only (predictor head, detached target) — does NOT push
representations directly (lesson of 185: hand losses on representations are poison).

Gates (judge = calibrated Exam v2 from 186):
  G1 CE preserved : next_tok >= endpoint_185(v2) - 0.03
  G2 novelty      : mean surprise on UNSEEN hold docs > on seen train docs
  G3 calibration  : predictive entropy after FAKE entity > after real entity (knows-it-doesn't-know)

SELF_MODEL_YES = G1 & G2 & G3.

  python _stage187_self_model.py
  python _stage187_self_model.py --steps 3000
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
import _stage181_ce_control as s181
import _stage185_tape_read as s185
RES = Path('results')
DATA = Path('data')
CKPT_DIR = Path('checkpoints')
LOG = RES / '_stage187_log.txt'
DECISION = RES / 'stage187_decision.json'
MINI = RES / 'stage187_mini.md'
EXAM = DATA / 'stage186_exam_v2.jsonl'
DEC186 = RES / 'stage186_decision.json'
CKPT_OUT = CKPT_DIR / 'stage187_self_model.pt'
TOK_PATH = s177.TOK_PATH
SEED = 185
D = 128
MAX_ARCS = s177.MAX_ARCS
MICRO = 16
LR = 0.0003
EVAL_EVERY = 1000
DEFAULT_STEPS = 3000
W_SELF = 0.1
N_MID_EVAL = 60
PAD = '[PAD]'
FAKES = ['Zorblax', 'Quenith', 'Marbune', 'Xaldera', 'Kessari', 'Vornak', 'Talmidex', 'Orsiphon', 'Pholmar', 'Girenth']

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

class SurpriseWriter(nn.Module):
    """Write-budget memory where write strength is gated by self-prediction error."""

    def __init__(self, d_in: int, d_slow: int=D):
        super().__init__()
        self.write = nn.Sequential(nn.Linear(d_in + d_slow, d_slow), nn.GELU(), nn.Linear(d_slow, d_slow))
        self.gate = nn.Linear(d_in + d_slow, 1)
        self.pred = nn.Sequential(nn.Linear(d_slow, d_slow), nn.GELU(), nn.Linear(d_slow, d_slow))
        self.k = nn.Parameter(torch.tensor(4.0))
        self.norm = nn.LayerNorm(d_slow)
        self.init = nn.Parameter(torch.zeros(d_slow))

    def forward(self, arcs: torch.Tensor, pad: torch.Tensor):
        B, A, _ = arcs.shape
        slow = self.init.unsqueeze(0).expand(B, -1).clone()
        budget = torch.ones(B, 1, device=arcs.device)
        outs, surprises, pred_losses = ([], [], [])
        for t in range(A):
            x = arcs[:, t]
            h = torch.cat([x, slow], dim=-1)
            expectation = self.pred(slow)
            w = self.write(h)
            surprise = 1.0 - F.cosine_similarity(expectation, w, dim=-1)
            mod = torch.sigmoid(self.k * (surprise.unsqueeze(-1) - 0.5))
            mask = (~pad[:, t]).float().unsqueeze(-1)
            g = torch.sigmoid(self.gate(h)) * budget * mod * mask
            slow = self.norm(slow + g * w)
            budget = (budget - g).clamp(min=0.0)
            outs.append(slow)
            surprises.append(surprise * mask.squeeze(-1))
            pred_losses.append((1.0 - F.cosine_similarity(expectation, w.detach(), dim=-1)) * mask.squeeze(-1))
        return (torch.stack(outs, 1), torch.stack(surprises, 1), torch.stack(pred_losses, 1))

class SelfModel(nn.Module):

    def __init__(self, n_char: int, V: int):
        super().__init__()
        self.arc_enc = s177.ArcEncoder(n_char, d=D)
        self.fast = s177.ArcTransformer(d=D)
        self.slow = SurpriseWriter(D, D)
        self.head = nn.Linear(2 * D, V, bias=False)

    def forward_all(self, char_ids: torch.Tensor, pad: torch.Tensor):
        arcs = self.arc_enc(char_ids)
        fast = self.fast(arcs, pad_mask=pad)
        slow, surprise, pred_loss = self.slow(arcs, pad)
        logits = self.head(torch.cat([fast, slow], dim=-1))
        return (logits, surprise, pred_loss)

    def logits(self, char_ids: torch.Tensor, pad: torch.Tensor, shuffle_tape: bool=False) -> torch.Tensor:
        return self.forward_all(char_ids, pad)[0]

@torch.no_grad()
def mean_surprise(model, docs, char_table, pad_id, device, rng, n=100) -> float:
    vals = []
    for _ in range(n):
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < 8:
            continue
        window = doc[:MAX_ARCS]
        if len(window) < MAX_ARCS:
            window = window + [pad_id] * (MAX_ARCS - len(window))
        ids = torch.tensor([window], dtype=torch.long, device=device)
        pad = ids == pad_id
        _, surprise, _ = model.forward_all(char_table[ids], pad)
        valid = ~pad
        vals.append(float(surprise[valid].mean()))
    return float(np.mean(vals))

@torch.no_grad()
def entropy_after(model, char_table, pad_id, ctx_ids, span_ids, device) -> float:
    seq = (ctx_ids + span_ids)[-MAX_ARCS:]
    x = torch.tensor([seq], dtype=torch.long, device=device)
    pad = x == pad_id
    logits = model.logits(char_table[x], pad)[0, len(seq) - 1]
    p = F.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-09)).sum())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage187 start {datetime.now(timezone.utc).isoformat()}')
    log('Self-model: predict-own-next-state + surprise-gated slow writes')
    items = [json.loads(l) for l in EXAM.read_text(encoding='utf-8').splitlines() if l.strip()]
    items_mid = [it for it in items if it['type'] == 'next_tok'][:N_MID_EVAL]
    d186 = json.loads(DEC186.read_text(encoding='utf-8'))
    base_next = d186['results']['endpoint_185']['next_tok_acc']
    log(f'exam v2 items={len(items)}; baseline endpoint_185 next_tok={base_next:.3f}')
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
    hold_docs = docs[int(0.8 * len(docs)):] or docs[-100:]
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    log(f'docs={len(docs)} V={V} n_char={len(itos)}')
    torch.manual_seed(SEED)
    model = SelfModel(len(itos), V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    running, run_self = (None, None)
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        ids = s185.sample_id_batch(train_docs, MICRO, rng, device, pad_id)
        pad = ids == pad_id
        logits, surprise, pred_loss = model.forward_all(char_table[ids], pad)
        target = ids[:, 1:]
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        lp = pred_loss[~pad].mean()
        loss = ce + W_SELF * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(ce) if running is None else 0.95 * running + 0.05 * float(ce)
        run_self = float(lp) if run_self is None else 0.95 * run_self + 0.05 * float(lp)
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            mid = s185.score_exam(model, char_table, pad_id, items_mid, device, only_type='next_tok')
            log(f"  step {step}: ce~{running:.3f} self~{run_self:.3f} k={float(model.slow.k):.2f} next_tok(mid)={mid.get('next_tok_acc', 0):.3f} ({time.time() - t0:.0f}s)")
            model.train()
            torch.save({'model': model.state_dict(), 'step': step}, CKPT_OUT)
    model.eval()
    full = s185.score_exam(model, char_table, pad_id, items, device)
    next_tok = full.get('next_tok_acc', 0.0)
    s_train = mean_surprise(model, train_docs, char_table, pad_id, device, random.Random(1))
    s_hold = mean_surprise(model, hold_docs, char_table, pad_id, device, random.Random(2))
    ent_items = [it for it in items if it['type'] == 'entity'][:80]
    rngf = random.Random(3)
    e_real, e_fake = ([], [])
    for it in ent_items:
        gold_ids = it['cand_ids'][it['gold_idx']]
        fake = FAKES[rngf.randint(0, len(FAKES) - 1)]
        fake_ids = [i for i in tok.encode(' ' + fake).ids if i != pad_id]
        e_real.append(entropy_after(model, char_table, pad_id, it['ctx_ids'], gold_ids, device))
        e_fake.append(entropy_after(model, char_table, pad_id, it['ctx_ids'], fake_ids, device))
    ent_real, ent_fake = (float(np.mean(e_real)), float(np.mean(e_fake)))
    g1 = next_tok >= base_next - 0.03
    g2 = s_hold > s_train
    g3 = ent_fake > ent_real
    overall = 'SELF_MODEL_YES' if g1 and g2 and g3 else 'SELF_MODEL_PARTIAL_' + ''.join((n for n, ok in (('1', g1), ('2', g2), ('3', g3)) if not ok))
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'self_model_surprise_187', 'overall': overall, 'gates': {'G1_ce_preserved': {'next_tok': next_tok, 'baseline': base_next, 'ok': g1}, 'G2_novelty': {'surprise_seen_train': s_train, 'surprise_unseen_hold': s_hold, 'ok': g2}, 'G3_calibration': {'entropy_after_real': ent_real, 'entropy_after_fake': ent_fake, 'ok': g3}}, 'exam_full': full, 'k_final': float(model.slow.k), 'steps': args.steps, 'note': 'aux loss = predictor only (detached target); no representation pushing'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage187 — self-model (surprise-gated writes)', '', f'**Overall:** `{overall}`', '', f'- G1 CE preserved: next_tok={next_tok:.3f} vs baseline {base_next:.3f} → {g1}', f'- G2 novelty: surprise seen={s_train:.4f} vs unseen={s_hold:.4f} → {g2}', f'- G3 calibration: entropy real={ent_real:.3f} vs fake={ent_fake:.3f} → {g3}', f"- entity={full.get('entity_acc', 0):.3f} ood={full.get('ood_acc', 0):.3f} k={float(model.slow.k):.2f}", '']), encoding='utf-8')
    log(f'[187] {overall} | G1 {next_tok:.3f}/{base_next:.3f} | G2 {s_train:.4f}<{s_hold:.4f}? | G3 {ent_real:.3f}<{ent_fake:.3f}?')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())