"""
Stage 199 — semantic invariance via consequence-prediction, WITHOUT breaking the stack.

Goal B (open since 179): representation groups by MEANING not spelling — paraphrases
close, surface-similar-different-meaning (car/cat) far. Route 1 (cheapest, no new data):
"meaning = what comes next". Train a contrastive predictive head so a window's embedding
predicts its own continuation vs other continuations (CPC). Paraphrases predict similar
futures -> pulled together; car/cat predict different futures -> pushed apart.

NON-DESTRUCTION (user constraint): P1 encoder stays FROZEN. Only a separate semantic
head z_sem is trained on top. Generation / FP-memory / calibration never touch z_sem, so
they cannot regress — verified by a parity regression gate (next_tok unchanged).

SCALABILITY (user constraint): train the SAME head at 3 data budgets (5% / 25% / 100% of
tokens), fixed steps. Monotone improvement of the semantic gap = evidence it scales;
recipe is size-invariant.

Metrics on 179 pairs (cosine):
  para_sim (want HIGH), hard_sim (want LOW). inversion = para_sim > hard_sim (true semantic win).
  baseline = raw frozen fast mean-pool (191b regime: hard≈0.89 > para≈0.71).

Gates:
  G_nondestruct : next_tok parity preserved (~0.867; encoder frozen by construction)
  G_semantic    : at full budget para_sim > hard_sim (INVERSION)   -> SEM_INV_YES
                  else head shrinks (hard-para) gap vs raw baseline -> SEM_INV_TREND
  G_scale       : (hard-para) gap decreases monotonically across budgets

  python _stage199_semantic_invariance.py
"""
from __future__ import annotations
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
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data, score_items, span_logprob_x
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
EXAM_V3 = Path('data/stage191_exam_v3.jsonl')
DECISION = RES / 'stage199_decision.json'
MINI = RES / 'stage199_mini.md'
LOG = RES / '_stage199_log.txt'
SEED = 199
L = 32
D_SEM = 128
TEMP = 0.07
STEPS = 1200
BATCH = 128
LR = 0.0003
BUDGETS = [0.05, 0.25, 1.0]
MAX_ARCS = 64

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

class SemHead(nn.Module):

    def __init__(self, d, d_sem=D_SEM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_sem))

    def forward(self, pooled):
        return F.normalize(self.net(pooled), dim=-1)

class Predictor(nn.Module):

    def __init__(self, d_sem=D_SEM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_sem, d_sem), nn.GELU(), nn.Linear(d_sem, d_sem))

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage199 start {datetime.now(timezone.utc).isoformat()}')
    log('semantic invariance via CPC on a FROZEN encoder + scale trend')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.head.in_features // 2
    log(f'encoder frozen (fast dim={d}) ({time.time() - t0:.0f}s)')

    @torch.no_grad()
    def pooled_fast(idb: torch.Tensor) -> torch.Tensor:
        pad = idb == pad_id
        arcs = model._arcs(char_table[idb], idb)
        fast = model.fast(arcs, pad_mask=pad)
        m = (~pad).float().unsqueeze(-1)
        return (fast * m).sum(1) / m.sum(1).clamp(min=1.0)

    @torch.no_grad()
    def pooled_text(t: str) -> torch.Tensor:
        ids = [i for i in tok.encode(t).ids if i != pad_id][:MAX_ARCS]
        x = torch.tensor([ids], device=device)
        return pooled_fast(x)[0]
    total = len(flat)

    def eligible_docs(budget_tokens):
        return [dd for dd in range(len(off) - 1) if off[dd + 1] <= budget_tokens and off[dd + 1] - off[dd] >= 2 * L]

    def sampler(docs, rng):

        def draw():
            xa = np.full((BATCH, L), pad_id, np.int64)
            xb = np.full((BATCH, L), pad_id, np.int64)
            for b in range(BATCH):
                dd = docs[rng.randint(0, len(docs) - 1)]
                s, e = (off[dd], off[dd + 1])
                st = s + rng.randint(0, e - s - 2 * L)
                xa[b] = flat[st:st + L]
                xb[b] = flat[st + L:st + 2 * L]
            return (torch.from_numpy(xa).to(device), torch.from_numpy(xb).to(device))
        return draw

    def measure_B(head: SemHead, raw=False):

        def z(t):
            p = pooled_text(t)
            return F.normalize(p, dim=-1) if raw else head(p.unsqueeze(0))[0]
        para = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.PARAPHRASE_PAIRS]))
        hard = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.HARD_PAIRS]))
        return {'para': para, 'hard': hard, 'gap_hard_minus_para': hard - para, 'inversion': para > hard}
    raw_base = measure_B(None, raw=True)
    log(f"raw frozen fast baseline: para={raw_base['para']:.3f} hard={raw_base['hard']:.3f} gap={raw_base['gap_hard_minus_para']:+.3f}")
    trend = {}
    heads = {}
    for frac in BUDGETS:
        budget_tokens = int(total * frac)
        docs = eligible_docs(budget_tokens)
        rng = random.Random(SEED)
        head = SemHead(d).to(device)
        pred = Predictor().to(device)
        opt = torch.optim.AdamW(list(head.parameters()) + list(pred.parameters()), lr=LR, weight_decay=0.01)
        draw = sampler(docs, rng)
        head.train()
        pred.train()
        running = None
        for step in range(1, STEPS + 1):
            xa, xb = draw()
            za = head(pooled_fast(xa))
            zb = head(pooled_fast(xb))
            pa = pred(za)
            logits = pa @ zb.T / TEMP
            labels = torch.arange(xa.size(0), device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running = float(loss) if running is None else 0.97 * running + 0.03 * float(loss)
        head.eval()
        b = measure_B(head)
        trend[f'{frac:.2f}'] = {'docs': len(docs), 'budget_tokens': budget_tokens, 'cpc_loss': running, **b}
        heads[frac] = head
        log(f"  budget {frac:.2f} ({len(docs)} docs): para={b['para']:.3f} hard={b['hard']:.3f} gap={b['gap_hard_minus_para']:+.3f} inv={b['inversion']} cpc~{running:.3f} ({time.time() - t0:.0f}s)")
    full = trend[f'{BUDGETS[-1]:.2f}']
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding='utf-8').splitlines()]
    nt = [it for it in items if it['type'] == 'next_tok'][:120]
    parity = score_items(lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), nt, 'next_tok')['next_tok_acc']
    log(f'non-destruct: next_tok(frozen)={parity:.3f} (expected ~0.867; head is a separate branch)')
    gaps = [trend[f'{f:.2f}']['gap_hard_minus_para'] for f in BUDGETS]
    g_scale = all((gaps[i + 1] <= gaps[i] + 1e-06 for i in range(len(gaps) - 1)))
    g_nondestruct = parity >= 0.8
    inversion_full = full['inversion']
    head_helps = full['gap_hard_minus_para'] < raw_base['gap_hard_minus_para'] - 0.02
    if g_nondestruct and inversion_full:
        overall = 'SEM_INV_YES'
    elif g_nondestruct and head_helps and g_scale:
        overall = 'SEM_INV_TREND'
    elif g_nondestruct and head_helps:
        overall = 'SEM_INV_PARTIAL'
    else:
        overall = 'SEM_INV_NO'
    gates = {'g_nondestruct': g_nondestruct, 'g_scale_monotone': g_scale, 'inversion_at_full': inversion_full, 'head_beats_raw': head_helps}
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'semantic_invariance_199', 'overall': overall, 'gates': gates, 'raw_frozen_baseline': raw_base, 'scale_trend': trend, 'next_tok_parity_frozen': parity, 'note': 'frozen P1 encoder + separate CPC-trained semantic head; consequence-prediction route to B; scale probe = same head at 5/25/100% token budgets'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage199 — semantic invariance (CPC on frozen encoder)', '', f'**Overall:** `{overall}`', '', f"- raw frozen baseline: para {raw_base['para']:.3f} / hard {raw_base['hard']:.3f} (gap {raw_base['gap_hard_minus_para']:+.3f})", '- scale trend (para / hard / gap hard−para):', *[f"  - budget {f:.2f}: {trend[f'{f:.2f}']['para']:.3f} / {trend[f'{f:.2f}']['hard']:.3f} / {trend[f'{f:.2f}']['gap_hard_minus_para']:+.3f} (inv={trend[f'{f:.2f}']['inversion']})" for f in BUDGETS], '', f'- non-destruct: next_tok(frozen) = {parity:.3f} (generation/memory/calib untouched)', f'- gates: {gates}']), encoding='utf-8')
    log(f"[199] {overall} | full para={full['para']:.3f} hard={full['hard']:.3f} gap={full['gap_hard_minus_para']:+.3f} parity={parity:.3f}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())