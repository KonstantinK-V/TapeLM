"""
Stage 216 — frozen char emb+pool; train FF (linear vs GELU control).

  python _stage216_split_arc_ff.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage213_arc_enc_freeze_finetune as s213
from _stage191_night import LR, MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows
from _stage194_fp_fact_memory import FpBank
RES = Path('results')
CKPT = Path('checkpoints/stage191_p1_curve.pt')
DOMAIN = Path('data/external_tinystories_100k_85.txt')
DECISION = RES / 'stage216_decision.json'
PROBE = ['the', 'cat', 'London', 'Elizabeth', 'running', 'quantum']
STEPS = 600

def fp_drift(bank: FpBank, ref: dict, words: list[str]) -> float:
    fps = bank.fp(words)
    return float(max((1 - float((ref[w] * fps[i]).sum()) for i, w in enumerate(words))))

def train_ff_only(model, mode: str, flat, off, char_table, pad_id, device, steps, rng):
    for p in model.parameters():
        p.requires_grad = False
    ae = model.arc_enc
    ae.emb.eval()
    for p in ae.emb.parameters():
        p.requires_grad = False
    if mode == 'linear':
        d = ae.ff[0].out_features
        ae.ff = nn.Sequential(nn.Linear(d, d), nn.Linear(d, d)).to(device)
    for p in ae.ff.parameters():
        p.requires_grad = True
    ae.ff.train()
    opt = torch.optim.AdamW(ae.ff.parameters(), lr=LR * 0.3)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        pad = ids == pad_id
        logits, _, pred_loss = model.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ae.eval()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(216)
    steps = 80 if args.smoke else STEPS
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    import _stage185_tape_read as s185
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    text_b = DOMAIN.read_text(encoding='utf-8', errors='ignore')
    flat_b, off_b = s213.build_flat_from_text(text_b, tok, pad_id, max_lines=300 if args.smoke else 5000)
    base = SelfModelXL(n_char, V).to(device)
    base.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    base.eval()
    ref = {w: FpBank(base, stoi, device).fp([w])[0].clone() for w in PROBE}
    m_lin = copy.deepcopy(base)
    train_ff_only(m_lin, 'linear', flat_b, off_b, char_table, pad_id, device, steps, rng)
    drift_lin = fp_drift(FpBank(m_lin, stoi, device), ref, PROBE)
    m_gelu = copy.deepcopy(base)
    train_ff_only(m_gelu, 'gelu', flat_b, off_b, char_table, pad_id, device, steps, rng)
    drift_gelu = fp_drift(FpBank(m_gelu, stoi, device), ref, PROBE)
    cos_lin = 1 - drift_lin
    cos_gelu = 1 - drift_gelu
    g1 = cos_lin > 0.95 and cos_gelu < cos_lin - 0.05
    g2 = cos_lin > 0.8
    overall = 'SPLIT_FF_LINEAR_WINS' if g1 else 'SPLIT_FF_PARTIAL' if cos_lin > cos_gelu else 'SPLIT_FF_NO'
    DECISION.write_text(json.dumps({'stage': 216, 'overall': overall, 'gates': {'G1_linear_vs_gelu': g1, 'G2_linear_recall_geom': g2}, 'cos_linear_min': cos_lin, 'cos_gelu_min': cos_gelu, 'timestamp': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
    print(f'216 {overall} cos_lin~{cos_lin:.3f} cos_gelu~{cos_gelu:.3f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())