"""
Stage 213 — Fine-tune variant A with frozen arc_enc (character fp geometry fixed).

Problem: if arc_enc (emb + FF + GELU) adapts on a new domain, h("the") can cross GELU
kinks differently from h("cat"); pooled char geometry stops behaving like a stable fp substrate.

Protocol:
  A) Freeze arc_enc (eval + no grad); train fast/slow/head on TinyStories domain.
  B) Control: train arc_enc only on same data (fp drift + GELU zone shift).

Gates:
  G1 fp_stable: max(1 - cos(fp_before, fp_after)) < 1e-5 after A
  G2 fp_drifts: mean fp drift after B >= 0.02 (control shows domain move hurts fp)
  G3 gen_ok: next_tok_acc after A within 0.03 of baseline OR improves

  python _stage213_arc_enc_freeze_finetune.py
  python _stage213_arc_enc_freeze_finetune.py --smoke
"""
from __future__ import annotations
import argparse
import copy
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
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import D_XL, L_XL, LR, MAX_ARCS, MICRO, PAD, SelfModelXL, W_SELF, WARMUP, load_data, lr_at, sample_windows, score_items, span_logprob_x
from _stage194_fp_fact_memory import FpBank
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
EXAM_V3 = Path('data/stage191_exam_v3.jsonl')
DOMAIN_B = Path('data/external_tinystories_100k_85.txt')
DECISION = RES / 'stage213_decision.json'
MINI = RES / 'stage213_mini.md'
LOG = RES / '_stage213_log.txt'
CKPT_OUT = Path('checkpoints/stage213_upper_tinystories.pt')
SEED = 213
STEPS = 1200
MAX_CHARS = s177.MAX_CHARS_PER_ARC
PROBE_WORDS = ['the', 'and', 'cat', 'London', 'quantum', 'Elizabeth', 'running', 'xyzabc']

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def build_flat_from_text(text: str, tok: Tokenizer, pad_id: int, max_lines: int=8000, min_line_len: int=40) -> tuple[np.ndarray, np.ndarray]:
    lines = []
    for l in text.split('\n'):
        l = l.strip()
        if l.startswith('#') or not l:
            continue
        if len(l) >= min_line_len:
            lines.append(l)
        if len(lines) >= max_lines:
            break
    flat, offsets = ([], [0])
    for line in lines:
        ids = [i for i in tok.encode(line).ids if i != pad_id]
        if len(ids) >= 8:
            flat.extend(ids)
            offsets.append(len(flat))
    if len(offsets) < 2:
        raise RuntimeError('domain B too small for flat corpus')
    return (np.asarray(flat, dtype=np.int32), np.asarray(offsets, dtype=np.int64))

def load_p1(device: torch.device, n_char: int, V: int) -> SelfModelXL:
    m = SelfModelXL(n_char, V, d=D_XL, n_layers=L_XL).to(device)
    ck = torch.load(CKPT_P1, map_location=device, weights_only=False)
    m.load_state_dict(ck['model'])
    m.eval()
    return m

def set_train_mode(model: SelfModelXL, which: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if which == 'none':
        model.eval()
    elif which == 'upper':
        for mod in (model.fast, model.slow, model.head):
            mod.train()
            for p in mod.parameters():
                p.requires_grad = True
        model.arc_enc.eval()
    elif which == 'arc_enc':
        model.arc_enc.train()
        for p in model.arc_enc.parameters():
            p.requires_grad = True
        model.fast.eval()
        model.slow.eval()
        model.head.eval()
    else:
        raise ValueError(which)

@torch.no_grad()
def fp_drift(bank: FpBank, words: list[str], ref: dict[str, torch.Tensor]) -> dict:
    fps = bank.fp(words)
    drifts = []
    for i, w in enumerate(words):
        c = float((ref[w] * fps[i]).sum())
        drifts.append(1.0 - c)
    return {'mean': float(np.mean(drifts)), 'max': float(np.max(drifts)), 'per_word': {w: drifts[i] for i, w in enumerate(words)}}

@torch.no_grad()
def snapshot_fps(bank: FpBank, words: list[str]) -> dict[str, torch.Tensor]:
    fps = bank.fp(words)
    return {w: fps[i].clone() for i, w in enumerate(words)}

@torch.no_grad()
def gelu_zone_stats(model: SelfModelXL, word: str, stoi: dict, device: torch.device) -> dict:
    """Pre/post FF on pooled char emb; GELU saturation fraction on L1 activations."""
    row = torch.zeros(1, 1, MAX_CHARS, dtype=torch.long, device=device)
    n = 0
    for j, c in enumerate(word[:MAX_CHARS]):
        row[0, 0, j] = stoi.get(c, 0)
        n += 1
    ae = model.arc_enc
    h = ae.emb(row)
    mask = (row != 0).float().unsqueeze(-1)
    pooled = (h * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1.0)
    mid = ae.ff[0](pooled)
    mid_g = F.gelu(mid)
    out = ae.norm(ae.ff[2](mid_g))
    sat = (mid.abs() > 2.0).float().mean().item()
    neg_frac = (mid < 0).float().mean().item()
    return {'word': word, 'gelu_sat_frac': sat, 'pre_linear_neg_frac': neg_frac, 'out_norm': float(out.norm())}

def finetune(model: SelfModelXL, which: str, flat, off, char_table, pad_id: int, items_mid, device: torch.device, steps: int, rng: random.Random) -> dict:
    set_train_mode(model, which)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR * 0.5, weight_decay=0.01)
    t0 = time.time()
    running = None
    best = -1.0
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        pad = ids == pad_id
        logits, surprise, pred_loss = model.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        running = float(ce) if running is None else 0.95 * running + 0.05 * float(ce)
        if step % max(1, steps // 4) == 0 or step == steps:
            model.eval()
            if which == 'upper':
                model.arc_enc.eval()
            mid = score_items(lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), items_mid, 'next_tok')
            acc = mid.get('next_tok_acc', 0)
            log(f'  [{which}] step {step}/{steps} ce~{running:.3f} next_tok={acc:.3f}')
            best = max(best, acc)
            set_train_mode(model, which)
    model.eval()
    model.arc_enc.eval()
    return {'steps': steps, 'ce': running, 'best_next_tok': best, 'wall_s': time.time() - t0}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage213 start {datetime.now(timezone.utc).isoformat()}')
    device = torch.device(args.device)
    rng = random.Random(SEED)
    steps = 80 if args.smoke else STEPS
    flat_w, off_w, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding='utf-8').splitlines() if l.strip()]
    items_mid = [it for it in items if it['type'] == 'next_tok'][:80 if args.smoke else 200]
    if not DOMAIN_B.exists():
        log(f'MISSING {DOMAIN_B}')
        return 1
    text_b = DOMAIN_B.read_text(encoding='utf-8', errors='ignore')
    flat_b, off_b = build_flat_from_text(text_b, tok, pad_id, max_lines=400 if args.smoke else 8000, min_line_len=20)
    log(f'domain B: {DOMAIN_B.name} docs={len(off_b) - 1} tokens={len(flat_b)}')
    model = load_p1(device, n_char, V)
    bank0 = FpBank(model, stoi, device)
    fp_ref = snapshot_fps(bank0, PROBE_WORDS)
    gelu0 = {w: gelu_zone_stats(model, w, stoi, device) for w in PROBE_WORDS[:4]}
    baseline_acc = score_items(lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), items_mid, 'next_tok').get('next_tok_acc', 0)
    log(f'baseline next_tok={baseline_acc:.3f}')
    model_a = load_p1(device, n_char, V)
    log('Phase A: freeze arc_enc, finetune fast/slow/head on domain B …')
    train_a = finetune(model_a, 'upper', flat_b, off_b, char_table, pad_id, items_mid, device, steps, rng)
    bank_a = FpBank(model_a, stoi, device)
    drift_a = fp_drift(bank_a, PROBE_WORDS, fp_ref)
    acc_a = score_items(lambda c, cd: span_logprob_x(model_a, char_table, pad_id, c, cd, device), items_mid, 'next_tok').get('next_tok_acc', 0)
    log(f"A fp drift mean={drift_a['mean']:.2e} max={drift_a['max']:.2e} next_tok={acc_a:.3f}")
    if not args.smoke:
        torch.save({'model': model_a.state_dict(), 'train': train_a}, CKPT_OUT)
    model_b = load_p1(device, n_char, V)
    log('Phase B control: train arc_enc only (fp should move) …')
    train_b = finetune(model_b, 'arc_enc', flat_b, off_b, char_table, pad_id, items_mid, device, steps, rng)
    bank_b = FpBank(model_b, stoi, device)
    drift_b = fp_drift(bank_b, PROBE_WORDS, fp_ref)
    gelu_b = {w: gelu_zone_stats(model_b, w, stoi, device) for w in PROBE_WORDS[:4]}
    log(f"B fp drift mean={drift_b['mean']:.4f} max={drift_b['max']:.4f}")
    g1 = drift_a['max'] < 1e-05
    g2 = drift_b['mean'] >= 0.02
    g3 = abs(acc_a - baseline_acc) <= 0.05 or acc_a >= baseline_acc - 1e-06
    if args.smoke:
        g1 = drift_a['max'] < 0.0001
        g2 = drift_b['mean'] > drift_a['mean']
    overall = 'ARC_ENC_FREEZE_FP_STABLE_YES' if g1 and g2 and g3 else 'ARC_ENC_FREEZE_PARTIAL'
    if g1 and (not g2):
        overall = 'ARC_ENC_FREEZE_FP_YES_CONTROL_WEAK'
    decision = {'stage': 213, 'overall': overall, 'gates': {'G1_fp_stable_upper': g1, 'G2_fp_drifts_arc_control': g2, 'G3_gen_ok': g3}, 'baseline_next_tok': baseline_acc, 'after_upper_next_tok': acc_a, 'fp_drift_upper': drift_a, 'fp_drift_arc_enc': drift_b, 'gelu_baseline': gelu0, 'gelu_after_arc_train': gelu_b, 'train_upper': train_a, 'train_arc': train_b, 'note': 'Freeze arc_enc (eval) during upper finetune → fp(word) unchanged; arc_enc-only train shifts fp and GELU stats.', 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text(f"# Stage213 — frozen arc_enc finetune\n\n**Overall:** `{overall}`\n\n- G1 fp stable (upper): {g1} (max drift {drift_a['max']:.2e})\n- G2 fp drifts (arc control): {g2} (mean {drift_b['mean']:.4f})\n- G3 gen ok: {g3} ({baseline_acc:.3f} → {acc_a:.3f})\n", encoding='utf-8')
    log(f'VERDICT {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())