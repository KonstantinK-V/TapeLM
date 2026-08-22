"""
Stage 221 — fp-remap adapter: arc_enc shift + tiny W on core vocab.

Protocol:
  1. fp_old = frozen P1 arc_enc on core vocab + old fact bank keys.
  2. Finetune arc_enc only on domain B (control shift) -> fp_new.
  3. Train W (d×d, optional bottleneck) so normalize(W @ fp_old) ≈ fp_new on core words.
  4. Gates:
     G_align  mean cos(W fp_old, fp_new) >= 0.85 on core vocab
     G_recall recall old facts with W @ key_old >= 0.80 * oracle reindex recall

  python _stage221_fp_remap_adapter.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
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
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import BottleneckRemap, DomainAdapter
RES = Path('results')
CKPT = Path('checkpoints/stage191_p1_curve.pt')
DOMAIN = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
DECISION = RES / 'stage221_decision.json'
MINI = RES / 'stage221_mini.md'
SEED = 221
CORE_N = 800
ARC_STEPS = 800
W_STEPS = 1200
BOTTLENECK_R = 32

def log(m: str) -> None:
    print(m, flush=True)

@torch.no_grad()
def fp_matrix(bank: FpBank, words: list[str]) -> torch.Tensor:
    return bank.fp(words)

def build_fact_bank(bank: FpBank, subjects, values, rng):
    keys, vals = ([], [])
    for S, v in zip(subjects, values):
        ctx = f'Official records state {S} was director of {v} in 1987 .'
        k = bank.fp([S])[0]
        c = bank.ctx_fp(ctx, exclude=v)
        if c is None:
            continue
        keys.append(F.normalize(k + c, dim=-1))
        vals.append(v)
    return (torch.stack(keys, 0), vals)

def recall_at(K: torch.Tensor, V: list[str], bank: FpBank, subjects, values, rng, key_transform=None):
    ok, n = (0, 0)
    for S, gold in zip(subjects, values):
        q = bank.ctx_fp(f'In the report {S} was linked to the organization.', exclude=gold)
        if q is None:
            continue
        if key_transform is not None:
            Kq = key_transform(K)
            qq = key_transform(q.unsqueeze(0))[0]
        else:
            Kq, qq = (K, q)
        cands = [gold] + [values[(i + 1) % len(values)] for i in range(3)]
        rng.shuffle(cands)
        g = cands.index(gold)
        sc = []
        for c in cands:
            idxs = [i for i, v in enumerate(V) if v == c]
            sc.append(float((Kq[idxs] @ qq).max()) if idxs else -1.0)
        ok += int(np.argmax(sc) == g)
        n += 1
    return (ok / max(1, n), n)

def train_remap(module: nn.Module, F_old, F_new, rng, steps: int, device, orth: bool=True):
    opt = torch.optim.AdamW(module.parameters(), lr=0.002)
    n = len(F_old)
    for step in range(1, steps + 1):
        idx = rng.sample(range(n), min(64, n))
        fo, fn = (F_old[idx], F_new[idx])
        pred = module(fo)
        loss = (1.0 - (pred * fn).sum(-1)).mean()
        if orth and isinstance(module, DomainAdapter):
            loss = loss + 0.02 * (module.w.weight @ module.w.weight.T - torch.eye(256, device=device)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        align = float((module(F_old) * F_new).sum(-1).mean())
    return (module, align)

def finetune_arc_enc(model: SelfModelXL, flat, off, char_table, pad_id: int, device: torch.device, steps: int, seed: int) -> SelfModelXL:
    """In-place arc_enc-only finetune (domain shift control)."""
    m = copy.deepcopy(model)
    s213.set_train_mode(m, 'arc_enc')
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR * 0.5)
    r2 = random.Random(seed)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, r2, pad_id).to(device)
        pad = ids == pad_id
        logits, _, pred_loss = m.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    m.eval()
    return m

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    arc_steps = 80 if args.smoke else ARC_STEPS
    w_steps = 100 if args.smoke else W_STEPS
    core_n = 80 if args.smoke else CORE_N
    n_facts = 15 if args.smoke else 60
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    import _stage185_tape_read as s185
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(2000000)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    model_old = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model_old.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model_old.eval()
    bank_old = FpBank(model_old, stoi, device)
    F_old = fp_matrix(bank_old, core)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(4000000)) if len(m.group(1)) >= 5)))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_old, V = build_fact_bank(bank_old, subs, vals, rng)
    acc_no_w, _ = recall_at(K_old, V, bank_old, subs, vals, rng, None)
    log(f'baseline recall (old fp, old bank) acc={acc_no_w:.3f}')
    text_b = DOMAIN.read_text(encoding='utf-8', errors='ignore')
    flat_b, off_b = s213.build_flat_from_text(text_b, tok, pad_id, max_lines=500 if args.smoke else 8000)
    model_new = finetune_arc_enc(model_old, flat_b, off_b, char_table, pad_id, device, arc_steps, SEED + 1)
    bank_new = FpBank(model_new, stoi, device)
    F_new = fp_matrix(bank_new, core)
    align_raw = float((F_old * F_new).sum(-1).mean())
    log(f'after arc_enc shift: mean cos(fp_old,fp_new) on core = {align_raw:.3f}')
    K_oracle, _ = build_fact_bank(bank_new, subs, vals, rng)
    acc_oracle, _ = recall_at(K_oracle, V, bank_new, subs, vals, rng, None)
    log(f'oracle reindex (new bank new fp) acc={acc_oracle:.3f}')
    Wmap, align_full = train_remap(DomainAdapter(256).to(device), F_old, F_new, rng, w_steps, device)

    def transform_full(K):
        return F.normalize(Wmap.map_raw(K), dim=-1)
    acc_full, _ = recall_at(K_old, V, bank_old, subs, vals, rng, transform_full)
    log(f'full 256x256: align={align_full:.3f} recall={acc_full:.3f}')
    r = 16 if args.smoke else BOTTLENECK_R
    Bmap, align_bot = train_remap(BottleneckRemap(256, r).to(device), F_old, F_new, rng, w_steps, device, orth=False)

    def transform_bot(K):
        return F.normalize(Bmap.map_raw(K), dim=-1)
    acc_bot, _ = recall_at(K_old, V, bank_old, subs, vals, rng, transform_bot)
    log(f'bottleneck d-{r}-d: align={align_bot:.3f} recall={acc_bot:.3f} params={r * 256 * 2}')
    acc_w = max(acc_full, acc_bot)
    align_w = align_full if acc_full >= acc_bot else align_bot
    best = 'full' if acc_full >= acc_bot else f'bottleneck_r{r}'
    g_align = align_w >= 0.85
    g_recall = acc_w >= 0.8 * max(acc_oracle, 0.01)
    overall = 'FP_REMAP_ADAPTER_YES' if g_align and g_recall else 'FP_REMAP_PARTIAL' if acc_w > acc_no_w + 0.05 else 'FP_REMAP_NO'
    out = {'stage': 221, 'overall': overall, 'gates': {'G_align_core': g_align, 'G_recall_W_keys': g_recall}, 'core_vocab_n': len(core), 'mean_cos_before_shift': align_raw, 'mean_cos_after_W': align_w, 'recall_old_fp': acc_no_w, 'recall_oracle_new': acc_oracle, 'recall_W_remapped': acc_w, 'remap_full': {'align': align_full, 'recall': acc_full, 'params': 256 * 256}, 'remap_bottleneck': {'r': r, 'align': align_bot, 'recall': acc_bot, 'params': r * 256 * 2}, 'best_remap': best, 'note': 'arc_enc domain shift then W on core vocab; old slots use W @ key_old', 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage221 fp-remap\n\n**{overall}** align_W={align_w:.3f} recall W={acc_w:.3f} oracle={acc_oracle:.3f}\n', encoding='utf-8')
    log(f'VERDICT {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())