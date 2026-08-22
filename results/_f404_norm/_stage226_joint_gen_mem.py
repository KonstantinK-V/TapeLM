"""
Stage 226 — Joint generation + memory (prose slots while generating in code domain).

Requires 227-style canonical bank. Protocol:
  - Write facts with frozen arc_enc (canonical keys).
  - Code domain: head_code (frozen arc_enc upper); retrieval uses W_code **qmap** (227).
  - Inject retrieved value into code-shaped prompt; 4-way rank of gold among distractors.

Gates:
  G_retrieve  cross-domain recall (canonical + W_code) >= 0.70
  G_joint     gen accuracy with gold inject >= baseline without inject + margin
              OR with retrieved inject >= 0.5 * gold-inject (retrieval useful)

  python _stage226_joint_gen_mem.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage225_family_fork as s225
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
RES = Path('results')
DECISION = RES / 'stage226_decision.json'
MINI = RES / 'stage226_mini.md'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
STORIES = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 226

def log(m: str) -> None:
    print(m, flush=True)

def w_apply(W):
    return lambda X: F.normalize(W.map_raw(X), dim=-1)

def retrieve_one(K, V, q, key_x):
    Kq = key_x(K)
    scores = Kq @ q
    i = int(scores.argmax())
    return (V[i], float(scores[i]))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    upper_steps = 80 if args.smoke else 600
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else s221.CORE_N
    n_facts = 12 if args.smoke else 40
    max_lines = 400 if args.smoke else 8000
    rng = random.Random(SEED)
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(2000000)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    model0 = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model0.eval()
    bank0 = FpBank(model0, stoi, device)
    F0 = s221.fp_matrix(bank0, core)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(4000000)) if len(m.group(1)) >= 5)))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_can, V = s221.build_fact_bank(bank0, subs, vals, rng)
    text_code = s225.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    model_c = s221.finetune_arc_enc(model0, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 2)
    bank_c = FpBank(model_c, stoi, device)
    F_c = s221.fp_matrix(bank_c, core)
    W_c, align = s221.train_remap(DomainAdapter(256).to(device), F0, F_c, rng, w_steps, device)
    W_c_bwd, align_bwd = s221.train_remap(DomainAdapter(256).to(device), F_c, F0, rng, w_steps, device)
    ok = n = 0
    retrieved = []
    for S, gold in zip(subs, vals):
        q = bank_c.ctx_fp(f'In the report {S} was linked to the organization.', exclude=gold)
        if q is None:
            continue
        qq = w_apply(W_c_bwd)(q.unsqueeze(0))[0]
        hit, sc = retrieve_one(K_can, V, qq, key_x=lambda K: K)
        ok += int(hit == gold)
        n += 1
        retrieved.append((S, gold, hit))
    recall = ok / max(1, n)
    head_code = s225.train_upper(model0, flat_c, off_c, char_table, pad_id, device, upper_steps, SEED + 3)
    ret_exact = sum((1 for _, g, h in retrieved if h == g)) / max(1, len(retrieved))

    def rank_gold_after_inject(model, injects: list[tuple[str, str, str]]) -> float:
        """(ctx_prefix, inject_value, gold) → fraction where gold scores highest among {gold}+3 distractors."""
        ok = n = 0
        for ctx, inj, gold in injects:
            text = f'# memory: {inj}\ndef label():\n    return '
            ids = tok.encode(text).ids
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            pad = x == pad_id
            logits, _, _ = model.forward_all(char_table[x], pad, ids=x)
            last = logits[0, -1]
            cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
            rng.shuffle(cands)
            scores = []
            for c in cands:
                cid = tok.encode(c).ids
                scores.append(float(last[cid[0]]) if cid else -1000000000.0)
            ok += int(cands[int(np.argmax(scores))] == gold)
            n += 1
        return ok / max(1, n)
    injects_gold = [(S, gold, gold) for S, gold, _ in retrieved]
    injects_ret = [(S, hit, gold) for S, gold, hit in retrieved]
    injects_none = [(S, 'UNKNOWN', gold) for S, gold, _ in retrieved]
    acc_none = rank_gold_after_inject(head_code, injects_none)
    acc_gold = rank_gold_after_inject(head_code, injects_gold)
    acc_ret = rank_gold_after_inject(head_code, injects_ret)
    g_ret = recall >= 0.7
    g_joint = acc_gold >= acc_none + 0.05 or acc_ret >= acc_none + 0.05
    overall = 'JOINT_GEN_MEM_OK' if g_ret and g_joint else 'JOINT_GEN_MEM_PARTIAL' if g_ret or g_joint else 'JOINT_GEN_MEM_NO'
    out = {'stage': 226, 'overall': overall, 'gates': {'G_retrieve': g_ret, 'G_joint': g_joint}, 'align_W_code_fwd': align, 'align_W_code_bwd_qmap': align_bwd, 'mean_cos_code_shift': float((F0 * F_c).sum(-1).mean()), 'recall_canonical_W_code_qmap': recall, 'ret_exact_rate': ret_exact, 'gen_rank4': {'no_inject': acc_none, 'gold_inject': acc_gold, 'retrieved_inject': acc_ret}, 'n_items': n, 'note': 'Canonical slots + code query via W_code qmap (227) + head_code; joint=4-way rank after inject', 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 226 joint gen+mem\n\n**{overall}** recall={recall:.3f} gen none/gold/ret={acc_none:.3f}/{acc_gold:.3f}/{acc_ret:.3f}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())