"""
Stage 228a — Counterfactual inject probe (226 boundary).

Tests whether head_code *follows* injected values vs lexical prior only:
  - none / gold_inject / wrong_inject (code-comment form)
  - 4-way rank of gold
  - span logprob of gold string
  - sensitivity: P(gold|gold_inject) - P(gold|wrong_inject)

  python _stage228a_counterfactual_inject.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
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
from _stage194_fp_fact_memory import ENT_RE
RES = Path('results')
DECISION = RES / 'stage228a_decision.json'
MINI = RES / 'stage228a_mini.md'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 2281

def log(m: str) -> None:
    print(m, flush=True)

def prompt_code_comment(S: str, inject_val: str | None) -> str:
    if inject_val is None:
        return f'# TODO\ndef org_of_{S}():\n    return '
    return f'# org[{S}] = {inject_val}\ndef org_of_{S}():\n    return '

@torch.no_grad()
def argmax_token(model, tok, pad_id, char_table, device, text: str) -> int:
    ids = tok.encode(text).ids
    if not ids:
        return -1
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pad = x == pad_id
    logits, _, _ = model.forward_all(char_table[x], pad, ids=x)
    return int(logits[0, -1].argmax())

@torch.no_grad()
def span_logprob(model, tok, pad_id, char_table, device, prefix: str, target: str) -> float:
    """Logprob of target token ids continuing prefix (teacher-forced on target)."""
    pids = tok.encode(prefix).ids
    tids = tok.encode(target).ids
    if not pids or not tids:
        return float('-inf')
    full = pids + tids
    x = torch.tensor([full], dtype=torch.long, device=device)
    pad = x == pad_id
    logits, _, _ = model.forward_all(char_table[x], pad, ids=x)
    lp = 0.0
    for i, tid in enumerate(tids):
        pos = len(pids) - 1 + i
        if pos < 0 or pos >= logits.shape[1]:
            break
        logp = F.log_softmax(logits[0, pos], dim=-1)[tid]
        lp += float(logp)
    return lp / max(1, len(tids))

@torch.no_grad()
def rank4_last(model, tok, pad_id, char_table, device, text: str, gold: str, pool: list[str], rng) -> bool:
    ids = tok.encode(text).ids
    if not ids:
        return False
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pad = x == pad_id
    logits, _, _ = model.forward_all(char_table[x], pad, ids=x)
    last = logits[0, -1]
    cands = [gold] + [p for p in pool if p != gold][:3]
    while len(cands) < 4:
        cands.append(pool[0])
    rng.shuffle(cands)
    scores = [float(last[tok.encode(c).ids[0]]) if tok.encode(c).ids else -1000000000.0 for c in cands]
    return cands[int(np.argmax(scores))] == gold

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    upper_steps = 80 if args.smoke else 600
    max_lines = 400 if args.smoke else 8000
    n_facts = 12 if args.smoke else 40
    rng = random.Random(SEED)
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    model0 = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model0.eval()
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(4000000)) if len(m.group(1)) >= 5)))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    text_code = s225.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    head_code = s225.train_upper(model0, flat_c, off_c, char_table, pad_id, device, upper_steps, SEED + 2)
    rank = {'none': 0, 'gold': 0, 'wrong': 0}
    follow_wrong = 0
    follow_gold = 0
    lp_gold_given_gold = []
    lp_gold_given_wrong = []
    lp_gold_given_none = []
    n = 0
    for i, (S, gold) in enumerate(zip(subs, vals)):
        wrong = vals[(i + 7) % len(vals)]
        if wrong == gold:
            wrong = vals[(i + 3) % len(vals)]
        pool = vals
        p_none = prompt_code_comment(S, None)
        p_gold = prompt_code_comment(S, gold)
        p_wrong = prompt_code_comment(S, wrong)
        for k, p in [('none', p_none), ('gold', p_gold), ('wrong', p_wrong)]:
            rank[k] += int(rank4_last(head_code, tok, pad_id, char_table, device, p, gold, pool, rng))
        tid_g = tok.encode(gold).ids[0] if tok.encode(gold).ids else -1
        pred_g = argmax_token(head_code, tok, pad_id, char_table, device, p_gold)
        pred_w = argmax_token(head_code, tok, pad_id, char_table, device, p_wrong)
        follow_gold += int(pred_g == tid_g)
        follow_wrong += int(pred_w == tok.encode(wrong).ids[0] if tok.encode(wrong).ids else -2)
        lp_gold_given_none.append(span_logprob(head_code, tok, pad_id, char_table, device, p_none, gold))
        lp_gold_given_gold.append(span_logprob(head_code, tok, pad_id, char_table, device, p_gold, gold))
        lp_gold_given_wrong.append(span_logprob(head_code, tok, pad_id, char_table, device, p_wrong, gold))
        n += 1
    n = max(1, n)
    r_none, r_gold, r_wrong = (rank['none'] / n, rank['gold'] / n, rank['wrong'] / n)
    sens_rank = r_gold - r_wrong
    sens_lp = float(np.mean(lp_gold_given_gold) - np.mean(lp_gold_given_wrong))
    margin_gold_vs_none = r_gold - r_none
    reads_inject = sens_rank >= 0.08 or sens_lp >= 0.5
    prior_only = abs(sens_rank) < 0.03 and abs(margin_gold_vs_none) < 0.03
    if reads_inject:
        overall = 'HEAD_READS_INJECT_YES'
    elif prior_only:
        overall = 'HEAD_LEXICAL_PRIOR_ONLY'
    else:
        overall = 'HEAD_INJECT_PARTIAL'
    out = {'stage': '228a', 'overall': overall, 'rank4_gold_target': {'none': r_none, 'gold_inject': r_gold, 'wrong_inject': r_wrong}, 'sensitivity_rank_gold_minus_wrong': sens_rank, 'span_logprob_gold': {'mean_given_none': float(np.mean(lp_gold_given_none)), 'mean_given_gold_inject': float(np.mean(lp_gold_given_gold)), 'mean_given_wrong_inject': float(np.mean(lp_gold_given_wrong)), 'sensitivity_gold_minus_wrong': sens_lp}, 'argmax_follows_inject_rate': {'gold_prompt': follow_gold / n, 'wrong_prompt': follow_wrong / n}, 'interpretation': 'If sensitivity ~0: head ignores inject; use fp-guided decode or head_mem train.' if prior_only or not reads_inject else 'Head partially follows inject; tune format / joint mem templates.', 'n_items': n, 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 228a counterfactual inject\n\n**{overall}** rank none/g/w={r_none:.3f}/{r_gold:.3f}/{r_wrong:.3f} sens={sens_rank:.3f} lp_sens={sens_lp:.3f}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())