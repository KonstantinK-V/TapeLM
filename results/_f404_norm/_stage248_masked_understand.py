"""
Stage 248 — Masked-CE understanding growth while facts live only in slots.

Branch from 247: CE on binding-stripped domain text (upper layers, frozen arc_enc);
novel facts written to canonical slots (+ optional hop gate). Measure:
  - understanding: next_tok exam (+ domain window acc)
  - memory: slot recall before/after long CE and after code-shift+W
  - edit: slot overwrite collateral
  - control arm: CE on FULL text with bindings (facts leak into weights via upper)

  python _stage248_masked_understand.py [--smoke] [--steps N]
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
import time
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
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data, sample_windows, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
RES = Path('results')
DECISION = RES / 'stage248_decision.json'
MINI = RES / 'stage248_mini.md'
LOG = RES / '_stage248_log.txt'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
EXAM = Path('data/stage191_exam_v3.jsonl')
SEED = 248

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def mask_facts(text: str, facts) -> str:
    out = text
    for f in facts:
        out = out.replace(f['sent'], 'The chronicle continues without naming any director.')
        out = out.replace(f['S'], 'Someone')
        out = out.replace(f['value'], 'somewhere')
    return out

def ce_upper(model, flat, off, char_table, pad_id, device, steps, seed, tag):
    m = copy.deepcopy(model)
    s213.set_train_mode(m, 'upper')
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0003, weight_decay=0.01)
    rng = random.Random(seed)
    for step in range(1, steps + 1):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
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
        if step % max(100, steps // 10) == 0:
            log(f'  {tag} step {step}/{steps}: ce={float(ce):.3f}')
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m

def next_tok(model, char_table, pad_id, items, device):
    if not items:
        return float('nan')
    ok = 0
    for it in items:
        sc = [span_logprob_x(model, char_table, pad_id, it['ctx_ids'], c, device) for c in it['cand_ids']]
        ok += int(int(np.argmax(sc)) == it['gold_idx'])
    return ok / len(items)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = args.steps or (150 if args.smoke else 8000)
    n_facts = 12 if args.smoke else 40
    arc_steps = 40 if args.smoke else 400
    w_steps = 40 if args.smoke else 500
    max_lines = 250 if args.smoke else 10000
    core_n = 50 if args.smoke else 300
    n_next = 40 if args.smoke else 120
    log(f'Stage248 start {datetime.now(timezone.utc).isoformat()} steps={steps}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model0 = SelfModelXL(n_char, V).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model0.eval()
    for p in model0.parameters():
        p.requires_grad_(False)
    bank0 = FpBank(model0, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(4000000 if args.smoke else 20000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 160]
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    F_can = s221.fp_matrix(bank0, core)
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 40) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[i]
        facts.append({'S': S, 'value': Vv, 'sent': f'{S} was appointed director of {Vv} in 1987 .', 'fid': i})
    all_values = [f['value'] for f in facts] + values_pool[n_facts:n_facts + 80]
    chunks = []
    for i, f in enumerate(facts):
        if i < len(paras):
            chunks.append(paras[i][:300])
        chunks.append(f['sent'])
    stream = ' '.join(chunks + paras[len(facts):len(facts) + 40])
    stream_m = mask_facts(stream, facts)
    flat_f, off_f = s213.build_flat_from_text(stream, tok, pad_id, max_lines=max_lines, min_line_len=16)
    flat_m, off_m = s213.build_flat_from_text(stream_m, tok, pad_id, max_lines=max_lines, min_line_len=16)
    items = []
    if EXAM.exists():
        with EXAM.open(encoding='utf-8') as f:
            for line in f:
                it = json.loads(line)
                if it.get('type') == 'next_tok':
                    items.append(it)
                if len(items) >= n_next:
                    break
    K, Vlist = L.write_tape_bank(bank0, facts)
    mem0 = L.tape_recall(facts, all_values, bank0, K, Vlist, SEED)
    nt0 = next_tok(model0, char_table, pad_id, items, device)
    log('arm MASKED: slots + CE without bindings')
    m_mask = ce_upper(model0, flat_m, off_m, char_table, pad_id, device, steps, SEED + 2, 'masked')
    mem_m = L.tape_recall(facts, all_values, bank0, K, Vlist, SEED)
    nt_m = next_tok(m_mask, char_table, pad_id, items, device)
    win_m = s225.window_next_tok_acc(m_mask, flat_m, off_m, char_table, pad_id, device, random.Random(SEED + 3), 12 if args.smoke else 24)
    log('arm FULL: CE with bindings (leak control)')
    m_full = ce_upper(model0, flat_f, off_f, char_table, pad_id, device, steps, SEED + 4, 'full')
    bank_full = FpBank(m_full, stoi, device)
    mem_full_raw = L.tape_recall(facts, all_values, bank_full, K, Vlist, SEED)
    nt_full = next_tok(m_full, char_table, pad_id, items, device)
    code = s227.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    model_b = s221.finetune_arc_enc(m_mask, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 5)
    bank_b = FpBank(model_b, stoi, device)
    W, align = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_b, core), F_can, rng, w_steps, device)
    mem_cf = L.tape_recall(facts, all_values, bank_b, K, Vlist, SEED, W_bwd=W)
    K2, V2 = (K.clone(), list(Vlist))
    V2[0] = values_pool[n_facts + 2]
    f0 = dict(facts[0])
    f0['value'] = V2[0]
    f0['sent'] = f"{f0['S']} was appointed director of {V2[0]} in 1987 ."
    k = bank0.fp([f0['S']])[0]
    c = bank0.ctx_fp(f0['sent'], exclude=V2[0])
    K2[0] = F.normalize(k + c, dim=-1) if c is not None else k
    edit_new = L.tape_recall([f0], all_values + [V2[0]], bank0, K2, V2, SEED)
    edit_ret = L.tape_recall(facts[1:], all_values, bank0, K2, V2, SEED)
    g_mem = mem_cf >= 0.8
    g_under = nt_m >= nt0 - 0.02
    g_better_than_full = mem_m >= mem_full_raw - 0.05 and nt_m + 0.02 >= nt_full
    g_edit = edit_new >= 0.8 and abs(edit_ret - L.tape_recall(facts[1:], all_values, bank0, K, Vlist, SEED)) <= 0.05
    if g_mem and g_under and g_edit:
        overall = 'MASKED_UNDERSTAND_OK'
    elif g_mem and (g_under or g_edit):
        overall = 'MASKED_UNDERSTAND_PARTIAL'
    else:
        overall = 'MASKED_UNDERSTAND_NO'
    out = {'stage': 248, 'overall': overall, 'steps': steps, 'gates': {'G_mem_after_shift_ge_0p80': g_mem, 'G_under_not_worse': g_under, 'G_masked_vs_full_ok': g_better_than_full, 'G_edit_clean': g_edit}, 'baseline': {'mem': mem0, 'next_tok': nt0}, 'masked': {'mem': mem_m, 'next_tok': nt_m, 'domain_win': win_m, 'mem_after_code_W': mem_cf, 'W_align': align}, 'full_bindings_CE': {'mem_query_drifted': mem_full_raw, 'next_tok': nt_full}, 'edit': {'new': edit_new, 'retained': edit_ret}, 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 248 masked understand\n\n**{overall}** steps={steps} mem_cf={mem_cf:.3f} nt_m={nt_m:.3f} nt_full={nt_full:.3f}\n', encoding='utf-8')
    log(json.dumps({'overall': overall, 'mem_cf': mem_cf, 'nt_m': nt_m}, indent=2))
    if not args.smoke and steps >= 2000:
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save({'model': m_mask.state_dict(), 'stage': 248, 'steps': steps}, 'checkpoints/stage248_masked_upper.pt')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())