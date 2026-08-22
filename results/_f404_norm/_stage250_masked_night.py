"""
Stage 250 — Long masked-only night: understanding CE without binding leak.

Single arm (no full-bindings control) so wall-clock goes into useful steps.
Periodically probe next_tok + slot mem; optional hop-gated extra writes mid-stream.
Saves checkpoint for resume.

  python _stage250_masked_night.py [--smoke] [--steps N] [--resume]
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
CKPT_IN = Path('checkpoints/stage191_p1_curve.pt')
CKPT_OUT = Path('checkpoints/stage250_masked_night.pt')
CKPT_248 = Path('checkpoints/stage248_masked_upper.pt')
DECISION = RES / 'stage250_decision.json'
MINI = RES / 'stage250_mini.md'
LOG = RES / '_stage250_log.txt'
WIKI = Path('data/_wikitext103_train.txt')
EXAM = Path('data/stage191_exam_v3.jsonl')
SEED = 250

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

def next_tok(model, char_table, pad_id, items, device):
    if not items:
        return float('nan')
    ok = 0
    for it in items:
        sc = [span_logprob_x(model, char_table, pad_id, it['ctx_ids'], c, device) for c in it['cand_ids']]
        ok += int(int(np_argmax(sc)) == it['gold_idx'])
    return ok / len(items)

def np_argmax(sc):
    import numpy as np
    return int(np.argmax(sc))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = args.steps or (200 if args.smoke else 60000)
    n_facts = 12 if args.smoke else 48
    probe_every = 50 if args.smoke else 5000
    arc_steps = 30 if args.smoke else 300
    w_steps = 30 if args.smoke else 400
    max_lines = 200 if args.smoke else 12000
    core_n = 40 if args.smoke else 300
    n_next = 30 if args.smoke else 100
    log(f'Stage250 start {datetime.now(timezone.utc).isoformat()} steps={steps} resume={args.resume}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model = SelfModelXL(n_char, V).to(device)
    src = CKPT_248 if args.resume and CKPT_248.exists() else CKPT_IN
    if args.resume and CKPT_OUT.exists():
        src = CKPT_OUT
    model.load_state_dict(torch.load(src, map_location=device, weights_only=False)['model'])
    model.eval()
    log(f'loaded {src}')
    bank0 = FpBank(model, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(3000000 if args.smoke else 25000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 150]
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 40) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[i]
        facts.append({'S': S, 'value': Vv, 'sent': f'{S} was appointed director of {Vv} in 1987 .', 'fid': i})
    all_values = [f['value'] for f in facts] + values_pool[n_facts:n_facts + 80]
    hop_q = bank0.ctx_fp('In the report the organization appointed a new director of governance.')
    if hop_q is None:
        hop_q = bank0.fp(['organization'])[0]
    scored = []
    for f in facts:
        k = bank0.fp([f['S']])[0]
        c = bank0.ctx_fp(f['sent'], exclude=f['value'])
        key = F.normalize(k + c, dim=-1) if c is not None else k
        scored.append((float((key * hop_q).sum()), f))
    scored.sort(key=lambda x: -x[0])
    facts_hop = [f for _, f in scored[:max(2, len(facts) // 2)]]
    chunks = []
    for i, f in enumerate(facts):
        if i < len(paras):
            chunks.append(paras[i][:280])
        chunks.append(f['sent'])
    stream = ' '.join(chunks + paras[len(facts):len(facts) + 80])
    stream_m = mask_facts(stream, facts)
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
    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_IN, map_location=device, weights_only=False)['model'])
    model_can.eval()
    bank_can = FpBank(model_can, stoi, device)
    K, Vlist = L.write_tape_bank(bank_can, facts_hop)
    F_can = s221.fp_matrix(bank_can, core)
    m = copy.deepcopy(model)
    s213.set_train_mode(m, 'upper')
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0003, weight_decay=0.01)
    trng = random.Random(SEED + 3)
    curve = []
    for step in range(1, steps + 1):
        ids = sample_windows(flat_m, off_m, MICRO, trng, pad_id).to(device)
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
        if step % max(100, steps // 20) == 0:
            log(f'  night step {step}/{steps}: ce={float(ce):.3f}')
        if step % probe_every == 0 or step == steps:
            m.eval()
            nt = next_tok(m, char_table, pad_id, items, device)
            mem = L.tape_recall_decision(facts_hop, all_values, bank_can, K, Vlist, SEED)
            win = s225.window_next_tok_acc(m, flat_m, off_m, char_table, pad_id, device, random.Random(SEED + step), 8 if args.smoke else 16)
            curve.append({'step': step, 'ce': float(ce), 'next_tok': nt, 'mem': mem, 'domain_win': win})
            log(f"  probe@{step}: nt={nt:.3f} mem={mem['four_way']:.3f} fb_top1={mem['full_bank_top1']:.3f} win={win:.3f}")
            Path('checkpoints').mkdir(exist_ok=True)
            torch.save({'model': m.state_dict(), 'stage': 250, 'step': step, 'curve': curve}, CKPT_OUT)
            m.train()
            s213.set_train_mode(m, 'upper')
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    code = s227.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(code, tok, pad_id, max_lines=min(max_lines, 8000), min_line_len=20)
    model_b = s221.finetune_arc_enc(m, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 9)
    bank_b = FpBank(model_b, stoi, device)
    W, align = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_b, core), F_can, rng, w_steps, device)
    mem_cf = L.tape_recall_decision(facts_hop, all_values, bank_b, K, Vlist, SEED, W_bwd=W)
    nt_final = curve[-1]['next_tok'] if curve else float('nan')
    nt0 = curve[0]['next_tok'] if curve else float('nan')
    g_mem = mem_cf['four_way'] >= 0.8
    g_under = nt_final == nt_final and nt_final >= (nt0 - 0.03 if nt0 == nt0 else 0.5)
    g_curve = len(curve) >= 2
    if g_mem and g_under:
        overall = 'MASKED_NIGHT_OK'
    elif g_mem or g_under:
        overall = 'MASKED_NIGHT_PARTIAL'
    else:
        overall = 'MASKED_NIGHT_NO'
    out = {'stage': 250, 'overall': overall, 'steps': steps, 'gates': {'G_mem_cf_ge_0p80': g_mem, 'G_under_stable': g_under, 'G_probes': g_curve}, 'curve': curve, 'mem_after_code_W': mem_cf, 'W_align': align, 'n_hop_facts': len(facts_hop), 'ckpt': str(CKPT_OUT), 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 250 masked night\n\n**{overall}** steps={steps} mem_cf={mem_cf['four_way']:.3f} fb_top1={mem_cf['full_bank_top1']:.3f} nt_final={nt_final}\n", encoding='utf-8')
    log(json.dumps({'overall': overall, 'mem_cf': mem_cf, 'probes': len(curve)}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())