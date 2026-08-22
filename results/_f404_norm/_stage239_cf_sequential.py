"""
Stage 239 — Sequential catastrophic forgetting (A → B) vs fair GPT control.

Protocol:
  1. Plant domain-A facts (novel subjects) into TapeLM slots on frozen P1.
  2. Matched GPT finetunes on A fact sentences until paraphrase-probe recall clears (205-style).
  3. Adapt to domain B (code):
       TapeLM — keep canonical slot keys; arc_enc shift on code + W_bwd qmap @ read (227).
       GPT    — continue CE finetune on code only (no A rehearsal; classic CF).
  4. Measure retained A-fact recall (shared paraphrase probe) + next_tok collateral.

Gates:
  G_memorize   both systems A-recall >= 0.70 after acquire
  G_tape_keep  TapeLM A-recall after B >= 0.80
  G_gpt_drop   GPT A-recall drops by >= 0.15 vs post-memorize
  G_gap        TapeLM_after - GPT_after >= 0.20

Note: vs GPT+RAG this is architectural (index can also keep A); vs parametric GPT it is CF capability.

  python _stage239_cf_sequential.py [--smoke]
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
import _stage227_canonical_slots as s227
from _stage191_night import PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _tapelm_ext import DomainAdapter
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
EXAM_V3 = Path('data/stage191_exam_v3.jsonl')
DECISION = RES / 'stage239_decision.json'
MINI = RES / 'stage239_mini.md'
LOG = RES / '_stage239_log.txt'
SEED = 239

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(line)
    except OSError:
        pass

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    try:
        LOG.write_text('', encoding='utf-8')
    except OSError:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_facts = 12 if args.smoke else 40
    ft_steps = 240 if args.smoke else 2400
    b_steps_gpt = 400 if args.smoke else 1600
    arc_steps = 60 if args.smoke else s221.ARC_STEPS
    w_steps = 80 if args.smoke else s221.W_STEPS
    core_n = 60 if args.smoke else 400
    n_next = 40 if args.smoke else 120
    max_lines = 300 if args.smoke else 8000
    n_batch, ft_len, ft_lr = (8, 64, 0.0003)
    b_lr = 0.0005
    mem_target = 0.72
    log(f'Stage239 start {datetime.now(timezone.utc).isoformat()} device={device}')
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
    bank = FpBank(model, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(4000000 if args.smoke else 20000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 200]
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 40) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[i]
        facts.append({'S': S, 'value': Vv, 'sent': f'{S} was appointed director of {Vv} in 1987 .', 'fid': i})
    all_values = [f['value'] for f in facts] + values_pool[n_facts:n_facts + 80]
    keys, vals = ([], [])
    for f in facts:
        k = bank.fp([f['S']])[0]
        c = bank.ctx_fp(f['sent'], exclude=f['value'])
        keys.append(F.normalize(k + c, dim=-1) if c is not None else k)
        vals.append(f['value'])
    K = torch.stack(keys, 0)

    def tape_recall(bank_q, Kmat, Vlist, W_bwd=None) -> float:
        ok, n = (0, 0)
        qrng = random.Random(SEED + 3)
        wx = (lambda X: F.normalize(W_bwd.map_raw(X), dim=-1)) if W_bwd is not None else None
        for f in facts:
            q = bank_q.ctx_fp(f"In the report {f['S']} was linked to the organization.", exclude=f['value'])
            if q is None:
                q = bank_q.fp([f['S']])[0]
            qq = wx(q.unsqueeze(0))[0] if wx else q
            others = [x for x in all_values if x != f['value']]
            qrng.shuffle(others)
            cands = [f['value']] + others[:3]
            order = list(range(4))
            qrng.shuffle(order)
            shuf = [cands[i] for i in order]
            sc = []
            for c in shuf:
                idxs = [j for j, v in enumerate(Vlist) if v == c]
                sc.append(float((Kmat[idxs] @ qq).max()) if idxs else -1.0)
            ok += int(int(np.argmax(sc)) == order.index(0))
            n += 1
        return ok / max(1, n)
    items = []
    if EXAM_V3.exists():
        with EXAM_V3.open(encoding='utf-8') as f:
            for line in f:
                it = json.loads(line)
                if it.get('type') == 'next_tok':
                    items.append(it)
                if len(items) >= n_next:
                    break

    def curve_next_tok(m) -> float:
        if not items:
            return float('nan')
        ok = 0
        for it in items:
            sc = [span_logprob_x(m, char_table, pad_id, it['ctx_ids'], c, device) for c in it['cand_ids']]
            ok += int(int(np.argmax(sc)) == it['gold_idx'])
        return ok / len(items)

    def gpt_next_tok(gm) -> float:
        if not items:
            return float('nan')
        ok = 0
        for it in items:
            sc = [gpt_span(gm, device, it['ctx_ids'], c) for c in it['cand_ids']]
            ok += int(int(np.argmax(sc)) == it['gold_idx'])
        return ok / len(items)
    tape_a0 = tape_recall(bank, K, vals)
    tape_nt0 = curve_next_tok(model)
    log(f'tape AFTER A write: A_recall={tape_a0:.3f} next_tok={tape_nt0:.3f} ({time.time() - t0:.0f}s)')
    gm = copy.deepcopy(load_gpt(device))
    gm.train()
    fact_ids = [[i for i in tok.encode(f['sent']).ids if i != pad_id] for f in facts]
    real_ids = [i for i in tok.encode(' '.join(paras[:400])[:150000]).ids if i != pad_id]

    def ft_batch(brng, pool=None, mix_real=True):
        rows = []
        src = pool if pool is not None else fact_ids
        for _ in range(n_batch):
            if not mix_real or brng.random() < 0.75 or (not real_ids):
                seq = []
                while len(seq) < ft_len:
                    seq += src[brng.randrange(len(src))]
                rows.append(seq[:ft_len])
            else:
                s = brng.randrange(max(1, len(real_ids) - ft_len - 1))
                rows.append(real_ids[s:s + ft_len])
        return torch.tensor(rows, device=device)

    def gpt_fact_recall(gm_eval) -> float:
        qrng = random.Random(SEED + 3)
        ok = 0
        for f in facts:
            ctx = [i for i in tok.encode(f"In the report {f['S']} was linked to the organization of").ids if i != pad_id]
            others = [x for x in all_values if x != f['value']]
            qrng.shuffle(others)
            cands = [f['value']] + others[:3]
            order = list(range(4))
            qrng.shuffle(order)
            shuf = [cands[i] for i in order]
            sc = [gpt_span(gm_eval, device, ctx, [i for i in tok.encode(' ' + c).ids if i != pad_id]) for c in shuf]
            ok += int(int(np.argmax(sc)) == order.index(0))
        return ok / max(1, len(facts))
    opt = torch.optim.AdamW(gm.parameters(), lr=ft_lr, weight_decay=0.01)
    brng = random.Random(SEED + 11)
    used_ft = 0
    check_every = 40 if args.smoke else 100
    for step in range(1, ft_steps + 1):
        x = ft_batch(brng)
        loss = gm(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        used_ft = step
        if step % check_every == 0:
            gm.eval()
            cur = gpt_fact_recall(gm)
            log(f'  gpt memorize A step {step}: loss={float(loss):.3f} recall={cur:.3f}')
            if cur >= mem_target:
                gm.train()
                break
            gm.train()
    gm.eval()
    gpt_a0 = gpt_fact_recall(gm)
    gpt_nt0 = gpt_next_tok(gm)
    log(f'gpt AFTER A memorize ({used_ft} steps): A_recall={gpt_a0:.3f} next_tok={gpt_nt0:.3f} ({time.time() - t0:.0f}s)')
    code_text = s227.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(code_text, tok, pad_id, max_lines=max_lines, min_line_len=20)
    code_ids = [i for i in tok.encode(code_text[:200000]).ids if i != pad_id]
    F_can = s221.fp_matrix(bank, core)
    model_b = s221.finetune_arc_enc(model, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 7)
    bank_b = FpBank(model_b, stoi, device)
    W_bwd, align = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_b, core), F_can, rng, w_steps, device)
    tape_a1_raw = tape_recall(bank_b, K, vals, W_bwd=None)
    tape_a1 = tape_recall(bank_b, K, vals, W_bwd=W_bwd)
    tape_nt1 = curve_next_tok(model)
    log(f'tape AFTER B (code shift + W): A_raw={tape_a1_raw:.3f} A_W={tape_a1:.3f} align={align:.3f} next_tok(frozen)={tape_nt1:.3f} ({time.time() - t0:.0f}s)')
    if len(code_ids) < ft_len + 2:
        raise RuntimeError('code corpus too short for GPT domain-B CE')
    gm.train()
    optb = torch.optim.AdamW(gm.parameters(), lr=b_lr, weight_decay=0.01)
    brng_b = random.Random(SEED + 17)

    def code_batch(brng):
        rows = []
        for _ in range(n_batch):
            s = brng.randrange(max(1, len(code_ids) - ft_len - 1))
            rows.append(code_ids[s:s + ft_len])
        return torch.tensor(rows, device=device)
    for step in range(1, b_steps_gpt + 1):
        x = code_batch(brng_b)
        loss = gm(input_ids=x, labels=x).loss
        optb.zero_grad(set_to_none=True)
        loss.backward()
        optb.step()
        if step % max(40, b_steps_gpt // 4) == 0:
            log(f'  gpt learn B step {step}: loss={float(loss):.3f}')
    gm.eval()
    gpt_a1 = gpt_fact_recall(gm)
    gpt_nt1 = gpt_next_tok(gm)
    log(f'gpt AFTER B (code ft): A_recall={gpt_a1:.3f} next_tok={gpt_nt1:.3f} ({time.time() - t0:.0f}s)')
    drop_gpt = gpt_a0 - gpt_a1
    gap = tape_a1 - gpt_a1
    g_mem = tape_a0 >= 0.7 and gpt_a0 >= 0.7
    g_tape = tape_a1 >= 0.8
    g_gpt = drop_gpt >= 0.15
    g_gap = gap >= 0.2
    overall = 'CF_SEQUENTIAL_OK' if g_mem and g_tape and g_gpt and g_gap else 'CF_SEQUENTIAL_PARTIAL' if g_mem and g_tape and (g_gpt or gap >= 0.1) else 'CF_SEQUENTIAL_NO'
    out = {'stage': 239, 'overall': overall, 'gates': {'G_memorize_both_ge_0p70': g_mem, 'G_tape_retain_after_B_ge_0p80': g_tape, 'G_gpt_A_drop_ge_0p15': g_gpt, 'G_gap_tape_minus_gpt_ge_0p20': g_gap}, 'n_facts': n_facts, 'tape': {'A_after_write': tape_a0, 'A_after_B_raw_no_W': tape_a1_raw, 'A_after_B_with_W': tape_a1, 'next_tok_before': tape_nt0, 'next_tok_frozen_after': tape_nt1, 'W_align': align}, 'gpt': {'A_after_memorize': gpt_a0, 'A_after_B': gpt_a1, 'A_drop': drop_gpt, 'next_tok_after_memorize': gpt_nt0, 'next_tok_after_B': gpt_nt1, 'memorize_steps': used_ft, 'domain_B_steps': b_steps_gpt}, 'gap_tape_minus_gpt_after_B': gap, 'note': 'vs parametric GPT = CF capability; vs GPT+RAG = architectural (index can keep A)', 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 239 CF sequential A→B\n\n**{overall}** tape_A={tape_a1:.3f} gpt_A={gpt_a1:.3f} drop_gpt={drop_gpt:.3f} gap={gap:.3f}\n', encoding='utf-8')
    log(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())