"""
Stage 240 — CF A→B: TapeLM vs GPT+RAG (frozen embedding index) vs parametric GPT.

Same A facts / code-B adapt as 239. Extra arm: freeze GPT embedding keys after memorize A;
after code CE, query with post-B GPT against frozen index (fair RAG).

Expected: RAG keeps A ≈ TapeLM (architectural). Surprise: query drift breaks RAG.

  python _stage240_cf_vs_rag.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage194_fp_fact_memory import FpBank
from _stage196_tapelm import load_gpt
from _tapelm_ext import DomainAdapter
SEED = 240
DECISION = L.RES / 'stage240_decision.json'
MINI = L.RES / 'stage240_mini.md'
LOG = L.RES / '_stage240_log.txt'

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    log = L.make_logger(LOG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = __import__('random').Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_facts = 12 if args.smoke else 40
    ft_steps = 240 if args.smoke else 2400
    b_steps = 400 if args.smoke else 1600
    arc_steps = 60 if args.smoke else s221.ARC_STEPS
    w_steps = 80 if args.smoke else s221.W_STEPS
    core_n = 60 if args.smoke else 400
    n_next = 40 if args.smoke else 120
    n_batch, ft_len, ft_lr, b_lr = (8, 64, 0.0003, 0.0005)
    mem_target = 0.72
    log(f'Stage240 start {datetime.now(timezone.utc).isoformat()} device={device}')
    _, _, stoi, n_char, tok, V, pad_id, char_table, model, bank = L.load_p1(device)
    _, values_pool, core, paras = L.wiki_bits(args.smoke, core_n, rng)
    facts, all_values = L.make_facts(n_facts, values_pool, rng)
    items = L.load_next_tok_items(n_next)
    K, vals = L.write_tape_bank(bank, facts)
    tape0 = L.tape_recall(facts, all_values, bank, K, vals, SEED)
    log(f'tape A write recall={tape0:.3f}')
    gm = copy.deepcopy(load_gpt(device))
    used, fact_ids, _ = L.memorize_gpt(gm, tok, pad_id, facts, all_values, paras, device, SEED, ft_steps, n_batch, ft_len, ft_lr, mem_target, 40 if args.smoke else 100, log)
    gpt0 = L.gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, SEED)
    Krag, Vrag = L.write_rag_bank(gm, tok, pad_id, device, facts)
    rag0 = L.rag_recall(gm, tok, pad_id, device, facts, all_values, Krag, Vrag, SEED)
    log(f'gpt memorize ({used}): param={gpt0:.3f} rag={rag0:.3f}')
    code_text = s227.ensure_code(__import__('random').Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(code_text, tok, pad_id, max_lines=300 if args.smoke else 8000, min_line_len=20)
    code_ids = [i for i in tok.encode(code_text[:200000]).ids if i != pad_id]
    F_can = s221.fp_matrix(bank, core)
    model_b = s221.finetune_arc_enc(model, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 7)
    bank_b = FpBank(model_b, stoi, device)
    W_bwd, align = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_b, core), F_can, rng, w_steps, device)
    tape1 = L.tape_recall(facts, all_values, bank_b, K, vals, SEED, W_bwd=W_bwd)
    tape1_raw = L.tape_recall(facts, all_values, bank_b, K, vals, SEED, W_bwd=None)
    L.code_ce(gm, code_ids, n_batch, ft_len, b_lr, b_steps, device, SEED, log)
    gpt1 = L.gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, SEED)
    rag1 = L.rag_recall(gm, tok, pad_id, device, facts, all_values, Krag, Vrag, SEED)
    nt1 = L.gpt_next_tok(gm, items, device)
    log(f'after B: tape_W={tape1:.3f} rag={rag1:.3f} param={gpt1:.3f} ({time.time() - t0:.0f}s)')
    g_mem = tape0 >= 0.7 and gpt0 >= 0.7 and (rag0 >= 0.7)
    g_tape = tape1 >= 0.8
    g_rag_keep = rag1 >= 0.8
    g_param_drop = gpt0 - gpt1 >= 0.15
    g_surprise = rag1 < tape1 - 0.15
    if g_mem and g_tape and g_param_drop and g_surprise and (not g_rag_keep):
        overall = 'CF_VS_RAG_SURPRISE'
    elif g_mem and g_tape and g_rag_keep and g_param_drop:
        overall = 'CF_VS_RAG_ARCHITECTURAL'
    elif g_mem and g_tape and (g_rag_keep or g_param_drop):
        overall = 'CF_VS_RAG_PARTIAL'
    else:
        overall = 'CF_VS_RAG_NO'
    out = {'stage': 240, 'overall': overall, 'gates': {'G_memorize_ge_0p70': g_mem, 'G_tape_keep_ge_0p80': g_tape, 'G_rag_keep_ge_0p80': g_rag_keep, 'G_param_drop_ge_0p15': g_param_drop, 'G_rag_surprise_gap': g_surprise}, 'tape': {'A0': tape0, 'A1_W': tape1, 'A1_raw': tape1_raw, 'W_align': align}, 'rag': {'A0': rag0, 'A1': rag1, 'drop': rag0 - rag1}, 'param_gpt': {'A0': gpt0, 'A1': gpt1, 'drop': gpt0 - gpt1, 'next_tok_after_B': nt1}, 'note': 'Frozen GPT emb index after A; queries with post-B GPT. Architectural if RAG~TapeLM.', 'timestamp': datetime.now(timezone.utc).isoformat()}
    L.dump(DECISION, MINI, out, 'Stage 240 CF vs GPT+RAG')
    log(str(out['overall']))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())