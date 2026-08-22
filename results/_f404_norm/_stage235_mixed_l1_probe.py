"""
Stage 235 — Mixed-domain L1 pretrain probe (scale branch).

Short arc_enc finetune on interleaved prose+code vs prose-only, then measure
cross-domain fp stability (mean core cos) and post-hoc W recall vs frozen P1.

Not full multi-domain pretrain — bounded exam for whether mixed L1 reduces W need.

  python _stage235_mixed_l1_probe.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, mean_core_cos
RES = Path('results')
DECISION = RES / 'stage235_decision.json'
MINI = RES / 'stage235_mini.md'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
STORIES = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 235

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc_steps = 50 if args.smoke else 400
    w_steps = 60 if args.smoke else 800
    core_n = 60 if args.smoke else 400
    n_facts = 10 if args.smoke else 50
    max_lines = 250 if args.smoke else 6000
    rng = random.Random(SEED)
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(2000000)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    model_can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model_can.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model_can.eval()
    bank_can = FpBank(model_can, stoi, device)
    F_can = s221.fp_matrix(bank_can, core)
    flat_s, off_s = s213.build_flat_from_text(STORIES.read_text(encoding='utf-8', errors='ignore'), tok, pad_id, max_lines=max_lines)
    code_text = s227.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(code_text, tok, pad_id, max_lines=max_lines, min_line_len=20)
    pl = STORIES.read_text(encoding='utf-8', errors='ignore').splitlines()[:max_lines]
    cl = code_text.splitlines()[:max_lines]
    mix_lines: list[str] = []
    for i in range(max(len(pl), len(cl))):
        if i < len(pl) and pl[i].strip():
            mix_lines.append(pl[i])
        if i < len(cl) and cl[i].strip():
            mix_lines.append(cl[i])
    flat_mix, off_mix = s213.build_flat_from_text('\n'.join(mix_lines), tok, pad_id, max_lines=max_lines * 2, min_line_len=20)
    model_prose = s221.finetune_arc_enc(model_can, flat_s, off_s, char_table, pad_id, device, arc_steps, SEED + 2)
    model_mixed = s221.finetune_arc_enc(model_can, flat_mix, off_mix, char_table, pad_id, device, arc_steps, SEED + 3)
    model_code_only = s221.finetune_arc_enc(model_can, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 4)
    bank_p = FpBank(model_prose, stoi, device)
    bank_m = FpBank(model_mixed, stoi, device)
    bank_co = FpBank(model_code_only, stoi, device)
    cos_p_can = mean_core_cos(bank_can, bank_p, core)
    cos_m_can = mean_core_cos(bank_can, bank_m, core)
    cos_co_can = mean_core_cos(bank_can, bank_co, core)
    cos_m_code = mean_core_cos(bank_co, bank_m, core)
    cos_p_code = mean_core_cos(bank_co, bank_p, core)
    W_m, _ = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_m, core), F_can, rng, w_steps, device)
    W_p, _ = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_p, core), F_can, rng, w_steps, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(2000000)) if len(m.group(1)) >= 5)))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_can, V = s221.build_fact_bank(bank_can, subs, vals, rng)
    acc_m = s227.recall(K_can, V, bank_co, subs, vals, rng, query_x=s227.w_apply(W_m))
    acc_p = s227.recall(K_can, V, bank_co, subs, vals, rng, query_x=s227.w_apply(W_p))
    g_stability = cos_m_code >= cos_p_code + 0.03
    g_recall = acc_m >= acc_p - 0.05
    g_cos = cos_m_can >= cos_p_can - 0.02
    overall = 'MIXED_L1_PROBE_OK' if g_stability and g_recall and (acc_m >= 0.72) else 'MIXED_L1_PROBE_PARTIAL' if g_stability or g_recall else 'MIXED_L1_PROBE_NO'
    out = {'stage': 235, 'branch': 'pretrain_L1_mixed_domain_probe', 'overall': overall, 'gates': {'G_mixed_closer_to_code_than_prose_only': g_stability, 'G_mixed_W_recall_not_worse': g_recall, 'G_mixed_cos_can_not_hurt': g_cos}, 'mean_cos_can_prose_ft': cos_p_can, 'mean_cos_can_mixed_ft': cos_m_can, 'mean_cos_code_vs_mixed': cos_m_code, 'mean_cos_code_vs_prose_ft': cos_p_code, 'recall_code_query_W_mixed': acc_m, 'recall_code_query_W_prose_ft': acc_p, 'arc_steps': arc_steps, 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 235 mixed L1 probe\n\n**{overall}** mixed_recall={acc_m:.3f} prose_W={acc_p:.3f}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())