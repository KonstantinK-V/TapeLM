"""
Stage 231 — Temporal W: matched qmap vs wrong-era qmap on cross-domain read.

Reuses 227 protocol: canonical bank; code encoder; W_prose_bwd vs W_code_bwd on query.
Gate: matched W recall ≥ wrong W + margin (224-style cross-drop).

  python _stage231_temporal_W.py [--smoke]
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
DECISION = RES / 'stage231_decision.json'
MINI = RES / 'stage231_mini.md'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
STORIES = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 231

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else s221.CORE_N
    n_facts = 12 if args.smoke else 60
    max_lines = 400 if args.smoke else 8000
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
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(4000000)) if len(m.group(1)) >= 5)))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_can, V = s221.build_fact_bank(bank_can, subs, vals, rng)
    flat_s, off_s = s213.build_flat_from_text(STORIES.read_text(encoding='utf-8', errors='ignore'), tok, pad_id, max_lines=max_lines)
    flat_c, off_c = s213.build_flat_from_text(s227.ensure_code(random.Random(SEED + 1), args.smoke), tok, pad_id, max_lines=max_lines, min_line_len=20)
    model_s = s221.finetune_arc_enc(model_can, flat_s, off_s, char_table, pad_id, device, arc_steps, SEED + 2)
    model_c = s221.finetune_arc_enc(model_can, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 3)
    bank_s = FpBank(model_s, stoi, device)
    bank_c = FpBank(model_c, stoi, device)
    F_s = s221.fp_matrix(bank_s, core)
    F_c = s221.fp_matrix(bank_c, core)
    W_s_bwd, _ = s221.train_remap(DomainAdapter(256).to(device), F_s, F_can, rng, w_steps, device)
    W_c_bwd, _ = s221.train_remap(DomainAdapter(256).to(device), F_c, F_can, rng, w_steps, device)
    acc_wrong = s227.recall(K_can, V, bank_c, subs, vals, rng, query_x=s227.w_apply(W_s_bwd))
    acc_matched = s227.recall(K_can, V, bank_c, subs, vals, rng, query_x=s227.w_apply(W_c_bwd))
    acc_prose_self = s227.recall(K_can, V, bank_s, subs, vals, rng, query_x=s227.w_apply(W_s_bwd))
    cos_s = mean_core_cos(bank_can, bank_s, core)
    cos_c = mean_core_cos(bank_can, bank_c, core)

    def pick_W(bank_q: FpBank) -> DomainAdapter:
        cq = mean_core_cos(bank_can, bank_q, core)
        return W_s_bwd if abs(cq - cos_s) <= abs(cq - cos_c) else W_c_bwd
    acc_picked = s227.recall(K_can, V, bank_c, subs, vals, rng, query_x=s227.w_apply(pick_W(bank_c)))
    margin = acc_matched - acc_wrong
    g_match = margin >= 0.08
    g_pick = acc_picked >= acc_wrong + 0.05
    g_self = acc_prose_self >= 0.75
    overall = 'TEMPORAL_W_OK' if g_match and g_pick and (acc_matched >= 0.72) else 'TEMPORAL_W_PARTIAL' if g_match or (margin >= 0.03 and acc_matched >= 0.8) else 'TEMPORAL_W_NO'
    out = {'stage': 231, 'overall': overall, 'gates': {'G_matched_beats_wrong_W': g_match, 'G_era_pick_beats_wrong': g_pick, 'G_prose_self_qmap': g_self}, 'recall_wrong_W_prose_on_code': acc_wrong, 'recall_matched_W_code_on_code': acc_matched, 'recall_era_picked_on_code': acc_picked, 'recall_prose_W_on_prose': acc_prose_self, 'margin_matched_minus_wrong': margin, 'mean_cos_can_prose': cos_s, 'mean_cos_can_code': cos_c, 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 231 temporal W\n\n**{overall}** matched={acc_matched:.3f} wrong={acc_wrong:.3f} Δ={margin:.3f}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())