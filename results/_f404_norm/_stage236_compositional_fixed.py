"""
Stage 236 — Fixed-exam compositional W (productize 234 algebra).

Persist a frozen fact list; re-run chained qmap vs direct W on that exam.
Gate: composed recall within 0.10 of direct; both ≥ 0.70.

  python _stage236_compositional_fixed.py [--smoke]
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
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, compose_w_bwd
RES = Path('results')
DATA = Path('data')
DECISION = RES / 'stage236_decision.json'
MINI = RES / 'stage236_mini.md'
EXAM = DATA / 'stage236_fixed_facts.json'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
STORIES = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 236

def ensure_exam(n_facts: int, smoke: bool) -> dict:
    if EXAM.exists() and (not smoke):
        return json.loads(EXAM.read_text(encoding='utf-8'))
    rng = random.Random(SEED)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(4000000)) if len(m.group(1)) >= 5)))
    from _stage192_fp_lexicon import gen_fakes
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    blob = {'seed': SEED, 'subjects': subs, 'values': vals, 'n': n_facts}
    if not smoke:
        EXAM.parent.mkdir(parents=True, exist_ok=True)
        EXAM.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding='utf-8')
    return blob

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc_steps = 60 if args.smoke else s221.ARC_STEPS
    w_steps = 80 if args.smoke else s221.W_STEPS
    core_n = 60 if args.smoke else s221.CORE_N
    n_facts = 10 if args.smoke else 55
    max_lines = 300 if args.smoke else 8000
    rng = random.Random(SEED)
    exam = ensure_exam(n_facts, args.smoke)
    subs, vals = (exam['subjects'][:n_facts], exam['values'][:n_facts])
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
    K_can, V = s221.build_fact_bank(bank_can, subs, vals, rng)
    flat_s, off_s = s213.build_flat_from_text(STORIES.read_text(encoding='utf-8', errors='ignore'), tok, pad_id, max_lines=max_lines)
    flat_c, off_c = s213.build_flat_from_text(s227.ensure_code(random.Random(SEED + 1), args.smoke), tok, pad_id, max_lines=max_lines, min_line_len=20)
    model_p = s221.finetune_arc_enc(model_can, flat_s, off_s, char_table, pad_id, device, arc_steps, SEED + 2)
    model_pc = s221.finetune_arc_enc(model_p, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 3)
    model_c = s221.finetune_arc_enc(model_can, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 4)
    bank_p = FpBank(model_p, stoi, device)
    bank_pc = FpBank(model_pc, stoi, device)
    bank_c = FpBank(model_c, stoi, device)
    F_p = s221.fp_matrix(bank_p, core)
    F_pc = s221.fp_matrix(bank_pc, core)
    F_c = s221.fp_matrix(bank_c, core)
    W_p_bwd, _ = s221.train_remap(DomainAdapter(256).to(device), F_p, F_can, rng, w_steps, device)
    W_step_bwd, _ = s221.train_remap(DomainAdapter(256).to(device), F_pc, F_p, rng, w_steps, device)
    W_direct_bwd, _ = s221.train_remap(DomainAdapter(256).to(device), F_c, F_can, rng, w_steps, device)
    W_comp = compose_w_bwd(W_p_bwd, W_step_bwd)
    acc_direct = s227.recall(K_can, V, bank_c, subs, vals, rng, query_x=s227.w_apply(W_direct_bwd))
    acc_comp = s227.recall(K_can, V, bank_c, subs, vals, rng, query_x=s227.w_apply(W_comp))
    gap = acc_direct - acc_comp
    g1 = acc_comp >= acc_direct - 0.1
    g2 = acc_comp >= 0.7
    overall = 'COMPOSITIONAL_FIXED_OK' if g1 and g2 else 'COMPOSITIONAL_FIXED_PARTIAL' if g1 or acc_comp >= 0.65 else 'COMPOSITIONAL_FIXED_NO'
    out = {'stage': 236, 'overall': overall, 'gates': {'G_composed_within_0p10_of_direct': g1, 'G_composed_ge_0p70': g2}, 'exam_path': str(EXAM) if EXAM.exists() else 'ephemeral_smoke', 'n_facts': len(subs), 'recall_direct': acc_direct, 'recall_composed': acc_comp, 'gap_direct_minus_composed': gap, 'api': 'compose_w_bwd', 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 236 compositional fixed\n\n**{overall}** direct={acc_direct:.3f} composed={acc_comp:.3f}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())