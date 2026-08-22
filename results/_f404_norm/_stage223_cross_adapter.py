"""
Stage 223 — Matched vs wrong domain W on the same A-era slot bank.

Train shift_B + W_B (Stories), shift_C + W_C (wiki windows from P1 corpus).
Recall with matched W vs cross W (B keys scenario but C adapter, etc.).

  python _stage223_cross_adapter.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
RES = Path('results')
DECISION = RES / 'stage223_decision.json'
MINI = RES / 'stage223_mini.md'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
DOMAIN_B = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 223

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else s221.CORE_N
    n_facts = 12 if args.smoke else 60
    rng = random.Random(SEED)
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(2000000)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    train_idx = list(range(len(core)))
    model_old = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model_old.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model_old.eval()
    bank_old = FpBank(model_old, stoi, device)
    F_old = s221.fp_matrix(bank_old, core)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(4000000)) if len(m.group(1)) >= 5)))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_old, V = s221.build_fact_bank(bank_old, subs, vals, rng)
    text_b = DOMAIN_B.read_text(encoding='utf-8', errors='ignore')
    flat_b, off_b = s213.build_flat_from_text(text_b, tok, pad_id, max_lines=500 if args.smoke else 8000)
    model_b = s221.finetune_arc_enc(model_old, flat_b, off_b, char_table, pad_id, device, arc_steps, SEED + 1)
    F_new_b = s221.fp_matrix(FpBank(model_b, stoi, device), core)
    Wb, _ = s221.train_remap(DomainAdapter(256).to(device), F_old, F_new_b, rng, w_steps, device)
    model_c = s221.finetune_arc_enc(model_old, flat, off, char_table, pad_id, device, arc_steps, SEED + 2)
    F_new_c = s221.fp_matrix(FpBank(model_c, stoi, device), core)
    Wc, _ = s221.train_remap(DomainAdapter(256).to(device), F_old, F_new_c, rng, w_steps, device)

    def tr(mod):
        return lambda K: F.normalize(mod.map_raw(K), dim=-1)
    bank_b = FpBank(model_b, stoi, device)

    def recall_k(K, bank_q, key_x):
        ok, n = (0, 0)
        for S, gold in zip(subs, vals):
            q = bank_q.ctx_fp(f'In the report {S} was linked to the organization.', exclude=gold)
            if q is None:
                continue
            Kq = key_x(K)
            cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
            rng.shuffle(cands)
            g = cands.index(gold)
            sc = []
            for c in cands:
                idxs = [i for i, v in enumerate(V) if v == c]
                sc.append(float((Kq[idxs] @ q).max()) if idxs else -1.0)
            ok += int(__import__('numpy').argmax(sc) == g)
            n += 1
        return ok / max(1, n)
    acc_bb = recall_k(K_old, bank_b, tr(Wb))
    acc_wrong_Wc_on_B = recall_k(K_old, bank_b, tr(Wc))
    bank_c = FpBank(model_c, stoi, device)
    acc_cc = recall_k(K_old, bank_c, tr(Wc))
    acc_wrong_Wb_on_C = recall_k(K_old, bank_c, tr(Wb))
    acc_legacy_Wb, _ = s221.recall_at(K_old, V, bank_old, subs, vals, rng, tr(Wb))
    acc_legacy_Wc, _ = s221.recall_at(K_old, V, bank_old, subs, vals, rng, tr(Wc))
    cross_drop_b = acc_bb - acc_wrong_Wc_on_B
    cross_drop_c = acc_cc - acc_wrong_Wb_on_C
    arch_ok = acc_bb >= 0.7 and acc_cc >= 0.7 and (cross_drop_b >= 0.03) and (cross_drop_c >= 0.03)
    overall = 'DOMAIN_W_SWITCH_OK' if arch_ok else 'DOMAIN_W_SWITCH_PARTIAL'
    out = {'stage': 223, 'overall': overall, 'recall_B_new_query_W_B_keys': acc_bb, 'recall_B_new_query_W_C_keys_WRONG': acc_wrong_Wc_on_B, 'recall_C_new_query_W_C_keys': acc_cc, 'recall_C_new_query_W_B_keys_WRONG': acc_wrong_Wb_on_C, 'recall_legacy_221_W_B': acc_legacy_Wb, 'recall_legacy_221_W_C': acc_legacy_Wc, 'cross_drop_B': cross_drop_b, 'cross_drop_C': cross_drop_c, 'note': '221-style: old fp extract + W on keys and queries; cross = wrong adapter', 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 223 cross adapter\n\n**{overall}** B={acc_bb:.3f} wrongWc={acc_wrong_Wc_on_B:.3f} C={acc_cc:.3f} wrongWb={acc_wrong_Wb_on_C:.3f}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())