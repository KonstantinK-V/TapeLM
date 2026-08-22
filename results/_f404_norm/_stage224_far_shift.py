"""
Stage 224 — far-domain arc_enc shifts + cross-W matrix (Stories vs code vs med).

Tests whether W is domain-specific (registry) or ~canonical (universal unwarp).

  python _stage224_far_shift.py [--smoke]
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
RES = Path('results')
DATA = Path('data')
DECISION = RES / 'stage224_decision.json'
MINI = RES / 'stage224_mini.md'
CODE_CORPUS = DATA / '_stage224_code_corpus.txt'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
STORIES = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 224
MED_RE = re.compile('\\b(patient|patients|clinical|diagnosis|treatment|therapy|disease|symptoms|hospital|physician|medical|cancer|cardiac|infection|chronic|acute)\\b', re.I)

def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_code_corpus(rng: random.Random, n_lines: int=12000) -> str:
    if CODE_CORPUS.exists() and CODE_CORPUS.stat().st_size > 10000:
        return CODE_CORPUS.read_text(encoding='utf-8')
    lines: list[str] = []
    for i in range(n_lines):
        fn = f'process_{rng.randint(0, 99999)}'
        var = f'arg_{rng.randint(0, 9999)}'
        mod = f'm{i % 200}'
        lines.append(f'def {fn}({var}: int) -> int:\n    import numpy as np  # {mod}\n    return np.abs({var}) + {rng.randint(0, 512)}')
        lines.append(f'class Handler_{i % 500}(object):\n    def __init__(self, {var}=None):\n        self.{var} = {var}')
    text = '\n'.join(lines)
    CODE_CORPUS.write_text(text, encoding='utf-8')
    return text

def ensure_med_corpus(max_lines: int=8000) -> str:
    out = DATA / '_stage224_med_corpus.txt'
    if out.exists() and out.stat().st_size > 10000:
        return out.read_text(encoding='utf-8')
    lines: list[str] = []
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) < 48 or not MED_RE.search(line):
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
    if len(lines) < 200:
        raise RuntimeError('med corpus too small; check WIKI path')
    text = '\n'.join(lines)
    out.write_text(text, encoding='utf-8')
    log(f'med corpus lines={len(lines)}')
    return text

def flat_from_domain(text: str, tok: Tokenizer, pad_id: int, max_lines: int, min_len: int=32):
    return s213.build_flat_from_text(text, tok, pad_id, max_lines=max_lines, min_line_len=min_len)

def recall_k(K, V, bank_q, subs, vals, rng, key_x):
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
        ok += int(np.argmax(sc) == g)
        n += 1
    return (ok / max(1, n), n)

def w_distance(Wa: torch.Tensor, Wb: torch.Tensor) -> dict:
    d = Wa - Wb
    return {'frobenius': float(d.pow(2).mean().sqrt()), 'cos_flat': float(F.cosine_similarity(Wa.flatten().unsqueeze(0), Wb.flatten().unsqueeze(0)))}

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
        text_wiki = f.read(2000000)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text_wiki) if len(w) <= 14)))[:core_n]
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
    domains: dict[str, tuple] = {}
    corpus = {'stories': STORIES.read_text(encoding='utf-8', errors='ignore'), 'code': ensure_code_corpus(random.Random(SEED + 11), n_lines=3000 if args.smoke else 12000), 'med': ensure_med_corpus(max_lines=max_lines)}
    W_maps: dict[str, DomainAdapter] = {}
    banks: dict[str, FpBank] = {}
    shifts: dict[str, float] = {}
    for i, (name, text) in enumerate(corpus.items()):
        log(f'arc shift domain={name} ...')
        flat_d, off_d = flat_from_domain(text, tok, pad_id, max_lines=max_lines)
        model_d = s221.finetune_arc_enc(model_old, flat_d, off_d, char_table, pad_id, device, arc_steps, SEED + 10 + i)
        bank_d = FpBank(model_d, stoi, device)
        F_new = s221.fp_matrix(bank_d, core)
        shifts[name] = float((F_old * F_new).sum(-1).mean())
        Wd, _ = s221.train_remap(DomainAdapter(256).to(device), F_old, F_new, rng, w_steps, device)
        W_maps[name] = Wd
        banks[name] = bank_d
        domains[name] = model_d
    names = list(corpus.keys())

    def tr(mod: DomainAdapter):
        return lambda K: F.normalize(mod.map_raw(K), dim=-1)
    recall_matrix: dict[str, dict[str, float]] = {}
    for d in names:
        recall_matrix[d] = {}
        for wname, Wmod in W_maps.items():
            acc, _ = recall_k(K_old, V, banks[d], subs, vals, rng, tr(Wmod))
            recall_matrix[d][wname] = acc
    cross_drops = {}
    for d in names:
        matched = recall_matrix[d][d]
        wrong = [recall_matrix[d][o] for o in names if o != d]
        best_wrong = max(wrong) if wrong else matched
        worst_wrong = min(wrong) if wrong else matched
        cross_drops[d] = {'matched': matched, 'best_wrong': best_wrong, 'worst_wrong': worst_wrong, 'drop_vs_best_wrong': matched - best_wrong, 'drop_vs_worst_wrong': matched - worst_wrong}
    W_weights = {n: W_maps[n].w.weight.detach() for n in names}
    w_dist = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            w_dist[f'{a}_vs_{b}'] = w_distance(W_weights[a], W_weights[b])
    max_drop = max((cross_drops[d]['drop_vs_best_wrong'] for d in names))
    min_drop = min((cross_drops[d]['drop_vs_best_wrong'] for d in names))
    max_frob = max((v['frobenius'] for v in w_dist.values()))
    if max_drop >= 0.2:
        overall = 'W_REGISTRY_NEEDED'
    elif max_drop < 0.05 and min_drop >= -0.02:
        overall = 'CANONICAL_W_CANDIDATE'
    else:
        overall = 'W_DOMAIN_PARTIAL'
    out = {'stage': 224, 'overall': overall, 'gates': {'G_cross_drop_ge_0p20_any': max_drop >= 0.2, 'G_cross_drop_lt_0p05_all': max_drop < 0.05}, 'mean_cos_shift_per_domain': shifts, 'recall_matrix_query_domain__W_adapter': recall_matrix, 'cross_drops': cross_drops, 'W_frobenius_cos_pairs': w_dist, 'summary': {'max_cross_drop_vs_best_wrong_W': max_drop, 'min_cross_drop_vs_best_wrong_W': min_drop, 'max_W_pair_frobenius': max_frob}, 'domains': {'stories': 'TinyStories prose', 'code': str(CODE_CORPUS), 'med': 'Wiki lines filtered medical lexicon'}, 'note': 'Rows=encoder domain for query; cols=W trained after shift on that domain', 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 224 far shift\n\n**{overall}** max_drop={max_drop:.3f} shifts={shifts}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())