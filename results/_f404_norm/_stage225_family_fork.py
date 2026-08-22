"""
Stage 225 — family fork for W + multi-head generation with frozen arc_enc.

Tests domain-bundle modularity on one shared substrate:
  shared: frozen arc_enc → one fp geometry
  bundle: {W_family, head_family}  (slots versioned later)

A) W_prose reuse on legal-ish wiki lines vs freshly learned W_legal (fork gate drop>=0.05).
B) Train head_prose / head_code with arc_enc frozen; matched vs cross next_tok on domain windows;
   fp drift must stay ~0 (shared map).

  python _stage225_family_fork.py [--smoke]
"""
from __future__ import annotations
import argparse
import copy
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
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data, lr_at, sample_windows
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, WFamilyPolicy
RES = Path('results')
DATA = Path('data')
DECISION = RES / 'stage225_decision.json'
MINI = RES / 'stage225_mini.md'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
STORIES = Path('data/external_tinystories_100k_85.txt')
WIKI = Path('data/_wikitext103_train.txt')
CODE = Path('data/_stage224_code_corpus.txt')
SEED = 225
LEGAL_RE = re.compile('\\b(court|law|legal|plaintiff|defendant|statute|contract|jurisdiction|legislation|attorney|verdict|constitution|amendment)\\b', re.I)

def log(m: str) -> None:
    print(m, flush=True)

def ensure_legal_corpus(max_lines: int) -> str:
    path = DATA / '_stage225_legal_corpus.txt'
    if path.exists() and path.stat().st_size > 5000:
        return path.read_text(encoding='utf-8')
    lines: list[str] = []
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) < 48 or not LEGAL_RE.search(line):
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
    if len(lines) < 80:
        raise RuntimeError('legal corpus too small')
    path.write_text('\n'.join(lines), encoding='utf-8')
    log(f'legal corpus lines={len(lines)}')
    return path.read_text(encoding='utf-8')

def ensure_code(rng: random.Random, smoke: bool) -> str:
    if CODE.exists() and CODE.stat().st_size > 10000:
        return CODE.read_text(encoding='utf-8')
    import _stage224_far_shift as s224
    return s224.ensure_code_corpus(rng, n_lines=2000 if smoke else 12000)

@torch.no_grad()
def window_next_tok_acc(model, flat, off, char_table, pad_id, device, rng, n_batches=20) -> float:
    ok = tot = 0
    for _ in range(n_batches):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        pad = ids == pad_id
        logits, _, _ = model.forward_all(char_table[ids], pad, ids=ids)
        pred = logits[:, :-1].argmax(-1)
        target = ids[:, 1:]
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        ok += int((pred[valid] == target[valid]).sum())
        tot += int(valid.sum())
    return ok / max(1, tot)

def train_upper(model, flat, off, char_table, pad_id, device, steps, seed):
    from _stage191_night import LR
    m = copy.deepcopy(model)
    s213.set_train_mode(m, 'upper')
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR * 0.5)
    r2 = random.Random(seed)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, r2, pad_id).to(device)
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
    m.eval()
    m.arc_enc.eval()
    return m

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
        sc = [float((Kq[[i for i, v in enumerate(V) if v == c]] @ q).max()) if any((v == c for v in V)) else -1.0 for c in cands]
        ok += int(np.argmax(sc) == g)
        n += 1
    return ok / max(1, n)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    upper_steps = 80 if args.smoke else 600
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else s221.CORE_N
    n_facts = 12 if args.smoke else 60
    max_lines = 400 if args.smoke else 8000
    rng = random.Random(SEED)
    flat_w, off_w, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_head = f.read(2000000)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', wiki_head) if len(w) <= 14)))[:core_n]
    model0 = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model0.eval()
    bank0 = FpBank(model0, stoi, device)
    F_old = s221.fp_matrix(bank0, core)
    fp_ref = {w: F_old[i].clone() for i, w in enumerate(core[:min(32, len(core))])}
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(4000000)) if len(m.group(1)) >= 5)))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_old, V = s221.build_fact_bank(bank0, subs, vals, rng)
    log('A: W_prose vs legal fork …')
    text_stories = STORIES.read_text(encoding='utf-8', errors='ignore')
    flat_s, off_s = s213.build_flat_from_text(text_stories, tok, pad_id, max_lines=max_lines)
    model_stories = s221.finetune_arc_enc(model0, flat_s, off_s, char_table, pad_id, device, arc_steps, SEED + 1)
    F_stories = s221.fp_matrix(FpBank(model_stories, stoi, device), core)
    cos_stories = float((F_old * F_stories).sum(-1).mean())
    W_prose, _ = s221.train_remap(DomainAdapter(256).to(device), F_old, F_stories, rng, w_steps, device)
    text_legal = ensure_legal_corpus(max_lines)
    flat_l, off_l = s213.build_flat_from_text(text_legal, tok, pad_id, max_lines=max_lines, min_line_len=40)
    model_legal = s221.finetune_arc_enc(model0, flat_l, off_l, char_table, pad_id, device, arc_steps, SEED + 2)
    bank_legal = FpBank(model_legal, stoi, device)
    F_legal = s221.fp_matrix(bank_legal, core)
    cos_legal = float((F_old * F_legal).sum(-1).mean())
    W_legal, _ = s221.train_remap(DomainAdapter(256).to(device), F_old, F_legal, rng, w_steps, device)

    def tr(W):
        return lambda K: F.normalize(W.map_raw(K), dim=-1)
    acc_reuse = recall_k(K_old, V, bank_legal, subs, vals, rng, tr(W_prose))
    acc_matched = recall_k(K_old, V, bank_legal, subs, vals, rng, tr(W_legal))
    fork = WFamilyPolicy.should_fork(acc_matched, acc_reuse, drop_tol=0.05)
    policy = WFamilyPolicy(registry={'prose': W_prose}, cos_identity=0.85, cos_family_floor=0.65)
    decide_legal = policy.decide(cos_legal, 'prose')
    log('B: multi-head freeze arc_enc …')
    text_code = ensure_code(random.Random(SEED + 3), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    head_prose = train_upper(model0, flat_s, off_s, char_table, pad_id, device, upper_steps, SEED + 4)
    head_code = train_upper(model0, flat_c, off_c, char_table, pad_id, device, upper_steps, SEED + 5)
    F_hp = s221.fp_matrix(FpBank(head_prose, stoi, device), list(fp_ref.keys()))
    F_hc = s221.fp_matrix(FpBank(head_code, stoi, device), list(fp_ref.keys()))
    F_ref = torch.stack([fp_ref[w] for w in fp_ref.keys()])
    drift_prose = float(1.0 - (F_hp * F_ref).sum(-1).mean())
    drift_code = float(1.0 - (F_hc * F_ref).sum(-1).mean())
    r_eval = random.Random(SEED + 9)
    nb = 8 if args.smoke else 24
    gen = {'stories_with_head_prose': window_next_tok_acc(head_prose, flat_s, off_s, char_table, pad_id, device, r_eval, nb), 'stories_with_head_code': window_next_tok_acc(head_code, flat_s, off_s, char_table, pad_id, device, r_eval, nb), 'code_with_head_code': window_next_tok_acc(head_code, flat_c, off_c, char_table, pad_id, device, r_eval, nb), 'code_with_head_prose': window_next_tok_acc(head_prose, flat_c, off_c, char_table, pad_id, device, r_eval, nb), 'baseline_stories_P1': window_next_tok_acc(model0, flat_s, off_s, char_table, pad_id, device, r_eval, nb), 'baseline_code_P1': window_next_tok_acc(model0, flat_c, off_c, char_table, pad_id, device, r_eval, nb)}
    gen['cross_drop_stories'] = gen['stories_with_head_prose'] - gen['stories_with_head_code']
    gen['cross_drop_code'] = gen['code_with_head_code'] - gen['code_with_head_prose']
    g_fp = drift_prose < 1e-05 and drift_code < 1e-05
    g_head = gen['cross_drop_stories'] >= 0.02 or gen['cross_drop_code'] >= 0.02
    if g_fp and g_head:
        overall = 'DOMAIN_BUNDLE_OK'
    elif g_fp:
        overall = 'DOMAIN_BUNDLE_PARTIAL'
    else:
        overall = 'DOMAIN_BUNDLE_NO'
    out = {'stage': 225, 'overall': overall, 'architecture': {'shared': 'frozen arc_enc → one fp R^d', 'bundle': '{W_family, head_family, slots_family*}', 'note': '*slots versioning deferred; A-era bank used for W tests'}, 'A_W_family_fork': {'cos_stories_shift': cos_stories, 'cos_legal_shift': cos_legal, 'recall_legal_query_W_prose_REUSE': acc_reuse, 'recall_legal_query_W_legal_MATCHED': acc_matched, 'drop_matched_minus_reuse': acc_matched - acc_reuse, 'fork_W_family': fork, 'policy_decide_legal': decide_legal}, 'B_multi_head_frozen_arc': {'fp_drift_head_prose': drift_prose, 'fp_drift_head_code': drift_code, 'generation': gen, 'gates': {'G_fp_shared': g_fp, 'G_head_specializes': g_head}}, 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 225 domain bundle\n\n**{overall}** fork={fork} reuse={acc_reuse:.3f} matchedW={acc_matched:.3f} fp_drift={drift_prose:.2e}/{drift_code:.2e} gen_cross_stories={gen['cross_drop_stories']:.3f} code={gen['cross_drop_code']:.3f}\n", encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())