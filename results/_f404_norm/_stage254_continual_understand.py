"""
Stage 254 — Continual curriculum with ONE shared upper (fix for 246).

246 trained a fresh head per domain from canonical P1, so `tape_next_tok` for wiki was the
identical 0.2998 in every phase and all drops were exactly 0.0: retention was trivially true
and understanding had nowhere to accumulate. Here a SINGLE upper walks wiki -> stories -> med
-> news with the 253 recipe (CE + 0.2*CPC), so forgetting is possible and growth is possible.

Per phase d:
  - inject fact sentences into domain lines, then STRIP bindings from the CE text
    (typed placeholders, not one repeated stub) -> facts never enter the weights
  - hop-gate admission -> admitted facts written to the shared canonical slot bank
  - train shared upper: joint CE + lam*CPC on (domain + replay of past domains)
  - learn W_bwd[d] (shifted arc_enc -> canonical) for reading slots after query drift

After every phase, for EVERY seen domain: held-out CE/PPL, slot recall against the
ACCUMULATED bank, parametric leak. Global: exam next_tok, uniformity.
Cross-domain 2-hop over the accumulated tape = the "thought" metric (chains are planted so a
value from domain i-1 is the subject of a fact in domain i).

Note: internal latent hops are closed (210/212 **THESIS_NO_AT_SCALE**) — hops here are external fp loops.

  python _stage254_continual_understand.py [--smoke] [--operators-only] [--token-budget N] [--domains wiki,med]

  --operators-only: frozen P1 upper; only W_query (+ growing tape). No joint CE/CPC, no arc shift.
"""
from __future__ import annotations
import argparse
import json
import math
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
import _stage224_far_shift as s224
import _stage246_domain_curriculum as s246
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter
RES = Path('results')
DECISION = RES / 'stage254_decision.json'
MINI = RES / 'stage254_mini.md'
LOG = RES / '_stage254_log.txt'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
CKPT_OUT = Path('checkpoints/stage254_continual_upper.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 254
LAM = 0.2
REPLAY_FRAC = 0.25
HOP_QUERY = 'In the report the organization appointed a new director linked to governance.'
OFF_THEME = 'The recipe for {S} calls for {V} simmered slowly in a copper pan .'
ON_THEME = '{S} was appointed director of {V} in the {D} chronicle of 1987 .'
PLACEHOLDERS = ['The chronicle continues with other institutional details .', 'A later passage turns to unrelated regional history .', 'The record notes routine administrative procedure .', 'Following lines cover general background only .', 'The section closes without naming any official .']

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)
STORIES_RAW = Path('data/_tinystories_raw_100k.txt')
MIN_LINE = 40

def build_filtered(out_path: Path, regex, max_lines: int, min_len: int=48) -> str:
    """Domain slice of wikitext; 246's caches were capped at 8000 lines, too small here."""
    if out_path.exists() and out_path.stat().st_size > 100000:
        return out_path.read_text(encoding='utf-8')
    lines: list[str] = []
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) < min_len or not regex.search(line):
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    log(f'  built {out_path.name}: {len(lines)} lines')
    return '\n'.join(lines)

def domain_lines(name: str, max_lines: int, smoke: bool, rng: random.Random) -> list[str]:
    if name == 'stories' and STORIES_RAW.exists():
        text = STORIES_RAW.read_text(encoding='utf-8', errors='ignore')[:4000000 if not smoke else 400000]
    elif name == 'med' and (not smoke):
        text = build_filtered(Path('data/_stage254_med.txt'), s224.MED_RE, max_lines)
    elif name == 'news' and (not smoke):
        text = build_filtered(Path('data/_stage254_news.txt'), s246.NEWS_RE, max_lines)
    else:
        text = s246.domain_text(name, max_lines, smoke, rng)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= MIN_LINE]
    return lines[:max_lines]

def plant_facts(values_pool, n_on, n_off, domain, chain_subjects, rng):
    """On-theme facts (some chained from the previous domain) + off-theme gate distractors.

    Half of each group carries wq_train=True: those fit W_query, the rest score recall.
    """
    subs = [w for w in gen_fakes(set(values_pool), rng, n_on + n_off + 30) if len(w) >= 5]
    facts = []
    used = 0
    for i in range(n_on):
        S = chain_subjects[i] if i < len(chain_subjects) else subs[used]
        if i >= len(chain_subjects):
            used += 1
        Vv = values_pool[rng.randrange(len(values_pool))]
        facts.append({'S': S, 'value': Vv, 'sent': ON_THEME.format(S=S, V=Vv, D=domain), 'domain': domain, 'theme': 'org', 'chained': i < len(chain_subjects), 'wq_train': i % 2 == 0, 'fid': f'{domain}_on_{i}'})
    for i in range(n_off):
        S = subs[used]
        used += 1
        Vv = values_pool[rng.randrange(len(values_pool))]
        facts.append({'S': S, 'value': Vv, 'sent': OFF_THEME.format(S=S, V=Vv), 'domain': domain, 'theme': 'food', 'chained': False, 'wq_train': i % 2 == 0, 'fid': f'{domain}_off_{i}'})
    return facts

def hop_gate(bank_can, facts, thresh_mode='median'):
    q = bank_can.ctx_fp(HOP_QUERY)
    if q is None:
        q = bank_can.fp(['organization'])[0]
    scores = []
    for f in facts:
        k = bank_can.fp([f['S']])[0]
        c = bank_can.ctx_fp(f['sent'], exclude=f['value'])
        key = F.normalize(k + c, dim=-1) if c is not None else k
        scores.append(float((key * q).sum()))
    thresh = float(np.median(scores)) if thresh_mode == 'median' else 0.0
    admitted = [f for f, s in zip(facts, scores) if s >= thresh]
    for f, s in zip(facts, scores):
        if f['chained'] and f not in admitted:
            admitted.append(f)
    return (admitted, scores, thresh)

def inject_and_mask(lines: list[str], facts, rng: random.Random) -> tuple[str, str]:
    """Append fact sentences into real lines; masked copy replaces only those sentences with placeholders."""
    out = list(lines)
    masked = list(lines)
    if not out:
        raise RuntimeError('empty domain')
    stride = max(1, len(out) // max(1, len(facts)))
    for i, f in enumerate(facts):
        j = min(len(out) - 1, i * stride)
        out[j] = out[j] + ' ' + f['sent']
        masked[j] = masked[j] + ' ' + PLACEHOLDERS[i % len(PLACEHOLDERS)]
    return ('\n'.join(out), '\n'.join(masked))

def concat_corpora(per_domain: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """One global flat/offsets so replay is just a doc-id mix."""
    parts, offs, ranges = ([], [0], {})
    for d, (flat, off) in per_domain.items():
        start_doc = len(offs) - 1
        base = offs[-1]
        parts.append(flat)
        for i in range(len(off) - 1):
            offs.append(base + int(off[i + 1]))
        ranges[d] = (start_doc, len(offs) - 1)
    return (np.concatenate(parts), np.asarray(offs, dtype=np.int64), ranges)

def retrieve_value(bank_q, K, Vlist, subject: str, W=None) -> str:
    q = bank_q.ctx_fp(f'In the report {subject} was linked to the organization.')
    if q is None:
        q = bank_q.fp([subject])[0]
    if W is not None:
        q = F.normalize(W.map_raw(q.unsqueeze(0)), dim=-1)[0]
    return Vlist[int((K @ q).argmax())]

def two_hop_acc(bank_can, K, Vlist, chains, all_values, seed: int) -> dict:
    """S_a -> mid -> final over the accumulated tape (external fp loop, 203-style)."""
    if not chains:
        return {'strict': float('nan'), 'four_way': float('nan'), 'n': 0, 'hop1': float('nan')}
    rng = random.Random(seed)
    ok_strict = ok_4 = ok_h1 = 0
    for S_a, mid, final in chains:
        got_mid = retrieve_value(bank_can, K, Vlist, S_a)
        ok_h1 += int(got_mid == mid)
        got_final = retrieve_value(bank_can, K, Vlist, got_mid)
        ok_strict += int(got_final == final)
        others = [v for v in all_values if v != final]
        rng.shuffle(others)
        cands = [final] + others[:3]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        q = bank_can.ctx_fp(f'In the report {got_mid} was linked to the organization.')
        if q is None:
            q = bank_can.fp([got_mid])[0]
        sc = []
        for c in shuf:
            idxs = [j for j, v in enumerate(Vlist) if v == c]
            sc.append(float((K[idxs] @ q).max()) if idxs else -1.0)
        ok_4 += int(int(np.argmax(sc)) == order.index(0))
    n = len(chains)
    return {'strict': ok_strict / n, 'four_way': ok_4 / n, 'hop1': ok_h1 / n, 'n': n}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--operators-only', action='store_true', help='frozen upper; train W_query only (tape grows). Skips joint CE/CPC and per-domain arc/W_bwd.')
    ap.add_argument('--token-budget', type=int, default=0, help='CE tokens per domain (joint mode only)')
    ap.add_argument('--domains', type=str, default='wiki,stories,med,news')
    args = ap.parse_args()
    global DECISION, MINI
    if args.operators_only:
        tag = 'operators_smoke' if args.smoke else 'operators'
        DECISION = RES / f'stage254_decision_{tag}.json'
        MINI = RES / f'stage254_mini_{tag}.md'
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    tb = args.token_budget or (150000 if args.smoke else 4000000)
    n_on = 6 if args.smoke else 16
    n_off = 4 if args.smoke else 10
    n_chain = 2 if args.smoke else 6
    n_exam = 40 if args.smoke else 120
    n_probe = 24 if args.smoke else 60
    n_hold = 6 if args.smoke else 16
    n_probes = 4 if args.smoke else 8
    max_lines = 300 if args.smoke else 40000
    max_epochs = 3.0
    core_n = 50 if args.smoke else 250
    arc_steps = 40 if args.smoke else 300
    w_steps = 40 if args.smoke else 400
    log(f'Stage254 start {datetime.now(timezone.utc).isoformat()} domains={domains} budget/domain={tb} operators_only={args.operators_only}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    Vtok = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, Vtok).to(device)
    model_can = SelfModelXL(n_char, Vtok).to(device)
    model_can.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(2000000 if args.smoke else 8000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', wtext) if len(w) <= 14)))[:core_n]
    F_can = s221.fp_matrix(bank_can, core)
    facts_by_dom, admitted_by_dom, corpora = ({}, {}, {})
    prev_values = []
    gate_stats = {}
    for i, d in enumerate(domains):
        lines = domain_lines(d, max_lines, args.smoke, rng)
        chain_subjects = prev_values[:n_chain] if i > 0 else []
        facts = plant_facts(values_pool, n_on, n_off, d, chain_subjects, rng)
        admitted, scores, thresh = hop_gate(bank_can, facts)
        on_adm = sum((1 for f in admitted if f['theme'] == 'org'))
        off_adm = sum((1 for f in admitted if f['theme'] == 'food'))
        gate_stats[d] = {'n_facts': len(facts), 'admitted': len(admitted), 'on_theme_admitted': on_adm, 'off_theme_admitted': off_adm, 'thresh': thresh}
        raw, masked = inject_and_mask(lines, facts, rng)
        flat, off = s213.build_flat_from_text(masked, tok, pad_id, max_lines=max_lines, min_line_len=20)
        corpora[d] = (flat, off)
        facts_by_dom[d], admitted_by_dom[d] = (facts, admitted)
        prev_values = [f['value'] for f in facts if f['theme'] == 'org']
        log(f'  {d}: docs={len(off) - 1} facts={len(facts)} admitted={len(admitted)} (on={on_adm}/{n_on}, off={off_adm}/{n_off})')
    by_subject = {}
    for d in domains:
        for f in facts_by_dom[d]:
            by_subject.setdefault(f['S'], f)
    real_chains = []
    for d in domains:
        for f in facts_by_dom[d]:
            if f['theme'] != 'org':
                continue
            nxt = by_subject.get(f['value'])
            if nxt is not None and nxt['domain'] != f['domain']:
                real_chains.append((f['S'], f['value'], nxt['value']))
    log(f'cross-domain 2-hop chains: {len(real_chains)}')
    budgets = {}
    for d in domains:
        n_tok = int(len(corpora[d][0]))
        if n_tok < (20000 if args.smoke else 200000):
            raise RuntimeError(f'domain {d} too small ({n_tok} tokens) — fix the source before running')
        budgets[d] = int(min(tb, max_epochs * n_tok))
        log(f'  {d}: corpus_tokens={n_tok} budget={budgets[d]} (~{budgets[d] / n_tok:.1f} epochs)')
    flat_all, off_all, ranges = concat_corpora(corpora)
    splits = {}
    for d in domains:
        lo, hi = ranges[d]
        n_docs = hi - lo
        n_h = max(4, int(n_docs * 0.05))
        splits[d] = (list(range(lo, hi - n_h)), list(range(hi - n_h, hi)))
        log(f'  {d}: train_docs={len(splits[d][0])} hold_docs={len(splits[d][1])}')
    hold_batches = {d: s252.make_hold_batches(flat_all, off_all, splits[d][1], pad_id, n_hold, SEED + 5 + i) for i, d in enumerate(domains)}
    items = s251.load_exam_next(n_exam)
    items_probe = items[:n_probe]
    all_values_union = list(dict.fromkeys((f['value'] for d in domains for f in facts_by_dom[d])))
    K_all = torch.zeros(0, F_can.size(-1), device=device)
    V_all: list[str] = []
    base = {'exam_next_tok': s251.next_tok_acc(model_can, char_table, pad_id, items, device), 'uniformity': s252.uniformity(model_can, flat_all, off_all, char_table, pad_id, device, splits[domains[0]][1], 48, SEED + 9), 'hold_ce': {d: s252.fixed_hold_ce(model_can, hold_batches[d], char_table, pad_id, device) for d in domains}, 'leak': {d: s251.curve_param_recall(model_can, char_table, pad_id, tok, facts_by_dom[d], all_values_union, device, SEED + 300) for d in domains}}
    log(f"baseline exam_nt={base['exam_next_tok']:.3f} hold={ {k: round(v, 3) for k, v in base['hold_ce'].items()}} leak={ {k: round(v, 3) for k, v in base['leak'].items()}}")
    q_steps = 40 if args.smoke else 180
    W_query = L.init_query_adapter(device)
    m = model_can
    W_bwd, banks_q, first_seen, matrix = ({}, {}, {}, [])
    for i, d in enumerate(domains):
        log(f'\n=== PHASE {i + 1}/{len(domains)}: {d} ===')
        keys = []
        for f in admitted_by_dom[d]:
            k = bank_can.fp([f['S']])[0]
            c = bank_can.ctx_fp(f['sent'], exclude=f['value'])
            keys.append(F.normalize(k + c, dim=-1) if c is not None else k)
        if keys:
            K_all = torch.cat([K_all, torch.stack(keys, 0)], 0)
            V_all = V_all + [f['value'] for f in admitted_by_dom[d]]
        log(f'  slots: +{len(keys)} -> bank={len(V_all)}')
        if not args.operators_only:
            flat_d, off_d = corpora[d]
            model_shift = s221.finetune_arc_enc(model_can, flat_d, off_d, char_table, pad_id, device, arc_steps, SEED + 10 + i)
            bank_shift = FpBank(model_shift, stoi, device)
            W, align = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_shift, core), F_can, rng, w_steps, device)
            W_bwd[d], banks_q[d] = (W, bank_shift)
            log(f'  W[{d}] align={align:.3f}')
            train_docs = list(splits[d][0])
            if i > 0 and REPLAY_FRAC > 0:
                past = [x for e in domains[:i] for x in splits[e][0]]
                n_rep = min(len(past), int(len(train_docs) * REPLAY_FRAC / max(1e-06, 1 - REPLAY_FRAC)))
                train_docs = train_docs + random.Random(SEED + 77 + i).sample(past, n_rep)
                log(f'  replay: +{n_rep} past docs ({REPLAY_FRAC:.0%} target)')
            seen_hold = [b for e in domains[:i + 1] for b in hold_batches[e]]
            m, meta = s252.train_joint(m, flat_all, off_all, char_table, pad_id, device, budgets[d], LAM, SEED + 100 + i, f'phase_{d}', train_docs, seen_hold, items_probe, early_stop=False, n_probes=n_probes)
        else:
            banks_q[d] = bank_can
            W_bwd[d] = None
            meta = {'tokens_ce': 0, 'tokens_cpc': 0, 'steps': 0}
            log('  operators-only: upper frozen — skip arc/W_bwd and joint train')
        wq_fit = [f for dd in domains[:i + 1] for f in admitted_by_dom[dd] if f['wq_train']]
        if wq_fit and len(V_all) > 0:
            L.train_query_adapter(W_query, bank_can, wq_fit, K_all, V_all, device, q_steps, SEED + 400 + i)
        row = {'after_phase': d, 'bank_size': len(V_all), 'domains': {}}
        for e in domains[:i + 1]:
            hold = s252.fixed_hold_ce(m, hold_batches[e], char_table, pad_id, device)
            held_out = [f for f in admitted_by_dom[e] if not f['wq_train']]
            mem_q = L.tape_recall_decision(held_out, all_values_union, bank_can, K_all, V_all, SEED, W_bwd=W_query) if held_out else {'four_way': float('nan'), 'full_bank_top1': float('nan'), 'full_bank_mrr': float('nan'), 'full_bank_median_rank': float('nan')}
            mem_shift = L.tape_recall_decision(held_out, all_values_union, banks_q[e], K_all, V_all, SEED, W_bwd=W_bwd[e]) if held_out and W_bwd.get(e) is not None else {'four_way': float('nan'), 'full_bank_top1': float('nan'), 'full_bank_mrr': float('nan'), 'full_bank_median_rank': float('nan')}
            leak = s251.curve_param_recall(m, char_table, pad_id, tok, facts_by_dom[e], all_values_union, device, SEED + 300)
            row['domains'][e] = {'hold_ce': hold, 'hold_ppl': math.exp(min(hold, 20)), 'mem': mem_q, 'mem_shift': mem_shift, 'leak': leak}
            first_seen.setdefault(e, dict(row['domains'][e]))
        row['exam_next_tok'] = s251.next_tok_acc(m, char_table, pad_id, items, device)
        row['uniformity'] = s252.uniformity(m, flat_all, off_all, char_table, pad_id, device, splits[d][1], 48, SEED + 9)
        row['inversion'] = s251.inversion_fast(m, char_table, pad_id, tok, device)
        row['hop'] = two_hop_acc(bank_can, K_all, V_all, real_chains, all_values_union, SEED + i)
        row['train_meta'] = {k: meta[k] for k in ('tokens_ce', 'tokens_cpc', 'steps')}
        matrix.append(row)
        log(f"  after {d}: exam={row['exam_next_tok']:.3f} hop2={row['hop']['four_way']:.3f} bank={len(V_all)} " + ' '.join((f"{e}[ce={row['domains'][e]['hold_ce']:.3f} mem={row['domains'][e]['mem']:.2f}]" for e in domains[:i + 1])) + f' ({time.time() - t0:.0f}s)')
    final = matrix[-1]
    forget_vs_first = {e: final['domains'][e]['hold_ce'] - first_seen[e]['hold_ce'] for e in final['domains']}
    forget_vs_p1 = {e: final['domains'][e]['hold_ce'] - base['hold_ce'][e] for e in final['domains']}
    max_forget_first = max(forget_vs_first.values())
    max_forget_p1 = max(forget_vs_p1.values())
    min_mem = min((v['mem'] for v in final['domains'].values() if not math.isnan(v['mem'])))
    max_leak = max((v['leak'] for v in final['domains'].values()))
    max_leak_delta = max((final['domains'][e]['leak'] - base['leak'][e] for e in final['domains']))
    exam_curve = [r['exam_next_tok'] for r in matrix]
    g_no_forget = max_forget_p1 <= 0.15
    g_peak_regress = max_forget_first <= 0.15
    g_grow = final['exam_next_tok'] >= base['exam_next_tok'] - 0.01
    g_mem = min_mem >= 0.75
    g_leak = max_leak_delta <= 0.12
    g_hop = final['hop']['four_way'] >= 0.5
    g_collapse = final['uniformity'] <= base['uniformity'] + 0.1
    if g_no_forget and g_grow and g_mem and g_leak and g_hop:
        overall = 'CONTINUAL_UNDERSTAND_OK'
    elif g_no_forget and g_mem and g_leak and (g_grow or g_hop):
        overall = 'CONTINUAL_UNDERSTAND_PARTIAL'
    else:
        overall = 'CONTINUAL_UNDERSTAND_NO'
    out = {'stage': 254, 'mode': 'operators_only' if args.operators_only else 'joint_upper', 'overall': overall, 'domains': domains, 'lambda': LAM, 'replay_frac': REPLAY_FRAC, 'token_budget_per_domain': tb, 'budget_used_per_domain': budgets, 'max_epochs_per_domain': max_epochs, 'gates': {'G_no_forget_vs_P1': g_no_forget, 'G_peak_hold_regress': g_peak_regress, 'G_understanding_holds': g_grow, 'G_mem_holds_full_bank': g_mem, 'G_no_param_leak': g_leak, 'G_cross_domain_hop': g_hop, 'G_no_collapse': g_collapse}, 'summary': {'exam_curve': exam_curve, 'exam_base': base['exam_next_tok'], 'max_forget_hold_ce_vs_first_phase': max_forget_first, 'max_forget_hold_ce_vs_P1': max_forget_p1, 'forget_per_domain_vs_first': forget_vs_first, 'forget_per_domain_vs_P1': forget_vs_p1, 'min_mem_full_bank': min_mem, 'max_leak': max_leak, 'max_leak_delta_vs_P1': max_leak_delta, 'baseline_leak': base['leak'], 'hop_final': final['hop'], 'bank_size_final': final['bank_size']}, 'gate_stats': gate_stats, 'baseline': base, 'matrix': matrix, 'note': 'operators_only: frozen P1 upper; local-mask CE corpus unused for weight updates; only W_query trains on wq_train facts; tape grows via hop-gate. mem=canonical+W_q.' if args.operators_only else 'One shared upper across domains. Canonical slot KEYS frozen; W_query trains each phase on wq_train facts only, mem is scored on the held-out half. mem=canonical+W_q; mem_shift=arc-shift+W_bwd. Leak gate: delta vs P1 baseline (fixed seed), not absolute 0.40.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    lines_md = ['# Stage 254 continual understanding' + (' (operators-only)' if args.operators_only else ' (shared upper)') + '', f"**{overall}** domains={'->'.join(domains)} budget={tb} tok/domain", '', f"- exam: {base['exam_next_tok']:.3f} -> " + ' -> '.join((f'{x:.3f}' for x in exam_curve)), f"- max forget vs P1 (hold CE): {max_forget_p1:+.3f} | vs first phase: {max_forget_first:+.3f} | min mem @bank {final['bank_size']}: {min_mem:.3f} | max leak: {max_leak:.3f}", f"- cross-domain 2-hop: 4way={final['hop']['four_way']:.3f} strict={final['hop']['strict']:.3f} n={final['hop']['n']}", '', '| after \\ domain | ' + ' | '.join(domains) + ' |', '|' + '---|' * (len(domains) + 1)]
    for r in matrix:
        cells = []
        for e in domains:
            v = r['domains'].get(e)
            cells.append(f"ce {v['hold_ce']:.2f} / mem {v['mem']:.2f}" if v else '-')
        lines_md.append(f"| {r['after_phase']} | " + ' | '.join(cells) + ' |')
    MINI.write_text('\n'.join(lines_md) + '\n', encoding='utf-8')
    log(json.dumps({'overall': overall, 'exam_curve': exam_curve, 'hop': final['hop']}, indent=2))
    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save({'model': m.state_dict(), 'stage': 254, 'domains': domains, 'W_query': W_query.state_dict()}, CKPT_OUT)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())