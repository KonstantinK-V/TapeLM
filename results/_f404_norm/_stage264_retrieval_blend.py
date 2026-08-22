"""
Stage 264 — Balance alternatives: flat mean, idf-mean, votes, cascade, fusion.

Open-bank retrieval (261f exam, zero train) compares all modes on the same items.
Optional --glue runs 256 decode EM per mode (263 harness).

  python _stage264_retrieval_blend.py [--smoke] [--glue]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage24x_lib as L
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage261_nl_query import collect, ctx_words, jaccard
from _stage261f_word_votes import content
from _retrieval_modes import CASCADE_POOL, FUSION_LAM, cascade_order, cosine_scores, ctx_vector, fusion_scores, rank_from_order, rank_from_scores, vote_scores
from _tape_index import nway_strict, vote_arm_fields
RES = Path('results')
DECISION = RES / 'stage264_decision.json'
MINI = RES / 'stage264_mini.md'
LOG = RES / '_stage264_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 2640
MODES = ('mean', 'idf_mean', 'votes', 'cascade', 'fusion')

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def build_open_bank(bank: FpBank, smoke: bool):
    rng = random.Random(SEED)
    n_ent = 60 if smoke else 400
    n_dist = 400 if smoke else 4000
    max_lines = 3000 if smoke else 25000
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(3000000 if smoke else 20000000)
    lines = [l.strip() for l in wtext.split('\n') if 80 <= len(l.strip()) <= 400][:max_lines]
    cands = collect(lines, bank)
    ents = sorted(cands)[:n_ent]
    rng.shuffle(ents)
    postings: dict[str, list[int]] = defaultdict(list)
    ctxw: list[list[str]] = []
    values: list[str] = []
    items = []
    for e in ents:
        occ = cands[e]
        a, b = (occ[0], occ[1])
        wctx = a['line'][max(0, a['start'] - 140):min(len(a['line']), a['end'] + 140)]
        qtext = b['line'][max(0, b['start'] - 200):b['start']].strip()
        ws = content(wctx, exclude=e)
        qs = content(qtext, exclude=e)
        if len(ws) < 4 or len(qs) < 4:
            continue
        cid = len(values)
        values.append(e)
        ctxw.append(ws)
        for w in ws:
            postings[w].append(cid)
        items.append({'ent': e, 'cid': cid, 'qwords': qs, 'overlap': jaccard(ctx_words(wctx, e), ctx_words(qtext, e))})
    n_exam = len(values)
    used = set(values)
    for ln in lines:
        if len(values) >= n_exam + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = (max(0, m.start() - 140), min(len(ln), m.end() + 140))
            ws = content(ln[lo:hi], exclude=e)
            if len(ws) < 4:
                continue
            cid = len(values)
            values.append(e)
            ctxw.append(ws)
            for w in ws:
                postings[w].append(cid)
            used.add(e)
            if len(values) >= n_exam + n_dist:
                break
    vocab = sorted(postings)
    idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in vocab}
    return (items, ctxw, postings, idf, values, n_exam)

@torch.no_grad()
def precompute_keys(bank: FpBank, ctxw: list[list[str]], idf: dict[str, float], device):
    mean_k, idf_k = ([], [])
    for ws in ctxw:
        mean_k.append(ctx_vector(bank, ws, None))
        idf_k.append(ctx_vector(bank, ws, idf))
    z = mean_k[0] if mean_k[0] is not None else torch.zeros(256, device=device)
    K_mean = torch.stack([k if k is not None else z for k in mean_k]).float().to(device)
    K_idf = torch.stack([k if k is not None else z for k in idf_k]).float().to(device)
    return (K_mean, K_idf)

def eval_mode(mode: str, items, ctxw, postings, idf, K_mean, K_idf, bank, n_way: int, med: float):
    n = len(ctxw)
    wrng = random.Random(SEED + 5)
    ranks, nway, rows = ([], [], [])
    for it in items:
        qw = it['qwords']
        gold = it['cid']
        vsc = vote_scores(qw, postings, idf)
        q_mean = ctx_vector(bank, qw, None)
        q_idf = ctx_vector(bank, qw, idf)
        low = it['overlap'] <= med
        if mode == 'mean':
            sc = cosine_scores(q_mean, K_mean)
        elif mode == 'idf_mean':
            sc = cosine_scores(q_idf, K_idf)
        elif mode == 'votes':
            sc = vsc
        elif mode == 'cascade':
            order = cascade_order(vsc, q_mean, K_mean, n, CASCADE_POOL)
            rank = rank_from_order(order, gold)
            ranks.append(rank)
            pool = [j for j in wrng.sample(range(n), min(n_way * 3, n)) if j != gold][:n_way - 1]
            if float(vsc.get(gold, 0.0)) <= 0.0:
                nway.append(0)
            else:
                sc = {cid: -float(i + 1) for i, cid in enumerate(order)}
                g = sc[gold]
                nway.append(int(nway_strict(g, (sc.get(j, -float(n + 1)) for j in pool))))
            rows.append({'gold_score': float(vsc.get(gold, 0.0)), 'rank': rank, 'low_overlap': low})
            continue
        elif mode == 'fusion':
            cos_sc = cosine_scores(q_mean, K_mean)
            sc = fusion_scores(vsc, cos_sc, FUSION_LAM)
        else:
            raise ValueError(mode)
        rank = rank_from_scores(sc, gold, n)
        ranks.append(rank)
        pool = [j for j in wrng.sample(range(n), min(n_way * 3, n)) if j != gold][:n_way - 1]
        g = sc.get(gold, 0.0)
        nway.append(int(nway_strict(g, (sc.get(j, 0.0) for j in pool))))
        silence_score = float(vsc.get(gold, 0.0)) if mode in ('votes', 'fusion') else float(g)
        rows.append({'gold_score': silence_score, 'rank': rank, 'low_overlap': low})
    silence = vote_arm_fields(rows)
    r = np.asarray(ranks, dtype=np.float64)
    return {'top1': float(np.mean(r == 1)), 'mrr': float(np.mean(1.0 / r)), 'median_rank': float(np.median(r)), f'acc_{n_way}way': float(np.mean(nway)), f'chance_{n_way}way': 1.0 / n_way, 'top1_low_overlap': silence['top1_low_overlap'], 'top1_high_overlap': silence['top1_high_overlap'], 'tie_at_zero_frac': silence['tie_at_zero_frac'], 'silence': silence, 'n': len(ranks)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--glue', action='store_true', help='256 decode EM per mode (slow)')
    ap.add_argument('--n-way', type=int, default=20)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--topk', type=int, default=8)
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    log(f'Stage264 retrieval blend start {datetime.now(timezone.utc).isoformat()} device={device}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bank = FpBank(model, stoi, device)
    items, ctxw, postings, idf, values, n_exam = build_open_bank(bank, args.smoke)
    if len(items) < 16:
        log('not enough eval pairs')
        return 1
    med = float(np.median([it['overlap'] for it in items]))
    log(f'  candidates={len(values)} ({n_exam} exam) eval={len(items)} overlap_med={med:.3f}')
    K_mean, K_idf = precompute_keys(bank, ctxw, idf, device)
    results = {}
    for mode in MODES:
        results[mode] = eval_mode(mode, items, ctxw, postings, idf, K_mean, K_idf, bank, args.n_way, med)
        log(f'  [{mode}] ' + json.dumps(results[mode]))
    ref_votes = results['votes'][f'acc_{args.n_way}way']
    ref_mean = results['mean'][f'acc_{args.n_way}way']
    best_mode_20way = max(MODES, key=lambda m: results[m][f'acc_{args.n_way}way'])
    best_nw = results[best_mode_20way][f'acc_{args.n_way}way']
    best_mode = max(MODES, key=lambda m: (results[m]['top1'], results[m][f'acc_{args.n_way}way']))
    g_idf_beats_flat = results['idf_mean'][f'acc_{args.n_way}way'] >= ref_mean + 0.05
    g_blend_beats_votes = any((results[m]['top1'] >= results['votes']['top1'] + 0.03 for m in ('cascade', 'fusion')))
    g_cascade_beats_votes = results['cascade']['top1'] >= results['votes']['top1'] + 0.03
    if g_blend_beats_votes:
        overall = 'BLEND_BEATS_SINGLE'
    elif results['idf_mean'][f'acc_{args.n_way}way'] >= ref_mean + 0.1:
        overall = 'IDF_MEAN_FIXES_FLAT'
    elif best_mode == 'votes' and results['votes']['top1'] >= results['mean']['top1'] + 0.05:
        overall = 'VOTES_BEST_OPEN_BANK'
    else:
        overall = 'NO_CLEAR_WINNER'
    glue_summary = None
    if args.glue:
        import _stage263_votes_vs_mean as s263
        glue_summary = {}
        steps = args.steps or (200 if args.smoke else 800)
        glue_modes = ('cosine', 'idf_mean', 'votes', 'cascade', 'fusion')
        for mode in glue_modes:
            glue_summary[mode] = s263.run_single_mode(mode, device, args.smoke, steps, args.topk)
        log('  glue summary: ' + json.dumps(glue_summary))
    out = {'stage': 264, 'overall': overall, 'modes': list(MODES), 'fusion_lambda': FUSION_LAM, 'cascade_pool': CASCADE_POOL, 'candidates': len(values), 'n_exam': n_exam, 'eval_pairs': len(items), 'gates': {'G_idf_mean_beats_flat_mean': g_idf_beats_flat, 'G_blend_beats_votes': g_blend_beats_votes, 'G_cascade_beats_votes': g_cascade_beats_votes}, 'summary': {'retrieval': results, 'read_full_bank': {m: {'full_bank_top1': results[m]['top1'], 'full_bank_median_rank': results[m]['median_rank'], f'acc_{args.n_way}way': results[m].get(f'acc_{args.n_way}way'), 'tie_at_zero_frac': results[m].get('tie_at_zero_frac'), 'top1_low_overlap_given_vote': results[m].get('silence', {}).get('top1_low_overlap_given_vote'), 'low_overlap_miss_is_silence_frac': results[m].get('silence', {}).get('low_overlap_miss_is_silence_frac')} for m in MODES}, 'best_mode_top1': best_mode, 'best_mode_20way': best_mode_20way, 'reference_261f': {'votes_20way_strict': 0.432, 'votes_20way_legacy_ge': 0.601, 'mean_20way': 0.226, 'tie_at_zero_frac': 0.488}}, 'fp_version': L.canonical_fp_version(), 'glue_em': glue_summary, 'note': 'n-way pessimistic via nway_strict for ALL modes (cascade: score=-rank). Overall uses top1 as headline (20-way alone had crowned cascade after a metric bug). silence.* permanent. Low-overlap hole ≈ silence → route to sem-q (258) when votes silent.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    lines = [f'# Stage 264 retrieval blend\n\n**{overall}** eval={len(items)} candidates={len(values)}\n\n']
    for mode in MODES:
        r = results[mode]
        s = r.get('silence') or {}
        lines.append(f"- **{mode}**: 20-way **{r[f'acc_{args.n_way}way']:.3f}** top1 **{r['top1']:.3f}** low-ov **{r['top1_low_overlap']:.3f}** tie0 **{r.get('tie_at_zero_frac', float('nan')):.3f}** miss=silence **{s.get('low_overlap_miss_is_silence_frac', float('nan')):.3f}**\n")
    lines.append(f"\nBest top1: **{best_mode}** ({results[best_mode]['top1']:.3f}) · best 20-way: **{best_mode_20way}** ({best_nw:.3f})\n")
    MINI.write_text(''.join(lines), encoding='utf-8')
    log(json.dumps({'overall': overall, 'best_top1': best_mode, 'best_20way': best_mode_20way}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())