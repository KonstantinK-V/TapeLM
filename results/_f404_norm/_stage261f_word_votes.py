"""
Stage 261f — One slot, one association: drop the averaging and drop the training.

  python _stage261f_word_votes.py [--smoke] [--soft] [--noise-typo 0.15]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage261_nl_query import collect, ctx_words, jaccard
import _stage24x_lib as L
RES = Path('results')
LOG = RES / '_stage261f_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
WORD_RE = re.compile('[A-Za-z][a-z]{2,}')
SEED = 2610
STOP = {'the', 'and', 'that', 'was', 'were', 'for', 'with', 'from', 'his', 'her', 'its', 'their', 'this', 'there', 'which', 'have', 'has', 'had', 'been', 'are', 'not', 'but', 'also', 'who', 'into', 'after', 'before', 'when', 'while', 'than', 'then', 'they', 'them', 'she', 'him'}

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)
from _tape_index import context_words, nway_strict, vote_arm_fields, vote_rank

def content(text: str, exclude: str | None=None, cap: int=40) -> list[str]:
    return context_words(text, exclude=exclude, cap=cap)

def typo(word: str, p: float, rng: random.Random) -> str:
    if p <= 0 or len(word) < 4:
        return word
    ws = list(word)
    for i in range(len(ws)):
        if rng.random() < p:
            ws[i] = rng.choice('abcdefghijklmnopqrstuvwxyz')
    return ''.join(ws)

def decision_paths(soft: bool, noise_typo: float) -> tuple[Path, Path]:
    tag = ''
    if soft:
        tag += '_soft'
    if noise_typo > 0:
        tag += f'_typo{int(round(noise_typo * 100)):03d}'
    dec = RES / (f'stage261f_decision{tag}.json' if tag else 'stage261f_decision.json')
    mini = RES / (f'stage261f_mini{tag}.md' if tag else 'stage261f_mini.md')
    return (dec, mini)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--entities', type=int, default=0)
    ap.add_argument('--distractor-entities', type=int, default=0)
    ap.add_argument('--soft', action='store_true')
    ap.add_argument('--soft-k', type=int, default=3)
    ap.add_argument('--noise-typo', type=float, default=0.0)
    ap.add_argument('--n-way', type=int, default=20)
    args = ap.parse_args()
    dec_path, mini_path = decision_paths(args.soft, args.noise_typo)
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_ent = args.entities or (60 if args.smoke else 400)
    n_dist = args.distractor_entities or (400 if args.smoke else 4000)
    max_lines = 3000 if args.smoke else 25000
    log(f'Stage261f word votes start {datetime.now(timezone.utc).isoformat()} device={device} soft={args.soft} typo={args.noise_typo}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bank = FpBank(model, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(3000000 if args.smoke else 20000000)
    lines = [l.strip() for l in wtext.split('\n') if 80 <= len(l.strip()) <= 400][:max_lines]
    cands = collect(lines, bank)
    ents = sorted(cands)[:n_ent]
    rng.shuffle(ents)
    log(f'  entities with >=2 natural mentions: {len(cands)} (using {len(ents)})')
    if len(ents) < 16:
        log('  not enough multi-mention entities')
        return 1
    postings: dict[str, list[int]] = defaultdict(list)
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
            for w in ws:
                postings[w].append(cid)
            used.add(e)
            if len(values) >= n_exam + n_dist:
                break
    vocab = sorted(postings)
    idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in vocab}
    med = float(np.median([it['overlap'] for it in items])) if items else 0.0
    log(f'  candidates={len(values)} ({n_exam} asked + {len(values) - n_exam} distractor) | vocab={len(vocab)} postings={sum((len(v) for v in postings.values()))} | eval={len(items)} overlap median={med:.3f}')
    if len(items) < 16:
        log('  not enough usable pairs')
        return 1
    Wfp = None
    if args.soft:
        Wfp = torch.stack([bank.fp([w])[0] for w in vocab], 0).to(device).float()
        log(f'  soft mode: word vocabulary embedded ({Wfp.shape[0]} x {Wfp.shape[1]})')

    def vote(qwords: list[str]) -> dict[int, float]:
        sc: dict[int, float] = defaultdict(float)
        for w in qwords:
            if args.soft and Wfp is not None:
                q = bank.fp([w])[0].to(device).float()
                sims, idx = torch.topk(Wfp @ q, min(args.soft_k, Wfp.size(0)))
                for s, j in zip(sims.tolist(), idx.tolist()):
                    if s <= 0:
                        continue
                    hit = vocab[j]
                    for cid in postings[hit]:
                        sc[cid] += s * idf[hit]
            else:
                for cid in postings.get(w, ()):
                    sc[cid] += idf[w]
        return sc
    nrng = random.Random(SEED + 3)
    wrng = random.Random(SEED + 5)
    ranks, nway, rows = ([], [], [])
    for it in items:
        qw = [typo(w, args.noise_typo, nrng) for w in it['qwords']]
        sc = vote(qw)
        gold, rank = vote_rank(sc, it['cid'], len(values))
        ranks.append(rank)
        pool = [j for j in wrng.sample(range(len(values)), min(args.n_way * 3, len(values))) if j != it['cid']][:args.n_way - 1]
        nway.append(int(nway_strict(gold, (sc.get(j, 0.0) for j in pool))))
        rows.append({'gold_score': float(gold), 'rank': rank, 'low_overlap': it['overlap'] <= med})
    silence = vote_arm_fields(rows)
    r = np.asarray(ranks, dtype=np.float64)
    res = {'top1': float(np.mean(r == 1)), 'mrr': float(np.mean(1.0 / r)), 'median_rank': float(np.median(r)), f'acc_{args.n_way}way': float(np.mean(nway)), f'chance_{args.n_way}way': 1.0 / args.n_way, 'top1_low_overlap': silence['top1_low_overlap'], 'top1_high_overlap': silence['top1_high_overlap'], 'tie_at_zero_frac': silence['tie_at_zero_frac'], 'n': len(ranks)}
    log('votes: ' + json.dumps(res))
    log('silence: ' + json.dumps(silence))
    perm = list(range(len(values)))
    random.Random(SEED + 7).shuffle(perm)
    post_shuf = {w: [perm[c] for c in v] for w, v in postings.items()}
    sh_ranks, sh_nway, sh_rows = ([], [], [])
    wrng2 = random.Random(SEED + 5)
    for it in items:
        sc: dict[int, float] = defaultdict(float)
        for w in it['qwords']:
            for cid in post_shuf.get(w, ()):
                sc[cid] += idf[w]
        gold, rank = vote_rank(sc, it['cid'], len(values))
        sh_ranks.append(rank)
        pool = [j for j in wrng2.sample(range(len(values)), min(args.n_way * 3, len(values))) if j != it['cid']][:args.n_way - 1]
        sh_nway.append(int(nway_strict(gold, (sc.get(j, 0.0) for j in pool))))
        sh_rows.append({'gold_score': float(gold), 'rank': rank, 'low_overlap': it['overlap'] <= med})
    pop_silence = vote_arm_fields(sh_rows)
    pop_floor = {'top1': float(np.mean(np.asarray(sh_ranks) == 1)), f'acc_{args.n_way}way': float(np.mean(sh_nway)), 'tie_at_zero_frac': pop_silence['tie_at_zero_frac'], 'silence': pop_silence}
    log('popularity floor (postings repointed, counts preserved): ' + json.dumps({k: v for k, v in pop_floor.items() if k != 'silence'}))
    nw = res[f'acc_{args.n_way}way']
    ch = 1.0 / args.n_way
    g_beats_mean_fp = nw >= 0.3
    g_signal = nw >= ch + 0.1
    g_low_overlap = not math.isnan(silence['top1_low_overlap_given_vote']) and silence['top1_low_overlap_given_vote'] > 0.0
    g_causal = pop_floor['top1'] <= 0.02
    g_beats_popularity = nw >= pop_floor[f'acc_{args.n_way}way'] + 0.15
    g_open_top1 = res['top1'] >= 0.3
    if g_causal and g_beats_popularity and g_open_top1 and g_low_overlap:
        overall = 'WORD_VOTES_OK'
    elif g_causal and g_beats_popularity and g_beats_mean_fp:
        overall = 'WORD_VOTES_BEATS_MEAN'
    elif g_causal and g_signal:
        overall = 'WORD_VOTES_SIGNAL_ONLY'
    else:
        overall = 'WORD_VOTES_NO'
    out = {'stage': '261f', 'overall': overall, 'soft': args.soft, 'soft_k': args.soft_k, 'noise_typo': args.noise_typo, 'n_way': args.n_way, 'candidates': len(values), 'asked': n_exam, 'distractors': len(values) - n_exam, 'vocab': len(vocab), 'postings': sum((len(v) for v in postings.values())), 'overlap_median': med, 'trained_parameters': 0, 'fp_version': L.canonical_fp_version(), 'read': {f'acc_{args.n_way}way': res.get(f'acc_{args.n_way}way'), 'full_bank_top1': res['top1'], 'full_bank_mrr': res['mrr'], 'full_bank_median_rank': res['median_rank'], 'tie_at_zero_frac': silence['tie_at_zero_frac'], 'top1_low_overlap': silence['top1_low_overlap'], 'top1_low_overlap_given_vote': silence['top1_low_overlap_given_vote'], 'low_overlap_miss_is_silence_frac': silence['low_overlap_miss_is_silence_frac']}, 'silence': silence, 'gates': {'G_signal': g_signal, 'G_beats_mean_fp': g_beats_mean_fp, 'G_low_overlap_works': g_low_overlap, 'G_open_top1': g_open_top1, 'G_causal_top1': g_causal, 'G_beats_popularity_20way': g_beats_popularity, 'G_low_overlap_uses': 'top1_low_overlap_given_vote'}, 'summary': {'votes': res, 'popularity_floor': pop_floor, 'reference_261_ctx_fp_mean': {'acc_20way': 0.226, 'top1': 0.034, 'top1_low_overlap': 0.0}}, 'note': 'Zero-train word postings + idf. n-way pessimistic (gold > distractor). silence.tie_at_zero_frac is permanent: share of queries with gold score 0 (index silent). top1_low_overlap hole is mostly silence — see low_overlap_miss_is_silence_frac and top1_low_overlap_given_vote. Causal read still on top1.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    dec_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    mini_path.write_text(f"# Stage 261f word votes (zero-train)\n\n**{overall}** candidates={len(values)} vocab={len(vocab)} soft={args.soft} typo={args.noise_typo}\n\n- {args.n_way}-way **{nw:.3f}** (chance {ch:.3f}, popularity floor {pop_floor[f'acc_{args.n_way}way']:.3f}) vs 261 ctx_fp mean **0.226**\n- open top1 **{res['top1']:.3f}** vs popularity floor **{pop_floor['top1']:.3f}** (261 mean: 0.034), mrr {res['mrr']:.3f}, median rank {res['median_rank']:.0f}\n- by overlap: low **{silence['top1_low_overlap']:.3f}** vs high **{silence['top1_high_overlap']:.3f}**\n- **silence:** tie_at_zero **{silence['tie_at_zero_frac']:.3f}** (low-ov **{silence['tie_at_zero_frac_low_overlap']:.3f}** / high **{silence['tie_at_zero_frac_high_overlap']:.3f}**); low-ov miss is silence **{silence['low_overlap_miss_is_silence_frac']:.3f}**; top1 low-ov | gold>0 **{silence['top1_low_overlap_given_vote']:.3f}**\n- trained parameters: **0**\n", encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())