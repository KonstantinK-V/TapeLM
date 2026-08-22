"""
Stage 186 — Exam v2: kill the unigram shortcut.

v1 flaw (found in 185): next_tok gold is usually a frequent token, distractors random
→ pure frequency scores 0.65. v2: distractors are FREQUENCY-MATCHED to gold
(nearest ranks in the corpus frequency table). Entity distractors matched by
entity-frequency rank too.

Gates:
  EXAM2_OK  = unigram <= chance+0.10 (shortcut dead) AND GPT >= chance+0.20 (still calibrated)

Systems: unigram, random, ce_gpt_181, endpoint_185, tape_185.

  python _stage186_exam_v2.py
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
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage181_ce_control as s181
import _stage184_exam_logprob as s184
import _stage185_tape_read as s185
RES = Path('results')
DATA = Path('data')
CKPT = Path('checkpoints')
LOG = RES / '_stage186_log.txt'
EXAM = DATA / 'stage186_exam_v2.jsonl'
DECISION = RES / 'stage186_decision.json'
MINI = RES / 'stage186_mini.md'
TOK_PATH = s177.TOK_PATH
SEED = 186
CTX_TOK = 40
N_NEXT = 150
N_ENTITY = 100
N_OOD = 60
N_CAND = 4
EXAM_CHARS = 800000
FREQ_WINDOW = 40
PAD = '[PAD]'
ENT_RE = re.compile('\\b([A-Z][a-z]{3,}|\\d{3,4})\\b')

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def freq_matched_pool(sorted_ids: list[int], rank_of: dict[int, int], gold: int, rng: random.Random, k: int):
    """k distractors with frequency rank near gold's."""
    r = rank_of[gold]
    lo = max(0, r - FREQ_WINDOW)
    hi = min(len(sorted_ids), r + FREQ_WINDOW + 1)
    window = [t for t in sorted_ids[lo:hi] if t != gold]
    rng.shuffle(window)
    return window[:k]

def build_exam_v2(text: str, tok: Tokenizer, pad_id: int, freq: np.ndarray, rng: random.Random) -> list[dict]:
    paras = [p.strip() for p in text.split('\n') if 120 < len(p.strip()) < 1000][:1200]
    hold = paras[int(0.8 * len(paras)):]
    train_blob = ' '.join(paras[:int(0.8 * len(paras))])[:200000].lower()
    log(f'  paras={len(paras)} hold={len(hold)}')
    sorted_ids = list(np.argsort(-freq))
    sorted_ids = [int(t) for t in sorted_ids if int(t) != pad_id]
    rank_of = {t: i for i, t in enumerate(sorted_ids)}
    items = []
    n_next = 0
    for p in hold * 3:
        if n_next >= N_NEXT:
            break
        ids = [i for i in tok.encode(p).ids if i != pad_id]
        if len(ids) < CTX_TOK + 2:
            continue
        pos = rng.randint(CTX_TOK, len(ids) - 2)
        ctx_ids = ids[max(0, pos - CTX_TOK):pos]
        gold = ids[pos]
        ds = freq_matched_pool(sorted_ids, rank_of, gold, rng, N_CAND - 1)
        if len(ds) < N_CAND - 1:
            continue
        cands = [[gold]] + [[d] for d in ds]
        order = list(range(len(cands)))
        rng.shuffle(order)
        items.append({'type': 'next_tok', 'ctx_ids': ctx_ids, 'cand_ids': [cands[k] for k in order], 'gold_idx': order.index(0)})
        n_next += 1
    ent_count: dict[str, int] = {}
    for p in paras:
        for m in ENT_RE.finditer(p):
            ent_count[m.group(1)] = ent_count.get(m.group(1), 0) + 1
    ents_sorted = [e for e, _ in sorted(ent_count.items(), key=lambda kv: -kv[1])]
    ent_rank = {e: i for i, e in enumerate(ents_sorted)}
    n_ent = 0
    for p in hold * 2:
        if n_ent >= N_ENTITY:
            break
        m = ENT_RE.search(p, 60)
        if not m:
            continue
        gold_str = m.group(1)
        if gold_str not in ent_rank:
            continue
        ctx_ids = [i for i in tok.encode(p[:m.start()]).ids if i != pad_id][-CTX_TOK:]
        if len(ctx_ids) < 8:
            continue
        r = ent_rank[gold_str]
        lo, hi = (max(0, r - FREQ_WINDOW), min(len(ents_sorted), r + FREQ_WINDOW + 1))
        pool = [e for e in ents_sorted[lo:hi] if e != gold_str]
        rng.shuffle(pool)
        ds = pool[:N_CAND - 1]
        gold_ids = [i for i in tok.encode(' ' + gold_str).ids if i != pad_id]
        if len(ds) < N_CAND - 1 or not gold_ids:
            continue
        cand_ids = [gold_ids] + [[i for i in tok.encode(' ' + d).ids if i != pad_id] for d in ds]
        order = list(range(len(cand_ids)))
        rng.shuffle(order)
        items.append({'type': 'entity', 'ctx_ids': ctx_ids, 'cand_ids': [cand_ids[k] for k in order], 'gold_idx': order.index(0)})
        n_ent += 1
    fakes = [f for f in ['Zorblax', 'Quenith', 'Marbune', 'Xaldera', 'Kessari', 'Vornak', 'Talmidex', 'Orsiphon', 'Pholmar', 'Girenth'] if f.lower() not in train_blob]
    n_ood = 0
    for p in hold:
        if n_ood >= N_OOD or len(fakes) < N_CAND:
            break
        m = ENT_RE.search(p, 60)
        if not m:
            continue
        ctx_ids = [i for i in tok.encode(p[:m.start()]).ids if i != pad_id][-CTX_TOK:]
        if len(ctx_ids) < 8:
            continue
        picks = rng.sample(fakes, N_CAND)
        cand_ids = [[i for i in tok.encode(' ' + w).ids if i != pad_id] for w in picks]
        items.append({'type': 'ood', 'ctx_ids': ctx_ids, 'cand_ids': cand_ids, 'gold_idx': rng.randint(0, N_CAND - 1)})
        n_ood += 1
    return items

def score_with(scorer, items) -> dict:
    acc = {}
    for it in items:
        scores = [scorer(it['ctx_ids'], c) for c in it['cand_ids']]
        pred = int(np.argmax(scores))
        ok, n = acc.get(it['type'], (0, 0))
        acc[it['type']] = (ok + int(pred == it['gold_idx']), n + 1)
    out = {}
    for t, (ok, n) in acc.items():
        out[f'{t}_acc'] = ok / max(1, n)
        out[f'{t}_n'] = n
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage186 start {datetime.now(timezone.utc).isoformat()}')
    log('Exam v2: frequency-matched distractors')
    rng = random.Random(SEED)
    device = torch.device(args.device)
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    text = s170.load_corpus(max_chars=20000000)
    chars = sorted(set(text) | {' '})
    itos = ['<pad>'] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = s181.build_id_docs(tok, text)
    train_docs = docs[:int(0.8 * len(docs))] or docs
    freq = np.ones(V)
    for doc in train_docs:
        for t in doc:
            freq[t] += 1
    log(f'docs={len(docs)} V={V}')
    items = build_exam_v2(text[:EXAM_CHARS], tok, pad_id, freq, rng)
    with EXAM.open('w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    counts = {t: sum((1 for i in items if i['type'] == t)) for t in ('next_tok', 'entity', 'ood')}
    log(f'exam v2 n={len(items)} {counts}')
    results = {}
    logfreq = np.log(freq / freq.sum())
    results['unigram'] = score_with(lambda c, cd: float(np.mean([logfreq[t] for t in cd])), items)
    rb = random.Random(0)
    results['random'] = score_with(lambda c, cd: rb.random(), items)
    log('score ce_gpt_181 …')
    gpt = s184.load_gpt(device)
    results['ce_gpt_181'] = score_with(lambda c, cd: s184.gpt_span_logprob(gpt, c, cd, device), items)
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    for mode in ('endpoint', 'tape'):
        path = CKPT / f'stage185_{mode}.pt'
        if not path.exists():
            log(f'skip {mode} (no ckpt)')
            continue
        log(f'score {mode}_185 …')
        m = s185.TapeReadModel(len(itos), V, mode).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
        m.eval()
        results[f'{mode}_185'] = score_with(lambda c, cd, _m=m: s185.span_logprob(_m, char_table, pad_id, c, cd, device), items)
        if mode == 'tape':
            shuf = score_with(lambda c, cd, _m=m: s185.span_logprob(_m, char_table, pad_id, c, cd, device, shuffle_tape=True), [it for it in items if it['type'] == 'next_tok'])
            results['tape_185']['next_tok_shuffled'] = shuf['next_tok_acc']
            log(f"  tape shuffle ablation: next_tok={shuf['next_tok_acc']:.3f}")
    chance = 1.0 / N_CAND
    for name, r in results.items():
        log(f"  {name}: next_tok={r.get('next_tok_acc', 0):.3f} entity={r.get('entity_acc', 0):.3f} ood={r.get('ood_acc', 0):.3f}")
    uni_ok = results['unigram']['next_tok_acc'] <= chance + 0.1
    gpt_ok = results['ce_gpt_181']['next_tok_acc'] >= chance + 0.2
    overall = 'EXAM2_OK' if uni_ok and gpt_ok else 'EXAM2_SHORTCUT_ALIVE' if not uni_ok else 'EXAM2_GPT_LOST_SIGNAL'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'exam_v2_freq_matched_186', 'overall': overall, 'chance': chance, 'gates': {'unigram<=chance+0.10': uni_ok, 'gpt>=chance+0.20': gpt_ok}, 'counts': counts, 'results': results, 'note': 'context credit is now score minus chance (unigram shortcut removed)'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Stage186 — Exam v2 (freq-matched distractors)', '', f'**Overall:** `{overall}`  chance={chance:.2f}', '']
    for name, r in results.items():
        lines.append(f"- `{name}`: next_tok={r.get('next_tok_acc', 0):.3f} entity={r.get('entity_acc', 0):.3f} ood={r.get('ood_acc', 0):.3f}")
    MINI.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log(f'[186] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())