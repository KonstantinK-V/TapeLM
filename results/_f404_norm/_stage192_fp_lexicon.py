"""
Stage 192 — FP-lexicon lexical surprise (old word-fingerprint theory × curve).

Idea: "I don't know this word" = "its fingerprint is in no one's lexicon", not
"its letters look odd". Zero training:
  - freeze night P1 curve-XL; fp(word) = normalize(arc_enc(word chars))
  - lexicon = fps of words seen in the 150M train corpus (count>=MIN_COUNT)
  - lex_surprise(w) = 1 - max_cos(fp(w), lexicon)

The gate that never passed (187/189/191-P3): surprise(fake) > surprise(real).
Here: real = entity words from exam v3 (seen in corpus), fake = generated
pseudo-words verified absent from corpus. Report means + AUC.
Baseline comparison: char-trigram novelty (the P3 rarity signal) on same words.

  python _stage192_fp_lexicon.py
"""
from __future__ import annotations
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
RES = Path('results')
DATA = Path('data')
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
DECISION = RES / 'stage192_decision.json'
MINI = RES / 'stage192_mini.md'
LOG = RES / '_stage192_log.txt'
EXAM_V3 = DATA / 'stage191_exam_v3.jsonl'
SEED = 192
CORPUS_CHARS = 150000000
MIN_COUNT = 2
MAX_LEX = 200000
MAX_CHARS = s177.MAX_CHARS_PER_ARC
N_FAKE = 150

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def gen_fakes(word_set: set[str], rng: random.Random, n: int) -> list[str]:
    """pronounceable pseudo-words, verified absent from corpus (any case)."""
    cons = 'bcdfghklmnprstvz'
    vow = 'aeiou'
    out = []
    tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        syls = rng.randint(2, 4)
        w = ''.join((rng.choice(cons) + rng.choice(vow) + (rng.choice(cons) if rng.random() < 0.4 else '') for _ in range(syls)))
        w = w.capitalize()
        if w.lower() not in word_set and w not in word_set and (5 <= len(w) <= 12) and (w not in out):
            out.append(w)
    return out

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) rank AUC."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage192 start {datetime.now(timezone.utc).isoformat()}')
    log('FP-lexicon lexical surprise on frozen P1 curve-XL')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    t0 = time.time()
    log('reading corpus words …')
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(CORPUS_CHARS)
    words = re.findall('[A-Za-z][a-z]+', text)
    cnt = Counter(words)
    word_set = set(cnt.keys()) | {w.lower() for w in cnt}
    lex_words = [w for w, c in cnt.most_common(MAX_LEX) if c >= MIN_COUNT]
    log(f'  unique={len(cnt)} lexicon={len(lex_words)} ({time.time() - t0:.0f}s)')
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model.eval()

    @torch.no_grad()
    def fp_batch(ws: list[str]) -> torch.Tensor:
        rows = torch.zeros(len(ws), 1, MAX_CHARS, dtype=torch.long)
        for i, w in enumerate(ws):
            for j, c in enumerate(w[:MAX_CHARS]):
                rows[i, 0, j] = stoi.get(c, 0)
        arcs = model.arc_enc(rows.to(device))[:, 0]
        return F.normalize(arcs, dim=-1)
    log('encoding lexicon fps …')
    lex_fps = []
    for i in range(0, len(lex_words), 4096):
        lex_fps.append(fp_batch(lex_words[i:i + 4096]))
    lex = torch.cat(lex_fps, 0)
    log(f'  lexicon fps={tuple(lex.shape)} ({time.time() - t0:.0f}s)')
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding='utf-8').splitlines()]
    real_words = []
    for it in items:
        if it['type'] != 'entity':
            continue
        s = tok.decode(it['cand_ids'][it['gold_idx']], skip_special_tokens=False).strip()
        w = re.findall('[A-Za-z][a-z]+', s)
        if w and w[0] in cnt:
            real_words.append(w[0])
    real_words = list(dict.fromkeys(real_words))
    fakes = gen_fakes(word_set, rng, N_FAKE)
    log(f'real entities={len(real_words)} fakes={len(fakes)}')

    @torch.no_grad()
    def lex_surprise(ws: list[str], exclude_self: bool=False) -> np.ndarray:
        fps = fp_batch(ws)
        sims = fps @ lex.T
        if exclude_self:
            top2 = sims.topk(2, dim=-1).values
            best = torch.where(top2[:, 0] > 0.999, top2[:, 1], top2[:, 0])
        else:
            best = sims.max(dim=-1).values
        return (1.0 - best).cpu().numpy()
    s_real = lex_surprise(real_words)
    s_fake = lex_surprise(fakes)
    tri = Counter()
    for w, c in cnt.most_common(100000):
        for i in range(len(w) - 2):
            tri[w[i:i + 3].lower()] += c
    tot = sum(tri.values()) or 1

    def tri_novelty(ws):
        out = []
        for w in ws:
            ts = [w[i:i + 3].lower() for i in range(len(w) - 2)]
            out.append(float(np.mean([-np.log((tri.get(t, 0) + 1) / (tot + 1)) for t in ts])) if ts else 0.0)
        return np.array(out)
    t_real, t_fake = (tri_novelty(real_words), tri_novelty(fakes))
    res = {'fp_lexicon': {'surprise_real_mean': float(s_real.mean()), 'surprise_fake_mean': float(s_fake.mean()), 'gate_fake_gt_real': bool(s_fake.mean() > s_real.mean()), 'auc': auc(s_fake, s_real)}, 'trigram_baseline': {'novelty_real_mean': float(t_real.mean()), 'novelty_fake_mean': float(t_fake.mean()), 'gate_fake_gt_real': bool(t_fake.mean() > t_real.mean()), 'auc': auc(t_fake, t_real)}}
    diag = []
    with torch.no_grad():
        fps = fp_batch(fakes[:8])
        sims = fps @ lex.T
        vals, idx = sims.max(dim=-1)
        for k in range(len(fps)):
            diag.append(f'{fakes[k]} → nn={lex_words[int(idx[k])]} cos={float(vals[k]):.3f}')
    for d in diag:
        log('  ' + d)
    g_fp = res['fp_lexicon']
    g_tr = res['trigram_baseline']
    log(f"FP-lexicon:  real={g_fp['surprise_real_mean']:.4f} fake={g_fp['surprise_fake_mean']:.4f} AUC={g_fp['auc']:.3f}")
    log(f"Trigram:     real={g_tr['novelty_real_mean']:.3f} fake={g_tr['novelty_fake_mean']:.3f} AUC={g_tr['auc']:.3f}")
    if g_fp['gate_fake_gt_real'] and g_fp['auc'] >= 0.8:
        overall = 'FP_LEXICON_SURPRISE_YES'
    elif g_fp['gate_fake_gt_real']:
        overall = 'FP_LEXICON_WEAK'
    else:
        overall = 'FP_LEXICON_NO'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'fp_lexicon_192', 'overall': overall, 'results': res, 'diag_nn': diag, 'lexicon_size': len(lex_words), 'n_real': len(real_words), 'n_fake': len(fakes), 'note': 'zero-training probe on frozen 191-P1; fp = normalized arc_enc(word); read-only by construction'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage192 — FP-lexicon lexical surprise', '', f'**Overall:** `{overall}`', '', f"- FP-lexicon: real={g_fp['surprise_real_mean']:.4f} fake={g_fp['surprise_fake_mean']:.4f} **AUC={g_fp['auc']:.3f}**", f"- Trigram baseline: AUC={g_tr['auc']:.3f}", ''] + [f'- {d}' for d in diag]), encoding='utf-8')
    log(f'[192] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())