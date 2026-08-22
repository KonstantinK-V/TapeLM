"""
Stage 201 — B-track on YOUR GPU: crack form-dominance with minimal-pair hard negatives.

199 showed a CPC head on FROZEN features can't move B (car≈cat baked into the substrate).
So here we touch the SUBSTRATE — but on a COPY of the encoder, leaving the product P1 frozen
and intact. Objective directly attacks form-dominance:
  - HARD NEGATIVES: edit-distance-1 word pairs mined from the corpus (car/cat, cold/bold,
    door/book-like) are pushed APART in fp-space.
  - ANCHOR: every word's new fp is kept near its original P1 fp (prevents collapse / preserves
    the structure that gives parity + memory).
Then measure B on the 179 sentence pairs through the fine-tuned encoder's fast channel.
Success = hard pairs drop below paraphrases (INVERSION), i.e. meaning finally beats spelling.

Fits a 3050: word-level contrastive on arc_enc only (fast/head frozen), few hundred steps.

Gates:
  G_invert  para_new > hard_new                                   -> SEM_HARDNEG_YES
  G_trend   hard_new <= baseline_hard - 0.10 and para_new >= 0.60 -> SEM_HARDNEG_TREND
  report    next_tok on the COPY (generation cost of touching the substrate; product P1 untouched)

  python _stage201_semantic_hardneg.py
"""
from __future__ import annotations
import copy
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data, score_items, span_logprob_x
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
EXAM_V3 = Path('data/stage191_exam_v3.jsonl')
DECISION = RES / 'stage201_decision.json'
MINI = RES / 'stage201_mini.md'
LOG = RES / '_stage201_log.txt'
SEED = 201
CORPUS_CHARS = 60000000
TOP_WORDS = 40000
MAX_PAIRS = 20000
STEPS = 500
BATCH_PAIRS = 256
BATCH_ANCHOR = 256
LR = 0.0002
LAMBDA_ANCHOR = 1.0
MAX_CHARS = s177.MAX_CHARS_PER_ARC
MAX_ARCS = 64

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def mine_minimal_pairs(words: list[str], rng: random.Random) -> list[tuple[str, str]]:
    """edit-distance-1 substitution pairs via wildcard buckets (car/cat, cold/bold)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for w in words:
        for i in range(len(w)):
            buckets[w[:i] + '*' + w[i + 1:]].append(w)
    pairs = set()
    for grp in buckets.values():
        if len(grp) < 2:
            continue
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                pairs.add((grp[a], grp[b]))
                if len(pairs) >= MAX_PAIRS * 2:
                    break
    pairs = list(pairs)
    rng.shuffle(pairs)
    return pairs[:MAX_PAIRS]

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage201 start {datetime.now(timezone.utc).isoformat()}')
    log('B-track: minimal-pair hard negatives on an encoder COPY (product P1 stays frozen)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    t0 = time.time()
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    p1 = SelfModelXL(n_char, V).to(device)
    p1.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    p1.eval()
    for p in p1.parameters():
        p.requires_grad_(False)
    m = copy.deepcopy(p1)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.arc_enc.parameters():
        p.requires_grad_(True)
    m.arc_enc.train()
    log(f'copy made; arc_enc trainable ({sum((p.numel() for p in m.arc_enc.parameters())) / 1000.0:.0f}k params)')

    def word_rows(words):
        rows = torch.zeros(len(words), 1, MAX_CHARS, dtype=torch.long)
        for i, w in enumerate(words):
            for j, c in enumerate(w[:MAX_CHARS]):
                rows[i, 0, j] = stoi.get(c, 0)
        return rows.to(device)

    def word_fp(words, mdl):
        return F.normalize(mdl.arc_enc(word_rows(words))[:, 0], dim=-1)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(CORPUS_CHARS)
    cnt = Counter(re.findall('[a-z]{3,}', text.lower()))
    del text
    words = [w for w, c in cnt.most_common(TOP_WORDS)]
    pairs = mine_minimal_pairs(words, rng)
    log(f'vocab={len(words)} minimal_pairs={len(pairs)} (e.g. {pairs[:5]}) ({time.time() - t0:.0f}s)')
    anchor_words = words[:8000]
    with torch.no_grad():
        anchor_fp = {}
        for i in range(0, len(anchor_words), 4096):
            chunk = anchor_words[i:i + 4096]
            fp = word_fp(chunk, p1)
            for w, v in zip(chunk, fp):
                anchor_fp[w] = v
    log(f'anchor fps={len(anchor_fp)} ({time.time() - t0:.0f}s)')

    @torch.no_grad()
    def pooled_text(t: str, mdl):
        ids = [i for i in tok.encode(t).ids if i != pad_id][:MAX_ARCS]
        x = torch.tensor([ids], device=device)
        pad = x == pad_id
        arcs = mdl._arcs(char_table[x], x)
        fast = mdl.fast(arcs, pad_mask=pad)
        return F.normalize(fast.mean(1)[0], dim=-1)

    def measure_B(mdl):
        para = float(np.mean([float(F.cosine_similarity(pooled_text(a, mdl), pooled_text(b, mdl), dim=-1)) for a, b in s179.PARAPHRASE_PAIRS]))
        hard = float(np.mean([float(F.cosine_similarity(pooled_text(a, mdl), pooled_text(b, mdl), dim=-1)) for a, b in s179.HARD_PAIRS]))
        return {'para': para, 'hard': hard, 'inversion': para > hard}
    base = measure_B(p1)
    log(f"P1 baseline B: para={base['para']:.3f} hard={base['hard']:.3f} inv={base['inversion']}")
    opt = torch.optim.Adam(m.arc_enc.parameters(), lr=LR)
    running = None
    for step in range(1, STEPS + 1):
        pb = [pairs[rng.randint(0, len(pairs) - 1)] for _ in range(BATCH_PAIRS)]
        wa = [p[0] for p in pb]
        wb = [p[1] for p in pb]
        fa = word_fp(wa, m)
        fb = word_fp(wb, m)
        neg = (fa * fb).sum(-1).clamp(min=-1, max=1).mean()
        aw = [anchor_words[rng.randint(0, len(anchor_words) - 1)] for _ in range(BATCH_ANCHOR)]
        fnew = word_fp(aw, m)
        fold = torch.stack([anchor_fp[w] for w in aw], 0)
        anchor = (1.0 - (fnew * fold).sum(-1)).mean()
        loss = neg + LAMBDA_ANCHOR * anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running = float(loss) if running is None else 0.97 * running + 0.03 * float(loss)
        if step % 100 == 0 or step == STEPS:
            b = measure_B(m)
            log(f"  step {step}: loss~{running:.3f} (neg~{float(neg):.3f} anchor~{float(anchor):.3f}) | para={b['para']:.3f} hard={b['hard']:.3f} inv={b['inversion']} ({time.time() - t0:.0f}s)")
    m.arc_enc.eval()
    after = measure_B(m)
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding='utf-8').splitlines()]
    nt = [it for it in items if it['type'] == 'next_tok'][:120]
    nt_copy = score_items(lambda c, cd: span_logprob_x(m, char_table, pad_id, c, cd, device), nt, 'next_tok')['next_tok_acc']
    nt_p1 = score_items(lambda c, cd: span_logprob_x(p1, char_table, pad_id, c, cd, device), nt, 'next_tok')['next_tok_acc']
    log(f'next_tok: P1(product)={nt_p1:.3f}  copy(fine-tuned)={nt_copy:.3f}')
    g_invert = after['para'] > after['hard']
    g_trend = after['hard'] <= base['hard'] - 0.1 and after['para'] >= 0.6
    if g_invert:
        overall = 'SEM_HARDNEG_YES'
    elif g_trend:
        overall = 'SEM_HARDNEG_TREND'
    else:
        overall = 'SEM_HARDNEG_NO'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'semantic_hardneg_201', 'overall': overall, 'baseline_B': base, 'after_B': after, 'next_tok_product_p1': nt_p1, 'next_tok_finetuned_copy': nt_copy, 'minimal_pairs': len(pairs), 'gates': {'g_invert': g_invert, 'g_trend': g_trend}, 'note': 'arc_enc COPY fine-tuned with edit-distance-1 hard negatives + anchor-to-P1; product P1 frozen'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage201 — B via minimal-pair hard negatives (encoder copy)', '', f'**Overall:** `{overall}`', '', f"- P1 baseline: para {base['para']:.3f} / hard {base['hard']:.3f} (inv={base['inversion']})", f"- after hard-neg: para {after['para']:.3f} / hard {after['hard']:.3f} (**inv={after['inversion']}**)", f'- next_tok: product P1 {nt_p1:.3f} | fine-tuned copy {nt_copy:.3f}', f'- minimal pairs mined: {len(pairs)}', '']), encoding='utf-8')
    log(f"[201] {overall} | para {base['para']:.2f}->{after['para']:.2f} hard {base['hard']:.2f}->{after['hard']:.2f} nt_copy={nt_copy:.2f}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())