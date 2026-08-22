"""
Stage 193 — wire FP-lexicon surprise into the head (the real crossover).

192 proved the signal (AUC 0.991, read-only). Now make it visible in behavior:
  - frozen night P1 curve-XL (nothing in the backbone trains — G1 safe by design)
  - per-position lexical surprise s_t: at each WORD boundary, s = 1 - max_cos(fp(word), lexicon)
  - logits_t / T_t,  T_t = 1 + softplus(w * s_t + b) — only w,b train (2 params, CE)

Gates:
  G1 next_tok drop <= 0.02 vs raw P1 (temperature must not hurt)
  G3 entropy after FAKE entity > after real (the never-passed one, now with real signal)

  python _stage193_fp_wired.py
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
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, sample_windows
RES = Path('results')
DATA = Path('data')
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
DECISION = RES / 'stage193_decision.json'
MINI = RES / 'stage193_mini.md'
LOG = RES / '_stage193_log.txt'
EXAM_V3 = DATA / 'stage191_exam_v3.jsonl'
SEED = 193
MAX_CHARS = s177.MAX_CHARS_PER_ARC
MAX_LEX = 200000
MIN_COUNT = 2
TEMP_STEPS = 600
TEMP_LR = 0.05
MICRO = 16
FAKES_GEN = 150

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

class LexSurprise:
    """word → 1 - max_cos(fp, lexicon); fp = normalized frozen arc_enc(word chars)."""

    def __init__(self, model, stoi, lex_words, device):
        self.model = model
        self.stoi = stoi
        self.device = device
        self.cache: dict[str, float] = {}
        log('  encoding lexicon fps …')
        fps = []
        for i in range(0, len(lex_words), 4096):
            fps.append(self._fp(lex_words[i:i + 4096]))
        self.lex = torch.cat(fps, 0)

    @torch.no_grad()
    def _fp(self, ws: list[str]) -> torch.Tensor:
        rows = torch.zeros(len(ws), 1, MAX_CHARS, dtype=torch.long)
        for i, w in enumerate(ws):
            for j, c in enumerate(w[:MAX_CHARS]):
                rows[i, 0, j] = self.stoi.get(c, 0)
        return F.normalize(self.model.arc_enc(rows.to(self.device))[:, 0], dim=-1)

    @torch.no_grad()
    def surprise(self, words: list[str]) -> list[float]:
        todo = [w for w in words if w not in self.cache]
        if todo:
            fps = self._fp(todo)
            best = (fps @ self.lex.T).max(dim=-1).values
            for w, b in zip(todo, best):
                self.cache[w] = float(1.0 - b)
        return [self.cache[w] for w in words]

def position_surprise(ids_row: list[int], piece_str: list[str], lexs: LexSurprise, pad_id: int) -> list[float]:
    """s_t > 0 at the last piece of each alphabetic word."""
    n = len(ids_row)
    s = [0.0] * n
    buf = ''
    start = 0
    words, bounds = ([], [])
    for t in range(n + 1):
        p = piece_str[ids_row[t]] if t < n and ids_row[t] != pad_id else ' '
        boundary = t == n or p.startswith(' ') or (p and (not p[0].isalpha()))
        if boundary and buf:
            m = re.search('[A-Za-z][a-z]+$', buf) or re.search('[A-Za-z]+$', buf)
            if m and len(m.group(0)) >= 4:
                words.append(m.group(0))
                bounds.append(t - 1)
            buf = ''
        if t < n and ids_row[t] != pad_id:
            buf = buf + p if not boundary else p
    if words:
        vals = lexs.surprise(words)
        for b, v in zip(bounds, vals):
            s[b] = v
    return s

def gen_fakes(word_set, rng, n):
    cons, vow = ('bcdfghklmnprstvz', 'aeiou')
    out = []
    tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        w = ''.join((rng.choice(cons) + rng.choice(vow) + (rng.choice(cons) if rng.random() < 0.4 else '') for _ in range(rng.randint(2, 4)))).capitalize()
        if w.lower() not in word_set and w not in word_set and (5 <= len(w) <= 12) and (w not in out):
            out.append(w)
    return out

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage193 start {datetime.now(timezone.utc).isoformat()}')
    log('Wire FP-lexicon surprise into head temperature (frozen backbone, 2 trainable params)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    t0 = time.time()
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    import _stage185_tape_read as s185
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    piece_str = [tok.decode([i], skip_special_tokens=False) or '' for i in range(V)]
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log('reading corpus words …')
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(150000000)
    cnt = Counter(re.findall('[A-Za-z][a-z]+', text))
    word_set = set(cnt.keys()) | {w.lower() for w in cnt}
    lex_words = [w for w, c in cnt.most_common(MAX_LEX) if c >= MIN_COUNT]
    del text
    lexs = LexSurprise(model, stoi, lex_words, device)
    log(f'  lexicon={len(lex_words)} ({time.time() - t0:.0f}s)')
    w_par = nn.Parameter(torch.tensor(10.0, device=device))
    b_par = nn.Parameter(torch.tensor(-2.0, device=device))
    opt = torch.optim.Adam([w_par, b_par], lr=TEMP_LR)

    def s_tensor(ids: torch.Tensor) -> torch.Tensor:
        rows = ids.tolist()
        return torch.tensor([position_surprise(r, piece_str, lexs, pad_id) for r in rows], device=device)

    def logits_with_T(ids: torch.Tensor):
        pad = ids == pad_id
        with torch.no_grad():
            logits, _, _ = model.forward_all(char_table[ids], pad, ids=ids)
        s = s_tensor(ids)
        T = 1.0 + F.softplus(w_par * s + b_par).unsqueeze(-1)
        return (logits / T, pad)
    log('training temperature (w,b) …')
    for step in range(1, TEMP_STEPS + 1):
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        logits, pad = logits_with_T(ids)
        target = ids[:, 1:]
        valid = ~pad[:, :-1] & ~pad[:, 1:]
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        opt.zero_grad(set_to_none=True)
        ce.backward()
        opt.step()
        if step % 150 == 0 or step == TEMP_STEPS:
            log(f'  step {step}: ce={float(ce):.4f} w={float(w_par):.3f} b={float(b_par):.3f}')
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding='utf-8').splitlines()]

    @torch.no_grad()
    def span_lp(ctx, cand, temp: bool):
        seq = (ctx + cand)[-MAX_ARCS:]
        n_ctx = len(seq) - len(cand)
        x = torch.tensor([seq], dtype=torch.long, device=device)
        if temp:
            logits, _ = logits_with_T(x)
            logits = logits[0]
        else:
            pad = x == pad_id
            logits = model.forward_all(char_table[x], pad, ids=x)[0][0]
        logp = F.log_softmax(logits, dim=-1)
        return sum((float(logp[n_ctx + k - 1, tid]) for k, tid in enumerate(cand))) / max(1, len(cand))

    def next_tok_acc(temp: bool, n=200):
        its = [it for it in items if it['type'] == 'next_tok'][:n]
        ok = 0
        for it in its:
            sc = [span_lp(it['ctx_ids'], c, temp) for c in it['cand_ids']]
            ok += int(int(np.argmax(sc)) == it['gold_idx'])
        return ok / len(its)
    acc_raw = next_tok_acc(False)
    acc_temp = next_tok_acc(True)
    log(f'next_tok raw={acc_raw:.3f} temp={acc_temp:.3f}')
    fakes = gen_fakes(word_set, rng, FAKES_GEN)
    ent_items = [it for it in items if it['type'] == 'entity'][:100]
    rngf = random.Random(3)

    @torch.no_grad()
    def entropy_after(ctx, span):
        seq = (ctx + span)[-MAX_ARCS:]
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits, _ = logits_with_T(x)
        p = F.softmax(logits[0, len(seq) - 1], dim=-1)
        return float(-(p * torch.log(p + 1e-09)).sum())
    e_real, e_fake = ([], [])
    for it in ent_items:
        gold_ids = it['cand_ids'][it['gold_idx']]
        fk = fakes[rngf.randint(0, len(fakes) - 1)]
        fk_ids = [i for i in tok.encode(' ' + fk).ids if i != pad_id]
        e_real.append(entropy_after(it['ctx_ids'], gold_ids))
        e_fake.append(entropy_after(it['ctx_ids'], fk_ids))
    ent_real, ent_fake = (float(np.mean(e_real)), float(np.mean(e_fake)))
    g1 = acc_temp >= acc_raw - 0.02
    g3 = ent_fake > ent_real
    overall = 'FP_WIRED_YES' if g1 and g3 else 'FP_WIRED_PARTIAL_' + ''.join((n for n, ok in (('1', g1), ('3', g3)) if not ok))
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'fp_wired_193', 'overall': overall, 'gates': {'G1': {'next_tok_raw': acc_raw, 'next_tok_temp': acc_temp, 'ok': g1}, 'G3': {'entropy_real': ent_real, 'entropy_fake': ent_fake, 'ok': g3}}, 'temp': {'w': float(w_par), 'b': float(b_par)}, 'note': 'frozen backbone; only (w,b) trained; surprise = FP-lexicon, read-only'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage193 — FP-lexicon wired into head temperature', '', f'**Overall:** `{overall}`', '', f'- G1: next_tok raw={acc_raw:.3f} → temp={acc_temp:.3f} → {g1}', f'- G3: entropy real={ent_real:.3f} fake={ent_fake:.3f} → {g3}', f'- learned w={float(w_par):.3f} b={float(b_par):.3f}', '']), encoding='utf-8')
    log(f'[193] {overall} | G1 {acc_raw:.3f}->{acc_temp:.3f} | G3 {ent_real:.3f} vs {ent_fake:.3f} | w={float(w_par):.2f} b={float(b_par):.2f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())