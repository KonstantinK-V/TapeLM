"""
Stage 207 — "curve as thinking": generative model whose OUTPUT space is the fp-space.

Same frozen char arc-encoder supplies word fingerprints for BOTH models; both share the same
causal-transformer trunk architecture and training budget. They differ ONLY in the output:

  CURVE : predicts the next word's FINGERPRINT in R^d, trained with in-batch InfoNCE (contrastive
          next-arc, NOT L2 -> no mean-collapse), decoded by SNAP to the nearest lexicon fp.
          Open metric vocabulary: can score/emit ANY word for which a fingerprint exists.
  CE    : predicts the next word ID via softmax over a CLOSED vocab (top-8k + UNK), trained with
          cross-entropy. Words outside the table collapse to UNK by construction.

Gates:
  G1 quality      curve k-way next-word acc within 0.05 of CE (on IN-VOCAB targets, fair to both)
  G2 drift (kill) free-run 50 steps: raw predicted fp must NOT walk off the lexicon manifold
                  (last-10 drift not >> first-10), i.e. snap is corrective, not a crutch for garbage
  G3 open-vocab   on OOV-for-CE targets (rank 8k..40k, real words): curve >> CE (CE ~ chance 0.25)
  G4 unification  the SAME trunk's hidden states work as memory keys for fact recall (>= 0.80)

  python _stage207_curve_thinking.py
"""
from __future__ import annotations
import json
import math
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _stage192_fp_lexicon import gen_fakes
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
DECISION = RES / 'stage207_decision.json'
MINI = RES / 'stage207_mini.md'
LOG = RES / '_stage207_log.txt'
SEED = 207
CORPUS_CHARS = 25000000
V_CE = 8000
V_LEX = 40000
RARE = V_LEX
MAXLEN = 48
D_MODEL = 256
N_LAYER = 4
N_HEAD = 4
STEPS = 3500
BATCH = 48
LR = 0.0003
TEMP = 0.07
N_EVAL = 800
WORD_RE = re.compile('[a-z]{2,}')

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

class Trunk(nn.Module):

    def __init__(self, d_out, d_in=256, d_model=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, MAXLEN, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=4 * d_model, batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d_model, d_out)

    def hidden(self, fps):
        T = fps.size(1)
        x = self.inp(fps) + self.pos[:, :T]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=fps.device), diagonal=1)
        return self.enc(x, mask=mask)

    def forward(self, fps):
        return self.head(self.hidden(fps))

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage207 start {datetime.now(timezone.utc).isoformat()}')
    log('curve-as-thinking: generate next fingerprint (InfoNCE + snap) vs closed-vocab token CE')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    p1 = SelfModelXL(n_char, V).to(device)
    p1.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    p1.eval()
    for p in p1.parameters():
        p.requires_grad_(False)
    bank = FpBank(p1, stoi, device)
    log(f'frozen arc-encoder loaded ({time.time() - t0:.0f}s)')
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(CORPUS_CHARS).lower()
    words = WORD_RE.findall(text)
    del text
    freq = Counter(words)
    vocab = [w for w, _ in freq.most_common(V_LEX)]
    word2rank = {w: i for i, w in enumerate(vocab)}
    log(f'words={len(words):,} distinct={len(freq):,} lexicon={len(vocab):,} ({time.time() - t0:.0f}s)')
    fp_rows = []
    for i in range(0, len(vocab), 4096):
        fp_rows.append(bank.fp(vocab[i:i + 4096]))
    FP = torch.cat(fp_rows, 0)
    ZERO = torch.zeros(256, device=device)
    log(f'fp lexicon table {tuple(FP.shape)} built ({time.time() - t0:.0f}s)')
    ranks = np.fromiter((word2rank.get(w, RARE) for w in words), dtype=np.int64, count=len(words))
    n_train = int(0.9 * len(ranks))
    train_r, eval_r = (ranks[:n_train], ranks[n_train:])

    def input_fp(ids):
        idx = torch.from_numpy(ids).to(device)
        out = torch.where((idx == RARE).unsqueeze(-1), ZERO.expand(*idx.shape, 256), FP[idx.clamp(max=V_LEX - 1)])
        return (out, idx)

    def draw(split, bsz):
        seqs = []
        hi = len(split) - MAXLEN - 1
        for _ in range(bsz):
            s = rng.randrange(hi)
            seqs.append(split[s:s + MAXLEN + 1])
        arr = np.stack(seqs, 0)
        return (arr[:, :-1], arr[:, 1:])
    curve = Trunk(d_out=256).to(device)
    ce = Trunk(d_out=V_CE + 1).to(device)
    n_par = sum((p.numel() for p in curve.parameters()))
    log(f'trunk params each ~{n_par / 1000000.0:.2f}M ({time.time() - t0:.0f}s)')
    opt_c = torch.optim.AdamW(curve.parameters(), lr=LR, weight_decay=0.01)
    opt_e = torch.optim.AdamW(ce.parameters(), lr=LR, weight_decay=0.01)

    def ce_target(next_ids):
        t = torch.from_numpy(np.where(next_ids < V_CE, next_ids, V_CE)).to(device)
        return t
    rc = re_ = None
    for step in range(1, STEPS + 1):
        xin, xnext = draw(train_r, BATCH)
        infp, _ = input_fp(xin)
        pred = F.normalize(curve(infp), dim=-1)
        nxt = torch.from_numpy(xnext).to(device)
        valid = nxt != RARE
        P = pred[valid]
        tgt_ids = nxt[valid]
        Tt = FP[tgt_ids]
        logits = P @ Tt.T / TEMP
        same = tgt_ids.unsqueeze(0) == tgt_ids.unsqueeze(1)
        eye = torch.eye(same.size(0), dtype=torch.bool, device=device)
        logits = logits.masked_fill(same & ~eye, float('-inf'))
        loss_c = F.cross_entropy(logits, torch.arange(P.size(0), device=device))
        opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        opt_c.step()
        logit_e = ce(infp)
        loss_e = F.cross_entropy(logit_e.reshape(-1, V_CE + 1), ce_target(xnext).reshape(-1))
        opt_e.zero_grad(set_to_none=True)
        loss_e.backward()
        opt_e.step()
        rc = float(loss_c) if rc is None else 0.98 * rc + 0.02 * float(loss_c)
        re_ = float(loss_e) if re_ is None else 0.98 * re_ + 0.02 * float(loss_e)
        if step % 500 == 0 or step == STEPS:
            log(f'  step {step}: curve_nce~{rc:.3f} ce~{re_:.3f} ({time.time() - t0:.0f}s)')
    curve.eval()
    ce.eval()

    @torch.no_grad()
    def eval_rank(lo, hi, n):
        """positions whose next-word rank in [lo,hi); 4-way, candidates drawn from [lo,hi)."""
        got_c = got_e = tot = 0
        erng = random.Random(SEED + 99)
        tries = 0
        while tot < n and tries < n * 40:
            tries += 1
            s = erng.randrange(len(eval_r) - MAXLEN - 1)
            seq = eval_r[s:s + MAXLEN + 1]
            nid = int(seq[-1])
            if not lo <= nid < hi:
                continue
            xin = seq[:-1][None, :]
            infp, _ = input_fp(xin)
            pred = F.normalize(curve(infp), dim=-1)[0, -1]
            logit_e = ce(infp)[0, -1]
            cand = [nid]
            while len(cand) < 4:
                c = erng.randrange(lo, hi)
                if c != nid and c not in cand:
                    cand.append(c)
            order = list(range(4))
            erng.shuffle(order)
            shuf = [cand[i] for i in order]
            gold = order.index(0)
            sc_c = [float(pred @ FP[c]) for c in shuf]
            sc_e = [float(logit_e[c if c < V_CE else V_CE]) + 1e-06 * erng.random() for c in shuf]
            got_c += int(int(np.argmax(sc_c)) == gold)
            got_e += int(int(np.argmax(sc_e)) == gold)
            tot += 1
        return (got_c / max(1, tot), got_e / max(1, tot), tot)
    g1_c, g1_e, n1 = eval_rank(0, V_CE, N_EVAL)
    g3_c, g3_e, n3 = eval_rank(V_CE, V_LEX, N_EVAL)
    log(f'G1 in-vocab (n={n1}): curve={g1_c:.3f} ce={g1_e:.3f}')
    log(f'G3 OOV-for-CE (n={n3}): curve={g3_c:.3f} ce={g3_e:.3f} (chance 0.25)')

    @torch.no_grad()
    def free_run(snap, steps=50):
        s = rng.randrange(len(eval_r) - MAXLEN - 1)
        seed = eval_r[s:s + 16]
        seq_fp = input_fp(seed[None, :])[0][0]
        drift, decoded = ([], [])
        cur = seq_fp
        for _ in range(steps):
            pred = F.normalize(curve(cur.unsqueeze(0)), dim=-1)[0, -1]
            sims = FP @ pred
            best = int(sims.argmax())
            drift.append(1.0 - float(sims[best]))
            decoded.append(best)
            nxt = FP[best] if snap else pred
            cur = torch.cat([cur, nxt.unsqueeze(0)], 0)[-MAXLEN:]
        return (drift, decoded)
    draw_raw, dec_raw = free_run(snap=False)
    draw_snap, dec_snap = free_run(snap=True)
    raw_first = float(np.mean(draw_raw[:10]))
    raw_last = float(np.mean(draw_raw[-10:]))
    snap_rep = sum((int(dec_snap[i] == dec_snap[i - 1]) for i in range(1, len(dec_snap)))) / (len(dec_snap) - 1)
    log(f'G2 raw drift first10={raw_first:.3f} last10={raw_last:.3f} | snap repetition={snap_rep:.3f}')

    @torch.no_grad()
    def trunk_key(words_list):
        ids = np.array([[word2rank.get(w, RARE) for w in words_list][:MAXLEN]], dtype=np.int64)
        if ids.shape[1] < 2:
            return None
        infp, _ = input_fp(ids)
        h = curve.hidden(infp)[0]
        return F.normalize(h.mean(0), dim=-1)
    subs = [w for w in gen_fakes(set(vocab), rng, 120) if len(w) >= 5][:80]
    valpool = vocab[100:100 + 80]
    facts = [{'S': subs[i], 'V': valpool[i]} for i in range(min(len(subs), len(valpool)))]
    keys, vals = ([], [])
    for f in facts:
        k = trunk_key(f"{f['S']} was appointed director of {f['V']} in 1987".split())
        if k is not None:
            keys.append(k)
            vals.append(f['V'])
    Kmat = torch.stack(keys, 0)
    qrng = random.Random(SEED + 5)
    ok = 0
    for i, f in enumerate(facts[:len(vals)]):
        q = trunk_key(f"{f['S']} was appointed director of".split())
        if q is None:
            continue
        sims = (Kmat @ q).tolist()
        best = {}
        for v, sc in zip(vals, sims):
            best[v] = max(best.get(v, -9.9), sc)
        others = [x for x in vals if x != f['V']]
        qrng.shuffle(others)
        cands = [f['V']] + others[:3]
        order = list(range(len(cands)))
        qrng.shuffle(order)
        shuf = [cands[j] for j in order]
        ok += int(int(np.argmax([best.get(c, -9.9) for c in shuf])) == order.index(0))
    g4 = ok / max(1, len(vals))
    log(f'G4 trunk-hidden memory recall (4-way): {g4:.3f}')
    g1 = g1_c >= g1_e - 0.05
    g2 = raw_last <= raw_first + 0.15
    g3 = g3_c >= g3_e + 0.2 and g3_c >= 0.5
    g4g = g4 >= 0.8
    passed = sum([g1, g2, g3, g4g])
    if g1 and g3 and g2 and g4g:
        overall = 'CURVE_THINKING_YES'
    elif g1 and g3:
        overall = 'CURVE_THINKING_PARTIAL'
    else:
        overall = 'CURVE_THINKING_NO'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'curve_as_thinking_207', 'overall': overall, 'gates_passed': f'{passed}/4', 'G1_quality_invocab': {'curve': g1_c, 'ce': g1_e, 'n': n1, 'pass': g1}, 'G3_open_vocab_oov': {'curve': g3_c, 'ce': g3_e, 'n': n3, 'chance': 0.25, 'pass': g3}, 'G2_drift': {'raw_first10': raw_first, 'raw_last10': raw_last, 'snap_repetition': snap_rep, 'pass': g2}, 'G4_unified_memory': {'recall': g4, 'pass': g4g}, 'config': {'V_CE': V_CE, 'V_LEX': V_LEX, 'MAXLEN': MAXLEN, 'D_MODEL': D_MODEL, 'N_LAYER': N_LAYER, 'STEPS': STEPS, 'params_each_M': round(n_par / 1000000.0, 2)}, 'note': "output space is the shared fp lexicon (open metric vocab) vs CE's closed softmax table; contrastive next-arc avoids L2 mean-collapse; snap-to-lexicon is the error-correcting decoder"}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage207 — curve as thinking (fp-generative vs token CE)', '', f'**Overall:** `{overall}` ({passed}/4 gates)', '', '| gate | curve | CE / baseline | pass |', '|------|-------|---------------|------|', f'| G1 quality (in-vocab, 4-way) | {g1_c:.3f} | {g1_e:.3f} | {g1} |', f'| G3 open-vocab OOV (4-way, chance 0.25) | **{g3_c:.3f}** | {g3_e:.3f} | {g3} |', f'| G2 drift raw first→last10 | {raw_first:.3f}→{raw_last:.3f} | snap rep {snap_rep:.3f} | {g2} |', f'| G4 unified trunk-hidden memory | {g4:.3f} | — | {g4g} |', '', f'- shared frozen arc-encoder input; both trunks {round(n_par / 1000000.0, 2)}M params, {STEPS} steps.', "- G3 is the essence gate: CE's closed softmax gives every OOV word the same UNK logit → chance;", '  the curve ranks them by fingerprint in an open metric vocabulary.']), encoding='utf-8')
    log(f'[207] {overall} ({passed}/4) | G1 {g1_c:.2f}/{g1_e:.2f} | G3 {g3_c:.2f}/{g3_e:.2f} | G2 {raw_first:.2f}->{raw_last:.2f} | G4 {g4:.2f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())