"""
Stage 217 — dual keys: ctx_fp (lex) + slow endpoint; noisy recall vs lex-only.

  python _stage217_slow_endpoint_slots.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, FpBank
from _stage204_noise_robustness import noisy
from _tapelm_ext import slow_endpoint_vec
RES = Path('results')
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
DECISION = RES / 'stage217_decision.json'
SEED = 217

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    n_facts = 15 if args.smoke else 50
    p_noise = 0.3
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model.eval()
    bank = FpBank(model, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        seen = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(3000000)) if len(m.group(1)) >= 5)))
    paras = [p.strip() for p in f.read().split('\n') if len(p.strip()) > 200][:200] if False else []
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(2000000)
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 200][:300]
    subs = gen_fakes(set(seen), rng, n_facts + 10)[:n_facts]
    vals = seen[:n_facts]
    Kl, Ks, V = ([], [], [])
    for S, v in zip(subs, vals):
        filler = paras[rng.randrange(len(paras))][:250]
        sent = f'{filler} {S} was appointed director of {v} in 1987 .'
        lo, hi = (0, len(sent))
        ctx_ids = [i for i in tok.encode(sent[lo:hi]).ids if i != pad_id]
        kx = bank.fp([S])[0]
        cx = bank.ctx_fp(sent, exclude=v)
        if cx is None:
            continue
        Kl.append(F.normalize(kx + cx, dim=-1))
        sv = slow_endpoint_vec(model, char_table, pad_id, ctx_ids, device)
        if sv is None:
            Kl.pop()
            continue
        Ks.append(sv)
        V.append(v)
    Kl, Ks = (torch.stack(Kl, 0), torch.stack(Ks, 0))
    ok_l, ok_b, n = (0, 0, 0)
    nrng = random.Random(SEED + 3)
    for S, v in zip(subs, vals):
        sent = f'According to reports {S} worked closely with {v} for many years.'
        q_sent = noisy(sent, p_noise, nrng)
        qx = bank.ctx_fp(q_sent, exclude=v)
        q_ids = [i for i in tok.encode(q_sent).ids if i != pad_id]
        qs = slow_endpoint_vec(model, char_table, pad_id, q_ids, device)
        if qx is None or qs is None:
            continue
        cands = [v] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        gold = cands.index(v)
        sc_l = []
        sc_b = []
        for c in cands:
            idxs = [i for i, vv in enumerate(V) if vv == c]
            if not idxs:
                sc_l.append(-1.0)
                sc_b.append(-1.0)
                continue
            sl = float((Kl[idxs] @ qx).max())
            sb = float(max(float((Kl[idxs] @ qx).max()), float((Ks[idxs] @ qs).max())))
            sc_l.append(sl)
            sc_b.append(sb)
        ok_l += int(np.argmax(sc_l) == gold)
        ok_b += int(np.argmax(sc_b) == gold)
        n += 1
    acc_l = ok_l / max(1, n)
    acc_b = ok_b / max(1, n)
    g1 = acc_b >= acc_l + 0.02
    overall = 'SLOW_ENDPOINT_WIN' if g1 else 'SLOW_ENDPOINT_INVALID_METHOD'
    DECISION.write_text(json.dumps({'stage': 217, 'overall': overall, 'gates': {'G1_noisy_dual': g1}, 'acc_lex': acc_l, 'acc_dual': acc_b, 'p_noise': p_noise, 'n': n, 'timestamp': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
    print(f'217 {overall} lex={acc_l:.3f} dual={acc_b:.3f} n={n}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())