"""
Stage 219 — slot age decay on stream recall (198-style stress).

  python _stage219_stream_decay.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from _stage191_night import load_data
from _stage194_fp_fact_memory import FpBank
from _stage192_fp_lexicon import gen_fakes
RES = Path('results')
CKPT = Path('checkpoints/stage191_p1_curve.pt')
DECISION = RES / 'stage219_decision.json'
SEED = 219

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    n_slots = 40 if args.smoke else 120
    from tokenizers import Tokenizer
    import _stage177_curve_bpe as s177
    from _stage191_night import SelfModelXL, PAD
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model.eval()
    bank = FpBank(model, stoi, device)
    words = [f'Ent{i}' for i in range(n_slots // 2)]
    stale_vals = [f'Old{i}' for i in range(len(words))]
    fresh_vals = [f'New{i}' for i in range(len(words))]
    keys, vals, ages = ([], [], [])
    for i, (w, ov, nv) in enumerate(zip(words, stale_vals, fresh_vals)):
        k = bank.fp([w])[0]
        keys.append(k)
        vals.append(ov)
        ages.append(100 + i)
        keys.append(k)
        vals.append(nv)
        ages.append(1 + i)
    K = torch.stack(keys, 0)
    max_age = max(ages)
    tau = max_age / 3.0

    def eval_use_decay(use: bool) -> float:
        ok, n = (0, 0)
        for i, w in enumerate(words):
            q = bank.fp([w])[0]
            sims = K @ q
            if use:
                wts = torch.tensor([math.exp(-a / tau) for a in ages], device=device, dtype=sims.dtype)
                sims = sims * wts
            sc = {}
            for j, v in enumerate(vals):
                sc[v] = max(sc.get(v, -1000000000.0), float(sims[j]))
            gold = fresh_vals[i]
            cands = [gold, stale_vals[i], fresh_vals[(i + 1) % len(words)], stale_vals[(i + 1) % len(words)]]
            rng.shuffle(cands)
            g = cands.index(gold)
            ok += int(int(np.argmax([sc[c] for c in cands])) == g)
            n += 1
        return ok / max(1, n)
    acc_flat = eval_use_decay(False)
    acc_decay = eval_use_decay(True)
    g1 = acc_decay >= acc_flat + 0.05
    overall = 'STREAM_DECAY_WIN' if g1 else 'STREAM_DECAY_NO'
    DECISION.write_text(json.dumps({'stage': 219, 'overall': overall, 'gates': {'G1_decay_vs_flat': g1}, 'acc_flat': acc_flat, 'acc_decay': acc_decay, 'timestamp': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
    print(f'219 {overall} flat={acc_flat:.3f} decay={acc_decay:.3f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())