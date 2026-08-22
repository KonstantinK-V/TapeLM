"""
Stage 249 — Online hop-gated slot admission on a long domain stream.

Stream of candidate facts + filler. Admission:
  hop: keep if cos(fp(fact), evolving hop query) in top-k / above median of batch
  surprise: keep if subject is novel fake (all planted are); baseline = first-B budget
Compare recall of hop-relevant gold set vs uniform budget under same B.

  python _stage249_hop_stream.py [--smoke] [--steps N]  (steps ≈ stream length proxy)
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
RES = Path('results')
DECISION = RES / 'stage249_decision.json'
MINI = RES / 'stage249_mini.md'
LOG = RES / '_stage249_log.txt'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 249

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0, help='unused train steps; scales n_events')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    t0 = time.time()
    n_rel = 20 if args.smoke else 80
    n_irrel = 20 if args.smoke else 80
    budget = 12 if args.smoke else 40
    scale = 1 if not args.steps else max(1, args.steps // 1000)
    n_rel *= min(4, scale)
    n_irrel *= min(4, scale)
    budget = min(budget * min(3, scale), n_rel)
    log(f'Stage249 start {datetime.now(timezone.utc).isoformat()} rel={n_rel} irrel={n_irrel} B={budget}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model.eval()
    bank = FpBank(model, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(6000000 if args.smoke else 25000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)

    def mk(n, theme, start):
        subs = [w for w in gen_fakes(set(values_pool), random.Random(SEED + start), n + 40) if len(w) >= 5][:n]
        out = []
        for i, S in enumerate(subs):
            Vv = values_pool[(start + i) % len(values_pool)]
            if theme == 'org':
                sent = f'{S} was appointed director of {Vv} in the organization chronicle .'
            else:
                sent = f'{S} crossed the river near {Vv} during the autumn migration .'
            out.append({'S': S, 'value': Vv, 'sent': sent, 'theme': theme, 'fid': f'{theme}_{i}'})
        return out
    rel = mk(n_rel, 'org', 11)
    irrel = mk(n_irrel, 'geo', 77)
    stream = rel + irrel
    rng.shuffle(stream)
    hop = bank.ctx_fp('In the report the organization appointed a new director of governance.')
    if hop is None:
        hop = bank.fp(['organization', 'director'])[0]

    def score(f):
        k = bank.fp([f['S']])[0]
        c = bank.ctx_fp(f['sent'], exclude=f['value'])
        key = F.normalize(k + c, dim=-1) if c is not None else k
        return float((key * hop).sum())
    scored = [(score(f), f) for f in stream]
    scored.sort(key=lambda x: -x[0])
    hop_keep = [f for _, f in scored[:budget]]
    uni_keep = stream[:budget]

    def bank_of(fs):
        if not fs:
            return (None, None)
        return L.write_tape_bank(bank, fs)
    Kh, Vh = bank_of(hop_keep)
    Ku, Vu = bank_of(uni_keep)
    all_v = [f['value'] for f in stream] + values_pool[:100]

    def recall_theme(fs_gold, K, V):
        gold = [f for f in fs_gold if any((f['fid'] == x['fid'] for x in (hop_keep if K is Kh else uni_keep) or []))]
        return L.tape_recall(fs_gold, all_v, bank, K, V, SEED) if K is not None else 0.0

    def theme_hit_rate(theme_facts, keep_set, K, V):
        ok = 0
        for f in theme_facts:
            if f['fid'] not in {x['fid'] for x in keep_set}:
                continue
            r = L.tape_recall([f], all_v, bank, K, V, SEED)
            ok += int(r >= 0.99)
        admitted = sum((1 for f in theme_facts if f['fid'] in {x['fid'] for x in keep_set}))
        return (ok / max(1, admitted), admitted)
    hop_rel_acc, hop_rel_n = theme_hit_rate(rel, hop_keep, Kh, Vh)
    uni_rel_acc, uni_rel_n = theme_hit_rate(rel, uni_keep, Ku, Vu)
    hop_rel_frac = sum((1 for f in hop_keep if f['theme'] == 'org')) / max(1, len(hop_keep))
    uni_rel_frac = sum((1 for f in uni_keep if f['theme'] == 'org')) / max(1, len(uni_keep))
    g_prec = hop_rel_frac >= uni_rel_frac + 0.15
    g_util = hop_rel_acc >= 0.8 and hop_rel_n >= budget // 3
    if g_prec and g_util:
        overall = 'HOP_STREAM_OK'
    elif g_prec or g_util:
        overall = 'HOP_STREAM_PARTIAL'
    else:
        overall = 'HOP_STREAM_NO'
    out = {'stage': 249, 'overall': overall, 'budget': budget, 'n_rel': n_rel, 'n_irrel': n_irrel, 'gates': {'G_hop_precision_vs_uniform': g_prec, 'G_admitted_rel_util': g_util}, 'hop': {'rel_frac': hop_rel_frac, 'rel_acc': hop_rel_acc, 'rel_admitted': hop_rel_n}, 'uniform': {'rel_frac': uni_rel_frac, 'rel_acc': uni_rel_acc, 'rel_admitted': uni_rel_n}, 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 249 hop stream\n\n**{overall}** hop_rel_frac={hop_rel_frac:.2f} uni={uni_rel_frac:.2f}\n', encoding='utf-8')
    log(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())