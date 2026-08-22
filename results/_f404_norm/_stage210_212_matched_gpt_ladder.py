"""
Refresh 210–212 verdict framing + matched-GPT ladder (cheap; no retrain of 210–212).

  python _stage210_212_matched_gpt_ladder.py           # patch decisions + write ladder JSON
  python _stage210_212_matched_gpt_ladder.py --run-gpt # also run 210 GPT parametric chain baseline

Does NOT change P1 or re-run SoftFollow training.
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, load_data
from _stage196_tapelm import gpt_span, load_gpt
from _stage210_softfollow_forward import CHAIN_LEN, encode_query, encode_word
RES = Path('results')
CLAIM = RES / 'internalization_210_212_claim_scope.json'
LADDER = RES / 'stage210_212_matched_gpt_ladder.json'
SEED = 210212
STAGES = {210: RES / 'stage210_decision.json', 211: RES / 'stage211_decision.json', 212: RES / 'stage212_decision.json'}

def load_claim() -> dict:
    if CLAIM.exists():
        return json.loads(CLAIM.read_text(encoding='utf-8'))
    return {}

def rename_overall(d: dict) -> dict:
    ov = d.get('overall')
    if ov == 'THESIS_NO':
        d['overall_legacy'] = 'THESIS_NO'
        d['overall'] = 'THESIS_NO_AT_SCALE'
    return d

def patch_stage(path: Path, claim: dict, ladder: dict) -> None:
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding='utf-8'))
    d = rename_overall(d)
    d['claim_scope'] = claim
    stage = int(path.name.replace('stage', '').split('_')[0])
    d['gpt_matched_ladder'] = ladder.get(f'stage_{stage}', {})
    if d['overall'] == 'THESIS_NO_AT_SCALE':
        d['interpretation'] = 'THESIS_NO_AT_SCALE: gates failed on frozen P1 @ d256/6L with controls run in this JSON. Not a permanent falsification; see gpt_matched_ladder and 209 for matched-GPT / scale context.'
    path.write_text(json.dumps(d, indent=2), encoding='utf-8')
    mini = path.with_name(path.name.replace('_decision.json', '_mini.md'))
    if mini.exists():
        body = mini.read_text(encoding='utf-8')
        body = body.replace('`THESIS_NO`', '`THESIS_NO_AT_SCALE`')
        if '**Overall:**' in body and 'legacy' not in body:
            body = body.replace('**Overall:** `THESIS_NO_AT_SCALE`', '**Overall:** `THESIS_NO_AT_SCALE` (legacy label: THESIS_NO)')
        mini.write_text(body, encoding='utf-8')

@torch.no_grad()
def gpt_chain_parametric(device, n_sample: int=80) -> dict:
    """210-style k-hop 4-way token ID — GPT parametric, no tape (matched LM baseline)."""
    from _stage192_fp_lexicon import gen_fakes
    import numpy as np
    _, _, stoi, _n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    gm = load_gpt(device)
    rng = random.Random(SEED)
    fakes = gen_fakes(set(), rng, 200)
    chains = [fakes[i * CHAIN_LEN:(i + 1) * CHAIN_LEN] for i in range(50)]
    chains = [c for c in chains if len(c) == CHAIN_LEN]
    samples = []
    for c in chains:
        for k in range(1, CHAIN_LEN):
            samples.append((c[0], k, c[k]))
    rng.shuffle(samples)
    samples = samples[:n_sample]
    ent_pool = list(dict.fromkeys((n for c in chains for n in c)))
    ok = {1: 0, 2: 0, 3: 0}
    tot = {1: 0, 2: 0, 3: 0}
    for a0, k, ak in samples:
        tot[k] += 1
        ctx = encode_query(tok, pad_id, a0, k)
        cands = [ak] + [ent_pool[rng.randint(0, len(ent_pool) - 1)] for _ in range(3)]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        gold = order.index(0)
        scores = [gpt_span(gm, device, ctx, encode_word(tok, pad_id, c)) for c in shuf]
        ok[k] += int(int(np.argmax(scores)) == gold)
    acc = {str(k): ok[k] / max(1, tot[k]) for k in (1, 2, 3)}
    return {'protocol': 'gpt_parametric_chain_4way', 'n': len(samples), 'acc_by_hop': acc, 'chance': 0.25}

def build_ladder(run_gpt: bool) -> dict:
    ladder: dict = {'timestamp': datetime.now(timezone.utc).isoformat(), 'reference': '209_sem_scaling_teacher_209', 'stage_209_summary': {'verdict': 'STRUCTURAL_BLOCK_NO', 'note': 'PAWS tracks matched GPT at d128/d192/d256 — shared scale ceiling, not curve-specific blindness'}}
    p211 = STAGES[211]
    if p211.exists():
        d211 = json.loads(p211.read_text(encoding='utf-8'))
        c = d211.get('clean', {})
        ladder['stage_211'] = {'status': 'gpt_control_in_original_run', 'internal_tape': c.get('internal_tape'), 'endpoint_only': c.get('endpoint_only'), 'external_slots': c.get('external_slots'), 'gpt_incontext': c.get('gpt_incontext'), 'parity_read': 'internal≈gpt_ic≈endpoint<<external; matched GPT does not solve task either'}
    p210 = STAGES[210]
    if p210.exists():
        d210 = json.loads(p210.read_text(encoding='utf-8'))
        ladder['stage_210'] = {'status': 'gpt_ladder_pending' if not run_gpt else 'gpt_baseline_run', 'curve_soft_follow_test': d210.get('soft_follow_token', {}).get('test'), 'curve_external_cosine': d210.get('external_loop_cosine_test')}
    p212 = STAGES[212]
    if p212.exists():
        d212 = json.loads(p212.read_text(encoding='utf-8'))
        ladder['stage_212'] = {'status': 'curve_only_collision_in_original_run', 'collision_4way': d212.get('t1_collision_4way'), 'para_hard': d212.get('t2_para_hard'), 'cross_ref': 'For substrate semantic parity vs GPT use 209; 212 tested instance head on curve tape only'}
    if run_gpt:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpt210 = gpt_chain_parametric(device)
        ladder.setdefault('stage_210', {})['gpt_parametric_chain'] = gpt210
        ladder['stage_210']['status'] = 'gpt_baseline_run'
        ladder['stage_210']['parity_read'] = 'Compare gpt_parametric_chain acc_by_hop to curve soft_follow_token test; external cosine is curve-only affordance'
    return ladder

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-gpt', action='store_true', help='Run 210 GPT parametric chain baseline (~1 min GPU)')
    args = ap.parse_args()
    claim = load_claim()
    ladder = build_ladder(args.run_gpt)
    LADDER.write_text(json.dumps(ladder, indent=2), encoding='utf-8')
    for p in STAGES.values():
        patch_stage(p, claim, ladder)
    print(f'Wrote {LADDER}; patched {[p.name for p in STAGES.values() if p.exists()]}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())