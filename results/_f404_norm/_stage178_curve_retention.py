"""
Stage 178 — Objective flip: force prefix (not new units).

Reuse Stage177 ByteLevel BPE pieces as arcs.
Change the teacher so the goal *requires* the prefix:

  1) RETENTION — same last piece, different prefixes → push final states apart
  2) PAST-BAG — from state_t predict mean of earlier arcs (exclude last)
  3) PREDICT-FAR — from state_t predict arc_{t+k}
  4) INSTANCE — random cue only on prefix arcs; recover cue from final state

Next-local kept weak (so dynamics don't die). NO text CE.

Gate A: same last BPE piece / different prefix (must move if retention works).

  python _stage178_curve_retention.py
  python _stage178_curve_retention.py --steps 12000
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
RES = Path('results')
CKPT_DIR = Path('checkpoints')
LOG = RES / '_stage178_log.txt'
DECISION = RES / 'stage178_decision.json'
MINI = RES / 'stage178_mini.md'
CKPT_OUT = CKPT_DIR / 'stage178_curve_retention.pt'
TOK_PATH = s177.TOK_PATH
PLAN = RES / 'plan_curve_dynamics.md'
SEED = 178
D = s177.D
MAX_ARCS = s177.MAX_ARCS
MICRO = 16
RET_PAIRS = 8
K_FAR = 4
PAST_SKIP = 1
LR = 0.0003
EVAL_EVERY = 1500
DEFAULT_STEPS = 12000
W_NEXT = 0.15
W_FAR = 0.5
W_PAST = 1.0
W_RET = 1.5
W_INST = 1.0
EARLY_FAIL_SAME = 0.985
EARLY_PASS_SAME = 0.9

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

class RetentionModel(nn.Module):

    def __init__(self, n_char: int):
        super().__init__()
        self.arc_enc = s177.ArcEncoder(n_char)
        self.tr = s177.ArcTransformer()
        self.pred_next = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.pred_far = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.pred_past = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.pred_inst = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

    def encode_arcs(self, char_ids: torch.Tensor) -> torch.Tensor:
        return self.arc_enc(char_ids)

    def forward_states(self, arc_emb: torch.Tensor, pad_mask: torch.Tensor | None=None) -> torch.Tensor:
        return self.tr(arc_emb, pad_mask=pad_mask)

def build_same_last_index(docs: list[list[str]], max_per_last: int=40) -> dict[str, list[list[str]]]:
    by_last: dict[str, list[list[str]]] = defaultdict(list)
    for doc in docs:
        if len(doc) < 12:
            continue
        for i in range(10, min(len(doc), 80)):
            last = doc[i]
            seq = doc[max(0, i - (MAX_ARCS - 1)):i + 1]
            if len(by_last[last]) < max_per_last:
                pref = tuple(seq[:-1])
                if all((tuple(s[:-1]) != pref for s in by_last[last])):
                    by_last[last].append(seq)
    return {k: v for k, v in by_last.items() if len(v) >= 2}

def sample_retention_pair_batch(index: dict[str, list[list[str]]], stoi: dict, n_pairs: int, rng: random.Random, device):
    keys = [k for k, v in index.items() if len(v) >= 2]
    if not keys:
        return None
    a_ids, b_ids, pads_a, pads_b = ([], [], [], [])
    for _ in range(n_pairs):
        last = keys[rng.randint(0, len(keys) - 1)]
        seqs = index[last]
        sa, sb = rng.sample(seqs, 2)

        def pack(seq):
            seq = seq[-MAX_ARCS:]
            pad_n = MAX_ARCS - len(seq)
            window = seq + [''] * pad_n
            return (s177.pieces_to_char_ids(window, stoi), torch.tensor([x == '' for x in window], dtype=torch.bool))
        ca, pa = pack(sa)
        cb, pb = pack(sb)
        a_ids.append(ca)
        b_ids.append(cb)
        pads_a.append(pa)
        pads_b.append(pb)
    return (torch.stack(a_ids).to(device), torch.stack(b_ids).to(device), torch.stack(pads_a).to(device), torch.stack(pads_b).to(device))

def last_valid_index(pad: torch.Tensor) -> torch.Tensor:
    B, A = pad.shape
    lengths = (~pad).sum(dim=1).clamp(min=1)
    return (lengths - 1).long()

def gather_last(states: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    idx = last_valid_index(pad)
    return states[torch.arange(states.size(0), device=states.device), idx]

def cos_loss_match(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=-1)).mean()

def dynamics_bundle(model: RetentionModel, char_ids: torch.Tensor, pad: torch.Tensor, rng_torch=None):
    """next (weak) + far + past-bag + instance-on-prefix."""
    B, A, _ = char_ids.shape
    device = char_ids.device
    arc = model.encode_arcs(char_ids)
    inst = F.normalize(torch.randn(B, D, device=device), dim=-1)
    lengths = (~pad).sum(dim=1).clamp(min=2)
    prefix_lens = (lengths.float() * 0.5).long().clamp(min=1)
    arc_inst = arc.clone()
    for b in range(B):
        pl = int(prefix_lens[b].item())
        arc_inst[b, :pl] = arc_inst[b, :pl] + 0.35 * inst[b]
    states = model.forward_states(arc_inst, pad_mask=pad)
    stats = {}
    losses = []
    valid_n = ~pad[:, :-1] & ~pad[:, 1:]
    if valid_n.sum() > 0:
        pred_n = model.pred_next(states[:, :-1])
        l_next = cos_loss_match(pred_n[valid_n], arc[:, 1:][valid_n])
        losses.append(W_NEXT * l_next)
        stats['cos_next'] = float(F.cosine_similarity(pred_n[valid_n], arc[:, 1:][valid_n].detach(), dim=-1).mean())
    else:
        stats['cos_next'] = 0.0
    if A > K_FAR + 1:
        valid_f = ~pad[:, :-K_FAR] & ~pad[:, K_FAR:]
        if valid_f.sum() > 0:
            pred_f = model.pred_far(states[:, :-K_FAR])
            tgt_f = arc[:, K_FAR:]
            l_far = cos_loss_match(pred_f[valid_f], tgt_f[valid_f])
            losses.append(W_FAR * l_far)
            stats['cos_far'] = float(F.cosine_similarity(pred_f[valid_f], tgt_f[valid_f].detach(), dim=-1).mean())
        else:
            stats['cos_far'] = 0.0
    else:
        stats['cos_far'] = 0.0
    cos_past_acc = []
    past_losses = []
    for t in range(3, A):
        valid = ~pad[:, t]
        if valid.sum() < 1:
            continue
        end = max(1, t - PAST_SKIP)
        bag = []
        for b in range(B):
            if pad[b, t]:
                bag.append(torch.zeros(D, device=device))
                continue
            early = arc[b, :end]
            early_mask = ~pad[b, :end]
            if early_mask.sum() < 1:
                bag.append(torch.zeros(D, device=device))
            else:
                bag.append(early[early_mask].mean(dim=0))
        bag_t = torch.stack(bag, 0)
        pred_p = model.pred_past(states[:, t])
        l = (1.0 - F.cosine_similarity(pred_p[valid], bag_t[valid].detach(), dim=-1)).mean()
        past_losses.append(l)
        cos_past_acc.append(float(F.cosine_similarity(pred_p[valid], bag_t[valid].detach(), dim=-1).mean()))
    if past_losses:
        l_past = torch.stack(past_losses).mean()
        losses.append(W_PAST * l_past)
        stats['cos_past'] = float(np.mean(cos_past_acc))
    else:
        stats['cos_past'] = 0.0
    z_last = gather_last(states, pad)
    pred_i = model.pred_inst(z_last)
    l_inst = cos_loss_match(pred_i, inst)
    losses.append(W_INST * l_inst)
    stats['cos_inst'] = float(F.cosine_similarity(pred_i, inst, dim=-1).mean())
    if not losses:
        return (states.sum() * 0.0, stats)
    loss = sum(losses)
    stats['loss_dyn'] = float(loss.detach())
    return (loss, stats)

def retention_loss(model: RetentionModel, batch, stoi_unused=None):
    a_ids, b_ids, pad_a, pad_b = batch
    za = gather_last(model.forward_states(model.encode_arcs(a_ids), pad_a), pad_a)
    zb = gather_last(model.forward_states(model.encode_arcs(b_ids), pad_b), pad_b)
    sim = F.cosine_similarity(za, zb, dim=-1)
    l = F.relu(sim - 0.5).mean() + 0.25 * sim.mean()
    return (W_RET * l, {'ret_cos': float(sim.mean().detach()), 'ret_hinge': float(l.detach())})

@torch.no_grad()
def encode_seq(model, pieces, stoi, device):
    pieces = pieces[-MAX_ARCS:] or ['.']
    char_ids = s177.pieces_to_char_ids(pieces, stoi).unsqueeze(0).to(device)
    pad = torch.zeros(1, len(pieces), dtype=torch.bool, device=device)
    return model.forward_states(model.encode_arcs(char_ids), pad)[0]

def cos(a, b) -> float:
    return float(F.cosine_similarity(F.normalize(a, dim=0), F.normalize(b, dim=0), dim=0))

def gate_A(model, docs, stoi, device, rng, n_pairs=80):
    return s177.gate_A(model, docs, stoi, device, rng, n_pairs=n_pairs)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage178 start {datetime.now(timezone.utc).isoformat()}')
    log('Objective flip: retention + past-bag + far + prefix-instance (BPE units from 177)')
    log(f'weights next={W_NEXT} far={W_FAR} past={W_PAST} ret={W_RET} inst={W_INST} k_far={K_FAR}')
    if not TOK_PATH.exists():
        raise FileNotFoundError(f'need {TOK_PATH} from Stage177')
    tok = Tokenizer.from_file(str(TOK_PATH))
    text = s170.load_corpus(max_chars=20000000)
    chars = sorted(set(text) | {' '})
    itos = ['<pad>'] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = s177.build_piece_docs(tok, text)
    hold = docs[int(0.8 * len(docs)):] or docs[-100:]
    train = docs[:int(0.8 * len(docs))] or docs
    index = build_same_last_index(train)
    log(f'docs={len(docs)} same-last keys={len(index)} V_bpe={tok.get_vocab_size()}')
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = RetentionModel(len(itos)).to(device)
    model.forward_states_gate = model.forward_states

    class GateWrap(nn.Module):

        def __init__(self, m):
            super().__init__()
            self.m = m

        def encode_arcs(self, x):
            return self.m.encode_arcs(x)

        def forward_states(self, char_ids, pad_mask=None):
            return self.m.forward_states(self.m.encode_arcs(char_ids), pad_mask)
    gmodel = GateWrap(model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    rng = random.Random(SEED)
    A0 = gate_A(gmodel, hold, stoi, device, random.Random(SEED))
    log(f"  init A: same={A0['mean_cos_same_last_piece']:.3f} diff={A0['mean_cos_diff_last_piece']:.3f} → {A0['verdict']}")
    model.train()
    Af = A0
    early = None
    running = None
    for step in range(1, args.steps + 1):
        x, pad = s177.sample_batch(train, stoi, MICRO, rng, device)
        loss_d, st = dynamics_bundle(model, x, pad)
        ret_batch = sample_retention_pair_batch(index, stoi, RET_PAIRS, rng, device)
        if ret_batch is not None:
            loss_r, st_r = retention_loss(model, ret_batch)
            loss = loss_d + loss_r
            st.update(st_r)
        else:
            loss = loss_d
            st['ret_cos'] = 1.0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(loss.detach()) if running is None else 0.95 * running + 0.05 * float(loss.detach())
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            Af = gate_A(gmodel, hold, stoi, device, random.Random(SEED + step))
            log(f"  step {step}: loss~{running:.3f} next={st.get('cos_next', 0):.3f} far={st.get('cos_far', 0):.3f} past={st.get('cos_past', 0):.3f} inst={st.get('cos_inst', 0):.3f} ret_cos={st.get('ret_cos', 0):.3f} A_same={Af['mean_cos_same_last_piece']:.3f} A_diff={Af['mean_cos_diff_last_piece']:.3f} → {Af['verdict']}")
            model.train()
            torch.save({'model': model.state_dict(), 'stoi': stoi, 'step': step, 'A': Af}, CKPT_OUT)
            if step >= 4500 and Af['mean_cos_same_last_piece'] >= EARLY_FAIL_SAME:
                early = 'EARLY_FAIL_STILL_WIPE'
                log(f'  [{early}] stop @ {step}')
                break
            if Af['mean_cos_same_last_piece'] < EARLY_PASS_SAME and 'PASS' in Af['verdict']:
                early = 'EARLY_PASS_PREFIX'
                log(f'  [{early}] stop @ {step}')
                break
    if 'PASS' in Af['verdict']:
        overall = 'CURVE_RETENTION_CONTEXT_YES'
    elif Af['mean_cos_same_last_piece'] >= 0.95:
        overall = 'CURVE_RETENTION_CONTEXT_NULL'
    else:
        overall = 'CURVE_RETENTION_CONTEXT_WEAK'
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'curve_retention_178', 'overall': overall, 'early': early, 'steps_ran': step, 'objective': {'retention': 'same-last-piece pairs → push endpoints apart', 'past_bag': 'state_t → mean of earlier arcs (exclude last)', 'predict_far': f'state_t → arc_t+{K_FAR}', 'instance': 'random cue on prefix half only → recover from final', 'next_local_weight': W_NEXT}, 'units': 'Stage177 ByteLevel BPE pieces (unchanged)', 'A': Af, 'init_A': A0, 'note': 'Falsify: can objective-flip beat last-unit wipe without new tokenizer?', 'next': 'If YES: harden + gate B. If NULL: this objective family insufficient at this scale — need stronger instance/handwriting or abandon context-from-curve under local ink.'}
    write_json(DECISION, out)
    MINI.write_text('\n'.join(['# Stage178 — retention objective', '', f'**Overall:** `{overall}`' + (f' ({early})' if early else ''), '', f"- A: {Af['verdict']} same={Af['mean_cos_same_last_piece']:.3f} diff={Af['mean_cos_diff_last_piece']:.3f}", '- losses: retention + past-bag + far + prefix-instance (next weak)', f"- {out['next']}", '']), encoding='utf-8')
    log(f'[178] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())