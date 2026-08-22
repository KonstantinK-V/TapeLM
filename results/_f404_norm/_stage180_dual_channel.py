"""
Stage 180 — Dual-channel curve: fast ink + slow instance.

Architecture (not just loss):
  FAST  — local ink dynamics over BPE arcs (can follow the suffix).
  SLOW  — gated cumulative write budget (write-once-ish): early arcs consume
          capacity; suffix cannot fully rewind the slow state.

Readout for gates = concat(fast_last, slow_last) (and slow-only probe).

Losses:
  - fast: weak next + far (ink)
  - slow: past-bag from slow state; recover random instance written only early
  - retention on slow endpoints (same last piece, different prefixes)
  - light combine consistency

NO text CE. Reuses Stage177 ByteLevel BPE.

  python _stage180_dual_channel.py
  python _stage180_dual_channel.py --steps 10000
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage178_curve_retention as s178
import _stage179_curve_harden_B as s179
RES = Path('results')
CKPT_DIR = Path('checkpoints')
LOG = RES / '_stage180_log.txt'
DECISION = RES / 'stage180_decision.json'
MINI = RES / 'stage180_mini.md'
CKPT_OUT = CKPT_DIR / 'stage180_dual_channel.pt'
TOK_PATH = s177.TOK_PATH
PLAN = RES / 'plan_curve_dynamics.md'
SEED = 180
D = 128
D_SLOW = 128
MAX_ARCS = s177.MAX_ARCS
MICRO = 12
RET_PAIRS = 6
K_FAR = 4
LR = 0.0003
EVAL_EVERY = 1500
DEFAULT_STEPS = 10000
W_FAST = 0.4
W_SLOW_PAST = 1.0
W_SLOW_INST = 1.0
W_RET = 1.2
W_ORTH = 0.1

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

class SlowWriter(nn.Module):
    """Cumulative write-budget memory: early writes spend capacity; late can't overwrite."""

    def __init__(self, d_in: int, d_slow: int=D_SLOW):
        super().__init__()
        self.write = nn.Sequential(nn.Linear(d_in + d_slow, d_slow), nn.GELU(), nn.Linear(d_slow, d_slow))
        self.gate = nn.Linear(d_in + d_slow, 1)
        self.norm = nn.LayerNorm(d_slow)
        self.init = nn.Parameter(torch.zeros(d_slow))

    def forward(self, arcs: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, A, _ = arcs.shape
        device = arcs.device
        slow = self.init.unsqueeze(0).expand(B, -1).clone()
        budget = torch.ones(B, 1, device=device)
        outs = []
        for t in range(A):
            x = arcs[:, t]
            h = torch.cat([x, slow], dim=-1)
            g = torch.sigmoid(self.gate(h)) * budget
            w = self.write(h)
            mask = (~pad[:, t]).float().unsqueeze(-1)
            g = g * mask
            slow = self.norm(slow + g * w)
            budget = (budget - g).clamp(min=0.0)
            outs.append(slow)
        return torch.stack(outs, dim=1)

class DualChannel(nn.Module):

    def __init__(self, n_char: int):
        super().__init__()
        self.arc_enc = s177.ArcEncoder(n_char, d=D)
        self.fast = s177.ArcTransformer(d=D)
        self.slow = SlowWriter(D, D_SLOW)
        self.pred_next = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.pred_far = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.pred_past_slow = nn.Sequential(nn.Linear(D_SLOW, D), nn.GELU(), nn.Linear(D, D))
        self.pred_inst_slow = nn.Sequential(nn.Linear(D_SLOW, D_SLOW), nn.GELU(), nn.Linear(D_SLOW, D_SLOW))

    def encode_arcs(self, char_ids: torch.Tensor) -> torch.Tensor:
        return self.arc_enc(char_ids)

    def forward_channels(self, char_ids: torch.Tensor, pad: torch.Tensor, inst_prefix: torch.Tensor | None=None):
        """
        inst_prefix: [B,D] added only into early arc embeddings (handwriting/instance cue),
        slow must carry it; fast may ignore.
        """
        arcs = self.encode_arcs(char_ids)
        if inst_prefix is not None:
            B, A, _ = arcs.shape
            lengths = (~pad).sum(dim=1).clamp(min=2)
            pl = (lengths.float() * 0.5).long().clamp(min=1)
            arcs = arcs.clone()
            for b in range(B):
                arcs[b, :int(pl[b])] = arcs[b, :int(pl[b])] + 0.4 * inst_prefix[b]
        fast = self.fast(arcs, pad_mask=pad)
        slow = self.slow(arcs, pad)
        return (arcs, fast, slow)

def last_state(states: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    idx = (~pad).sum(dim=1).clamp(min=1).long() - 1
    return states[torch.arange(states.size(0), device=states.device), idx]

def cos_match(pred, target) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=-1)).mean()

def train_step_loss(model: DualChannel, char_ids, pad, device):
    B = char_ids.size(0)
    inst = F.normalize(torch.randn(B, D_SLOW, device=device), dim=-1)
    inst_fast = inst if D_SLOW == D else inst[:, :D]
    arcs, fast, slow = model.forward_channels(char_ids, pad, inst_prefix=inst_fast)
    stats = {}
    losses = []
    valid_n = ~pad[:, :-1] & ~pad[:, 1:]
    if valid_n.sum() > 0:
        pn = model.pred_next(fast[:, :-1])
        losses.append(W_FAST * cos_match(pn[valid_n], arcs[:, 1:][valid_n]))
        stats['cos_next'] = float(F.cosine_similarity(pn[valid_n], arcs[:, 1:][valid_n].detach(), dim=-1).mean())
    if arcs.size(1) > K_FAR + 1:
        valid_f = ~pad[:, :-K_FAR] & ~pad[:, K_FAR:]
        if valid_f.sum() > 0:
            pf = model.pred_far(fast[:, :-K_FAR])
            losses.append(W_FAST * cos_match(pf[valid_f], arcs[:, K_FAR:][valid_f]))
            stats['cos_far'] = float(F.cosine_similarity(pf[valid_f], arcs[:, K_FAR:][valid_f].detach(), dim=-1).mean())
    A = arcs.size(1)
    past_ls = []
    past_cs = []
    for t in range(3, A):
        valid = ~pad[:, t]
        if valid.sum() < 1:
            continue
        bags = []
        for b in range(B):
            if pad[b, t]:
                bags.append(torch.zeros(D, device=device))
                continue
            early = arcs[b, :t]
            em = ~pad[b, :t]
            bags.append(early[em].mean(0) if em.any() else torch.zeros(D, device=device))
        bag = torch.stack(bags, 0)
        pred = model.pred_past_slow(slow[:, t])
        past_ls.append(cos_match(pred[valid], bag[valid]))
        past_cs.append(float(F.cosine_similarity(pred[valid], bag[valid].detach(), dim=-1).mean()))
    if past_ls:
        losses.append(W_SLOW_PAST * torch.stack(past_ls).mean())
        stats['cos_past_slow'] = float(np.mean(past_cs))
    z_slow = last_state(slow, pad)
    pi = model.pred_inst_slow(z_slow)
    losses.append(W_SLOW_INST * cos_match(pi, inst))
    stats['cos_inst_slow'] = float(F.cosine_similarity(pi, inst, dim=-1).mean())
    z_fast = last_state(fast, pad)
    if z_slow.size(-1) == z_fast.size(-1):
        sim = F.cosine_similarity(z_slow, z_fast.detach(), dim=-1).abs().mean()
        losses.append(W_ORTH * sim)
        stats['slow_fast_sim'] = float(sim.detach())
    loss = sum(losses) if losses else fast.sum() * 0.0
    stats['loss'] = float(loss.detach())
    return (loss, stats, fast, slow)

def retention_slow(model: DualChannel, batch):
    a_ids, b_ids, pad_a, pad_b = batch
    _, _, slow_a = model.forward_channels(a_ids, pad_a, inst_prefix=None)
    _, _, slow_b = model.forward_channels(b_ids, pad_b, inst_prefix=None)
    za, zb = (last_state(slow_a, pad_a), last_state(slow_b, pad_b))
    sim = F.cosine_similarity(za, zb, dim=-1)
    l = F.relu(sim - 0.4).mean() + 0.2 * sim.mean()
    return (W_RET * l, {'ret_slow_cos': float(sim.mean().detach())})

class DualGateWrap(nn.Module):
    """Expose combined (or slow) states for A/B gates expecting forward_states(char_ids)."""

    def __init__(self, m: DualChannel, mode: str='combined'):
        super().__init__()
        self.m = m
        self.mode = mode

    def forward_states(self, char_ids, pad_mask=None):
        if pad_mask is None:
            pad_mask = torch.zeros(char_ids.size(0), char_ids.size(1), dtype=torch.bool, device=char_ids.device)
        arcs, fast, slow = self.m.forward_channels(char_ids, pad_mask, inst_prefix=None)
        if self.mode == 'slow':
            return slow
        if self.mode == 'fast':
            return fast
        return 0.5 * fast + 0.5 * slow

def gate_A_modes(model, docs, stoi, device, rng):
    out = {}
    for mode in ('combined', 'slow', 'fast'):
        wrap = DualGateWrap(model, mode=mode).to(device)
        out[mode] = s177.gate_A(wrap, docs, stoi, device, rng, n_pairs=60)
    return out

def gate_B_modes(model, tok, stoi, device, rng):
    out = {}
    for mode in ('combined', 'slow', 'fast'):
        wrap = DualGateWrap(model, mode=mode).to(device)
        out[mode] = s179.gate_B(wrap, tok, stoi, device, rng)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage180 start {datetime.now(timezone.utc).isoformat()}')
    log('Dual-channel: FAST ink Transformer + SLOW write-budget memory')
    log(f'plan={PLAN}')
    if not TOK_PATH.exists():
        raise FileNotFoundError(TOK_PATH)
    tok = Tokenizer.from_file(str(TOK_PATH))
    text = s170.load_corpus(max_chars=20000000)
    chars = sorted(set(text) | {' '})
    itos = ['<pad>'] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = s177.build_piece_docs(tok, text)
    hold = docs[int(0.8 * len(docs)):] or docs[-100:]
    train = docs[:int(0.8 * len(docs))] or docs
    index = s178.build_same_last_index(train)
    log(f'docs={len(docs)} same-last={len(index)} V={tok.get_vocab_size()} d={D}')
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = DualChannel(len(itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    rng = random.Random(SEED)
    model.eval()
    A0 = gate_A_modes(model, hold, stoi, device, random.Random(SEED))
    B0 = gate_B_modes(model, tok, stoi, device, random.Random(SEED + 1))
    for mode in ('combined', 'slow', 'fast'):
        a, b = (A0[mode], B0[mode])
        log(f"  init [{mode}] A_same={a['mean_cos_same_last_piece']:.3f}→{a['verdict']} | B para={b['mean_cos_paraphrase']:.3f} hard={b['mean_cos_hard_spelling']:.3f}→{b['verdict']}")
    Af, Bf = (A0, B0)
    running = None
    model.train()
    for step in range(1, args.steps + 1):
        x, pad = s177.sample_batch(train, stoi, MICRO, rng, device)
        loss, st, _, _ = train_step_loss(model, x, pad, device)
        ret_batch = s178.sample_retention_pair_batch(index, stoi, RET_PAIRS, rng, device)
        if ret_batch is not None:
            loss_r, st_r = retention_slow(model, ret_batch)
            loss = loss + loss_r
            st.update(st_r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = st['loss'] if running is None else 0.95 * running + 0.05 * st['loss']
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            Af = gate_A_modes(model, hold, stoi, device, random.Random(SEED + step))
            Bf = gate_B_modes(model, tok, stoi, device, random.Random(SEED + step + 3))
            ac, bc = (Af['combined'], Bf['combined'])
            as_, bs = (Af['slow'], Bf['slow'])
            log(f"  step {step}: loss~{running:.3f} inst={st.get('cos_inst_slow', 0):.3f} past={st.get('cos_past_slow', 0):.3f} ret={st.get('ret_slow_cos', 0):.3f} sf_sim={st.get('slow_fast_sim', 0):.3f} | A_comb={ac['mean_cos_same_last_piece']:.3f} A_slow={as_['mean_cos_same_last_piece']:.3f} | B_comb {bc['verdict']} para={bc['mean_cos_paraphrase']:.3f} hard={bc['mean_cos_hard_spelling']:.3f} | B_slow {bs['verdict']}")
            model.train()
            torch.save({'model': model.state_dict(), 'stoi': stoi, 'step': step, 'A': Af, 'B': Bf}, CKPT_OUT)
    ac, bc = (Af['combined'], Bf['combined'])
    as_, bs = (Af['slow'], Bf['slow'])
    a_ok = ac['mean_cos_same_last_piece'] < 0.9
    if a_ok and 'PASS' in bc['verdict']:
        overall = 'DUAL_A_YES_B_YES'
    elif a_ok and 'WEAK' in bc['verdict']:
        overall = 'DUAL_A_YES_B_WEAK'
    elif a_ok:
        overall = 'DUAL_A_YES_B_FAIL'
    else:
        overall = 'DUAL_A_FAIL'
    arch = {'slow_A_same': as_['mean_cos_same_last_piece'], 'fast_A_same': Af['fast']['mean_cos_same_last_piece'], 'slow_better_than_fast_for_A': as_['mean_cos_same_last_piece'] + 0.05 < Af['fast']['mean_cos_same_last_piece']}
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'dual_channel_180', 'overall': overall, 'architecture': {'fast': 'causal Transformer over BPE arc ink', 'slow': "cumulative write-budget memory (early spends, late can't fully overwrite)", 'readout': 'combined = 0.5 fast + 0.5 slow; also report slow/fast alone'}, 'arch_check': arch, 'final_A': Af, 'final_B': Bf, 'init_A': A0, 'init_B': B0, 'note': 'Architectural anti-coil: slow channel write-once-ish. B may still need semantic pressure.', 'next': 'If slow A << fast A: architecture works for retention. If B still form>>meaning: put semantic/contrastive load on SLOW only. Do not soak fast next-local.'}
    write_json(DECISION, out)
    MINI.write_text('\n'.join(['# Stage180 — dual channel', '', f'**Overall:** `{overall}`', '', f"- A combined same={ac['mean_cos_same_last_piece']:.3f}; slow={as_['mean_cos_same_last_piece']:.3f}; fast={Af['fast']['mean_cos_same_last_piece']:.3f}", f"- B combined: {bc['verdict']} para={bc['mean_cos_paraphrase']:.3f} hard={bc['mean_cos_hard_spelling']:.3f}", f"- arch: slow_better_A={arch['slow_better_than_fast_for_A']}", f"- {out['next']}", '']), encoding='utf-8')
    log(f'[180] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())