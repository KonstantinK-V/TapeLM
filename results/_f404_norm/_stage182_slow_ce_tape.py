"""
Stage 182 — Hybrid: dual-channel tape + CE on SLOW («какой следующий id куска?»).

Matches what we want:
  - FAST/SLOW draw the BPE tape (ink + write-budget memory)
  - From SLOW state at t, predict next BPE piece id (CE) — GPT-like use of context
  - FAST keeps weak local ink (optional, low weight)

Same tokenizer/corpus/scale as 180/181.
Gates: A/B on slow (+ combined); prefix-shuffle ablation on slow-CE (like 181).

  python _stage182_slow_ce_tape.py
  python _stage182_slow_ce_tape.py --steps 10000
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
import _stage180_dual_channel as s180
import _stage181_ce_control as s181
RES = Path('results')
CKPT_DIR = Path('checkpoints')
LOG = RES / '_stage182_log.txt'
DECISION = RES / 'stage182_decision.json'
MINI = RES / 'stage182_mini.md'
CKPT_OUT = CKPT_DIR / 'stage182_slow_ce_tape.pt'
TOK_PATH = s177.TOK_PATH
PLAN = RES / 'plan_curve_dynamics.md'
DEC181 = RES / 'stage181_decision.json'
SEED = 182
D = s180.D
D_SLOW = s180.D_SLOW
MAX_ARCS = s177.MAX_ARCS
MICRO = 16
LR = 0.0003
EVAL_EVERY = 1500
DEFAULT_STEPS = 10000
W_CE = 1.0
W_FAST = 0.15
W_RET = 0.3
PAD = '[PAD]'

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

class DualSlowCE(nn.Module):

    def __init__(self, n_char: int, vocab: int):
        super().__init__()
        self.backbone = s180.DualChannel(n_char)
        self.lm_head = nn.Linear(D_SLOW, vocab, bias=False)

    def forward_channels(self, char_ids, pad, inst_prefix=None):
        return self.backbone.forward_channels(char_ids, pad, inst_prefix=inst_prefix)

    def logits_from_slow(self, slow: torch.Tensor) -> torch.Tensor:
        return self.lm_head(slow)

def ids_to_char_batch(tok: Tokenizer, id_batch: torch.Tensor, stoi: dict, pad_id: int) -> torch.Tensor:
    """[B,A] token ids → [B,A,C] char ids for arc ink encoder."""
    B, A = id_batch.shape
    rows = []
    for b in range(B):
        pieces = []
        for t in range(A):
            tid = int(id_batch[b, t].item())
            if tid == pad_id:
                pieces.append('')
            else:
                pieces.append(tok.decode([tid], skip_special_tokens=False) or '')
        rows.append(s177.pieces_to_char_ids(pieces, stoi))
    return torch.stack(rows, 0)

def sample_id_batch(docs: list[list[int]], batch: int, rng: random.Random, device, pad_id: int):
    xs = []
    for _ in range(batch):
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < 8:
            doc = doc * 4
        max_start = max(0, len(doc) - MAX_ARCS)
        s = rng.randint(0, max_start) if max_start > 0 else 0
        window = doc[s:s + MAX_ARCS]
        if len(window) < MAX_ARCS:
            window = window + [pad_id] * (MAX_ARCS - len(window))
        xs.append(window)
    return torch.tensor(xs, dtype=torch.long, device=device)

def train_loss(model: DualSlowCE, tok, id_batch, stoi, pad_id, device):
    pad = id_batch == pad_id
    char_ids = ids_to_char_batch(tok, id_batch, stoi, pad_id).to(device)
    arcs, fast, slow = model.forward_channels(char_ids, pad, inst_prefix=None)
    logits = model.logits_from_slow(slow[:, :-1])
    target = id_batch[:, 1:]
    valid = ~pad[:, :-1] & ~pad[:, 1:]
    if valid.sum() < 1:
        return (fast.sum() * 0.0, {'ce': 0.0, 'ppl': 0.0, 'cos_next': 0.0})
    ce = F.cross_entropy(logits[valid], target[valid])
    loss = W_CE * ce
    stats = {'ce': float(ce.detach()), 'ppl': float(torch.exp(ce.detach().clamp(max=20)))}
    valid_n = valid
    if valid_n.sum() > 0:
        pn = model.backbone.pred_next(fast[:, :-1])
        l_fast = (1.0 - F.cosine_similarity(pn[valid_n], arcs[:, 1:][valid_n].detach(), dim=-1)).mean()
        loss = loss + W_FAST * l_fast
        stats['cos_next'] = float(F.cosine_similarity(pn[valid_n], arcs[:, 1:][valid_n].detach(), dim=-1).mean())
    return (loss, stats)

def retention_slow(model: DualSlowCE, tok, stoi, pad_id, batch_ids_a, batch_ids_b, device):
    """batch_ids: [P,A] each"""

    def run(ids):
        pad = ids == pad_id
        char_ids = ids_to_char_batch(tok, ids, stoi, pad_id).to(device)
        _, _, slow = model.forward_channels(char_ids, pad)
        return s180.last_state(slow, pad)
    za, zb = (run(batch_ids_a), run(batch_ids_b))
    sim = F.cosine_similarity(za, zb, dim=-1)
    l = F.relu(sim - 0.5).mean() + 0.15 * sim.mean()
    return (W_RET * l, {'ret_cos': float(sim.mean().detach())})

def sample_ret_id_pairs(index_by_last: dict, n_pairs: int, rng: random.Random, pad_id: int, device):
    keys = [k for k, v in index_by_last.items() if len(v) >= 2]
    if not keys:
        return None
    a_list, b_list = ([], [])
    for _ in range(n_pairs):
        last = keys[rng.randint(0, len(keys) - 1)]
        sa, sb = rng.sample(index_by_last[last], 2)

        def pack(seq):
            seq = seq[-MAX_ARCS:]
            if len(seq) < MAX_ARCS:
                seq = seq + [pad_id] * (MAX_ARCS - len(seq))
            return seq
        a_list.append(pack(sa))
        b_list.append(pack(sb))
    return (torch.tensor(a_list, dtype=torch.long, device=device), torch.tensor(b_list, dtype=torch.long, device=device))

def build_same_last_id_index(docs: list[list[int]], max_per: int=40):
    from collections import defaultdict
    by_last = defaultdict(list)
    for doc in docs:
        if len(doc) < 12:
            continue
        for i in range(10, min(len(doc), 80)):
            last = doc[i]
            seq = doc[max(0, i - (MAX_ARCS - 1)):i + 1]
            if len(by_last[last]) < max_per:
                pref = tuple(seq[:-1])
                if all((tuple(s[:-1]) != pref for s in by_last[last])):
                    by_last[last].append(seq)
    return {k: v for k, v in by_last.items() if len(v) >= 2}

class SlowGateWrap(nn.Module):

    def __init__(self, model: DualSlowCE, tok, stoi, pad_id, mode='slow'):
        super().__init__()
        self.model = model
        self.tok = tok
        self.stoi = stoi
        self.pad_id = pad_id
        self.mode = mode

    def forward_states(self, char_ids, pad_mask=None):
        if pad_mask is None:
            pad_mask = torch.zeros(char_ids.size(0), char_ids.size(1), dtype=torch.bool, device=char_ids.device)
        _, fast, slow = self.model.forward_channels(char_ids, pad_mask)
        if self.mode == 'fast':
            return fast
        if self.mode == 'combined':
            return 0.5 * fast + 0.5 * slow
        return slow

@torch.no_grad()
def encode_id_seq_slow(model, tok, stoi, pad_id, ids: list[int], device):
    ids = ids[-MAX_ARCS:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pad = x == pad_id
    char_ids = ids_to_char_batch(tok, x, stoi, pad_id).to(device)
    _, _, slow = model.forward_channels(char_ids, pad)
    return slow[0]

def gate_A_ids(model, docs, tok, stoi, pad_id, device, rng, n_pairs=80):
    from collections import defaultdict
    by_last = defaultdict(list)
    for doc in docs:
        if len(doc) < 12:
            continue
        for i in range(8, min(len(doc), 80)):
            last = doc[i]
            seq = doc[max(0, i - (MAX_ARCS - 1)):i + 1]
            if len(by_last[last]) < 40:
                pref = tuple(seq[:-1])
                if all((tuple(s[:-1]) != pref for s in by_last[last])):
                    by_last[last].append(seq)
    pairs_same = []
    for last, seqs in by_last.items():
        uniq = {}
        for s in seqs:
            key = tuple(s[:-1])
            if key not in uniq:
                uniq[key] = s
            if len(uniq) >= 2:
                break
        if len(uniq) >= 2:
            vals = list(uniq.values())
            pairs_same.append((vals[0], vals[1]))
        if len(pairs_same) >= n_pairs:
            break
    rng.shuffle(pairs_same)
    pairs_same = pairs_same[:n_pairs]
    flat = [s for seqs in list(by_last.values())[:200] for s in seqs[:3]]
    pairs_diff = []
    for _ in range(n_pairs * 4):
        if len(flat) < 2:
            break
        a, b = rng.sample(flat, 2)
        if a[-1] != b[-1]:
            pairs_diff.append((a, b))
        if len(pairs_diff) >= n_pairs:
            break

    def last_h(seq):
        h = encode_id_seq_slow(model, tok, stoi, pad_id, seq, device)
        return h[-1]

    def c(a, b):
        return float(F.cosine_similarity(F.normalize(a, dim=0), F.normalize(b, dim=0), dim=0))
    cos_same = [c(last_h(a), last_h(b)) for a, b in pairs_same]
    cos_diff = [c(last_h(a), last_h(b)) for a, b in pairs_diff]
    m_same = float(np.mean(cos_same)) if cos_same else 1.0
    m_diff = float(np.mean(cos_diff)) if cos_diff else 0.0
    if m_same >= 0.98:
        verdict = 'A_FAIL_LAST_TOKEN_WIPES'
    elif m_same < 0.9 and m_same - m_diff < 0.35:
        verdict = 'A_PASS_PREFIX_VISIBLE'
    else:
        verdict = 'A_WEAK_PARTIAL'
    return {'verdict': verdict, 'mean_cos_same_last_piece': m_same, 'mean_cos_diff_last_piece': m_diff, 'n_same': len(cos_same), 'n_diff': len(cos_diff)}

def gate_B_slow(model, tok, stoi, pad_id, device, rng):
    wrap = SlowGateWrap(model, tok, stoi, pad_id, mode='slow').to(device)
    return s179.gate_B(wrap, tok, stoi, device, rng)

@torch.no_grad()
def slow_ce_ablation(model, tok, stoi, docs, pad_id, device, rng, n=40):
    """CE of next-ids from slow: natural vs prefix-shuffled (same suffix)."""
    nat, shuf = ([], [])
    for _ in range(n * 2):
        if len(nat) >= n:
            break
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < MAX_ARCS:
            continue
        s = rng.randint(0, len(doc) - MAX_ARCS)
        window = doc[s:s + MAX_ARCS]
        suf = max(8, MAX_ARCS // 3)
        prefix, suffix = (window[:-suf], window[-suf:])
        shuf_p = prefix.copy()
        rng.shuffle(shuf_p)

        def ce_of(ids):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            pad = x == pad_id
            char_ids = ids_to_char_batch(tok, x, stoi, pad_id).to(device)
            _, _, slow = model.forward_channels(char_ids, pad)
            logits = model.logits_from_slow(slow[:, :-1])
            target = x[:, 1:]
            start = max(0, MAX_ARCS - suf - 1)
            logits_s = logits[:, start:]
            target_s = target[:, start:]
            pad_s = pad[:, 1:][:, start:]
            valid = ~pad_s
            if valid.sum() < 1:
                return None
            return float(F.cross_entropy(logits_s[valid], target_s[valid]))
        a, b = (ce_of(window), ce_of(shuf_p + suffix))
        if a is not None and b is not None:
            nat.append(a)
            shuf.append(b)
    if not nat:
        return {'delta_shuf_minus_nat': 0.0, 'nat': 0.0, 'shuf': 0.0, 'n': 0}
    mn, ms = (float(np.mean(nat)), float(np.mean(shuf)))
    return {'mean_ce_natural': mn, 'mean_ce_prefix_shuffled': ms, 'delta_shuf_minus_nat': ms - mn, 'n': len(nat), 'note': 'positive => slow-CE uses prefix (context)'}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage182 start {datetime.now(timezone.utc).isoformat()}')
    log('Hybrid: dual-channel tape + SLOW predicts next BPE id (CE)')
    log(f'plan={PLAN} | unlock: piece-id CE from slow only')
    if not TOK_PATH.exists():
        raise FileNotFoundError(TOK_PATH)
    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    V = tok.get_vocab_size()
    text = s170.load_corpus(max_chars=20000000)
    chars = sorted(set(text) | {' '})
    itos = ['<pad>'] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = s181.build_id_docs(tok, text)
    hold = docs[int(0.8 * len(docs)):] or docs[-100:]
    train = docs[:int(0.8 * len(docs))] or docs
    index = build_same_last_id_index(train)
    log(f'docs={len(docs)} V={V} same-last={len(index)} d={D}')
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = DualSlowCE(len(itos), V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    model.eval()
    A0 = gate_A_ids(model, hold, tok, stoi, pad_id, device, random.Random(SEED))
    B0 = gate_B_slow(model, tok, stoi, pad_id, device, random.Random(SEED + 1))
    Ab0 = slow_ce_ablation(model, tok, stoi, hold, pad_id, device, random.Random(SEED + 2))
    log(f"  init A: same={A0['mean_cos_same_last_piece']:.3f} → {A0['verdict']}")
    log(f"  init B: para={B0['mean_cos_paraphrase']:.3f} hard={B0['mean_cos_hard_spelling']:.3f} → {B0['verdict']}")
    log(f"  init ablΔ={Ab0['delta_shuf_minus_nat']:+.4f}")
    history = []
    Af, Bf, Abf = (A0, B0, Ab0)
    running = None
    model.train()
    for step in range(1, args.steps + 1):
        ids = sample_id_batch(train, MICRO, rng, device, pad_id)
        loss, st = train_loss(model, tok, ids, stoi, pad_id, device)
        pair = sample_ret_id_pairs(index, 4, rng, pad_id, device)
        if pair is not None:
            loss_r, st_r = retention_slow(model, tok, stoi, pad_id, pair[0], pair[1], device)
            loss = loss + loss_r
            st.update(st_r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = st['ce'] if running is None else 0.95 * running + 0.05 * st['ce']
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            Af = gate_A_ids(model, hold, tok, stoi, pad_id, device, random.Random(SEED + step))
            Bf = gate_B_slow(model, tok, stoi, pad_id, device, random.Random(SEED + step + 3))
            Abf = slow_ce_ablation(model, tok, stoi, hold, pad_id, device, random.Random(SEED + step + 5))
            gap = Bf['mean_cos_hard_spelling'] - Bf['mean_cos_paraphrase']
            row = {'step': step, 'ce': running, 'A_same': Af['mean_cos_same_last_piece'], 'A': Af['verdict'], 'para': Bf['mean_cos_paraphrase'], 'hard': Bf['mean_cos_hard_spelling'], 'gap': gap, 'B': Bf['verdict'], 'ablation_delta': Abf['delta_shuf_minus_nat']}
            history.append(row)
            log(f"  step {step}: ce~{running:.3f} A_same={row['A_same']:.3f}→{row['A']} | para={row['para']:.3f} hard={row['hard']:.3f} gap={row['gap']:.3f} | ablΔ={row['ablation_delta']:+.4f}")
            model.train()
            torch.save({'model': model.state_dict(), 'step': step, 'A': Af, 'B': Bf, 'Ab': Abf}, CKPT_OUT)
    abl_ok = Abf['delta_shuf_minus_nat'] > 0.5
    a_ok = Af['mean_cos_same_last_piece'] < 0.9
    if abl_ok and a_ok:
        overall = 'SLOW_CE_TAPE_CONTEXT_YES'
    elif abl_ok:
        overall = 'SLOW_CE_ABLATION_YES_A_WEAK'
    elif a_ok:
        overall = 'SLOW_CE_A_YES_ABLATION_WEAK'
    else:
        overall = 'SLOW_CE_FLAT'
    vs181 = None
    if DEC181.exists():
        d = json.loads(DEC181.read_text(encoding='utf-8'))
        vs181 = {'gpt_ablation': d.get('final_ablation', {}).get('delta_shuf_minus_nat'), 'gpt_A_same': d.get('final_A', {}).get('mean_cos_same_last_piece'), 'tape_ablation': Abf['delta_shuf_minus_nat'], 'tape_A_same': Af['mean_cos_same_last_piece']}
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'slow_ce_dual_tape_182', 'overall': overall, 'design': 'fast/slow draw BPE tape; slow state → next piece-id CE (main); weak fast ink + light retention', 'contract_note': 'Piece-id CE unlocked on SLOW only — hybrid path, not 169 word-CE revival', 'final_A': Af, 'final_B': Bf, 'final_ablation': Abf, 'init_A': A0, 'init_B': B0, 'history': history, 'vs_181_gpt': vs181, 'next': 'If ablation approaches GPT: hybrid works. If A wipe returns: raise W_RET / write-budget. If flat: CE head underpowered vs GPT attn.'}
    write_json(DECISION, out)
    MINI.write_text('\n'.join(['# Stage182 — slow-CE on dual tape', '', f'**Overall:** `{overall}`', '', f"- A: {Af['verdict']} same={Af['mean_cos_same_last_piece']:.3f}", f"- B: para={Bf['mean_cos_paraphrase']:.3f} hard={Bf['mean_cos_hard_spelling']:.3f}", f"- ablation Δ={Abf['delta_shuf_minus_nat']:+.4f}", f'- vs GPT181: {vs181}', f"- {out['next']}", '']), encoding='utf-8')
    log(f'[182] {overall}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())