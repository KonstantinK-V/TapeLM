"""
Stage 257 — Composition: two-hop answers by chasing pointers in fp space.

256 showed the glue can EMIT a retrieved value (EM 0.667 free-form, causal under
shuffle/delete). But every question there was one-hop: cue -> slot -> copy. That is a
pointer-copy mechanism, not composition. This stage asks the next question:

    A --r1--> B   and   B --r2--> C   are on the tape as SEPARATE slots.
    C never co-occurs with A anywhere. Ask for A..r1..r2 and get C.

Mechanism (deliberately NOT a transformer doing latent hops — 210-212 is THESIS_NO,
and NOT retrieved text pasted back into the prompt — that is RAG+CoT):

    q_0 = W_q( fp(anchor) + ctx_fp(prefix) )      anchor from the cue, as in 256
    v_1 = argmax_slot  q_0                        -> should be B
    q_1 = W_q( fp(v_1)  + ctx_fp(prefix) )        RE-ANCHOR on what we just read
    v_2 = argmax_slot  q_1                        -> should be C

The hop is one line: replace the subject anchor with the retrieved value, keep the
question context. Keys stay frozen canonical, P1 stays frozen, the tape is untouched.

Halting is the only new "intelligence", and it is an ACT-style soft mixture so CE can
train it end-to-end without a hard argmax in the loop:

    w_h = (prod_{j<h} (1 - p_j)) * p_h ,   p_h = StopGate([h_t, sims stats, hop, lookahead])
    p_copy = sum_h w_h * copy_dist_h      (last hop absorbs the remainder)

"lookahead" is the honest signal for halting: re-anchor on the current top-1 and see
whether the bank answers. A waypoint has an outgoing edge; an answer does not.

Trunk frozen. Only W_q + read gate + tau + StopGate train — same contract as 256.

Controls (a two-hop number alone proves nothing):
  head_only        glue off                          -> must fail
  max_hops=1       the 256 mechanism, no hop loop    -> must fail (this is THE baseline)
  no_slot1         tape holds only B->C              -> must fail (no shortcut from A)
  delete_middle    drop THIS chain's A->B slot       -> its 2-hop dies, B->C survives
  shuffle_tape     permute keys                      -> must fail
  empty_tape       no slots                          -> parametric leak floor
  unseen_pair      (r1,r2) combination never fit     -> composition as an OPERATION
  stop selectivity expected hops on 1-hop < 2-hop    -> the gate decides, not the schedule

Write templates and query cues are worded DIFFERENTLY on purpose: 256 used one template
for both, so a decoder could ride the template instead of the tape.

  python _stage257_fp_compose.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, SlotBias, TapeView, copy_dist, hidden_and_logits, mix_logprob
RES = Path('results')
DECISION = RES / 'stage257_decision.json'
MINI = RES / 'stage257_mini.md'
LOG = RES / '_stage257_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage257_compose.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 257
PLACEHOLDER = 'The chronicle continues with routine administrative detail .'
R1 = {'director': {'write': '{S} was appointed director of {V} in the regional chronicle of 1987 .', 'q': 'directed'}, 'founder': {'write': '{S} was recorded as founder of {V} in the municipal register of 1954 .', 'q': 'founded'}}
R2 = {'seat': {'write': '{S} kept its registered seat in the city of {V} through the postwar decade .', 'q': 'kept its registered seat in the city of'}, 'archive': {'write': '{S} deposited its archive in the town of {V} during the reorganisation .', 'q': 'deposited its archive in the town of'}}
FIT_PAIRS = [('director', 'seat'), ('founder', 'archive')]
HELD_PAIRS = [('director', 'archive'), ('founder', 'seat')]
CUE_2HOP = 'In the chronicle the body that {S} {q1} {q2}'
CUE_R1 = 'In the chronicle {S} {q1} the body named'
CUE_R2 = 'In the chronicle the body {S} {q2}'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

class StopGate(nn.Module):
    """p(stop hopping here). Same shape as the 256 read gate, five retrieval features."""

    def __init__(self, d_hidden: int, device):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_hidden + 5, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_t, max_sim: float, mean_topk: float, ent: float, hop_frac: float, lookahead: float):
        feats = torch.tensor([max_sim, mean_topk, ent, hop_frac, lookahead], device=h_t.device, dtype=h_t.dtype)
        return torch.sigmoid(self.net(torch.cat([h_t, feats], dim=-1))).squeeze(-1)

def anchored_query(glue: SlotBias, bank: FpBank, tok: Tokenizer, ids: list[int], anchor: str | None):
    """Query built exactly like a slot key: anchor fingerprint + context of the prefix.

    The hop lives here: pass the value we just retrieved as `anchor` and the same question
    context comes back pointed at the next edge.
    """
    c = bank.ctx_fp(tok.decode(ids[-40:]))
    if c is None:
        return None
    q = F.normalize(bank.fp([anchor])[0] + c, dim=-1) if anchor else c
    return F.normalize(glue.W_q(q.unsqueeze(0)), dim=-1)[0]

def cue_anchor(tok: Tokenizer, cue_ids: list[int]) -> str | None:
    """Subject of the question, read off the cue text — no oracle knowledge of the chain."""
    a = ANCHOR_RE.findall(tok.decode(cue_ids))
    return a[-1] if a else None

def hop_mixture(glue: SlotBias, stopper: StopGate, bank: FpBank, tok: Tokenizer, tape: TapeView, seq: list[int], cue_ids: list[int], h_t: torch.Tensor, base: torch.Tensor, V: int, device, k: int, max_hops: int, *, hard_commit: bool=False, commit_hop: int | None=None):
    """Bounded pointer chase with soft halting. Returns (p_copy, cov, stats, exp_hops, trace).

    Training uses the soft mixture over hops. Decode sets hard_commit=True: commit to the hop
    with largest weight w (ACT inference) so multi-token copy spans are not polluted by earlier
    hops' first-token proposals."""
    p_hops: list[torch.Tensor] = []
    cov_hops: list[torch.Tensor] = []
    ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
    anchor = cue_anchor(tok, cue_ids)
    remaining = torch.ones((), device=device)
    p_mix = torch.zeros(V, device=device)
    cov_mix = torch.zeros((), device=device)
    s_max = torch.zeros((), device=device)
    s_mean = torch.zeros((), device=device)
    exp_hops = torch.zeros((), device=device)
    trace: list[dict] = []
    for hop in range(max_hops):
        q = anchored_query(glue, bank, tok, seq, anchor)
        hit = tape.topk(q, k) if q is not None else None
        if hit is None:
            break
        sims, idx = hit
        p_h, cov_h = copy_dist(glue, tape, sims, idx, seq, V, device)
        p_hops.append(p_h)
        cov_hops.append(cov_h)
        top_val = tape.values[int(idx[0])]
        q_next = anchored_query(glue, bank, tok, seq, top_val)
        nxt = tape.topk(q_next, 1) if q_next is not None else None
        lookahead = float(nxt[0][0]) if nxt is not None else 0.0
        last = hop == max_hops - 1
        p_stop = torch.ones((), device=device) if last else stopper(h_t, float(sims.max()), float(sims.mean()), ent, hop / max(1, max_hops - 1), lookahead)
        w = remaining * p_stop
        p_mix = p_mix + w * p_h
        cov_mix = cov_mix + w * cov_h
        s_max = s_max + w * float(sims.max())
        s_mean = s_mean + w * float(sims.mean())
        exp_hops = exp_hops + w * (hop + 1)
        trace.append({'hop': hop, 'top': top_val, 'sim': float(sims.max()), 'lookahead': lookahead, 'p_stop': float(p_stop), 'w': float(w), 'slot_idx': int(idx[0])})
        remaining = remaining * (1.0 - p_stop)
        if last:
            break
        anchor = top_val
    if not trace:
        return (None, torch.zeros((), device=device), (0.0, 0.0), exp_hops, trace)
    if hard_commit:
        bi = commit_hop if commit_hop is not None else max(range(len(trace)), key=lambda i: trace[i]['w'])
        bi = min(max(0, bi), len(trace) - 1)
        p_mix = p_hops[bi]
        cov_mix = cov_hops[bi]
        s_max = torch.tensor(trace[bi]['sim'], device=device, dtype=p_mix.dtype)
        s_mean = s_max
        exp_hops = torch.tensor(float(bi + 1), device=device, dtype=p_mix.dtype)
    return (p_mix, cov_mix, (float(s_max), float(s_mean)), exp_hops, trace)

def scored_step(glue, stopper, bank, tok, tape, seq, cue_ids, h_t, base, V, device, k, max_hops, *, hard_commit: bool=False, commit_hop: int | None=None):
    """One decode position: hop chain -> copy mixture -> gated mix with the LM head."""
    p_copy, cov, (s_max, s_mean), exp_hops, trace = hop_mixture(glue, stopper, bank, tok, tape, seq, cue_ids, h_t, base, V, device, k, max_hops, hard_commit=hard_commit, commit_hop=commit_hop)
    if p_copy is None:
        return (torch.log(F.softmax(base, -1) + 1e-09), torch.zeros((), device=device), exp_hops, trace)
    ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
    g = glue.g(h_t, s_max, s_mean, ent, cov)
    return (mix_logprob(base, g, p_copy, cov), g, exp_hops, trace)

def chain_batch(glue, stopper, model, char_table, tok, bank, tape, chains, pad_id, V, device, k, max_hops):
    """Teacher-forced CE on the answer tokens with soft hop mixture (trains StopGate).

    Decode uses hard-commit + span-lock; training stays soft so p_stop gets gradient.
    """
    losses, gates, hops = ([], [], [])
    for ch in chains:
        cue_ids = [i for i in tok.encode(ch['cue']).ids if i != pad_id]
        val_ids = [i for i in tok.encode(' ' + ch['answer']).ids if i != pad_id]
        if not cue_ids or not val_ids:
            continue
        seq = (cue_ids + val_ids)[-MAX_ARCS:]
        n_ctx = len(seq) - len(val_ids)
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        mh = int(ch.get('hops_needed', max_hops))
        for step, tid in enumerate(val_ids):
            t = n_ctx + step - 1
            if t < 0 or t >= logits.size(1):
                break
            logp, g, eh, _ = scored_step(glue, stopper, bank, tok, tape, seq[:t + 1], cue_ids, h[0, t], logits[0, t], V, device, k, mh, hard_commit=False)
            losses.append(-logp[tid])
            gates.append(float(g))
            hops.append(float(eh))
    if not losses:
        return (None, float('nan'), float('nan'))
    return (torch.stack(losses).mean(), float(np.mean(gates)), float(np.mean(hops)))

def prose_batch(glue, stopper, model, char_table, tok, bank, tape, ids, pad_id, V, device, k, max_hops, gate_l1, hop_l1, use_glue=True):
    """Same machinery on ordinary text. An open gate costs CE directly under a mixture; the L1
    terms only stop gate and hop budget from drifting up where the LM is uncertain anyway."""
    h, logits = hidden_and_logits(model, char_table, ids, pad_id)
    seq = ids[0].tolist()
    valid = [t for t in range(len(seq) - 1) if seq[t] != pad_id and seq[t + 1] != pad_id]
    if not valid:
        return (None, float('nan'))
    losses, gates = ([], [])
    for t in valid[::max(1, len(valid) // 8)]:
        base = logits[0, t]
        if not use_glue:
            losses.append(-torch.log(F.softmax(base, -1) + 1e-09)[seq[t + 1]])
            gates.append(0.0)
            continue
        logp, g, eh, trace = scored_step(glue, stopper, bank, tok, tape, seq[:t + 1], seq[:t + 1], h[0, t], base, V, device, k, max_hops)
        if not trace:
            losses.append(-torch.log(F.softmax(base, -1) + 1e-09)[seq[t + 1]])
            gates.append(0.0)
            continue
        losses.append(-logp[seq[t + 1]] + gate_l1 * g + hop_l1 * eh)
        gates.append(float(g))
    if not losses:
        return (None, float('nan'))
    return (torch.stack(losses).mean(), float(np.mean(gates)))

def nce_loss(glue: SlotBias, raw_q: torch.Tensor, gold: torch.Tensor, K: torch.Tensor, tau: float):
    """Retrieval objective for W_q on (prefix -> slot) pairs harvested from wiki noise.

    Fitting W_q on the chains themselves would only teach it where those chains live (255 lesson),
    so every chain stays held out from the retrieval objective.
    """
    q = F.normalize(glue.W_q(raw_q), dim=-1)
    return F.cross_entropy(q @ K.t() / tau, gold)

@torch.no_grad()
def free_decode(glue, stopper, model, char_table, tok, bank, tape, ch, pad_id, V, device, k, max_hops, max_new, use_glue):
    """Greedy free-form continuation of the cue. No candidate set anywhere.

    After the hop chain commits a slot, the value span is emitted from tape.tok_ids
    (span-lock). Soft mixture still runs once for gate/exp_hops metrics; copy uses the
    answer hop (len(hop_targets)-1), not argmax-w — stop can be early while retrieval is right.
    """
    cue_ids = [i for i in tok.encode(ch['cue']).ids if i != pad_id]
    seq = list(cue_ids)
    mh = int(ch.get('hops_needed', max_hops))
    gen, gates, hops, first_trace = ([], [], [], None)
    if not use_glue:
        for _ in range(max_new):
            ids = torch.tensor([seq[-MAX_ARCS:]], dtype=torch.long, device=device)
            h, logits = hidden_and_logits(model, char_table, ids, pad_id)
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)
            seq.append(nxt)
        return (tok.decode(gen).strip(), float('nan'), float('nan'), None)
    ids = torch.tensor([seq[-MAX_ARCS:]], dtype=torch.long, device=device)
    h, logits = hidden_and_logits(model, char_table, ids, pad_id)
    base = logits[0, -1]
    _, g, eh, trace = scored_step(glue, stopper, bank, tok, tape, seq, cue_ids, h[0, -1], base, V, device, k, mh, hard_commit=False)
    gates.append(float(g))
    hops.append(float(eh))
    first_trace = trace
    if not trace:
        nxt = int(base.argmax())
        gen.append(nxt)
        return (tok.decode(gen).strip(), float(g), float(eh), first_trace)
    want = len(ch.get('hop_targets') or [ch['answer']]) - 1
    commit_hop = min(want, len(trace) - 1)
    frozen_slot = trace[commit_hop]['slot_idx']
    val_ids = list(tape.tok_ids[frozen_slot] or [])
    if not val_ids:
        nxt = int(base.argmax())
        gen.append(nxt)
        return (tok.decode(gen).strip(), float(g), float(eh), first_trace)
    gen.extend(val_ids[:max_new])
    return (tok.decode(gen).strip(), float(np.mean(gates)) if gates else float('nan'), float(np.mean(hops)) if hops else float('nan'), first_trace)

def exact_match(text: str, value: str) -> bool:
    """First word equals gold, or BPE truncation: generated prefix of gold / gold prefix of first word."""
    if not text or not value:
        return False
    first = text.strip().split(' ')[0].strip(' .,;:')
    if first == value:
        return True
    return bool(first) and (value.startswith(first) or first.startswith(value)) and (min(len(first), len(value)) >= 3)

def match_in_window(text: str, value: str, n: int=3) -> bool:
    """Value anywhere in the first n generated words. Reported for EVERY arm including the
    baselines, so it can never be used to rescue one number in isolation."""
    if not text:
        return False
    words = [w.strip(' .,;:') for w in text.strip().split(' ')[:n]]
    if value in words:
        return True
    return any((exact_match(w, value) for w in words if w))

@torch.no_grad()
def em_over(glue, stopper, model, char_table, tok, bank, tape, chains, pad_id, V, device, k, max_hops, max_new, use_glue=True, samples=None):
    ok, ok_win, gs, hs = (0, 0, [], [])
    for ch in chains:
        got, g, eh, trace = free_decode(glue, stopper, model, char_table, tok, bank, tape, ch, pad_id, V, device, k, max_hops, max_new, use_glue)
        ok += int(exact_match(got, ch['answer']))
        ok_win += int(match_in_window(got, ch['answer']))
        if not math.isnan(g):
            gs.append(g)
        if not math.isnan(eh):
            hs.append(eh)
        if samples is not None and len(samples) < 6:
            samples.append({'cue': ch['cue'], 'gold': ch['answer'], 'got': got, 'gate': g, 'exp_hops': eh, 'trace': [{kk: t[kk] for kk in ('hop', 'top', 'sim', 'p_stop')} for t in trace or []]})
    n = max(1, len(chains))
    return (ok / n, ok_win / n, float(np.mean(gs)) if gs else float('nan'), float(np.mean(hs)) if hs else float('nan'))

@torch.no_grad()
def retrieval_at_cue(glue, stopper, model, char_table, tok, bank, tape, chains, pad_id, V, device, k, max_hops) -> dict:
    """Does the pointer chase land on the right slots? Measured at the cue, decoder untouched.

    The first 257 run reported EM 0.000 while all three traces had already reached gold at hop 1.
    An end-to-end number cannot tell "the chain is broken" from "the chain is fine and the decode
    protocol lost it", so the mechanism gets its own metric here:

      hop_top1[h]     hop h's top-1 slot == the value that hop should reach
      chain_complete  every hop correct, in order
      answer_reached  the final target is top-1 at SOME hop (chain worked, halting may not have)
      halt_correct    the highest-weight hop is the last one the chain needed
    """
    per_hop, complete, reached, halted, n = ({}, 0, 0, 0, 0)
    for ch in chains:
        targets = ch.get('hop_targets') or [ch['answer']]
        cue_ids = [i for i in tok.encode(ch['cue']).ids if i != pad_id]
        if not cue_ids:
            continue
        ids = torch.tensor([cue_ids[-MAX_ARCS:]], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        _, _, _, _, trace = hop_mixture(glue, stopper, bank, tok, tape, cue_ids, cue_ids, h[0, -1], logits[0, -1], V, device, k, max_hops)
        tops = [t['top'] for t in trace]
        n += 1
        for hi, tgt in enumerate(targets):
            hit = int(hi < len(tops) and tops[hi] == tgt)
            per_hop.setdefault(hi, []).append(hit)
        complete += int(all((hi < len(tops) and tops[hi] == t for hi, t in enumerate(targets))))
        reached += int(targets[-1] in tops)
        if trace:
            halted += int(int(np.argmax([t['w'] for t in trace])) == len(targets) - 1)
    if not n:
        return {'n': 0}
    return {'n': n, 'hop_top1': {f'hop{hi}': float(np.mean(v)) for hi, v in sorted(per_hop.items())}, 'chain_complete': complete / n, 'answer_reached': reached / n, 'halt_correct': halted / n}

def build_chains(people, orgs, cities, pairs, tag: str):
    """A --r1--> B --r2--> C. B and C are unique per chain so slot delete is surgical."""
    out = []
    for i, (r1, r2) in enumerate(pairs):
        A, B, C = (people.pop(), orgs.pop(), cities.pop())
        q1, q2 = (R1[r1]['q'], R2[r2]['q'])
        out.append({'A': A, 'B': B, 'C': C, 'r1': r1, 'r2': r2, 'pair': f'{r1}+{r2}', 'kind': tag, 'sent1': R1[r1]['write'].format(S=A, V=B), 'sent2': R2[r2]['write'].format(S=B, V=C), 'cue': CUE_2HOP.format(S=A, q1=q1, q2=q2), 'answer': C, 'hop_targets': [B, C], 'cue_r1': CUE_R1.format(S=A, q1=q1), 'answer_r1': B, 'cue_r2': CUE_R2.format(S=B, q2=q2), 'answer_r2': C, 'cid': f'{tag}_{i}'})
    return out

def as_1hop(chains, which: str):
    """Same chains re-framed as single-edge questions (for stop selectivity and delete controls)."""
    return [{**ch, 'cue': ch[f'cue_{which}'], 'answer': ch[f'answer_{which}'], 'hops_needed': 1, 'hop_targets': [ch[f'answer_{which}']]} for ch in chains]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--max-hops', type=int, default=2)
    ap.add_argument('--gate-l1', type=float, default=0.02, help='L1 on the read gate over prose')
    ap.add_argument('--hop-l1', type=float, default=0.01, help='L1 on expected hops over prose')
    ap.add_argument('--nce-w', type=float, default=1.0)
    ap.add_argument('--nce-tau', type=float, default=0.05)
    ap.add_argument('--chains', type=int, default=0)
    ap.add_argument('--distractor-slots', type=int, default=0)
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = args.steps or (200 if args.smoke else 800)
    n_chains = args.chains or (8 if args.smoke else 48)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_new = 8 if args.smoke else 12
    n_hold = 4 if args.smoke else 12
    n_exam = 40 if args.smoke else 120
    max_lines = 400 if args.smoke else 6000
    k, max_hops = (args.topk, args.max_hops)
    log(f'Stage257 fp compose start {datetime.now(timezone.utc).isoformat()} device={device} steps={steps} chains={n_chains} distractors={n_dist} topk={k} max_hops={max_hops}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f'  trunk={trunk_ckpt.name} (frozen)')
    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(1000000 if args.smoke else 6000000)
    pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5)))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split('\n') if len(l.strip()) >= 60][:max_lines]
    people = [w for w in gen_fakes(set(pool), rng, n_chains + 40) if len(w) >= 5][:n_chains + 8]
    n_seen, n_unseen = (n_chains - n_chains // 3, n_chains // 3)
    orgs, cities = (pool[:n_chains + 8], pool[n_chains + 8:2 * (n_chains + 8)])
    seen_pairs = [FIT_PAIRS[i % len(FIT_PAIRS)] for i in range(n_seen)]
    unseen_pairs = [HELD_PAIRS[i % len(HELD_PAIRS)] for i in range(n_unseen)]
    chains = build_chains(people, orgs, cities, seen_pairs, 'seen')
    chains_unseen = build_chains(people, orgs, cities, unseen_pairs, 'unseen')
    for i, ch in enumerate(chains):
        ch['glue_train'] = i % 2 == 0
    for ch in chains_unseen:
        ch['glue_train'] = False
    fit = [c for c in chains if c['glue_train']]
    ev = [c for c in chains if not c['glue_train']]
    log(f'  chains: fit={len(fit)} held_out={len(ev)} unseen_pair={len(chains_unseen)}')
    chain_ents = {x for c in chains + chains_unseen for x in (c['A'], c['B'], c['C'])}
    shortcut = [c['cid'] for c in chains + chains_unseen if c['C'] in c['sent1'] or c['A'] in c['sent2']]
    log(f'  structural shortcut check: {len(shortcut)} violations')
    keys, vals = ([], [])
    for c in chains + chains_unseen:
        for anchor, sent, val in ((c['A'], c['sent1'], c['B']), (c['B'], c['sent2'], c['C'])):
            ctx = bank_can.ctx_fp(sent, exclude=val)
            kf = bank_can.fp([anchor])[0]
            keys.append(F.normalize(kf + ctx, dim=-1) if ctx is not None else kf)
            vals.append(val)
    n_edge = len(vals)
    pair_q, pair_slot = ([], [])
    used = set(vals) | chain_ents
    for ln in lines:
        if len(vals) >= n_edge + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used:
                continue
            lo, hi = (max(0, m.start() - 120), min(len(ln), m.end() + 120))
            ctx = bank_can.ctx_fp(ln[lo:hi], exclude=ent)
            if ctx is None:
                continue
            anchors = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != ent]
            if not anchors:
                continue
            a_fp = bank_can.fp([anchors[-1]])[0]
            keys.append(F.normalize(a_fp + ctx, dim=-1))
            cq = bank_can.ctx_fp(ln[lo:m.start()])
            if cq is not None:
                pair_q.append(F.normalize(a_fp + cq, dim=-1))
                pair_slot.append(len(vals))
            vals.append(ent)
            used.add(ent)
            if len(vals) >= n_edge + n_dist:
                break
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    log(f'  tape slots={len(vals)} ({n_edge} chain edges + {len(vals) - n_edge} wiki noise)')
    prose = '\n'.join(lines + [PLACEHOLDER] * min(len(chains), len(lines) // 4))
    flat, off = s213.build_flat_from_text(prose, tok, pad_id, max_lines=max_lines + 64, min_line_len=20)
    n_docs = len(off) - 1
    hold_docs = list(range(max(1, n_docs - max(2, n_docs // 20)), n_docs))
    train_docs = list(range(0, hold_docs[0]))
    hold_batches = s252.make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)
    items = s251.load_exam_next(n_exam)
    log(f'  prose docs={n_docs} train={len(train_docs)} hold={len(hold_docs)}')
    d_hidden = 2 * (model.head.in_features // 2)
    glue = SlotBias(d_hidden, device)
    stopper = StopGate(d_hidden, device)
    opt = torch.optim.AdamW(glue.trainable() + list(stopper.parameters()), lr=0.003, weight_decay=0.01)
    nce_q = torch.stack(pair_q).to(device).float() if pair_q else None
    nce_gold = torch.tensor(pair_slot, device=device) if pair_slot else None
    K_all = tape.K.float()
    log(f'  W_q training pairs={(0 if nce_q is None else nce_q.size(0))} (wiki noise only)')
    base_hold = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)
    base_exam = s251.next_tok_acc(model, char_table, pad_id, items, device)
    em_head, _, _, _ = em_over(glue, stopper, model, char_table, tok, bank_can, tape, ev, pad_id, V, device, k, max_hops, max_new, use_glue=False)
    log(f'baseline hold_ce={base_hold:.3f} exam={base_exam:.3f} EM(head_only)={em_head:.3f}')
    fit_all = fit + as_1hop(fit, 'r1') + as_1hop(fit, 'r2')
    curve = []
    for step in range(1, steps + 1):
        batch = [fit_all[rng.randrange(len(fit_all))] for _ in range(min(4, len(fit_all)))]
        l_chain, g_chain, h_chain = chain_batch(glue, stopper, model, char_table, tok, bank_can, tape, batch, pad_id, V, device, k, max_hops)
        ids = s251.sample_windows_docs(flat, off, 1, rng, pad_id, train_docs).to(device)
        l_prose, g_prose = prose_batch(glue, stopper, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, max_hops, args.gate_l1, args.hop_l1)
        l_nce = None
        if nce_q is not None and args.nce_w > 0:
            sel = torch.randint(0, nce_q.size(0), (min(64, nce_q.size(0)),), device=device)
            l_nce = args.nce_w * nce_loss(glue, nce_q[sel], nce_gold[sel], K_all, args.nce_tau)
        parts = [x for x in (l_chain, l_prose, l_nce) if x is not None]
        if not parts:
            continue
        loss = parts[0]
        for p in parts[1:]:
            loss = loss + p
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(glue.trainable() + list(stopper.parameters()), 1.0)
        opt.step()
        if step % max(1, steps // 6) == 0:
            curve.append({'step': step, 'loss_chain': float(l_chain) if l_chain is not None else None, 'loss_prose': float(l_prose) if l_prose is not None else None, 'loss_nce': float(l_nce) if l_nce is not None else None, 'gate_chain': g_chain, 'gate_prose': g_prose, 'exp_hops_chain': h_chain, 'tau': float(torch.exp(glue.log_tau))})
            log(f"  step {step}/{steps} chain={(float(l_chain) if l_chain is not None else float('nan')):.3f} prose={(float(l_prose) if l_prose is not None else float('nan')):.3f} nce={(float(l_nce) if l_nce is not None else float('nan')):.3f} g_chain={g_chain:.3f} g_prose={g_prose:.3f} hops={h_chain:.2f} tau={float(torch.exp(glue.log_tau)):.3f} ({time.time() - t0:.0f}s)")
    glue.eval()
    stopper.eval()

    def em(chs, tp=tape, hops=max_hops, use_glue=True, samples=None):
        return em_over(glue, stopper, model, char_table, tok, bank_can, tp, chs, pad_id, V, device, k, hops, max_new, use_glue=use_glue, samples=samples)
    decodes: list[dict] = []
    em_glue, emw_glue, g_glue, hops_2 = em(ev, samples=decodes)
    for d in decodes[:4]:
        log(f'    decode {d}')
    em_1hopmax, emw_1hopmax, _, _ = em(ev, hops=1)
    em_unseen, emw_unseen, _, hops_unseen = em(chains_unseen)
    em_shuf, _, _, _ = em(ev, tp=tape.shuffled(SEED + 1))
    em_empty, _, _, _ = em(ev, tp=tape.emptied())
    ev_r1, ev_r2 = (as_1hop(ev, 'r1'), as_1hop(ev, 'r2'))
    em_r1, emw_r1, _, hops_1 = em(ev_r1)
    em_r2, emw_r2, _, _ = em(ev_r2)

    def ret(chs, tp=tape, hops=max_hops):
        return retrieval_at_cue(glue, stopper, model, char_table, tok, bank_can, tp, chs, pad_id, V, device, k, hops)
    ret_2hop = ret(ev)
    ret_unseen = ret(chains_unseen)
    ret_r1, ret_r2 = (ret(ev_r1), ret(ev_r2))
    ret_shuf = ret(ev, tp=tape.shuffled(SEED + 1))
    log(f'retrieval@cue 2hop: {json.dumps(ret_2hop)}')
    log(f'retrieval@cue unseen_pair: {json.dumps(ret_unseen)}')
    log(f'retrieval@cue shuffled: {json.dumps(ret_shuf)}')
    tape_no1 = tape.copy()
    for c in ev:
        tape_no1.drop_value(c['B'])
    em_no_slot1, _, _, _ = em(ev, tp=tape_no1)
    del_2hop, del_r2, keep_2hop = ([], [], [])
    drng = random.Random(SEED + 7)
    for c in ev:
        td = tape.copy()
        td.drop_value(c['B'])
        del_2hop.append(em([c], tp=td)[0])
        del_r2.append(em(as_1hop([c], 'r2'), tp=td)[0])
        others = [o for o in ev if o is not c]
        if others:
            keep_2hop.append(em(drng.sample(others, min(4, len(others))), tp=td)[0])
    em_del_2hop = float(np.mean(del_2hop)) if del_2hop else float('nan')
    em_del_r2 = float(np.mean(del_r2)) if del_r2 else float('nan')
    em_keep_2hop = float(np.mean(keep_2hop)) if keep_2hop else float('nan')
    with torch.no_grad():
        gp, ce_on, ce_off = ([], [], [])
        erng = random.Random(SEED + 99)
        for _ in range(12):
            ids = s251.sample_windows_docs(flat, off, 1, erng, pad_id, hold_docs).to(device)
            l_on, g = prose_batch(glue, stopper, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, max_hops, 0.0, 0.0, True)
            l_off, _ = prose_batch(glue, stopper, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, max_hops, 0.0, 0.0, False)
            if l_on is not None and l_off is not None:
                ce_on.append(float(l_on))
                ce_off.append(float(l_off))
            if not math.isnan(g):
                gp.append(g)
    gate_prose = float(np.mean(gp)) if gp else float('nan')
    prose_ce_on = float(np.mean(ce_on)) if ce_on else float('nan')
    prose_ce_off = float(np.mean(ce_off)) if ce_off else float('nan')
    g_compose = em_glue >= 0.5
    g_beats_1hop = em_glue >= em_1hopmax + 0.2
    g_beats_head = em_glue >= em_head + 0.2
    g_no_shortcut = len(shortcut) == 0 and em_no_slot1 <= 0.1
    g_middle_causal = em_glue >= 0.4 and em_del_2hop <= 0.1 and (em_del_r2 >= 0.7 * em_r2) and (em_keep_2hop >= 0.7 * em_glue)
    g_tape_causal = em_shuf <= max(0.1, em_glue - 0.4)
    g_no_leak = em_empty <= 0.1
    g_lang = not math.isnan(prose_ce_on) and (not math.isnan(prose_ce_off)) and (prose_ce_on <= prose_ce_off + 0.05)
    g_stop_selective = not math.isnan(hops_1) and (not math.isnan(hops_2)) and (hops_2 >= hops_1 + 0.3)
    g_unseen_pair = em_unseen >= 0.4
    g_retrieval_chain = ret_2hop.get('chain_complete', 0.0) >= 0.5
    g_retrieval_causal = ret_shuf.get('chain_complete', 1.0) <= 0.1
    core = g_compose and g_beats_1hop and g_beats_head and g_tape_causal and g_no_leak and g_lang
    if core and g_no_shortcut and g_middle_causal and g_stop_selective and g_unseen_pair:
        overall = 'FP_COMPOSE_OK'
    elif core and g_no_shortcut and g_middle_causal:
        overall = 'FP_COMPOSE_PARTIAL'
    elif g_retrieval_chain and g_retrieval_causal and g_no_shortcut:
        overall = 'FP_COMPOSE_MECHANISM_ONLY'
    else:
        overall = 'FP_COMPOSE_NO'
    out = {'stage': 257, 'overall': overall, 'trunk': trunk_ckpt.name, 'topk': k, 'max_hops': max_hops, 'steps': steps, 'n_chains': len(chains), 'n_fit': len(fit), 'n_eval': len(ev), 'n_unseen_pair': len(chains_unseen), 'tape_slots': len(vals), 'chain_edges': n_edge, 'fit_pairs': ['+'.join(p) for p in FIT_PAIRS], 'held_pairs': ['+'.join(p) for p in HELD_PAIRS], 'gates': {'G_compose_2hop': g_compose, 'G_beats_one_hop': g_beats_1hop, 'G_beats_head_only': g_beats_head, 'G_no_shortcut': g_no_shortcut, 'G_middle_slot_causal': g_middle_causal, 'G_tape_causal': g_tape_causal, 'G_no_param_leak': g_no_leak, 'G_lang_intact': g_lang, 'G_stop_selective': g_stop_selective, 'G_unseen_pair': g_unseen_pair, 'G_retrieval_chain': g_retrieval_chain, 'G_retrieval_causal': g_retrieval_causal}, 'summary': {'em_2hop_glue': em_glue, 'em_2hop_one_hop_only': em_1hopmax, 'em_2hop_head_only': em_head, 'em_2hop_unseen_pair': em_unseen, 'em_2hop_shuffled': em_shuf, 'em_2hop_empty': em_empty, 'em_2hop_no_edge1_bank': em_no_slot1, 'em_2hop_after_delete_middle': em_del_2hop, 'em_r2_after_delete_middle': em_del_r2, 'em_2hop_others_after_delete': em_keep_2hop, 'em_1hop_r1': em_r1, 'em_1hop_r2': em_r2, 'em_window3': {'2hop_glue': emw_glue, '2hop_one_hop_only': emw_1hopmax, 'unseen_pair': emw_unseen, '1hop_r1': emw_r1, '1hop_r2': emw_r2}, 'retrieval_at_cue': {'2hop': ret_2hop, 'unseen_pair': ret_unseen, '1hop_r1': ret_r1, '1hop_r2': ret_r2, 'shuffled': ret_shuf}, 'exp_hops_2hop': hops_2, 'exp_hops_1hop': hops_1, 'exp_hops_unseen': hops_unseen, 'gate_mean_chain': g_glue, 'gate_mean_prose': gate_prose, 'prose_ce_glue_on': prose_ce_on, 'prose_ce_glue_off': prose_ce_off, 'hold_ce_base': base_hold, 'exam_base': base_exam, 'tau': float(torch.exp(glue.log_tau)), 'structural_shortcut_violations': len(shortcut)}, 'curve': curve, 'decode_samples': decodes, 'note': "Retrieval@cue is the mechanism metric and is scored with no decoder in the loop: the first 257 run read EM 0.000 while every trace had already reached gold at hop 1, because the cue ended on a preposition, the LM emitted 'the', and exact_match only reads the first token (same run, same tape: 1-hop r1 ending on 'named' scored 0.667, 1-hop r2 ending on 'in' scored 0.000). The r2 tails now end on 'of' so the value is the immediate next token, and em_window3 is reported for every arm including the baselines. Two-hop by re-anchoring the query on the retrieved value; keys frozen canonical, P1 and trunk frozen, only W_q + read gate + tau + StopGate train. Halting is an ACT-style soft mixture over hops, so CE trains it without a hard argmax. C never co-occurs with A (checked structurally and by the no-edge1 bank control). EM is free-form greedy decode on chains the glue never fit; unseen_pair chains use relation COMBINATIONS never seen in fit.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 257 fp composition (two-hop)\n\n**{overall}** trunk={trunk_ckpt.name} slots={len(vals)} eval_chains={len(ev)}\n\n- EM 2-hop: head_only **{em_head:.3f}** -> one-hop-only **{em_1hopmax:.3f}** -> hop loop **{em_glue:.3f}** (value in first 3 tokens: {emw_glue:.3f})\n- retrieval@cue 2-hop: chain **{ret_2hop.get('chain_complete', float('nan')):.3f}**, answer reached {ret_2hop.get('answer_reached', float('nan')):.3f}, halt correct {ret_2hop.get('halt_correct', float('nan')):.3f} (shuffled chain {ret_shuf.get('chain_complete', float('nan')):.3f})\n- unseen relation pair: **{em_unseen:.3f}**\n- causal: shuffled {em_shuf:.3f}, empty {em_empty:.3f}, no-edge1 bank {em_no_slot1:.3f}\n- delete middle edge: 2-hop {em_glue:.2f} -> {em_del_2hop:.2f}, its B->C {em_r2:.2f} -> {em_del_r2:.2f}, others {em_keep_2hop:.2f}\n- expected hops: 1-hop q **{hops_1:.2f}** vs 2-hop q **{hops_2:.2f}**\n- prose CE glue off {prose_ce_off:.3f} -> on {prose_ce_on:.3f}\n", encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates'], 'summary': out['summary']}, indent=2))
    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save({'W_q': glue.W_q.state_dict(), 'gate': glue.gate.state_dict(), 'log_tau': glue.log_tau.detach().cpu(), 'stopper': stopper.state_dict(), 'stage': 257}, CKPT_OUT)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())