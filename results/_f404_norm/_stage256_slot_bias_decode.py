"""
Stage 256 — Glue layer: slot-bias decoding (228c fp-decode x free-form head).

228c showed the fp path picks the right value 1.0 of the time, but only as a CONSTRAINED
choice over a 4-way candidate set. The head generates freely but never sees which slots were
retrieved (228a: HEAD_LEXICAL_PRIOR_ONLY, sensitivity ~0.036). This stage glues them:

  p'_t = (1 - g_t) * p_LM(t) + g_t * p_copy(t | tape, q_t)

  q_t      = W_q(ctx_fp(prefix))            queries move, tape KEYS stay frozen canonical
  p_copy   = span-aware distribution over the next token of top-k retrieved slot values
  g_t      = sigmoid(MLP([h_t, max_sim, mean_topk, entropy, coverage]))  "read the tape now?"

Mixing in PROBABILITY space, not as an additive logit bonus, is what makes the gate honest: an
additive bias has to out-shout logits of order ~10, and leaving the gate open costs nothing, so
it saturates at 1.0 and the tape stays decorative. Under a mixture, g_t=1 means "answer purely
from the tape", so on ordinary prose (where p_copy puts ~0 on the true next token) an open gate
is paid for directly in CE. A small L1 on g_t over prose keeps it from drifting back up.

Trunk is FROZEN. Only the glue trains: W_q + gate MLP + tau. Values live in the tape
only — the CE text has the fact sentence replaced by a placeholder, so the gradient toward the
right value can flow ONLY through the bias path. That keeps 244-style unlearning honest: delete
the slot and the answer dies.

Ablations that make the test strong (not just "number looks good"):
  head_only       glue off
  shuffle_tape    permute keys, breaking key<->value pairing
  slot_delete     drop the target slot, check target dies and retained survive
  empty_tape      no slots at all (parametric leak floor)
  prose gate      mean g_t on ordinary wiki windows must stay low

  python _stage256_slot_bias_decode.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
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
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
RES = Path('results')
DECISION = RES / 'stage256_decision.json'
MINI = RES / 'stage256_mini.md'
DECODE_AUDIT = RES / 'stage256_decode_miss_audit.md'
DECODE_AUDIT_JSON = RES / 'stage256_decode_miss_audit.json'
LOG = RES / '_stage256_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage256_slot_bias.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 256
FACT_TMPL = '{S} was appointed director of {V} in the regional chronicle of 1987 .'
CUE = '{S} was appointed director of'
PLACEHOLDER = 'The chronicle continues with routine administrative detail .'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)
import _inprint_glue as glue_lib
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, DEFAULT_RETRIEVE_MODE, RetrieveStats, SlotBias, TapeView, VOTES_AUTO_MIN_SLOTS, copy_dist, hidden_and_logits, mix_logprob, raw_query, full_bank_cue_summary, retrieve_topk, slot_query_words
from _retrieval_modes import vote_scores
CUE = DEFAULT_CUE
FACT_TMPL = DEFAULT_FACT_TMPL

def nce_loss(glue: SlotBias, raw_q: torch.Tensor, gold_mask: torch.Tensor, K: torch.Tensor, tau: float):
    """InfoNCE over the whole bank: pull the adapted cue query onto its slot, push off the rest.

    CE through the copy mixture only nudges retrieval second-hand (it can lower the loss by
    sharpening tau instead), so W_q needs a direct retrieval objective.
    """
    q = F.normalize(glue.W_q(raw_q), dim=-1)
    sims = q @ K.t() / tau
    pos = torch.where(gold_mask, sims, torch.full_like(sims, -10000.0)).logsumexp(dim=-1)
    return (sims.logsumexp(dim=-1) - pos).mean()

def fact_batch(glue, model, char_table, tok, bank, tape, facts, pad_id, V, device, k: int, retrieve_mode: str=DEFAULT_RETRIEVE_MODE):
    """Teacher-forced CE on the value tokens, logits corrected by the gated slot bias."""
    losses, gates = ([], [])
    for f in facts:
        cue_ids = [i for i in tok.encode(CUE.format(S=f['S'])).ids if i != pad_id]
        val_ids = [i for i in tok.encode(' ' + f['value']).ids if i != pad_id]
        if not cue_ids or not val_ids:
            continue
        seq = (cue_ids + val_ids)[-MAX_ARCS:]
        n_ctx = len(seq) - len(val_ids)
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        for step, tid in enumerate(val_ids):
            t = n_ctx + step - 1
            if t < 0 or t >= logits.size(1):
                break
            prefix = seq[:t + 1]
            base = logits[0, t]
            hit = retrieve_topk(retrieve_mode, glue, bank, tok, tape, prefix, cue_ids, k)
            if hit is None:
                logp = torch.log(F.softmax(base, -1) + 1e-09)
                g_val = torch.zeros((), device=device)
            else:
                sims, idx = hit
                ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
                p_copy, cov = copy_dist(glue, tape, sims, idx, prefix, V, device)
                g_val = glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov)
                logp = mix_logprob(base, g_val, p_copy, cov)
            losses.append(-logp[tid])
            gates.append(float(g_val))
    if not losses:
        return (None, float('nan'))
    return (torch.stack(losses).mean(), float(np.mean(gates)))

def prose_batch(glue, model, char_table, tok, bank, tape, ids: torch.Tensor, pad_id, V, device, k: int, gate_l1: float, use_glue: bool=True, retrieve_mode: str=DEFAULT_RETRIEVE_MODE):
    """Same glue on ordinary text. Under a mixture an open gate directly costs CE here; the L1 term
    only stops it from drifting up where the LM happens to be uncertain anyway."""
    h, logits = hidden_and_logits(model, char_table, ids, pad_id)
    losses, gates = ([], [])
    seq = ids[0].tolist()
    valid = [t for t in range(len(seq) - 1) if seq[t] != pad_id and seq[t + 1] != pad_id]
    if not valid:
        return (None, float('nan'))
    for t in valid[::max(1, len(valid) // 8)]:
        base = logits[0, t]
        prefix = seq[:t + 1]
        if not use_glue:
            losses.append(-torch.log(F.softmax(base, -1) + 1e-09)[seq[t + 1]])
            gates.append(0.0)
            continue
        hit = retrieve_topk(retrieve_mode, glue, bank, tok, tape, prefix, None, k)
        if hit is None:
            losses.append(-torch.log(F.softmax(base, -1) + 1e-09)[seq[t + 1]])
            gates.append(0.0)
            continue
        sims, idx = hit
        ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
        p_copy, cov = copy_dist(glue, tape, sims, idx, prefix, V, device)
        g_val = glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov)
        logp = mix_logprob(base, g_val, p_copy, cov)
        losses.append(-logp[seq[t + 1]] + gate_l1 * g_val)
        gates.append(float(g_val))
    if not losses:
        return (None, float('nan'))
    return (torch.stack(losses).mean(), float(np.mean(gates)))

@torch.no_grad()
def free_decode(glue, model, char_table, tok, bank, tape, fact, pad_id, V, device, k: int, max_new: int, use_glue: bool, retrieve_mode: str=DEFAULT_RETRIEVE_MODE, stats: RetrieveStats | None=None) -> tuple[str, float]:
    """Greedy free-form continuation of the cue; no candidate set anywhere."""
    cue_ids = [i for i in tok.encode(CUE.format(S=fact['S'])).ids if i != pad_id]
    seq = list(cue_ids)
    gen, gates = ([], [])
    for _ in range(max_new):
        ids = torch.tensor([seq[-MAX_ARCS:]], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        base = logits[0, -1]
        score = torch.log(F.softmax(base, -1) + 1e-09)
        if use_glue:
            hit = retrieve_topk(retrieve_mode, glue, bank, tok, tape, seq, cue_ids, k, stats=stats)
            if hit is not None:
                sims, idx = hit
                ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
                p_copy, cov = copy_dist(glue, tape, sims, idx, seq, V, device)
                g_val = glue.g(h[0, -1], float(sims.max()), float(sims.mean()), ent, cov)
                score = mix_logprob(base, g_val, p_copy, cov)
                gates.append(float(g_val))
        nxt = int(score.argmax())
        gen.append(nxt)
        seq.append(nxt)
    return (tok.decode(gen).strip(), float(np.mean(gates)) if gates else float('nan'))

@torch.no_grad()
def retrieval_report(glue, bank, tok, tape: TapeView, facts, pad_id, k: int) -> list[dict]:
    """At the cue (the exact state free decode starts from): where does the gold slot rank?"""
    rows = []
    for f in facts:
        cue_ids = [i for i in tok.encode(CUE.format(S=f['S'])).ids if i != pad_id]
        gold = [j for j, v in enumerate(tape.values) if v == f['value']]
        if tape.postings is not None and tape.n_live() >= VOTES_AUTO_MIN_SLOTS:
            words = slot_query_words(tok.decode(cue_ids))
            sc = vote_scores(words, tape.postings.postings, tape.postings.idf)
            gsc = max((sc.get(j, 0.0) for j in gold), default=0.0)
            rank = 1 + sum((1 for v in sc.values() if v > gsc))
            top_j = max(sc, key=sc.get) if sc else 0
            top = tape.values[top_j]
            gsim = gsc
            hit = retrieve_topk(DEFAULT_RETRIEVE_MODE, glue, bank, tok, tape, cue_ids, cue_ids, k)
            w = glue.weights(hit[0]) if hit is not None else torch.zeros(0)
        else:
            from _inprint_glue import ctx_query
            q = ctx_query(glue, bank, tok, cue_ids, anchor_ids=cue_ids)
            if q is None:
                rows.append({'S': f['S'], 'rank': None})
                continue
            sims = tape.K @ q
            gsim = float(sims[gold].max()) if gold else float('-inf')
            rank = 1 + int((sims > gsim).sum())
            top = tape.values[int(sims.argmax())]
            w = glue.weights(torch.topk(sims, min(k, sims.numel()))[0])
        rows.append({'S': f['S'], 'gold': f['value'], 'rank': rank, 'top1': top, 'gold_sim': gsim, 'w_max': float(w.max())})
    return rows

def exact_match(text: str, value: str) -> bool:
    return text.strip().split(' ')[0].strip(' .,;:') == value if text else False

def match_in_window(text: str, value: str, n: int=3) -> bool:
    """Value in first n words (257 em_window3). Catches metric misses like Sharaif Shara."""
    if not text:
        return False
    words = [w.strip(' .,;:') for w in text.strip().split(' ')[:n]]
    return value in words

def _token_rank(prob: torch.Tensor, tok_id: int) -> tuple[float, int]:
    p = float(prob[tok_id])
    return (p, 1 + int((prob > p).sum().item()))

def _rand_values(n: int, rng: random.Random, forbid: set[str]) -> list[str]:
    """Nonsense strings — not English dictionary words; BPE usually multi-token."""
    vowels, cons = ('aeiou', 'bcdfghjklmnpqrstvwxyz')
    out, seen = ([], set())
    while len(out) < n:
        L = rng.randint(6, 11)
        chars = []
        for i in range(L):
            chars.append(rng.choice(cons if i % 2 == 0 else vowels))
        w = ''.join(chars).capitalize()
        if w in forbid or w in seen or len(w) < 5:
            continue
        seen.add(w)
        out.append(w)
    return out

@torch.no_grad()
def decode_step_audit(glue, model, char_table, tok, bank, tape, fact, pad_id, V, device, k: int, max_new: int=6, retrieve_mode: str=DEFAULT_RETRIEVE_MODE) -> dict:
    """Per-step gate / p_copy / mix during greedy decode — copy has no end-of-value state."""
    cue_ids = [i for i in tok.encode(CUE.format(S=fact['S'])).ids if i != pad_id]
    val_ids = [i for i in tok.encode(' ' + fact['value']).ids if i != pad_id]
    if not cue_ids or not val_ids:
        return {'S': fact['S'], 'error': 'empty cue or value'}
    seq = list(cue_ids)
    gen, steps = ([], [])
    n_val = len(val_ids)
    copy_restart = False
    for t in range(max_new):
        ids = torch.tensor([seq[-MAX_ARCS:]], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        base = logits[0, -1]
        p_lm = F.softmax(base, dim=-1)
        lm_top = int(p_lm.argmax())
        gold_id = val_ids[t] if t < n_val else None
        hit = retrieve_topk(retrieve_mode, glue, bank, tok, tape, seq, cue_ids, k)
        if hit is None:
            nxt = lm_top
            steps.append({'t': t, 'gate': 0.0, 'p_copy_gold': None, 'copy_rank_gold': None, 'mix_top': tok.decode([nxt]), 'lm_top': tok.decode([lm_top]), 'gold_tok': tok.decode([gold_id]) if gold_id is not None else None, 'past_value_end': t >= n_val})
        else:
            sims, idx = hit
            ent = float(-(p_lm * F.log_softmax(base, -1)).sum())
            p_copy, cov = copy_dist(glue, tape, sims, idx, seq, V, device)
            g_val = float(glue.g(h[0, -1], float(sims.max()), float(sims.mean()), ent, cov))
            score = mix_logprob(base, g_val, p_copy, cov)
            mix_top = int(score.argmax())
            first_val = val_ids[0]
            p_first, rank_first = _token_rank(p_copy, first_val)
            if t >= n_val and rank_first == 1 and (g_val >= 0.35):
                copy_restart = True
            p_gold, copy_rank = (None, None)
            if gold_id is not None:
                p_gold, copy_rank = _token_rank(p_copy, gold_id)
            steps.append({'t': t, 'gate': g_val, 'cov': float(cov), 'p_copy_gold': p_gold, 'copy_rank_gold': copy_rank, 'p_copy_first_val': p_first, 'copy_rank_first_val': rank_first, 'mix_top': tok.decode([mix_top]), 'lm_top': tok.decode([lm_top]), 'gold_tok': tok.decode([gold_id]) if gold_id is not None else None, 'past_value_end': t >= n_val})
            nxt = mix_top
        gen.append(nxt)
        seq.append(nxt)
    got = tok.decode(gen).strip()
    em_ok = exact_match(got, fact['value'])
    win3 = match_in_window(got, fact['value'], 3)
    g0 = steps[0]['gate'] if steps else float('nan')
    g_mean = float(np.mean([s['gate'] for s in steps])) if steps else float('nan')
    step0 = steps[0] if steps else {}
    step0_prefix_ok = step0.get('copy_rank_gold') == 1 and step0.get('gate', 0) >= 0.35
    if em_ok:
        diagnosis = 'ok'
    elif win3 and (not em_ok):
        diagnosis = 'metric_first_word'
    elif g0 is not None and g0 < 0.35:
        diagnosis = 'gate_low'
    elif step0.get('copy_rank_gold', 999) not in (1, None) and step0.get('copy_rank_gold'):
        diagnosis = 'readout_copy'
    elif step0_prefix_ok and (not em_ok):
        diagnosis = 'copy_no_span_lock'
    else:
        diagnosis = 'other'
    return {'S': fact['S'], 'gold': fact['value'], 'n_val_tokens': n_val, 'got': got, 'em_ok': em_ok, 'em_window3': win3, 'gate_step0': g0, 'gate_mean_decode': g_mean, 'copy_restart_after_value': copy_restart, 'diagnosis': diagnosis, 'steps': steps}

@torch.no_grad()
def em_over(glue, model, char_table, tok, bank, tape, facts, pad_id, V, device, k, max_new, use_glue=True, samples=None, retrieve_mode: str=DEFAULT_RETRIEVE_MODE, stats: RetrieveStats | None=None):
    ok, gs = (0, [])
    for f in facts:
        got, g = free_decode(glue, model, char_table, tok, bank, tape, f, pad_id, V, device, k, max_new, use_glue, retrieve_mode=retrieve_mode, stats=stats)
        ok += int(exact_match(got, f['value']))
        if not math.isnan(g):
            gs.append(g)
        if samples is not None and len(samples) < 6:
            samples.append({'cue_S': f['S'], 'gold': f['value'], 'got': got, 'gate': g})
    return (ok / max(1, len(facts)), float(np.mean(gs)) if gs else float('nan'))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--gate-l1', type=float, default=0.02, help='L1 on g_t over prose steps')
    ap.add_argument('--nce-w', type=float, default=1.0, help='weight of the retrieval InfoNCE term')
    ap.add_argument('--nce-tau', type=float, default=0.05)
    ap.add_argument('--retrieve-mode', default=DEFAULT_RETRIEVE_MODE, choices=('auto', 'cosine', 'votes'), help='glue retrieval during train/eval (eval also logs all three after train)')
    ap.add_argument('--nce-pool', choices=('wiki', 'facts'), default='wiki', help='train W_q on bank-wide (prefix->slot) pairs, or overfit the fit facts (ablation)')
    ap.add_argument('--eval-only', action='store_true', help='load glue ckpt, skip training (audit retrieve paths)')
    ap.add_argument('--decode-audit', action='store_true', help='with eval-only: per-step gate/p_copy audit + miss diagnosis for held-out facts')
    ap.add_argument('--random-values', action='store_true', help='planted values = nonsense strings (control: EM should fall to single-token fraction)')
    ap.add_argument('--facts', type=int, default=0)
    ap.add_argument('--distractor-slots', type=int, default=0, help='real wiki entities added as bank noise')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = 0 if args.eval_only else args.steps or (200 if args.smoke else 800)
    n_facts = args.facts or (8 if args.smoke else 48)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_new = 4 if args.smoke else 6
    n_hold = 4 if args.smoke else 12
    n_exam = 40 if args.smoke else 120
    max_lines = 400 if args.smoke else 6000
    k = args.topk
    log(f'Stage256 slot-bias glue start {datetime.now(timezone.utc).isoformat()} device={device} steps={steps} facts={n_facts} distractors={n_dist} topk={k}')
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
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    lines = [l.strip() for l in wtext.split('\n') if len(l.strip()) >= 60][:max_lines]
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 30) if len(w) >= 5][:n_facts]
    if args.random_values:
        planted_vals = _rand_values(n_facts, rng, set(values_pool) | set(subs))
        log(f'  random-values control: {n_facts} nonsense strings (not wiki entities)')
    else:
        planted_vals = values_pool[:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = planted_vals[i]
        facts.append({'S': S, 'value': Vv, 'sent': FACT_TMPL.format(S=S, V=Vv), 'fid': f'f{i}', 'glue_train': i % 2 == 0})
    fit_facts = [f for f in facts if f['glue_train']]
    eval_facts = [f for f in facts if not f['glue_train']]
    n_single_tok = sum((1 for f in eval_facts if len([i for i in tok.encode(' ' + f['value']).ids if i != pad_id]) == 1))
    log(f'  facts: fit={len(fit_facts)} held_out={len(eval_facts)} eval_single_token_values={n_single_tok}/{len(eval_facts)}')
    if args.random_values and args.eval_only:
        log('random-values + eval-only: refuse (needs train on new tape values)')
        return 1
    keys, vals, ctxw = ([], [], [])
    pair_q, pair_slot = ([], [])
    for f in facts:
        kf = bank_can.fp([f['S']])[0]
        c = bank_can.ctx_fp(f['sent'], exclude=f['value'])
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(f['value'])
        ctxw.append(slot_query_words(f['sent'], exclude=f['value']))
    used = set(vals)
    for ln in lines:
        if len(vals) >= n_facts + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used:
                continue
            lo, hi = (max(0, m.start() - 120), min(len(ln), m.end() + 120))
            c = bank_can.ctx_fp(ln[lo:hi], exclude=ent)
            if c is None:
                continue
            anchors = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != ent]
            if not anchors:
                continue
            keys.append(F.normalize(bank_can.fp([anchors[-1]])[0] + c, dim=-1))
            ctxw.append(slot_query_words(ln[lo:hi], exclude=ent))
            cq = bank_can.ctx_fp(ln[lo:m.start()])
            if cq is not None:
                pair_q.append(F.normalize(bank_can.fp([anchors[-1]])[0] + cq, dim=-1))
                pair_slot.append(len(vals))
            vals.append(ent)
            used.add(ent)
            if len(vals) >= n_facts + n_dist:
                break
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id, ctxw=ctxw)
    log(f'  tape slots={len(vals)} ({len(facts)} planted + {len(vals) - len(facts)} wiki noise) retrieve=auto (votes if >={VOTES_AUTO_MIN_SLOTS})')
    prose = '\n'.join(lines + [PLACEHOLDER] * min(len(facts), len(lines) // 4))
    flat, off = s213.build_flat_from_text(prose, tok, pad_id, max_lines=max_lines + 64, min_line_len=20)
    n_docs = len(off) - 1
    hold_docs = list(range(max(1, n_docs - max(2, n_docs // 20)), n_docs))
    train_docs = list(range(0, hold_docs[0]))
    hold_batches = s252.make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)
    items = s251.load_exam_next(n_exam)
    log(f'  prose docs={n_docs} train={len(train_docs)} hold={len(hold_docs)}')
    glue = SlotBias(2 * (model.head.in_features // 2), device)
    if args.eval_only:
        if not CKPT_OUT.exists():
            log(f'eval-only: missing {CKPT_OUT}')
            return 1
        ck = torch.load(CKPT_OUT, map_location=device, weights_only=False)
        glue.W_q.load_state_dict(ck['W_q_glue'] if 'W_q_glue' in ck else ck['W_q'])
        glue.gate.load_state_dict(ck['gate'])
        with torch.no_grad():
            glue.log_tau.data.copy_(ck['log_tau'].to(device).reshape_as(glue.log_tau.data))
        glue.eval()
        log(f'eval-only: loaded glue from {CKPT_OUT.name}')
    opt = torch.optim.AdamW(glue.trainable(), lr=0.003, weight_decay=0.01)
    if args.nce_pool == 'facts':
        with torch.no_grad():
            pq, ps = ([], [])
            for f in fit_facts:
                cue_ids = [i for i in tok.encode(CUE.format(S=f['S'])).ids if i != pad_id]
                rq = raw_query(bank_can, tok, cue_ids, anchor_ids=cue_ids)
                if rq is None:
                    continue
                pq.append(rq)
                ps.append(vals.index(f['value']))
        pair_q, pair_slot = (pq, ps)
    nce_q = torch.stack(pair_q).to(device).float() if pair_q else None
    nce_slot = torch.tensor(pair_slot, device=device) if pair_slot else None
    K_all = tape.K.float()
    log(f'  W_q training pairs={(0 if nce_q is None else nce_q.size(0))} (pool={args.nce_pool})')
    base_hold = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)
    base_exam = s251.next_tok_acc(model, char_table, pad_id, items, device)
    em_head, _ = em_over(glue, model, char_table, tok, bank_can, tape, eval_facts, pad_id, V, device, k, max_new, use_glue=False)
    log(f'baseline hold_ce={base_hold:.3f} exam={base_exam:.3f} EM(head_only)={em_head:.3f}')
    curve = []
    retrieve_mode = args.retrieve_mode
    if steps > 0:
        for step in range(1, steps + 1):
            batch = [fit_facts[rng.randrange(len(fit_facts))] for _ in range(min(4, len(fit_facts)))]
            l_fact, g_fact = fact_batch(glue, model, char_table, tok, bank_can, tape, batch, pad_id, V, device, k, retrieve_mode)
            ids = s251.sample_windows_docs(flat, off, 1, rng, pad_id, train_docs).to(device)
            l_prose, g_prose = prose_batch(glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, args.gate_l1, retrieve_mode=retrieve_mode)
            l_nce = None
            if nce_q is not None and args.nce_w > 0:
                sel = torch.randint(0, nce_q.size(0), (min(64, nce_q.size(0)),), device=device)
                gold = F.one_hot(nce_slot[sel], K_all.size(0)).bool()
                l_nce = args.nce_w * nce_loss(glue, nce_q[sel], gold, K_all, args.nce_tau)
            parts = [x for x in (l_fact, l_prose, l_nce) if x is not None]
            if not parts:
                continue
            loss = parts[0]
            for p in parts[1:]:
                loss = loss + p
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(glue.trainable(), 1.0)
            opt.step()
            if step % max(1, steps // 6) == 0:
                curve.append({'step': step, 'loss_fact': float(l_fact) if l_fact is not None else None, 'loss_prose': float(l_prose) if l_prose is not None else None, 'loss_nce': float(l_nce) if l_nce is not None else None, 'gate_fact': g_fact, 'gate_prose': g_prose, 'tau': float(torch.exp(glue.log_tau))})
                log(f"  step {step}/{steps} fact={(float(l_fact) if l_fact is not None else float('nan')):.3f} prose={(float(l_prose) if l_prose is not None else float('nan')):.3f} nce={(float(l_nce) if l_nce is not None else float('nan')):.3f} g_fact={g_fact:.3f} g_prose={g_prose:.3f} tau={float(torch.exp(glue.log_tau)):.3f} ({time.time() - t0:.0f}s)")
    glue.eval()
    decode_audit_rows: list[dict] = []
    if args.decode_audit:
        decode_audit_rows = [decode_step_audit(glue, model, char_table, tok, bank_can, tape, f, pad_id, V, device, k, max_new, retrieve_mode) for f in eval_facts]
        misses = [r for r in decode_audit_rows if not r.get('em_ok')]
        mechanism_misses = [r for r in misses if r.get('diagnosis') != 'metric_first_word']
        diag_counts: dict[str, int] = {}
        for r in misses:
            d = r.get('diagnosis', '?')
            diag_counts[d] = diag_counts.get(d, 0) + 1
        n_restart = sum((1 for r in decode_audit_rows if r.get('copy_restart_after_value')))
        n_win3 = sum((1 for r in decode_audit_rows if r.get('em_window3')))
        log(f'decode step audit: EM-miss {len(misses)}/{len(eval_facts)} (mechanism {len(mechanism_misses)}; metric_only {len(misses) - len(mechanism_misses)}); em_window3={n_win3}/{len(eval_facts)}; copy_restart={n_restart}/{len(eval_facts)}')
        for r in misses:
            s0 = (r.get('steps') or [{}])[0]
            log(f"  MISS {r['S']} gold={r['gold']} got={r.get('got')} g0={r.get('gate_step0', float('nan')):.3f} g_mean={r.get('gate_mean_decode', float('nan')):.3f} n_tok={r.get('n_val_tokens')} restart={r.get('copy_restart_after_value')} -> {r.get('diagnosis')}")
            for st in (r.get('steps') or [])[:4]:
                log(f"    t={st['t']} g={st['gate']:.3f} past_end={st.get('past_value_end')} copy_rank_gold={st.get('copy_rank_gold')} copy_rank_1st={st.get('copy_rank_first_val')} mix={st.get('mix_top')!r} gold={st.get('gold_tok')!r}")
        audit_out = {'n_eval': len(eval_facts), 'n_miss_em': len(misses), 'n_miss_mechanism': len(mechanism_misses), 'n_em_window3': n_win3, 'n_copy_restart': n_restart, 'diagnosis_counts': diag_counts, 'rows': decode_audit_rows, 'note': 'Tape supplies ~first BPE; rest is LM spelling prior. copy_restart_after_value = after value tokens exhausted, p_copy still ranks first value token #1 (no span-lock). metric_first_word = em_window3 would pass (first-word EM too strict).'}
        audit_path = RES / ('stage256_decode_miss_audit_random_values.md' if args.random_values else 'stage256_decode_miss_audit.md')
        audit_json = RES / ('stage256_decode_miss_audit_random_values.json' if args.random_values else 'stage256_decode_miss_audit.json')
        audit_json.write_text(json.dumps(audit_out, indent=2), encoding='utf-8')
        md = ['# Stage 256 — decode audit (per-step)\n\n', f'Held-out **{len(eval_facts)}** · first-word EM miss **{len(misses)}** · mechanism miss **{len(mechanism_misses)}** · em_window3 **{n_win3}** · copy_restart **{n_restart}** · retrieve `{retrieve_mode}`' + (' · **random-values**' if args.random_values else '') + '\n\n', 'Retrieval @ cue was rank **1.0**. Gate opens on step 0; copy has **no end-of-value** (restarts at first token). Fix = **257 span-lock**.\n\n', '## Diagnosis counts\n\n']
        for d, c in sorted(diag_counts.items()):
            md.append(f'- **{d}**: {c}\n')
        md.append('\n| S | gold | got | g0 | g_mean | n_tok | restart | diagnosis |\n|---|------|-----|---:|-------:|------:|:-------:|----------|\n')
        for r in misses:
            md.append(f"| {r['S']} | {r['gold']} | {r.get('got', '')} | {r.get('gate_step0', float('nan')):.3f} | {r.get('gate_mean_decode', float('nan')):.3f} | {r.get('n_val_tokens')} | {r.get('copy_restart_after_value')} | {r.get('diagnosis')} |\n")
        oks = [r for r in decode_audit_rows if r.get('em_ok') and r.get('copy_restart_after_value')][:4]
        if oks:
            md.append('\n## OK but copy restart (same disease)\n\n')
            for r in oks:
                md.append(f"- **{r['S']}** `{r['gold']}` → `{r['got']}` (g_mean={r['gate_mean_decode']:.3f})\n")
        audit_path.write_text(''.join(md), encoding='utf-8')
        log(f'  wrote {audit_path.name} ({len(misses)} EM-miss / {len(mechanism_misses)} mechanism)')
    ret_eval = retrieval_report(glue, bank_can, tok, tape, eval_facts, pad_id, k)
    ret_fit = retrieval_report(glue, bank_can, tok, tape, fit_facts, pad_id, k)
    r_eval = [r['rank'] for r in ret_eval if r.get('rank')]
    r_fit = [r['rank'] for r in ret_fit if r.get('rank')]
    log(f'retrieval at cue: held-out top1={np.mean([r == 1 for r in r_eval]):.2f} median_rank={np.median(r_eval):.0f} | fit top1={np.mean([r == 1 for r in r_fit]):.2f}')
    for r in ret_eval[:4]:
        log(f'    {r}')
    decodes: list[dict] = []
    retrieve_decode_steps: dict[str, dict] = {}
    n_live = tape.n_live()
    log(f"retrieve audit: n_live={n_live} auto_eff={glue_lib.resolve_retrieve_mode('auto', n_live)} postings={('yes' if tape.postings else 'no')}")
    retrieve_em: dict[str, float] = {}
    em_glue, g_glue = (float('nan'), float('nan'))
    for mode in ('auto', 'cosine', 'votes'):
        st = RetrieveStats()
        em, g_mean = em_over(glue, model, char_table, tok, bank_can, tape, eval_facts, pad_id, V, device, k, max_new, retrieve_mode=mode, stats=st, samples=decodes if mode == retrieve_mode else None)
        retrieve_em[mode] = em
        retrieve_decode_steps[mode] = st.to_dict()
        if mode == retrieve_mode:
            em_glue, g_glue = (em, g_mean)
    log(f'retrieve EM (same glue, eval decode): {json.dumps(retrieve_em)}')
    log(f'retrieve decode steps: {json.dumps(retrieve_decode_steps)}')
    full_bank_cue_eval: dict[str, dict] = {}
    for mode in ('auto', 'cosine', 'votes'):
        full_bank_cue_eval[mode] = full_bank_cue_summary(mode, glue, bank_can, tok, tape, eval_facts, pad_id, cue_tmpl=CUE)
    log(f'full_bank at cue (held-out): {json.dumps(full_bank_cue_eval)}')
    for d in decodes[:4]:
        log(f'    decode {d}')
    em_shuf, _ = em_over(glue, model, char_table, tok, bank_can, tape.shuffled(SEED + 1), eval_facts, pad_id, V, device, k, max_new)
    em_empty, _ = em_over(glue, model, char_table, tok, bank_can, tape.emptied(), eval_facts, pad_id, V, device, k, max_new)
    per_fact_after, retained = ([], [])
    for f in eval_facts:
        tape_del = tape.copy()
        tape_del.drop_value(f['value'])
        em_f, _ = em_over(glue, model, char_table, tok, bank_can, tape_del, [f], pad_id, V, device, k, max_new)
        per_fact_after.append(em_f)
        others = [o for o in eval_facts if o is not f]
        if others:
            em_o, _ = em_over(glue, model, char_table, tok, bank_can, tape_del, others, pad_id, V, device, k, max_new)
            retained.append(em_o)
    em_tgt_before = em_glue
    em_tgt_after = float(np.mean(per_fact_after)) if per_fact_after else float('nan')
    em_ret_after = float(np.mean(retained)) if retained else float('nan')
    with torch.no_grad():
        gp, ce_on, ce_off = ([], [], [])
        erng = random.Random(SEED + 99)
        for _ in range(12):
            ids = s251.sample_windows_docs(flat, off, 1, erng, pad_id, hold_docs).to(device)
            l_on, g = prose_batch(glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, 0.0, True)
            l_off, _ = prose_batch(glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, 0.0, False)
            if l_on is not None and l_off is not None:
                ce_on.append(float(l_on))
                ce_off.append(float(l_off))
            if not math.isnan(g):
                gp.append(g)
    gate_prose = float(np.mean(gp)) if gp else float('nan')
    prose_ce_glue = float(np.mean(ce_on)) if ce_on else float('nan')
    prose_ce_plain = float(np.mean(ce_off)) if ce_off else float('nan')
    hold_after = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)
    g_freeform = em_glue >= 0.6
    g_beats_head = em_glue >= em_head + 0.2
    g_tape_causal = em_shuf <= max(0.1, em_glue - 0.4)
    g_slot_delete = em_tgt_before >= 0.4 and em_tgt_after <= 0.1 and (em_ret_after >= 0.7 * em_glue)
    g_no_leak = em_empty <= 0.1
    g_lang_intact = not math.isnan(prose_ce_glue) and (not math.isnan(prose_ce_plain)) and (prose_ce_glue <= prose_ce_plain + 0.05)
    g_gate_selective = not math.isnan(g_glue) and (not math.isnan(gate_prose)) and (g_glue >= gate_prose + 0.2)
    core = g_freeform and g_beats_head and g_tape_causal and g_no_leak and g_lang_intact
    if core and g_slot_delete and g_gate_selective:
        overall = 'SLOT_BIAS_GLUE_OK'
    elif g_beats_head and g_tape_causal and g_no_leak and g_lang_intact:
        overall = 'SLOT_BIAS_GLUE_PARTIAL'
    else:
        overall = 'SLOT_BIAS_GLUE_NO'
    out = {'stage': 256, 'overall': overall, 'trunk': trunk_ckpt.name, 'topk': k, 'steps': steps, 'n_facts': len(facts), 'n_fit': len(fit_facts), 'n_eval': len(eval_facts), 'tape_slots': len(vals), 'random_values': bool(args.random_values), 'gates': {'G_freeform_value': g_freeform, 'G_beats_head_only': g_beats_head, 'G_tape_causal': g_tape_causal, 'G_slot_delete_clean': g_slot_delete, 'G_no_param_leak': g_no_leak, 'G_lang_intact': g_lang_intact, 'G_gate_selective': g_gate_selective}, 'summary': {'em_head_only': em_head, 'em_glue': em_glue, 'em_shuffled_tape': em_shuf, 'em_empty_tape': em_empty, 'em_target_before_delete': em_tgt_before, 'em_target_after_delete': em_tgt_after, 'em_retained_after_delete': em_ret_after, 'gate_mean_fact': g_glue, 'gate_mean_prose': gate_prose, 'prose_ce_glue_on': prose_ce_glue, 'prose_ce_glue_off': prose_ce_plain, 'hold_ce_base': base_hold, 'hold_ce_after': hold_after, 'exam_base': base_exam, 'tau': float(torch.exp(glue.log_tau)), 'gate_l1': args.gate_l1, 'retrieve_mode_train': retrieve_mode, 'retrieve_em_eval': retrieve_em, 'retrieve_decode_steps': retrieve_decode_steps, 'full_bank_cue_eval': full_bank_cue_eval, 'fp_version': CKPT_P1.name, 'eval_single_token_values': n_single_tok, 'eval_single_token_frac': n_single_tok / max(1, len(eval_facts))}, 'curve': curve, 'retrieval_at_cue': {'held_out_top1': float(np.mean([r == 1 for r in r_eval])) if r_eval else None, 'held_out_median_rank': float(np.median(r_eval)) if r_eval else None, 'fit_top1': float(np.mean([r == 1 for r in r_fit])) if r_fit else None, 'rows': ret_eval[:8]}, 'decode_samples': decodes, 'note': "Glue only: trunk frozen, W_q + gate MLP + tau trained. Copy mixture p' = (1-g)p_LM + g*p_copy, so an open gate is paid for in CE. Values exist in the tape only, so CE toward the right value can flow only through the bias path. EM is free-form greedy decode (no candidate set); scored on facts the glue never fit.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION_PATH = RES / ('stage256_decision_random_values.json' if args.random_values else 'stage256_decision.json')
    DECISION_PATH.write_text(json.dumps(out, indent=2), encoding='utf-8')
    mini_path = RES / ('stage256_mini_random_values.md' if args.random_values else 'stage256_mini.md')
    mini_path.write_text(f'# Stage 256 slot-bias glue\n\n**{overall}** trunk={trunk_ckpt.name} slots={len(vals)} eval_facts={len(eval_facts)}\n\n- EM free-form: head_only **{em_head:.3f}** -> glue **{em_glue:.3f}**\n- causal: shuffled **{em_shuf:.3f}**, empty **{em_empty:.3f}**\n- slot delete: target {em_tgt_before:.2f} -> {em_tgt_after:.2f}, retained {em_ret_after:.2f}\n- gate: fact **{g_glue:.3f}** vs prose **{gate_prose:.3f}**\n- prose CE glue off {prose_ce_plain:.3f} -> on {prose_ce_glue:.3f} (hold CE {base_hold:.3f})\n', encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates'], 'summary': out['summary']}, indent=2))
    if not args.smoke and (not args.random_values):
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save({'W_q': glue.W_q.state_dict(), 'W_q_glue': glue.W_q.state_dict(), 'gate': glue.gate.state_dict(), 'log_tau': glue.log_tau.detach().cpu(), 'stage': 256}, CKPT_OUT)
    elif args.random_values and (not args.smoke):
        alt = Path('checkpoints/stage256_slot_bias_random_values.pt')
        alt.parent.mkdir(exist_ok=True)
        torch.save({'W_q': glue.W_q.state_dict(), 'W_q_glue': glue.W_q.state_dict(), 'gate': glue.gate.state_dict(), 'log_tau': glue.log_tau.detach().cpu(), 'stage': 256, 'random_values': True}, alt)
        log(f'  saved random-values glue -> {alt.name} (did not overwrite {CKPT_OUT.name})')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())