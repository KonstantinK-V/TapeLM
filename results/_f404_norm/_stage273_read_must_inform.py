"""
Stage 273 — Read must inform the answer (SOTE course correction).

272's confirmatory READ before ANSWER-by-agreement was a ritual: majority lived in cand
`agreement` / `max_agree`, so the policy could ignore h(transcript) and still look like it
"read". That is lookup-with-features wearing a READ — off course for a neural mind over tape.

Here the answer branch is denied the majority channel:

    READ bonus  <- [score, agreement, was_read]   (whom to open)
    ANSWER bonus <- [score, was_said, was_read]    (was_said = value appears in transcript)

Global feats drop `max_agree`. After a READ, choosing ANSWER_i must track what entered the
transcript — copy from what was read — not count siblings in the retrieve list.

Oracle (teacher may use kind; policy does not):
    clean   ASK_Q → ANSWER truth          (no read; score ranks the single witness)
    lying   ASK_Q → READ a truth witness → ANSWER that value (now was_said)

BC only by default. Same gates as 271/272.

  python _stage273_read_must_inform.py --smoke --witnesses 5 --liars 2 --read-cost 0.02
  python _stage273_read_must_inform.py --witnesses 5 --liars 2 --read-cost 0.02
"""
from __future__ import annotations
import argparse
import json
import random
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
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import hidden_and_logits
from _tape_index import context_words
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage273_read_must_inform.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 273
N_FEAT = 5

def paths(frozen: bool):
    t = '_frozen' if frozen else ''
    return (RES / f'stage273_decision{t}.json', RES / f'stage273_mini{t}.md', RES / f'_stage273_log{t}.txt')
LOG_PATH = RES / '_stage273_log.txt'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line)

class Policy(nn.Module):
    """Separate READ vs ANSWER cand scorers — ANSWER never sees agreement."""

    def __init__(self, d_hidden: int, k: int, device):
        super().__init__()
        self.k = k
        self.n_actions = 2 + 2 * k + 1
        self.f = nn.Sequential(nn.Linear(d_hidden + N_FEAT, 128), nn.GELU(), nn.Linear(128, self.n_actions)).to(device)
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        self.read_scorer = nn.Sequential(nn.Linear(d_hidden + N_FEAT + 3, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.read_scorer[-1].weight)
        nn.init.zeros_(self.read_scorer[-1].bias)
        self.ans_scorer = nn.Sequential(nn.Linear(d_hidden + N_FEAT + 3, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.ans_scorer[-1].weight)
        nn.init.zeros_(self.ans_scorer[-1].bias)

    def forward(self, h, feats, mask, read_feats=None, ans_feats=None):
        x = torch.cat([h, feats], dim=-1)
        logits = self.f(x)
        if read_feats is not None and read_feats.numel():
            n = read_feats.size(0)
            xc = torch.cat([x.unsqueeze(0).expand(n, -1), read_feats], dim=-1)
            logits = logits.index_add(0, torch.arange(2, 2 + n, device=logits.device), self.read_scorer(xc).squeeze(-1))
        if ans_feats is not None and ans_feats.numel():
            n = ans_feats.size(0)
            xc = torch.cat([x.unsqueeze(0).expand(n, -1), ans_feats], dim=-1)
            logits = logits.index_add(0, torch.arange(2 + self.k, 2 + self.k + n, device=logits.device), self.ans_scorer(xc).squeeze(-1))
        return logits.masked_fill(~mask, -1000000000.0)

def _answer_value(cands, tape, k, value: str) -> int:
    for i, c in enumerate(cands):
        if tape.values[c] == value:
            return 2 + k + i
    return 2 + k

def oracle_action(item, cands, seen_reads, tape, k, max_steps, n_reads, transcript: str) -> int:
    if not cands:
        return s271.ASK_Q
    vals = [tape.values[c] for c in cands]
    if item.get('kind') == 'clean':
        target = item['truth'] if item['truth'] in vals else vals[0]
        return _answer_value(cands, tape, k, target)
    own = set(item['slots'])
    if not any((tape.values[c] == item['truth'] and c in seen_reads for c in cands)):
        if n_reads + 1 < max_steps:
            for i, c in enumerate(cands):
                if c in own and tape.values[c] == item['truth'] and (c not in seen_reads):
                    return 2 + i
            for i, c in enumerate(cands):
                if c in own and c not in seen_reads:
                    return 2 + i
            for i, c in enumerate(cands):
                if c not in seen_reads:
                    return 2 + i
    target = item['truth'] if item['truth'] in vals else vals[0]
    return _answer_value(cands, tape, k, target)

def _state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads, last_read_words, n_reads, pad_id, device, k, max_steps):
    tape = pack['tape']
    ids = [i for i in tok.encode(transcript).ids if i != pad_id][-MAX_ARCS:]
    if not ids:
        return None
    t = torch.tensor([ids], dtype=torch.long, device=device)
    h, _ = hidden_and_logits(model, char_table, t, pad_id)
    h = h[0, -1]
    scores = [pack.get('_sc', {}).get(c, 0.0) for c in cands]
    top = max(scores) if scores else 0.0
    second = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0
    feats = torch.tensor([top, top - second, float(len(cands)) / max(1, k), float(n_reads) / max_steps, float(bool(last_read_words))], device=device, dtype=h.dtype)
    mask = torch.zeros(policy.n_actions, dtype=torch.bool, device=device)
    mask[s271.ASK_Q] = True
    mask[s271.ASK_READ] = bool(last_read_words)
    for i in range(len(cands)):
        mask[2 + i] = cands[i] not in seen_reads
        mask[2 + k + i] = True
    mask[-1] = True
    read_feats = ans_feats = None
    if cands:
        vals_c = [tape.values[c] for c in cands]
        cnt_c = Counter(vals_c)
        mx = max(scores) if scores and max(scores) > 0 else 1.0
        r_rows, a_rows = ([], [])
        for i, c in enumerate(cands):
            sc = scores[i] / mx
            agree = cnt_c[vals_c[i]] / len(cands)
            was_r = 1.0 if c in seen_reads else 0.0
            was_said = 1.0 if vals_c[i] and vals_c[i] in transcript else 0.0
            r_rows.append([sc, agree, was_r])
            a_rows.append([sc, was_said, was_r])
        read_feats = torch.tensor(r_rows, device=device, dtype=h.dtype)
        ans_feats = torch.tensor(a_rows, device=device, dtype=h.dtype)
    logits = policy(h, feats, mask, read_feats, ans_feats)
    return (logits, mask)

def _apply(a, *, item, pack, qwords, transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, k, policy):
    tape, postings, idf = (pack['tape'], pack['postings'], pack['idf'])
    if a in (s271.ASK_Q, s271.ASK_READ):
        words = qwords if a == s271.ASK_Q else last_read_words
        cands, sc = s271.vote(words, postings, idf, k)
        pack['_sc'] = sc
        return (transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, False)
    if a == policy.n_actions - 1:
        return (transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, True)
    if a < 2 + k:
        i = a - 2
        if i >= len(cands):
            return (transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, True)
        slot = cands[i]
        txt = pack['texts'][slot]
        transcript = (transcript + ' | ' + txt)[-2000:]
        last_read_words = context_words(txt, exclude=tape.values[slot])
        seen_reads = set(seen_reads) | {slot}
        return (transcript, cands, last_read_words, seen_reads, n_reads + 1, answered, reward, False)
    i = a - 2 - k
    if i >= len(cands):
        return (transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, True)
    answered = tape.values[cands[i]]
    reward = 1.0 if answered == item['truth'] else 0.0
    return (transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, True)

def bc_episode(policy, model, char_table, tok, pack, item, pad_id, device, *, k, max_steps, read_cost, answer_after_read_weight: float=5.0):
    tape = pack['tape']
    qtext = s271.CUE.format(S=item['S'])
    qwords = context_words(qtext)
    transcript = qtext
    cands: list[int] = []
    last_read_words: list[str] = []
    seen_reads: set[int] = set()
    n_reads, answered, reward = (0, None, 0.0)
    losses, weights, trace = ([], [], [])
    for _ in range(max_steps):
        st = _state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads, last_read_words, n_reads, pad_id, device, k, max_steps)
        if st is None:
            break
        logits, _ = st
        a = oracle_action(item, cands, seen_reads, tape, k, max_steps, n_reads, transcript)
        if not torch.isfinite(logits[a]) or logits[a] < -100000000.0:
            break
        is_read = 2 <= a < 2 + k
        is_ans = 2 + k <= a < 2 + 2 * k
        if is_read:
            w = 1.0
        elif is_ans and n_reads > 0:
            w = float(answer_after_read_weight)
        elif is_ans:
            w = 2.5
        else:
            w = 2.0
        losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([a], device=device)))
        weights.append(w)
        trace.append(s271.act_names(k)[a])
        transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, done = _apply(a, item=item, pack=pack, qwords=qwords, transcript=transcript, cands=cands, last_read_words=last_read_words, seen_reads=seen_reads, n_reads=n_reads, answered=answered, reward=reward, k=k, policy=policy)
        if done:
            break
    reward -= read_cost * n_reads
    if losses:
        lw = torch.tensor(weights, device=device, dtype=losses[0].dtype)
        loss = (torch.stack(losses) * lw).sum() / lw.sum()
    else:
        loss = torch.zeros((), device=device)
    return {'loss': loss, 'reward': reward, 'correct': int(answered == item['truth']), 'n_reads': n_reads, 'trace': trace, 'kind': item.get('kind'), 'answer_is_slot': answered is None or answered in set(tape.values)}

def run_episode(policy, model, char_table, tok, pack, item, pad_id, device, *, k, max_steps, read_cost, greedy=True):
    tape = pack['tape']
    qtext = s271.CUE.format(S=item['S'])
    qwords = context_words(qtext)
    transcript = qtext
    cands: list[int] = []
    last_read_words: list[str] = []
    seen_reads: set[int] = set()
    logps, ents, trace = ([], [], [])
    n_reads, answered, reward = (0, None, 0.0)
    for _ in range(max_steps):
        st = _state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads, last_read_words, n_reads, pad_id, device, k, max_steps)
        if st is None:
            break
        logits, _ = st
        dist = torch.distributions.Categorical(logits=logits)
        a = int(logits.argmax()) if greedy else int(dist.sample())
        logps.append(dist.log_prob(torch.tensor(a, device=device)))
        ents.append(dist.entropy())
        trace.append(s271.act_names(k)[a])
        transcript, cands, last_read_words, seen_reads, n_reads, answered, reward, done = _apply(a, item=item, pack=pack, qwords=qwords, transcript=transcript, cands=cands, last_read_words=last_read_words, seen_reads=seen_reads, n_reads=n_reads, answered=answered, reward=reward, k=k, policy=policy)
        if done:
            break
    reward -= read_cost * n_reads
    return {'logps': logps, 'entropy': ents, 'reward': reward, 'correct': int(answered == item['truth']), 'answered': answered, 'n_reads': n_reads, 'trace': trace, 'answer_is_slot': answered is None or answered in set(tape.values), 'kind': item.get('kind')}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--bc-episodes', type=int, default=0)
    ap.add_argument('--rl-episodes', type=int, default=0)
    ap.add_argument('--tape-period', type=int, default=0)
    ap.add_argument('--clean', type=int, default=6)
    ap.add_argument('--lying', type=int, default=6)
    ap.add_argument('--witnesses', type=int, default=5)
    ap.add_argument('--liars', type=int, default=2)
    ap.add_argument('--distractor-slots', type=int, default=0)
    ap.add_argument('--topk', type=int, default=4)
    ap.add_argument('--max-steps', type=int, default=6)
    ap.add_argument('--read-cost', type=float, default=0.02)
    ap.add_argument('--entropy-bonus', type=float, default=0.01)
    ap.add_argument('--answer-after-read-weight', type=float, default=5.0)
    ap.add_argument('--lr-policy', type=float, default=0.001)
    ap.add_argument('--lr-upper', type=float, default=3e-05)
    ap.add_argument('--frozen-trunk', action='store_true')
    args = ap.parse_args()
    global LOG_PATH
    DECISION, MINI, LOG_PATH = paths(args.frozen_trunk)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_bc = args.bc_episodes or (400 if args.smoke else 4000)
    n_rl = max(0, args.rl_episodes)
    tape_period = args.tape_period or (50 if args.smoke else 200)
    n_dist = args.distractor_slots or (150 if args.smoke else 1000)
    k = args.topk
    mode = 'none' if args.frozen_trunk else 'upper'
    log(f'Stage273 read-must-inform start {datetime.now(timezone.utc).isoformat()} device={device} bc={n_bc} rl={n_rl} tape_period={tape_period} clean={args.clean} lying={args.lying} wit={args.witnesses} liars={args.liars} k={k} mode={mode}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)['model'])
    s213.set_train_mode(model, mode)
    arc0 = s271.arc_enc_hash(model)
    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank = FpBank(model_can, stoi, device)
    log(f'  trunk={trunk_ckpt.name} mode={mode} fp_version={s271.fp_version()} arc={arc0[:12]}…')
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(1500000 if args.smoke else 8000000)
    pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5)))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split('\n') if len(l.strip()) >= 60][:400 if args.smoke else 6000]
    policy = Policy(2 * (model.head.in_features // 2), k, device)
    live = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': policy.parameters(), 'lr': args.lr_policy}] + ([{'params': live, 'lr': args.lr_upper}] if live else []), weight_decay=0.01)
    used: set[str] = set()
    pack = None
    baseline = 0.0
    curve = []

    def new_tape(r):
        return s271.build_episode_tape(bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, pool=pool, lines=lines, used=used, n_clean=args.clean, n_lying=args.lying, n_wit=args.witnesses, n_liars=args.liars, n_dist=n_dist)
    policy.train()
    model.train(mode != 'none')
    for ep in range(1, n_bc + 1):
        if pack is None or (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack['items'][rng.randrange(len(pack['items']))]
        out = bc_episode(policy, model, char_table, tok, pack, item, pad_id, device, k=k, max_steps=args.max_steps, read_cost=args.read_cost, answer_after_read_weight=args.answer_after_read_weight)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_bc // 10) == 0:
            curve.append({'phase': 'bc', 'episode': ep, 'loss': float(out['loss']), 'reward': out['reward'], 'trace': out['trace']})
            log(f"  bc {ep}/{n_bc} loss={float(out['loss']):.3f} last_trace={out['trace']} ({time.time() - t0:.0f}s)")
    for ep in range(1, n_rl + 1):
        if pack is None or (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack['items'][rng.randrange(len(pack['items']))]
        out = run_episode(policy, model, char_table, tok, pack, item, pad_id, device, k=k, max_steps=args.max_steps, read_cost=args.read_cost, greedy=False)
        if not out['logps']:
            continue
        baseline = 0.99 * baseline + 0.01 * out['reward']
        adv = out['reward'] - baseline
        ent = torch.stack(out['entropy']).sum() if out['entropy'] else torch.zeros((), device=device)
        loss = -adv * torch.stack(out['logps']).sum() - args.entropy_bonus * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_rl // 10) == 0 or ep == n_rl:
            curve.append({'phase': 'rl', 'episode': ep, 'baseline': baseline, 'reward': out['reward'], 'trace': out['trace']})
            log(f"  rl {ep}/{n_rl} baseline={baseline:.3f} last_trace={out['trace']} ({time.time() - t0:.0f}s)")
    policy.eval()
    model.eval()
    arc1 = s271.arc_enc_hash(model)

    @torch.no_grad()
    def evaluate(p):
        res = {'clean': [], 'lying': [], 'reads': [], 'reads_clean': [], 'reads_lying': [], 'slot_ok': [], 'lookup': {'clean': [], 'lying': []}, 'major': {'clean': [], 'lying': []}, 'traces': []}
        for it in p['items']:
            o = run_episode(policy, model, char_table, tok, p, it, pad_id, device, k=k, max_steps=args.max_steps, read_cost=args.read_cost, greedy=True)
            res[it['kind']].append(o['correct'])
            res['reads'].append(o['n_reads'])
            res[f"reads_{it['kind']}"].append(o['n_reads'])
            res['slot_ok'].append(int(o['answer_is_slot']))
            res['lookup'][it['kind']].append(s271.fixed_lookup(p, it, k))
            res['major'][it['kind']].append(s271.fixed_majority(p, it, k))
            res['traces'].append({'kind': it['kind'], 'trace': o['trace'], 'correct': o['correct']})
        m = lambda xs: float(np.mean(xs)) if xs else float('nan')
        return {'policy_clean': m(res['clean']), 'policy_lying': m(res['lying']), 'lookup_clean': m(res['lookup']['clean']), 'lookup_lying': m(res['lookup']['lying']), 'majority_lying': m(res['major']['lying']), 'mean_reads': m(res['reads']), 'mean_reads_clean': m(res['reads_clean']), 'mean_reads_lying': m(res['reads_lying']), 'answer_is_slot': m(res['slot_ok']), 'n': len(p['items']), 'traces': res['traces']}
    train_eval = evaluate(pack)
    pack_novel = new_tape(random.Random(SEED + 99))
    novel = evaluate(pack_novel)
    log(f'  TRAIN {json.dumps(train_eval)}')
    log(f'  NOVEL {json.dumps(novel)}')
    g_arc = arc0 == arc1
    g_slot = novel['answer_is_slot'] >= 0.99
    g_beats_lookup = novel['policy_lying'] >= novel['lookup_lying'] + 0.1
    g_beats_major = novel['policy_lying'] >= novel['majority_lying'] - 0.05
    g_clean_kept = novel['policy_clean'] >= 0.7
    g_novel = novel['policy_lying'] >= train_eval['policy_lying'] - 0.1
    g_economical = novel['mean_reads'] <= args.max_steps * 0.6
    g_read_informed = novel['mean_reads_lying'] >= 0.5 and novel['mean_reads_clean'] <= novel['mean_reads_lying'] + 0.25
    if not (g_arc and g_slot):
        overall = 'READ_INFORM_INVALID'
    elif g_beats_lookup and g_clean_kept and g_novel and g_read_informed:
        overall = 'READ_INFORM_OK'
    elif g_clean_kept or g_beats_lookup or g_read_informed:
        overall = 'READ_INFORM_PARTIAL'
    else:
        overall = 'READ_INFORM_NO'
    torch.save({'policy': policy.state_dict(), 'model': model.state_dict(), 'stage': 273, 'arc_enc_hash': arc1}, CKPT_OUT)
    out = {'stage': 273, 'overall': overall, 'frozen_trunk': args.frozen_trunk, 'trunk_mode': mode, 'smoke': args.smoke, 'seed': SEED, 'bc_episodes': n_bc, 'rl_episodes': n_rl, 'tape_period': tape_period, 'actions': s271.act_names(k), 'topk': k, 'max_steps': args.max_steps, 'read_cost': args.read_cost, 'entropy_bonus': args.entropy_bonus, 'answer_after_read_weight': args.answer_after_read_weight, 'witnesses': args.witnesses, 'liars': args.liars, 'fp_version': s271.fp_version(), 'used_pool_final': len(used), 'gates': {'G_arc_enc_frozen': g_arc, 'G_answer_is_slot': g_slot, 'G_beats_lookup': g_beats_lookup, 'G_beats_majority': g_beats_major, 'G_clean_kept': g_clean_kept, 'G_novel_tape': g_novel, 'G_reads_economical': g_economical, 'G_read_informed': g_read_informed}, 'train_tape': train_eval, 'novel_tape': novel, 'arc_enc_hash_before': arc0, 'arc_enc_hash_after': arc1, 'curve': curve, 'note': 'Course correction after 272: ANSWER cand scorer sees was_said (value in transcript) instead of agreement; max_agree removed from globals. READ is the only way to light was_said for the truth on lying items — majority-in-retrieve is not an answer feature.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 273 read-must-inform{(' (frozen trunk)' if args.frozen_trunk else '')}\n\n**{overall}** · bc={n_bc} rl={n_rl} · actions={len(s271.act_names(k))}{(' · SMOKE' if args.smoke else '')}\n\n| arm | clean | lying |\n|---|---:|---:|\n| policy (novel tape) | **{novel['policy_clean']:.3f}** | **{novel['policy_lying']:.3f}** |\n| fixed lookup | {novel['lookup_clean']:.3f} | {novel['lookup_lying']:.3f} |\n| fixed majority | — | {novel['majority_lying']:.3f} |\n\n- mean reads {novel['mean_reads']:.2f} (clean {novel['mean_reads_clean']:.2f} / lying {novel['mean_reads_lying']:.2f})\n- train tape lying {train_eval['policy_lying']:.3f} → novel {novel['policy_lying']:.3f}\n\n## Gates\n\n" + ''.join((f'- {kk}: **{vv}**\n' for kk, vv in out['gates'].items())), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())