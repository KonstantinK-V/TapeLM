"""
Stage 275 — Knowing that you do not know.

Aggregation is not the interesting claim. A tally over retrieved witnesses is arithmetic, and 270
showed plain majority already does it. The claim worth making is the one neither lookup nor
majority can express: recognising that the tape does not settle the question, and declining.

So every bank carries three families and the policy is never told which it is looking at:

    clean       one witness                     answer it
    decidable   3 witnesses agree, 2 share a lie   read, tally, answer the majority
    tie         2 and 2, no majority exists      there is no right answer — abstain

Reward makes silence a real option rather than a way to dodge work:

    +1.0  correct answer      -0.3  wrong answer      0.0  abstain      -read_cost per read

A policy that always answers loses on ties. A policy that always abstains scores zero and is
caught by G_answers_when_decidable, which is a validity gate for exactly that reason: the
degenerate solution is cheap and has to be excluded before any abstention number means anything.

The teacher stays executable — it consults no gold value and no family label. A *repeat* is the
dispute signal:

    lead, second = top two frequencies among opened values
    if lead >= 2 and lead > second:   ANSWER the leader     # decidable (and keep reading while
                                                            # second==0 and unread remain, else a
                                                            # 2-0 prefix would fake a majority on ties)
    if lead >= 2 and lead == second:  STOP / abstain        # tie
    else:                             READ if unread left, else ANSWER top retrieve score  # clean

No repeats means no contradiction — answer by retrieval. A repeat with a lead means answer the
majority. A repeat without a lead is the only place to stay silent. 274's strict majority and the
first 275 margin-≥2 rule both failed on clean: unique opens never produce a lead of 2, so the
teacher abstained everywhere and the policy copied silence (ABSTAIN_INVALID).

topk defaults to 7 so five witnesses are not crowded out of the retrieve set by distractors.

The headline is not accuracy. It is risk-coverage: how much of the exam the policy chose to answer,
and how right it was on that part. A mind that answers 60% at 0.95 is better than one that answers
100% at 0.7, and only this table can say so.

  python _stage275_abstain.py --smoke
  python _stage275_abstain.py --rl-episodes 3000
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
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
import _stage274_truthfree_oracle as s274
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, TapeView
from _tape_index import context_words
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage275_abstain.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 275
FAMILIES = ('clean', 'decidable', 'tie')

def paths(frozen: bool):
    t = '_frozen' if frozen else ''
    return (RES / f'stage275_decision{t}.json', RES / f'stage275_mini{t}.md', RES / f'_stage275_log{t}.txt')
LOG_PATH = RES / '_stage275_log.txt'

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(line)

def build_tape(*, bank, tok, pad_id, device, rng, pool, lines, used, n_clean, n_dec, n_tie, n_wit, n_liars, n_dist):
    """clean / decidable (n_wit-n_liars vs n_liars) / tie (half and half, no majority)."""
    avail = [w for w in pool if w not in used and len(w) >= 5]
    rng.shuffle(avail)
    n_items = n_clean + n_dec + n_tie
    subs = [w for w in gen_fakes(set(used) | set(avail), rng, n_items + 80) if len(w) >= 5 and w not in used]
    subs = list(dict.fromkeys(subs))
    need = n_clean + 2 * (n_dec + n_tie)
    if len(subs) < n_items or len(avail) < need:
        raise RuntimeError(f'pool exhausted: subs={len(subs)} vals={len(avail)} need={need}')
    keys, vals, texts, items = ([], [], [], [])
    vi = 0

    def add(S, value, ti):
        sent = s271.WITNESS_TMPL[ti % len(s271.WITNESS_TMPL)].format(S=S, V=value)
        c = bank.ctx_fp(sent, exclude=value)
        kf = bank.fp([S])[0]
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(value)
        texts.append(sent)
        return len(vals) - 1
    si = 0

    def witness_block(kind, n_true, n_false):
        nonlocal si, vi
        S = subs[si]
        si += 1
        truth = avail[vi]
        vi += 1
        other = avail[vi]
        vi += 1
        order = [truth] * n_true + [other] * n_false
        rng.shuffle(order)
        sids = [add(S, v, j) for j, v in enumerate(order)]
        items.append({'S': S, 'truth': None if kind == 'tie' else truth, 'values': order, 'slots': sids, 'kind': kind})
        used.add(truth)
        used.add(other)
        used.add(S)
    for _ in range(n_clean):
        S = subs[si]
        si += 1
        v = avail[vi]
        vi += 1
        sid = add(S, v, 0)
        items.append({'S': S, 'truth': v, 'values': [v], 'slots': [sid], 'kind': 'clean'})
        used.add(v)
        used.add(S)
    for _ in range(n_dec):
        witness_block('decidable', n_wit - n_liars, n_liars)
    for _ in range(n_tie):
        half = n_wit // 2
        witness_block('tie', half, half)
    seen = set(vals)
    target = len(vals) + n_dist
    for ln in lines:
        if len(vals) >= target:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in seen:
                continue
            lo, hi = (max(0, m.start() - 120), min(len(ln), m.end() + 120))
            c = bank.ctx_fp(ln[lo:hi], exclude=e)
            if c is None:
                continue
            an = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != e]
            if not an:
                continue
            keys.append(F.normalize(bank.fp([an[-1]])[0] + c, dim=-1))
            vals.append(e)
            texts.append(ln[lo:hi])
            seen.add(e)
            if len(vals) >= target:
                break
    from collections import defaultdict as _dd
    postings = _dd(list)
    for cid, (v, t) in enumerate(zip(vals, texts)):
        for w in context_words(t, exclude=v):
            postings[w].append(cid)
    import math as _m
    idf = {w: 1.0 / _m.log(2.0 + len(postings[w])) for w in postings}
    return {'tape': TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id), 'texts': texts, 'items': items, 'postings': postings, 'idf': idf}

def teacher(*, cands, seen_reads, opened_values, n_reads, max_steps, max_reads, k, cand_scores=None):
    """Executable. Abstains only on a repeated tie. No gold, no family label.

    Repeat is the dispute signal. If the retrieve list itself has no duplicated values, no
    amount of reading can produce a repeat — answer by score immediately (clean). Otherwise
    open slots until: lead>=2 and lead>second with margin ≥2 (or unread exhausted) → ANSWER;
    lead>=2 and lead==second → STOP; exhausted without that → ANSWER by score / first open.
    """
    if not cands:
        return s271.ASK_Q
    sc = cand_scores or {}

    def answer_value(val: str) -> int:
        for i, c in enumerate(cands):
            if s274._cand_value(c) == val:
                return 2 + k + i
        return 2 + k

    def answer_top_score() -> int:
        best_i, best = (0, float('-inf'))
        for i, c in enumerate(cands):
            s = float(sc.get(c, 0.0))
            if s > best:
                best, best_i = (s, i)
        return 2 + k + best_i
    retrieve_counts = Counter((s274._cand_value(c) for c in cands))
    if retrieve_counts and max(retrieve_counts.values()) <= 1:
        return answer_top_score()
    cnt = Counter(opened_values)
    ranked = cnt.most_common(2)
    lead = ranked[0][1] if ranked else 0
    second = ranked[1][1] if len(ranked) > 1 else 0
    unread = [i for i, c in enumerate(cands) if c not in seen_reads]
    if lead >= 2 and lead == second:
        if not unread or n_reads >= max_reads or n_reads + 2 > max_steps:
            return 2 + 2 * k
        return 2 + unread[0]
    if lead >= 2 and lead > second and (lead - second >= 2 or not unread):
        return answer_value(ranked[0][0])
    if unread and n_reads < max_reads and (n_reads + 2 <= max_steps):
        return 2 + unread[0]
    if opened_values:
        return answer_value(opened_values[0])
    return answer_top_score()

def rollout(policy, model, char_table, tok, pack, item, pad_id, device, *, k, max_steps, max_reads, read_cost, wrong_cost, bc=False, greedy=True, teacher_only=False):
    tape, postings, idf = (pack['tape'], pack['postings'], pack['idf'])
    s274._VALUE_OF = {i: v for i, v in enumerate(tape.values)}
    qtext = s271.CUE.format(S=item['S'])
    qwords = context_words(qtext)
    transcript = qtext
    cands: list[int] = []
    last_read_words: list[str] = []
    seen_reads: set[int] = set()
    opened: list[str] = []
    losses, logps, ents, trace = ([], [], [], [])
    n_reads, answered, abstained = (0, None, False)
    for _ in range(max_steps):
        if teacher_only:
            a = teacher(cands=cands, seen_reads=seen_reads, opened_values=opened, n_reads=n_reads, max_steps=max_steps, max_reads=max_reads, k=k, cand_scores=pack.get('_sc'))
        else:
            st = s274.state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads, opened, last_read_words, n_reads, pad_id, device, k, max_steps)
            if st is None:
                break
            logits, _ = st
            if bc:
                a = teacher(cands=cands, seen_reads=seen_reads, opened_values=opened, n_reads=n_reads, max_steps=max_steps, max_reads=max_reads, k=k, cand_scores=pack.get('_sc'))
                if not torch.isfinite(logits[a]) or logits[a] < -100000000.0:
                    break
                losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([a], device=device)))
            else:
                dist = torch.distributions.Categorical(logits=logits)
                a = int(logits.argmax()) if greedy else int(dist.sample())
                logps.append(dist.log_prob(torch.tensor(a, device=device)))
                ents.append(dist.entropy())
        trace.append(s271.act_names(k)[a])
        if a in (s271.ASK_Q, s271.ASK_READ):
            words = qwords if a == s271.ASK_Q else last_read_words
            cands, sc = s271.vote(words, postings, idf, k)
            own = [c for c in cands if item['S'] in pack['texts'][c]]
            cands = own if own else cands
            pack['_sc'] = {c: sc.get(c, 0.0) for c in cands}
        elif a == 2 + 2 * k:
            abstained = True
            break
        elif a < 2 + k:
            i = a - 2
            if i >= len(cands):
                break
            slot = cands[i]
            transcript = (transcript + ' | ' + pack['texts'][slot])[-2000:]
            last_read_words = context_words(pack['texts'][slot], exclude=tape.values[slot])
            seen_reads.add(slot)
            opened.append(tape.values[slot])
            n_reads += 1
        else:
            i = a - 2 - k
            if i >= len(cands):
                break
            answered = tape.values[cands[i]]
            break
    if abstained or answered is None:
        correct, reward = (0, 0.0)
        abstained = True
    else:
        correct = int(item['truth'] is not None and answered == item['truth'])
        reward = 1.0 if correct else -wrong_cost
    reward -= read_cost * n_reads
    return {'loss': torch.stack(losses).mean() if losses else torch.zeros((), device=device), 'logps': logps, 'entropy': ents, 'reward': reward, 'correct': correct, 'abstained': abstained, 'n_reads': n_reads, 'trace': trace, 'kind': item['kind'], 'answer_is_slot': answered is None or answered in set(tape.values)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--bc-episodes', type=int, default=0)
    ap.add_argument('--rl-episodes', type=int, default=0)
    ap.add_argument('--tape-period', type=int, default=0)
    ap.add_argument('--clean', type=int, default=4)
    ap.add_argument('--decidable', type=int, default=4)
    ap.add_argument('--tie', type=int, default=4)
    ap.add_argument('--witnesses', type=int, default=5)
    ap.add_argument('--liars', type=int, default=2)
    ap.add_argument('--distractor-slots', type=int, default=0)
    ap.add_argument('--topk', type=int, default=7)
    ap.add_argument('--max-steps', type=int, default=10)
    ap.add_argument('--max-reads', type=int, default=7)
    ap.add_argument('--read-cost', type=float, default=0.02)
    ap.add_argument('--wrong-cost', type=float, default=0.3)
    ap.add_argument('--entropy-bonus', type=float, default=0.01)
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
    log(f'Stage275 abstain start {datetime.now(timezone.utc).isoformat()} device={device} bc={n_bc} rl={n_rl} clean={args.clean} dec={args.decidable} tie={args.tie} wit={args.witnesses} liars={args.liars} k={k} wrong_cost={args.wrong_cost} mode={mode}')
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
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(1500000 if args.smoke else 8000000)
    pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5)))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split('\n') if len(l.strip()) >= 60][:400 if args.smoke else 6000]
    policy = s274.Policy(2 * (model.head.in_features // 2), k, device)
    live = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': policy.parameters(), 'lr': args.lr_policy}] + ([{'params': live, 'lr': args.lr_upper}] if live else []), weight_decay=0.01)
    used: set[str] = set()
    pack, baseline, curve = (None, 0.0, [])
    common = dict(k=k, max_steps=args.max_steps, max_reads=args.max_reads, read_cost=args.read_cost, wrong_cost=args.wrong_cost)

    def new_tape(r):
        return build_tape(bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, pool=pool, lines=lines, used=used, n_clean=args.clean, n_dec=args.decidable, n_tie=args.tie, n_wit=args.witnesses, n_liars=args.liars, n_dist=n_dist)
    policy.train()
    model.train(mode != 'none')
    for ep in range(1, n_bc + 1):
        if pack is None or (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack['items'][rng.randrange(len(pack['items']))]
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device, bc=True, **common)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_bc // 8) == 0:
            curve.append({'phase': 'bc', 'episode': ep, 'loss': float(out['loss']), 'kind': out['kind'], 'trace': out['trace']})
            log(f"  bc {ep}/{n_bc} loss={float(out['loss']):.4f} [{out['kind']}] {out['trace']}")
    for ep in range(1, n_rl + 1):
        if (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack['items'][rng.randrange(len(pack['items']))]
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device, greedy=False, **common)
        if not out['logps']:
            continue
        baseline = 0.99 * baseline + 0.01 * out['reward']
        ent = torch.stack(out['entropy']).sum() if out['entropy'] else torch.zeros((), device=device)
        loss = -(out['reward'] - baseline) * torch.stack(out['logps']).sum() - args.entropy_bonus * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_rl // 8) == 0:
            curve.append({'phase': 'rl', 'episode': ep, 'baseline': baseline, 'kind': out['kind'], 'trace': out['trace']})
            log(f"  rl {ep}/{n_rl} baseline={baseline:.3f} [{out['kind']}] {out['trace']}")
    policy.eval()
    model.eval()
    arc1 = s271.arc_enc_hash(model)

    @torch.no_grad()
    def evaluate(p):
        per = {f: {'correct': [], 'abstain': [], 'reads': [], 'reward': []} for f in FAMILIES}
        tper = {f: {'correct': [], 'abstain': [], 'reward': []} for f in FAMILIES}
        slot_ok, traces = ([], [])
        for it in p['items']:
            o = rollout(policy, model, char_table, tok, p, it, pad_id, device, **common)
            t = rollout(policy, model, char_table, tok, p, it, pad_id, device, teacher_only=True, **common)
            f = it['kind']
            per[f]['correct'].append(o['correct'])
            per[f]['abstain'].append(int(o['abstained']))
            per[f]['reads'].append(o['n_reads'])
            per[f]['reward'].append(o['reward'])
            tper[f]['correct'].append(t['correct'])
            tper[f]['abstain'].append(int(t['abstained']))
            tper[f]['reward'].append(t['reward'])
            slot_ok.append(int(o['answer_is_slot']))
            traces.append({'kind': f, 'trace': o['trace'], 'correct': o['correct'], 'abstained': o['abstained']})
        m = lambda xs: float(np.mean(xs)) if xs else float('nan')
        out = {'answer_is_slot': m(slot_ok), 'traces': traces, 'reward_total': m([r for f in FAMILIES for r in per[f]['reward']]), 'teacher_reward_total': m([r for f in FAMILIES for r in tper[f]['reward']])}
        answered_correct, answered_n = (0, 0)
        for f in FAMILIES:
            cov = 1.0 - m(per[f]['abstain'])
            n_ans = sum((1 for a in per[f]['abstain'] if not a))
            acc_ans = sum(per[f]['correct']) / n_ans if n_ans else float('nan')
            answered_correct += sum(per[f]['correct'])
            answered_n += n_ans
            out[f] = {'coverage': cov, 'acc_answered': acc_ans, 'acc_all': m(per[f]['correct']), 'abstain': m(per[f]['abstain']), 'mean_reads': m(per[f]['reads']), 'reward': m(per[f]['reward']), 'teacher_abstain': m(tper[f]['abstain']), 'teacher_acc_all': m(tper[f]['correct'])}
        out['coverage_all'] = answered_n / max(1, len(p['items']))
        out['acc_answered_all'] = answered_correct / max(1, answered_n)
        return out
    train_eval = evaluate(pack)
    novel = evaluate(new_tape(random.Random(SEED + 99)))
    log(f"  NOVEL {json.dumps({kk: vv for kk, vv in novel.items() if kk != 'traces'})}")
    g_arc = arc0 == arc1
    g_slot = novel['answer_is_slot'] >= 0.99
    g_answers_when_decidable = novel['clean']['abstain'] <= 0.15 and novel['decidable']['abstain'] <= 0.25
    g_abstain_on_tie = novel['tie']['abstain'] >= 0.7
    g_teacher_abstains = novel['tie']['teacher_abstain'] >= 0.7
    g_beats_always_answer = novel['reward_total'] > 0.0
    g_novel_tape = novel['reward_total'] >= train_eval['reward_total'] - 0.15
    g_acc_when_answering = novel['acc_answered_all'] >= 0.75
    if not (g_arc and g_slot and g_answers_when_decidable):
        overall = 'ABSTAIN_INVALID'
    elif not g_teacher_abstains:
        overall = 'TEACHER_CANNOT_ABSTAIN'
    elif g_abstain_on_tie and g_acc_when_answering and g_novel_tape:
        overall = 'KNOWS_WHAT_IT_DOES_NOT_KNOW'
    elif g_abstain_on_tie or g_acc_when_answering:
        overall = 'ABSTAIN_PARTIAL'
    else:
        overall = 'ABSTAIN_NO'
    torch.save({'policy': policy.state_dict(), 'model': model.state_dict(), 'stage': 275, 'arc_enc_hash': arc1}, CKPT_OUT)
    out = {'stage': 275, 'overall': overall, 'frozen_trunk': args.frozen_trunk, 'trunk_mode': mode, 'smoke': args.smoke, 'seed': SEED, 'bc_episodes': n_bc, 'rl_episodes': n_rl, 'families': {'clean': args.clean, 'decidable': args.decidable, 'tie': args.tie}, 'witnesses': args.witnesses, 'liars': args.liars, 'topk': k, 'reward': {'correct': 1.0, 'wrong': -args.wrong_cost, 'abstain': 0.0, 'read': -args.read_cost}, 'teacher': 'ASK_Q; read while hunting repeats; ANSWER leader if lead>=2 and lead>second (after opposition seen or unread exhausted); STOP if lead>=2 and lead==second; else ANSWER top retrieve score. No gold, no family.', 'fp_version': s271.fp_version(), 'used_pool_final': len(used), 'gates': {'G_arc_enc_frozen': g_arc, 'G_answer_is_slot': g_slot, 'G_answers_when_decidable': g_answers_when_decidable, 'G_teacher_abstains_on_tie': g_teacher_abstains, 'G_abstain_on_tie': g_abstain_on_tie, 'G_acc_when_answering': g_acc_when_answering, 'G_beats_always_answer': g_beats_always_answer, 'G_novel_tape': g_novel_tape}, 'train_tape': {kk: vv for kk, vv in train_eval.items() if kk != 'traces'}, 'novel_tape': novel, 'arc_enc_hash_before': arc0, 'arc_enc_hash_after': arc1, 'curve': curve, 'note': 'Three families, never labelled to the policy: clean, decidable (3 vs 2), tie (2 vs 2, no truth exists). Reward +1 / -0.3 / 0 makes silence a real option, and G_answers_when_decidable is a validity gate because always abstaining scores zero and would otherwise look like wisdom. The teacher reads until the leader is ahead by two — 274 used a strict majority, which one read satisfies, so it answered after a single read and never aggregated. Headline is risk-coverage, not accuracy: answering 60% at 0.95 beats answering everything at 0.7, and only the table shows it.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 275 abstain\n\n**{overall}**{(' · SMOKE' if args.smoke else '')}\n\n| family (novel) | coverage | acc answered | abstain | teacher abstain | reads |\n|---|---:|---:|---:|---:|---:|\n" + ''.join((f"| {f} | {novel[f]['coverage']:.2f} | {novel[f]['acc_answered']:.2f} | {novel[f]['abstain']:.2f} | {novel[f]['teacher_abstain']:.2f} | {novel[f]['mean_reads']:.1f} |\n" for f in FAMILIES)) + f"\n- overall coverage {novel['coverage_all']:.2f} at accuracy {novel['acc_answered_all']:.2f}\n- reward: policy {novel['reward_total']:.3f} vs teacher {novel['teacher_reward_total']:.3f}\n\n## Gates\n\n" + ''.join((f'- {kk}: **{vv}**\n' for kk, vv in out['gates'].items())), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())