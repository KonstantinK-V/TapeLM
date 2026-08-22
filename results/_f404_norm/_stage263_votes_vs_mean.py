"""
Stage 263 — Was the context mean a bottleneck everywhere, or only in 261?

`ctx_fp(text) = normalize(mean over up to 40 word fingerprints)` was decided in 194 and never
questioned again. Every slot key since is `norm(fp(anchor) + ctx_fp(...))` - 194 through 198,
255, 256, 257, 258, 260. Seventy stages on one unexamined choice.

261f found the first exam where the ceiling of that mean is visible: one posting per content
word, zero trained parameters, scored 20-way 0.601 against the mean's 0.226 and open top1 0.246
against 0.034. A mean of forty unit vectors is nearly a constant direction; it cannot
discriminate. On the other stages the candidate set is closed - four ways, or eight relations of
one subject - so the mean is good enough and the ceiling never shows.

That is not evidence there is no ceiling. This runs 256's exam, which has a decode and a
published number to beat (em_glue 0.667), with ONE thing changed:

    cosine arm   retrieve by cos(q, key)              the mean, as shipped
    votes arm    retrieve by word postings + idf      one slot per word, nothing fitted

Everything else is identical and shared: same facts, same tape, same glue, same copy mixture,
same gate, same training loop, same seeds. Only the retrieval step differs, so a difference in
EM is a difference in retrieval and nothing else.

  python _stage263_votes_vs_mean.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, DEFAULT_CUE, DEFAULT_FACT_TMPL, SlotBias, SlotPostings, TapeView, copy_dist, full_bank_cue_summary, hidden_and_logits, mix_logprob
from _stage261f_word_votes import content
from _retrieval_modes import CASCADE_POOL, FUSION_LAM, cascade_order, cosine_scores, ctx_vector, fusion_scores, vote_scores
RES = Path('results')
DECISION = RES / 'stage263_decision.json'
MINI = RES / 'stage263_mini.md'
LOG = RES / '_stage263_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED_256 = 256

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def build(bank: FpBank, tok: Tokenizer, pad_id: int, device, smoke: bool):
    """256's data build, kept verbatim, plus the write-context words each slot came from."""
    rng = random.Random(SEED_256)
    n_facts = 8 if smoke else 48
    n_dist = 150 if smoke else 1200
    max_lines = 400 if smoke else 6000
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(1000000 if smoke else 6000000)
    pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5)))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split('\n') if len(l.strip()) >= 60][:max_lines]
    subs = [w for w in gen_fakes(set(pool), rng, n_facts + 30) if len(w) >= 5][:n_facts]
    facts = [{'S': S, 'value': pool[i], 'sent': DEFAULT_FACT_TMPL.format(S=S, V=pool[i]), 'fid': f'f{i}', 'glue_train': i % 2 == 0} for i, S in enumerate(subs)]
    keys, vals, ctxw, anchors = ([], [], [], [])
    pair_q, pair_slot = ([], [])
    for f_ in facts:
        kf = bank.fp([f_['S']])[0]
        ws = content(f_['sent'], exclude=f_['value'])
        c = bank.ctx_fp(f_['sent'], exclude=f_['value'])
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        anchors.append(kf)
        vals.append(f_['value'])
        ctxw.append(ws)
    used = set(vals)
    for ln in lines:
        if len(vals) >= n_facts + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = (max(0, m.start() - 120), min(len(ln), m.end() + 120))
            c = bank.ctx_fp(ln[lo:hi], exclude=e)
            if c is None:
                continue
            an = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != e]
            if not an:
                continue
            a_fp = bank.fp([an[-1]])[0]
            ws = content(ln[lo:hi], exclude=e)
            keys.append(F.normalize(a_fp + c, dim=-1))
            anchors.append(a_fp)
            cq = bank.ctx_fp(ln[lo:m.start()])
            if cq is not None:
                pair_q.append(F.normalize(a_fp + cq, dim=-1))
                pair_slot.append(len(vals))
            vals.append(e)
            ctxw.append(ws)
            used.add(e)
            if len(vals) >= n_facts + n_dist:
                break
    postings: dict[str, list[int]] = defaultdict(list)
    for j, ws in enumerate(ctxw):
        for w in ws:
            postings[w].append(j)
    idf = {w: 1.0 / math.log(2.0 + len(v)) for w, v in postings.items()}
    idf_keys = []
    for anc, ws in zip(anchors, ctxw):
        ci = ctx_vector(bank, ws, idf)
        idf_keys.append(F.normalize(anc + ci, dim=-1) if ci is not None else anc)
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id, ctxw=ctxw)
    K_idf = torch.stack(idf_keys, 0).to(device)
    Q = torch.stack(pair_q).to(device).float() if pair_q else None
    G = torch.tensor(pair_slot, device=device) if pair_slot else None
    return (facts, tape, ctxw, Q, G, K_idf)

class Votes(SlotPostings):
    """Alias for stage 263 ablation arms."""
    pass

def _adapted_q(glue, bank, tok, seq, cue_ids, idf: dict[str, float] | None):
    words = content(tok.decode(seq[-40:]))
    c = ctx_vector(bank, words, idf)
    if c is None:
        return None
    an = ANCHOR_RE.findall(tok.decode(cue_ids))
    q = F.normalize(bank.fp([an[-1]])[0] + c, dim=-1) if an else c
    return F.normalize(glue.W_q(q.unsqueeze(0)), dim=-1)[0]

def _topk_indices(sc: dict[int, float], k: int, device):
    idx = sorted(sc, key=lambda j: -sc[j])[:k]
    v = torch.tensor([sc[j] for j in idx], dtype=torch.float32, device=device)
    v = v / v.max().clamp_min(1e-06)
    return (v, torch.tensor(idx, dtype=torch.long, device=device))

def retrieve(mode, glue, bank, tok, tape, votes, seq, cue_ids, k, K_idf):
    """Retrieval mode: cosine | idf_mean | votes | cascade | fusion."""
    words = content(tok.decode(seq[-60:]))
    device = tape.K.device
    n = len(tape.values)
    if mode == 'votes':
        return votes.topk(words, k, tape.alive)
    q = _adapted_q(glue, bank, tok, seq, cue_ids, None)
    if q is None:
        return None
    if mode == 'cosine':
        return tape.topk(q, k)
    if mode == 'idf_mean':
        q_i = _adapted_q(glue, bank, tok, seq, cue_ids, votes.idf)
        if q_i is None:
            return None
        sims = (K_idf.float() @ q_i).masked_fill(~tape.alive, -10000.0)
        kk = min(k, int(tape.alive.sum()))
        v, idx = torch.topk(sims, kk)
        return (v, idx)
    if mode == 'cascade':
        vsc = vote_scores(words, votes.postings, votes.idf)
        order = cascade_order(vsc, q, tape.K.float(), n, CASCADE_POOL)
        pick = order[:k]
        sims = torch.tensor([float(tape.K[j] @ q) for j in pick], dtype=torch.float32, device=device)
        sims = sims / sims.max().clamp_min(1e-06)
        return (sims, torch.tensor(pick, dtype=torch.long, device=device))
    if mode == 'fusion':
        vsc = vote_scores(words, votes.postings, votes.idf)
        cos_sc = cosine_scores(q, tape.K.float())
        fsc = fusion_scores(vsc, cos_sc, FUSION_LAM)
        return _topk_indices(fsc, k, device)
    raise ValueError(mode)

def step_logp(mode, glue, model, char_table, tok, bank, tape, votes, seq, cue_ids, h_t, base, V, device, k, K_idf):
    got = retrieve(mode, glue, bank, tok, tape, votes, seq, cue_ids, k, K_idf)
    if got is None:
        return (torch.log(F.softmax(base, -1) + 1e-09), torch.zeros((), device=device))
    sims, idx = got
    ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
    p_copy, cov = copy_dist(glue, tape, sims, idx, seq, V, device)
    g = glue.g(h_t, float(sims.max()), float(sims.mean()), ent, cov)
    return (mix_logprob(base, g, p_copy, cov), g)

@torch.no_grad()
def decode(mode, glue, model, char_table, tok, bank, tape, votes, fact, pad_id, V, device, k, max_new, K_idf, use_glue=True):
    cue = [i for i in tok.encode(DEFAULT_CUE.format(S=fact['S'])).ids if i != pad_id]
    seq, gen = (list(cue), [])
    for _ in range(max_new):
        ids = torch.tensor([seq[-MAX_ARCS:]], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        base = logits[0, -1]
        if use_glue:
            logp, _ = step_logp(mode, glue, model, char_table, tok, bank, tape, votes, seq, cue, h[0, -1], base, V, device, k, K_idf)
        else:
            logp = torch.log(F.softmax(base, -1) + 1e-09)
        nxt = int(logp.argmax())
        gen.append(nxt)
        seq.append(nxt)
    return tok.decode(gen).strip()

def em(mode, glue, model, char_table, tok, bank, tape, votes, facts, pad_id, V, device, k, max_new, K_idf, use_glue=True):
    ok = 0
    for f_ in facts:
        got = decode(mode, glue, model, char_table, tok, bank, tape, votes, f_, pad_id, V, device, k, max_new, K_idf, use_glue)
        ok += int(bool(got) and got.strip().split(' ')[0].strip(' .,;:') == f_['value'])
    return ok / max(1, len(facts))

def run_single_mode(mode: str, device, smoke: bool, steps: int, topk: int, *, log_fn=log) -> dict:
    """Train glue once for a retrieval mode; return EM metrics."""
    torch.manual_seed(SEED_256)
    t0 = time.time()
    max_new = 4 if smoke else 6
    k = topk
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    trunk = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    can = SelfModelXL(n_char, V).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)
    facts, tape, ctxw, Q, G, K_idf = build(bank, tok, pad_id, device, smoke)
    votes = Votes(ctxw, device)
    fit = [f for f in facts if f['glue_train']]
    ev = [f for f in facts if not f['glue_train']]
    rng = random.Random(SEED_256 + 11)
    glue = SlotBias(2 * (model.head.in_features // 2), device)
    opt = torch.optim.AdamW(glue.trainable(), lr=0.003, weight_decay=0.01)
    for step in range(1, steps + 1):
        f_ = fit[rng.randrange(len(fit))]
        cue = [i for i in tok.encode(DEFAULT_CUE.format(S=f_['S'])).ids if i != pad_id]
        val = [i for i in tok.encode(' ' + f_['value']).ids if i != pad_id]
        if not cue or not val:
            continue
        seq = (cue + val)[-MAX_ARCS:]
        n_ctx = len(seq) - len(val)
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        losses = []
        for si, tid in enumerate(val):
            t = n_ctx + si - 1
            if t < 0 or t >= logits.size(1):
                break
            logp, _ = step_logp(mode, glue, model, char_table, tok, bank, tape, votes, seq[:t + 1], cue, h[0, t], logits[0, t], V, device, k, K_idf)
            losses.append(-logp[tid])
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        if mode != 'votes' and Q is not None:
            sel = torch.randint(0, Q.size(0), (min(32, Q.size(0)),), device=device)
            qn = F.normalize(glue.W_q(Q[sel]), dim=-1)
            loss = loss + F.cross_entropy(qn @ tape.K.float().t() / 0.05, G[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(glue.trainable(), 1.0)
        opt.step()
    glue.eval()
    e = em(mode, glue, model, char_table, tok, bank, tape, votes, ev, pad_id, V, device, k, max_new, K_idf)
    del_ = []
    for f_ in ev:
        td = tape.copy()
        td.drop_value(f_['value'])
        vv = votes
        if mode == 'votes':
            vv = Votes([[] if tape.values[j] == f_['value'] else ctxw[j] for j in range(len(ctxw))], device)
        del_.append(em(mode, glue, model, char_table, tok, bank, td, vv, [f_], pad_id, V, device, k, max_new, K_idf))
    res = {'em': e, 'em_after_delete': float(np.mean(del_)) if del_ else float('nan'), 'wall_s': time.time() - t0}
    log_fn(f'[263/{mode}] ' + json.dumps(res))
    return res

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--topk', type=int, default=8)
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED_256)
    t0 = time.time()
    steps = args.steps or (200 if args.smoke else 800)
    max_new = 4 if args.smoke else 6
    k = args.topk
    log(f'Stage263 votes vs mean start {datetime.now(timezone.utc).isoformat()} device={device}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    trunk = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    can = SelfModelXL(n_char, V).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)
    facts, tape, ctxw, Q, G, K_idf = build(bank, tok, pad_id, device, args.smoke)
    votes = Votes(ctxw, device)
    fit = [f for f in facts if f['glue_train']]
    ev = [f for f in facts if not f['glue_train']]
    log(f'  trunk={trunk.name} slots={len(tape.values)} fit={len(fit)} eval={len(ev)} | vocab={len(votes.postings)} postings={sum((len(v) for v in votes.postings.values()))} | W_q pairs={(0 if Q is None else Q.size(0))}')

    def run(mode):
        torch.manual_seed(SEED_256)
        rng = random.Random(SEED_256 + 11)
        glue = SlotBias(2 * (model.head.in_features // 2), device)
        opt = torch.optim.AdamW(glue.trainable(), lr=0.003, weight_decay=0.01)
        for step in range(1, steps + 1):
            f_ = fit[rng.randrange(len(fit))]
            cue = [i for i in tok.encode(DEFAULT_CUE.format(S=f_['S'])).ids if i != pad_id]
            val = [i for i in tok.encode(' ' + f_['value']).ids if i != pad_id]
            if not cue or not val:
                continue
            seq = (cue + val)[-MAX_ARCS:]
            n_ctx = len(seq) - len(val)
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            h, logits = hidden_and_logits(model, char_table, ids, pad_id)
            losses = []
            for si, tid in enumerate(val):
                t = n_ctx + si - 1
                if t < 0 or t >= logits.size(1):
                    break
                logp, _ = step_logp(mode, glue, model, char_table, tok, bank, tape, votes, seq[:t + 1], cue, h[0, t], logits[0, t], V, device, k, K_idf)
                losses.append(-logp[tid])
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if mode != 'votes' and Q is not None:
                sel = torch.randint(0, Q.size(0), (min(32, Q.size(0)),), device=device)
                qn = F.normalize(glue.W_q(Q[sel]), dim=-1)
                loss = loss + F.cross_entropy(qn @ tape.K.float().t() / 0.05, G[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(glue.trainable(), 1.0)
            opt.step()
            if step % max(1, steps // 4) == 0:
                log(f'  [{mode}] step {step}/{steps} loss={float(loss):.3f} ({time.time() - t0:.0f}s)')
        glue.eval()
        fb = full_bank_cue_summary(mode, glue, bank, tok, tape, ev, pad_id, cue_tmpl=DEFAULT_CUE)
        e = em(mode, glue, model, char_table, tok, bank, tape, votes, ev, pad_id, V, device, k, max_new, K_idf)
        e_shuf = em(mode, glue, model, char_table, tok, bank, tape.shuffled(SEED_256 + 1), votes, ev, pad_id, V, device, k, max_new, K_idf) if mode == 'cosine' else None
        del_ = []
        for f_ in ev:
            td = tape.copy()
            td.drop_value(f_['value'])
            vv = votes
            if mode == 'votes':
                vv = Votes([[] if tape.values[j] == f_['value'] else ctxw[j] for j in range(len(ctxw))], device)
            del_.append(em(mode, glue, model, char_table, tok, bank, td, vv, [f_], pad_id, V, device, k, max_new, K_idf))
        res = {'em': e, 'em_after_delete': float(np.mean(del_)) if del_ else float('nan'), **fb}
        if e_shuf is not None:
            res['em_shuffled_keys'] = e_shuf
        log(f'[{mode}] ' + json.dumps(res))
        return res
    r_cos = run('cosine')
    r_vote = run('votes')
    g_repro = r_cos['em'] >= 0.5
    g_delta = r_vote['em'] - r_cos['em']
    g_votes_causal = r_vote['em_after_delete'] <= 0.1
    g_cos_causal = r_cos['em_after_delete'] <= 0.1
    if not g_repro:
        overall = 'COSINE_BASELINE_INVALID'
    elif g_delta >= 0.1 and g_votes_causal:
        overall = 'VOTES_BEAT_MEAN'
    elif abs(g_delta) < 0.1:
        overall = 'VOTES_TIE_MEAN'
    else:
        overall = 'MEAN_BEATS_VOTES'
    out = {'stage': 263, 'overall': overall, 'trunk': trunk.name, 'steps': steps, 'topk': k, 'fp_version': L.canonical_fp_version(), 'slots': len(tape.values), 'n_fit': len(fit), 'n_eval': len(ev), 'vocab': len(votes.postings), 'gates': {'G_votes_causal': g_votes_causal, 'G_cosine_causal': g_cos_causal, 'G_cosine_reproduces_256': g_repro}, 'summary': {'cosine_mean': r_cos, 'word_votes': r_vote, 'delta_em': g_delta, 'delta_full_bank_top1': r_vote.get('full_bank_top1', float('nan')) - r_cos.get('full_bank_top1', float('nan')), 'reference_256_em_glue': 0.667, 'reference_261f': {'votes_20way': 0.601, 'mean_20way': 0.226}}, 'note': "256's exam with one line changed: retrieval by cos(q, key) over the context mean, or by word postings with an idf weight and nothing fitted. Cosine arm gets wiki InfoNCE on W_q like 256; votes arm does not. If G_cosine_reproduces_256 is false, delta_em is not readable.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 263 word votes vs context mean (256's exam)\n\n**{overall}** slots={len(tape.values)} eval={len(ev)}\n\n- EM: cosine/mean **{r_cos['em']:.3f}** -> votes **{r_vote['em']:.3f}** (delta {g_delta:+.3f}; 256 published 0.667; G_cosine_reproduces_256={g_repro})\n- slot deleted: cosine {r_cos['em_after_delete']:.3f} | votes {r_vote['em_after_delete']:.3f}\n- votes are zero-train; cosine arm also trains W_q via wiki InfoNCE\n", encoding='utf-8')
    log(json.dumps({'overall': overall, 'delta_em': g_delta, 'G_cosine_reproduces_256': g_repro}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())