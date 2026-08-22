"""
Stage 212b — instance disambiguation, retried with the 258 channel.

212 closed THESIS_NO: a read-only 2-layer MLP over frozen P1 state could not tell WHICH
occurrence of a surface form we are in, so it could not resolve collisions a surface-keyed
memory is blind to (`g1_collision` false). That was one scale, one architecture, no control —
the same shape of evidence 210-212 were criticised for.

Since then 258 built the mechanism 212 was missing: instead of a standalone head scored on its
own contrastive objective, project the trunk state INTO KEY SPACE and blend it with the fp
query, trained by InfoNCE against the actual bank:

    q = normalize( (1 - a) * W_q(fp query) + a * W_sem(h_t) ),   a = sigmoid(MLP([h_t, fp conf]))

Collisions are the case where fp is blind BY CONSTRUCTION: one surface form, four occurrences,
four different values, so every candidate key carries the identical fingerprint. Chance 0.25.
Anything above it has to come from context, which is exactly what the semantic channel carries.

Store from one half of an occurrence window, query from the DISJOINT other half — no lexical
overlap shortcut inside an occurrence, same rule 212 used.

  fp_only     the 256 path — pinned at chance by construction
  fp + sem    the 258 channel
  gpt2 + sem  matched control, so a negative separates scale from architecture
  shuffled    keys permuted — causal floor

  python _stage212b_instance_sem.py [--smoke] [--no-gpt-control]
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import hidden_and_logits
RES = Path('results')
DECISION = RES / 'stage212b_decision.json'
MINI = RES / 'stage212b_mini.md'
LOG = RES / '_stage212b_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
WIKI = Path('data/_wikitext103_train.txt')
WORD_RE = re.compile('[A-Za-z][a-z]{2,}')
SEED = 212
N_SIB = 4
CHANCE = 1.0 / N_SIB

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

class SemQuery(nn.Module):

    def __init__(self, in_dim: int, device):
        super().__init__()
        self.proj = nn.Linear(in_dim, 256).to(device)
        self.blend = nn.Sequential(nn.Linear(in_dim + 2, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.blend[-1].weight)
        nn.init.constant_(self.blend[-1].bias, -2.0)

    def q(self, h):
        return F.normalize(self.proj(h), dim=-1)

    def a(self, h, conf):
        return torch.sigmoid(self.blend(torch.cat([h, conf], dim=-1))).squeeze(-1)

def fp_conf(q_fp, K):
    sims = q_fp @ K.t()
    two = torch.topk(sims, min(2, sims.size(-1)), dim=-1).values
    if two.size(-1) < 2:
        return torch.stack([two[..., 0], two[..., 0]], dim=-1)
    return torch.stack([two[..., 0], two[..., 0] - two[..., 1]], dim=-1)

@torch.no_grad()
def state(model, char_table, tok, pad_id, device, text):
    ids = [i for i in tok.encode(text).ids if i != pad_id][-MAX_ARCS:]
    if not ids:
        return None
    h, _ = hidden_and_logits(model, char_table, torch.tensor([ids], device=device), pad_id)
    return h[0, -1].detach().float()

def collisions(lines, want: int, win: int=220):
    """Surface forms with >= N_SIB occurrences; each occurrence contributes two DISJOINT halves
    of its own window: one writes the key, the other asks."""
    occ = defaultdict(list)
    for ln in lines:
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5:
                continue
            lo, hi = (max(0, m.start() - win), min(len(ln), m.end() + win))
            seg = ln[lo:hi]
            if len(WORD_RE.findall(seg)) < 12:
                continue
            cut = len(seg) // 2
            occ[e].append((seg[:cut], seg[cut:]))
    out = {e: v[:N_SIB] for e, v in occ.items() if len(v) >= N_SIB}
    return dict(list(out.items())[:want])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--forms', type=int, default=0)
    ap.add_argument('--no-gpt-control', action='store_true')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = args.steps or (200 if args.smoke else 800)
    n_forms = args.forms or (24 if args.smoke else 120)
    max_lines = 4000 if args.smoke else 30000
    log(f'Stage212b instance-sem start {datetime.now(timezone.utc).isoformat()} device={device}')
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
    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank = FpBank(model_can, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(4000000 if args.smoke else 25000000)
    lines = [l.strip() for l in wtext.split('\n') if 200 <= len(l.strip()) <= 600][:max_lines]
    forms = collisions(lines, n_forms)
    log(f'  surface forms with >={N_SIB} occurrences: {len(forms)}')
    if len(forms) < 8:
        log('  not enough colliding forms')
        return 1
    keys, meta, items = ([], [], [])
    for e, occs in forms.items():
        base = bank.fp([e])[0]
        sib = []
        for j, (half_w, half_q) in enumerate(occs):
            c = bank.ctx_fp(half_w, exclude=e)
            if c is None:
                continue
            keys.append(F.normalize(base + c, dim=-1))
            sib.append(len(meta))
            meta.append({'form': e, 'occ': j, 'slot': len(meta)})
        for idx, (half_w, half_q) in zip(sib, occs):
            qc = bank.ctx_fp(half_q, exclude=e)
            st = state(model, char_table, tok, pad_id, device, half_q)
            if qc is None or st is None:
                continue
            items.append({'form': e, 'slot': idx, 'sib': sib, 'raw': F.normalize(base + qc, dim=-1), 'h': st, 'qtext': half_q})
    if len(items) < 16:
        log('  not enough usable occurrences')
        return 1
    K = torch.stack(keys, 0).to(device).float()
    rng.shuffle(items)
    n_fit = len(items) // 2
    fit, ev = (items[:n_fit], items[n_fit:])
    log(f'  slots={len(meta)} fit={len(fit)} eval={len(ev)} chance={CHANCE:.3f}')
    semq = SemQuery(int(fit[0]['h'].numel()), device)
    W_q = L.init_query_adapter(device)
    opt = torch.optim.AdamW(list(semq.parameters()) + list(W_q.parameters()), lr=0.002, weight_decay=0.01)
    Rq = torch.stack([it['raw'] for it in fit]).to(device).float()
    Hq = torch.stack([it['h'] for it in fit]).to(device).float()
    Gq = torch.tensor([it['slot'] for it in fit], device=device)
    for step in range(1, steps + 1):
        sel = torch.randint(0, Rq.size(0), (min(32, Rq.size(0)),), device=device)
        q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
        a = semq.a(Hq[sel], fp_conf(q_fp, K)).unsqueeze(-1)
        q = F.normalize((1 - a) * q_fp + a * semq.q(Hq[sel]), dim=-1)
        loss = F.cross_entropy(q @ K.t() / args.tau, Gq[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(semq.parameters()) + list(W_q.parameters()), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0:
            log(f'  step {step}/{steps} loss={float(loss):.3f} a={float(a.mean()):.3f}')
    semq.eval()

    @torch.no_grad()
    def score(use_sem, Kmat=K):
        ok, alphas = ([], [])
        for it in ev:
            q_fp = F.normalize(W_q(it['raw'].unsqueeze(0)), dim=-1)[0]
            if use_sem:
                a = semq.a(it['h'], fp_conf(q_fp, Kmat))
                q = F.normalize((1 - a) * q_fp + a * semq.q(it['h']), dim=-1)
                alphas.append(float(a))
            else:
                q = q_fp
            sims = Kmat @ q
            best = max(it['sib'], key=lambda j: float(sims[j]))
            ok.append(int(best == it['slot']))
        return {'collision_4way': float(np.mean(ok)), 'n': len(ok), 'alpha': float(np.mean(alphas)) if alphas else 0.0}
    fp_only, sem = (score(False), score(True))
    perm = torch.randperm(K.size(0), generator=torch.Generator().manual_seed(SEED + 1))
    shuf = score(True, Kmat=K[perm.to(K.device)])
    log(f"fp_only={fp_only['collision_4way']:.3f} sem={sem['collision_4way']:.3f} shuffled={shuf['collision_4way']:.3f} (chance {CHANCE:.3f})")
    gpt = None
    if not args.no_gpt_control:
        try:
            gm = L.load_gpt(device)
            ok = True
            for it in items:
                e = L.gpt_emb(gm, tok, pad_id, device, [i for i in tok.encode(it['qtext']).ids if i != pad_id])
                if e is None:
                    ok = False
                    break
                it['h_gpt'] = e.detach().float()
            if ok:
                semg = SemQuery(int(items[0]['h_gpt'].numel()), device)
                Wg = L.init_query_adapter(device)
                og = torch.optim.AdamW(list(semg.parameters()) + list(Wg.parameters()), lr=0.002)
                Hg = torch.stack([it['h_gpt'] for it in fit]).to(device).float()
                for _ in range(steps):
                    sel = torch.randint(0, Rq.size(0), (min(32, Rq.size(0)),), device=device)
                    qf = F.normalize(Wg(Rq[sel]), dim=-1)
                    a = semg.a(Hg[sel], fp_conf(qf, K)).unsqueeze(-1)
                    q = F.normalize((1 - a) * qf + a * semg.q(Hg[sel]), dim=-1)
                    lo_ = F.cross_entropy(q @ K.t() / args.tau, Gq[sel])
                    og.zero_grad(set_to_none=True)
                    lo_.backward()
                    og.step()
                semg.eval()
                with torch.no_grad():
                    okc = []
                    for it in ev:
                        qf = F.normalize(Wg(it['raw'].unsqueeze(0)), dim=-1)[0]
                        a = semg.a(it['h_gpt'], fp_conf(qf, K))
                        q = F.normalize((1 - a) * qf + a * semg.q(it['h_gpt']), dim=-1)
                        s = K @ q
                        okc.append(int(max(it['sib'], key=lambda j: float(s[j])) == it['slot']))
                    gpt = {'collision_4way': float(np.mean(okc))}
                log(f"gpt2+sem={gpt['collision_4way']:.3f}")
        except Exception as e:
            log(f'  gpt control unavailable: {type(e).__name__}: {e}')
    g_fp_blind = fp_only['collision_4way'] <= CHANCE + 0.1
    g_sem = sem['collision_4way'] >= CHANCE + 0.2
    g_beats_fp = sem['collision_4way'] >= fp_only['collision_4way'] + 0.15
    g_causal = shuf['collision_4way'] <= CHANCE + 0.1
    gpt_also_fails = gpt is not None and gpt['collision_4way'] < CHANCE + 0.2
    if not g_fp_blind:
        overall = 'INSTANCE_SEM_INVALID'
    elif g_sem and g_beats_fp and g_causal:
        overall = 'INSTANCE_SEM_OK'
    elif gpt_also_fails:
        overall = 'INSTANCE_SEM_NO_AT_SCALE'
    else:
        overall = 'INSTANCE_SEM_NO'
    out = {'stage': '212b', 'overall': overall, 'trunk': trunk_ckpt.name, 'chance': CHANCE, 'steps': steps, 'slots': len(meta), 'n_fit': len(fit), 'n_eval': len(ev), 'gates': {'G_fp_blind_by_construction': g_fp_blind, 'G_sem_above_chance': g_sem, 'G_beats_fp_only': g_beats_fp, 'G_tape_causal': g_causal}, 'summary': {'fp_only': fp_only, 'fp_plus_sem': sem, 'shuffled_keys': shuf, 'gpt_control': gpt}, 'note': 'Retry of 212 with the mechanism it lacked. 212 scored a standalone read-only MLP on its own contrastive objective; here the trunk state is projected INTO KEY SPACE and blended with the fp query, trained by InfoNCE against the real bank (the 258 channel). Collisions are where fp is blind by construction — one surface form, four occurrences, four values, identical fingerprint on every sibling key — so G_fp_blind must hold or the exam leaked. Store and query halves of an occurrence window are disjoint, as in 212. P1 and trunk frozen; only W_q, W_sem and the blend train.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f"# Stage 212b instance channel, retried via W_sem\n\n**{overall}** chance={CHANCE:.2f} slots={len(meta)} eval={len(ev)}\n\n- collision 4-way: fp-only **{fp_only['collision_4way']:.3f}** -> fp+sem **{sem['collision_4way']:.3f}** (shuffled {shuf['collision_4way']:.3f})\n- blend a {sem['alpha']:.3f}\n" + (f"- matched GPT-2: {gpt['collision_4way']:.3f}\n" if gpt else '- matched GPT-2: not run\n'), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())