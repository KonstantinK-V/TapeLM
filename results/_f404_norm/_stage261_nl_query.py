"""
Stage 261 — Can a natural question drive the retrieval, with no cue template anywhere?

256/257 used hand-written cues. 258 removed the cue's lexical overlap but still drew wording
from a fixed per-relation dictionary, so W_sem could learn "this template -> that relation".
Both sides of the exam were still authored. This stage authors neither.

The fact is written from one REAL wikitext sentence mentioning entity E. The question is a
DIFFERENT real wikitext sentence mentioning the same E, truncated just before it. Both
contexts are natural prose, written by different people about the same thing, and nothing in
between was designed by us:

    slot   key = norm( fp(anchor_A) + ctx_fp(sentence A, exclude=E) ),  value = E
    query  from sentence B, prefix up to where E begins        gold = that slot

The discriminator, and the reason this stage exists: LEXICAL OVERLAP between the two contexts.
Bag-of-spellings retrieval works when A and B happen to share words. Report accuracy split by
overlap quartile — if it only works in the high-overlap half, the query is still spelling
matching and NL_QUERY_LEXICAL_ONLY is the verdict, not a win.

Channels compared on identical queries:
    fp-only          W_q(anchor fp + ctx_fp)              = the 256 path
    fp + semantic    blend with W_sem(h_t), as in 258     = trunk understanding in the query

The bank also has to contain entities that are NOT on the exam. The first run built it from the
exam entities alone - 53 slots for 26 fit and 27 eval items - so InfoNCE could satisfy itself by
learning "point at one of these 26", drove the loss to 0.007 by step 40, and then sent eval
queries to the same places: fp+sem 0.037 against fp-only 0.148, below even the shuffled control.
Wiki noise slots make that shortcut worthless.

Keys canonical frozen fp; P1 and trunk frozen; only W_q, W_sem and the blend train. Entities
used for fitting and for evaluation are disjoint.

  python _stage261_nl_query.py [--smoke] [--no-gpt-control]
  python _stage261_nl_query.py --recipe fix1p   # strong fp floor + alpha cap + max-score eval
  python _stage261_nl_query.py --recipe fix1q   # fp-only W_q pretrain, freeze W_q, then sem mixer
  python _stage261_nl_query.py --recipe tape_rerank   # fp top-k, trunk reranks (no alpha)
  python _stage261_nl_query.py --recipe tape_dualkey  # fp + sem keys, read = max (no alpha)
  python _stage261_nl_query.py --recipe tape_symkey   # h_t in key and query symmetrically
  python _stage261_nl_query.py --recipe tape_rerank_val  # fp top-k, rerank reads entity value (fp)
  python _stage261_nl_query.py --recipe tape_qkey     # write-side key = predicted question h
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
import _stage260f_open_gate as s260f
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, hidden_and_logits
from _stage262_trunk_swap import ExternalTrunk
RES = Path('results')
DECISION = RES / 'stage261_decision.json'
MINI = RES / 'stage261_mini.md'
LOG = RES / '_stage261_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_JOINT = Path('checkpoints/stage253_joint_l02.pt')
CKPT_OUT = Path('checkpoints/stage261_nl_query.pt')
WIKI = Path('data/_wikitext103_train.txt')
WORD_RE = re.compile('[A-Za-z][a-z]{2,}')
SEED = 261

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)
RECIPES = ('baseline', 'fix1', 'fix1m', 'fix1p', 'fix1q', 'fix2', 'fix3', 'all', 'tape_rerank', 'tape_dualkey', 'tape_symkey', 'tape_rerank_val', 'tape_qkey')
RRF_K = 60
TAPE_MODES = {'tape_rerank': 'rerank', 'tape_dualkey': 'dualkey', 'tape_symkey': 'symkey', 'tape_rerank_val': 'rerank_val', 'tape_qkey': 'qkey'}

def recipe_flags(name: str) -> dict:
    name = name if name in RECIPES else 'baseline'
    tape_mode = TAPE_MODES.get(name)
    return {'name': name, 'tape_mode': tape_mode, 'fp_floor': name in ('fix1', 'fix1m', 'fix1p', 'fix1q', 'fix2', 'fix3', 'all'), 'fp_floor_strong': name in ('fix1p', 'fix1q'), 'rrf': name in ('fix2', 'fix3', 'all'), 'feat_gate': name in ('fix3', 'all'), 'hygiene': name == 'all', 'alpha_cap': 0.35 if name in ('fix1m', 'fix1p', 'fix1q') else None if name != 'all' else 0.3, 'score_max_fusion': name in ('fix1m', 'fix1p', 'fix1q'), 'wq_fp_pretrain_steps': 800 if name == 'fix1q' else 0, 'freeze_wq': name == 'fix1q'}

def alpha_warmup_cap(step: int, total_warmup: int, max_cap: float=0.3) -> float:
    if total_warmup <= 0:
        return 1.0
    return max_cap * min(1.0, step / total_warmup)

def fp_floor_loss(q_fp: torch.Tensor, q_blend: torch.Tensor, K: torch.Tensor, gold: torch.Tensor, tau: float) -> torch.Tensor:
    """Blend must not score gold below fp-only on the same batch."""
    s_fp = (q_fp * K[gold]).sum(dim=-1)
    s_bl = (q_blend * K[gold]).sum(dim=-1)
    return F.relu(s_fp - s_bl).mean()

def fp_floor_loss_strong(q_fp: torch.Tensor, q_blend: torch.Tensor, K: torch.Tensor, gold: torch.Tensor, tau: float, *, lam_gold: float=8.0, lam_top: float=4.0) -> torch.Tensor:
    """Gold score floor + when fp argmax is gold, blend must not promote a distractor above gold."""
    s_fp = q_fp @ K.t() / tau
    s_bl = q_blend @ K.t() / tau
    g = gold.unsqueeze(1)
    s_fp_g = s_fp.gather(1, g).squeeze(1)
    s_bl_g = s_bl.gather(1, g).squeeze(1)
    l_gold = F.relu(s_fp_g - s_bl_g).mean()
    fp_top = s_fp.argmax(dim=-1)
    mask = fp_top == gold
    if mask.any():
        bl_max = s_bl.max(dim=-1).values
        l_top = F.relu(bl_max[mask] - s_bl_g[mask]).mean()
    else:
        l_top = s_fp.new_zeros(())
    return lam_gold * l_gold + lam_top * l_top

def rrf_scores(s_fp: torch.Tensor, s_sem: torch.Tensor, k: int=RRF_K) -> torch.Tensor:
    """Reciprocal rank fusion; sem cannot demote fp-only winner outside top-k rerank pool."""
    n = s_fp.size(-1)
    k = min(k, n)
    top_fp = torch.topk(s_fp, k, dim=-1).indices
    out = s_fp.clone()
    for b in range(s_fp.size(0)):
        for j in top_fp[b]:
            r_fp = 1 + int((s_fp[b] > s_fp[b, j]).sum())
            r_sem = 1 + int((s_sem[b] > s_sem[b, j]).sum())
            out[b, j] = 1.0 / (RRF_K + r_fp) + 1.0 / (RRF_K + r_sem)
    return out

class SemQuery(nn.Module):
    """Trunk state -> key space; blend gate from fp conf (2) or 260f retrieval feats (5)."""

    def __init__(self, in_dim: int, device, *, feat_gate: bool=False, key_only: bool=False):
        super().__init__()
        self.feat_gate = feat_gate
        self.key_only = key_only
        self.proj = nn.Linear(in_dim, 256).to(device)
        if not key_only:
            d_blend = len(s260f.FEAT_NAMES) if feat_gate else in_dim + 2
            self.blend = nn.Sequential(nn.Linear(d_blend, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
            nn.init.zeros_(self.blend[-1].weight)
            nn.init.constant_(self.blend[-1].bias, -2.0)
        if feat_gate:
            self.register_buffer('mu', torch.zeros(len(s260f.FEAT_NAMES), device=device))
            self.register_buffer('sd', torch.ones(len(s260f.FEAT_NAMES), device=device))

    def q(self, h):
        return F.normalize(self.proj(h), dim=-1)

    def fit_feat_norm(self, rows: list[torch.Tensor]) -> None:
        if not rows or not self.feat_gate:
            return
        M = torch.stack(rows)
        self.mu.copy_(M.mean(0))
        self.sd.copy_(M.std(0).clamp_min(0.001))

    def blend_input(self, h, q_fp: torch.Tensor, K: torch.Tensor, tau: float):
        sims = q_fp @ K.t()
        logits = sims / tau
        if self.feat_gate:
            feats = s260f.retrieval_feats(sims[0], logits[0])
            z = (feats - self.mu) / self.sd
            return z.unsqueeze(0)
        h1 = h if h.dim() == 1 else h.reshape(-1)
        conf = fp_conf(q_fp, K).reshape(-1)[:2]
        return torch.cat([h1, conf], dim=-1).unsqueeze(0)

    def a(self, h, q_fp: torch.Tensor, K: torch.Tensor, tau: float=0.05):
        if self.key_only:
            return q_fp.new_zeros(())
        if q_fp.dim() == 1:
            q_fp = q_fp.unsqueeze(0)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        alphas = []
        for i in range(q_fp.size(0)):
            x = self.blend_input(h[i], q_fp[i:i + 1], K, tau)
            alphas.append(torch.sigmoid(self.blend(x)).squeeze(-1))
        return torch.stack(alphas)

class RerankHead(nn.Module):
    """Trunk read step: score fp top-k slot keys with query hidden state (no vector blend)."""

    def __init__(self, in_dim: int, device):
        super().__init__()
        self.proj = nn.Linear(in_dim, 256).to(device)

    def scores(self, h_q: torch.Tensor, k_cand: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.proj(h_q if h_q.dim() == 1 else h_q.reshape(-1)), dim=-1)
        if k_cand.dim() == 1:
            return k_cand @ q
        return k_cand @ q

def train_wq_fp_only(W_q, Rq, Gq, K, train_mask, steps: int, tau: float, device, *, log_every: int) -> None:
    opt = torch.optim.AdamW(W_q.parameters(), lr=0.002, weight_decay=0.01)
    for step in range(1, steps + 1):
        pool = torch.where(train_mask)[0]
        sel = pool[torch.randint(0, pool.numel(), (min(32, pool.numel()),))]
        q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
        loss = F.cross_entropy(q_fp @ K.t() / tau, Gq[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(W_q.parameters(), 1.0)
        opt.step()
        if step == steps or step % max(1, log_every) == 0:
            log(f'  wq-fp {step}/{steps} loss={float(loss):.3f}')

def _fp_topk_cands(s_fp: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, s_fp.numel())
    return torch.topk(s_fp, k).indices

def _gold_in_fp_topk(s_fp: torch.Tensor, gold: int, k: int) -> tuple[bool, torch.Tensor]:
    topi = _fp_topk_cands(s_fp, k)
    return (int(gold) in topi.tolist(), topi)

def train_tape_rerank(W_q, rerank, fit, K, Hq, Gq, Rq, train_mask, steps, tau, rerank_k, device, *, read_vecs: torch.Tensor | None=None) -> dict:
    """Rerank only within fp top-k; never inject gold (avoids low-fp outlier leak)."""
    opt = torch.optim.AdamW(list(rerank.parameters()), lr=0.002, weight_decay=0.01)
    read = read_vecs if read_vecs is not None else K
    n_skip, n_train = (0, 0)
    for step in range(1, steps + 1):
        pool = torch.where(train_mask)[0]
        sel = pool[torch.randint(0, pool.numel(), (min(24, pool.numel()),))]
        loss_acc = []
        for idx in sel.tolist():
            q_fp = F.normalize(W_q(Rq[idx:idx + 1]), dim=-1)[0]
            s_fp = K @ q_fp
            ok, topi = _gold_in_fp_topk(s_fp, int(Gq[idx]), rerank_k)
            if not ok:
                n_skip += 1
                continue
            n_train += 1
            logits = rerank.scores(Hq[idx], read[topi]) / tau
            tgt = (topi == Gq[idx]).nonzero(as_tuple=True)[0].squeeze()
            loss_acc.append(F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0)))
        if not loss_acc:
            continue
        loss = torch.stack(loss_acc).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 40 or step % max(1, steps // 5) == 0:
            log(f'  rerank {step}/{steps} loss={float(loss):.3f} train={n_train} skip={n_skip}')
    return {'rerank_train_steps': n_train, 'rerank_skip_gold_not_in_topk': n_skip}

class QKeyTape(nn.Module):
    """Write-side key = address in question space; read-side query from actual question h_t."""

    def __init__(self, in_dim: int, device):
        super().__init__()
        self.write = nn.Linear(in_dim, 256).to(device)
        self.read = nn.Linear(in_dim, 256).to(device)

    def keys(self, h_write: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.write(h_write), dim=-1)

    def query(self, h_read: torch.Tensor) -> torch.Tensor:
        if h_read.dim() == 1:
            return F.normalize(self.read(h_read), dim=-1)
        return F.normalize(self.read(h_read), dim=-1)

def train_tape_qkey(qkey: QKeyTape, H_write, fit_tensors, train_mask, steps, tau, device):
    Rq, Hq, Gq, _K = fit_tensors
    opt = torch.optim.AdamW(qkey.parameters(), lr=0.002, weight_decay=0.01)
    for step in range(1, steps + 1):
        Kq = qkey.keys(H_write)
        pool = torch.where(train_mask)[0]
        sel = pool[torch.randint(0, pool.numel(), (min(32, pool.numel()),))]
        q = qkey.query(Hq[sel])
        loss = F.cross_entropy(q @ Kq.t() / tau, Gq[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 40 or step % max(1, steps // 5) == 0:
            log(f'  qkey {step}/{steps} loss={float(loss):.3f}')

def train_tape_dualkey(W_q, semq, H_write, fit_tensors, train_mask, steps, tau, device):
    Rq, Hq, Gq, K_fp = fit_tensors
    opt = torch.optim.AdamW(list(W_q.parameters()) + list(semq.parameters()), lr=0.002, weight_decay=0.01)
    for step in range(1, steps + 1):
        pool = torch.where(train_mask)[0]
        sel = pool[torch.randint(0, pool.numel(), (min(32, pool.numel()),))]
        K_sem = semq.q(H_write)
        q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
        q_sem = semq.q(Hq[sel])
        s = torch.maximum(q_fp @ K_fp.t(), q_sem @ K_sem.t())
        loss = F.cross_entropy(s / tau, Gq[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 40 or step % max(1, steps // 5) == 0:
            log(f'  dualkey {step}/{steps} loss={float(loss):.3f}')

def train_tape_symkey(W_q, semq, H_write, fit_tensors, train_mask, steps, tau, device):
    Rq, Hq, Gq, K_fp = fit_tensors
    opt = torch.optim.AdamW(list(W_q.parameters()) + list(semq.parameters()), lr=0.002, weight_decay=0.01)
    for step in range(1, steps + 1):
        pool = torch.where(train_mask)[0]
        sel = pool[torch.randint(0, pool.numel(), (min(32, pool.numel()),))]
        k_joint = F.normalize(K_fp + semq.q(H_write), dim=-1)
        q = F.normalize(W_q(Rq[sel]) + semq.q(Hq[sel]), dim=-1)
        loss = F.cross_entropy(q @ k_joint.t() / tau, Gq[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 40 or step % max(1, steps // 5) == 0:
            log(f'  symkey {step}/{steps} loss={float(loss):.3f}')

@torch.no_grad()
def acc_20way_batch(W_q, semq, bank, K, items, states, use_sem: bool, med: float, *, rrf: bool, tau: float) -> float:
    wrng = random.Random(SEED + 99)
    hits = []
    for it in items:
        q_fp = F.normalize(W_q(it['raw'].unsqueeze(0)), dim=-1)
        if use_sem and semq is not None:
            a = float(semq.a(it['h'], q_fp[0], K, tau).reshape(-1)[0])
            q_sem = F.normalize(semq.q(it['h'].unsqueeze(0)), dim=-1).reshape(-1)
            if rrf:
                s_fp = (K @ q_fp[0]).unsqueeze(0)
                s_sem = (K @ q_sem).unsqueeze(0)
                sims = rrf_scores(s_fp, s_sem)[0]
            else:
                q = F.normalize((1 - a) * q_fp[0] + a * q_sem, dim=-1)
                sims = K @ q
        else:
            sims = K @ q_fp[0]
        pool = [j for j in wrng.sample(range(K.size(0)), min(20 * 3, K.size(0))) if j != it['slot']][:19]
        hits.append(int(all((float(sims[it['slot']]) >= float(sims[j]) for j in pool))))
    return float(np.mean(hits)) if hits else 0.0

def fp_conf(q_fp, K):
    sims = q_fp @ K.t()
    two = torch.topk(sims, min(2, sims.size(-1)), dim=-1).values
    if two.size(-1) < 2:
        return torch.stack([two[..., 0], two[..., 0]], dim=-1)
    return torch.stack([two[..., 0], two[..., 0] - two[..., 1]], dim=-1)

def ctx_words(text: str, exclude: str | None=None) -> set:
    return {w.lower() for w in WORD_RE.findall(text) if w != exclude}

def entity_in_query(ent: str, qtext: str) -> bool:
    """True if gold entity string appears in the natural query prefix (before truncation at E)."""
    return re.search(f'\\b{re.escape(ent)}\\b', qtext, re.IGNORECASE) is not None

def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))

def fp_raw(bank: FpBank, text: str, use_anchor: bool=True):
    """256 recipe is anchor fp + context — shared anchor in 258; 261 write/ask anchors differ."""
    c = bank.ctx_fp(text)
    if c is None:
        return None
    if not use_anchor:
        return c
    an = ANCHOR_RE.findall(text)
    return F.normalize(bank.fp([an[-1]])[0] + c, dim=-1) if an else c

@torch.no_grad()
def trunk_state(model, char_table, tok, pad_id, device, text):
    ids = [i for i in tok.encode(text).ids if i != pad_id][-MAX_ARCS:]
    if not ids:
        return None
    h, _ = hidden_and_logits(model, char_table, torch.tensor([ids], device=device), pad_id)
    return h[0, -1].detach().float()

def collect(lines, bank, min_occ=2):
    """Entities appearing in at least two different real sentences: one writes the slot, the
    other asks the question. Neither sentence was authored by us."""
    by_ent = defaultdict(list)
    for ln in lines:
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5:
                continue
            lo = max(0, m.start() - 140)
            an = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != e]
            if an and len(by_ent[e]) < 4:
                by_ent[e].append({'line': ln, 'start': m.start(), 'end': m.end(), 'anchor': an[-1]})
    return {e: v for e, v in by_ent.items() if len(v) >= min_occ}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--entities', type=int, default=0)
    ap.add_argument('--distractor-slots', type=int, default=0, help='wiki entities added to the bank that no query ever asks for')
    ap.add_argument('--no-anchor', action='store_true', help='ctx only on keys and queries (write/ask anchors are different entities)')
    ap.add_argument('--query-names-entity', action='store_true', help='variant-2 ceiling: append gold entity to query fp+trunk (names what to retrieve)')
    ap.add_argument('--model', type=str, default='', help='external frozen causal LM for h_t (262 ExternalTrunk); empty = curve trunk')
    ap.add_argument('--no-gpt-control', action='store_true')
    ap.add_argument('--recipe', type=str, default='baseline', choices=RECIPES)
    ap.add_argument('--rerank-k', type=int, default=32, help='fp pool size for tape_rerank read step')
    args = ap.parse_args()
    rf = recipe_flags(args.recipe)
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    use_anchor = not args.no_anchor
    query_names_entity = args.query_names_entity
    steps = args.steps or (200 if args.smoke else 800)
    n_ent = args.entities or (60 if args.smoke else 400)
    n_dist = args.distractor_slots or (400 if args.smoke else 4000)
    max_lines = 3000 if args.smoke else 25000
    log(f"Stage261 nl query start {datetime.now(timezone.utc).isoformat()} device={device} steps={steps} recipe={rf['name']} anchor={use_anchor} query_names_entity={query_names_entity} trunk={args.model or 'curve'}")
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
    ext: ExternalTrunk | None = None
    if args.model:
        try:
            ext = ExternalTrunk(args.model, device)
            log(f'  external trunk: {args.model} hidden={ext.dim} (query h_t only; fp keys unchanged)')
        except Exception as e:
            log(f'  could not load {args.model}: {type(e).__name__}: {e}')
            return 1

    def query_h(text: str):
        if ext is not None:
            return ext.state(text)
        return trunk_state(model, char_table, tok, pad_id, device, text)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wtext = f.read(3000000 if args.smoke else 20000000)
    lines = [l.strip() for l in wtext.split('\n') if 80 <= len(l.strip()) <= 400][:max_lines]
    cands = collect(lines, bank)
    ents = sorted(cands)[:n_ent]
    rng.shuffle(ents)
    log(f'  entities with >=2 natural mentions: {len(cands)} (using {len(ents)})')
    if len(ents) < 16:
        log('  not enough multi-mention entities')
        return 1
    keys, vals, items, h_writes = ([], [], [], [])
    for e in ents:
        occ = cands[e]
        a, b = (occ[0], occ[1])
        wctx = a['line'][max(0, a['start'] - 140):min(len(a['line']), a['end'] + 140)]
        k = bank.ctx_fp(wctx, exclude=e)
        if k is None:
            continue
        h_w = trunk_state(model, char_table, tok, pad_id, device, wctx)
        qtext = b['line'][max(0, b['start'] - 200):b['start']].strip()
        if len(WORD_RE.findall(qtext)) < 4:
            continue
        ent_in_q = entity_in_query(e, qtext)
        q_use = f'{qtext} {e}' if query_names_entity else qtext
        raw = fp_raw(bank, q_use, use_anchor)
        st = query_h(q_use)
        if raw is None or st is None or h_w is None:
            continue
        keys.append(F.normalize(bank.fp([a['anchor']])[0] + k, dim=-1) if use_anchor else k)
        h_writes.append(h_w)
        items.append({'ent': e, 'slot': len(vals), 'qtext': qtext, 'q_use': q_use, 'raw': raw, 'h': st, 'h_write': h_w, 'ent_in_query': ent_in_q, 'overlap': jaccard(ctx_words(wctx, e), ctx_words(qtext, e))})
        vals.append(e)
    if len(items) < 16:
        log('  not enough usable (write, ask) pairs')
        return 1
    n_exam = len(keys)
    used = {it['ent'] for it in items}
    for ln in lines:
        if len(keys) >= n_exam + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = (max(0, m.start() - 140), min(len(ln), m.end() + 140))
            c = bank.ctx_fp(ln[lo:hi], exclude=e)
            if c is None:
                continue
            an = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != e]
            if use_anchor and (not an):
                continue
            h_w = trunk_state(model, char_table, tok, pad_id, device, ln[lo:hi])
            if h_w is None:
                continue
            keys.append(F.normalize(bank.fp([an[-1]])[0] + c, dim=-1) if use_anchor else c)
            h_writes.append(h_w)
            vals.append(e)
            used.add(e)
            if len(keys) >= n_exam + n_dist:
                break
    K = torch.stack(keys, 0).to(device).float()
    V_val = F.normalize(bank.fp(vals), dim=-1).to(device)
    log(f'  bank: {n_exam} exam slots + {len(keys) - n_exam} wiki noise = {len(keys)}')
    n_fit = len(items) // 2
    fit, ev = (items[:n_fit], items[n_fit:])
    med = float(np.median([it['overlap'] for it in ev]))
    log(f'  exam_slots={n_exam} fit={len(fit)} eval={len(ev)} | overlap median={med:.3f}')
    H_write = torch.stack(h_writes, 0).to(device).float()
    Rq = torch.stack([it['raw'] for it in fit]).to(device).float()
    Hq = torch.stack([it['h'] for it in fit]).to(device).float()
    Gq = torch.tensor([it['slot'] for it in fit], device=device)
    n_hold = max(4, len(fit) // 5)
    hold_idx = list(range(len(fit) - n_hold, len(fit)))
    train_mask = torch.ones(len(fit), dtype=torch.bool)
    train_mask[hold_idx] = False
    hold_items = [fit[i] for i in hold_idx]
    tape_mode = rf.get('tape_mode')
    rerank_stats: dict = {}
    rerank_k = max(8, args.rerank_k)
    fp_steps = steps if tape_mode else max(400, steps // 2)
    tape_steps = steps
    steps_done = steps
    W_q = L.init_query_adapter(device)
    W_q_fp_only = L.init_query_adapter(device)
    semq = None
    rerank = None
    qkey = None
    if tape_mode:
        log(f'  tape mode={tape_mode} fp_steps={fp_steps} tape_steps={tape_steps} rerank_k={rerank_k}')
        h_dim = int(fit[0]['h'].numel())
        fit_tensors = (Rq, Hq, Gq, K)
        if tape_mode != 'qkey':
            train_wq_fp_only(W_q, Rq, Gq, K, train_mask, fp_steps, args.tau, device, log_every=max(1, fp_steps // 4))
            W_q_fp_only.load_state_dict(W_q.state_dict())
        else:
            train_wq_fp_only(W_q, Rq, Gq, K, train_mask, fp_steps, args.tau, device, log_every=max(1, fp_steps // 4))
            W_q_fp_only.load_state_dict(W_q.state_dict())
        if tape_mode in ('rerank', 'rerank_val'):
            rerank = RerankHead(h_dim, device)
            for p in W_q.parameters():
                p.requires_grad_(False)
            read_mat = V_val if tape_mode == 'rerank_val' else K
            rerank_stats = train_tape_rerank(W_q, rerank, fit, K, Hq, Gq, Rq, train_mask, tape_steps, args.tau, rerank_k, device, read_vecs=read_mat)
            rerank.eval()
        elif tape_mode == 'qkey':
            qkey = QKeyTape(h_dim, device)
            train_tape_qkey(qkey, H_write, fit_tensors, train_mask, tape_steps, args.tau, device)
            qkey.eval()
        else:
            semq = SemQuery(h_dim, device, key_only=True)
            if tape_mode == 'dualkey':
                train_tape_dualkey(W_q, semq, H_write, fit_tensors, train_mask, tape_steps, args.tau, device)
            else:
                train_tape_symkey(W_q, semq, H_write, fit_tensors, train_mask, tape_steps, args.tau, device)
            semq.eval()

        @torch.no_grad()
        def score_fp(items_, Kmat=K, n_way: int=20, Wq=W_q):
            wrng = random.Random(SEED + 5)
            ranks, lo, hi, nway = ([], [], [], [])
            for it in items_:
                q_fp = F.normalize(Wq(it['raw'].unsqueeze(0)), dim=-1)[0]
                sims = Kmat @ q_fp
                r = 1 + int((sims > sims[it['slot']]).sum())
                ranks.append(r)
                (hi if it['overlap'] > med else lo).append(int(r == 1))
                pool = [j for j in wrng.sample(range(Kmat.size(0)), min(n_way * 3, Kmat.size(0))) if j != it['slot']][:n_way - 1]
                nway.append(int(all((float(sims[it['slot']]) >= float(sims[j]) for j in pool))))
            r = np.asarray(ranks, dtype=np.float64)
            return {'top1': float(np.mean(r == 1)), 'mrr': float(np.mean(1.0 / r)), 'median_rank': float(np.median(r)), 'top1_low_overlap': float(np.mean(lo)) if lo else float('nan'), 'top1_high_overlap': float(np.mean(hi)) if hi else float('nan'), 'alpha': 0.0, 'n': len(ranks), f'acc_{n_way}way': float(np.mean(nway)) if nway else float('nan'), f'chance_{n_way}way': 1.0 / n_way}

        @torch.no_grad()
        def score_tape(items_, Kmat=K, H_w=H_write, V_read=V_val, n_way: int=20):
            wrng = random.Random(SEED + 5)
            ranks, lo, hi, nway = ([], [], [], [])
            neg_inf = -10000.0
            read_mat = V_read if tape_mode == 'rerank_val' else Kmat
            for it in items_:
                q_fp = F.normalize(W_q(it['raw'].unsqueeze(0)), dim=-1)[0]
                gold = int(it['slot'])
                if tape_mode in ('rerank', 'rerank_val'):
                    s_fp = Kmat @ q_fp
                    ok, topi = _gold_in_fp_topk(s_fp, gold, rerank_k)
                    if ok:
                        sims = torch.full((Kmat.size(0),), neg_inf, device=Kmat.device, dtype=s_fp.dtype)
                        sims[topi] = rerank.scores(it['h'], read_mat[topi])
                    else:
                        sims = s_fp
                elif tape_mode == 'qkey':
                    Kq = qkey.keys(H_w)
                    sims = Kq @ qkey.query(it['h'])
                elif tape_mode == 'dualkey':
                    k_sem = semq.q(H_w)
                    q_sem = semq.q(it['h'].unsqueeze(0))[0]
                    sims = torch.maximum(Kmat @ q_fp, k_sem @ q_sem)
                else:
                    k_joint = F.normalize(Kmat + semq.q(H_w), dim=-1)
                    q = F.normalize(W_q(it['raw'].unsqueeze(0)) + semq.q(it['h'].unsqueeze(0)), dim=-1)[0]
                    sims = k_joint @ q
                r = 1 + int((sims > sims[gold]).sum())
                ranks.append(r)
                (hi if it['overlap'] > med else lo).append(int(r == 1))
                pool = [j for j in wrng.sample(range(Kmat.size(0)), min(n_way * 3, Kmat.size(0))) if j != gold][:n_way - 1]
                if tape_mode in ('rerank', 'rerank_val'):
                    nway.append(int(all((float(sims[gold]) >= float(sims[j]) for j in pool))))
                elif tape_mode == 'qkey':
                    nway.append(int(all((float(sims[gold]) >= float(sims[j]) for j in pool))))
                else:
                    nway.append(int(all((float(sims[gold]) >= float(sims[j]) for j in pool))))
            r = np.asarray(ranks, dtype=np.float64)
            return {'top1': float(np.mean(r == 1)), 'mrr': float(np.mean(1.0 / r)), 'median_rank': float(np.median(r)), 'top1_low_overlap': float(np.mean(lo)) if lo else float('nan'), 'top1_high_overlap': float(np.mean(hi)) if hi else float('nan'), 'alpha': 0.0, 'n': len(ranks), f'acc_{n_way}way': float(np.mean(nway)) if nway else float('nan'), f'chance_{n_way}way': 1.0 / n_way}
        fp_only = score_fp(ev)
        sem = score_tape(ev)
        fp_frozen = score_fp(ev, Wq=W_q_fp_only)
        perm = torch.randperm(K.size(0), generator=torch.Generator().manual_seed(SEED + 1)).to(K.device)
        K_sh = K[perm]
        H_sh = H_write[perm] if tape_mode in ('dualkey', 'symkey', 'qkey') else H_write
        shuf = score_tape(ev, Kmat=K_sh, H_w=H_sh)
        log(f'fp-only: {json.dumps(fp_only)}')
        log(f'tape ({tape_mode}): {json.dumps(sem)}')
        log(f"shuffled keys: top1={shuf['top1']:.3f}")
    else:
        semq = SemQuery(int(fit[0]['h'].numel()), device, feat_gate=rf['feat_gate'])
        W_q = L.init_query_adapter(device)
        W_q_fp_only = L.init_query_adapter(device)
        W_q_fp_only.load_state_dict(W_q.state_dict())
        for p in W_q_fp_only.parameters():
            p.requires_grad_(False)
        if rf['feat_gate']:
            feat_rows = []
            W0 = L.init_query_adapter(device)
            with torch.no_grad():
                for it in fit:
                    qf = F.normalize(W0(it['raw'].unsqueeze(0)), dim=-1)
                    sims = qf @ K.t()
                    feat_rows.append(s260f.retrieval_feats(sims[0], sims[0] / args.tau))
            semq.fit_feat_norm(feat_rows)
        pretrain = int(rf.get('wq_fp_pretrain_steps') or 0)
        if pretrain > 0:
            opt_fp = torch.optim.AdamW(W_q.parameters(), lr=0.002, weight_decay=0.01)
            for pstep in range(1, pretrain + 1):
                pool = torch.where(train_mask)[0]
                sel = pool[torch.randint(0, pool.numel(), (min(32, pool.numel()),))]
                q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
                loss_fp = F.cross_entropy(q_fp @ K.t() / args.tau, Gq[sel])
                opt_fp.zero_grad(set_to_none=True)
                loss_fp.backward()
                torch.nn.utils.clip_grad_norm_(W_q.parameters(), 1.0)
                opt_fp.step()
                if pstep == pretrain or pstep % max(1, pretrain // 4) == 0:
                    log(f'  wq-pretrain {pstep}/{pretrain} loss={float(loss_fp):.3f}')
            W_q_fp_only.load_state_dict(W_q.state_dict())
            if rf['freeze_wq']:
                for p in W_q.parameters():
                    p.requires_grad_(False)
        wq_params = [] if rf['freeze_wq'] else list(W_q.parameters())
        opt = torch.optim.AdamW(list(semq.parameters()) + wq_params, lr=0.002, weight_decay=0.01)
        warmup_steps = 200 if rf['hygiene'] else 0
        best_step, stop_step = (steps, steps)
        for step in range(1, steps + 1):
            pool = torch.where(train_mask)[0]
            sel = pool[torch.randint(0, pool.numel(), (min(32, pool.numel()),))]
            q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
            a_raw = semq.a(Hq[sel], q_fp, K, args.tau)
            a = a_raw.reshape(-1, 1)
            if rf['hygiene']:
                cap = alpha_warmup_cap(step, warmup_steps, max_cap=float(rf['alpha_cap'] or 0.3))
                a = (a * cap).clamp(0.0, cap)
            elif rf['alpha_cap'] is not None:
                a = a.clamp(0.0, float(rf['alpha_cap']))
            q_sem = semq.q(Hq[sel])
            q = F.normalize((1 - a) * q_fp + a * q_sem, dim=-1)
            logits = q @ K.t() / args.tau
            loss = F.cross_entropy(logits, Gq[sel])
            if rf['fp_floor_strong']:
                loss = loss + fp_floor_loss_strong(q_fp, q, K, Gq[sel], args.tau)
                if not rf['freeze_wq']:
                    with torch.no_grad():
                        q_fp_f = F.normalize(W_q_fp_only(Rq[sel]), dim=-1)
                    loss = loss + 0.5 * F.cross_entropy(q_fp_f @ K.t() / args.tau, Gq[sel])
            elif rf['fp_floor']:
                loss = loss + fp_floor_loss(q_fp, q, K, Gq[sel], args.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(semq.parameters()) + wq_params, 1.0)
            opt.step()
            if rf['hygiene'] and step >= 80 and (step % 40 == 0):
                semq.eval()
                fp20 = acc_20way_batch(W_q, None, bank, K, hold_items, {}, False, med, rrf=False, tau=args.tau)
                s20 = acc_20way_batch(W_q, semq, bank, K, hold_items, {}, True, med, rrf=rf['rrf'], tau=args.tau)
                semq.train()
                if s20 + 1e-06 < fp20:
                    stop_step = step
                    log(f'  early-stop @ {step}: holdout 20-way fp={fp20:.3f} sem={s20:.3f}')
                    break
                best_step = step
            if step == 40 or step % max(1, steps // 5) == 0:
                log(f'  step {step}/{steps} loss={float(loss):.3f} a={float(a.mean()):.3f}')
        semq.eval()
        steps_done = stop_step

        def _alpha_eval(raw_a: torch.Tensor) -> float:
            a = float(raw_a.reshape(-1)[0])
            if rf['alpha_cap'] is not None:
                return min(a, float(rf['alpha_cap']))
            return a

        @torch.no_grad()
        def score(items_, use_sem, Kmat=K, n_way: int=20, Wq=W_q):
            wrng = random.Random(SEED + 5)
            ranks, alphas, lo, hi, nway = ([], [], [], [], [])
            for it in items_:
                q_fp = F.normalize(Wq(it['raw'].unsqueeze(0)), dim=-1)[0]
                if use_sem:
                    af = _alpha_eval(semq.a(it['h'], q_fp, Kmat, args.tau))
                    alphas.append(af)
                    q_sem = F.normalize(semq.q(it['h'].unsqueeze(0)), dim=-1).reshape(-1)
                    s_fp_v = Kmat @ q_fp
                    if rf['rrf']:
                        sims = rrf_scores(s_fp_v.unsqueeze(0), (Kmat @ q_sem).unsqueeze(0))[0]
                    elif rf['score_max_fusion']:
                        q_bl = F.normalize((1 - af) * q_fp + af * q_sem, dim=-1)
                        sims = torch.maximum(s_fp_v, Kmat @ q_bl)
                    else:
                        q = F.normalize((1 - af) * q_fp + af * q_sem, dim=-1)
                        sims = Kmat @ q
                else:
                    sims = Kmat @ q_fp
                r = 1 + int((sims > sims[it['slot']]).sum())
                ranks.append(r)
                (hi if it['overlap'] > med else lo).append(int(r == 1))
                pool = [j for j in wrng.sample(range(Kmat.size(0)), min(n_way * 3, Kmat.size(0))) if j != it['slot']][:n_way - 1]
                nway.append(int(all((float(sims[it['slot']]) >= float(sims[j]) for j in pool))))
            r = np.asarray(ranks, dtype=np.float64)
            return {'top1': float(np.mean(r == 1)), 'mrr': float(np.mean(1.0 / r)), 'median_rank': float(np.median(r)), 'top1_low_overlap': float(np.mean(lo)) if lo else float('nan'), 'top1_high_overlap': float(np.mean(hi)) if hi else float('nan'), 'alpha': float(np.mean(alphas)) if alphas else 0.0, 'n': len(ranks), f'acc_{n_way}way': float(np.mean(nway)) if nway else float('nan'), f'chance_{n_way}way': 1.0 / n_way}
        fp_only, sem = (score(ev, False), score(ev, True))
        fp_frozen = score(ev, False, Wq=W_q_fp_only)
        perm = torch.randperm(K.size(0), generator=torch.Generator().manual_seed(SEED + 1))
        shuf = score(ev, True, Kmat=K[perm.to(K.device)])
        log(f'fp-only: {json.dumps(fp_only)}')
        log(f'fp+sem : {json.dumps(sem)}')
        log(f"shuffled keys: top1={shuf['top1']:.3f}")
    gpt = None
    if not args.no_gpt_control and (not tape_mode):
        try:
            gm = L.load_gpt(device)
            hs = []
            for it in items:
                e = L.gpt_emb(gm, tok, pad_id, device, [i for i in tok.encode(it['qtext']).ids if i != pad_id])
                hs.append(None if e is None else e.detach().float())
            if all((x is not None for x in hs)):
                for it, x in zip(items, hs):
                    it['h_gpt'] = x
                semg = SemQuery(int(hs[0].numel()), device, feat_gate=rf['feat_gate'])
                Wg = L.init_query_adapter(device)
                og = torch.optim.AdamW(list(semg.parameters()) + list(Wg.parameters()), lr=0.002)
                Hg = torch.stack([it['h_gpt'] for it in fit]).to(device).float()
                for si in range(steps_done):
                    sel = torch.randint(0, Rq.size(0), (min(32, Rq.size(0)),), device=device)
                    qf = F.normalize(Wg(Rq[sel]), dim=-1)
                    a = semg.a(Hg[sel], qf, K, args.tau).unsqueeze(-1)
                    q = F.normalize((1 - a) * qf + a * semg.q(Hg[sel]), dim=-1)
                    lo_ = F.cross_entropy(q @ K.t() / args.tau, Gq[sel])
                    if rf['fp_floor']:
                        lo_ = lo_ + fp_floor_loss(qf, q, K, Gq[sel], args.tau)
                    og.zero_grad(set_to_none=True)
                    lo_.backward()
                    og.step()
                semg.eval()
                with torch.no_grad():
                    ranks = []
                    for it in ev:
                        qf = F.normalize(Wg(it['raw'].unsqueeze(0)), dim=-1)[0]
                        a = semg.a(it['h_gpt'], qf, K, args.tau)
                        af = float(a if a.numel() == 1 else a.mean())
                        q_sem = semg.q(it['h_gpt'].unsqueeze(0))[0]
                        if rf['rrf']:
                            s_fp = (K @ qf).unsqueeze(0)
                            s_sem = (K @ q_sem).unsqueeze(0)
                            s = rrf_scores(s_fp, s_sem)[0]
                        else:
                            q = F.normalize((1 - af) * qf + af * q_sem, dim=-1)
                            s = K @ q
                        ranks.append(1 + int((s > s[it['slot']]).sum()))
                    rr = np.asarray(ranks, dtype=np.float64)
                    gpt = {'top1': float(np.mean(rr == 1)), 'mrr': float(np.mean(1.0 / rr))}
                log(f'gpt2+sem: {json.dumps(gpt)}')
        except Exception as e:
            log(f'  gpt control unavailable: {type(e).__name__}: {e}')

    @torch.no_grad()
    def fp_subset_metrics(items_sub):
        if not items_sub:
            return {'n': 0, 'top1': float('nan'), 'acc_20way': float('nan')}
        wrng = random.Random(SEED + 7)
        ranks, nway = ([], [])
        for it in items_sub:
            q_fp = F.normalize(W_q(it['raw'].unsqueeze(0)), dim=-1)[0]
            sims = K @ q_fp
            gold = int(it['slot'])
            ranks.append(1 + int((sims > sims[gold]).sum()))
            pool = [j for j in wrng.sample(range(K.size(0)), min(20 * 3, K.size(0))) if j != gold][:19]
            nway.append(int(all((float(sims[gold]) >= float(sims[j]) for j in pool))))
        r = np.asarray(ranks, dtype=np.float64)
        return {'n': len(items_sub), 'top1': float(np.mean(r == 1)), 'acc_20way': float(np.mean(nway)) if nway else float('nan')}
    ev_leak = [it for it in ev if it.get('ent_in_query')]
    ev_clean = [it for it in ev if not it.get('ent_in_query')]
    query_entity_diag = {'query_names_entity': query_names_entity, 'eval_n': len(ev), 'ent_in_natural_query_n': len(ev_leak), 'fp_ent_absent': fp_subset_metrics(ev_clean), 'fp_ent_present_leak': fp_subset_metrics(ev_leak)}
    log(f'  query entity diag: {json.dumps(query_entity_diag)}')
    chance = 1.0 / len(vals)
    fp20 = fp_only.get('acc_20way', 0.0)
    fp20_init = fp_frozen.get('acc_20way', fp20)
    sem20 = sem.get('acc_20way', 0.0)
    sh20 = shuf.get('acc_20way', 0.0)
    g_works = sem['top1'] >= 0.3
    g_beats_fp = sem['top1'] >= fp_only['top1'] + 0.1
    g_low_overlap = not math.isnan(sem['top1_low_overlap']) and sem['top1_low_overlap'] >= 0.25
    g_not_lexical = not math.isnan(sem['top1_low_overlap']) and (not math.isnan(sem['top1_high_overlap'])) and (sem['top1_low_overlap'] > 0.0) and (sem['top1_low_overlap'] >= 0.6 * sem['top1_high_overlap'])
    g_signal_fp_20way = fp20 >= 0.05 + 1.0 / 20
    g_signal_sem_20way = sem20 >= 0.05 + 1.0 / 20
    g_sem_harms = sem20 < fp20 - 0.03
    g_sem_neutral = sem20 >= fp20 * 0.9 and fp20 >= 0.12
    g_causal = shuf['top1'] <= max(0.05, chance * 3)
    g_causal_20way = sh20 >= 0.045 and sh20 <= 0.085
    gpt_also_fails = gpt is not None and gpt['top1'] <= max(0.02, fp_only['top1'])
    if g_works and g_beats_fp and g_causal and g_not_lexical and g_low_overlap:
        overall = 'NL_QUERY_OK'
    elif g_works and g_causal and (not g_not_lexical):
        overall = 'NL_QUERY_LEXICAL_ONLY'
    elif g_works and g_causal:
        overall = 'NL_QUERY_PARTIAL'
    elif g_sem_neutral and g_signal_fp_20way and g_causal and g_causal_20way:
        overall = 'NL_QUERY_MIXER_OK'
    elif g_signal_fp_20way and g_causal and g_causal_20way and g_sem_harms:
        overall = 'NL_QUERY_NWAY_FP_ONLY'
    elif g_signal_sem_20way and g_causal and g_causal_20way and (not g_sem_harms) and (fp20 >= 0.12):
        overall = 'NL_QUERY_NWAY_ONLY'
    elif gpt is not None and gpt_also_fails:
        overall = 'NL_QUERY_NO_AT_SCALE'
    elif gpt is None and fp_only['top1'] <= 0.05 and (sem['top1'] <= 0.05):
        overall = 'NL_QUERY_NO_AT_SCALE'
    else:
        overall = 'NL_QUERY_NO'
    out = {'stage': 261, 'overall': overall, 'recipe': rf['name'], 'recipe_flags': rf, 'use_anchor': use_anchor, 'query_names_entity': query_names_entity, 'query_entity_diag': query_entity_diag, 'trunk': trunk_ckpt.name, 'fp_version': L.canonical_fp_version(), 'external_model': args.model or None, 'external_hidden': ext.dim if ext is not None else None, 'steps': steps, 'steps_done': steps_done, 'slots': len(vals), 'exam_slots': n_exam, 'noise_slots': len(vals) - n_exam, 'n_fit': len(fit), 'n_eval': len(ev), 'chance': chance, 'overlap_median': med, **({'rerank_stats': rerank_stats} if rerank_stats else {}), 'gates': {'G_works': g_works, 'G_beats_fp_only': g_beats_fp, 'G_low_overlap_works': g_low_overlap, 'G_not_lexical': g_not_lexical, 'G_tape_causal': g_causal, 'G_tape_causal_20way': g_causal_20way, 'G_signal_fp_20way': g_signal_fp_20way, 'G_signal_sem_20way': g_signal_sem_20way, 'G_sem_harms_fp': g_sem_harms, 'G_sem_neutral_vs_fp': g_sem_neutral}, 'read': {'acc_20way_fp': fp20, 'acc_20way_fp_init_Wq': fp20_init, 'acc_20way_fp_trained_Wq': fp_only.get('acc_20way', 0.0), 'acc_20way_sem': sem20, 'acc_20way_shuffled': sh20, 'sem_over_fp_20way': sem20 / max(fp20, 1e-09), 'blend_alpha_eval': sem.get('alpha', 0.0), 'full_bank_top1_fp': fp_only.get('top1'), 'full_bank_top1_sem': sem.get('top1'), 'full_bank_median_rank_fp': fp_only.get('median_rank'), 'full_bank_median_rank_sem': sem.get('median_rank')}, 'summary': {'fp_only': fp_only, 'fp_only_frozen_Wq': fp_frozen, 'fp_plus_sem': sem, 'shuffled_keys': shuf, 'gpt_control': gpt}, 'gpt_parity': bool(gpt_also_fails) if gpt is not None else None, 'note': 'Full: 353 exam + 4000 wiki noise. Headline NL_QUERY_NO_AT_SCALE (GPT top1 0). acc_20way: fp-only beats chance (~4.4x); fp+sem at high alpha can harm fp — see results/stage261_close.md. G_not_lexical requires top1_low_overlap > 0.', 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    if query_names_entity and rf['name'] == 'baseline':
        dec_path = RES / 'stage261_decision_query_names_entity.json'
        mini_path = RES / 'stage261_mini_query_names_entity.md'
    elif args.model and rf['name'] == 'baseline':
        slug = args.model.replace('/', '_')
        dec_path = RES / f'stage261_decision_{slug}.json'
        mini_path = RES / f'stage261_mini_{slug}.md'
    elif not use_anchor and rf['name'] == 'baseline':
        dec_path = RES / 'stage261_decision_no_anchor.json'
        mini_path = RES / 'stage261_mini_no_anchor.md'
    else:
        dec_path = DECISION if rf['name'] == 'baseline' else RES / f"stage261_decision_{rf['name']}.json"
        mini_path = MINI if rf['name'] == 'baseline' else RES / f"stage261_mini_{rf['name']}.md"
    dec_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    mini_path.write_text(f"# Stage 261 natural-question retrieval ({rf['name']})\n\n**{overall}** slots={len(vals)} eval={len(ev)} chance={chance:.4f}\n\n- top1: fp-only **{fp_only['top1']:.3f}** -> fp+sem **{sem['top1']:.3f}** (shuffled {shuf['top1']:.3f})\n- by overlap: low **{sem['top1_low_overlap']:.3f}** vs high **{sem['top1_high_overlap']:.3f}** (median {med:.3f})\n- 20-way (chance 0.05): fp-only **{fp_only.get('acc_20way', float('nan')):.3f}** (init Wq {fp_frozen.get('acc_20way', float('nan')):.3f}) -> fp+sem **{sem.get('acc_20way', float('nan')):.3f}** (shuffled {shuf.get('acc_20way', float('nan')):.3f})\n- mrr {sem['mrr']:.3f}, median rank {sem['median_rank']:.0f}, blend a {sem['alpha']:.3f}\n" + (f"- matched GPT-2: top1 {gpt['top1']:.3f}\n" if gpt else '- matched GPT-2: not run\n'), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        ckpt = {'W_q': W_q.state_dict(), 'stage': 261, 'recipe': rf['name']}
        if semq is not None:
            ckpt['sem'] = semq.state_dict()
        if rerank is not None:
            ckpt['rerank'] = rerank.state_dict()
        if qkey is not None:
            ckpt['qkey'] = qkey.state_dict()
        torch.save(ckpt, CKPT_OUT)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())