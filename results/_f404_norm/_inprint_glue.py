"""Inprint slot-bias glue — inference API (stage 256). Trunk frozen; W_q + gate + copy mixture."""
from __future__ import annotations
import math
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, SelfModelXL
from _stage194_fp_fact_memory import FpBank
from _retrieval_modes import vote_scores
DEFAULT_CUE = '{S} was appointed director of'
DEFAULT_FACT_TMPL = '{S} was appointed director of {V} in the regional chronicle of 1987 .'
ANCHOR_RE = re.compile('\\b([A-Z][a-z]{2,})\\b')
from _tape_index import VOTES_AUTO_MIN_SLOTS, DEFAULT_RETRIEVE_TOPK
DEFAULT_RETRIEVE_MODE = 'auto'
GLUE_CKPT = Path('checkpoints/stage256_slot_bias.pt')
JOINT_CKPT = Path('checkpoints/stage253_joint_l02.pt')
P1_CKPT = Path('checkpoints/stage191_p1_curve.pt')
from _tape_index import context_words

class RetrieveStats:
    """Counts glue retrieval steps by effective backend (decode/train diagnostics)."""
    __slots__ = ('votes', 'cosine', 'miss')

    def __init__(self) -> None:
        self.votes = 0
        self.cosine = 0
        self.miss = 0

    def record(self, eff: str, hit: object | None) -> None:
        if eff == 'votes':
            self.votes += 1
        elif eff == 'cosine':
            self.cosine += 1
        if hit is None:
            self.miss += 1

    def to_dict(self) -> dict:
        t = self.votes + self.cosine
        return {'votes_steps': self.votes, 'cosine_steps': self.cosine, 'miss': self.miss, 'total_glue_steps': t}

def slot_query_words(text: str, exclude: str | None=None) -> list[str]:
    return context_words(text, exclude=exclude)

class SlotPostings:
    """Word -> slot postings with idf weights (zero-train retrieval index)."""

    def __init__(self, ctxw: list[list[str]], device: torch.device):
        self.postings: dict[str, list[int]] = defaultdict(list)
        for j, ws in enumerate(ctxw):
            for w in ws:
                self.postings[w].append(j)
        self.idf = {w: 1.0 / math.log(2.0 + len(v)) for w, v in self.postings.items()}
        self.device = device
        self.n = len(ctxw)

    @classmethod
    def from_ctxw(cls, ctxw: list[list[str]], device: torch.device) -> SlotPostings:
        return cls(ctxw, device)

    def topk(self, words: list[str], k: int, alive: torch.Tensor | None=None):
        sc = vote_scores(words, self.postings, self.idf)
        if alive is not None:
            sc = {j: v for j, v in sc.items() if j < alive.numel() and bool(alive[j])}
        if not sc:
            return None
        idx = sorted(sc, key=lambda j: -sc[j])[:k]
        v = torch.tensor([sc[j] for j in idx], dtype=torch.float32, device=self.device)
        v = v / v.max().clamp_min(1e-06)
        return (v, torch.tensor(idx, dtype=torch.long, device=self.device))

def resolve_retrieve_mode(mode: str, n_live_slots: int) -> str:
    if mode != 'auto':
        return mode
    return 'votes' if n_live_slots >= VOTES_AUTO_MIN_SLOTS else 'cosine'

@torch.no_grad()
def full_bank_cue_summary(retrieve_mode: str, glue: SlotBias | None, bank: FpBank, tok: Tokenizer, tape: TapeView, facts: list[dict], pad_id: int, *, cue_tmpl: str=DEFAULT_CUE) -> dict:
    """Gold slot rank over all live tape slots at the decode cue (open bank)."""
    ranks: list[int] = []
    n_live = int(tape.alive.sum()) if tape.alive is not None else len(tape.values)
    eff = resolve_retrieve_mode(retrieve_mode, n_live)
    for f in facts:
        cue_ids = [i for i in tok.encode(cue_tmpl.format(S=f['S'])).ids if i != pad_id]
        gold = [j for j, v in enumerate(tape.values) if v == f['value']]
        if not gold:
            continue
        use_votes = eff == 'votes' and tape.postings is not None
        if use_votes:
            words = slot_query_words(tok.decode(cue_ids))
            sc = vote_scores(words, tape.postings.postings, tape.postings.idf)
            gsc = max((sc.get(j, 0.0) for j in gold), default=0.0)
            rank = 1 + sum((1 for v in sc.values() if v > gsc))
        else:
            if glue is None:
                continue
            q = ctx_query(glue, bank, tok, cue_ids, anchor_ids=cue_ids)
            if q is None:
                continue
            sims = tape.K @ q
            if tape.alive is not None:
                sims = sims.masked_fill(~tape.alive, float('-inf'))
            gsim = float(sims[gold].max())
            rank = 1 + int((sims > gsim).sum().item())
        ranks.append(rank)
    if not ranks:
        return {'full_bank_top1': float('nan'), 'full_bank_mrr': float('nan'), 'full_bank_median_rank': float('nan'), 'n': 0}
    r = np.asarray(ranks, dtype=np.float64)
    return {'full_bank_top1': float(np.mean(r == 1)), 'full_bank_mrr': float(np.mean(1.0 / r)), 'full_bank_median_rank': float(np.median(r)), 'n': len(ranks)}

def retrieve_topk(mode: str, glue: SlotBias | None, bank: FpBank, tok: Tokenizer, tape: TapeView, prefix_ids: list[int], cue_ids: list[int] | None, k: int, stats: RetrieveStats | None=None):
    """Unified retrieval: auto picks votes at scale, cosine on small banks."""
    n_live = int(tape.alive.sum()) if tape.alive is not None else len(tape.values)
    eff = resolve_retrieve_mode(mode, n_live)
    if eff == 'votes':
        if tape.postings is None:
            eff = 'cosine'
        else:
            words = slot_query_words(tok.decode(prefix_ids[-60:]))
            hit = tape.postings.topk(words, k, tape.alive)
            if stats is not None:
                stats.record('votes', hit)
            return hit
    if glue is None:
        if stats is not None:
            stats.record('cosine', None)
        return None
    q = ctx_query(glue, bank, tok, prefix_ids, anchor_ids=cue_ids)
    hit = tape.topk(q, k) if q is not None else None
    if stats is not None:
        stats.record('cosine', hit)
    return hit

class SlotBias(nn.Module):
    """Retrieved slots -> copy distribution, mixed into LM logits by a per-step gate."""

    def __init__(self, d_hidden: int, device):
        super().__init__()
        self.W_q = L.init_query_adapter(device)
        self.gate = nn.Sequential(nn.Linear(d_hidden + 4, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.log_tau = nn.Parameter(torch.tensor(-1.5, device=device))

    def trainable(self):
        return list(self.W_q.parameters()) + list(self.gate.parameters()) + [self.log_tau]

    def weights(self, sims: torch.Tensor) -> torch.Tensor:
        return F.softmax(sims / torch.exp(self.log_tau).clamp(0.001, 10.0), dim=-1)

    def g(self, h_t, max_sim: float, mean_topk: float, entropy: float, coverage) -> torch.Tensor:
        cov = coverage if torch.is_tensor(coverage) else torch.tensor(coverage, device=h_t.device)
        feats = torch.stack([torch.tensor(max_sim, device=h_t.device, dtype=h_t.dtype), torch.tensor(mean_topk, device=h_t.device, dtype=h_t.dtype), torch.tensor(entropy, device=h_t.device, dtype=h_t.dtype), cov.to(h_t.dtype).reshape(())])
        return torch.sigmoid(self.gate(torch.cat([h_t, feats], dim=-1))).squeeze(-1)

class TapeView:
    """Read-only slot bank for glue decode."""

    def __init__(self, K: torch.Tensor, values: list[str], tok: Tokenizer, pad_id: int, ctxw: list[list[str]] | None=None):
        self.K = K
        self.values = values
        self.tok_ids = [[i for i in tok.encode(' ' + v).ids if i != pad_id] for v in values]
        self.alive = torch.ones(len(values), dtype=torch.bool, device=K.device)
        self.ctxw = [list(w) for w in ctxw] if ctxw is not None else None
        self.postings = SlotPostings.from_ctxw(self.ctxw, K.device) if self.ctxw else None

    def n_live(self) -> int:
        return int(self.alive.sum())

    def topk(self, q: torch.Tensor, k: int):
        if q is None or not bool(self.alive.any()):
            return None
        sims = self.K @ q
        sims = sims.masked_fill(~self.alive, -10000.0)
        k = min(k, int(self.alive.sum()))
        v, idx = torch.topk(sims, k)
        return (v, idx)

    def copy(self) -> 'TapeView':
        t = TapeView.__new__(TapeView)
        t.K, t.values, t.tok_ids = (self.K, self.values, self.tok_ids)
        t.alive = self.alive.clone()
        t.ctxw = [list(w) for w in self.ctxw] if self.ctxw is not None else None
        t.postings = SlotPostings.from_ctxw(t.ctxw, self.K.device) if t.ctxw is not None else None
        return t

    def reindex(self, j: int, new_ctx_words: list[str]) -> None:
        """Replace slot j's write-context words and rebuild postings."""
        if self.ctxw is None:
            raise ValueError('tape has no ctxw/postings index')
        if j < 0 or j >= len(self.ctxw):
            raise ValueError(f'slot index {j} out of range')
        self.ctxw = [list(w) for w in self.ctxw]
        self.ctxw[j] = list(new_ctx_words)
        self.postings = SlotPostings.from_ctxw(self.ctxw, self.K.device)

    def drop_value(self, value: str) -> int:
        n = 0
        for j, v in enumerate(self.values):
            if v == value and self.alive[j]:
                self.alive[j] = False
                n += 1
        return n

    def shuffled(self, seed: int) -> 'TapeView':
        t = self.copy()
        g = torch.Generator(device='cpu').manual_seed(seed)
        perm = torch.randperm(self.K.size(0), generator=g).to(self.K.device)
        t.K = self.K[perm]
        return t

    def emptied(self) -> 'TapeView':
        t = self.copy()
        t.alive = torch.zeros_like(t.alive)
        return t

    def with_value(self, old: str, new: str, tok: Tokenizer, pad_id: int, *, new_ctx_words: list[str] | None=None) -> 'TapeView':
        """Update a fact: same slot, same KEY, new value — zero gradient steps.

        Keys are written as norm(fp(anchor) + ctx_fp(sentence, exclude=value)), so the value
        never enters its own key; replacing it leaves the key bit-identical and this stays a
        fact update rather than a re-index. Context changes must use ``reindex()``, not this
        method — silent postings drift is not allowed.
        """
        if new_ctx_words is not None:
            raise ValueError('write-context change requires TapeView.reindex(j, new_ctx_words); with_value only replaces the value string')
        t = self.copy()
        t.values = list(self.values)
        t.tok_ids = list(self.tok_ids)
        ids = [i for i in tok.encode(' ' + new).ids if i != pad_id]
        for j, v in enumerate(self.values):
            if v == old:
                t.values[j] = new
                t.tok_ids[j] = ids
        return t

def copy_dist(glue, tape, sims, idx, prefix_ids, V, device):
    w = glue.weights(sims)
    p = torch.zeros(V, device=device, dtype=w.dtype)
    cov = torch.zeros((), device=device, dtype=w.dtype)
    for pos, j in enumerate(idx.tolist()):
        ids = tape.tok_ids[j]
        if not ids:
            continue
        step = 0
        for cut in range(min(len(ids), len(prefix_ids)), 0, -1):
            if prefix_ids[-cut:] == ids[:cut]:
                step = cut
                break
        if step >= len(ids):
            continue
        p = p.index_add(0, torch.tensor([ids[step]], device=device), w[pos].reshape(1))
        cov = cov + w[pos]
    if float(cov) > 1e-06:
        p = p / cov
    return (p, cov)

def mix_logprob(base_logits: torch.Tensor, g: torch.Tensor, p_copy: torch.Tensor, cov) -> torch.Tensor:
    p_lm = F.softmax(base_logits, dim=-1)
    if p_copy is None or float(cov) <= 1e-06:
        return torch.log(p_lm + 1e-09)
    return torch.log((1.0 - g) * p_lm + g * p_copy + 1e-09)

def hidden_and_logits(model: SelfModelXL, char_table, ids: torch.Tensor, pad_id: int):
    pad = ids == pad_id
    arcs = model._arcs(char_table[ids], ids)
    fast = model.fast(arcs, pad_mask=pad)
    slow, _, _ = model.slow(arcs, pad)
    h = torch.cat([fast, slow], dim=-1)
    return (h, model.head(h))

def raw_query(bank: FpBank, tok: Tokenizer, ids: list[int], anchor_ids: list[int] | None=None):
    text = tok.decode(ids[-40:])
    c = bank.ctx_fp(text)
    if c is None:
        return None
    anchors = ANCHOR_RE.findall(tok.decode(anchor_ids) if anchor_ids is not None else text)
    if anchors:
        c = F.normalize(bank.fp([anchors[-1]])[0] + c, dim=-1)
    return c

def ctx_query(glue, bank, tok, ids, anchor_ids=None):
    q = raw_query(bank, tok, ids, anchor_ids)
    if q is None:
        return None
    return F.normalize(glue.W_q(q.unsqueeze(0)), dim=-1)[0]

def build_planted_keys(bank: FpBank, subjects: list[str], values: list[str], sent_tmpl: str=DEFAULT_FACT_TMPL):
    keys, kept_s, kept_v = ([], [], [])
    for S, Vv in zip(subjects, values):
        sent = sent_tmpl.format(S=S, V=Vv)
        kf = bank.fp([S])[0]
        c = bank.ctx_fp(sent, exclude=Vv)
        if c is None:
            continue
        keys.append(F.normalize(kf + c, dim=-1))
        kept_s.append(S)
        kept_v.append(Vv)
    if not keys:
        return (torch.zeros(0, 256), [], [])
    return (torch.stack(keys, 0), kept_s, kept_v)

def load_glue(model: SelfModelXL, device, ckpt_path: Path | None=None) -> SlotBias | None:
    p = ckpt_path or GLUE_CKPT
    if not p.is_file():
        return None
    d = 2 * (model.head.in_features // 2)
    glue = SlotBias(d, device)
    st = torch.load(p, map_location=device, weights_only=False)
    with torch.no_grad():
        glue.W_q.load_state_dict(st['W_q'])
        glue.gate.load_state_dict(st['gate'])
        glue.log_tau.copy_(st['log_tau'].to(device))
    glue.eval()
    return glue

def trunk_ckpt_path() -> Path:
    return JOINT_CKPT if JOINT_CKPT.is_file() else P1_CKPT

@torch.no_grad()
def free_decode_value(glue, model, char_table, tok, bank, tape: TapeView, subject: str, pad_id: int, vocab_size: int, device, *, cue: str=DEFAULT_CUE, k: int=8, max_new: int=6, use_glue: bool=True) -> tuple[str, float]:
    cue_ids = [i for i in tok.encode(cue.format(S=subject)).ids if i != pad_id]
    seq = list(cue_ids)
    gates = []
    for _ in range(max_new):
        ids = torch.tensor([seq[-MAX_ARCS:]], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        base = logits[0, -1]
        score = torch.log(F.softmax(base, -1) + 1e-09)
        if use_glue and glue is not None:
            hit = retrieve_topk(DEFAULT_RETRIEVE_MODE, glue, bank, tok, tape, seq, cue_ids, k)
            if hit is not None:
                sims, idx = hit
                ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
                p_copy, cov = copy_dist(glue, tape, sims, idx, seq, vocab_size, device)
                g_val = glue.g(h[0, -1], float(sims.max()), float(sims.mean()), ent, cov)
                score = mix_logprob(base, g_val, p_copy, cov)
                gates.append(float(g_val))
        nxt = int(score.argmax())
        seq.append(nxt)
    text = tok.decode(seq[len(cue_ids):]).strip()
    g_mean = float(sum(gates) / len(gates)) if gates else float('nan')
    return (text, g_mean)

def value_exact_match(generated: str, gold: str) -> bool:
    if not generated:
        return False
    return generated.strip().split(' ')[0].strip(' .,;:') == gold