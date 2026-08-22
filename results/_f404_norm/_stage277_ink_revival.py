"""
Stage 277 — Are word votes just the τ→1 limit of the ink, and is the region τ<1 worth anything?

263 and 261f left the retrieval question in an odd place. Votes beat the context mean 7x on the
open bank (top1 0.246 vs 0.034, median rank 76.5 vs 1036.5), so the shipped dense channel was
abandoned. But votes are exact string match, and exact match has one failure that no amount of
idf fixes: SILENCE. When no query word appears in the gold slot's write context the gold scores
zero, and a zero cannot be ranked - 266 showed what happens when you pretend it can.

The claim this stage tests is that "replace votes with a better vector" is the wrong framing,
because in the limit a perfectly discriminative vector IS a one-hot hash, i.e. IS votes. The
right framing is that votes are one endpoint of a family:

    score(q, slot) = Σ_{w∈q} idf(w) · max_{c∈slot} k(fp(w), fp(c))
    k(x, y) = relu(cos(x, y) − τ)^p

At τ→1 the kernel fires only on an identical fingerprint, the max collapses to an indicator, and
the expression is EXACTLY votes with idf - because content() dedups, so sum and max coincide.
At τ<1 the same expression is votes with a soft kernel: a declension, a typo, a compound form
lands near its neighbour instead of landing nowhere. Nothing is trained; τ is one scalar.

Four scorers over one bank, one item list, one seed:

    votes     exact postings + idf                       the incumbent (261f, 263)
    sum       cos(q, normalize(Σ idf(w)·fp(w)))          idf-weighted averaging, the honest
                                                         version of the mean 261f killed
    sum_sif   the same with the top principal component removed (all-but-the-top)
    maxsim    late interaction, the kernel above          votes generalised, zero trained params
    hybrid    votes + α·maxsim                            the deployable form if neither wins alone

and one diagnostic that has been owed since 264: split the queries where votes go SILENT on the
gold into those with a near-string route to the gold (shared 4-gram or edit distance ≤ 2) and
those with none, then ask what maxsim does on each half. The first half is the part the ink can
close by construction. The second half is paraphrase with no shared characters, and a character
encoder cannot close it - saying so with a number is the point.

Ranks use 266's correction throughout: a gold with zero score ranks LAST, not first, and n-way
is strict (gold must beat every distractor outright, ties are misses).

The first run answered INK_NO, and two things about that answer were wrong. Every gate read top1
on a 4338-slot open bank, which is the wrong instrument for a channel whose job is the tail:
hybrid was beating votes on the strict 20-way exam clean (0.432 vs 0.417) AND under character
noise (0.393 vs 0.352), and under that noise votes' median rank collapsed to the 4338 floor
while hybrid held 635. That IS 204's property; the gate simply was not looking at it. Each claim
is now gated on the metric the claim is about.

The second thing the run exposed is geometry. tau=0 and tau=0.3 scored identically, which can
only mean almost no pair of fingerprints sits below cos 0.3 - the ink is crowded into a narrow
cone and there is no range for a threshold to act on. sum_sif removed the common direction from
the SLOT matrix; the crowding is in the WORD matrix and was never touched. --whiten-fp K removes
the top-K principal directions of the vocabulary (or whitens its covariance with --whiten-mode
zca), fitted to the ink's own shape and to no label, and --cos-hist prints the two distributions
that decide whether any tau can work: cosine between unrelated words, and cosine to the nearest
OTHER word, which is what max_c returns on a miss.

That histogram came back and answered the question, though not in favour of whitening. Unrelated
words sit at cos 0.34 - the space is not uniformly crowded - but the NEAREST other word sits at
0.913 median and at exactly 1.0 for the top 5% of the vocabulary. A cosine of 1 between two
different strings is a collision, and whitening is a linear map: it cannot separate two vectors
that are equal.

The cause is upstream of everything this repo has measured. ArcEncoder pools its character
embeddings BEFORE the feed-forward and has no positional code at all, so fp(w) is a function of
w's normalised character histogram and of nothing else. listen and silent are bit-identical, and
so are ab and abab. That single fact explains 261's low-overlap 0.000, QUERY_MUST_BE_WORDS, the
collapse of the ctx_fp mean into near-constancy, and why maxsim at tau=0.6 fires on every slot:
the max over a slot's sixteen words clears 0.6 by chance alone. It also explains the one thing
the ink was always good at - a typo barely moves a histogram - and shows that the robustness and
the non-discrimination were never two properties, only one.

--fp-ngram N feeds the SAME frozen arc_enc overlapping character N-grams and pools those. Each
gram is still internally a bag, but the multiset of grams is not, so order returns without a
weight moving and with the arc_enc hash gate still passing. A typo corrupts N grams out of
len(w)-N+1, so 204's property survives. G_fp_no_collisions is the gate that says whether it
worked.

--fp-ngram 3 did half of that. Collisions fell from 0.098 to 0.030, so order genuinely returned,
but every cosine moved UP - unrelated pairs 0.329 -> 0.438, nearest 0.913 -> 0.942 - and the
separation shrank from 0.304 to 0.260. Pooling the grams is the same averaging one level down:
the mean of k vectors shrinks variance and drags everything toward the common direction. So two
last measurements before the ink is filed where it belongs.

--gram-maxsim N never averages a vector. A word is a SET of gram vectors; similarity is the mean
over the query's grams of the max over the candidate's grams, and the outer mean is over scalars
taken AFTER the maxima, which is not the same operation as averaging first and comparing once.

The spectrum print settles the width question for good. pooled = mean(emb(chars)) is a convex
combination of the alphabet's embedding rows, so its rank is bounded by the charset and not by
d, and ff is a fixed smooth map - it can fold that manifold but cannot add information to it. If
95% of the variance sits in a few dozen directions then 256 is already an order of magnitude
more room than the ink uses, and no width changes what it can tell apart.

  python _stage277_ink_revival.py --cos-hist                       # the shipped ink + spectrum
  python _stage277_ink_revival.py --cos-hist --gram-maxsim 3       # gram space, no averaging
  python _stage277_ink_revival.py [--smoke] [--gram-maxsim 3] [--tau 0.6] [--whiten-fp 8]
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
from _stage191_night import SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage261_nl_query import collect, ctx_words, jaccard
from _stage261f_word_votes import content, typo
RES = Path('results')
DECISION = RES / 'stage277_decision.json'
MINI = RES / 'stage277_mini.md'
LOG = RES / '_stage277_log.txt'
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 2770

def log(m: str) -> None:
    line = m if m.endswith('\n') else m + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def rank_of(v: np.ndarray, cid: int) -> int:
    """A gold that scored nothing ranks LAST. 266 read 0.477 top1 off 71 empty answers because
    this line said `1 + (v > gold).sum()` and every tie sat above the gold."""
    g = float(v[cid])
    if g <= 0.0:
        return int(v.shape[0])
    return 1 + int((v > g).sum())

def nway_strict(v: np.ndarray, cid: int, pool: list[int]) -> int:
    """Pessimistic n-way: the gold must beat every distractor OUTRIGHT. A silent gold loses."""
    g = float(v[cid])
    if g <= 0.0:
        return 0
    return int(all((g > float(v[j]) for j in pool)))

def report(name: str, vecs: list[np.ndarray], items: list[dict], pools: list[list[int]], med_overlap: float, n_way: int) -> dict:
    ranks, nway, lo, hi = ([], [], [], [])
    for v, it, pool in zip(vecs, items, pools):
        r = rank_of(v, it['cid'])
        ranks.append(r)
        nway.append(nway_strict(v, it['cid'], pool))
        (hi if it['overlap'] > med_overlap else lo).append(int(r == 1))
    r = np.asarray(ranks, dtype=np.float64)
    out = {'top1': float(np.mean(r == 1)), 'mrr': float(np.mean(1.0 / r)), 'median_rank': float(np.median(r)), f'acc_{n_way}way_strict': float(np.mean(nway)), 'top1_low_overlap': float(np.mean(lo)) if lo else float('nan'), 'top1_high_overlap': float(np.mean(hi)) if hi else float('nan'), 'n': len(ranks)}
    log(f'  [{name}] ' + json.dumps(out))
    return out

def shares_ngram(a: str, b: str, n: int=4) -> bool:
    if len(a) < n or len(b) < n:
        return a == b
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    return any((b[i:i + n] in ga for i in range(len(b) - n + 1)))

def edit_le(a: str, b: str, k: int=2) -> bool:
    """Bounded Levenshtein: returns True iff distance ≤ k. Bails out as soon as it cannot be."""
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > k:
            return False
        prev = cur
    return prev[-1] <= k

def grams(w: str, n: int) -> list[str]:
    return [w[i:i + n] for i in range(len(w) - n + 1)] if len(w) >= n else [w]

class Ink:
    """fp(w), with the option of writing the word in n-grams instead of in letters.

    ArcEncoder pools its character embeddings BEFORE the feed-forward and carries no positional
    code (_stage177_curve_bpe.py:183-187), so fp(w) is a function of w's normalised character
    histogram and nothing else. listen and silent are bit-identical; so are ab and abab. That is
    what the first histogram found - a nearest-neighbour cosine whose p95 is exactly 1.0 is not
    crowding, it is collisions, and no linear whitening separates equal vectors.

    It also explains why the ink is robust to typos: one changed letter barely moves a histogram.
    Robustness and non-discrimination are the same property here, not two.

    Feeding the SAME frozen encoder overlapping n-grams and pooling those restores order without
    touching a weight - arc_enc stays byte-identical and its hash gate still passes. Each gram is
    still internally a bag, but the multiset of grams is not: listen gives {lis,ist,ste,ten} and
    silent gives {sil,ile,len,ent}. A typo still only corrupts n grams out of len(w)-n+1, so the
    204 property survives.
    """

    def __init__(self, bank: FpBank, n: int, device):
        self.bank, self.n, self.device = (bank, n, device)
        self.cache: dict[str, torch.Tensor] = {}

    def __call__(self, words: list[str]) -> torch.Tensor:
        if self.n <= 0:
            return self.bank.fp(words).to(self.device).float()
        todo = [w for w in words if w not in self.cache]
        if todo:
            flat, spans = ([], [])
            for w in todo:
                g = grams(w, self.n)
                spans.append((len(flat), len(g)))
                flat.extend(g)
            E = self.bank.fp(flat).to(self.device).float()
            for w, (a, ln) in zip(todo, spans):
                self.cache[w] = F.normalize(E[a:a + ln].mean(0), dim=-1)
        return torch.stack([self.cache[w] for w in words], 0)

@torch.no_grad()
def spectrum(W: torch.Tensor) -> dict:
    """How many dimensions does the ink actually occupy?

    pooled = mean(emb(chars)) is a convex combination of the alphabet's embedding rows, so its
    rank is bounded by the CHARSET, not by d, and ff is a fixed smooth map that can fold that
    manifold but cannot add information to it. If 95% of the variance lives in ~30 directions
    then d=256 is already an order of magnitude more room than the ink uses, and no width - 64,
    1024, anything - changes what it can tell apart. This prints the number that settles it.
    """
    W0 = (W - W.mean(0, keepdim=True)).double()
    ev = torch.linalg.svdvals(W0) ** 2
    part = ev / ev.sum()
    c = torch.cumsum(part, 0)
    out = {'d': int(W.size(1)), 'top1_var_frac': float(part[0])}
    for t in (0.9, 0.95, 0.99):
        out[f'dims_{int(t * 100)}pct'] = int((c < t).sum()) + 1
    out['participation_ratio'] = float(1.0 / (part ** 2).sum())
    return out

def apply_whiten(X: torch.Tensor, mu, M) -> torch.Tensor:
    Y = X - mu
    return F.normalize(Y if M is None else Y @ M, dim=-1)

class GramIndex:
    """MaxSim one level down: a word is a SET of gram vectors, never their mean.

    Averaging the grams cut the collision rate three-fold but pushed every cosine up - the mean
    of k vectors shrinks variance and drags everything toward the common direction, which is the
    same disease the whole-word recipe has, one level lower. This path never averages a vector:

        sim(q, c)      = mean over q's grams of  max over c's grams of  cos
        score(q, slot) = sum_q idf(q) * max over the slot's words of sim(q, c)

    The outer mean is over SCALARS, after the maxima, which is a different operation from
    averaging the vectors first and comparing once.
    """

    def __init__(self, vocab: list[str], bank: FpBank, n: int, device, whiten_k: int, whiten_mode: str):
        self.n, self.bank, self.device = (n, bank, device)
        gidx: dict[str, int] = {}
        ww, wg = ([], [])
        for wi, w in enumerate(vocab):
            for g in grams(w, n):
                ww.append(wi)
                wg.append(gidx.setdefault(g, len(gidx)))
        self.gvocab = sorted(gidx, key=gidx.get)
        self.wg_word = torch.tensor(ww, dtype=torch.long, device=device)
        self.wg_gram = torch.tensor(wg, dtype=torch.long, device=device)
        self.n_vocab = len(vocab)
        raw = F.normalize(bank.fp(self.gvocab).to(device).float(), dim=-1)
        self.raw_cos = cos_stats(raw)
        self.spectrum = spectrum(raw)
        self.mu, self.M = fit_whiten(raw, whiten_k, whiten_mode)
        self.G = apply_whiten(raw, self.mu, self.M)
        self.cos = cos_stats(self.G) if whiten_k else self.raw_cos
        self._cache: dict[str, torch.Tensor] = {}

    def wordsim(self, w: str) -> torch.Tensor:
        """Similarity of one query word to every vocabulary word, [n_vocab]."""
        hit = self._cache.get(w)
        if hit is not None:
            return hit
        Q = apply_whiten(F.normalize(self.bank.fp(grams(w, self.n)).to(self.device).float(), dim=-1), self.mu, self.M)
        S = self.G @ Q.t()
        out = torch.zeros(self.n_vocab, dtype=torch.float32, device=self.device)
        seg = torch.empty(self.n_vocab, dtype=torch.float32, device=self.device)
        for j in range(S.size(1)):
            seg.fill_(-1.0)
            seg.scatter_reduce_(0, self.wg_word, S[self.wg_gram, j], reduce='amax')
            out += seg
        out /= S.size(1)
        if len(self._cache) < 200000:
            self._cache[w] = out
        return out

def fit_whiten(W: torch.Tensor, k: int, mode: str):
    """Fitted on the vocabulary matrix, applied to queries too. Returns (mean, [d,d] or None).

    abtt  remove the top-k principal directions - the common component every fingerprint shares
    zca   whiten the whole covariance, so no direction carries more variance than any other
    Both are zero-train: nothing here is fitted to a label, only to the shape of the ink itself.
    """
    mu = W.mean(0, keepdim=True)
    if k <= 0:
        return (mu * 0.0, None)
    W0 = W - mu
    if mode == 'zca':
        C = W0.t() @ W0 / max(1, W0.size(0) - 1)
        ev, U = torch.linalg.eigh(C.double())
        ev = ev.clamp_min(1e-08)
        return (mu, (U @ torch.diag(ev.rsqrt()) @ U.t()).to(W.dtype))
    Vh = torch.linalg.svd(W0, full_matrices=False).Vh[:k]
    return (mu, torch.eye(W.size(1), device=W.device, dtype=W.dtype) - Vh.t() @ Vh)

@torch.no_grad()
def cos_stats(W: torch.Tensor, n_pairs: int=200000, seed: int=11) -> dict:
    """Two distributions, and the second is the one that decides whether a threshold can work.

    random_pair  cos between two unrelated words - where the kernel's noise floor sits
    nearest      cos to the closest OTHER word - what max_c actually returns on a miss

    If the nearest-neighbour cosine is ~1 for everything, max_c saturates and no tau separates a
    morphological variant from an unrelated word: the kernel is thresholding on nothing.
    """
    n, d = W.shape
    g = torch.Generator(device='cpu').manual_seed(seed)
    a = torch.randint(0, n, (n_pairs,), generator=g).to(W.device)
    b = torch.randint(0, n, (n_pairs,), generator=g).to(W.device)
    keep = a != b
    rp = (W[a[keep]] * W[b[keep]]).sum(-1)
    m = min(n, 4000)
    sel = torch.randperm(n, generator=g)[:m].to(W.device)
    nn_parts = []
    for a0 in range(0, m, 512):
        s = sel[a0:a0 + 512]
        S = W[s] @ W.t()
        S[torch.arange(s.numel(), device=W.device), s] = -2.0
        nn_parts.append(S.max(dim=1).values)
    nn = torch.cat(nn_parts)
    q = [1, 5, 25, 50, 75, 95, 99]

    def pct(x):
        v = torch.quantile(x.float(), torch.tensor([p / 100 for p in q], device=x.device))
        return {'mean': float(x.mean()), 'std': float(x.std()), **{f'p{p}': float(t) for p, t in zip(q, v)}}
    return {'random_pair': pct(rp), 'nearest_other': pct(nn), 'n_pairs': int(keep.sum()), 'collision_rate': float((nn >= 0.9999).float().mean()), 'n_words_sampled': int(m)}

def near_string(qwords: list[str], slot_words: list[str]) -> bool:
    for q in qwords:
        for c in slot_words:
            if shares_ngram(q, c) or edit_le(q, c):
                return True
    return False

class Index:
    """One flat posting list shared by every scorer, so an arm can never differ by its data.

    pos_word[i] / pos_slot[i] is the i-th (word, slot) incidence. votes, maxsim and the two dense
    arms are all functions of these two arrays plus a kernel.
    """

    def __init__(self, slot_ws: list[list[str]], bank: FpBank, device, chunk: int=20000, whiten_k: int=0, whiten_mode: str='abtt', ngram: int=0, gram_maxsim: int=0):
        self.device = device
        self.n_slots = len(slot_ws)
        self.slot_ws = slot_ws
        postings: dict[str, list[int]] = defaultdict(list)
        for cid, ws in enumerate(slot_ws):
            for w in ws:
                postings[w].append(cid)
        self.postings = postings
        self.vocab = sorted(postings)
        self.widx = {w: i for i, w in enumerate(self.vocab)}
        self.idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in self.vocab}
        self.idf_oov = 1.0 / math.log(2.0)
        pw, ps = ([], [])
        for cid, ws in enumerate(slot_ws):
            for w in ws:
                pw.append(self.widx[w])
                ps.append(cid)
        self.pos_word = torch.tensor(pw, dtype=torch.long, device=device)
        self.pos_slot = torch.tensor(ps, dtype=torch.long, device=device)
        self.n_post = len(pw)
        self.bank = bank
        self.chunk = chunk
        self.ngram = ngram
        self.ink = Ink(bank, ngram, device)
        self._dense: dict[bool, tuple] = {}
        raw = F.normalize(self.ink(self.vocab), dim=-1)
        self.raw_cos = cos_stats(raw)
        self.mu, self.M = fit_whiten(raw, whiten_k, whiten_mode)
        self.whiten_k, self.whiten_mode = (whiten_k, whiten_mode)
        self.Wfp = self._apply(raw)
        self.cos = cos_stats(self.Wfp) if whiten_k else self.raw_cos
        self.spectrum = spectrum(raw)
        self.idf_vec = torch.tensor([self.idf[w] for w in self.vocab], dtype=torch.float32, device=device)
        self.gram = None
        if gram_maxsim > 0:
            self.gram = GramIndex(self.vocab, bank, gram_maxsim, device, whiten_k, whiten_mode)
            self.raw_cos, self.cos = (self.gram.raw_cos, self.gram.cos)
            self.spectrum = self.gram.spectrum

    def _apply(self, X: torch.Tensor) -> torch.Tensor:
        Y = X - self.mu
        if self.M is not None:
            Y = Y @ self.M
        return F.normalize(Y, dim=-1)

    def embed(self, words: list[str]) -> torch.Tensor:
        """Query words take the SAME transform the vocabulary was fitted with - an OOV form has
        to land in the whitened space or the kernel compares two different geometries."""
        return self._apply(F.normalize(self.ink(words), dim=-1))

    def votes(self, qwords: list[str]) -> np.ndarray:
        v = np.zeros(self.n_slots, dtype=np.float64)
        for w in qwords:
            if w in self.postings:
                s = self.idf[w]
                for cid in self.postings[w]:
                    v[cid] += s
        return v

    def maxsim(self, qwords: list[str], tau: float, p: float) -> np.ndarray:
        if not qwords:
            return np.zeros(self.n_slots, dtype=np.float64)
        if self.gram is not None:
            cols = torch.stack([self.gram.wordsim(w) for w in qwords], 1)
        else:
            cols = self.Wfp @ self.embed(qwords).t()
        K = torch.clamp(cols - tau, min=0.0)
        if p != 1.0:
            K = K.pow(p)
        out = torch.zeros(self.n_slots, dtype=torch.float32, device=self.device)
        seg = torch.empty(self.n_slots, dtype=torch.float32, device=self.device)
        for i, w in enumerate(qwords):
            seg.zero_()
            seg.scatter_reduce_(0, self.pos_slot, K[self.pos_word, i], reduce='amax')
            out += self.idf.get(w, self.idf_oov) * seg
        return out.double().cpu().numpy()

    def _dense_bank(self, sif: bool):
        if sif in self._dense:
            return self._dense[sif]
        S = torch.zeros(self.n_slots, self.Wfp.size(1), dtype=torch.float32, device=self.device)
        for a in range(0, self.n_post, self.chunk):
            b = min(a + self.chunk, self.n_post)
            wi = self.pos_word[a:b]
            S.index_add_(0, self.pos_slot[a:b], self.Wfp[wi] * self.idf_vec[wi].unsqueeze(1))
        mu = S.mean(0, keepdim=True)
        u = None
        if sif:
            S0 = S - mu
            u = torch.linalg.svd(S0, full_matrices=False).Vh[0]
            S = S0 - (S0 @ u).unsqueeze(1) * u.unsqueeze(0)
        self._dense[sif] = (F.normalize(S, dim=-1), mu, u)
        return self._dense[sif]

    def summed(self, qwords: list[str], sif: bool) -> np.ndarray:
        S, mu, u = self._dense_bank(sif)
        if not qwords:
            return np.zeros(self.n_slots, dtype=np.float64)
        Qw = self.embed(qwords)
        wts = torch.tensor([self.idf.get(w, self.idf_oov) for w in qwords], dtype=torch.float32, device=self.device).unsqueeze(1)
        q = (Qw * wts).sum(0, keepdim=True)
        if sif:
            q = q - mu
            q = q - (q @ u).unsqueeze(1) * u.unsqueeze(0)
        q = F.normalize(q, dim=-1)
        return (S @ q.t()).squeeze(1).double().cpu().numpy()

    def repointed(self, perm: list[int]):
        """Popularity floor: every posting keeps its word and its COUNT, only the slot it points
        at is permuted. A well-connected slot still collects hits, so this is a floor on
        connectivity rather than a random floor."""
        other = Index.__new__(Index)
        other.__dict__.update(self.__dict__)
        p = torch.tensor(perm, dtype=torch.long, device=self.device)
        other.pos_slot = p[self.pos_slot]
        other.postings = {w: [perm[c] for c in v] for w, v in self.postings.items()}
        other._dense = {}
        return other

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--entities', type=int, default=0)
    ap.add_argument('--distractor-entities', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.6, help='kernel threshold of the headline arm')
    ap.add_argument('--kernel-p', type=float, default=1.0)
    ap.add_argument('--tau-sweep', type=str, default='0.0,0.3,0.5,0.6,0.7,0.8,0.9,0.999', help='the curve that shows votes as the tau->1 endpoint of the same kernel')
    ap.add_argument('--alpha', type=float, default=0.25, help='hybrid = votes + alpha * maxsim')
    ap.add_argument('--whiten-fp', type=int, default=0, metavar='K', help="de-crowd the WORD fingerprint matrix before the kernel: remove the top-K principal directions (abtt) or whiten the covariance (zca). 0 = off. Zero-train - fitted to the ink's own shape, never to a label.")
    ap.add_argument('--whiten-mode', choices=['abtt', 'zca'], default='abtt')
    ap.add_argument('--fp-ngram', type=int, default=0, metavar='N', help='write each word as overlapping character N-grams through the SAME frozen arc_enc instead of as one bag of letters. 0 = the shipped recipe. N=3 is the first setting where an anagram stops colliding.')
    ap.add_argument('--gram-maxsim', type=int, default=0, metavar='N', help='score at the GRAM level and never average a vector: a word is a set of N-gram vectors, similarity is mean-over-query-grams of max-over-candidate-grams. Supersedes --fp-ngram for the kernel arm.')
    ap.add_argument('--cos-hist', action='store_true', help='print the cosine histogram of the word matrix and exit without scoring')
    ap.add_argument('--n-way', type=int, default=20)
    ap.add_argument('--typo-rate', type=float, default=0.15)
    ap.add_argument('--no-typo-arm', action='store_true')
    args = ap.parse_args()
    LOG.write_text('', encoding='utf-8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_ent = args.entities or (60 if args.smoke else 400)
    n_dist = args.distractor_entities or (400 if args.smoke else 4000)
    max_lines = 3000 if args.smoke else 25000
    taus = [float(x) for x in args.tau_sweep.split(',') if x.strip()]
    log(f'Stage277 ink revival start {datetime.now(timezone.utc).isoformat()} device={device} tau={args.tau} p={args.kernel_p} alpha={args.alpha}')
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    bank = FpBank(model, stoi, device)
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
    slot_ws: list[list[str]] = []
    values: list[str] = []
    items: list[dict] = []
    for e in ents:
        occ = cands[e]
        a, b = (occ[0], occ[1])
        wctx = a['line'][max(0, a['start'] - 140):min(len(a['line']), a['end'] + 140)]
        qtext = b['line'][max(0, b['start'] - 200):b['start']].strip()
        ws = content(wctx, exclude=e)
        qs = content(qtext, exclude=e)
        if len(ws) < 4 or len(qs) < 4:
            continue
        cid = len(values)
        values.append(e)
        slot_ws.append(ws)
        items.append({'ent': e, 'cid': cid, 'qwords': qs, 'overlap': jaccard(ctx_words(wctx, e), ctx_words(qtext, e))})
    n_exam = len(values)
    used = set(values)
    for ln in lines:
        if len(values) >= n_exam + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = (max(0, m.start() - 140), min(len(ln), m.end() + 140))
            ws = content(ln[lo:hi], exclude=e)
            if len(ws) < 4:
                continue
            values.append(e)
            slot_ws.append(ws)
            used.add(e)
            if len(values) >= n_exam + n_dist:
                break
    if len(items) < 16:
        log('  not enough usable pairs')
        return 1
    idx = Index(slot_ws, bank, device, whiten_k=args.whiten_fp, whiten_mode=args.whiten_mode, ngram=args.fp_ngram, gram_maxsim=args.gram_maxsim)
    med = float(np.median([it['overlap'] for it in items]))
    log(f'  slots={idx.n_slots} ({n_exam} asked + {idx.n_slots - n_exam} distractor) | vocab={len(idx.vocab)} postings={idx.n_post} | eval={len(items)} overlap median={med:.3f} | ({time.time() - t0:.0f}s)')
    unit = 'whole word (shipped)' if not args.fp_ngram else f'{args.fp_ngram}-gram mean'
    if args.gram_maxsim:
        unit = f'{args.gram_maxsim}-gram SET (no vector averaging)'
    log(f'  fp unit = {unit}' + (f' | gram vocab={len(idx.gram.gvocab)}' if idx.gram is not None else ''))
    log(f'  fp spectrum {json.dumps(idx.spectrum)}')
    log(f"  fp cos raw       random_pair {json.dumps(idx.raw_cos['random_pair'])}")
    log(f"  fp cos raw       nearest     {json.dumps(idx.raw_cos['nearest_other'])}")
    log(f"  fp collisions (nearest cos >= 0.9999): {idx.raw_cos['collision_rate']:.4f} of {idx.raw_cos['n_words_sampled']} sampled words")
    if args.whiten_fp:
        log(f"  fp cos whitened  random_pair {json.dumps(idx.cos['random_pair'])} ({args.whiten_mode} k={args.whiten_fp})")
        log(f"  fp cos whitened  nearest     {json.dumps(idx.cos['nearest_other'])}")
    if args.cos_hist:
        log('  --cos-hist: geometry only, no scoring')
        return 0
    wrng = random.Random(SEED + 5)
    pools = [[j for j in wrng.sample(range(idx.n_slots), min(args.n_way * 3, idx.n_slots)) if j != it['cid']][:args.n_way - 1] for it in items]
    V_votes = [idx.votes(it['qwords']) for it in items]
    r_votes = report('votes', V_votes, items, pools, med, args.n_way)
    V_max = [idx.maxsim(it['qwords'], args.tau, args.kernel_p) for it in items]
    r_max = report(f'maxsim tau={args.tau}', V_max, items, pools, med, args.n_way)
    r_sum = report('sum', [idx.summed(it['qwords'], False) for it in items], items, pools, med, args.n_way)
    r_sif = report('sum_sif', [idx.summed(it['qwords'], True) for it in items], items, pools, med, args.n_way)
    vmax = max((float(v.max()) for v in V_votes if v.size), default=1.0) or 1.0
    mmax = max((float(v.max()) for v in V_max if v.size), default=1.0) or 1.0
    V_hyb = [v + args.alpha * (vmax / mmax) * m for v, m in zip(V_votes, V_max)]
    r_hyb = report(f'hybrid a={args.alpha}', V_hyb, items, pools, med, args.n_way)
    sweep = {}
    for t in taus:
        vt = [idx.maxsim(it['qwords'], t, args.kernel_p) for it in items]
        rr = np.asarray([rank_of(v, it['cid']) for v, it in zip(vt, items)], dtype=np.float64)
        sweep[f'{t:g}'] = {'top1': float(np.mean(rr == 1)), 'median_rank': float(np.median(rr)), 'mrr': float(np.mean(1.0 / rr))}
        log(f'  [sweep tau={t:g}] ' + json.dumps(sweep[f'{t:g}']))
    t_hi = f'{max(taus):g}'
    reduces = abs(sweep[t_hi]['top1'] - r_votes['top1']) <= 0.02 if taus else False
    sil = [k for k, (v, it) in enumerate(zip(V_votes, items)) if v[it['cid']] <= 0.0]
    near = [k for k in sil if near_string(items[k]['qwords'], slot_ws[items[k]['cid']])]
    near_set = set(near)
    pure = [k for k in sil if k not in near_set]

    def sub(name, ks):
        if not ks:
            return {'n': 0}
        rm = np.asarray([rank_of(V_max[k], items[k]['cid']) for k in ks], dtype=np.float64)
        rh = np.asarray([rank_of(V_hyb[k], items[k]['cid']) for k in ks], dtype=np.float64)
        o = {'n': len(ks), 'maxsim_top1': float(np.mean(rm == 1)), 'maxsim_median_rank': float(np.median(rm)), 'maxsim_top10': float(np.mean(rm <= 10)), 'hybrid_top1': float(np.mean(rh == 1)), 'votes_top1': 0.0, 'votes_median_rank': float(idx.n_slots)}
        log(f'  [silent/{name}] ' + json.dumps(o))
        return o
    silence = {'n_queries': len(items), 'n_votes_silent': len(sil), 'frac_votes_silent': len(sil) / max(1, len(items)), 'near_string': sub('near_string', near), 'purely_semantic': sub('purely_semantic', pure)}
    perm = list(range(idx.n_slots))
    random.Random(SEED + 7).shuffle(perm)
    shuf = idx.repointed(perm)
    fl_v = [shuf.votes(it['qwords']) for it in items]
    fl_m = [shuf.maxsim(it['qwords'], args.tau, args.kernel_p) for it in items]
    floor = {'votes_top1': float(np.mean([rank_of(v, it['cid']) == 1 for v, it in zip(fl_v, items)])), 'maxsim_top1': float(np.mean([rank_of(v, it['cid']) == 1 for v, it in zip(fl_m, items)]))}
    log('  [popularity floor] ' + json.dumps(floor))
    noise = {}
    if not args.no_typo_arm and args.typo_rate > 0:
        nrng = random.Random(SEED + 3)
        qn = [[typo(w, args.typo_rate, nrng) for w in it['qwords']] for it in items]
        nv = [idx.votes(q) for q in qn]
        nm = [idx.maxsim(q, args.tau, args.kernel_p) for q in qn]
        nh = [v + args.alpha * (vmax / mmax) * m for v, m in zip(nv, nm)]
        noise = {'typo_rate': args.typo_rate, 'votes': report('votes+typo', nv, items, pools, med, args.n_way), 'maxsim': report('maxsim+typo', nm, items, pools, med, args.n_way), 'hybrid': report('hybrid+typo', nh, items, pools, med, args.n_way)}
    nw = f'acc_{args.n_way}way_strict'
    g_causal = floor['maxsim_top1'] <= 0.02 and floor['votes_top1'] <= 0.02
    g_reduces = bool(reduces)
    g_beats_votes = r_max['top1'] >= r_votes['top1'] + 0.02
    g_hybrid_beats = r_hyb['top1'] >= r_votes['top1'] + 0.02 or r_hyb[nw] >= r_votes[nw] + 0.01
    ns_med = silence['near_string'].get('maxsim_median_rank', float(idx.n_slots))
    g_not_silent = silence['near_string'].get('n', 0) > 0 and ns_med <= 0.75 * idx.n_slots
    g_near_closed = silence['near_string'].get('maxsim_top10', 0.0) >= 0.2
    g_sum_still_loses = max(r_sum['top1'], r_sif['top1']) < r_max['top1']
    g_typo = not noise or (noise['maxsim']['median_rank'] < noise['votes']['median_rank'] or noise['hybrid'][nw] > noise['votes'][nw])
    g_range = idx.cos['nearest_other']['p50'] < 0.95 and idx.cos['random_pair']['p95'] < 0.8
    g_no_collisions = idx.cos['collision_rate'] <= 0.01
    if not g_causal:
        overall = 'INK_INVALID_FLOOR'
    elif g_beats_votes and g_reduces and g_not_silent:
        overall = 'INK_REPLACES_VOTES'
    elif g_hybrid_beats and g_typo:
        overall = 'INK_FILLS_SILENCE'
    elif g_not_silent or g_typo:
        overall = 'INK_SIGNAL_ONLY'
    elif not (g_range and g_no_collisions):
        overall = 'INK_CROWDED'
    else:
        overall = 'INK_NO'
    out = {'stage': 277, 'overall': overall, 'trained_parameters': 0, 'tau': args.tau, 'kernel_p': args.kernel_p, 'alpha': args.alpha, 'n_way': args.n_way, 'whiten_fp': args.whiten_fp, 'whiten_mode': args.whiten_mode, 'fp_ngram': args.fp_ngram, 'gram_maxsim': args.gram_maxsim, 'fp_unit': unit, 'slots': idx.n_slots, 'asked': n_exam, 'distractors': idx.n_slots - n_exam, 'vocab': len(idx.vocab), 'postings': idx.n_post, 'overlap_median': med, 'gates': {'G_causal_popularity_floor': g_causal, 'G_kernel_reduces_to_votes': g_reduces, 'G_maxsim_beats_votes': g_beats_votes, 'G_hybrid_beats_votes': g_hybrid_beats, 'G_nonzero_where_votes_silent': g_not_silent, 'G_near_string_hole_closed': g_near_closed, 'G_sum_still_loses': g_sum_still_loses, 'G_typo_favours_ink': g_typo, 'G_fp_has_dynamic_range': g_range, 'G_fp_no_collisions': g_no_collisions}, 'summary': {'fp_geometry': {'raw': idx.raw_cos, 'spectrum': idx.spectrum, 'whitened': idx.cos if args.whiten_fp else None}, 'votes': r_votes, 'maxsim': r_max, 'sum': r_sum, 'sum_sif': r_sif, 'hybrid': r_hyb, 'tau_sweep': sweep, 'silence': silence, 'popularity_floor': floor, 'typo_arm': noise, 'reference_261f': {'votes_top1': 0.246, 'votes_median_rank': 76.5, 'mean_top1': 0.034, 'mean_median_rank': 1036.5}}, 'note': "Votes are not an alternative to the fingerprints, they are the tau->1 limit of score(q,slot) = sum_w idf(w) * max_c relu(cos(fp(w),fp(c)) - tau)^p. content() dedups a slot's words, so at tau->1 the max collapses to an indicator and the expression is votes with idf, exactly - G_kernel_reduces_to_votes checks that empirically rather than on paper. The question the sweep answers is whether the interior tau<1 buys anything the endpoint does not. Ranks use 266's correction (a zero-scoring gold ranks LAST) and n-way is strict, because both failures previously read as accuracy. The silence split is the diagnostic owed since 264: near_string is the half a character encoder can reach by construction, purely_semantic is the half it cannot, and reporting the second honestly is what keeps this from being a claim about meaning.", 'timestamp': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    ns = silence['near_string']
    ps = silence['purely_semantic']
    MINI.write_text(f"# Stage 277 ink revival: votes as the tau->1 limit\n\n**{overall}** slots={idx.n_slots} vocab={len(idx.vocab)} eval={len(items)} tau={args.tau} trained params **0**\n\n| arm | top1 | median rank | {args.n_way}-way strict |\n|---|---|---|---|\n| votes (incumbent) | {r_votes['top1']:.3f} | {r_votes['median_rank']:.0f} | {r_votes[f'acc_{args.n_way}way_strict']:.3f} |\n| sum (idf-weighted) | {r_sum['top1']:.3f} | {r_sum['median_rank']:.0f} | {r_sum[f'acc_{args.n_way}way_strict']:.3f} |\n| sum + all-but-top | {r_sif['top1']:.3f} | {r_sif['median_rank']:.0f} | {r_sif[f'acc_{args.n_way}way_strict']:.3f} |\n| **maxsim** | **{r_max['top1']:.3f}** | {r_max['median_rank']:.0f} | {r_max[f'acc_{args.n_way}way_strict']:.3f} |\n| hybrid (a={args.alpha}) | {r_hyb['top1']:.3f} | {r_hyb['median_rank']:.0f} | {r_hyb[f'acc_{args.n_way}way_strict']:.3f} |\n\n- kernel reduces to votes at tau={t_hi}: **{g_reduces}** (maxsim {sweep.get(t_hi, {}).get('top1', float('nan')):.3f} vs votes {r_votes['top1']:.3f})\n- votes silent on gold: **{silence['n_votes_silent']}/{len(items)}** ({silence['frac_votes_silent']:.3f}) -> near-string {ns.get('n', 0)}, purely semantic {ps.get('n', 0)}\n- on the near-string half maxsim top1 {ns.get('maxsim_top1', float('nan')):.3f}, top10 {ns.get('maxsim_top10', float('nan')):.3f}; on the semantic half top10 {ps.get('maxsim_top10', float('nan')):.3f}\n- popularity floor: votes {floor['votes_top1']:.3f}, maxsim {floor['maxsim_top1']:.3f}\n- fp geometry raw: random pair p50 {idx.raw_cos['random_pair']['p50']:.3f} / p95 {idx.raw_cos['random_pair']['p95']:.3f}, nearest-other p50 {idx.raw_cos['nearest_other']['p50']:.3f}" + (f" -> whitened ({args.whiten_mode} k={args.whiten_fp}) random p50 {idx.cos['random_pair']['p50']:.3f} / p95 {idx.cos['random_pair']['p95']:.3f}, nearest p50 {idx.cos['nearest_other']['p50']:.3f}" if args.whiten_fp else '') + f"; range usable **{g_range}**\n- fp unit: **{unit}**, collisions (nearest cos >= 0.9999) **{idx.cos['collision_rate']:.4f}** (raw {idx.raw_cos['collision_rate']:.4f})\n- spectrum: d={idx.spectrum['d']}, 95% of variance in **{idx.spectrum['dims_95pct']}** dims, participation ratio {idx.spectrum['participation_ratio']:.1f} - width is not the bottleneck if this is far below d\n" + (f"- typo {args.typo_rate}: votes {noise['votes']['top1']:.3f} -> maxsim {noise['maxsim']['top1']:.3f}, hybrid {noise['hybrid']['top1']:.3f}\n" if noise else ''), encoding='utf-8')
    log(json.dumps({'overall': overall, 'gates': out['gates']}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())