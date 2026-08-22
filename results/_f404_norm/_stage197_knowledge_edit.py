"""
Stage 197 — knowledge edit for the tape: make one-shot fact acquisition a CLEAN
win vs vanilla GPT, and find the one in-space advantage over GPT+RAG.

196 left edit weak (0.417, mean-ctx key, disjoint windows). Here we test stronger
zero-training write mechanisms and the surprise-gated write policy.

Setup: synthetic one-shot facts on a held-out corpus slice. Each fact ties a brand
-new fake object F to an EXISTING subject entity S: WRITE window mentions "S F",
disjoint QUERY window mentions S only. Recall F by anchoring on S (4-way, chance 0.25).

Mechanisms (all zero-train, one frozen fp-space):
  M1 mean-ctx   : key=fp(context words)                         (196 baseline)
  M2 subject    : key=fp(S)                                     (variant 2)
  M3 blend      : key=norm(fp(S)+fp(context))                   (variant 3)
  M4 multi-key  : slots on fp(S) + top-2 rarest context words   (variant 3+4, fp-lexicon anchors)

Controls:
  GPT parametric : logprob(" "+F | query)  -> chance (F unseen, cannot ingest write)
  GPT+RAG        : key=GPT embed(write), query=GPT embed(query) -> nearest rival

Selective write (the in-space advantage over generic RAG):
  stream of facts = few NOVEL (fake F, high fp-lexicon surprise) + many boilerplate (real, low).
  budget B slots. gated = write top-B by fp-lexicon surprise; uniform = write first-B (ingestion order).
  metric = recall of NOVEL facts. A generic store has no in-space novelty prior; the tape gets it free.

Gates:
  G_acquire  best curve mech >= 0.60 and > gpt_param + 0.20   (clean win over vanilla GPT)
  G_vs_rag   best curve mech >= gpt_rag - 0.03                 (not worse than nearest rival)
  G_select   gated_recall > uniform_recall + 0.15             (free write policy from fp-space)

  python _stage197_knowledge_edit.py
"""
from __future__ import annotations
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank
from _stage196_tapelm import GptBank, load_gpt
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
DECISION = RES / 'stage197_decision.json'
MINI = RES / 'stage197_mini.md'
LOG = RES / '_stage197_log.txt'
SEED = 197
CORPUS_CHARS = 150000000
MID_START = 60000000
MID_CHARS = 4000000
MIN_COUNT = 2
MAX_LEX = 200000
N_ACQ = 100
MAX_ARCS = 64
MAX_CHARS = s177.MAX_CHARS_PER_ARC

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def build_acq_facts(paras, rng, existing_fakes) -> list[dict]:
    facts = []
    fi = [0]

    def next_fake():
        f = existing_fakes[fi[0] % len(existing_fakes)]
        fi[0] += 1
        return f
    for p in paras:
        if len(facts) >= N_ACQ:
            break
        occ = defaultdict(list)
        for m in ENT_RE.finditer(p):
            occ[m.group(1)].append((m.start(), m.end()))
        cand = [(s, ps) for s, ps in occ.items() if len(ps) >= 2 and len(s) >= 4]
        if not cand:
            continue
        S, ps = cand[rng.randint(0, len(cand) - 1)]
        (o1s, o1e), (o2s, o2e) = (ps[0], ps[-1])
        if o2s - o1e < CTX_WIN:
            continue
        F_new = next_fake()
        w1 = p[max(0, o1s - CTX_WIN):o1e]
        w1 = w1 + ' ' + F_new + ' ' + p[o1e:min(len(p), o1e + 20)]
        w2 = p[max(0, o2s - CTX_WIN):min(len(p), o2e + CTX_WIN)]
        if F_new in w2 or S not in w2 or len(WORD_RE.findall(w2)) < 4:
            continue
        facts.append({'S': S, 'F': F_new, 'write': w1, 'query': w2})
    return facts

class Mech:
    """A write/query key generator over a fp-bank exposing .fp(list) and .ctx_fp(text, exclude)."""

    def __init__(self, name, bank, lexs=None, lex=None):
        self.name = name
        self.bank = bank
        self.lexs = lexs
        self.lex = lex

    def _rare_anchors(self, text, exclude, k=2):
        ws = [w for w in WORD_RE.findall(text) if w != exclude and w[0:1].isupper() is False]
        ws = list(dict.fromkeys(ws))
        if not ws or self.lex is None:
            return []
        fps = self.bank.fp(ws)
        sur = 1.0 - (fps @ self.lex.T).max(dim=-1).values
        idx = torch.topk(sur, min(k, len(ws))).indices.tolist()
        return [ws[i] for i in idx]

    def write_keys(self, f):
        S, Fx = (f['S'], f['F'])
        if self.name == 'M1_meanctx':
            k = self.bank.ctx_fp(f['write'], exclude=Fx)
            return [k] if k is not None else []
        if self.name == 'M2_subject':
            return [self.bank.fp([S])[0]]
        if self.name == 'M3_blend':
            c = self.bank.ctx_fp(f['write'], exclude=Fx)
            base = self.bank.fp([S])[0]
            return [F.normalize(base + c, dim=-1)] if c is not None else [base]
        if self.name == 'M4_multikey':
            keys = [self.bank.fp([S])[0]]
            for a in self._rare_anchors(f['write'], Fx):
                keys.append(self.bank.fp([a])[0])
            return keys
        raise ValueError(self.name)

    def query_keys(self, f):
        S = f['S']
        if self.name == 'M1_meanctx':
            k = self.bank.ctx_fp(f['query'])
            return [k] if k is not None else []
        if self.name == 'M2_subject':
            return [self.bank.fp([S])[0]]
        if self.name == 'M3_blend':
            c = self.bank.ctx_fp(f['query'])
            base = self.bank.fp([S])[0]
            return [F.normalize(base + c, dim=-1)] if c is not None else [base]
        if self.name == 'M4_multikey':
            keys = [self.bank.fp([S])[0]]
            for a in self._rare_anchors(f['query'], None):
                keys.append(self.bank.fp([a])[0])
            return keys
        raise ValueError(self.name)

def score_mech(mech: Mech, facts, rng, device) -> float:
    krows, vrows = ([], [])
    for f in facts:
        for k in mech.write_keys(f):
            krows.append(k)
            vrows.append(f['F'])
    if not krows:
        return 0.0
    Kmat = torch.stack(krows, 0)
    allF = [f['F'] for f in facts]
    ok = n = 0
    for f in facts:
        qs = mech.query_keys(f)
        if not qs:
            continue
        Q = torch.stack(qs, 0)
        best_slot = (Q @ Kmat.T).max(0).values
        others = [x for x in allF if x != f['F']]
        rng.shuffle(others)
        cands = [f['F']] + others[:3]
        order = list(range(len(cands)))
        rng.shuffle(order)
        shuffled = [cands[i] for i in order]
        gold = order.index(0)
        sc = []
        for c in shuffled:
            idxs = [i for i, v in enumerate(vrows) if v == c]
            sc.append(float(best_slot[idxs].max()) if idxs else -1.0)
        ok += int(int(np.argmax(sc)) == gold)
        n += 1
    return ok / max(1, n)

@torch.no_grad()
def gpt_param_edit(gm, tok, pad_id, device, facts, rng) -> float:
    allF = [f['F'] for f in facts]
    ok = n = 0
    for f in facts:
        others = [x for x in allF if x != f['F']]
        rng.shuffle(others)
        cands = [f['F']] + others[:3]
        order = list(range(len(cands)))
        rng.shuffle(order)
        shuffled = [cands[i] for i in order]
        gold = order.index(0)
        ctx = [i for i in tok.encode(f['query']).ids if i != pad_id][-MAX_ARCS:]
        sc = []
        for c in shuffled:
            cid = [i for i in tok.encode(' ' + c).ids if i != pad_id]
            seq = (ctx + cid)[-MAX_ARCS:]
            nc = len(seq) - len(cid)
            x = torch.tensor([seq], device=device)
            logp = F.log_softmax(gm(input_ids=x).logits[0], dim=-1)
            sc.append(sum((float(logp[nc + k - 1, t]) for k, t in enumerate(cid))) / max(1, len(cid)))
        ok += int(int(np.argmax(sc)) == gold)
        n += 1
    return ok / max(1, n)

def score_gpt_rag(gbank: GptBank, facts, rng) -> float:
    krows, vrows = ([], [])
    for f in facts:
        k = gbank.ctx_fp(f['write'], exclude=f['F'])
        if k is not None:
            krows.append(k)
            vrows.append(f['F'])
    if not krows:
        return 0.0
    Kmat = torch.stack(krows, 0)
    allF = [f['F'] for f in facts]
    ok = n = 0
    for f in facts:
        q = gbank.ctx_fp(f['query'])
        if q is None:
            continue
        best_slot = Kmat @ q
        others = [x for x in allF if x != f['F']]
        rng.shuffle(others)
        cands = [f['F']] + others[:3]
        order = list(range(len(cands)))
        rng.shuffle(order)
        shuffled = [cands[i] for i in order]
        gold = order.index(0)
        sc = []
        for c in shuffled:
            idxs = [i for i, v in enumerate(vrows) if v == c]
            sc.append(float(best_slot[idxs].max()) if idxs else -1.0)
        ok += int(int(np.argmax(sc)) == gold)
        n += 1
    return ok / max(1, n)

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage197 start {datetime.now(timezone.utc).isoformat()}')
    log('knowledge edit: anchored/multikey write + surprise-gated policy')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(SEED)
    t0 = time.time()
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model.eval()
    gm = load_gpt(device)
    bank = FpBank(model, stoi, device)
    log(f'models loaded ({time.time() - t0:.0f}s)')
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(CORPUS_CHARS)
    mid = text[MID_START:MID_START + MID_CHARS]
    mid_paras = [p.strip() for p in mid.split('\n') if 120 < len(p.strip()) < 1000][:1500]
    words = re.findall('[A-Za-z][a-z]+', text)
    del text
    cnt = Counter(words)
    word_set = set(cnt.keys())
    lex_words = [w for w, c in cnt.most_common(MAX_LEX) if c >= MIN_COUNT]
    fps = []
    for i in range(0, len(lex_words), 4096):
        fps.append(bank.fp(lex_words[i:i + 4096]))
    lex = torch.cat(fps, 0)
    log(f'mid_paras={len(mid_paras)} lexicon={len(lex_words)} ({time.time() - t0:.0f}s)')
    fakes = gen_fakes(word_set, rng, N_ACQ * 3)
    facts = build_acq_facts(mid_paras, rng, fakes)
    log(f'acquisition facts={len(facts)}')
    R = {'acquire': {}}
    for name in ('M1_meanctx', 'M2_subject', 'M3_blend', 'M4_multikey'):
        acc = score_mech(Mech(name, bank, lex=lex), facts, random.Random(SEED), device)
        R['acquire'][name] = acc
        log(f'  [{name}] acc={acc:.3f} ({time.time() - t0:.0f}s)')
    best_curve = max(R['acquire'].values())
    best_mech = max(R['acquire'], key=R['acquire'].get)
    gpt_param = gpt_param_edit(gm, tok, pad_id, device, facts, random.Random(SEED))
    gbank = GptBank(gm, tok, pad_id, device)
    gpt_rag = score_gpt_rag(gbank, facts, random.Random(SEED))
    R['controls'] = {'gpt_parametric': gpt_param, 'gpt_rag': gpt_rag, 'chance': 0.25}
    log(f'  [gpt_param]={gpt_param:.3f} [gpt_rag]={gpt_rag:.3f} ({time.time() - t0:.0f}s)')
    novel = [{'S': f['S'], 'F': f['F'], 'write': f['write'], 'query': f['query'], 'novel': True} for f in facts]
    boiler = []
    for p in mid_paras:
        if len(boiler) >= len(novel) * 5:
            break
        m = ENT_RE.search(p)
        if not m:
            continue
        ent = m.group(1)
        if ent in word_set or cnt.get(ent.lower(), 0) > 0:
            lo, hi = (max(0, m.start() - CTX_WIN), min(len(p), m.end() + CTX_WIN))
            boiler.append({'S': ent, 'F': ent, 'write': p[lo:hi], 'query': p[lo:hi], 'novel': False})
    stream = novel + boiler
    rng.shuffle(stream)
    budget = len(novel)
    vals = [it['F'] for it in stream]
    sur = 1.0 - (bank.fp(vals) @ lex.T).max(dim=-1).values
    gated_idx = set(torch.topk(sur, budget).indices.tolist())
    gated_written = {stream[i]['F'] for i in gated_idx if stream[i]['novel']}
    uniform_written = {it['F'] for it in stream[:budget] if it['novel']}
    n_novel = len(novel)
    gated_recall = len(gated_written) / max(1, n_novel)
    uniform_recall = len(uniform_written) / max(1, n_novel)
    R['selective'] = {'budget': budget, 'n_novel': n_novel, 'n_stream': len(stream), 'gated_novel_kept': gated_recall, 'uniform_novel_kept': uniform_recall}
    log(f'  [selective] gated_kept={gated_recall:.3f} uniform_kept={uniform_recall:.3f}')
    g_acquire = best_curve >= 0.6 and best_curve > gpt_param + 0.2
    g_vs_rag = best_curve >= gpt_rag - 0.03
    g_select = gated_recall > uniform_recall + 0.15
    if g_acquire and g_select and g_vs_rag:
        overall = 'EDIT_CLEAN_WIN'
    elif g_acquire and g_vs_rag:
        overall = 'EDIT_ACQUIRE_FIXED'
    elif g_acquire:
        overall = 'EDIT_ACQUIRE_FIXED_RAG_AHEAD'
    else:
        overall = 'EDIT_STILL_WEAK'
    gates = {'g_acquire': g_acquire, 'g_vs_rag': g_vs_rag, 'g_select': g_select}
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'knowledge_edit_197', 'overall': overall, 'best_mechanism': best_mech, 'best_curve_acc': best_curve, 'gates': gates, 'axes': R, 'note': 'zero-training; anchored/multikey write on one frozen fp-space; selective-write policy = fp-lexicon surprise (free, in-space) vs ingestion order'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage197 — knowledge edit (anchored write + surprise-gated policy)', '', f'**Overall:** `{overall}`  (best mech: `{best_mech}` = {best_curve:.3f})', '', 'acquisition (4-way, chance 0.25):', *[f'- {k}: {v:.3f}' for k, v in R['acquire'].items()], f'- gpt_parametric: {gpt_param:.3f}  |  gpt+rag: {gpt_rag:.3f}', '', f'selective write (budget={budget}, novel={n_novel}): gated kept {gated_recall:.3f} vs uniform {uniform_recall:.3f}', '', f'gates: {gates}']), encoding='utf-8')
    log(f'[197] {overall} | best {best_mech}={best_curve:.3f} gpt={gpt_param:.3f} rag={gpt_rag:.3f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())