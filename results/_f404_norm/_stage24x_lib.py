"""Shared helpers for unexpected-comparison stages 240–245."""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage196_tapelm import gpt_span, load_gpt
from _tapelm_ext import DomainAdapter
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
CKPT_MIXED = Path('checkpoints/stage238_mixed_scratch.pt')
WIKI = Path('data/_wikitext103_train.txt')
EXAM_V3 = Path('data/stage191_exam_v3.jsonl')
STORIES = Path('data/external_tinystories_100k_85.txt')
WORD_RE = re.compile("[A-Za-z][A-Za-z0-9'\\-]{2,}")

def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = msg if msg.endswith('\n') else msg + '\n'
        try:
            print(line, end='', flush=True)
        except UnicodeEncodeError:
            print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(line)
    return log

def load_p1(device):
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bank = FpBank(model, stoi, device)
    return (flat, off, stoi, n_char, tok, V, pad_id, char_table, model, bank)

def load_curve_ckpt(path: Path, n_char: int, V: int, stoi, device):
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return (model, FpBank(model, stoi, device))

def wiki_bits(smoke: bool, core_n: int, rng: random.Random):
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(4000000 if smoke else 20000000)
    values_pool = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5)))
    rng.shuffle(values_pool)
    core = list(dict.fromkeys((w for w in re.findall('[A-Za-z][a-z]{2,}', text) if len(w) <= 14)))[:core_n]
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 200]
    return (text, values_pool, core, paras)

def make_facts(n_facts: int, values_pool, rng: random.Random):
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 40) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[i]
        facts.append({'S': S, 'value': Vv, 'sent': f'{S} was appointed director of {Vv} in 1987 .', 'fid': i})
    all_values = [f['value'] for f in facts] + values_pool[n_facts:n_facts + 80]
    return (facts, all_values)

def write_tape_bank(bank: FpBank, facts):
    keys, vals = ([], [])
    for f in facts:
        k = bank.fp([f['S']])[0]
        c = bank.ctx_fp(f['sent'], exclude=f['value'])
        keys.append(F.normalize(k + c, dim=-1) if c is not None else k)
        vals.append(f['value'])
    return (torch.stack(keys, 0), vals)

def _tape_query(bank_q, fact, W_bwd=None):
    """Query built like a key: subject fingerprint + context of a prefix that stops where the value
    would go. A context-only query leaves the shared template dominating the similarity, and it also
    does not match the (prefix -> slot) pairs W_q is trained on."""
    a = bank_q.fp([fact['S']])[0]
    c = bank_q.ctx_fp(f"In the report {fact['S']} was appointed director of")
    q = F.normalize(a + c, dim=-1) if c is not None else a
    if W_bwd is not None:
        q = F.normalize(W_bwd.map_raw(q.unsqueeze(0)), dim=-1)[0]
    return q

def init_query_adapter(device: torch.device) -> DomainAdapter:
    """Trainable query warp; tape KEYS stay frozen canonical — only queries move.

    Naming in checkpoints:
      W_q_glue  — stage 256 SlotBias.W_q (decode glue + gate)
      W_q_stream — stage 255 stream ingest W_query (continual query adapter)
    """
    W = DomainAdapter(256).to(device)
    with torch.no_grad():
        W.w.weight.copy_(torch.eye(256, device=device) + 0.02 * torch.randn(256, 256, device=device))
    return W

def _gold_slot_indices(value: str, Vlist: list[str]) -> list[int]:
    return [i for i, v in enumerate(Vlist) if v == value]

def train_query_adapter(W: DomainAdapter, bank_q, facts: list[dict], Kmat: torch.Tensor, Vlist: list[str], device: torch.device, steps: int, seed: int, lr: float=0.002, micro: int=8) -> float:
    """Contrastive align adapted queries to frozen keys (InfoNCE over full bank)."""
    if not facts or Kmat.numel() == 0 or steps <= 0:
        return float('nan')
    W.train()
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed + 17)
    K = Kmat.detach().to(device, torch.float32)
    losses: list[float] = []
    for _ in range(steps):
        batch = [facts[rng.randrange(len(facts))] for _ in range(min(micro, len(facts)))]
        tot = 0.0
        n = 0
        for f in batch:
            gidx = _gold_slot_indices(f['value'], Vlist)
            if not gidx:
                continue
            q = _tape_query(bank_q, f, None).to(device)
            q = W(q.unsqueeze(0)).squeeze(0)
            sc = K @ q
            loss = -(torch.logsumexp(sc[gidx], 0) - torch.logsumexp(sc, 0))
            tot = tot + loss
            n += 1
        if n == 0:
            continue
        loss = tot / n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(W.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    W.eval()
    return sum(losses) / max(1, len(losses))

def train_query_adapter_pairs(W: DomainAdapter, pairs: list[dict], Kmat: torch.Tensor, Vlist: list[str], device: torch.device, steps: int, seed: int, lr: float=0.002, batch: int=64, tau: float=0.05) -> float:
    """InfoNCE over the full bank using (prefix -> slot) pairs harvested from the stream itself.

    Fitting W_q on a handful of planted probe facts only teaches it where those facts live, so the
    recall number then says nothing about the rest of the bank. Ingested entities supply thousands
    of pairs, which lets the probe facts stay fully held out.
    """
    if not pairs or Kmat.numel() == 0 or steps <= 0:
        return float('nan')
    by_val: dict[str, int] = {}
    for i, v in enumerate(Vlist):
        by_val.setdefault(v, i)
    usable = [p for p in pairs if p['value'] in by_val]
    if not usable:
        return float('nan')
    K = Kmat.detach().to(device, torch.float32)
    Q = torch.stack([p['q'] for p in usable]).to(device, torch.float32)
    gold = torch.tensor([by_val[p['value']] for p in usable], device=device)
    W.train()
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator(device='cpu').manual_seed(seed + 31)
    losses: list[float] = []
    for _ in range(steps):
        sel = torch.randint(0, Q.size(0), (min(batch, Q.size(0)),), generator=g).to(device)
        q = F.normalize(W(Q[sel]), dim=-1)
        loss = F.cross_entropy(q @ K.t() / tau, gold[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(W.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    W.eval()
    return sum(losses) / max(1, len(losses))

def tape_recall_metrics(facts, all_values, bank_q, Kmat, Vlist, seed: int, W_bwd=None, block: int=200000, postings=None, retrieve_mode: str='auto') -> dict:
    """Fixed-seed 4-way distractors plus full-bank rank metrics (scores are time-invariant if arc_enc frozen)."""
    if not facts or Kmat.numel() == 0:
        return {'four_way': float('nan'), 'top1': float('nan'), 'mrr': float('nan'), 'median_rank': float('nan')}
    from _inprint_glue import VOTES_AUTO_MIN_SLOTS, resolve_retrieve_mode, slot_query_words
    from _retrieval_modes import vote_scores
    qrng = random.Random(seed + 3)
    by_val: dict[str, list[int]] = {}
    for j, v in enumerate(Vlist):
        by_val.setdefault(v, []).append(j)
    K = Kmat.detach().to('cpu', torch.float32) if Kmat.is_cuda else Kmat.float()
    n_slots = K.size(0)
    use_votes = postings is not None and n_slots >= VOTES_AUTO_MIN_SLOTS and (resolve_retrieve_mode(retrieve_mode, n_slots) == 'votes')
    ok4 = 0
    ranks: list[int] = []
    for f in facts:
        if use_votes:
            words = slot_query_words(f"In the report {f['S']} was appointed director of")
            sc = vote_scores(words, postings.postings, postings.idf)
            gold = f['value']
            gsc = max((sc.get(j, 0.0) for j in by_val.get(gold, ())), default=0.0)
            rank = 1 + sum((1 for v in sc.values() if v > gsc))
        else:
            qq = _tape_query(bank_q, f, W_bwd).detach().cpu().float()
            sc_all = []
            for i in range(0, n_slots, block):
                sc_all.append(K[i:i + block] @ qq)
            sc_col = torch.cat(sc_all) if sc_all else torch.zeros(0)
            gold = f['value']
            gold_sc = float(sc_col[by_val[gold]].max()) if gold in by_val else -1.0
            rank = 1 + int((sc_col > gold_sc).sum().item())
        ranks.append(rank)
        others = [x for x in all_values if x != gold]
        qrng.shuffle(others)
        cands = [gold] + others[:3]
        order = list(range(4))
        qrng.shuffle(order)
        shuf = [cands[i] for i in order]
        if use_votes:
            sc4 = [max((sc.get(j, 0.0) for j in by_val.get(c, ())), default=-1.0) for c in shuf]
        else:
            sc4 = [float(sc_col[by_val[c]].max()) if c in by_val else -1.0 for c in shuf]
        ok4 += int(int(np.argmax(sc4)) == order.index(0))
    r = np.asarray(ranks, dtype=np.float64)
    return {'four_way': ok4 / len(facts), 'top1': float(np.mean(r == 1)), 'mrr': float(np.mean(1.0 / r)), 'median_rank': float(np.median(r))}

def tape_recall(facts, all_values, bank_q, Kmat, Vlist, seed: int, W_bwd=None) -> float:
    return tape_recall_metrics(facts, all_values, bank_q, Kmat, Vlist, seed, W_bwd=W_bwd)['four_way']

def tape_recall_decision(facts, all_values, bank_q, Kmat, Vlist, seed: int, W_bwd=None, **kwargs) -> dict:
    """Closed-pool 4-way plus open full-bank rank — use in decision JSON."""
    m = tape_recall_metrics(facts, all_values, bank_q, Kmat, Vlist, seed, W_bwd=W_bwd, **kwargs)
    return {'four_way': m['four_way'], 'full_bank_top1': m['top1'], 'full_bank_mrr': m['mrr'], 'full_bank_median_rank': m['median_rank']}

def canonical_fp_version() -> str:
    return CKPT_P1.name

def load_next_tok_items(n_next: int):
    items = []
    if EXAM_V3.exists():
        with EXAM_V3.open(encoding='utf-8') as f:
            for line in f:
                it = json.loads(line)
                if it.get('type') == 'next_tok':
                    items.append(it)
                if len(items) >= n_next:
                    break
    return items

def curve_next_tok(model, char_table, pad_id, items, device) -> float:
    if not items:
        return float('nan')
    ok = 0
    for it in items:
        sc = [span_logprob_x(model, char_table, pad_id, it['ctx_ids'], c, device) for c in it['cand_ids']]
        ok += int(int(np.argmax(sc)) == it['gold_idx'])
    return ok / len(items)

def gpt_next_tok(gm, items, device) -> float:
    if not items:
        return float('nan')
    ok = 0
    for it in items:
        sc = [gpt_span(gm, device, it['ctx_ids'], c) for c in it['cand_ids']]
        ok += int(int(np.argmax(sc)) == it['gold_idx'])
    return ok / len(items)

def gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, seed: int) -> float:
    qrng = random.Random(seed + 3)
    ok = 0
    for f in facts:
        ctx = [i for i in tok.encode(f"In the report {f['S']} was linked to the organization of").ids if i != pad_id]
        others = [x for x in all_values if x != f['value']]
        qrng.shuffle(others)
        cands = [f['value']] + others[:3]
        order = list(range(4))
        qrng.shuffle(order)
        shuf = [cands[i] for i in order]
        sc = [gpt_span(gm, device, ctx, [i for i in tok.encode(' ' + c).ids if i != pad_id]) for c in shuf]
        ok += int(int(np.argmax(sc)) == order.index(0))
    return ok / max(1, len(facts))

def ft_batch(brng, fact_ids, real_ids, n_batch, ft_len, device, mix_real=True, fact_p=0.75):
    rows = []
    for _ in range(n_batch):
        if not mix_real or brng.random() < fact_p or (not real_ids):
            seq = []
            while len(seq) < ft_len:
                seq += fact_ids[brng.randrange(len(fact_ids))]
            rows.append(seq[:ft_len])
        else:
            s = brng.randrange(max(1, len(real_ids) - ft_len - 1))
            rows.append(real_ids[s:s + ft_len])
    return torch.tensor(rows, device=device)

def memorize_gpt(gm, tok, pad_id, facts, all_values, paras, device, seed, ft_steps, n_batch, ft_len, ft_lr, mem_target, check_every, log):
    fact_ids = [[i for i in tok.encode(f['sent']).ids if i != pad_id] for f in facts]
    real_ids = [i for i in tok.encode(' '.join(paras[:400])[:150000]).ids if i != pad_id]
    opt = torch.optim.AdamW(gm.parameters(), lr=ft_lr, weight_decay=0.01)
    brng = random.Random(seed + 11)
    used = 0
    gm.train()
    for step in range(1, ft_steps + 1):
        x = ft_batch(brng, fact_ids, real_ids, n_batch, ft_len, device)
        loss = gm(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        used = step
        if step % check_every == 0:
            gm.eval()
            cur = gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, seed)
            log(f'  gpt memorize step {step}: loss={float(loss):.3f} recall={cur:.3f}')
            if cur >= mem_target:
                gm.train()
                break
            gm.train()
    gm.eval()
    return (used, fact_ids, real_ids)

def code_ce(gm, code_ids, n_batch, ft_len, lr, steps, device, seed, log, tag='B', fact_ids=None, rehearsal=0.0):
    if len(code_ids) < ft_len + 2:
        raise RuntimeError('code corpus too short')
    opt = torch.optim.AdamW(gm.parameters(), lr=lr, weight_decay=0.01)
    brng = random.Random(seed + 17)
    gm.train()
    for step in range(1, steps + 1):
        rows = []
        for _ in range(n_batch):
            use_fact = fact_ids is not None and rehearsal > 0 and (brng.random() < rehearsal)
            if use_fact:
                seq = []
                while len(seq) < ft_len:
                    seq += fact_ids[brng.randrange(len(fact_ids))]
                rows.append(seq[:ft_len])
            else:
                s = brng.randrange(max(1, len(code_ids) - ft_len - 1))
                rows.append(code_ids[s:s + ft_len])
        x = torch.tensor(rows, device=device)
        loss = gm(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(40, steps // 4) == 0:
            log(f'  gpt {tag} step {step}: loss={float(loss):.3f}')
    gm.eval()

@torch.no_grad()
def gpt_emb(gm, tok, pad_id, device, ids):
    ids = [i for i in ids if i != pad_id][-MAX_ARCS:]
    if not ids:
        return None
    h = gm.transformer(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0].mean(0)
    return F.normalize(h, dim=-1)

def gpt_word(gm, tok, pad_id, device, w):
    return gpt_emb(gm, tok, pad_id, device, tok.encode(' ' + w).ids)

def gpt_ctx(gm, tok, pad_id, device, textv, exclude=None):
    ws = [x for x in WORD_RE.findall(textv) if x != exclude][:40]
    return gpt_emb(gm, tok, pad_id, device, tok.encode(' '.join(ws)).ids) if len(ws) >= 3 else None

def write_rag_bank(gm, tok, pad_id, device, facts):
    rows, vals = ([], [])
    for f in facts:
        k = gpt_word(gm, tok, pad_id, device, f['S'])
        c = gpt_ctx(gm, tok, pad_id, device, f['sent'], exclude=f['value'])
        rows.append(F.normalize(k + c, dim=-1) if c is not None else k)
        vals.append(f['value'])
    return (torch.stack(rows, 0), vals)

def rag_recall(gm, tok, pad_id, device, facts, all_values, Kmat, Vlist, seed: int) -> float:
    ok, n = (0, 0)
    qrng = random.Random(seed + 3)
    for f in facts:
        q = gpt_ctx(gm, tok, pad_id, device, f"In the report {f['S']} was linked to the organization.", exclude=f['value'])
        if q is None:
            q = gpt_word(gm, tok, pad_id, device, f['S'])
        others = [x for x in all_values if x != f['value']]
        qrng.shuffle(others)
        cands = [f['value']] + others[:3]
        order = list(range(4))
        qrng.shuffle(order)
        shuf = [cands[i] for i in order]
        sc = []
        for c in shuf:
            idxs = [j for j, v in enumerate(Vlist) if v == c]
            sc.append(float((Kmat[idxs] @ q).max()) if idxs else -1.0)
        ok += int(int(np.argmax(sc)) == order.index(0))
        n += 1
    return ok / max(1, n)

def dump(decision: Path, mini: Path, out: dict, title: str):
    RES.mkdir(parents=True, exist_ok=True)
    decision.write_text(json.dumps(out, indent=2), encoding='utf-8')
    mini.write_text(f"# {title}\n\n**{out['overall']}**\n\n```json\n{json.dumps(out, indent=2)}\n```\n", encoding='utf-8')