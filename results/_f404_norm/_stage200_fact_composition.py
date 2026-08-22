"""
Stage 200 — do we OPERATE on facts (in-space vectors) or just READ them (index)?

User's question: our facts live inside the model's own fp-space; RAG's live in an
external text index. Does in-space storage give a real compositional/chaining
advantage over index lookup?

Test: planted k-hop chains of NOVEL entities (A0->A1->...->Ak), each edge stated in a
distinctive sentence spread across the corpus (beyond the attention window). Answer:
given A0, follow k hops to Ak. Memory also holds thousands of real-edge distractors.

Systems (fair subject-anchoring for both):
  curve_string : per hop  q=fp(current) -> argmax over keys -> value STRING -> next   (fp encoder)
  curve_vector : per hop  q=value FP (vector-native) -> argmax -> value FP -> ...      (never decodes;
                 pure operable-vector chaining — the thing RAG structurally cannot do)
  rag          : per hop  q=gpt_word_embed(current) -> argmax -> value STRING -> next  (GPT encoder = index)
  gpt_incontext: sees only last window of the concatenated edge sentences (beyond-window control)

Also: binding one-shot 2-hop (old SOTE edge_fp) — answer a 2-hop query with composed vectors,
no sequential re-retrieval.

Gates:
  G_external   curve & rag at k>=2 >> gpt_incontext        (external memory required)
  G_chain      curve_string at k=3 >= 0.50
  G_vs_rag     curve_string(k=3) - rag(k=3) >= 0.10         (fp is a better chainer -> operable win)
  G_vectornative curve_vector at k=3 >= 0.50                (chaining survives with NO decoding)

  python _stage200_fact_composition.py
"""
from __future__ import annotations
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank
from _stage196_tapelm import load_gpt
RES = Path('results')
CKPT_P1 = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
DECISION = RES / 'stage200_decision.json'
MINI = RES / 'stage200_mini.md'
LOG = RES / '_stage200_log.txt'
SEED = 200
CORPUS_CHARS = 150000000
MID_START = 80000000
MID_CHARS = 4000000
N_CHAIN = 60
K = 3
N_DISTRACT = 6000
MAX_ARCS = 64

def log(msg: str) -> None:
    line = msg if msg.endswith('\n') else msg + '\n'
    try:
        print(line, end='', flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'), end='', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line)

def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text('', encoding='utf-8')
    log(f'Stage200 start {datetime.now(timezone.utc).isoformat()}')
    log('fact composition: in-space chaining vs index-RAG vs vanilla GPT')
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

    @torch.no_grad()
    def gpt_word(w: str) -> torch.Tensor:
        ids = [i for i in tok.encode(' ' + w).ids if i != pad_id][:MAX_ARCS]
        if not ids:
            ids = [pad_id]
        x = torch.tensor([ids], device=device)
        h = gm.transformer(input_ids=x).last_hidden_state[0].mean(0)
        return F.normalize(h, dim=-1)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        text = f.read(CORPUS_CHARS)
    mid = text[MID_START:MID_START + MID_CHARS]
    mid_paras = [p.strip() for p in mid.split('\n') if 120 < len(p.strip()) < 1000][:3000]
    del text
    log(f'mid_paras={len(mid_paras)} ({time.time() - t0:.0f}s)')
    key_fp, val_str, val_fp, key_g = ([], [], [], [])

    def add_slot(u, v):
        key_fp.append(bank.fp([u])[0])
        val_str.append(v)
        val_fp.append(bank.fp([v])[0])
        key_g.append(gpt_word(u))
    for p in mid_paras:
        if len(val_str) >= N_DISTRACT:
            break
        ents = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(p))))
        for i in range(len(ents) - 1):
            if len(val_str) >= N_DISTRACT:
                break
            if len(ents[i]) >= 4 and len(ents[i + 1]) >= 4:
                add_slot(ents[i], ents[i + 1])
    log(f'distractor slots={len(val_str)} ({time.time() - t0:.0f}s)')
    fakes = gen_fakes(set(), rng, N_CHAIN * (K + 1) + 50)
    chains = []
    fi = 0
    for _ in range(N_CHAIN):
        nodes = fakes[fi:fi + K + 1]
        fi += K + 1
        if len(nodes) < K + 1:
            break
        chains.append(nodes)
    edges_curve = []
    for ch in chains:
        for i in range(len(ch) - 1):
            u, v = (ch[i], ch[i + 1])
            add_slot(u, v)
            edges_curve.append((bank.fp([u])[0], bank.fp([v])[0]))
    K_fp = torch.stack(key_fp, 0)
    K_g = torch.stack(key_g, 0)
    Vfp = torch.stack(val_fp, 0)
    log(f'total slots={len(val_str)} chains={len(chains)} ({time.time() - t0:.0f}s)')
    ent_pool = list(dict.fromkeys([n for ch in chains for n in ch]))
    pool_fp = bank.fp(ent_pool)

    def chain_string(encoder_keys, q_of):
        accs = {}
        for k in range(1, K + 1):
            ok = 0
            for ch in chains:
                cur = ch[0]
                for _ in range(k):
                    q = q_of(cur)
                    idx = int((encoder_keys @ q).argmax())
                    cur = val_str[idx]
                ok += int(cur == ch[k])
            accs[k] = ok / len(chains)
        return accs

    def chain_vector():
        accs = {}
        for k in range(1, K + 1):
            ok = 0
            for ch in chains:
                v = bank.fp([ch[0]])[0]
                for _ in range(k):
                    idx = int((K_fp @ v).argmax())
                    v = Vfp[idx]
                pred = ent_pool[int((pool_fp @ v).argmax())]
                ok += int(pred == ch[k])
            accs[k] = ok / len(chains)
        return accs
    curve_string = chain_string(K_fp, lambda w: bank.fp([w])[0])
    rag = chain_string(K_g, gpt_word)
    curve_vector = chain_vector()
    log(f'curve_string={curve_string}')
    log(f'rag={rag}')
    log(f'curve_vector={curve_vector}')
    E = torch.stack([F.normalize(a * b, dim=-1) for a, b in edges_curve], 0)
    ok_bind = 0
    n_bind = 0
    for ch in chains:
        A0, A2 = (ch[0], ch[2])
        cands = [A2] + [ent_pool[rng.randint(0, len(ent_pool) - 1)] for _ in range(3)]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        gold = order.index(0)
        fa = bank.fp([A0])[0]
        sc = []
        for C in shuf:
            fc = bank.fp([C])[0]
            best = -9.9
            for m in ent_pool:
                fm = bank.fp([m])[0]
                s1 = float((E @ F.normalize(fa * fm, dim=-1)).max())
                s2 = float((E @ F.normalize(fm * fc, dim=-1)).max())
                best = max(best, s1 + s2)
            sc.append(best)
        ok_bind += int(int(np.argmax(sc)) == gold)
        n_bind += 1
    bind_2hop = ok_bind / max(1, n_bind)
    log(f'binding one-shot 2-hop acc={bind_2hop:.3f} (chance 0.25)')

    @torch.no_grad()
    def gpt_incontext_2hop():
        ok = n = 0
        for ch in chains:
            cue = [i for i in tok.encode(' ' + ch[0] + ' leads to').ids if i != pad_id]
            ctx = cue[-MAX_ARCS:]
            cands = [ch[2]] + [ent_pool[rng.randint(0, len(ent_pool) - 1)] for _ in range(3)]
            order = list(range(4))
            rng.shuffle(order)
            shuf = [cands[i] for i in order]
            gold = order.index(0)
            sc = []
            for c in shuf:
                cid = [i for i in tok.encode(' ' + c).ids if i != pad_id]
                seq = (ctx + cid)[-MAX_ARCS:]
                nc = len(seq) - len(cid)
                x = torch.tensor([seq], device=device)
                logp = F.log_softmax(gm(input_ids=x).logits[0], dim=-1)
                sc.append(sum((float(logp[nc + j - 1, t]) for j, t in enumerate(cid))) / max(1, len(cid)))
            ok += int(int(np.argmax(sc)) == gold)
            n += 1
        return ok / max(1, n)
    gpt_ic = gpt_incontext_2hop()
    log(f'gpt_incontext 2-hop acc={gpt_ic:.3f}')
    g_external = curve_string[2] - gpt_ic >= 0.3 and rag[2] - gpt_ic >= 0.0
    g_chain = curve_string[K] >= 0.5
    g_vs_rag = curve_string[K] - rag[K] >= 0.1
    g_vectornative = curve_vector[K] >= 0.5
    if g_chain and g_external and g_vs_rag and g_vectornative:
        overall = 'COMPOSE_OPERABLE_WIN'
    elif g_chain and g_external and g_vectornative:
        overall = 'COMPOSE_CHAINS_BUT_RAG_PARITY'
    elif g_chain and g_external:
        overall = 'COMPOSE_CHAINS_STRING_ONLY'
    else:
        overall = 'COMPOSE_WEAK'
    gates = {'g_external': g_external, 'g_chain': g_chain, 'g_vs_rag': g_vs_rag, 'g_vectornative': g_vectornative}
    R = {'curve_string': curve_string, 'curve_vector': curve_vector, 'rag_index': rag, 'binding_2hop': bind_2hop, 'gpt_incontext_2hop': gpt_ic, 'slots': len(val_str), 'chains': len(chains), 'K': K, 'chance_exact': round(1.0 / len(ent_pool), 4)}
    out = {'timestamp': datetime.now(timezone.utc).isoformat(), 'protocol': 'fact_composition_200', 'overall': overall, 'gates': gates, 'results': R, 'note': 'in-space fp chaining (string & vector-native) + binding one-shot vs GPT-embedding index-RAG and vanilla in-context GPT; tests whether facts are OPERABLE vectors vs READ-by-index documents'}
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    MINI.write_text('\n'.join(['# Stage200 — fact composition: operate vs read-by-index', '', f'**Overall:** `{overall}`', '', f'- curve_string k1/k2/k3: {curve_string[1]:.2f} / {curve_string[2]:.2f} / {curve_string[3]:.2f}', f'- curve_vector (no decode) k1/k2/k3: {curve_vector[1]:.2f} / {curve_vector[2]:.2f} / {curve_vector[3]:.2f}', f'- rag_index k1/k2/k3: {rag[1]:.2f} / {rag[2]:.2f} / {rag[3]:.2f}', f'- binding one-shot 2-hop: {bind_2hop:.3f} (chance 0.25)', f'- gpt_incontext 2-hop: {gpt_ic:.3f}', '', f'slots={len(val_str)} chains={len(chains)} K={K}', f'gates: {gates}']), encoding='utf-8')
    log(f'[200] {overall} | curve_str k3={curve_string[3]:.2f} vec k3={curve_vector[3]:.2f} rag k3={rag[3]:.2f} gpt={gpt_ic:.2f}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())