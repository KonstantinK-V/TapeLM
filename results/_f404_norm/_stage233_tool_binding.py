"""
Stage 233 — Tool binding: fp(tool) ⊙ fp(entity) keys for structured memory ops.

Simulated tools (lookup / set / hop) bind to entity fps; values are facts.
Retrieve with bind(query_tool, query_entity) vs naive entity-only baseline.

  python _stage233_tool_binding.py [--smoke]
"""
from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
from _stage191_night import SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import fp_bind
RES = Path('results')
DECISION = RES / 'stage233_decision.json'
MINI = RES / 'stage233_mini.md'
CKPT = Path('checkpoints/stage191_p1_curve.pt')
WIKI = Path('data/_wikitext103_train.txt')
SEED = 233
TOOLS = ('lookup', 'set', 'hop')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 12 if args.smoke else 60
    rng = random.Random(SEED)
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)['model'])
    model.eval()
    bank = FpBank(model, stoi, device)
    with WIKI.open('r', encoding='utf-8', errors='ignore') as f:
        wiki_words = list(dict.fromkeys((m.group(1) for m in ENT_RE.finditer(f.read(2000000)) if len(m.group(1)) >= 5)))
    n_tools = len(TOOLS)
    n_ent = max(4, n // n_tools)
    ents = gen_fakes(set(wiki_words), rng, n_ent + 3)[:n_ent]
    keys_bind, keys_ent, vals, queries = ([], [], [], [])
    for i, ent in enumerate(ents):
        fe = bank.fp([ent])[0]
        for j, tool in enumerate(TOOLS):
            fact = f'{tool}:{ent}:{wiki_words[(i + j) % len(wiki_words)]}'
            ft = bank.fp([tool])[0]
            keys_bind.append(fp_bind(ft, fe)[0])
            keys_ent.append(fe)
            vals.append(fact)
            queries.append((tool, ent, fact))
    K_bind = torch.stack(keys_bind, 0)
    K_ent = torch.stack(keys_ent, 0)
    ok_bind = ok_ent = 0
    n_q = len(queries)
    by_ent: dict[str, list[tuple[str, str]]] = {}
    for tool, ent, fact in queries:
        by_ent.setdefault(ent, []).append((tool, fact))
    for tool, ent, gold in queries:
        fe = bank.fp([ent])[0]
        ft = bank.fp([tool])[0]
        qb = fp_bind(ft, fe)[0]
        cands = [gold]
        for t2, f2 in by_ent[ent]:
            if f2 != gold:
                cands.append(f2)
        for t2, ent2, f2 in queries:
            if ent2 != ent and len(cands) < 4:
                cands.append(f2)
        while len(cands) < 4:
            cands.append(queries[rng.randint(0, n_q - 1)][2])
        cands = cands[:4]
        rng.shuffle(cands)
        g = cands.index(gold)
        sc_b = []
        for v in cands:
            idxs = [j for j, vv in enumerate(vals) if vv == v]
            sc_b.append(float((K_bind[idxs] @ qb).max()) if idxs else -1.0)
        sc_e = []
        for v in cands:
            idxs = [j for j, vv in enumerate(vals) if vv == v]
            sc_e.append(float((K_ent[idxs] @ fe).max()) if idxs else -1.0)
        ok_bind += int(int(np.argmax(sc_b)) == g)
        ok_ent += int(int(np.argmax(sc_e)) == g)
    acc_bind = ok_bind / max(1, n_q)
    acc_ent = ok_ent / max(1, n_q)
    g1 = acc_bind >= 0.85
    g2 = acc_bind >= acc_ent + 0.15
    g3 = acc_ent <= 0.65
    overall = 'TOOL_BINDING_OK' if g1 and g2 and g3 else 'TOOL_BINDING_PARTIAL' if g1 and g2 else 'TOOL_BINDING_NO'
    out = {'stage': 233, 'overall': overall, 'gates': {'G_bind_acc_ge_0p85': g1, 'G_bind_beats_entity_by_0p20': g2, 'G_entity_baseline_le_0p55': g3}, 'acc_bind_key': acc_bind, 'acc_entity_key': acc_ent, 'tools': list(TOOLS), 'timestamp': datetime.now(timezone.utc).isoformat()}
    DECISION.write_text(json.dumps(out, indent=2), encoding='utf-8')
    MINI.write_text(f'# Stage 233 tool binding\n\n**{overall}** bind={acc_bind:.3f} entity={acc_ent:.3f}\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())