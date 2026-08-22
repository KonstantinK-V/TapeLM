"""187b — control: does GPT-181 raise entropy after fake entities? (G3 reference)"""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import _stage177_curve_bpe as s177
import _stage184_exam_logprob as s184
import _stage187_self_model as s187
EXAM = Path('data/stage186_exam_v2.jsonl')

@torch.no_grad()
def gpt_entropy_after(model, ctx_ids, span_ids, device) -> float:
    seq = ctx_ids + span_ids
    x = torch.tensor([seq], device=device)
    logits = model(input_ids=x).logits[0, -1]
    p = F.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-09)).sum())

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id('[PAD]') or 0
    items = [json.loads(l) for l in EXAM.read_text(encoding='utf-8').splitlines() if l.strip()]
    ent_items = [it for it in items if it['type'] == 'entity'][:80]
    gpt = s184.load_gpt(device)
    rngf = random.Random(3)
    e_real, e_fake = ([], [])
    for it in ent_items:
        gold_ids = it['cand_ids'][it['gold_idx']]
        fake = s187.FAKES[rngf.randint(0, len(s187.FAKES) - 1)]
        fake_ids = [i for i in tok.encode(' ' + fake).ids if i != pad_id]
        e_real.append(gpt_entropy_after(gpt, it['ctx_ids'], gold_ids, device))
        e_fake.append(gpt_entropy_after(gpt, it['ctx_ids'], fake_ids, device))
    print(f'GPT181: entropy_after_real={np.mean(e_real):.3f} entropy_after_fake={np.mean(e_fake):.3f} fake>real={np.mean(e_fake) > np.mean(e_real)}')
if __name__ == '__main__':
    main()