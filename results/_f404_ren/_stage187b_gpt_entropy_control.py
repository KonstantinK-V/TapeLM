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
v0 = v2('data/stage186_exam_v2.jsonl')

@v22.v11()
def gpt_entropy_after(v3, v4, v5, v6) -> v1:
    v7 = v4 + v5
    v8 = v22.v20([v7], device=v6)
    v9 = v3(input_ids=v8).v9[0, -1]
    v10 = v33.v21(v9, dim=-1)
    return v1(-(v10 * v22.v61(v10 + 1e-09)).v45())

def main():
    v6 = v22.v6('cuda' if v22.v55.v46() else 'cpu')
    v12 = v34.v23(v35(v47.v36))
    v13 = v12.v37('[PAD]') or 0
    v14 = [v48.v38(v39) for v39 in v0.v57(encoding='utf-8').v49() if v39.v50()]
    v15 = [v18 for v18 in v14 if v18['type'] == 'entity'][:80]
    v16 = v40.v24(v6)
    v17 = v41.v25(3)
    v26, v27 = ([], [])
    for v18 in v15:
        v28 = v18['cand_ids'][v18['gold_idx']]
        v29 = v51.v42[v17.v52(0, v58(v51.v42) - 1)]
        v30 = [v43 for v43 in v12.v59(' ' + v29).v53 if v43 != v13]
        v26.v44(v54(v16, v18['ctx_ids'], v28, v6))
        v27.v44(v54(v16, v18['ctx_ids'], v30, v6))
    v31(f'GPT181: entropy_after_real={v60.v56(v26):.3f} entropy_after_fake={v60.v56(v27):.3f} fake>real={v60.v56(v27) > v60.v56(v26)}')
if v19 == '__main__':
    v32()