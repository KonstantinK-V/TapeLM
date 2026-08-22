"""
Stage 189 — read-only surprise (fix 188's Goodhart collapse).

188 lesson: when surprise modulates the head temperature WITH gradient, CE kills
the signal (self-error 0.056→0.003, surprise flat 0.0015 everywhere) — the model
games its own introspection.

Fix: temperature reads surprise.DETACHED. Calibration params (w,b) still learn,
but no gradient path lets CE reduce loss by suppressing surprise itself.

Same gates as 188 (judge = Exam v2, baseline 187 next_tok=0.727).

  python _stage189_readonly_surprise.py
  python _stage189_readonly_surprise.py --steps 3000
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import _stage187_self_model as s187
import _stage188_surprise_head as s188

class ReadOnlySurpriseModel(v2.v0):

    def forward_all(v9, v10: v25.v19, v11: v25.v19):
        v12 = v9.v20(v10)
        v13 = v9.v13(v12, pad_mask=v11)
        v21, v22, v23 = v9.v21(v12, v11)
        v14 = v9.v24(v25.v26([v13, v21], dim=-1))
        v15 = 1.0 + v29.v28(v9.v31 * v22.v32() + v9.v30).v27(-1)
        return (v14 / v15, v22, v23)

def main() -> v1:
    from pathlib import Path
    v2.v0 = v3
    v2.v4 = v16('results/_stage189_log.txt')
    v2.v5 = v16('results/stage189_decision.json')
    v2.v6 = v16('results/stage189_mini.md')
    v2.v7 = v16('checkpoints/stage189_readonly_surprise.pt')
    return v2.v17()
if v8 == '__main__':
    raise v18(v17())