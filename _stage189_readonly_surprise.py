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


class ReadOnlySurpriseModel(s188.SurpriseHeadModel):
    def forward_all(self, char_ids: torch.Tensor, pad: torch.Tensor):
        arcs = self.arc_enc(char_ids)
        fast = self.fast(arcs, pad_mask=pad)
        slow, surprise, pred_loss = self.slow(arcs, pad)
        logits = self.head(torch.cat([fast, slow], dim=-1))
        # READ-ONLY: no gradient into surprise from the CE/temperature path
        T = 1.0 + F.softplus(self.temp_w * surprise.detach() + self.temp_b).unsqueeze(-1)
        return logits / T, surprise, pred_loss


def main() -> int:
    # reuse the whole 188 pipeline with swapped model class and 189 paths
    from pathlib import Path

    s188.SurpriseHeadModel = ReadOnlySurpriseModel  # type: ignore
    s188.LOG = Path("results/_stage189_log.txt")
    s188.DECISION = Path("results/stage189_decision.json")
    s188.MINI = Path("results/stage189_mini.md")
    s188.CKPT_OUT = Path("checkpoints/stage189_readonly_surprise.pt")
    return s188.main()


if __name__ == "__main__":
    raise SystemExit(main())
