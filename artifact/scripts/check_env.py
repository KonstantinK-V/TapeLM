#!/usr/bin/env python3
"""Verify Python deps and optional CUDA. No checkpoints required."""
from __future__ import annotations

import sys


def main() -> int:
    ok = True
    for mod in ("numpy", "torch", "tokenizers", "transformers"):
        try:
            __import__(mod)
            print(f"  OK  {mod}")
        except ImportError:
            print(f"  FAIL  {mod} — pip install -r artifact/requirements.txt")
            ok = False
    if ok:
        import torch

        print(f"  torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    repo = __import__("pathlib").Path(__file__).resolve().parents[2]
    p1 = repo / "checkpoints" / "stage191_p1_curve.pt"
    p2 = repo / "checkpoints" / "stage191_p2_gpt.pt"
    wreg = repo / "checkpoints" / "w_registry" / "w_registry.json"
    print(f"  P1 checkpoint: {'found' if p1.exists() else 'MISSING'}")
    print(f"  P2 checkpoint: {'found' if p2.exists() else 'MISSING'}")
    print(f"  W registry:    {'found' if wreg.exists() else 'optional (export_w_registry.py)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
