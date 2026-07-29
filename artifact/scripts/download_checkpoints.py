#!/usr/bin/env python3
"""Download TapeLM P1/P2 checkpoints from Hugging Face into checkpoints/."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "checkpoints"

# Change if your HF model id differs
HF_REPO = "Kostya03v/TapeLM-P1"
FILES = (
    "stage191_p1_curve.pt",
    "stage191_p2_gpt.pt",
)


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return 1

    CKPT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = CKPT / name
        if dest.exists():
            print(f"OK exists: {dest}")
            continue
        print(f"Downloading {HF_REPO}/{name} ...")
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=name,
            local_dir=str(CKPT),
            local_dir_use_symlinks=False,
        )
        print(f"  -> {path}")
    print("Done. Run: python artifact/scripts/run_demo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
