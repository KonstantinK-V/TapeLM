#!/usr/bin/env python3
"""
Upload checkpoints/w_registry/ to Hugging Face (TapeLM-P1).

Requires: hf auth login (or HF_TOKEN), local folder from export_w_registry.py

  python artifact/scripts/export_w_registry.py        # or --smoke
  python artifact/scripts/upload_w_registry.py
  python artifact/scripts/upload_w_registry.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WREG = REPO / "checkpoints" / "w_registry"
HF_REPO = "Kostya03v/TapeLM-P1"


def hf_cmd() -> list[str]:
    if shutil.which("hf"):
        return ["hf", "upload"]
    return ["huggingface-cli", "upload"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print command only")
    args = ap.parse_args()

    manifest = WREG / "w_registry.json"
    if not manifest.is_file():
        print(f"Missing {manifest}")
        print("  python artifact/scripts/export_w_registry.py")
        return 1

    base = hf_cmd()
    cmd = [
        *base,
        HF_REPO,
        str(WREG),
        "w_registry",
        "--repo-type",
        "model",
    ]
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        print("Auth required: hf auth login   (or set HF_TOKEN)")
        print("Docs: docs/HUGGINGFACE.md")
        return r.returncode
    print(f"Uploaded. Verify: https://huggingface.co/{HF_REPO}/tree/main/w_registry")
    print("Visitors: python artifact/scripts/download_checkpoints.py --with-w-registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
