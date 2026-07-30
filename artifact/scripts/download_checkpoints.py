#!/usr/bin/env python3
"""Download TapeLM P1/P2 checkpoints from Hugging Face into checkpoints/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "checkpoints"
WREG = CKPT / "w_registry"

HF_REPO = "Kostya03v/TapeLM-P1"
CORE_FILES = (
    "stage191_p1_curve.pt",
    "stage191_p2_gpt.pt",
)


def download_one(hf_hub_download, repo_id: str, remote_name: str, local_root: Path) -> Path | None:
    """Download `remote_name` into `local_root` preserving subdirs."""
    dest = local_root / remote_name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"OK exists: {dest}")
        return dest
    print(f"Downloading {repo_id}/{remote_name} ...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_name,
            local_dir=str(local_root),
            local_dir_use_symlinks=False,
        )
        print(f"  -> {path}")
        return Path(path)
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def w_registry_remote_paths(manifest: dict) -> list[str]:
    paths = ["w_registry/w_registry.json"]
    seen = set(paths)
    for spec in manifest.get("families", {}).values():
        for rel in spec.get("files", {}).values():
            remote = f"w_registry/{rel}"
            if remote not in seen:
                seen.add(remote)
                paths.append(remote)
    return paths


def download_w_registry(hf_hub_download) -> bool:
    manifest_path = download_one(hf_hub_download, HF_REPO, "w_registry/w_registry.json", CKPT)
    if manifest_path is None or not manifest_path.is_file():
        print("w_registry not on Hugging Face yet.")
        print("  Upload: python artifact/scripts/upload_w_registry.py")
        print("  Or local: python artifact/scripts/export_w_registry.py")
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = True
    for remote in w_registry_remote_paths(manifest):
        if remote.endswith("w_registry.json"):
            continue
        if download_one(hf_hub_download, HF_REPO, remote, CKPT) is None:
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Download TapeLM weights from Hugging Face")
    ap.add_argument(
        "--with-w-registry",
        action="store_true",
        help="Also download checkpoints/w_registry/ (family W for cross-domain memory)",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return 1

    CKPT.mkdir(parents=True, exist_ok=True)
    for name in CORE_FILES:
        if download_one(hf_hub_download, HF_REPO, name, CKPT) is None:
            return 1

    if args.with_w_registry:
        WREG.mkdir(parents=True, exist_ok=True)
        if not download_w_registry(hf_hub_download):
            return 2

    print("Done.")
    print("  Product demo: python artifact/scripts/run_product.py")
    if not args.with_w_registry:
        print("  Family W:     python artifact/scripts/download_checkpoints.py --with-w-registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
