#!/usr/bin/env python3
"""
Train and persist family W matrices for canonical slot read (227 qmap).

Writes:
  checkpoints/w_registry/w_registry.json
  checkpoints/w_registry/W_prose_bwd.pt (+ optional fwd)
  checkpoints/w_registry/W_code_bwd.pt (+ optional fwd)

Canonical geometry = frozen P1 arc_enc. Each family W is fit on ~800 core words
after an arc_enc-only finetune on a representative corpus (stories / code).

  python artifact/scripts/export_w_registry.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage225_family_fork as s225
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tapelm_ext import DomainAdapter, W_REGISTRY_DIR, save_w_family

CKPT = REPO / "checkpoints/stage191_p1_curve.pt"
STORIES = REPO / "data/external_tinystories_100k_85.txt"
WIKI = REPO / "data/_wikitext103_train.txt"
SEED = 9001


def train_family_W(
    model_can: SelfModelXL,
    flat,
    off,
    char_table,
    pad_id: int,
    device: torch.device,
    stoi,
    rng: random.Random,
    core: list[str],
    F_can: torch.Tensor,
    arc_steps: int,
    w_steps: int,
    seed: int,
    family: str,
    corpus_tag: str,
) -> tuple[DomainAdapter, DomainAdapter, dict]:
    model_d = s221.finetune_arc_enc(model_can, flat, off, char_table, pad_id, device, arc_steps, seed)
    bank_d = FpBank(model_d, stoi, device)
    F_d = s221.fp_matrix(bank_d, core)
    mean_cos = float((F_can * F_d).sum(-1).mean())
    W_fwd, align_fwd = s221.train_remap(DomainAdapter(256).to(device), F_can, F_d, rng, w_steps, device)
    W_bwd, align_bwd = s221.train_remap(DomainAdapter(256).to(device), F_d, F_can, rng, w_steps, device)
    meta_base = {
        "family": family,
        "corpus": corpus_tag,
        "mean_cos_canonical_to_domain": mean_cos,
        "canonical_ckpt": "stage191_p1_curve.pt",
        "core_n": len(core),
        "arc_steps": arc_steps,
        "w_steps": w_steps,
    }
    meta_fwd = {**meta_base, "direction": "canonical_to_domain", "policy": "keylift", "align": align_fwd}
    meta_bwd = {**meta_base, "direction": "domain_to_canonical", "policy": "qmap", "align": align_bwd}
    return W_fwd, W_bwd, {"fwd": meta_fwd, "bwd": meta_bwd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else s221.CORE_N
    max_lines = 400 if args.smoke else 8000
    rng = random.Random(SEED)

    if not CKPT.is_file():
        print(f"Missing {CKPT}; run download_checkpoints.py")
        return 1

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(2_000_000)
    core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:core_n]

    model_can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model_can.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model_can.eval()
    bank_can = FpBank(model_can, stoi, device)
    F_can = s221.fp_matrix(bank_can, core)

    out_dir = REPO / W_REGISTRY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_s, off_s = s213.build_flat_from_text(
        STORIES.read_text(encoding="utf-8", errors="ignore"), tok, pad_id, max_lines=max_lines
    )
    text_code = s225.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=max_lines, min_line_len=20)

    print("Training W_prose (stories shift)…", flush=True)
    W_p_fwd, W_p_bwd, meta_p = train_family_W(
        model_can, flat_s, off_s, char_table, pad_id, device, stoi, rng, core, F_can, arc_steps, w_steps, SEED + 2, "prose", "tinystories"
    )
    print("Training W_code (code shift)…", flush=True)
    W_c_fwd, W_c_bwd, meta_c = train_family_W(
        model_can, flat_c, off_c, char_table, pad_id, device, stoi, rng, core, F_can, arc_steps, w_steps, SEED + 3, "code", "stage224_code"
    )

    files = {}
    for family, W_fwd, W_bwd, meta in (
        ("prose", W_p_fwd, W_p_bwd, meta_p),
        ("code", W_c_fwd, W_c_bwd, meta_c),
    ):
        rel_fwd = f"W_{family}_fwd.pt"
        rel_bwd = f"W_{family}_bwd.pt"
        save_w_family(out_dir / rel_fwd, W_fwd, meta["fwd"])
        save_w_family(out_dir / rel_bwd, W_bwd, meta["bwd"])
        files[family] = {
            "files": {"fwd": rel_fwd, "bwd": rel_bwd},
            "mean_cos_canonical_to_domain": meta["bwd"]["mean_cos_canonical_to_domain"],
            "align_bwd": meta["bwd"]["align"],
        }

    manifest = {
        "version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "canonical": "checkpoints/stage191_p1_curve.pt",
        "read_policy_default": "qmap",
        "decode_api": "fp_decode_pick_retrieved_4way in _tapelm_ext.py (228c)",
        "families": files,
    }
    manifest_path = out_dir / "w_registry.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
