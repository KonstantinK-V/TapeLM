"""
Stage 221-probe — characterise W-remap (geometry, OOV, multi-domain, incremental).

Runs after the same protocol as 221; writes results/stage221_probe_decision.json.

  python _stage221_probe.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter

RES = Path("results")
DECISION = RES / "stage221_probe_decision.json"
MINI = RES / "stage221_probe_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
DOMAIN_B = Path("data/external_tinystories_100k_85.txt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 2211

OOV_WORDS = [
    "Zorblax",
    "Qjxtrv",
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "floccinaucinihilipilification",
    "xyzabc",
]


@torch.no_grad()
def align_pairs(module: DomainAdapter, F_old: torch.Tensor, F_new: torch.Tensor) -> float:
    pred = F.normalize(module.map_raw(F_old), dim=-1)
    return float((pred * F_new).sum(-1).mean())


@torch.no_grad()
def w_gram_stats(W: torch.Tensor) -> dict:
    """W is (d,d) linear map fp' = W @ fp (column convention: Linear weight)."""
    g = W.T @ W
    d = g.shape[0]
    I = torch.eye(d, device=g.device, dtype=g.dtype)
    frob = float((g - I).pow(2).mean().sqrt())
    sym_frob = float((g - g.T).pow(2).mean().sqrt())
    evals = torch.linalg.eigvalsh(g).cpu().numpy()
    sv = torch.linalg.svdvals(W).cpu().numpy()
    det = float(torch.linalg.det(W))
    return {
        "WtW_frobenius_to_I": frob,
        "WtW_asymmetry": sym_frob,
        "WtW_eig_min": float(evals.min()),
        "WtW_eig_max": float(evals.max()),
        "WtW_eig_mean": float(evals.mean()),
        "singular_value_min": float(sv.min()),
        "singular_value_max": float(sv.max()),
        "singular_value_mean": float(sv.mean()),
        "det_W": det,
        "interpretation": (
            "near_orthogonal" if frob < 0.15 and sym_frob < 0.05 else "general_linear_warp"
        ),
    }


def train_w_on_indices(
    F_old: torch.Tensor,
    F_new: torch.Tensor,
    idx: list[int],
    steps: int,
    device: torch.device,
    seed: int,
) -> tuple[DomainAdapter, float]:
    rng = random.Random(seed)
    mod = DomainAdapter(256).to(device)
    Fo, Fn = F_old[idx], F_new[idx]
    mod, _ = s221.train_remap(mod, Fo, Fn, rng, steps, device, orth=True)
    return mod, align_pairs(mod, Fo, Fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 120 if args.smoke else s221.CORE_N
    n_facts = 12 if args.smoke else 60
    rng = random.Random(SEED)

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(2_000_000)
    core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:core_n]
    hold_n = max(20, core_n // 4)
    train_words = core[:-hold_n]
    hold_words = core[-hold_n:]

    model_old = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model_old.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model_old.eval()
    bank_old = FpBank(model_old, stoi, device)
    F_old_full = s221.fp_matrix(bank_old, core)

    text_b = DOMAIN_B.read_text(encoding="utf-8", errors="ignore")
    flat_b, off_b = s213.build_flat_from_text(text_b, tok, pad_id, max_lines=500 if args.smoke else 8000)
    flat_w, off_w = flat, off

    model_b = s221.finetune_arc_enc(model_old, flat_b, off_b, char_table, pad_id, device, arc_steps, SEED + 1)
    bank_b = FpBank(model_b, stoi, device)
    F_new_b = s221.fp_matrix(bank_b, core)

    train_idx = [core.index(w) for w in train_words]
    hold_idx = [core.index(w) for w in hold_words]

    Wmap, _ = train_w_on_indices(F_old_full, F_new_b, train_idx, w_steps, device, SEED + 2)
    W = Wmap.w.weight.detach()
    gram = w_gram_stats(W)

    align_train = align_pairs(Wmap, F_old_full[train_idx], F_new_b[train_idx])
    align_hold = align_pairs(Wmap, F_old_full[hold_idx], F_new_b[hold_idx])
    oov_list = list(dict.fromkeys(OOV_WORDS + hold_words[:5]))
    Fo_oov = s221.fp_matrix(bank_old, oov_list)
    Fn_oov = s221.fp_matrix(bank_b, oov_list)
    align_oov = align_pairs(Wmap, Fo_oov, Fn_oov)
    cos_raw_oov = float((Fo_oov * Fn_oov).sum(-1).mean())
    cos_raw_hold = float((F_old_full[hold_idx] * F_new_b[hold_idx]).sum(-1).mean())

    inc_sizes = [100, 200, 400, 800] if not args.smoke else [30, 60, 90]
    incremental = []
    for n in inc_sizes:
        if n > len(train_idx):
            continue
        sub = train_idx[:n]
        mod_n, _ = train_w_on_indices(F_old_full, F_new_b, sub, w_steps, device, SEED + 100 + n)
        incremental.append(
            {
                "n_train_words": n,
                "align_train_subset": align_pairs(mod_n, F_old_full[sub], F_new_b[sub]),
                "align_hold_out": align_pairs(mod_n, F_old_full[hold_idx], F_new_b[hold_idx]),
            }
        )

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(4_000_000)) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_old, V = s221.build_fact_bank(bank_old, subs, vals, rng)

    def tr(Wmod):
        return lambda K: F.normalize(Wmod.map_raw(K), dim=-1)

    acc_mismatch, _ = s221.recall_at(K_old, V, bank_b, subs, vals, rng, None)
    acc_w_b, _ = s221.recall_at(K_old, V, bank_old, subs, vals, rng, tr(Wmap))

    model_c = s221.finetune_arc_enc(model_old, flat_w, off_w, char_table, pad_id, device, arc_steps, SEED + 3)
    bank_c = FpBank(model_c, stoi, device)
    F_new_c = s221.fp_matrix(bank_c, core)
    Wc, _ = train_w_on_indices(F_old_full, F_new_c, train_idx, w_steps, device, SEED + 4)
    acc_w_c, _ = s221.recall_at(K_old, V, bank_old, subs, vals, rng, tr(Wc))

    oov_generalizes = align_oov >= 0.80 * align_train and align_oov >= align_hold - 0.05
    multi_ok = acc_w_b >= 0.70 and acc_w_c >= 0.70
    inc_ok = len(incremental) >= 2 and incremental[-1]["align_hold_out"] >= incremental[0]["align_hold_out"] - 0.08

    out = {
        "stage": "221-probe",
        "overall": "W_REMAP_CHARACTERIZED",
        "q1_W_linear_gram": gram,
        "q2_oov": {
            "train_align": align_train,
            "holdout_align": align_hold,
            "holdout_mean_cos_raw": cos_raw_hold,
            "oov_align_after_W": align_oov,
            "oov_mean_cos_raw": cos_raw_oov,
            "oov_words": oov_list,
            "generalizes_beyond_core": oov_generalizes,
            "note": "holdout = core words not used in W train; OOV includes nonsense/long forms",
        },
        "q3_multi_W": {
            "recall_W_B_stories_shift": acc_w_b,
            "recall_W_C_wiki_shift": acc_w_c,
            "recall_mismatch_old_keys_new_queries_no_W": acc_mismatch,
            "note": "W_B/W_C each trained for its domain shift; same A-era slot keys; 221-style W on keys+queries",
            "one_bank_two_projections_both_ge_0p70": multi_ok,
        },
        "q4_incremental": {"curve": incremental, "monotone_enough": inc_ok},
        "prior_test_validation": {
            "mean_cos_word_fp_after_B_shift": float((F_old_full * F_new_b).sum(-1).mean()),
            "note": "221 exam uses 4-way MC + composite slot keys; word cos ~0.68 is real but not same as slot recall",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 221-probe\n\nOOV={out['q2_oov']['oov_align_after_W']:.3f} "
        f"multi={out['q3_multi_W']['one_bank_two_projections_both_ge_0p70']}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
