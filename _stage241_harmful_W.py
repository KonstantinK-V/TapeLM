"""
Stage 241 — Harmful W vs no-W: does wrong-family qmap beat raw (no W)?

Canonical bank; code query encoder. Compare matched W_code, wrong W_stories, no W.

  python _stage241_harmful_W.py [--smoke]
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage194_fp_fact_memory import FpBank
from _tapelm_ext import DomainAdapter

SEED = 241
DECISION = L.RES / "stage241_decision.json"
MINI = L.RES / "stage241_mini.md"
LOG = L.RES / "_stage241_log.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    LOG.write_text("", encoding="utf-8")
    log = L.make_logger(LOG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    arc_steps = 60 if args.smoke else s221.ARC_STEPS
    w_steps = 80 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else 400
    n_facts = 12 if args.smoke else 60
    max_lines = 300 if args.smoke else 8000

    log(f"Stage241 start {datetime.now(timezone.utc).isoformat()}")
    _, _, stoi, _, tok, _, pad_id, char_table, model, bank = L.load_p1(device)
    _, values_pool, core, _ = L.wiki_bits(args.smoke, core_n, rng)
    facts, all_values = L.make_facts(n_facts, values_pool, rng)
    K, vals = L.write_tape_bank(bank, facts)
    F_can = s221.fp_matrix(bank, core)

    flat_s, off_s = s213.build_flat_from_text(
        L.STORIES.read_text(encoding="utf-8", errors="ignore"), tok, pad_id, max_lines=max_lines
    )
    code = s227.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    model_s = s221.finetune_arc_enc(model, flat_s, off_s, char_table, pad_id, device, arc_steps, SEED + 2)
    model_c = s221.finetune_arc_enc(model, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 3)
    bank_s, bank_c = FpBank(model_s, stoi, device), FpBank(model_c, stoi, device)
    W_s, _ = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_s, core), F_can, rng, w_steps, device)
    W_c, align = s221.train_remap(
        DomainAdapter(256).to(device), s221.fp_matrix(bank_c, core), F_can, rng, w_steps, device
    )

    acc_none = L.tape_recall(facts, all_values, bank_c, K, vals, SEED, W_bwd=None)
    acc_wrong = L.tape_recall(facts, all_values, bank_c, K, vals, SEED, W_bwd=W_s)
    acc_match = L.tape_recall(facts, all_values, bank_c, K, vals, SEED, W_bwd=W_c)
    log(f"code query: none={acc_none:.3f} wrong_stories_W={acc_wrong:.3f} matched={acc_match:.3f}")

    hurt = acc_none - acc_wrong
    help_m = acc_match - acc_none
    g_hurt = hurt >= 0.05
    g_match_helps = help_m >= 0.05
    g_match_best = acc_match >= acc_wrong + 0.10
    if g_hurt and g_match_helps and g_match_best:
        overall = "WRONG_W_HURTS_OK"
    elif g_hurt or (acc_wrong < acc_none and g_match_best):
        overall = "WRONG_W_HURTS_PARTIAL"
    else:
        overall = "WRONG_W_HURTS_NO"

    out = {
        "stage": 241,
        "overall": overall,
        "gates": {
            "G_wrong_worse_than_none_by_0p05": g_hurt,
            "G_matched_helps_vs_none_0p05": g_match_helps,
            "G_matched_beats_wrong_0p10": g_match_best,
        },
        "recall": {"no_W": acc_none, "wrong_W_stories": acc_wrong, "matched_W_code": acc_match},
        "deltas": {"none_minus_wrong": hurt, "matched_minus_none": help_m},
        "W_align_code": align,
        "note": "Deploy guard: prefer no-W over wrong-family W when hurt>0.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    L.dump(DECISION, MINI, out, "Stage 241 harmful W vs no-W")
    log(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
