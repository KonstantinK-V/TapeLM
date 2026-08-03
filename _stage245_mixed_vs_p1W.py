"""
Stage 245 — Mixed-scratch encoder without W vs product P1 + W after code shift.

Uses checkpoints/stage238_mixed_scratch.pt and stage191_p1_curve.pt.
Same fact strings; each encoder writes its own bank, then code-shifts.
Compare mixed recall(no W) vs P1 recall(W).

  python _stage245_mixed_vs_p1W.py [--smoke]
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

SEED = 245
DECISION = L.RES / "stage245_decision.json"
MINI = L.RES / "stage245_mini.md"
LOG = L.RES / "_stage245_log.txt"


def arm(name, model, stoi, core, facts, all_values, tok, pad_id, char_table, device, smoke, arc_steps, w_steps, rng, use_W: bool):
    bank = FpBank(model, stoi, device)
    K, vals = L.write_tape_bank(bank, facts)
    a0 = L.tape_recall(facts, all_values, bank, K, vals, SEED)
    code = s227.ensure_code(random.Random(SEED + 1), smoke)
    flat_c, off_c = s213.build_flat_from_text(
        code, tok, pad_id, max_lines=300 if smoke else 8000, min_line_len=20
    )
    F_can = s221.fp_matrix(bank, core)
    model_b = s221.finetune_arc_enc(model, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 7)
    bank_b = FpBank(model_b, stoi, device)
    W_bwd, align = s221.train_remap(
        DomainAdapter(256).to(device), s221.fp_matrix(bank_b, core), F_can, rng, w_steps, device
    )
    a_raw = L.tape_recall(facts, all_values, bank_b, K, vals, SEED, W_bwd=None)
    a_W = L.tape_recall(facts, all_values, bank_b, K, vals, SEED, W_bwd=W_bwd)
    chosen = a_W if use_W else a_raw
    return {
        "name": name,
        "A0": a0,
        "A_raw": a_raw,
        "A_W": a_W,
        "chosen": chosen,
        "use_W": use_W,
        "W_align": align,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    LOG.write_text("", encoding="utf-8")
    log = L.make_logger(LOG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    if not L.CKPT_MIXED.exists():
        raise SystemExit(f"missing {L.CKPT_MIXED}")

    arc_steps = 50 if args.smoke else s221.ARC_STEPS
    w_steps = 60 if args.smoke else s221.W_STEPS
    core_n = 60 if args.smoke else 400
    n_facts = 12 if args.smoke else 50

    log(f"Stage245 start {datetime.now(timezone.utc).isoformat()}")
    _, _, stoi, n_char, tok, V, pad_id, char_table, model_p1, _ = L.load_p1(device)
    model_m, _ = L.load_curve_ckpt(L.CKPT_MIXED, n_char, V, stoi, device)
    _, values_pool, core, _ = L.wiki_bits(args.smoke, core_n, rng)
    facts, all_values = L.make_facts(n_facts, values_pool, rng)

    p1 = arm("P1+W", model_p1, stoi, core, facts, all_values, tok, pad_id, char_table, device, args.smoke, arc_steps, w_steps, rng, True)
    mix = arm(
        "mixed_no_W", model_m, stoi, core, facts, all_values, tok, pad_id, char_table, device, args.smoke, arc_steps, w_steps, rng, False
    )
    log(f"P1+W chosen={p1['chosen']:.3f} (raw={p1['A_raw']:.3f} W={p1['A_W']:.3f})")
    log(f"mixed noW chosen={mix['chosen']:.3f} (raw={mix['A_raw']:.3f} W={mix['A_W']:.3f})")

    gap = mix["chosen"] - p1["chosen"]
    g_p1 = p1["chosen"] >= 0.70
    g_mix = mix["chosen"] >= 0.70
    g_mix_beats = gap >= 0.05
    g_mix_raw_ge_W = mix["A_raw"] + 0.02 >= mix["A_W"]
    if g_p1 and g_mix and g_mix_beats:
        overall = "MIXED_NO_W_BEATS_P1W"
    elif g_p1 and g_mix and abs(gap) < 0.05:
        overall = "MIXED_NO_W_TIES_P1W"
    elif g_p1 and (not g_mix_beats):
        overall = "P1W_BEATS_MIXED_NO_W"
    else:
        overall = "MIXED_VS_P1W_NO"

    out = {
        "stage": 245,
        "overall": overall,
        "gates": {
            "G_p1W_floor_0p70": g_p1,
            "G_mixed_floor_0p70": g_mix,
            "G_mixed_beats_p1W_0p05": g_mix_beats,
            "G_mixed_raw_ge_own_W": g_mix_raw_ge_W,
        },
        "p1": p1,
        "mixed": mix,
        "gap_mixed_minus_p1": gap,
        "note": "Unexpected if mixed-no-W >= P1+W on same fact strings after code shift.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    L.dump(DECISION, MINI, out, "Stage 245 mixed no-W vs P1+W")
    log(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
