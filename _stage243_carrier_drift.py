"""
Stage 243 — Same domain-B corpus, different memory carrier.

A facts acquired in TapeLM slots and GPT weights; adapt on identical code corpus:
  TapeLM — query arc_enc shift + W (slots untouched)
  GPT    — CE overwrite (parametric carrier)

Reports which carrier retains A. Related to 239; framed as carrier contrast.

  python _stage243_carrier_drift.py [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime, timezone

import torch

import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
import _stage24x_lib as L
from _stage194_fp_fact_memory import FpBank
from _stage196_tapelm import load_gpt
from _tapelm_ext import DomainAdapter

SEED = 243
DECISION = L.RES / "stage243_decision.json"
MINI = L.RES / "stage243_mini.md"
LOG = L.RES / "_stage243_log.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    LOG.write_text("", encoding="utf-8")
    log = L.make_logger(LOG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = __import__("random").Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    n_facts = 12 if args.smoke else 40
    ft_steps = 240 if args.smoke else 2400
    b_steps = 400 if args.smoke else 1600
    arc_steps = 60 if args.smoke else s221.ARC_STEPS
    w_steps = 80 if args.smoke else s221.W_STEPS
    core_n = 60 if args.smoke else 400
    n_next = 40 if args.smoke else 120
    n_batch, ft_len, ft_lr, b_lr = 8, 64, 3e-4, 5e-4
    mem_target = 0.72

    log(f"Stage243 start {datetime.now(timezone.utc).isoformat()}")
    _, _, stoi, _, tok, _, pad_id, char_table, model, bank = L.load_p1(device)
    _, values_pool, core, paras = L.wiki_bits(args.smoke, core_n, rng)
    facts, all_values = L.make_facts(n_facts, values_pool, rng)
    items = L.load_next_tok_items(n_next)
    K, vals = L.write_tape_bank(bank, facts)
    tape0 = L.tape_recall(facts, all_values, bank, K, vals, SEED)
    nt0 = L.curve_next_tok(model, char_table, pad_id, items, device)

    gm = copy.deepcopy(load_gpt(device))
    used, _, _ = L.memorize_gpt(
        gm, tok, pad_id, facts, all_values, paras, device, SEED, ft_steps, n_batch, ft_len, ft_lr, mem_target,
        40 if args.smoke else 100, log,
    )
    gpt0 = L.gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, SEED)

    code_text = s227.ensure_code(__import__("random").Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(
        code_text, tok, pad_id, max_lines=300 if args.smoke else 8000, min_line_len=20
    )
    code_ids = [i for i in tok.encode(code_text[:200_000]).ids if i != pad_id]
    F_can = s221.fp_matrix(bank, core)
    model_b = s221.finetune_arc_enc(model, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 7)
    bank_b = FpBank(model_b, stoi, device)
    W_bwd, align = s221.train_remap(
        DomainAdapter(256).to(device), s221.fp_matrix(bank_b, core), F_can, rng, w_steps, device
    )
    tape1 = L.tape_recall(facts, all_values, bank_b, K, vals, SEED, W_bwd=W_bwd)
    nt1 = L.curve_next_tok(model, char_table, pad_id, items, device)

    L.code_ce(gm, code_ids, n_batch, ft_len, b_lr, b_steps, device, SEED, log)
    gpt1 = L.gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, SEED)
    gpt_nt1 = L.gpt_next_tok(gm, items, device)
    log(f"carriers after same code-B: slots+W={tape1:.3f} weights={gpt1:.3f} ({time.time()-t0:.0f}s)")

    gap = tape1 - gpt1
    g_mem = tape0 >= 0.70 and gpt0 >= 0.70
    g_slots = tape1 >= 0.80
    g_weights_drop = (gpt0 - gpt1) >= 0.15
    g_gap = gap >= 0.20
    g_gen_safe = abs(nt1 - nt0) < 1e-9 or (nt1 >= nt0 - 0.02)
    if g_mem and g_slots and g_weights_drop and g_gap:
        overall = "CARRIER_DRIFT_OK"
    elif g_mem and g_slots and (g_weights_drop or g_gap >= 0.10):
        overall = "CARRIER_DRIFT_PARTIAL"
    else:
        overall = "CARRIER_DRIFT_NO"

    out = {
        "stage": 243,
        "overall": overall,
        "gates": {
            "G_memorize": g_mem,
            "G_slots_retain_ge_0p80": g_slots,
            "G_weights_drop_ge_0p15": g_weights_drop,
            "G_gap_ge_0p20": g_gap,
            "G_frozen_gen_stable": g_gen_safe,
        },
        "slots": {"A0": tape0, "A1": tape1, "next_tok_0": nt0, "next_tok_1": nt1},
        "weights": {"A0": gpt0, "A1": gpt1, "drop": gpt0 - gpt1, "next_tok_1": gpt_nt1},
        "gap_slots_minus_weights": gap,
        "W_align": align,
        "memorize_steps": used,
        "note": "Same B corpus; carrier = slots+W vs parametric weights.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    L.dump(DECISION, MINI, out, "Stage 243 carrier drift")
    log(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
