"""
Stage 242 — GPT rehearsal dose during domain-B: how much A mix to match TapeLM retain.

After shared A acquire + TapeLM code+W retain, sweep GPT code CE with rehearsal in
{0, 0.05, 0.15, 0.30, 0.50}. Report minimal dose where GPT A >= tape_A - 0.05.

  python _stage242_rehearsal_dose.py [--smoke]
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

SEED = 242
DECISION = L.RES / "stage242_decision.json"
MINI = L.RES / "stage242_mini.md"
LOG = L.RES / "_stage242_log.txt"
RATES = [0.0, 0.05, 0.15, 0.30, 0.50]


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

    n_facts = 10 if args.smoke else 32
    ft_steps = 200 if args.smoke else 2000
    b_steps = 200 if args.smoke else 800
    arc_steps = 50 if args.smoke else s221.ARC_STEPS
    w_steps = 60 if args.smoke else s221.W_STEPS
    core_n = 60 if args.smoke else 400
    n_batch, ft_len, ft_lr, b_lr = 8, 64, 3e-4, 5e-4
    mem_target = 0.72
    rates = [0.0, 0.15, 0.50] if args.smoke else RATES

    log(f"Stage242 start {datetime.now(timezone.utc).isoformat()}")
    _, _, stoi, _, tok, _, pad_id, char_table, model, bank = L.load_p1(device)
    _, values_pool, core, paras = L.wiki_bits(args.smoke, core_n, rng)
    facts, all_values = L.make_facts(n_facts, values_pool, rng)
    K, vals = L.write_tape_bank(bank, facts)

    gm0 = copy.deepcopy(load_gpt(device))
    used, fact_ids, _ = L.memorize_gpt(
        gm0, tok, pad_id, facts, all_values, paras, device, SEED, ft_steps, n_batch, ft_len, ft_lr, mem_target,
        40 if args.smoke else 100, log,
    )
    gpt0 = L.gpt_fact_recall(gm0, tok, pad_id, facts, all_values, device, SEED)

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
    target = tape1 - 0.05
    log(f"tape retain={tape1:.3f} gpt0={gpt0:.3f} target_gpt>={target:.3f}")

    curve = {}
    min_dose = None
    for r in rates:
        gm = copy.deepcopy(gm0)
        L.code_ce(
            gm, code_ids, n_batch, ft_len, b_lr, b_steps, device, SEED + int(r * 100), log,
            tag=f"reh{r}", fact_ids=fact_ids, rehearsal=r,
        )
        acc = L.gpt_fact_recall(gm, tok, pad_id, facts, all_values, device, SEED)
        curve[str(r)] = acc
        log(f"  rehearsal={r:.2f} -> A={acc:.3f}")
        if min_dose is None and acc >= target:
            min_dose = r

    g_tape = tape1 >= 0.80
    g_zero_fails = curve[str(rates[0])] < target
    g_found = min_dose is not None
    if g_tape and g_zero_fails and g_found:
        overall = "REHEARSAL_DOSE_OK"
    elif g_tape and (g_zero_fails or g_found):
        overall = "REHEARSAL_DOSE_PARTIAL"
    else:
        overall = "REHEARSAL_DOSE_NO"

    out = {
        "stage": 242,
        "overall": overall,
        "gates": {
            "G_tape_retain_ge_0p80": g_tape,
            "G_zero_rehearsal_below_target": g_zero_fails,
            "G_found_dose": g_found,
        },
        "tape_A_after_B": tape1,
        "gpt_A0": gpt0,
        "target_gpt": target,
        "min_rehearsal_to_match": min_dose,
        "curve": curve,
        "W_align": align,
        "memorize_steps": used,
        "note": "Price of anti-CF in weights = fraction of A tokens mixed into B CE.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    L.dump(DECISION, MINI, out, "Stage 242 rehearsal dose")
    log(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
