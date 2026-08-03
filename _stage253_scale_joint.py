"""
Stage 253 — Scale joint CE + 0.2*CPC (252 fork SCALE_JOINT_TOKENS).

Single arm lambda=0.2, 16M CE tokens from P1, no early stop (full budget burn).
252 @4M reference: nt=0.850, hold_ce=4.199, gap=+0.137.

  python _stage253_scale_joint.py [--smoke] [--token-budget N]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank

RES = Path("results")
DECISION = RES / "stage253_decision.json"
MINI = RES / "stage253_mini.md"
LOG = RES / "_stage253_log.txt"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
CKPT_OUT = Path("checkpoints/stage253_joint_l02.pt")
SEED = 253
LAM = 0.2
REF_252 = {"next_tok": 0.85, "hold_ce": 4.199211339155833, "gap": 0.13654892843961712}


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--token-budget", type=int, default=0)
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    tb = args.token_budget or (200_000 if args.smoke else 16_000_000)
    n_facts = 8 if args.smoke else 20
    n_exam = 40 if args.smoke else 120
    n_probe = 24 if args.smoke else 60
    n_hold = 8 if args.smoke else 32
    n_probes = 8 if args.smoke else 16

    log(f"Stage253 scale joint CE+{LAM}*CPC start {datetime.now(timezone.utc).isoformat()} budget={tb}")
    flat, off, stoi, n_char = load_data()
    train_docs, hold_docs = s251.split_train_hold(off)
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model0 = SelfModelXL(n_char, V).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model0.eval()
    for p in model0.parameters():
        p.requires_grad_(False)

    facts, all_values = s251.make_facts(rng, n_facts, args.smoke)
    bank0 = FpBank(model0, stoi, device)
    K, Vlist = L.write_tape_bank(bank0, facts)
    items = s251.load_exam_next(n_exam)
    items_probe = items[:n_probe]
    hold_batches = s252.make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)

    base_eval = s252.evaluate(
        model0, char_table, pad_id, tok, stoi, device, flat, off, hold_docs, hold_batches,
        items, facts, all_values, K, Vlist,
    )
    log(
        f"baseline nt={base_eval['next_tok']:.3f} hold={base_eval['hold_ce']:.3f} "
        f"gap={base_eval['inversion']['gap_hard_minus_para']:+.3f}"
    )

    m, meta = s252.train_joint(
        model0, flat, off, char_table, pad_id, device, tb, LAM, SEED + 1, "scale_l02",
        train_docs, hold_batches, items_probe, early_stop=False, n_probes=n_probes,
    )
    ev = s252.evaluate(
        m, char_table, pad_id, tok, stoi, device, flat, off, hold_docs, hold_batches,
        items, facts, all_values, K, Vlist,
    )
    log(
        f"DONE nt={ev['next_tok']:.3f} hold={ev['hold_ce']:.3f} "
        f"gap={ev['inversion']['gap_hard_minus_para']:+.3f} unif={ev['uniformity']:.3f} "
        f"mem={ev['slot_mem']:.3f} leak={ev['param_leak']:.3f} wall={time.time()-t0:.0f}s"
    )

    g_nt = ev["next_tok"] >= REF_252["next_tok"] - 0.01
    g_gap = ev["inversion"]["gap_hard_minus_para"] <= REF_252["gap"] - 0.005
    g_hold = ev["hold_ce"] <= REF_252["hold_ce"] + 0.03
    g_mem = ev["slot_mem"] >= 0.75 and ev["param_leak"] <= 0.40
    g_scale = meta["tokens_ce"] >= tb * 0.98

    if g_nt and g_gap and g_hold and g_mem and g_scale:
        overall = "SCALE_JOINT_OK"
    elif g_mem and g_scale and (g_nt or g_gap):
        overall = "SCALE_JOINT_PARTIAL"
    else:
        overall = "SCALE_JOINT_NO"

    out = {
        "stage": 253,
        "overall": overall,
        "lambda": LAM,
        "token_budget": tb,
        "tokens_ce": meta["tokens_ce"],
        "reference_252_4M": REF_252,
        "gates": {
            "G_nt_vs_252": g_nt,
            "G_gap_vs_252": g_gap,
            "G_hold_vs_252": g_hold,
            "G_memory_clean": g_mem,
            "G_full_budget": g_scale,
        },
        "baseline": base_eval,
        "final": ev,
        "train_meta": meta,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 253 scale joint (λ={LAM})\n\n**{overall}** budget={tb} tokens_ce={meta['tokens_ce']}\n"
        f"nt {base_eval['next_tok']:.3f}->{ev['next_tok']:.3f} "
        f"gap {base_eval['inversion']['gap_hard_minus_para']:+.3f}->"
        f"{ev['inversion']['gap_hard_minus_para']:+.3f} hold {ev['hold_ce']:.3f}\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "nt": ev["next_tok"], "gap": ev["inversion"]["gap_hard_minus_para"]}, indent=2))

    if not args.smoke and meta["tokens_ce"] >= 500_000:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save(
            {"model": m.state_dict(), "stage": 253, "lambda": LAM, "tokens_ce": meta["tokens_ce"]},
            CKPT_OUT,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
