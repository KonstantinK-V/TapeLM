"""Audit stage-255 recall metrics on a saved tape (CPU)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _stage24x_lib as L
import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from tokenizers import Tokenizer

RECALL_SEED = 255 + 9000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="results/stream255")
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    run = Path(args.run)
    st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    facts = [f for fs in st["probe_facts"].values() for f in fs]
    d = torch.load(run / "tape.pt", map_location="cpu", weights_only=False)
    values, K = d["values"], d["K"]

    device = torch.device("cpu")
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load("checkpoints/stage191_p1_curve.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)
    all_vals = list(dict.fromkeys([f["value"] for f in facts] + values))

    fixed = L.tape_recall_metrics(facts, all_vals, bank, K, values, RECALL_SEED)
    print(f"tape slots={len(values)} probe_facts={len(facts)}")
    print(f"fixed-seed metrics: {fixed}")

    import random

    draws = []
    for s in range(args.seeds):
        m = L.tape_recall_metrics(facts, all_vals, bank, K, values, 1000 + s)
        draws.append(m["four_way"])
    print(
        f"4-way over {args.seeds} distractor seeds: mean={np.mean(draws):.3f} sd={np.std(draws):.3f} "
        f"min={min(draws):.3f} max={max(draws):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
