"""
Stage 232 — L3 stream: age decay + W_version mismatch penalty on canonical slots.

When age ties, wrong `w_version` must lose to correct era. Decay-only cannot
disambiguate; version penalty can.

  python _stage232_stream_w_version.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from _stage191_night import load_data
from _stage194_fp_fact_memory import FpBank
from _tapelm_ext import weighted_slot_sims

RES = Path("results")
DECISION = RES / "stage232_decision.json"
MINI = RES / "stage232_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
SEED = 232


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    n_subj = 24 if args.smoke else 100

    from tokenizers import Tokenizer
    import _stage177_curve_bpe as s177
    from _stage191_night import SelfModelXL

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)

    words = [f"Ent{i}" for i in range(n_subj)]
    wrong_val = [f"StaleWrong{i}" for i in range(n_subj)]
    gold_val = [f"FreshRight{i}" for i in range(n_subj)]
    keys, vals, ages, wvers = [], [], [], []
    active_w = "prose_v2"
    tau = 50.0

    for i, w in enumerate(words):
        k = bank.fp([w])[0]
        # Tie age: decay-only prefers arbitrary max cos → often wrong version
        keys.append(k)
        vals.append(wrong_val[i])
        ages.append(40)
        wvers.append("prose_v1")
        keys.append(k)
        vals.append(gold_val[i])
        ages.append(41)
        wvers.append(active_w)
        # Stale distraction (old age, correct version) — should not beat gold
        keys.append(k)
        vals.append(f"OldRight{i}")
        ages.append(500)
        wvers.append(active_w)

    K = torch.stack(keys, 0)

    def eval_mode(mode: str) -> float:
        ok, n = 0, 0
        for i, w in enumerate(words):
            q = bank.fp([w])[0]
            sims = K @ q
            if mode == "flat":
                pass
            elif mode == "decay":
                sims = weighted_slot_sims(sims, ages, wvers, active_w, tau, version_penalty=1.0)
            elif mode == "decay_version":
                sims = weighted_slot_sims(sims, ages, wvers, active_w, tau, version_penalty=0.05)
            sc = {}
            for j, v in enumerate(vals):
                sc[v] = max(sc.get(v, -1e9), float(sims[j]))
            gold = gold_val[i]
            cands = [gold, wrong_val[i], gold_val[(i + 1) % n_subj], wrong_val[(i + 1) % n_subj]]
            rng.shuffle(cands)
            g = cands.index(gold)
            ok += int(int(np.argmax([sc[c] for c in cands])) == g)
            n += 1
        return ok / max(1, n)

    acc_flat = eval_mode("flat")
    acc_decay = eval_mode("decay")
    acc_dv = eval_mode("decay_version")
    g1 = acc_dv >= acc_flat + 0.15
    g2 = acc_dv >= acc_decay + 0.10
    overall = "STREAM_W_VERSION_OK" if g1 and g2 else ("STREAM_W_VERSION_PARTIAL" if g1 else "STREAM_W_VERSION_NO")

    out = {
        "stage": 232,
        "overall": overall,
        "gates": {"G_decay_version_beats_flat": g1, "G_decay_version_beats_decay_only": g2},
        "acc_flat": acc_flat,
        "acc_decay_only": acc_decay,
        "acc_decay_plus_w_version": acc_dv,
        "active_w_version": active_w,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(f"# Stage 232 stream + W version\n\n**{overall}** dv={acc_dv:.3f} decay={acc_decay:.3f} flat={acc_flat:.3f}\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
