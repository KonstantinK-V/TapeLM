"""
Stage 226c — Joint gen + mem with official 228c fp decode (e2e trunk).

Fixes 226 protocol gaps:
  - Retrieval: 4-way qmap recall (227), not global argmax (226 was ~0.60).
  - Utilization: fp_retrieved_4way at code return position (228c), vs head_only.

  python _stage226c_joint_fp_decode.py [--smoke]
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
import _stage225_family_fork as s225
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import (
    DomainAdapter,
    apply_qmap,
    fp_cos_scores,
    fp_decode_pick_retrieved_4way,
    slot_retrieve_4way,
)

RES = Path("results")
DECISION = RES / "stage226c_decision.json"
MINI = RES / "stage226c_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 2263


@torch.no_grad()
def head_pick(model, tok, pad_id, char_table, device, prefix: str, cands: list[str]) -> str:
    ids = tok.encode(prefix).ids
    if not ids:
        return cands[0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pad = x == pad_id
    logits, _, _ = model.forward_all(char_table[x], pad, ids=x)
    last = logits[0, -1]
    scores = []
    for c in cands:
        tid = tok.encode(c).ids
        scores.append(float(last[tid[0]]) if tid else -1e9)
    return cands[int(np.argmax(scores))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    upper_steps = 80 if args.smoke else 600
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else s221.CORE_N
    n_facts = 12 if args.smoke else 60
    max_lines = 400 if args.smoke else 8000
    rng = random.Random(SEED)

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(2_000_000)
    core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:core_n]

    model0 = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model0.eval()
    bank_can = FpBank(model0, stoi, device)
    F0 = s221.fp_matrix(bank_can, core)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(4_000_000)) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_can, V = s221.build_fact_bank(bank_can, subs, vals, rng)

    text_code = s225.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    model_c = s221.finetune_arc_enc(model0, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 2)
    bank_c = FpBank(model_c, stoi, device)
    F_c = s221.fp_matrix(bank_c, core)
    W_bwd, align_bwd = s221.train_remap(DomainAdapter(256).to(device), F_c, F0, rng, w_steps, device)
    head_code = s225.train_upper(model0, flat_c, off_c, char_table, pad_id, device, upper_steps, SEED + 3)

    ctx_tpl = "In the report {S} was linked to the organization."
    prefix_tpl = "def org_of_{S}():\n    return "

    ret_ok = fp_ok = head_ok = n = 0
    for S, gold in zip(subs, vals):
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        ctx = ctx_tpl.format(S=S)
        q = bank_c.ctx_fp(ctx, exclude=gold)
        if q is None:
            continue
        qq = apply_qmap(W_bwd, q)
        hit4 = slot_retrieve_4way(K_can, V, qq, cands)
        ret_ok += int(hit4 == gold)

        _, pick_fp = fp_decode_pick_retrieved_4way(
            bank_can, K_can, V, W_bwd, bank_c, ctx, gold, cands
        )
        fp_ok += int(pick_fp == gold)

        prefix = prefix_tpl.format(S=S)
        pick_head = head_pick(head_code, tok, pad_id, char_table, device, prefix, cands)
        head_ok += int(pick_head == gold)
        n += 1

    n = max(1, n)
    recall_4way = ret_ok / n
    acc_fp = fp_ok / n
    acc_head = head_ok / n
    lift = acc_fp - acc_head

    g_ret = recall_4way >= 0.70
    g_fp = acc_fp >= 0.70 and lift >= 0.08
    overall = (
        "JOINT_FP_DECODE_OK"
        if g_ret and g_fp
        else ("JOINT_FP_DECODE_PARTIAL" if g_ret or g_fp else "JOINT_FP_DECODE_NO")
    )

    out = {
        "stage": "226c",
        "overall": overall,
        "gates": {"G_recall_4way": g_ret, "G_fp_decode_util": g_fp},
        "align_W_bwd": align_bwd,
        "mean_cos_code_shift": float((F0 * F_c).sum(-1).mean()),
        "recall_4way_qmap": recall_4way,
        "accuracy": {"head_only": acc_head, "fp_retrieved_4way": acc_fp},
        "lift_fp_minus_head": lift,
        "n_items": n,
        "note": "226 e2e: canonical bank + code qmap + 228c decode at return token",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 226c joint fp decode\n\n**{overall}** recall4={recall_4way:.3f} "
        f"fp={acc_fp:.3f} head={acc_head:.3f} lift={lift:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
