"""
Stage 228c — Fix 228b: align retrieve with 227, then fp-guided decode.

228b failed because global argmax retrieve was ~33% exact while 227 4-way qmap
was ~0.95. Oracle fp-scorer was 1.0 → mechanism OK, protocol mismatch.

Modes (4-way org pick at code return position):
  head_only          — LM first-BPE logit
  fp_query           — cos(fp_can(c), W_bwd(q_code))   [= 227 recall as decode]
  fp_retrieved_4way  — 4-way retrieve value, then cos(fp(c), fp(retrieved))
  fp_retrieved_global— 228b broken protocol (global argmax), for contrast
  fp_oracle          — cos(fp(c), fp(gold))
  hybrid_query       — zscore(head) + zscore(fp_query)

  python _stage228c_fp_decode_fix.py [--smoke]
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
    slot_retrieve_4way,
    slot_retrieve_global,
)

RES = Path("results")
DECISION = RES / "stage228c_decision.json"
MINI = RES / "stage228c_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 2283


def zscore(xs: list[float]) -> list[float]:
    a = np.asarray(xs, dtype=np.float64)
    if len(a) < 2:
        return xs
    m, s = a.mean(), a.std()
    if s < 1e-9:
        return [0.0] * len(xs)
    return ((a - m) / s).tolist()


@torch.no_grad()
def head_scores(model, tok, pad_id, char_table, device, prefix: str, cands: list[str]) -> list[float]:
    ids = tok.encode(prefix).ids
    if not ids:
        return [-1e9] * len(cands)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pad = x == pad_id
    logits, _, _ = model.forward_all(char_table[x], pad, ids=x)
    last = logits[0, -1]
    out = []
    for c in cands:
        tid = tok.encode(c).ids
        out.append(float(last[tid[0]]) if tid else -1e9)
    return out


@torch.no_grad()
def fp_vs_anchor(bank: FpBank, anchor_fp: torch.Tensor, cands: list[str]) -> list[float]:
    fps = bank.fp(cands)
    return [float((fps[i] * anchor_fp).sum()) for i in range(len(cands))]


def pick(scores: list[float], cands: list[str], gold: str) -> bool:
    return cands[int(np.argmax(scores))] == gold


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

    rng227 = random.Random(227)
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(4_000_000)) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng227, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_can, V = s221.build_fact_bank(bank_can, subs, vals, rng227)

    text_code = s225.ensure_code(random.Random(SEED + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    model_c = s221.finetune_arc_enc(model0, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 2)
    bank_c = FpBank(model_c, stoi, device)
    F_c = s221.fp_matrix(bank_c, core)
    W_bwd, align = s221.train_remap(DomainAdapter(256).to(device), F_c, F0, rng227, w_steps, device)
    head_code = s225.train_upper(model0, flat_c, off_c, char_table, pad_id, device, upper_steps, SEED + 3)

    keys = [
        "head_only",
        "fp_query",
        "fp_retrieved_4way",
        "fp_retrieved_global",
        "fp_oracle",
        "hybrid_query",
    ]
    acc = {k: 0 for k in keys}
    ret4 = ret_g = 0
    n = 0

    for S, gold in zip(subs, vals):
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        prefix = f"def org_of_{S}():\n    return "
        q = bank_c.ctx_fp(f"In the report {S} was linked to the organization.", exclude=gold)
        if q is None:
            continue
        qq = apply_qmap(W_bwd, q)

        hit4 = slot_retrieve_4way(K_can, V, qq, cands)
        hit_g = slot_retrieve_global(K_can, V, qq)
        ret4 += int(hit4 == gold)
        ret_g += int(hit_g == gold)

        hs = head_scores(head_code, tok, pad_id, char_table, device, prefix, cands)
        fq = fp_vs_anchor(bank_can, qq, cands)
        fr4 = fp_cos_scores(bank_can, hit4, cands)
        frg = fp_cos_scores(bank_can, hit_g, cands)
        fo = fp_cos_scores(bank_can, gold, cands)
        hy = [a + b for a, b in zip(zscore(hs), zscore(fq))]

        acc["head_only"] += int(pick(hs, cands, gold))
        acc["fp_query"] += int(pick(fq, cands, gold))
        acc["fp_retrieved_4way"] += int(pick(fr4, cands, gold))
        acc["fp_retrieved_global"] += int(pick(frg, cands, gold))
        acc["fp_oracle"] += int(pick(fo, cands, gold))
        acc["hybrid_query"] += int(pick(hy, cands, gold))
        n += 1

    n = max(1, n)
    rates = {k: acc[k] / n for k in keys}
    ret4_r, ret_g_r = ret4 / n, ret_g / n
    lift = rates["fp_retrieved_4way"] - rates["head_only"]
    g_fp = rates["fp_retrieved_4way"] >= 0.70 and lift >= 0.08
    overall = "FP_DECODE_FIX_YES" if g_fp else ("FP_DECODE_FIX_PARTIAL" if lift > 0.03 else "FP_DECODE_FIX_NO")

    out = {
        "stage": "228c",
        "overall": overall,
        "contract": "decode-time: 4-way slot retrieve (227) → fp-scorer vs retrieved value; not raw query·fp(c)",
        "gates": {
            "G_fp_ret4_ge_0p70": rates["fp_retrieved_4way"] >= 0.70,
            "G_lift_ge_0p08": lift >= 0.08,
            "G_ret4_exact": ret4_r >= 0.90,
        },
        "retrieve_exact": {"four_way": ret4_r, "global_argmax": ret_g_r},
        "align_W_bwd": align,
        "mean_cos_code": float((F0 * F_c).sum(-1).mean()),
        "accuracy_4way": rates,
        "lift_fp_retrieved_4way_minus_head": lift,
        "lift_fp_query_minus_head": rates["fp_query"] - rates["head_only"],
        "n_items": n,
        "note": (
            "fp_retrieved_4way=YES path; fp_query fails because qq is ctx mix not value fp; "
            "fp_retrieved_global reproduces 228b (~global argmax)"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 228c fp decode fix\n\n**{overall}** head={rates['head_only']:.3f} "
        f"fp_query={rates['fp_query']:.3f} ret4={rates['fp_retrieved_4way']:.3f} "
        f"ret_global={rates['fp_retrieved_global']:.3f} oracle={rates['fp_oracle']:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
