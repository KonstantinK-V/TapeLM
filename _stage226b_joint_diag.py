"""
Stage 226b — Diagnose 226 NO: reconcile recall vs 227 + utilization inject forms.

Hypothesis check:
  H1: 0.60 vs 0.95 is exam/seed variance (same factual query form as 227, not code-gen query).
  H2: gold_inject == no_inject because head_code ignores prose; code-native comment may help (path C).

Does NOT do joint SFT (path B) — keeps zero-train substrate narrative.

  python _stage226b_joint_diag.py [--smoke]
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
from _tapelm_ext import DomainAdapter

RES = Path("results")
DECISION = RES / "stage226b_decision.json"
MINI = RES / "stage226b_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 2262  # diag; also report SEED 227-matched subset


def log(m: str) -> None:
    print(m, flush=True)


def w_apply(W):
    return lambda X: F.normalize(W.map_raw(X), dim=-1)


def recall_qmap(K, V, bank_q, W_bwd, subs, vals, rng):
    ok = n = 0
    for S, gold in zip(subs, vals):
        q = bank_q.ctx_fp(f"In the report {S} was linked to the organization.", exclude=gold)
        if q is None:
            continue
        qq = w_apply(W_bwd)(q.unsqueeze(0))[0]
        sc = []
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        g = cands.index(gold)
        for c in cands:
            idxs = [i for i, v in enumerate(V) if v == c]
            sc.append(float((K[idxs] @ qq).max()) if idxs else -1.0)
        ok += int(np.argmax(sc) == g)
        n += 1
    return ok / max(1, n)


@torch.no_grad()
def rank4(model, tok, pad_id, char_table, device, texts_gold: list[tuple[str, str, list[str]]], rng) -> float:
    """(prompt_text, gold, distractor_pool) → 4-way rank of first BPE of gold."""
    ok = n = 0
    for text, gold, pool in texts_gold:
        ids = tok.encode(text).ids
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        pad = x == pad_id
        logits, _, _ = model.forward_all(char_table[x], pad, ids=x)
        last = logits[0, -1]
        cands = [gold] + [p for p in pool if p != gold][:3]
        while len(cands) < 4:
            cands.append(pool[len(cands) % len(pool)])
        rng.shuffle(cands)
        scores = []
        for c in cands:
            cid = tok.encode(c).ids
            scores.append(float(last[cid[0]]) if cid else -1e9)
        ok += int(cands[int(np.argmax(scores))] == gold)
        n += 1
    return ok / max(1, n)


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
    bank0 = FpBank(model0, stoi, device)
    F0 = s221.fp_matrix(bank0, core)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(4_000_000)) if len(m.group(1)) >= 5))

    # --- Match 227 fact protocol (SEED 227, 60 facts) ---
    rng227 = random.Random(227)
    subs = gen_fakes(set(wiki_words), rng227, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_can, V = s221.build_fact_bank(bank0, subs, vals, rng227)

    text_code = s225.ensure_code(random.Random(227 + 1), args.smoke)
    flat_c, off_c = s213.build_flat_from_text(text_code, tok, pad_id, max_lines=max_lines, min_line_len=20)
    # same arc seed family as 227 code shift
    model_c = s221.finetune_arc_enc(model0, flat_c, off_c, char_table, pad_id, device, arc_steps, 227 + 3)
    bank_c = FpBank(model_c, stoi, device)
    F_c = s221.fp_matrix(bank_c, core)
    cos = float((F0 * F_c).sum(-1).mean())
    W_bwd, align = s221.train_remap(DomainAdapter(256).to(device), F_c, F0, rng227, w_steps, device)

    r_qmap = recall_qmap(K_can, V, bank_c, W_bwd, subs, vals, rng227)
    ok = n = 0
    for S, gold in zip(subs, vals):
        q = bank_c.ctx_fp(f"In the report {S} was linked to the organization.", exclude=gold)
        if q is None:
            continue
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng227.shuffle(cands)
        g = cands.index(gold)
        sc = [
            float((K_can[[i for i, v in enumerate(V) if v == c]] @ q).max()) if any(v == c for v in V) else -1.0
            for c in cands
        ]
        ok += int(np.argmax(sc) == g)
        n += 1
    r_no_w = ok / max(1, n)

    head_code = s225.train_upper(model0, flat_c, off_c, char_table, pad_id, device, upper_steps, SEED + 3)
    rng = random.Random(SEED)

    items = []
    for S, gold in zip(subs, vals):
        pool = vals
        items.append((S, gold, pool))

    def pack(form: str):
        out = []
        for S, gold, pool in items:
            if form == "none":
                text = f"# TODO\ndef org_of_{S}():\n    return "
            elif form == "prose":
                text = f"# Note: {S} was director of {gold}.\ndef org_of_{S}():\n    return "
            elif form == "code_comment":
                text = f"# org[{S}] = {gold}\ndef org_of_{S}():\n    return "
            elif form == "assignment":
                text = f"ORG = {gold!r}\ndef org_of_{S}():\n    return "
            else:
                raise ValueError(form)
            out.append((text, gold, pool))
        return out

    util = {
        "none": rank4(head_code, tok, pad_id, char_table, device, pack("none"), rng),
        "prose_inject": rank4(head_code, tok, pad_id, char_table, device, pack("prose"), rng),
        "code_comment_inject": rank4(head_code, tok, pad_id, char_table, device, pack("code_comment"), rng),
        "assignment_inject": rank4(head_code, tok, pad_id, char_table, device, pack("assignment"), rng),
    }
    # also P1 head for comparison
    util_p1 = {
        "none": rank4(model0, tok, pad_id, char_table, device, pack("none"), rng),
        "assignment_inject": rank4(model0, tok, pad_id, char_table, device, pack("assignment"), rng),
    }

    g_recall_reconcile = r_qmap >= 0.85
    best_util = max(util.values())
    g_util = best_util >= util["none"] + 0.08
    overall = "RETRIEVAL_OK_UTIL_BOUNDARY"
    if g_recall_reconcile and g_util:
        overall = "JOINT_DIAG_UTIL_WIN"
    elif g_recall_reconcile:
        overall = "RETRIEVAL_OK_UTIL_BOUNDARY"

    out = {
        "stage": "226b",
        "overall": overall,
        "H1_recall_vs_227": {
            "protocol": "SEED227 facts + factual query + code shift + qmap",
            "mean_cos_code": cos,
            "align_W_bwd": align,
            "recall_qmap": r_qmap,
            "recall_no_W": r_no_w,
            "reconciles_with_227_cross_0p95": g_recall_reconcile,
            "note": "226's 0.60 used SEED226 / n=40 — not code-gen query; same factual template",
        },
        "H2_utilization_inject_forms": {
            "head_code": util,
            "head_P1_ref": util_p1,
            "best_minus_none": best_util - util["none"],
            "path_C_code_native_helps": util["code_comment_inject"] >= util["prose_inject"] + 0.03
            or util["assignment_inject"] >= util["prose_inject"] + 0.03,
        },
        "contract_note": (
            "One canonical bank + W@read (227). Utilization is head/policy layer; "
            "path A boundary unless inject form or joint train (B) helps."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 226b joint diag\n\n**{overall}** qmap={r_qmap:.3f} util none/prose/code/assign="
        f"{util['none']:.3f}/{util['prose_inject']:.3f}/{util['code_comment_inject']:.3f}/{util['assignment_inject']:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
