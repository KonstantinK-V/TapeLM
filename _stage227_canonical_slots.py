"""
Stage 227 — Canonical slot storage + domain-conditioned read.

Write always with frozen arc_enc (canonical keys).
Read under domain shift X with either:
  P_keylift:  score(W_fwd @ K_can, q_domain)     # W: old→new (221)
  P_qmap:     score(K_can, W_bwd @ q_domain)     # W: new→old

Gate: cross-family (code query on canonical prose-era slots) drop vs same-family
      < 0.10 on best policy → CANONICAL_STORAGE_OK (one bank, disposable W).

  python _stage227_canonical_slots.py [--smoke]
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter

RES = Path("results")
DECISION = RES / "stage227_decision.json"
MINI = RES / "stage227_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
STORIES = Path("data/external_tinystories_100k_85.txt")
WIKI = Path("data/_wikitext103_train.txt")
CODE = Path("data/_stage224_code_corpus.txt")
SEED = 227


def log(m: str) -> None:
    print(m, flush=True)


def ensure_code(rng: random.Random, smoke: bool) -> str:
    if CODE.exists() and CODE.stat().st_size > 10_000:
        return CODE.read_text(encoding="utf-8")
    import _stage224_far_shift as s224

    return s224.ensure_code_corpus(rng, n_lines=2000 if smoke else 12000)


def recall(K, V, bank_q, subs, vals, rng, key_x=None, query_x=None):
    ok, n = 0, 0
    for S, gold in zip(subs, vals):
        q = bank_q.ctx_fp(f"In the report {S} was linked to the organization.", exclude=gold)
        if q is None:
            continue
        qq = query_x(q.unsqueeze(0))[0] if query_x else q
        Kq = key_x(K) if key_x else K
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        g = cands.index(gold)
        sc = []
        for c in cands:
            idxs = [i for i, v in enumerate(V) if v == c]
            sc.append(float((Kq[idxs] @ qq).max()) if idxs else -1.0)
        ok += int(np.argmax(sc) == g)
        n += 1
    return ok / max(1, n)


def w_apply(W: DomainAdapter):
    return lambda X: F.normalize(W.map_raw(X), dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
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

    model_can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model_can.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model_can.eval()
    bank_can = FpBank(model_can, stoi, device)
    F_can = s221.fp_matrix(bank_can, core)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(4_000_000)) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_can, V = s221.build_fact_bank(bank_can, subs, vals, rng)

    # shifts
    flat_s, off_s = s213.build_flat_from_text(
        STORIES.read_text(encoding="utf-8", errors="ignore"), tok, pad_id, max_lines=max_lines
    )
    flat_c, off_c = s213.build_flat_from_text(
        ensure_code(random.Random(SEED + 1), args.smoke), tok, pad_id, max_lines=max_lines, min_line_len=20
    )
    log("arc shift prose(stories)…")
    model_s = s221.finetune_arc_enc(model_can, flat_s, off_s, char_table, pad_id, device, arc_steps, SEED + 2)
    bank_s = FpBank(model_s, stoi, device)
    F_s = s221.fp_matrix(bank_s, core)
    log("arc shift code…")
    model_c = s221.finetune_arc_enc(model_can, flat_c, off_c, char_table, pad_id, device, arc_steps, SEED + 3)
    bank_c = FpBank(model_c, stoi, device)
    F_c = s221.fp_matrix(bank_c, core)

    cos_s = float((F_can * F_s).sum(-1).mean())
    cos_c = float((F_can * F_c).sum(-1).mean())

    W_s_fwd, al_sf = s221.train_remap(DomainAdapter(256).to(device), F_can, F_s, rng, w_steps, device)
    W_c_fwd, al_cf = s221.train_remap(DomainAdapter(256).to(device), F_can, F_c, rng, w_steps, device)
    W_s_bwd, al_sb = s221.train_remap(DomainAdapter(256).to(device), F_s, F_can, rng, w_steps, device)
    W_c_bwd, al_cb = s221.train_remap(DomainAdapter(256).to(device), F_c, F_can, rng, w_steps, device)

    # baseline canonical
    r_can = recall(K_can, V, bank_can, subs, vals, rng)

    modes = {}
    # same-family prose: stories query
    modes["same_prose_no_W"] = recall(K_can, V, bank_s, subs, vals, rng)
    modes["same_prose_keylift"] = recall(K_can, V, bank_s, subs, vals, rng, key_x=w_apply(W_s_fwd))
    modes["same_prose_qmap"] = recall(K_can, V, bank_s, subs, vals, rng, query_x=w_apply(W_s_bwd))
    # cross-family: code query on canonical (prose-era) slots
    modes["cross_code_no_W"] = recall(K_can, V, bank_c, subs, vals, rng)
    modes["cross_code_keylift"] = recall(K_can, V, bank_c, subs, vals, rng, key_x=w_apply(W_c_fwd))
    modes["cross_code_qmap"] = recall(K_can, V, bank_c, subs, vals, rng, query_x=w_apply(W_c_bwd))

    best_same = max(modes["same_prose_keylift"], modes["same_prose_qmap"])
    best_cross = max(modes["cross_code_keylift"], modes["cross_code_qmap"])
    best_same_name = "keylift" if modes["same_prose_keylift"] >= modes["same_prose_qmap"] else "qmap"
    best_cross_name = "keylift" if modes["cross_code_keylift"] >= modes["cross_code_qmap"] else "qmap"
    drop = best_same - best_cross
    # also compare cross best vs canonical baseline
    drop_vs_can = r_can - best_cross

    g_unify = drop < 0.10 and best_cross >= 0.70
    overall = "CANONICAL_STORAGE_OK" if g_unify else ("CANONICAL_STORAGE_PARTIAL" if best_cross > modes["cross_code_no_W"] + 0.05 else "CANONICAL_STORAGE_NO")

    out = {
        "stage": 227,
        "overall": overall,
        "gates": {"G_cross_drop_lt_0p10": drop < 0.10, "G_cross_recall_ge_0p70": best_cross >= 0.70},
        "recall_canonical_baseline": r_can,
        "mean_cos_shift": {"stories": cos_s, "code": cos_c},
        "align": {"W_s_fwd": al_sf, "W_c_fwd": al_cf, "W_s_bwd": al_sb, "W_c_bwd": al_cb},
        "modes": modes,
        "best_same_policy": best_same_name,
        "best_cross_policy": best_cross_name,
        "best_same_recall": best_same,
        "best_cross_recall": best_cross,
        "cross_drop_same_minus_cross": drop,
        "drop_vs_canonical_baseline": drop_vs_can,
        "note": "Keys always canonical (frozen P1). W disposable at read: keylift=old→domain, qmap=domain→old.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 227 canonical slots\n\n**{overall}** same={best_same:.3f} cross={best_cross:.3f} drop={drop:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
