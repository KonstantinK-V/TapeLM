"""
Stage 222 — Which fp pipeline actually needs W after arc_enc shift?

Compares recall modes on the same fact bank after one Stories shift:
  old/old, old keys + new query, W on keys only, W on query only, W both (221), oracle reindex.

  python _stage222_fp_deploy_modes.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

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
DECISION = RES / "stage222_decision.json"
MINI = RES / "stage222_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
DOMAIN = Path("data/external_tinystories_100k_85.txt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 222


def recall_new_queries(K, V, bank_q, bank_k_unused, subs, vals, rng, key_x=None, query_x=None):
    """bank_q for ctx_fp; keys optionally transformed; queries optionally transformed."""
    ok, n = 0, 0
    for S, gold in zip(subs, vals):
        q = bank_q.ctx_fp(f"In the report {S} was linked to the organization.", exclude=gold)
        if q is None:
            continue
        qq = query_x(q.unsqueeze(0))[0] if query_x else q
        Kuse = key_x(K) if key_x else K
        cands = [gold] + [vals[(i + 1) % len(vals)] for i in range(3)]
        rng.shuffle(cands)
        g = cands.index(gold)
        sc = []
        for c in cands:
            idxs = [i for i, v in enumerate(V) if v == c]
            sc.append(float((Kuse[idxs] @ qq).max()) if idxs else -1.0)
        ok += int(__import__("numpy").argmax(sc) == g)
        n += 1
    return ok / max(1, n), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arc_steps = 80 if args.smoke else s221.ARC_STEPS
    w_steps = 100 if args.smoke else s221.W_STEPS
    core_n = 80 if args.smoke else s221.CORE_N
    n_facts = 12 if args.smoke else 60
    rng = random.Random(SEED)

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(2_000_000)
    core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:core_n]

    model_old = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model_old.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model_old.eval()
    bank_old = FpBank(model_old, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(4_000_000)) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K_old, V = s221.build_fact_bank(bank_old, subs, vals, rng)

    text_b = DOMAIN.read_text(encoding="utf-8", errors="ignore")
    flat_b, off_b = s213.build_flat_from_text(text_b, tok, pad_id, max_lines=500 if args.smoke else 8000)
    model_new = s221.finetune_arc_enc(model_old, flat_b, off_b, char_table, pad_id, device, arc_steps, SEED + 1)
    bank_new = FpBank(model_new, stoi, device)
    K_oracle, _ = s221.build_fact_bank(bank_new, subs, vals, rng)

    F_old = s221.fp_matrix(bank_old, core)
    F_new = s221.fp_matrix(bank_new, core)
    Wmap, align_w = s221.train_remap(DomainAdapter(256).to(device), F_old, F_new, rng, w_steps, device)

    def w_raw(x):
        return F.normalize(Wmap.map_raw(x), dim=-1)

    modes = {}
    modes["M1_old_keys_old_query"], _ = recall_new_queries(K_old, V, bank_old, bank_old, subs, vals, rng)
    modes["M2_old_keys_new_query_no_W"], _ = recall_new_queries(K_old, V, bank_new, bank_old, subs, vals, rng)
    modes["M3_W_keys_old_query"], _ = recall_new_queries(
        K_old, V, bank_old, bank_old, subs, vals, rng, key_x=w_raw, query_x=None
    )
    modes["M4_old_keys_W_old_query"], _ = recall_new_queries(
        K_old, V, bank_old, bank_old, subs, vals, rng, key_x=None, query_x=w_raw
    )
    modes["M5_W_keys_W_old_query_221"], _ = recall_new_queries(
        K_old, V, bank_old, bank_old, subs, vals, rng, key_x=w_raw, query_x=w_raw
    )
    modes["M6_W_keys_new_query"], _ = recall_new_queries(
        K_old, V, bank_new, bank_old, subs, vals, rng, key_x=w_raw, query_x=None
    )
    modes["M7_oracle_new_keys_new_query"], _ = recall_new_queries(K_oracle, V, bank_new, bank_new, subs, vals, rng)

    best = max(modes, key=modes.get)
    overall = "FP_DEPLOY_MODES_OK" if modes["M5_W_keys_W_old_query_221"] >= 0.75 else "FP_DEPLOY_MODES_MIXED"

    out = {
        "stage": 222,
        "overall": overall,
        "modes": modes,
        "best_mode": best,
        "align_W_core": align_w,
        "mean_cos_word_shift": float((F_old * F_new).sum(-1).mean()),
        "interpretation": (
            "If M2 ~ M7 and >> M5, W is for legacy old-fp extraction; "
            "if M6 ~ M7, deploy new encoder on queries + W on keys only."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# Stage 222 deploy modes\n"] + [f"- **{k}**: {v:.3f}" for k, v in modes.items()]
    lines.append(f"\n**{overall}** best={best}\n")
    MINI.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
