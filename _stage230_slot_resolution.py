"""
Stage 230 — Resolution policy on contradictory slots (229 follow-up).

229 showed multi-hit + small gaps; raw argmax always picked the first-written value.
This stage tests **upper-layer** policies (provenance / recency / query cue / composite).

  python _stage230_slot_resolution.py [--smoke]
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
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import resolve_slot_contradiction, subject_slot_hits

RES = Path("results")
DECISION = RES / "stage230_decision.json"
MINI = RES / "stage230_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 230

QUERY_NEUTRAL = "In the report {S} was linked to the organization."
QUERY_OFFICIAL = "Per the 1987 official records, {S} was linked to the organization."
QUERY_REVISION = "Per the 1999 revision, {S} was linked to the organization."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 8 if args.smoke else 30
    rng = random.Random(SEED)

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(4_000_000)) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng, n + 5)[:n]
    vals_a = wiki_words[:n]
    vals_b = wiki_words[n : 2 * n]
    if len(vals_b) < n:
        vals_b = list(reversed(wiki_words[:n]))

    keys, vals, meta, subject_indices = [], [], [], []
    for S, a, b in zip(subs, vals_a, vals_b):
        ctx_a = f"Official records state {S} was director of {a} in 1987 ."
        ctx_b = f"Later revision claims {S} was director of {b} in 1999 ."
        ka = bank.fp([S])[0]
        ca = bank.ctx_fp(ctx_a, exclude=a)
        cb = bank.ctx_fp(ctx_b, exclude=b)
        if ca is None or cb is None:
            continue
        ia = len(keys)
        keys.append(F.normalize(ka + ca, dim=-1))
        vals.append(a)
        meta.append({"provenance": "official", "year": 1987, "subject": S})
        ib = len(keys)
        keys.append(F.normalize(ka + cb, dim=-1))
        vals.append(b)
        meta.append({"provenance": "revision", "year": 1999, "subject": S})
        subject_indices.append((S, a, b, [ia, ib]))
    K = torch.stack(keys, 0)

    policies = ["argmax", "recency", "query_cue", "composite"]
    suites = [
        ("neutral", QUERY_NEUTRAL, lambda a, b: b),
        ("official_cue", QUERY_OFFICIAL, lambda a, b: a),
        ("revision_cue", QUERY_REVISION, lambda a, b: b),
    ]
    acc = {p: {name: 0 for name, _, _ in suites} for p in policies}
    counts = {name: 0 for name, _, _ in suites}
    both_top2 = 0
    n_q = 0

    for S, a, b, idxs in subject_indices:
        for suite_name, tmpl, gold_fn in suites:
            qtext = tmpl.format(S=S)
            q = bank.ctx_fp(qtext, exclude=None)
            if q is None:
                continue
            hits = subject_slot_hits(K, vals, q, idxs, meta)
            top_vals = [h.value for h in hits[:2]]
            both_top2 += int(a in top_vals and b in top_vals)
            gold = gold_fn(a, b)
            for p in policies:
                pick = resolve_slot_contradiction(hits, qtext, policy=p)
                acc[p][suite_name] += int(pick == gold)
            counts[suite_name] += 1
            n_q += 1

    rates = {
        p: {name: acc[p][name] / max(1, counts[name]) for name in counts}
        for p in policies
    }
    macro = {p: sum(rates[p].values()) / max(1, len(rates[p])) for p in policies}
    rate_both = both_top2 / max(1, n_q)

    g_cue = rates["query_cue"]["official_cue"] >= 0.85 and rates["query_cue"]["revision_cue"] >= 0.85
    g_comp = macro["composite"] >= macro["argmax"] + 0.10
    g_neutral = rates["composite"]["neutral"] >= 0.70
    overall = (
        "RESOLUTION_POLICY_OK"
        if g_cue and g_comp and g_neutral
        else ("RESOLUTION_POLICY_PARTIAL" if g_cue or g_comp else "RESOLUTION_POLICY_NO")
    )

    out = {
        "stage": 230,
        "overall": overall,
        "gates": {
            "G_query_cue_cued_ge_0p85": g_cue,
            "G_composite_beats_argmax_macro": g_comp,
            "G_composite_neutral_ge_0p70": g_neutral,
        },
        "rate_both_values_in_top2": rate_both,
        "accuracy_by_policy_suite": rates,
        "macro_accuracy": macro,
        "argmax_bias_note": "229: argmax always preferred first-written official slot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 230 resolution\n\n**{overall}** macro composite={macro['composite']:.3f} "
        f"argmax={macro['argmax']:.3f} cue={rates['query_cue']['revision_cue']:.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
