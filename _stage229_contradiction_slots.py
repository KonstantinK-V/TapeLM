"""
Stage 229 — Adversarial slot injection (contradiction / multi-hit).

Write two slots with same/similar keys, conflicting values.
Query entity; measure top-2 retrieval, score gap, whether both survive.

Contract expectation: fp memory returns candidates (feature); resolution is upper-layer.

  python _stage229_contradiction_slots.py [--smoke]
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

RES = Path("results")
DECISION = RES / "stage229_decision.json"
MINI = RES / "stage229_mini.md"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 229


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

    keys, vals, tags = [], [], []
    for S, a, b in zip(subs, vals_a, vals_b):
        ctx_a = f"Official records state {S} was director of {a} in 1987 ."
        ctx_b = f"Later revision claims {S} was director of {b} in 1999 ."
        ka = bank.fp([S])[0]
        ca = bank.ctx_fp(ctx_a, exclude=a)
        cb = bank.ctx_fp(ctx_b, exclude=b)
        if ca is None or cb is None:
            continue
        keys.append(F.normalize(ka + ca, dim=-1))
        vals.append(a)
        tags.append("A")
        keys.append(F.normalize(ka + cb, dim=-1))
        vals.append(b)
        tags.append("B")
    K = torch.stack(keys, 0)

    both_in_top2 = 0
    a_wins = 0
    b_wins = 0
    gaps = []
    n_q = 0
    for S, a, b in zip(subs, vals_a, vals_b):
        q = bank.ctx_fp(f"In the report {S} was linked to the organization.", exclude=None)
        if q is None:
            continue
        sc = (K @ q).tolist()
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
        top = order[:4]
        top_vals = [vals[i] for i in top]
        hit_a = a in top_vals[:2]
        hit_b = b in top_vals[:2]
        both_in_top2 += int(hit_a and hit_b)
        # among slots for this subject only
        idxs = [i for i, v in enumerate(vals) if v in (a, b)]
        # better: indices written for this S — use pairs by position
        # find the two keys for this S by matching values a,b in order of appearance
        ia = next((i for i, v in enumerate(vals) if v == a), None)
        ib = next((i for i, v in enumerate(vals) if v == b), None)
        if ia is None or ib is None:
            continue
        sa, sb = sc[ia], sc[ib]
        gaps.append(abs(sa - sb))
        if sa >= sb:
            a_wins += 1
        else:
            b_wins += 1
        n_q += 1

    rate_both = both_in_top2 / max(1, n_q)
    mean_gap = float(sum(gaps) / max(1, len(gaps)))
    # Honest memory: both often survive OR small gap (ambiguous)
    g_multi = rate_both >= 0.40 or mean_gap < 0.08
    overall = "CONTRADICTION_RAW_MEMORY_OK" if g_multi else "CONTRADICTION_COLLAPSE"

    out = {
        "stage": 229,
        "overall": overall,
        "n_queries": n_q,
        "rate_both_values_in_top2": rate_both,
        "mean_abs_score_gap_A_vs_B": mean_gap,
        "A_wins": a_wins,
        "B_wins": b_wins,
        "interpretation": (
            "fp nearest-neighbor returns conflicting candidates; resolution is not in the slot layer"
            if g_multi
            else "one value dominates — possible key collision / ctx dominates"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 229 contradiction\n\n**{overall}** both_top2={rate_both:.3f} gap={mean_gap:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
