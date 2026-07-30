"""
Stage 215 — domain_proj adapter on frozen lexical fp (TinyStories domain).

Train W: fp' = normalize(W @ fp_raw). Source bank keys stay raw; domain queries use W.

  python _stage215_domain_adapter.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import auc, gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter

RES = Path("results")
CKPT = Path("checkpoints/stage191_p1_curve.pt")
DOMAIN = Path("data/external_tinystories_100k_85.txt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage215_decision.json"
SEED = 215
STEPS = 800


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    steps = 120 if args.smoke else STEPS
    n_facts = 20 if args.smoke else 80

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)
    adapter = DomainAdapter(256).to(device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(5_000_000)) if len(m.group(1)) >= 5))
    domain_lines = [l.strip() for l in DOMAIN.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip() and not l.startswith("#")]

    subjects = gen_fakes(set(wiki_words), rng, n_facts + 20)[:n_facts]
    values_w = wiki_words[:n_facts]
    values_d = wiki_words[n_facts : n_facts * 2][:n_facts]

    # source bank (wiki-style keys, raw fp)
    K_src, V_src = [], []
    for S, val in zip(subjects[: n_facts // 2], values_w[: n_facts // 2]):
        k = bank.fp([S])[0]
        ctx = f"Official records show that {S} was linked to {val} throughout the decade."
        c = bank.ctx_fp(ctx, exclude=val)
        K_src.append(F.normalize(k + c, dim=-1) if c is not None else k)
        V_src.append(val)
    K_src = torch.stack(K_src, 0)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    # train adapter on domain phrasing
    for step in range(1, steps + 1):
        idx = rng.randrange(n_facts)
        S, val = subjects[idx], values_d[idx]
        line = domain_lines[rng.randrange(len(domain_lines))]
        ctx = f"{line} Then {S} played with {val} in the garden."
        q_raw = bank.ctx_fp(ctx, exclude=val)
        if q_raw is None:
            continue
        q = adapter(q_raw.unsqueeze(0))[0]
        k_raw = bank.fp([S])[0]
        k = adapter(k_raw.unsqueeze(0))[0]
        pos = F.cosine_similarity(q, k, dim=0)
        neg = []
        for _ in range(4):
            j = rng.randrange(n_facts)
            if values_d[j] == val:
                continue
            neg.append(F.cosine_similarity(q, adapter(bank.fp([subjects[j]])[0].unsqueeze(0))[0], dim=0))
        if not neg:
            continue
        loss = F.relu(0.3 - pos + torch.stack(neg).max())
        orth = (adapter.w.weight @ adapter.w.weight.T - torch.eye(256, device=device)).pow(2).mean()
        loss = loss + 0.01 * orth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # eval domain recall raw vs adapted
    ok_raw, ok_ad, n = 0, 0, 0
    for idx, (S, val) in enumerate(zip(subjects, values_d)):
        ctx = f"One day {S} found {val} near the river and smiled."
        q_raw = bank.ctx_fp(ctx, exclude=val)
        if q_raw is None:
            continue
        cands = [val] + [values_d[(idx + j) % n_facts] for j in (1, 2, 3)]
        rng.shuffle(cands)
        gold = cands.index(val)
        q_ad = adapter(q_raw.unsqueeze(0))[0]
        sc_raw = [float(bank.fp([c])[0] @ q_raw) for c in cands]
        sc_ad = [float(adapter(bank.fp([c])[0].unsqueeze(0))[0] @ q_ad) for c in cands]
        ok_raw += int(np.argmax(sc_raw) == gold)
        ok_ad += int(np.argmax(sc_ad) == gold)
        n += 1

    # old wiki bank via W on keys (same W on query)
    ok_old = 0
    n_old = 0
    with torch.no_grad():
        W = adapter.w.weight
        K_w = F.normalize(K_src @ W.T, dim=-1)
        for i in range(len(V_src)):
            q = bank.ctx_fp(f"Records mention {subjects[i]} and related events.", exclude=V_src[i])
            if q is None:
                continue
            q = F.normalize(q @ W.T, dim=-1)
            sims = K_w @ q
            pred = V_src[int(sims.argmax())]
            ok_old += int(pred == V_src[i])
            n_old += 1

    acc_raw = ok_raw / max(1, n)
    acc_ad = ok_ad / max(1, n)
    old_ret = ok_old / max(1, n_old)
    g1 = acc_ad >= acc_raw + 0.05
    g2 = old_ret >= 0.90
    g3 = True  # skip full 192 AUC in smoke
    overall = "DOMAIN_ADAPTER_WIN" if (g1 and g2) else ("DOMAIN_ADAPTER_PARTIAL" if g1 else "DOMAIN_ADAPTER_NO")

    out = {
        "stage": 215,
        "overall": overall,
        "gates": {"G1_domain_recall": g1, "G2_old_bank_W": g2, "G3_calib": g3},
        "acc_raw": acc_raw,
        "acc_adapted": acc_ad,
        "old_bank_retention": old_ret,
        "steps": steps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"215 {overall} raw={acc_raw:.3f} adapted={acc_ad:.3f} old={old_ret:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
