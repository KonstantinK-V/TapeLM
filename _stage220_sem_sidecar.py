"""
Stage 220 — semantic sidecar vs lexical fp on PAWS (frozen P1).

  python _stage220_sem_sidecar.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage202_semantic_paws import SemHead, lexical_overlap

RES = Path("results")
CKPT = Path("checkpoints/stage191_p1_curve.pt")
DECISION = RES / "stage220_decision.json"
SEED = 220


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    n_train = 400 if args.smoke else 2000
    n_test = 200 if args.smoke else 800

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    head = SemHead(256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-4)

    ds = load_dataset("paws", "labeled_final", split="train[:8000]" if args.smoke else "train[:25000]")
    pairs = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds]
    rng = np.random.RandomState(SEED)
    rng.shuffle(pairs)
    train, test = pairs[:n_train], pairs[n_train : n_train + n_test]

    def encode_sent(text):
        ids = [i for i in tok.encode(text).ids if i != pad_id][-MAX_ARCS:]
        x = torch.tensor([ids], device=device)
        pad = x == pad_id
        arcs = model._arcs(char_table[x], ids=x)
        fast = model.fast(arcs, pad_mask=pad)
        return fast, ~pad

    for epoch in range(2 if args.smoke else 3):
        rng.shuffle(train)
        for s1, s2, lab in train[:200] if args.smoke else train:
            st1, m1 = encode_sent(s1)
            st2, m2 = encode_sent(s2)
            z1, z2 = head(st1, m1), head(st2, m2)
            y = torch.tensor([lab], device=device, dtype=torch.float32)
            cos = F.cosine_similarity(z1, z2)
            loss = F.binary_cross_entropy((cos + 1) / 2, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def acc_pairs(subset, use_sem=True):
        ok, n = 0, 0
        for s1, s2, lab in subset:
            if use_sem:
                st1, m1 = encode_sent(s1)
                st2, m2 = encode_sent(s2)
                z1, z2 = head(st1, m1), head(st2, m2)
                pred = int(F.cosine_similarity(z1, z2).item() > 0.5)
            else:
                pred = int(lexical_overlap(s1, s2) > 0.5)
            ok += int(pred == lab)
            n += 1
        return ok / max(1, n)

    acc_sem = acc_pairs(test, True)
    acc_lex = acc_pairs(test, False)
    g1 = acc_sem > acc_lex + 0.03
    overall = "SEM_SIDECAR_WIN" if g1 else "SEM_SIDECAR_INVALID_METHOD"
    DECISION.write_text(
        json.dumps(
            {
                "stage": 220,
                "overall": overall,
                "gates": {"G1_sem_vs_lexical_baseline": g1},
                "paws_acc_sem": acc_sem,
                "paws_acc_lexical_proxy": acc_lex,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"220 {overall} sem={acc_sem:.3f} lex={acc_lex:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
