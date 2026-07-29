"""191b — fairness probe: curve meaning measured like GPT's (mean-pooled states).

P4 compared GPT mean-pooled hidden vs curve slow ENDPOINT. Re-measure curve P1
with mean-pooled FAST channel (and fast+slow concat) — same pooling as GPT.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data

CKPT = Path("checkpoints/stage191_p1_curve.pt")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()

    @torch.no_grad()
    def states(ids_list):
        x = torch.tensor([ids_list[-MAX_ARCS:]], dtype=torch.long, device=device)
        pad = x == pad_id
        arcs = model._arcs(char_table[x], x)
        fast = model.fast(arcs, pad_mask=pad)
        slow, _, _ = model.slow(arcs, pad)
        v = (~pad)[0]
        return fast[0][v], slow[0][v]

    def z_fast(ids_list):
        f, _ = states(ids_list)
        return f.mean(0)

    def z_cat(ids_list):
        f, s = states(ids_list)
        return torch.cat([f.mean(0), s.mean(0)])

    def gate_B(z_of):
        def zt(t):
            return z_of([i for i in tok.encode(t).ids if i != pad_id])

        cos = lambda a, b: float(F.cosine_similarity(a, b, dim=-1))
        para = [cos(zt(a), zt(b)) for a, b in s179.PARAPHRASE_PAIRS]
        hard = [cos(zt(a), zt(b)) for a, b in s179.HARD_PAIRS]
        return {"para": float(np.mean(para)), "hard": float(np.mean(hard)), "gap": float(np.mean(hard) - np.mean(para))}

    def doclink(z_of, n=80):
        rng = random.Random(7)
        nd = len(off) - 1
        ok = 0
        for _ in range(n):
            d1, d2 = rng.randint(0, nd - 1), rng.randint(0, nd - 1)
            s1, e1 = off[d1], off[d1 + 1]
            s2, e2 = off[d2], off[d2 + 1]
            if e1 - s1 < MAX_ARCS + 16 or e2 - s2 < MAX_ARCS:
                continue
            half = (s1 + e1) // 2
            a = flat[s1 : min(s1 + MAX_ARCS, half)].tolist()
            b = flat[half : half + MAX_ARCS].tolist()
            c = flat[s2 : s2 + MAX_ARCS].tolist()
            za, zb, zc = z_of(a), z_of(b), z_of(c)
            ok += int(float(F.cosine_similarity(za, zb, dim=-1)) > float(F.cosine_similarity(za, zc, dim=-1)))
        return ok / max(1, n)

    for name, z in (("fast_meanpool", z_fast), ("fast+slow_meanpool", z_cat)):
        r = {"gateB": gate_B(z), "doclink": doclink(z)}
        print(name, json.dumps(r))
    Path("results/stage191_p4b_fair.json").write_text(
        json.dumps({n: {"gateB": gate_B(z), "doclink": doclink(z)} for n, z in (("fast_meanpool", z_fast), ("fast+slow_meanpool", z_cat))}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
