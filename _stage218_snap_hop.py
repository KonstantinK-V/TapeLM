"""
Stage 218 — explicit lexicon snap on latent hops (subset of 206 protocol).

  python _stage218_snap_hop.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage204_noise_robustness import noisy

RES = Path("results")
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage218_decision.json"
SEED = 218


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    n_chain = 20 if args.smoke else 60
    k = 4
    p_noise = 0.15

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    model = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        real = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(2_000_000)) if len(m.group(1)) >= 5))
    ENT_RE2 = ENT_RE
    ents = gen_fakes(set(real), rng, n_chain * 5)[: n_chain * 4]
    chains = [ents[i : i + 4] for i in range(0, len(ents) - 3, 4)][:n_chain]
    pool = list(dict.fromkeys(w for c in chains for w in c))
    pool_c = bank.fp(pool)
    nrng = random.Random(SEED + 1)
    Kc, Vc = [], []
    for c in chains:
        for i in range(len(c) - 1):
            Kc.append(bank.fp([noisy(c[i], p_noise, nrng)])[0])
            Vc.append(bank.fp([noisy(c[i + 1], p_noise, nrng)])[0])
    Kc, Vc = torch.stack(Kc, 0), torch.stack(Vc, 0)

    def run(snap: bool) -> float:
        ok = 0
        for c in chains:
            v = bank.fp([noisy(c[0], p_noise, nrng)])[0]
            for _ in range(k):
                v = Vc[int((Kc @ v).argmax())]
                if snap:
                    v = pool_c[int((pool_c @ v).argmax())]
            scores = pool_c @ v
            gold = c[min(k, len(c) - 1)]
            others = [x for x in pool if x != gold][:3]
            cands = [gold] + others
            order = list(range(4))
            rng.shuffle(order)
            cands = [cands[i] for i in order]
            gold_i = cands.index(gold)
            sc = [float(scores[pool.index(x)]) for x in cands]
            ok += int(int(np.argmax(sc)) == gold_i)
        return ok / len(chains)

    acc_raw = run(False)
    acc_snap = run(True)
    g1 = acc_snap >= acc_raw + 0.02
    overall = "SNAP_HOP_WIN" if g1 else "SNAP_HOP_INVALID_METHOD"
    DECISION.write_text(
        json.dumps(
            {
                "stage": 218,
                "overall": overall,
                "gates": {"G1_snap_noisy": g1},
                "acc_no_snap": acc_raw,
                "acc_snap": acc_snap,
                "k": k,
                "p_noise": p_noise,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"218 {overall} raw={acc_raw:.3f} snap={acc_snap:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
