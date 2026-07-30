"""
Stage 214 — Recency-weighted ctx_fp (zero-train extension of 194).

Sweep lambda on entity recall vs mean-pool baseline (lambda=0).

  python _stage214_recency_ctx.py
  python _stage214_recency_ctx.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import build_memory, score_entity_items
from _tapelm_ext import RecencyFpBank

RES = Path("results")
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
EXAM_V3 = Path("data/stage191_exam_v3.jsonl")
DECISION = RES / "stage214_decision.json"
MINI = RES / "stage214_mini.md"
LOG = RES / "_stage214_log.txt"
CORPUS_CHARS = 150_000_000
EXAM_TAIL_CHARS = 3_000_000
LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.35]


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage214 start {datetime.now(timezone.utc).isoformat()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS if not args.smoke else 2_000_000)
    tail = text[-EXAM_TAIL_CHARS:]
    tail_paras = [p.strip() for p in tail.split("\n") if 120 < len(p.strip()) < 1000][:200 if args.smoke else 1200]

    items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
    lambdas = [0.0, 0.1] if args.smoke else LAMBDAS
    sweep = {}
    for lam in lambdas:
        bank = RecencyFpBank(model, stoi, device, lam=lam)
        K, vals = build_memory(tail_paras, bank, f"lam={lam}")
        res = score_entity_items(items, tok, pad_id, bank, K, vals)
        sweep[str(lam)] = res
        log(f"  lam={lam}: acc={res['acc']:.3f} n={res['n']}")

    base = sweep["0.0"]["acc"]
    best_lam = max(lambdas, key=lambda l: sweep[str(l)]["acc"])
    best = sweep[str(best_lam)]["acc"]
    delta = best - base
    g1 = delta >= 0.02 and best_lam > 0
    g2 = best >= 0.50
    overall = "RECENCY_CTX_WIN" if (g1 and g2) else ("RECENCY_CTX_MARGINAL" if delta > 0 else "RECENCY_CTX_NO")

    out = {
        "stage": 214,
        "overall": overall,
        "gates": {"G1_delta_vs_mean": g1, "G2_acc": g2},
        "baseline_acc": base,
        "best_lambda": best_lam,
        "best_acc": best,
        "delta": delta,
        "sweep": sweep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(f"# Stage214\n\n**{overall}** best_lam={best_lam} acc={best:.3f} (base {base:.3f})\n", encoding="utf-8")
    log(f"VERDICT {overall} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
