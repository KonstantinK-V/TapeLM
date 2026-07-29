"""
Stage 188 — S3b: wire surprise into the output (fix 187's G3).

187 showed: internal surprise EXISTS (G2 pass) but is not visible in the output
distribution — the model gets MORE confident after fake entities (G3 fail),
while GPT gets less confident (rarity signal for free from BPE).

Fix: surprise-conditioned temperature. Per position t:
  T_t = 1 + softplus(w * surprise_t + b)     (learnable w,b)
  logits_t = head([fast_t ; slow_t]) / T_t
CE itself calibrates w,b: when surprised, softening predictions lowers loss on
hard positions. No new hand loss.

Extra diagnostic: mean surprise AT fake span vs AT real span — does the ink
channel even see fakes as unusual? If not, no head wiring can fix G3 and the
next target is a rarity signal in the ink encoder itself.

Gates (judge = Exam v2):
  G1 next_tok >= 0.727 - 0.03 (don't lose 187's CE parity)
  G2 surprise unseen > seen (keep novelty)
  G3 entropy after fake > after real (the fix target)

  python _stage188_surprise_head.py
  python _stage188_surprise_head.py --steps 3000
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

import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage181_ce_control as s181
import _stage185_tape_read as s185
import _stage187_self_model as s187

RES = Path("results")
DATA = Path("data")
CKPT_DIR = Path("checkpoints")
LOG = RES / "_stage188_log.txt"
DECISION = RES / "stage188_decision.json"
MINI = RES / "stage188_mini.md"
EXAM = DATA / "stage186_exam_v2.jsonl"
DEC187 = RES / "stage187_decision.json"
CKPT_OUT = CKPT_DIR / "stage188_surprise_head.pt"
TOK_PATH = s177.TOK_PATH

SEED = 185  # matched stream
MAX_ARCS = s177.MAX_ARCS
MICRO = 16
LR = 3e-4
EVAL_EVERY = 1000
DEFAULT_STEPS = 3000
W_SELF = 0.1
N_MID_EVAL = 60
PAD = "[PAD]"


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class SurpriseHeadModel(s187.SelfModel):
    def __init__(self, n_char: int, V: int):
        super().__init__(n_char, V)
        self.temp_w = nn.Parameter(torch.tensor(4.0))
        self.temp_b = nn.Parameter(torch.tensor(-2.0))

    def forward_all(self, char_ids: torch.Tensor, pad: torch.Tensor):
        arcs = self.arc_enc(char_ids)
        fast = self.fast(arcs, pad_mask=pad)
        slow, surprise, pred_loss = self.slow(arcs, pad)
        logits = self.head(torch.cat([fast, slow], dim=-1))
        T = 1.0 + F.softplus(self.temp_w * surprise + self.temp_b).unsqueeze(-1)
        return logits / T, surprise, pred_loss


@torch.no_grad()
def surprise_at_span(model, char_table, pad_id, ctx_ids, span_ids, device) -> float:
    seq = (ctx_ids + span_ids)[-MAX_ARCS:]
    n_ctx = len(seq) - len(span_ids)
    x = torch.tensor([seq], dtype=torch.long, device=device)
    pad = x == pad_id
    _, surprise, _ = model.forward_all(char_table[x], pad)
    return float(surprise[0, n_ctx:].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage188 start {datetime.now(timezone.utc).isoformat()}")
    log("S3b: surprise-conditioned temperature on the head")

    items = [json.loads(l) for l in EXAM.read_text(encoding="utf-8").splitlines() if l.strip()]
    items_mid = [it for it in items if it["type"] == "next_tok"][:N_MID_EVAL]
    base_next = json.loads(DEC187.read_text(encoding="utf-8"))["gates"]["G1_ce_preserved"]["next_tok"]
    log(f"exam v2 items={len(items)}; baseline (187) next_tok={base_next:.3f}")

    device = torch.device(args.device)
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0

    text = s170.load_corpus(max_chars=20_000_000)
    chars = sorted(set(text) | {" "})
    itos = ["<pad>"] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = s181.build_id_docs(tok, text)
    train_docs = docs[: int(0.8 * len(docs))] or docs
    hold_docs = docs[int(0.8 * len(docs)) :] or docs[-100:]
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    log(f"docs={len(docs)} V={V} n_char={len(itos)}")

    torch.manual_seed(SEED)
    model = SurpriseHeadModel(len(itos), V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    running, run_self = None, None
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        ids = s185.sample_id_batch(train_docs, MICRO, rng, device, pad_id)
        pad = ids == pad_id
        logits, surprise, pred_loss = model.forward_all(char_table[ids], pad)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        lp = pred_loss[~pad].mean()
        loss = ce + W_SELF * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(ce) if running is None else 0.95 * running + 0.05 * float(ce)
        run_self = float(lp) if run_self is None else 0.95 * run_self + 0.05 * float(lp)
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            mid = s185.score_exam(model, char_table, pad_id, items_mid, device, only_type="next_tok")
            log(
                f"  step {step}: ce~{running:.3f} self~{run_self:.3f} tw={float(model.temp_w):.2f} "
                f"tb={float(model.temp_b):.2f} next_tok(mid)={mid.get('next_tok_acc', 0):.3f} ({time.time()-t0:.0f}s)"
            )
            model.train()
            torch.save({"model": model.state_dict(), "step": step}, CKPT_OUT)

    model.eval()
    full = s185.score_exam(model, char_table, pad_id, items, device)
    next_tok = full.get("next_tok_acc", 0.0)

    s_train = s187.mean_surprise(model, train_docs, char_table, pad_id, device, random.Random(1))
    s_hold = s187.mean_surprise(model, hold_docs, char_table, pad_id, device, random.Random(2))

    ent_items = [it for it in items if it["type"] == "entity"][:80]
    rngf = random.Random(3)
    e_real, e_fake, sp_real, sp_fake = [], [], [], []
    for it in ent_items:
        gold_ids = it["cand_ids"][it["gold_idx"]]
        fake = s187.FAKES[rngf.randint(0, len(s187.FAKES) - 1)]
        fake_ids = [i for i in tok.encode(" " + fake).ids if i != pad_id]
        e_real.append(s187.entropy_after(model, char_table, pad_id, it["ctx_ids"], gold_ids, device))
        e_fake.append(s187.entropy_after(model, char_table, pad_id, it["ctx_ids"], fake_ids, device))
        sp_real.append(surprise_at_span(model, char_table, pad_id, it["ctx_ids"], gold_ids, device))
        sp_fake.append(surprise_at_span(model, char_table, pad_id, it["ctx_ids"], fake_ids, device))
    ent_real, ent_fake = float(np.mean(e_real)), float(np.mean(e_fake))
    s_real, s_fake = float(np.mean(sp_real)), float(np.mean(sp_fake))

    g1 = next_tok >= base_next - 0.03
    g2 = s_hold > s_train
    g3 = ent_fake > ent_real
    overall = "SURPRISE_HEAD_YES" if (g1 and g2 and g3) else "SURPRISE_HEAD_PARTIAL_" + "".join(
        n for n, ok in (("1", g1), ("2", g2), ("3", g3)) if not ok
    )

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "surprise_head_188",
        "overall": overall,
        "gates": {
            "G1_ce_preserved": {"next_tok": next_tok, "baseline_187": base_next, "ok": g1},
            "G2_novelty": {"surprise_seen_train": s_train, "surprise_unseen_hold": s_hold, "ok": g2},
            "G3_calibration": {"entropy_after_real": ent_real, "entropy_after_fake": ent_fake, "ok": g3},
        },
        "diagnostic_surprise_at_span": {"real": s_real, "fake": s_fake, "fake_gt_real": s_fake > s_real},
        "temp": {"w": float(model.temp_w), "b": float(model.temp_b)},
        "exam_full": full,
        "steps": args.steps,
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage188 — surprise-conditioned head",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- G1: next_tok={next_tok:.3f} vs 187 {base_next:.3f} → {g1}",
                f"- G2: surprise seen={s_train:.4f} unseen={s_hold:.4f} → {g2}",
                f"- G3: entropy real={ent_real:.3f} fake={ent_fake:.3f} → {g3}",
                f"- diag: surprise@span real={s_real:.4f} fake={s_fake:.4f} (fake>real={s_fake > s_real})",
                f"- temp w={float(model.temp_w):.2f} b={float(model.temp_b):.2f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(
        f"[188] {overall} | G1 {next_tok:.3f} | G2 {s_train:.4f}<{s_hold:.4f} | G3 {ent_real:.3f}<{ent_fake:.3f} | "
        f"diag span s_real={s_real:.4f} s_fake={s_fake:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
