"""
Stage 190 — S4: content invariant on slow (attack MEANING).

Base = 187 self-model (best curve config). Add ONE representation objective:
doc-level InfoNCE on the slow endpoint — two windows of the SAME doc should map
close, windows of other docs far. This is the only natural "same content, different
surface" supervision the corpus gives for free (no handcrafted paraphrases).

Per the 185 rule, a representation loss is allowed ONLY if it survives the A/B:
  G1 next_tok(v2) >= 0.727 - 0.03      (CE not poisoned — else revert)
  G2 doc-link: same-doc vs cross-doc pairing acc > 187 baseline (invariant learned)
  G3 gate B (179 pairs): (hard - para) gap shrinks vs 187 baseline;
     strong win = para > hard (meaning beats form — never achieved before)

  python _stage190_content_invariant.py
  python _stage190_content_invariant.py --steps 3000
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
import _stage179_curve_harden_B as s179
import _stage181_ce_control as s181
import _stage185_tape_read as s185
import _stage187_self_model as s187

RES = Path("results")
DATA = Path("data")
CKPT_DIR = Path("checkpoints")
LOG = RES / "_stage190_log.txt"
DECISION = RES / "stage190_decision.json"
MINI = RES / "stage190_mini.md"
EXAM = DATA / "stage186_exam_v2.jsonl"
CKPT_OUT = CKPT_DIR / "stage190_content_invariant.pt"
CKPT_187 = CKPT_DIR / "stage187_self_model.pt"
TOK_PATH = s177.TOK_PATH

SEED = 185
MAX_ARCS = s177.MAX_ARCS
MICRO = 16
P_CON = 8          # docs per contrast batch (2 windows each)
TAU = 0.2
W_CON = 0.2
W_SELF = 0.1
LR = 3e-4
EVAL_EVERY = 1000
DEFAULT_STEPS = 3000
N_MID_EVAL = 60
PAD = "[PAD]"
BASE_NEXT = 0.727  # 187 G1 value


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def slow_endpoint(model: s187.SelfModel, ids: torch.Tensor, char_table, pad_id) -> torch.Tensor:
    pad = ids == pad_id
    arcs = model.arc_enc(char_table[ids])
    slow, _, _ = model.slow(arcs, pad)
    lengths = (~pad).sum(dim=1).clamp(min=1)
    return slow[torch.arange(ids.size(0), device=ids.device), lengths - 1]


def sample_doc_windows(docs, rng, pad_id, device, n_docs=P_CON):
    """Two disjoint windows per doc: one from first half, one from second half."""
    a_list, b_list = [], []
    tries = 0
    while len(a_list) < n_docs and tries < 200:
        tries += 1
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < MAX_ARCS + 16:
            continue
        half = len(doc) // 2
        wa_start = rng.randint(0, max(0, half - MAX_ARCS // 2))
        wb_start = rng.randint(half, max(half, len(doc) - MAX_ARCS // 2))

        def pack(s):
            w = doc[s : s + MAX_ARCS]
            if len(w) < MAX_ARCS:
                w = w + [pad_id] * (MAX_ARCS - len(w))
            return w

        a_list.append(pack(wa_start))
        b_list.append(pack(wb_start))
    if not a_list:
        return None
    return (
        torch.tensor(a_list, dtype=torch.long, device=device),
        torch.tensor(b_list, dtype=torch.long, device=device),
    )


def infonce(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    sim = za @ zb.t() / TAU  # [P,P]
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


@torch.no_grad()
def doclink_acc(model, docs, char_table, pad_id, device, rng, n=100) -> float:
    ok = 0
    done = 0
    while done < n:
        pair = sample_doc_windows(docs, rng, pad_id, device, n_docs=2)
        if pair is None:
            break
        a, b = pair  # 2 docs, windows a[i] ~ b[i]
        za = slow_endpoint(model, a, char_table, pad_id)
        zb = slow_endpoint(model, b, char_table, pad_id)
        same = F.cosine_similarity(za[0], zb[0], dim=-1)
        cross = F.cosine_similarity(za[0], zb[1], dim=-1)
        ok += int(float(same) > float(cross))
        done += 1
    return ok / max(1, done)


@torch.no_grad()
def gate_B_slow(model, tok, char_table, pad_id, device) -> dict:
    def z_of(text: str) -> torch.Tensor:
        ids = [i for i in tok.encode(text).ids if i != pad_id][-MAX_ARCS:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        return slow_endpoint(model, x, char_table, pad_id)[0]

    def cos(a, b):
        return float(F.cosine_similarity(a, b, dim=-1))

    para = [cos(z_of(a), z_of(b)) for a, b in s179.PARAPHRASE_PAIRS]
    hard = [cos(z_of(a), z_of(b)) for a, b in s179.HARD_PAIRS]
    return {"para": float(np.mean(para)), "hard": float(np.mean(hard)), "gap_hard_minus_para": float(np.mean(hard) - np.mean(para))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage190 start {datetime.now(timezone.utc).isoformat()}")
    log("S4: doc-level content invariant (InfoNCE on slow endpoint) + gate B")

    items = [json.loads(l) for l in EXAM.read_text(encoding="utf-8").splitlines() if l.strip()]
    items_mid = [it for it in items if it["type"] == "next_tok"][:N_MID_EVAL]

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

    # 187 baseline for G2/G3 comparison
    base = s187.SelfModel(len(itos), V).to(device)
    base.load_state_dict(torch.load(CKPT_187, map_location=device, weights_only=False)["model"])
    base.eval()
    base_doclink = doclink_acc(base, hold_docs, char_table, pad_id, device, random.Random(7))
    base_B = gate_B_slow(base, tok, char_table, pad_id, device)
    log(f"187 baseline: doclink={base_doclink:.3f} B para={base_B['para']:.3f} hard={base_B['hard']:.3f} gap={base_B['gap_hard_minus_para']:.3f}")

    torch.manual_seed(SEED)
    model = s187.SelfModel(len(itos), V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    running, run_con = None, None
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        ids = s185.sample_id_batch(train_docs, MICRO, rng, device, pad_id)
        pad = ids == pad_id
        logits, surprise, pred_loss = model.forward_all(char_table[ids], pad)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()

        pair = sample_doc_windows(train_docs, rng, pad_id, device)
        lc = torch.tensor(0.0, device=device)
        if pair is not None:
            za = slow_endpoint(model, pair[0], char_table, pad_id)
            zb = slow_endpoint(model, pair[1], char_table, pad_id)
            lc = infonce(za, zb)
            loss = loss + W_CON * lc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(ce) if running is None else 0.95 * running + 0.05 * float(ce)
        run_con = float(lc) if run_con is None else 0.95 * run_con + 0.05 * float(lc)
        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            mid = s185.score_exam(model, char_table, pad_id, items_mid, device, only_type="next_tok")
            dl = doclink_acc(model, hold_docs, char_table, pad_id, device, random.Random(7), n=40)
            log(
                f"  step {step}: ce~{running:.3f} con~{run_con:.3f} next_tok(mid)={mid.get('next_tok_acc', 0):.3f} "
                f"doclink={dl:.3f} ({time.time()-t0:.0f}s)"
            )
            model.train()
            torch.save({"model": model.state_dict(), "step": step}, CKPT_OUT)

    model.eval()
    full = s185.score_exam(model, char_table, pad_id, items, device)
    next_tok = full.get("next_tok_acc", 0.0)
    dl = doclink_acc(model, hold_docs, char_table, pad_id, device, random.Random(7))
    B = gate_B_slow(model, tok, char_table, pad_id, device)

    g1 = next_tok >= BASE_NEXT - 0.03
    g2 = dl > base_doclink
    g3 = B["gap_hard_minus_para"] < base_B["gap_hard_minus_para"]
    strong = B["para"] > B["hard"]
    if g1 and g2 and strong:
        overall = "MEANING_OVER_FORM_YES"
    elif g1 and g2 and g3:
        overall = "CONTENT_INV_PROGRESS"
    else:
        overall = "CONTENT_INV_PARTIAL_" + "".join(n for n, ok in (("1", g1), ("2", g2), ("3", g3)) if not ok)

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "content_invariant_190",
        "overall": overall,
        "gates": {
            "G1_ce_preserved": {"next_tok": next_tok, "baseline_187": BASE_NEXT, "ok": g1},
            "G2_doclink": {"model": dl, "baseline_187": base_doclink, "ok": g2},
            "G3_gateB": {"model": B, "baseline_187": base_B, "gap_shrunk": g3, "para_gt_hard": strong},
        },
        "exam_full": full,
        "w_con": W_CON,
        "steps": args.steps,
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage190 — content invariant (doc-level InfoNCE on slow)",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- G1: next_tok={next_tok:.3f} vs 0.727 → {g1}",
                f"- G2: doclink={dl:.3f} vs 187 {base_doclink:.3f} → {g2}",
                f"- G3: para={B['para']:.3f} hard={B['hard']:.3f} gap={B['gap_hard_minus_para']:.3f} "
                f"(187 gap {base_B['gap_hard_minus_para']:.3f}) shrunk={g3} para>hard={strong}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(
        f"[190] {overall} | G1 {next_tok:.3f} | G2 {dl:.3f}/{base_doclink:.3f} | "
        f"G3 para={B['para']:.3f} hard={B['hard']:.3f} (187: {base_B['para']:.3f}/{base_B['hard']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
