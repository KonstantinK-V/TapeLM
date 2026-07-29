"""
Stage 202b — decisive B-capability test: fine-tune the encoder END-TO-END on PAWS.

202 showed a head on FROZEN features plateaus (~0.65) for BOTH curve and GPT — the
frozen small encoder is the bottleneck. Here we UNFREEZE the encoder (on a COPY; product
P1 stays frozen) and train encoder+head jointly on PAWS. Fair GPT control fine-tuned the
same way. Question: given a meaning signal AND a trainable encoder, can the CURVE reach
semantic invariance (inversion para>hard) and match GPT?

Runs on 4GB: d256/6L, short sentences (<=64 tok), small batch.

Gates:
  G_paws       curve PAWS test acc >= 0.75
  G_inversion  179 para_sim > hard_sim
  G_parity     |curve_acc - gpt_acc| <= 0.03
  verdict: G_paws & G_inversion -> SEM_B_CAP_CONFIRMED ; G_paws -> SEM_B_CAP_PARTIAL ; else SEM_B_CAP_NO

  python _stage202b_paws_finetune.py
"""
from __future__ import annotations

from datasets import load_dataset  # before torch (Windows segfault guard)

import copy
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data, score_items, span_logprob_x

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_GPT = Path("checkpoints/stage191_p2_gpt.pt")
EXAM_V3 = Path("data/stage191_exam_v3.jsonl")
DECISION = RES / "stage202b_decision.json"
MINI = RES / "stage202b_mini.md"
LOG = RES / "_stage202b_log.txt"

SEED = 2022
D_SEM = 128
MAX_ARCS = 64
EPOCHS = 3
BATCH = 48
LR_ENC = 1e-4
LR_HEAD = 5e-4
MARGIN = 0.4
WORD_RE = re.compile(r"[A-Za-z]+")


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class SemHead(nn.Module):
    def __init__(self, d, d_sem=D_SEM):
        super().__init__()
        self.att = nn.Linear(d, 1)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_sem))

    def forward(self, states, mask):
        scores = self.att(states).squeeze(-1).masked_fill(~mask, -1e9)
        w = scores.softmax(-1).unsqueeze(-1)
        return F.normalize(self.proj((w * states).sum(1)), dim=-1)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage202b start {datetime.now(timezone.utc).isoformat()}")
    log("decisive B: fine-tune encoder end-to-end on PAWS (copy; product P1 frozen)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    p1 = SelfModelXL(n_char, V).to(device)
    p1.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    p1.eval()
    d = p1.head.in_features // 2

    ds = load_dataset("paws", "labeled_final")
    train = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["train"]]
    val = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["validation"]]
    test = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["test"]]
    log(f"PAWS train={len(train)} val={len(val)} test={len(test)} ({time.time()-t0:.0f}s)")

    def ids_of(sents):
        ids = [[i for i in tok.encode(s).ids if i != pad_id][:MAX_ARCS] or [pad_id] for s in sents]
        T = max(len(x) for x in ids)
        x = torch.full((len(ids), T), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(ids):
            x[i, : len(seq)] = torch.tensor(seq, device=device)
        return x

    def curve_states(x, enc):
        pad = x == pad_id
        arcs = enc._arcs(char_table[x], x)
        fast = enc.fast(arcs, pad_mask=pad)
        return fast, ~pad

    def gpt_states(x, enc):
        m = (x != pad_id)
        h = enc.transformer(input_ids=x, attention_mask=m.long()).last_hidden_state
        return h, m

    def run(kind):
        if kind == "curve":
            enc = copy.deepcopy(p1)
            enc.train()
            states_fn = curve_states
            enc_params = list(enc.arc_enc.parameters()) + list(enc.fast.parameters())
        else:
            ck = torch.load(CKPT_GPT, map_location=device, weights_only=False)
            enc = GPT2LMHeadModel(GPT2Config(**ck["conf"])).to(device)
            enc.load_state_dict(ck["model"])
            enc.train()
            states_fn = gpt_states
            enc_params = list(enc.transformer.parameters())
        head = SemHead(d).to(device)
        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": LR_ENC}, {"params": head.parameters(), "lr": LR_HEAD}],
            weight_decay=0.01,
        )
        idx = list(range(len(train)))
        for ep in range(EPOCHS):
            rng.shuffle(idx)
            running = None
            for b in range(0, len(idx), BATCH):
                batch = [train[i] for i in idx[b : b + BATCH]]
                y = torch.tensor([x[2] for x in batch], dtype=torch.float, device=device)
                st1, m1 = states_fn(ids_of([x[0] for x in batch]), enc)
                st2, m2 = states_fn(ids_of([x[1] for x in batch]), enc)
                cos = (head(st1, m1) * head(st2, m2)).sum(-1)
                loss = (y * (1 - cos) + (1 - y) * F.relu(cos - MARGIN)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(enc_params + list(head.parameters()), 1.0)
                opt.step()
                running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
            log(f"  [{kind}] epoch {ep+1}/{EPOCHS} loss~{running:.4f} ({time.time()-t0:.0f}s)")
        enc.eval()
        head.eval()

        @torch.no_grad()
        def cos_of(split):
            out = []
            for b in range(0, len(split), 128):
                bs = split[b : b + 128]
                st1, m1 = states_fn(ids_of([x[0] for x in bs]), enc)
                st2, m2 = states_fn(ids_of([x[1] for x in bs]), enc)
                out.append(((head(st1, m1) * head(st2, m2)).sum(-1)).cpu().numpy())
            return np.concatenate(out)

        yv, cv = np.array([x[2] for x in val]), cos_of(val)
        thr = max(np.linspace(-0.2, 0.95, 60), key=lambda t: ((cv >= t).astype(int) == yv).mean())
        yt, ct = np.array([x[2] for x in test]), cos_of(test)
        acc = float(((ct >= thr).astype(int) == yt).mean())

        @torch.no_grad()
        def z(t):
            st, m = states_fn(ids_of([t]), enc)
            return head(st, m)[0]

        para = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.PARAPHRASE_PAIRS]))
        hard = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.HARD_PAIRS]))
        res = {"paws_acc": acc, "para": para, "hard": hard, "inversion": para > hard}
        # generation cost for curve copy (product P1 untouched)
        if kind == "curve":
            items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
            nt = [it for it in items if it["type"] == "next_tok"][:120]
            res["next_tok_copy"] = score_items(lambda c, cd: span_logprob_x(enc, char_table, pad_id, c, cd, device), nt, "next_tok")["next_tok_acc"]
        return res

    curve = run("curve")
    log(f"curve: paws={curve['paws_acc']:.3f} para={curve['para']:.3f} hard={curve['hard']:.3f} inv={curve['inversion']} nt_copy={curve.get('next_tok_copy')}")
    gpt = run("gpt")
    log(f"gpt:   paws={gpt['paws_acc']:.3f} para={gpt['para']:.3f} hard={gpt['hard']:.3f} inv={gpt['inversion']}")

    g_paws = curve["paws_acc"] >= 0.75
    g_inv = curve["inversion"]
    g_parity = abs(curve["paws_acc"] - gpt["paws_acc"]) <= 0.03
    if g_paws and g_inv:
        overall = "SEM_B_CAP_CONFIRMED"
    elif g_paws:
        overall = "SEM_B_CAP_PARTIAL"
    else:
        overall = "SEM_B_CAP_NO"

    gates = {"g_paws": g_paws, "g_inversion": g_inv, "g_parity": g_parity}
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "paws_finetune_202b",
        "overall": overall,
        "gates": gates,
        "curve": curve,
        "gpt": gpt,
        "note": "encoder fine-tuned end-to-end on PAWS (copy; product P1 frozen); decisive B-capability test",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage202b — decisive B: encoder fine-tune on PAWS",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- curve: PAWS **{curve['paws_acc']:.3f}** | 179 para {curve['para']:.3f} / hard {curve['hard']:.3f} "
                f"(**inversion={curve['inversion']}**) | next_tok(copy) {curve.get('next_tok_copy')}",
                f"- gpt:   PAWS {gpt['paws_acc']:.3f} | para {gpt['para']:.3f} / hard {gpt['hard']:.3f} (inversion={gpt['inversion']})",
                "",
                f"gates: {gates}",
                "",
                "Encoder fine-tuned end-to-end (copy); product P1 frozen. Tests whether the curve substrate "
                "CAN reach semantic invariance given a meaning signal + trainable encoder, at parity with GPT.",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[202b] {overall} | curve paws={curve['paws_acc']:.3f} inv={curve['inversion']} gpt={gpt['paws_acc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
