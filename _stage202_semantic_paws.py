"""
Stage 202 — B capability test (path A): can the FROZEN curve representation be made
semantically invariant when given a real meaning signal? Confirm-or-refute on 4GB.

Data: PAWS (adversarial paraphrase — high lexical overlap, label = same-meaning or not).
This is exactly the hard-pair problem: surface says "same", meaning may differ.

Method (NON-DESTRUCTIVE): P1 encoder FROZEN. Train only a semantic head with ATTENTION
pooling over per-token fast states (learns to down-weight shared/function words that make
hard pairs collapse under mean-pool) -> z_sem. Online-contrastive loss on PAWS labels.

Eval:
  - PAWS test accuracy (best cos threshold on val) vs lexical-overlap baseline (~chance by design)
  - INVERSION on the 179 pairs: para_sim > hard_sim (meaning finally beats spelling)
  - fair GPT baseline: identical head on GPT hidden states (is the curve competitive?)

Gates:
  G_paws       curve test acc >= 0.70 (chance/lexical ~0.55)
  G_inversion  179 para_sim > hard_sim
  G_vs_gpt     curve acc >= gpt acc - 0.03
  verdict: G_paws & G_inversion -> SEM_B_CONFIRMED ; G_paws only -> SEM_B_PARTIAL ; else SEM_B_NO

  python _stage202_semantic_paws.py
"""
from __future__ import annotations

# NOTE: datasets/pyarrow MUST be imported before torch on this Windows box, else segfault.
from datasets import load_dataset

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

import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
from _stage191_night import PAD, SelfModelXL, load_data
from _stage196_tapelm import load_gpt

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
DECISION = RES / "stage202_decision.json"
MINI = RES / "stage202_mini.md"
LOG = RES / "_stage202_log.txt"

SEED = 202
D_SEM = 128
MAX_ARCS = 64
EPOCHS = 4
BATCH = 128
LR = 5e-4
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

    def forward(self, states, mask):  # states[B,T,d], mask[B,T] True=valid
        scores = self.att(states).squeeze(-1).masked_fill(~mask, -1e9)
        w = scores.softmax(-1).unsqueeze(-1)
        pooled = (w * states).sum(1)
        return F.normalize(self.proj(pooled), dim=-1)


def lexical_overlap(a, b):
    wa, wb = set(WORD_RE.findall(a.lower())), set(WORD_RE.findall(b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage202 start {datetime.now(timezone.utc).isoformat()}")
    log("B capability via PAWS on FROZEN curve encoder + attention head (4GB)")
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
    for p in p1.parameters():
        p.requires_grad_(False)
    gm = load_gpt(device)
    d = p1.head.in_features // 2
    log(f"models loaded (fast dim={d}) ({time.time()-t0:.0f}s)")

    ds = load_dataset("paws", "labeled_final")
    train = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["train"]]
    val = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["validation"]]
    test = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["test"]]
    log(f"PAWS train={len(train)} val={len(val)} test={len(test)} ({time.time()-t0:.0f}s)")

    # lexical-overlap baseline (PAWS is adversarial -> should be ~chance)
    def lex_acc(split):
        best = 0.0
        ov = np.array([lexical_overlap(a, b) for a, b, _ in split])
        y = np.array([l for _, _, l in split])
        for thr in np.linspace(0.1, 0.95, 40):
            best = max(best, max(((ov >= thr) == y).mean(), ((ov < thr) == y).mean()))
        return float(best)

    lex = lex_acc(test)
    log(f"lexical-overlap baseline test acc={lex:.3f}")

    # ---- batched frozen encoders returning per-token states ----
    def curve_states(sents):
        ids = [[i for i in tok.encode(s).ids if i != pad_id][:MAX_ARCS] or [pad_id] for s in sents]
        T = max(len(x) for x in ids)
        x = torch.full((len(ids), T), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(ids):
            x[i, : len(seq)] = torch.tensor(seq, device=device)
        pad = x == pad_id
        with torch.no_grad():
            arcs = p1._arcs(char_table[x], x)
            fast = p1.fast(arcs, pad_mask=pad)
        return fast, ~pad

    @torch.no_grad()
    def gpt_states(sents):
        ids = [[i for i in tok.encode(s).ids if i != pad_id][:MAX_ARCS] or [pad_id] for s in sents]
        T = max(len(x) for x in ids)
        x = torch.full((len(ids), T), pad_id, dtype=torch.long, device=device)
        m = torch.zeros((len(ids), T), dtype=torch.bool, device=device)
        for i, seq in enumerate(ids):
            x[i, : len(seq)] = torch.tensor(seq, device=device)
            m[i, : len(seq)] = True
        h = gm.transformer(input_ids=x, attention_mask=m.long()).last_hidden_state
        return h, m

    def train_head(states_fn, tag):
        head = SemHead(d).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
        idx = list(range(len(train)))
        for ep in range(EPOCHS):
            rng.shuffle(idx)
            running = None
            for b in range(0, len(idx), BATCH):
                batch = [train[i] for i in idx[b : b + BATCH]]
                s1 = [x[0] for x in batch]
                s2 = [x[1] for x in batch]
                y = torch.tensor([x[2] for x in batch], dtype=torch.float, device=device)
                st1, m1 = states_fn(s1)
                st2, m2 = states_fn(s2)
                z1 = head(st1, m1)
                z2 = head(st2, m2)
                cos = (z1 * z2).sum(-1)
                loss = (y * (1 - cos) + (1 - y) * F.relu(cos - MARGIN)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
            log(f"  [{tag}] epoch {ep+1}/{EPOCHS} loss~{running:.4f} ({time.time()-t0:.0f}s)")
        head.eval()

        @torch.no_grad()
        def cos_of(split):
            out = []
            for b in range(0, len(split), 256):
                bs = split[b : b + 256]
                st1, m1 = states_fn([x[0] for x in bs])
                st2, m2 = states_fn([x[1] for x in bs])
                out.append(((head(st1, m1) * head(st2, m2)).sum(-1)).cpu().numpy())
            return np.concatenate(out)

        yv = np.array([x[2] for x in val])
        cv = cos_of(val)
        thr = max(np.linspace(-0.2, 0.95, 60), key=lambda t: ((cv >= t).astype(int) == yv).mean())
        yt = np.array([x[2] for x in test])
        ct = cos_of(test)
        acc = float(((ct >= thr).astype(int) == yt).mean())

        # 179 pairs
        @torch.no_grad()
        def z(t):
            st, m = states_fn([t])
            return head(st, m)[0]

        para = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.PARAPHRASE_PAIRS]))
        hard = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.HARD_PAIRS]))
        return {"paws_acc": acc, "thr": float(thr), "para": para, "hard": hard, "inversion": para > hard}

    curve = train_head(curve_states, "curve")
    log(f"curve: paws={curve['paws_acc']:.3f} para={curve['para']:.3f} hard={curve['hard']:.3f} inv={curve['inversion']}")
    gpt = train_head(gpt_states, "gpt")
    log(f"gpt:   paws={gpt['paws_acc']:.3f} para={gpt['para']:.3f} hard={gpt['hard']:.3f} inv={gpt['inversion']}")

    g_paws = curve["paws_acc"] >= 0.70
    g_inv = curve["inversion"]
    g_vs_gpt = curve["paws_acc"] >= gpt["paws_acc"] - 0.03
    if g_paws and g_inv:
        overall = "SEM_B_CONFIRMED"
    elif g_paws:
        overall = "SEM_B_PARTIAL"
    else:
        overall = "SEM_B_NO"

    gates = {"g_paws": g_paws, "g_inversion": g_inv, "g_vs_gpt": g_vs_gpt}
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "semantic_paws_202",
        "overall": overall,
        "gates": gates,
        "curve": curve,
        "gpt_baseline": gpt,
        "lexical_overlap_test_acc": lex,
        "note": "FROZEN P1 encoder + attention-pool semantic head trained on PAWS; non-destructive "
        "(generation/memory/calibration untouched); tests B CAPABILITY, not free emergence",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage202 — B capability via PAWS (frozen encoder + attention head)",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- curve: PAWS test acc **{curve['paws_acc']:.3f}** | 179 para {curve['para']:.3f} / hard {curve['hard']:.3f} "
                f"(**inversion={curve['inversion']}**)",
                f"- gpt baseline: PAWS acc {gpt['paws_acc']:.3f} | para {gpt['para']:.3f} / hard {gpt['hard']:.3f} (inv={gpt['inversion']})",
                f"- lexical-overlap baseline: {lex:.3f} (PAWS adversarial ~chance)",
                "",
                f"gates: {gates}",
                "",
                "Non-destructive: P1 frozen; head is a separate branch. Confirms whether the curve representation "
                "CAN encode meaning over spelling given a meaning signal.",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[202] {overall} | curve paws={curve['paws_acc']:.3f} inv={curve['inversion']} (para {curve['para']:.2f}/hard {curve['hard']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
