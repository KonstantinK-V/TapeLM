"""
Stage 209 — on 3050: refute "variant A is structurally blind to meaning"?

Two probes (same PAWS harness as 202):
  1) Scaling grid: CE-pretrained curve + matched GPT at d128/2L, d192/4L, d256/6L (P1/P2 ckpt).
     Frozen encoder + PAWS attention head. Gates: monotonic curve PAWS acc; parity vs GPT each scale.
  2) Teacher sufficiency (MiniLM-L6-v2): train head to match teacher cosine (MSE) from frozen states.
     Gate: curve teacher-corr >= gpt - 0.05 at d256 (is semantic info in the substrate?).

  python _stage209_sem_scaling.py
"""
from __future__ import annotations

from datasets import load_dataset

import json
import math
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
import _stage191_night as s191
from _stage191_night import PAD, SelfModelXL, load_data, lr_at, sample_windows, span_logprob_x
from _stage202_semantic_paws import SemHead, lexical_overlap

RES = Path("results")
CKPT = Path("checkpoints")
CKPT_P1 = CKPT / "stage191_p1_curve.pt"
CKPT_GPT = CKPT / "stage191_p2_gpt.pt"
DECISION = RES / "stage209_decision.json"
MINI = RES / "stage209_mini.md"
LOG = RES / "_stage209_log.txt"

SEED = 209
MAX_ARCS = 64
MICRO = 24
LR = 3e-4
W_SELF = 0.05
PAWS_EPOCHS = 3
PAWS_BATCH = 128
PAWS_LR = 5e-4
MARGIN = 0.4
TEACHER_EPOCHS = 3
TEACHER_N = 2500
WORD_RE = re.compile(r"[A-Za-z]+")

SCALES = (
    {"d": 128, "L": 2, "steps": 3200, "budget_s": 720, "gpt_heads": 4},
    {"d": 192, "L": 4, "steps": 4200, "budget_s": 900, "gpt_heads": 4},
    {"d": 256, "L": 6, "steps": 0, "budget_s": 0, "gpt_heads": 8},
)


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def train_curve_quick(model, flat, off, char_table, pad_id, device, steps, budget_s, tag):
    if steps <= 0:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED + tag.__hash__() % 10000)
    t0 = time.time()
    model.train()
    running = None
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        pad = ids == pad_id
        logits, _, pred_loss = model.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(ce) if running is None else 0.95 * running + 0.05 * float(ce)
        if step % 800 == 0 or step == steps:
            log(f"    [{tag}] pretrain step {step}/{steps} ce~{running:.3f} ({time.time()-t0:.0f}s)")
        if time.time() - t0 > budget_s:
            log(f"    [{tag}] pretrain budget stop @ {step}")
            break
    model.eval()
    torch.save({"model": model.state_dict()}, CKPT / f"stage209_{tag}.pt")


def train_gpt_quick(d, L, n_head, V, pad_id, flat, off, device, steps, budget_s, tag):
    if steps <= 0:
        return
    conf = GPT2Config(
        vocab_size=V, n_positions=MAX_ARCS, n_embd=d, n_layer=L, n_head=n_head,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
    )
    gm = GPT2LMHeadModel(conf).to(device)
    opt = torch.optim.AdamW(gm.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED + 17 + d)
    t0 = time.time()
    gm.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        loss = gm(input_ids=ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 800 == 0 or step == steps:
            log(f"    [{tag}] gpt step {step}/{steps} loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
        if time.time() - t0 > budget_s:
            break
    gm.eval()
    torch.save({"model": gm.state_dict(), "conf": conf.to_dict()}, CKPT / f"stage209_{tag}.pt")
    return gm


def load_gpt_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    conf = GPT2Config(**ck["conf"]) if "conf" in ck else None
    if conf is None:
        conf = GPT2Config(**ck.get("config", {}))
    gm = GPT2LMHeadModel(conf).to(device)
    gm.load_state_dict(ck["model"])
    gm.eval()
    return gm


def paws_eval(curve_model, gpt_model, char_table, pad_id, tok, d_fast, device, train, val, test, rng):
    ds_train, ds_val, ds_test = train, val, test

    def curve_states(p1, sents):
        ids = [[i for i in tok.encode(s).ids if i != pad_id][:MAX_ARCS] or [pad_id] for s in sents]
        T = max(len(x) for x in ids)
        x = torch.full((len(ids), T), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(ids):
            x[i, : len(seq)] = torch.tensor(seq, device=device)
        pad = x == pad_id
        with torch.no_grad():
            fast = p1.fast(p1._arcs(char_table[x], x), pad_mask=pad)
        return fast, ~pad

    @torch.no_grad()
    def gpt_states(gm, sents):
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
        head = SemHead(d_fast).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=PAWS_LR, weight_decay=0.01)
        idx = list(range(len(ds_train)))
        for ep in range(PAWS_EPOCHS):
            rng.shuffle(idx)
            for b in range(0, len(idx), PAWS_BATCH):
                batch = [ds_train[i] for i in idx[b : b + PAWS_BATCH]]
                st1, m1 = states_fn([x[0] for x in batch])
                st2, m2 = states_fn([x[1] for x in batch])
                y = torch.tensor([x[2] for x in batch], dtype=torch.float, device=device)
                z1, z2 = head(st1, m1), head(st2, m2)
                cos = (z1 * z2).sum(-1)
                loss = (y * (1 - cos) + (1 - y) * F.relu(cos - MARGIN)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        head.eval()

        @torch.no_grad()
        def cos_of(split):
            out = []
            for b in range(0, len(split), 256):
                bs = split[b : b + 256]
                st1, m1 = states_fn([x[0] for x in bs])
                st2, m2 = states_fn([x[1] for x in bs])
                out.append((head(st1, m1) * head(st2, m2)).sum(-1).cpu().numpy())
            return np.concatenate(out)

        yv = np.array([x[2] for x in ds_val])
        cv = cos_of(ds_val)
        thr = max(np.linspace(-0.2, 0.95, 60), key=lambda t: ((cv >= t).astype(int) == yv).mean())
        yt = np.array([x[2] for x in ds_test])
        acc = float(((cos_of(ds_test) >= thr).astype(int) == yt).mean())

        def zsent(t):
            st, m = states_fn([t])
            return head(st, m)[0]

        para = float(np.mean([float(F.cosine_similarity(zsent(a), zsent(b), dim=-1)) for a, b in s179.PARAPHRASE_PAIRS]))
        hard = float(np.mean([float(F.cosine_similarity(zsent(a), zsent(b), dim=-1)) for a, b in s179.HARD_PAIRS]))
        return {"paws_acc": acc, "para": para, "hard": hard, "inversion": para > hard}

    c = train_head(lambda s: curve_states(curve_model, s), "curve")
    g = train_head(lambda s: gpt_states(gpt_model, s), "gpt")
    return c, g


def teacher_probe(curve_model, gpt_model, char_table, pad_id, tok, d_fast, device, teacher, train, test, rng):
    """Train head to match MiniLM cosine; report Pearson r on test."""
    sub = rng.sample(train, min(TEACHER_N, len(train)))
    s1, s2 = [x[0] for x in sub], [x[1] for x in sub]
    with torch.no_grad():
        e1 = teacher.encode(s1, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        e2 = teacher.encode(s2, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        tgt = F.cosine_similarity(e1, e2, dim=-1).cpu().numpy()

    def curve_states(sents):
        ids = [[i for i in tok.encode(s).ids if i != pad_id][:MAX_ARCS] or [pad_id] for s in sents]
        T = max(len(x) for x in ids)
        x = torch.full((len(ids), T), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(ids):
            x[i, : len(seq)] = torch.tensor(seq, device=device)
        pad = x == pad_id
        with torch.no_grad():
            fast = curve_model.fast(curve_model._arcs(char_table[x], x), pad_mask=pad)
        return fast, ~pad

    @torch.no_grad()
    def gpt_states(gm, sents):
        ids = [[i for i in tok.encode(s).ids if i != pad_id][:MAX_ARCS] or [pad_id] for s in sents]
        T = max(len(x) for x in ids)
        x = torch.full((len(ids), T), pad_id, dtype=torch.long, device=device)
        m = torch.zeros((len(ids), T), dtype=torch.bool, device=device)
        for i, seq in enumerate(ids):
            x[i, : len(seq)] = torch.tensor(seq, device=device)
            m[i, : len(seq)] = True
        h = gpt_model.transformer(input_ids=x, attention_mask=m.long()).last_hidden_state
        return h, m

    def fit(states_fn):
        head = SemHead(d_fast).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=PAWS_LR)
        for ep in range(TEACHER_EPOCHS):
            perm = rng.sample(range(len(sub)), len(sub))
            for b in range(0, len(perm), PAWS_BATCH):
                ix = perm[b : b + PAWS_BATCH]
                batch = [sub[i] for i in ix]
                st1, m1 = states_fn([x[0] for x in batch])
                st2, m2 = states_fn([x[1] for x in batch])
                pred = (head(st1, m1) * head(st2, m2)).sum(-1)
                loss = F.mse_loss(pred, torch.tensor(tgt[ix], device=device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        head.eval()

        @torch.no_grad()
        def pred_split(split):
            ps, ts = [], []
            for b in range(0, len(split), 128):
                bs = split[b : b + 128]
                t1 = [x[0] for x in bs]
                t2 = [x[1] for x in bs]
                st1, m1 = states_fn(t1)
                st2, m2 = states_fn(t2)
                ps.append((head(st1, m1) * head(st2, m2)).sum(-1).cpu().numpy())
                e1 = teacher.encode(t1, batch_size=64, show_progress_bar=False)
                e2 = teacher.encode(t2, batch_size=64, show_progress_bar=False)
                ts.append(np.array([float(F.cosine_similarity(torch.tensor(e1[i]), torch.tensor(e2[i]), dim=0))
                                    for i in range(len(t1))]))
            return np.concatenate(ps), np.concatenate(ts)

        p, t = pred_split(test)
        r = float(np.corrcoef(p, t)[0, 1]) if len(p) > 2 else 0.0
        return r

    rc = fit(lambda s: curve_states(s))
    rg = fit(lambda s: gpt_states(gpt_model, s))
    return {"curve_teacher_r": rc, "gpt_teacher_r": rg}


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage209 start {datetime.now(timezone.utc).isoformat()}")
    log("scaling + MiniLM teacher probe: is A structurally blocked on meaning?")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    ds = load_dataset("paws", "labeled_final")
    train = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["train"]]
    val = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["validation"]]
    test = [(r["sentence1"], r["sentence2"], int(r["label"])) for r in ds["test"]]
    log(f"PAWS loaded ({time.time()-t0:.0f}s)")

    try:
        from sentence_transformers import SentenceTransformer
        teacher = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
        log(f"MiniLM teacher on {device} ({time.time()-t0:.0f}s)")
    except Exception as e:
        log(f"ERROR: need sentence-transformers: {e}")
        return 1

    grid = []
    teacher_d256 = None
    for sc in SCALES:
        d, L = sc["d"], sc["L"]
        tag = f"d{d}_L{L}"
        log(f"=== scale {tag} ===")
        curve = SelfModelXL(n_char, V, d=d, n_layers=L).to(device)
        if d == 256 and L == 6:
            curve.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
            gpt = load_gpt_ckpt(CKPT_GPT, device)
            log(f"  loaded P1 + P2 ckpts ({time.time()-t0:.0f}s)")
        else:
            train_curve_quick(curve, flat, off, char_table, pad_id, device, sc["steps"], sc["budget_s"], f"curve_{tag}")
            curve.load_state_dict(torch.load(CKPT / f"stage209_curve_{tag}.pt", map_location=device, weights_only=False)["model"])
            train_gpt_quick(d, L, sc["gpt_heads"], V, pad_id, flat, off, device, sc["steps"], sc["budget_s"], f"gpt_{tag}")
            gpt = load_gpt_ckpt(CKPT / f"stage209_gpt_{tag}.pt", device)
        for p in curve.parameters():
            p.requires_grad_(False)
        d_fast = d
        c, g = paws_eval(curve, gpt, char_table, pad_id, tok, d_fast, device, train, val, test, rng)
        log(f"  PAWS curve={c['paws_acc']:.3f} gpt={g['paws_acc']:.3f} para/hard {c['para']:.2f}/{c['hard']:.2f}")
        row = {"scale": tag, "d": d, "L": L, "curve": c, "gpt": g, "delta": c["paws_acc"] - g["paws_acc"]}
        grid.append(row)
        if d == 256:
            teacher_d256 = teacher_probe(curve, gpt, char_table, pad_id, tok, d_fast, device, teacher, train, test, rng)
            log(f"  teacher r: curve={teacher_d256['curve_teacher_r']:.3f} gpt={teacher_d256['gpt_teacher_r']:.3f}")

    paws_c = [r["curve"]["paws_acc"] for r in grid]
    mono = paws_c[2] + 0.02 >= paws_c[1] >= paws_c[0] - 0.02
    parity = all(r["curve"]["paws_acc"] >= r["gpt"]["paws_acc"] - 0.03 for r in grid)
    t_ok = teacher_d256 and teacher_d256["curve_teacher_r"] >= teacher_d256["gpt_teacher_r"] - 0.05
    t_info = teacher_d256 and teacher_d256["curve_teacher_r"] >= 0.25
    if parity and mono and t_ok and t_info:
        overall = "STRUCTURAL_BLOCK_NO"
    elif parity and (mono or paws_c[2] >= 0.65):
        overall = "STRUCTURAL_BLOCK_UNLIKELY"
    else:
        overall = "STRUCTURAL_BLOCK_UNCLEAR"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "sem_scaling_teacher_209",
        "overall": overall,
        "grid": grid,
        "teacher_d256": teacher_d256,
        "gates": {
            "g_monotone_paws": mono,
            "g_parity_all_scales": parity,
            "g_teacher_parity": t_ok,
            "g_teacher_signal": t_info,
        },
        "interpretation": "STRUCTURAL_BLOCK_NO means: curve PAWS tracks GPT at every scale, "
        "accuracy tends to rise with scale, and curve states carry MiniLM geometry at least as well as GPT — "
        "3050 cannot prove B at scale, but refutes 'A can never be meaningful'.",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                f"# Stage209 — scaling + teacher probe\n\n**Overall:** `{overall}`\n",
                "| scale | curve PAWS | gpt PAWS | Δ |",
                "|-------|------------|----------|---|",
            ]
            + [f"| {r['scale']} | {r['curve']['paws_acc']:.3f} | {r['gpt']['paws_acc']:.3f} | {r['delta']:+.3f} |" for r in grid]
            + (
                [
                    "",
                    f"Teacher Pearson r @ d256: curve **{teacher_d256['curve_teacher_r']:.3f}** vs gpt {teacher_d256['gpt_teacher_r']:.3f}",
                    f"gates: mono={mono} parity={parity} teacher={t_ok} signal={t_info}",
                ]
                if teacher_d256
                else []
            )
        ),
        encoding="utf-8",
    )
    log(f"[209] {overall} | paws {[round(x,3) for x in paws_c]} | gates mono={mono} parity={parity} teacher={t_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
