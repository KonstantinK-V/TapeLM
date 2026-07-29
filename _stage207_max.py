"""
Stage 207 MAX — same falsification protocol as 207, scaled to full WikiText-103 train.

Streaming corpus (no full-text RAM hold): two-pass word-rank array.
Defaults tuned for ~500MB wiki + long training on RTX 3050.

  python _stage207_max.py
  python _stage207_max.py --steps 12000 --v-lex 60000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import FpBank
from _stage207_curve_thinking import (
    D_MODEL,
    MAXLEN,
    N_EVAL,
    N_HEAD,
    N_LAYER,
    TEMP,
    Trunk,
    log,
)

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage207_max_decision.json"
MINI = RES / "stage207_max_mini.md"
LOG = RES / "_stage207_max_log.txt"

SEED = 2071
CHUNK = 8_000_000
V_CE = 8_000
MAX_NCE = 512  # cap InfoNCE rows per step (full batch*T can OOM on 4GB)
RANKS_MEMMAP = RES / "_stage207_max_ranks.mmap"
WORD_RE = re.compile(r"[a-z]{2,}")


def stream_words(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            yield from WORD_RE.findall(block.lower())


def build_ranks_safe(path: Path, v_lex: int):
    log("pass 1: word frequencies (streaming)…")
    t0 = time.time()
    freq = Counter(stream_words(path))
    vocab = [w for w, _ in freq.most_common(v_lex)]
    word2rank = {w: i for i, w in enumerate(vocab)}
    rare = v_lex
    log(f"  distinct={len(freq):,} lexicon={len(vocab):,} ({time.time()-t0:.0f}s)")
    log("pass 2: rank sequence (streaming, memmap)…")
    t1 = time.time()
    # count tokens
    n_tok = sum(1 for _ in stream_words(path))
    log(f"  counting done: {n_tok:,} tokens ({time.time()-t1:.0f}s)")
    t2 = time.time()
    mm = np.memmap(RANKS_MEMMAP, dtype=np.int32, mode="w+", shape=(n_tok,))
    j = 0
    for w in stream_words(path):
        mm[j] = word2rank.get(w, rare)
        j += 1
        if j % 10_000_000 == 0:
            mm.flush()
            log(f"  wrote {j:,}/{n_tok:,} ({time.time()-t2:.0f}s)")
    mm.flush()
    log(f"  memmap {RANKS_MEMMAP.name} ({time.time()-t2:.0f}s)")
    return mm, vocab, word2rank, rare, len(freq), n_tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--v-lex", type=int, default=80_000)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message=".*nested_tensor.*")
    LOG.write_text("", encoding="utf-8")
    log(f"Stage207-MAX start {datetime.now(timezone.utc).isoformat()}")
    log(f"full wiki stream | steps={args.steps} batch={args.batch} V_LEX={args.v_lex}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    if not WIKI.exists():
        log(f"ERROR: missing {WIKI}")
        return 1
    log(f"wiki bytes={WIKI.stat().st_size:,}")

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    p1 = SelfModelXL(n_char, V).to(device)
    p1.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    p1.eval()
    for p in p1.parameters():
        p.requires_grad_(False)
    bank = FpBank(p1, stoi, device)

    ranks, vocab, word2rank, RARE, n_distinct, n_tok = build_ranks_safe(WIKI, args.v_lex)
    V_LEX = len(vocab)

    fp_rows = []
    for i in range(0, V_LEX, 2048):
        fp_rows.append(bank.fp(vocab[i : i + 2048]))
        if (i // 2048) % 10 == 0 and i:
            log(f"  fp lexicon encode {i}/{V_LEX} ({time.time()-t0:.0f}s)")
    FP = torch.cat(fp_rows, 0)
    ZERO = torch.zeros(256, device=device)
    log(f"fp table {tuple(FP.shape)} ({time.time()-t0:.0f}s)")

    n_train = int(0.9 * n_tok)
    train_hi = n_train - MAXLEN - 1
    eval_r = ranks[n_train:]
    eval_len = n_tok - n_train

    def input_fp(ids):
        idx = torch.from_numpy(ids).to(device)
        return (
            torch.where(
                (idx == RARE).unsqueeze(-1),
                ZERO.expand(*idx.shape, 256),
                FP[idx.clamp(max=V_LEX - 1)],
            ),
            idx,
        )

    def draw_train(bsz):
        seqs = []
        for _ in range(bsz):
            s = rng.randrange(train_hi)
            seqs.append(np.asarray(ranks[s : s + MAXLEN + 1], dtype=np.int64))
        arr = np.stack(seqs, 0)
        return arr[:, :-1], arr[:, 1:]

    curve = Trunk(d_out=256).to(device)
    ce = Trunk(d_out=V_CE + 1).to(device)
    n_par = sum(p.numel() for p in curve.parameters())
    opt_c = torch.optim.AdamW(curve.parameters(), lr=args.lr, weight_decay=0.01)
    opt_e = torch.optim.AdamW(ce.parameters(), lr=args.lr, weight_decay=0.01)

    def ce_target(next_ids):
        return torch.from_numpy(np.where(next_ids < V_CE, next_ids, V_CE)).to(device)

    rc = re_ = None
    uniform_floor = np.log(max(2, args.batch * MAXLEN))
    for step in range(1, args.steps + 1):
        xin, xnext = draw_train(args.batch)
        infp, _ = input_fp(xin)
        pred = F.normalize(curve(infp), dim=-1)
        nxt = torch.from_numpy(xnext).to(device)
        valid = nxt != RARE
        P = pred[valid]
        tgt_ids = nxt[valid]
        if P.size(0) < 2:
            continue
        if P.size(0) > MAX_NCE:
            sel = torch.randperm(P.size(0), device=device)[:MAX_NCE]
            P = P[sel]
            tgt_ids = tgt_ids[sel]
        Tt = FP[tgt_ids]
        logits = (P @ Tt.T) / TEMP
        same = tgt_ids.unsqueeze(0) == tgt_ids.unsqueeze(1)
        eye = torch.eye(same.size(0), dtype=torch.bool, device=device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
        loss_c = F.cross_entropy(logits, torch.arange(P.size(0), device=device))
        opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        opt_c.step()

        logit_e = ce(infp)
        loss_e = F.cross_entropy(logit_e.reshape(-1, V_CE + 1), ce_target(xnext).reshape(-1))
        opt_e.zero_grad(set_to_none=True)
        loss_e.backward()
        opt_e.step()

        rc = float(loss_c) if rc is None else 0.995 * rc + 0.005 * float(loss_c)
        re_ = float(loss_e) if re_ is None else 0.995 * re_ + 0.005 * float(loss_e)
        if step % args.log_every == 0 or step == args.steps:
            log(
                f"  step {step}/{args.steps}: curve_nce~{rc:.3f} ce~{re_:.3f} "
                f"(floor~{uniform_floor:.2f}) ({time.time()-t0:.0f}s)"
            )
    curve.eval()
    ce.eval()

    @torch.no_grad()
    def eval_rank(lo, hi, n):
        got_c = got_e = tot = 0
        erng = random.Random(SEED + 99)
        tries = 0
        while tot < n and tries < n * 50:
            tries += 1
            s = erng.randrange(eval_len - MAXLEN - 1)
            seq = np.asarray(eval_r[s : s + MAXLEN + 1], dtype=np.int64)
            nid = int(seq[-1])
            if not (lo <= nid < hi):
                continue
            xin = seq[:-1][None, :]
            infp, _ = input_fp(xin)
            pred = F.normalize(curve(infp), dim=-1)[0, -1]
            logit_e = ce(infp)[0, -1]
            cand = [nid]
            while len(cand) < 4:
                c = erng.randrange(lo, hi)
                if c != nid and c not in cand:
                    cand.append(c)
            order = list(range(4))
            erng.shuffle(order)
            shuf = [cand[i] for i in order]
            gold = order.index(0)
            sc_c = [float(pred @ FP[c]) for c in shuf]
            sc_e = [float(logit_e[c if c < V_CE else V_CE]) + 1e-6 * erng.random() for c in shuf]
            got_c += int(int(np.argmax(sc_c)) == gold)
            got_e += int(int(np.argmax(sc_e)) == gold)
            tot += 1
        return got_c / max(1, tot), got_e / max(1, tot), tot

    g1_c, g1_e, n1 = eval_rank(0, V_CE, N_EVAL)
    g3_c, g3_e, n3 = eval_rank(V_CE, V_LEX, N_EVAL)

    @torch.no_grad()
    def free_run(snap, steps=50):
        s = rng.randrange(eval_len - MAXLEN - 1)
        seed = np.asarray(eval_r[s : s + 16], dtype=np.int64)
        cur = input_fp(seed[None, :])[0][0]
        drift, decoded = [], []
        for _ in range(steps):
            pred = F.normalize(curve(cur.unsqueeze(0)), dim=-1)[0, -1]
            sims = FP @ pred
            best = int(sims.argmax())
            drift.append(1.0 - float(sims[best]))
            decoded.append(best)
            nxt = FP[best] if snap else pred
            cur = torch.cat([cur, nxt.unsqueeze(0)], 0)[-MAXLEN:]
        return drift, decoded

    draw_raw, _ = free_run(snap=False)
    _, dec_snap = free_run(snap=True)
    raw_first = float(np.mean(draw_raw[:10]))
    raw_last = float(np.mean(draw_raw[-10:]))
    snap_rep = sum(int(dec_snap[i] == dec_snap[i - 1]) for i in range(1, len(dec_snap))) / (len(dec_snap) - 1)

    @torch.no_grad()
    def trunk_key(words_list):
        ids = np.array([[word2rank.get(w, RARE) for w in words_list][:MAXLEN]], dtype=np.int64)
        if ids.shape[1] < 2:
            return None
        infp, _ = input_fp(ids)
        return F.normalize(curve.hidden(infp)[0].mean(0), dim=-1)

    subs = [w for w in gen_fakes(set(vocab), rng, 120) if len(w) >= 5][:80]
    valpool = vocab[100:180]
    facts = [{"S": subs[i], "V": valpool[i]} for i in range(min(len(subs), len(valpool)))]
    keys, vals = [], []
    for f in facts:
        k = trunk_key(f"{f['S']} was appointed director of {f['V']} in 1987".split())
        if k is not None:
            keys.append(k)
            vals.append(f["V"])
    Kmat = torch.stack(keys, 0)
    qrng = random.Random(SEED + 5)
    ok = 0
    for f in facts[: len(vals)]:
        q = trunk_key(f"{f['S']} was appointed director of".split())
        if q is None:
            continue
        sims = (Kmat @ q).tolist()
        best = {}
        for v, sc in zip(vals, sims):
            best[v] = max(best.get(v, -9.9), sc)
        others = [x for x in vals if x != f["V"]]
        qrng.shuffle(others)
        cands = [f["V"]] + others[:3]
        order = list(range(len(cands)))
        qrng.shuffle(order)
        shuf = [cands[j] for j in order]
        ok += int(int(np.argmax([best.get(c, -9.9) for c in shuf])) == order.index(0))
    g4 = ok / max(1, len(vals))

    g1 = g1_c >= g1_e - 0.05
    g2 = raw_last <= raw_first + 0.15
    g3 = g3_c >= g3_e + 0.20 and g3_c >= 0.50
    g4g = g4 >= 0.80
    passed = sum([g1, g2, g3, g4g])
    overall = (
        "CURVE_THINKING_YES"
        if g1 and g3 and g2 and g4g
        else "CURVE_THINKING_PARTIAL"
        if g1 and g3
        else "CURVE_THINKING_NO"
    )

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "curve_as_thinking_207_max",
        "overall": overall,
        "gates_passed": f"{passed}/4",
        "G1_quality_invocab": {"curve": g1_c, "ce": g1_e, "n": n1, "pass": g1},
        "G3_open_vocab_oov": {"curve": g3_c, "ce": g3_e, "n": n3, "chance": 0.25, "pass": g3},
        "G2_drift": {"raw_first10": raw_first, "raw_last10": raw_last, "snap_repetition": snap_rep, "pass": g2},
        "G4_unified_memory": {"recall": g4, "pass": g4g},
        "config": {
            "wiki_bytes": WIKI.stat().st_size,
            "train_tokens": int(n_train),
            "eval_tokens": int(eval_len),
            "n_distinct": n_distinct,
            "V_CE": V_CE,
            "V_LEX": V_LEX,
            "STEPS": args.steps,
            "BATCH": args.batch,
            "params_each_M": round(n_par / 1e6, 2),
        },
        "compare_baseline_207_smoke": {
            "note": "207 used 25M chars / 3500 steps / V_LEX 40k; this run is full wiki stream",
        },
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        f"# Stage207 MAX\n\n**Overall:** `{overall}` ({passed}/4)\n\n"
        f"- tokens train/eval: {n_train:,} / {eval_len:,}\n"
        f"- G1 curve/ce: {g1_c:.3f} / {g1_e:.3f}\n"
        f"- G3 curve/ce: {g3_c:.3f} / {g3_e:.3f}\n"
        f"- G2 drift: {raw_first:.3f} -> {raw_last:.3f}\n"
        f"- G4 memory: {g4:.3f}\n",
        encoding="utf-8",
    )
    log(
        f"[207-MAX] {overall} ({passed}/4) | G1 {g1_c:.3f}/{g1_e:.3f} | G3 {g3_c:.3f}/{g3_e:.3f} | "
        f"G2 {raw_first:.3f}->{raw_last:.3f} | G4 {g4:.3f} | train_tokens={n_train:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
