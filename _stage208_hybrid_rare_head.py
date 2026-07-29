"""
Stage 208 — hybrid for variant A: a word-level fp RERANKER for rare words, on top of the frozen A stack.

Why this is not a rerun of 207: 207 asked the model to GENERATE the next fingerprint (regress a spelling
code) and failed. Here the fp side is DISCRIMINATIVE — it only has to prefer the right candidate among a
few, using the frozen context state that we already know is informative (191 next_tok 0.867).

  A (baseline)  : mean per-piece log-prob of the candidate word's BPE pieces (frozen CE head)
  fp reranker   : score = <W(h_ctx), fp(word)> ; only W trains, encoder frozen
  combined      : z-scored A + w * z-scored fp
  gated         : w chosen per item by fp-lexicon surprise of the CANDIDATES (read-only, no gold peeking)

Candidates are frequency-matched (exam-v2 discipline) so nothing can be won by unigram frequency.

Gates:
  G1 no_degrade  gated overall acc >= A overall acc - 0.01
  G2 rare_win    gated acc on the RARE band >= A + 0.10
  G3 read_only   encoder/head params unchanged (assert)

  python _stage208_hybrid_rare_head.py
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, span_logprob_x
from _stage194_fp_fact_memory import FpBank

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage208_decision.json"
MINI = RES / "stage208_mini.md"
LOG = RES / "_stage208_log.txt"

SEED = 208
CORPUS_CHARS = 12_000_000
TRAIN_FRAC = 0.85
V_COMMON = 8_000     # "common" band / fp-lexicon of seen words
V_RARE = 40_000      # rare band upper bound
N_TRAIN_POS = 12_000
N_EVAL_BAND = 700
N_NEG = 15
STEPS = 2500
BATCH = 128
LR = 3e-4
WORD_RE = re.compile(r"[A-Za-z]{3,}")
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def zs(v):
    a = np.asarray(v, dtype=np.float64)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage208 start {datetime.now(timezone.utc).isoformat()}")
    log("hybrid: discriminative word-level fp reranker for rare words over frozen A")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sig_before = sum(float(p.abs().sum()) for p in model.parameters())
    bank = FpBank(model, stoi, device)
    log(f"frozen A loaded ({time.time()-t0:.0f}s)")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    enc = tok.encode(text)
    ids, offs = enc.ids, enc.offsets
    tok_at = {a: i for i, (a, b) in enumerate(offs)}
    freq = Counter(WORD_RE.findall(text))
    vocab = [w for w, _ in freq.most_common(V_RARE)]
    rank = {w: i for i, w in enumerate(vocab)}
    band_words = {"common": vocab[:V_COMMON], "rare": vocab[V_COMMON:V_RARE]}
    log(f"tokens={len(ids):,} distinct_words={len(freq):,} common={len(band_words['common'])} rare={len(band_words['rare'])} ({time.time()-t0:.0f}s)")

    # word-level items: (token index of the leading-space token, word)
    items_all = []
    for m in WORD_RE.finditer(text):
        s, e = m.start(), m.end()
        if s == 0 or text[s - 1] != " ":
            continue
        i = tok_at.get(s - 1)
        if i is None or i < MAX_ARCS + 1:  # need a FULL context: no padding anywhere (pads break attention)
            continue
        w = m.group(0)
        r = rank.get(w)
        if r is None:
            continue
        items_all.append((i, w, r))
    split = int(TRAIN_FRAC * len(items_all))
    train_items, eval_items = items_all[:split], items_all[split:]
    log(f"word positions: train={len(train_items):,} eval={len(eval_items):,} ({time.time()-t0:.0f}s)")

    # ---------- frozen context state h = [fast; slow] at the last context position ----------
    @torch.no_grad()
    def ctx_states(tok_idx_list, bsz=64):
        out = []
        for b0 in range(0, len(tok_idx_list), bsz):
            chunk = tok_idx_list[b0 : b0 + bsz]
            rows = [list(ids[i - MAX_ARCS : i]) for i in chunk]  # exactly MAX_ARCS real arcs, no padding
            x = torch.tensor(rows, dtype=torch.long, device=device)
            pad = x == pad_id
            arcs = model._arcs(char_table[x], x)
            fast = model.fast(arcs, pad_mask=pad)
            slow, _, _ = model.slow(arcs, pad)
            out.append(torch.cat([fast, slow], dim=-1)[:, -1].float())
        return torch.cat(out, 0)

    # ---------- baseline A: mean per-piece logprob of a candidate word ----------
    cand_cache: dict[str, list[int]] = {}

    def pieces(w):
        if w not in cand_cache:
            cand_cache[w] = [t for t in tok.encode(" " + w).ids if t != pad_id][:8] or [pad_id]
        return cand_cache[w]

    def a_scores(tok_i, cands):
        """Use the established span scorer verbatim (truncate-only, no padding)."""
        ctx = list(ids[max(0, tok_i - MAX_ARCS) : tok_i])
        return [span_logprob_x(model, char_table, pad_id, ctx, pieces(w), device) for w in cands]

    # ---------- fp lexicon (common words) for the read-only gate ----------
    lex = torch.cat([bank.fp(band_words["common"][i : i + 4096]) for i in range(0, V_COMMON, 4096)], 0)

    def surprise(words):
        return (1.0 - (bank.fp(words) @ lex.T).max(dim=-1).values).mean().item()

    # ---------- train the fp reranker (discriminative) ----------
    tr = train_items[: N_TRAIN_POS]
    H = ctx_states([i for i, _, _ in tr])
    tgt_fp = bank.fp([w for _, w, _ in tr])
    log(f"train states {tuple(H.shape)} ({time.time()-t0:.0f}s)")

    class Rerank(nn.Module):
        def __init__(self, d_in, d_fp=256):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, d_fp))

        def forward(self, h):
            return F.normalize(self.net(h), dim=-1)

    rr = Rerank(H.size(1)).to(device)
    opt = torch.optim.AdamW(rr.parameters(), lr=LR, weight_decay=0.01)
    ranks_tr = np.array([r for _, _, r in tr])
    running = None
    for step in range(1, STEPS + 1):
        sel = torch.randint(0, H.size(0), (BATCH,))
        h = H[sel.to(device)]
        pos = tgt_fp[sel.to(device)]
        # frequency-matched negatives: same band window as the positive
        negs = []
        for j in sel.tolist():
            r = int(ranks_tr[j])
            lo, hi = max(0, int(r * 0.5)), min(V_RARE - 1, max(int(r * 2), r + 20))
            negs.append([vocab[rng.randint(lo, hi)] for _ in range(N_NEG)])
        neg_fp = bank.fp([w for row in negs for w in row]).view(BATCH, N_NEG, -1)
        q = rr(h)
        s_pos = (q * pos).sum(-1, keepdim=True)
        s_neg = torch.einsum("bd,bnd->bn", q, neg_fp)
        loss = F.cross_entropy(torch.cat([s_pos, s_neg], 1) / 0.07, torch.zeros(BATCH, dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running = float(loss) if running is None else 0.98 * running + 0.02 * float(loss)
        if step % 500 == 0 or step == STEPS:
            log(f"  step {step}: rerank_loss~{running:.3f} ({time.time()-t0:.0f}s)")
    rr.eval()

    # ---------- build eval items per band, dev/test split ----------
    def make_band(band, n):
        lo, hi = (0, V_COMMON) if band == "common" else (V_COMMON, V_RARE)
        pool = [it for it in eval_items if lo <= it[2] < hi]
        rng.shuffle(pool)
        out = []
        for i, w, r in pool[:n]:
            cl, ch = max(lo, int(r * 0.5)), min(hi - 1, max(int(r * 2), r + 20))
            cands = [w]
            while len(cands) < 4:
                c = vocab[rng.randint(cl, ch)]
                if c not in cands:
                    cands.append(c)
            order = list(range(4))
            rng.shuffle(order)
            shuf = [cands[k] for k in order]
            out.append({"tok_i": i, "cands": shuf, "gold": order.index(0), "band": band})
        return out

    ev = make_band("common", N_EVAL_BAND) + make_band("rare", N_EVAL_BAND)
    rng.shuffle(ev)
    dev, test = ev[: len(ev) // 2], ev[len(ev) // 2 :]
    log(f"eval items dev={len(dev)} test={len(test)} ({time.time()-t0:.0f}s)")

    @torch.no_grad()
    def annotate(rows):
        Hs = ctx_states([r["tok_i"] for r in rows])
        Q = rr(Hs)
        for k, r in enumerate(rows):
            r["a"] = a_scores(r["tok_i"], r["cands"])
            r["fp"] = [float(Q[k] @ f) for f in bank.fp(r["cands"])]
            r["sur"] = surprise(r["cands"])
        return rows

    dev, test = annotate(dev), annotate(test)
    sur_thresh = float(np.median([r["sur"] for r in dev]))

    def acc(rows, w_fn):
        ok_by = {"common": [0, 0], "rare": [0, 0], "all": [0, 0]}
        for r in rows:
            w = w_fn(r)
            comb = zs(r["a"]) + w * zs(r["fp"])
            hit = int(int(np.argmax(comb)) == r["gold"])
            for key in (r["band"], "all"):
                ok_by[key][0] += hit
                ok_by[key][1] += 1
        return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in ok_by.items()}

    a_only = acc(test, lambda r: 0.0)
    fp_only = acc(test, lambda r: 1e6)  # fp dominates
    best_w, best_dev = 0.0, -1.0
    for w in W_GRID:
        d = acc(dev, lambda r, w=w: w)["all"]
        if d > best_dev:
            best_dev, best_w = d, w
    combined = acc(test, lambda r: best_w)
    # gated: separate weights for low/high candidate-surprise items, both tuned on dev
    best_pair, best_dev_g = (0.0, 0.0), -1.0
    for wl in W_GRID:
        for wh in W_GRID:
            d = acc(dev, lambda r, wl=wl, wh=wh: wh if r["sur"] > sur_thresh else wl)["all"]
            if d > best_dev_g:
                best_dev_g, best_pair = d, (wl, wh)
    gated = acc(test, lambda r: best_pair[1] if r["sur"] > sur_thresh else best_pair[0])
    sig_after = sum(float(p.abs().sum()) for p in model.parameters())

    log(f"A only     : all={a_only['all']:.3f} common={a_only['common']:.3f} rare={a_only['rare']:.3f}")
    log(f"fp only    : all={fp_only['all']:.3f} common={fp_only['common']:.3f} rare={fp_only['rare']:.3f}")
    log(f"combined w={best_w}: all={combined['all']:.3f} common={combined['common']:.3f} rare={combined['rare']:.3f}")
    log(f"gated {best_pair}: all={gated['all']:.3f} common={gated['common']:.3f} rare={gated['rare']:.3f}")

    g1 = gated["all"] >= a_only["all"] - 0.01
    g2 = gated["rare"] >= a_only["rare"] + 0.10
    g3 = abs(sig_before - sig_after) < 1e-3
    if g1 and g2 and g3:
        overall = "HYBRID_RARE_WIN"
    elif g1 and gated["rare"] >= a_only["rare"] + 0.03 and g3:
        overall = "HYBRID_RARE_PARTIAL"
    else:
        overall = "HYBRID_NO_GAIN"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "hybrid_rare_word_head_208",
        "overall": overall,
        "A_only": a_only,
        "fp_only": fp_only,
        "combined": {"weight": best_w, **combined},
        "gated": {"weights_low_high": list(best_pair), "surprise_threshold": sur_thresh, **gated},
        "gates": {"g1_no_degrade": g1, "g2_rare_win": g2, "g3_read_only": g3},
        "config": {
            "N_TRAIN_POS": len(tr), "STEPS": STEPS, "N_EVAL_BAND": N_EVAL_BAND,
            "V_COMMON": V_COMMON, "V_RARE": V_RARE, "chance": 0.25,
        },
        "note": "candidates are frequency-matched within band (no unigram shortcut); the fp side is a "
        "discriminative reranker over the frozen context state, not a generative fp predictor (cf. 207)",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage208 — hybrid rare-word fp reranker over frozen A",
                "",
                f"**Overall:** `{overall}`",
                "",
                "| scorer | all | common band | rare band |",
                "|--------|-----|-------------|-----------|",
                f"| A only (BPE CE head) | {a_only['all']:.3f} | {a_only['common']:.3f} | {a_only['rare']:.3f} |",
                f"| fp reranker only | {fp_only['all']:.3f} | {fp_only['common']:.3f} | {fp_only['rare']:.3f} |",
                f"| combined (w={best_w}) | {combined['all']:.3f} | {combined['common']:.3f} | {combined['rare']:.3f} |",
                f"| gated by fp-surprise {best_pair} | **{gated['all']:.3f}** | {gated['common']:.3f} | **{gated['rare']:.3f}** |",
                "",
                f"- 4-way, frequency-matched candidates within band, chance 0.25; test n={len(test)}",
                f"- gates: no_degrade={g1} rare_win={g2} read_only={g3}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[208] {overall} | A all={a_only['all']:.3f} rare={a_only['rare']:.3f} -> gated all={gated['all']:.3f} rare={gated['rare']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
