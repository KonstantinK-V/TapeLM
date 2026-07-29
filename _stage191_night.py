"""
Stage 191 — NIGHT-9h scale run (plan: results/plan_stage191_night9h.md).

Phases (idempotent — done phase = json exists, rerun skips):
  P0 data+exam : 150M chars → proper line-split id docs (npz cache) + Exam v3 (freq-matched)
  P1 curve-XL  : self-model dual-channel d256/6L, clean CE + read-only surprise, ≤15k steps
  P2 gpt-XL    : matched GPT-2 (d256/6L/T64), same data/steps
  P3 rarity    : P1 + char-trigram rarity feature in ink + surprise temperature (S3-G3 fix)
  P4 sweep     : gate B + doclink for P1/P2/P3/187-old (does scale move meaning?)
  P5 report    : night report + verdicts

NOTE: 181's build_id_docs split on \n\n (absent in this wiki file) → all prior runs actually
trained on a ~2M-char fallback slice. Tonight = ~75x data.

  python _stage191_night.py --phase all
  python _stage191_night.py --phase all --smoke   (fast end-to-end check)
"""
from __future__ import annotations

import argparse
import json
import math
import random
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
import _stage186_exam_v2 as s186
import _stage187_self_model as s187

RES = Path("results")
DATA = Path("data")
CKPT = Path("checkpoints")
LOG = RES / "_stage191_log.txt"
WIKI = Path("data/_wikitext103_train.txt")
DOCS_NPZ = DATA / "stage191_docs.npz"
CHARSET_JSON = DATA / "stage191_charset.json"
EXAM_V3 = DATA / "stage191_exam_v3.jsonl"
REPORT = RES / "stage191_night_report.md"
TOK_PATH = s177.TOK_PATH

SEED = 191
D_XL = 256
L_XL = 6
MAX_ARCS = s177.MAX_ARCS  # 64
MICRO = 16
LR = 3e-4
WARMUP = 200
W_SELF = 0.1
PAD = "[PAD]"
CORPUS_CHARS = 150_000_000
EXAM_TAIL_CHARS = 3_000_000

# budgets (seconds)
BUDGET = {"p1": 3.2 * 3600, "p2": 1.8 * 3600, "p3": 1.7 * 3600}
STEPS = {"p1": 15_000, "p2": 15_000, "p3": 10_000}
EVAL_EVERY = 2500
N_MID = 80

FAKES = s187.FAKES


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def pj(phase: str) -> Path:
    return RES / f"stage191_{phase}.json"


def save_phase(phase: str, obj: dict) -> None:
    obj["timestamp"] = datetime.now(timezone.utc).isoformat()
    pj(phase).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------- P0: data + exam ----------------


def phase0(smoke: bool) -> None:
    if pj("p0").exists():
        log("P0 done, skip")
        return
    t0 = time.time()
    n_chars = 6_000_000 if smoke else CORPUS_CHARS
    log(f"P0: reading {n_chars//1_000_000}M chars …")
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(n_chars)

    chars = sorted(set(text) | {" "})
    CHARSET_JSON.write_text(json.dumps(chars, ensure_ascii=False), encoding="utf-8")
    log(f"  charset={len(chars)+1}")

    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0

    # proper doc split: single newlines, sizable lines
    train_text = text[: -EXAM_TAIL_CHARS] if len(text) > EXAM_TAIL_CHARS * 2 else text[: len(text) // 2]
    lines = [l.strip() for l in train_text.split("\n") if len(l.strip()) >= 120]
    log(f"  lines={len(lines)}; tokenizing …")
    encs = tok.encode_batch(lines)
    flat, offsets = [], [0]
    for e in encs:
        ids = [i for i in e.ids if i != pad_id]
        if len(ids) >= 24:
            flat.extend(ids)
            offsets.append(len(flat))
    flat_a = np.asarray(flat, dtype=np.int32)
    off_a = np.asarray(offsets, dtype=np.int64)
    np.savez_compressed(DOCS_NPZ, flat=flat_a, offsets=off_a)
    log(f"  docs={len(off_a)-1} tokens={len(flat_a)} ({time.time()-t0:.0f}s)")

    # exam v3 from unseen tail
    tail = text[-EXAM_TAIL_CHARS:]
    V = tok.get_vocab_size()
    freq = np.bincount(flat_a, minlength=V).astype(np.float64) + 1.0
    s186.N_NEXT, s186.N_ENTITY, s186.N_OOD = (60, 30, 20) if smoke else (300, 150, 100)
    rng = random.Random(SEED)
    items = s186.build_exam_v2(tail, tok, pad_id, freq, rng)
    with EXAM_V3.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = {t: sum(1 for i in items if i["type"] == t) for t in ("next_tok", "entity", "ood")}

    # baselines
    logfreq = np.log(freq / freq.sum())
    uni = s186.score_with(lambda c, cd: float(np.mean([logfreq[t] for t in cd])), items)
    rb = random.Random(0)
    rnd = s186.score_with(lambda c, cd: rb.random(), items)
    log(f"  exam v3 {counts} | unigram next_tok={uni['next_tok_acc']:.3f} random={rnd['next_tok_acc']:.3f}")
    save_phase("p0", {"counts": counts, "unigram": uni, "random": rnd, "docs": len(off_a) - 1, "tokens": int(len(flat_a)), "charset": len(chars) + 1})


# ---------------- shared training utils ----------------


def load_data():
    z = np.load(DOCS_NPZ)
    flat, off = z["flat"], z["offsets"]
    chars = json.loads(CHARSET_JSON.read_text(encoding="utf-8"))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    return flat, off, stoi, len(chars) + 1


def sample_windows(flat, off, batch, rng, pad_id):
    n_docs = len(off) - 1
    xs = np.full((batch, MAX_ARCS), pad_id, dtype=np.int64)
    for b in range(batch):
        d = rng.randint(0, n_docs - 1)
        s, e = off[d], off[d + 1]
        ln = e - s
        if ln <= MAX_ARCS:
            xs[b, :ln] = flat[s:e]
        else:
            st = s + rng.randint(0, ln - MAX_ARCS)
            xs[b] = flat[st : st + MAX_ARCS]
    return torch.from_numpy(xs)


def lr_at(step, total):
    if step < WARMUP:
        return LR * step / WARMUP
    p = (step - WARMUP) / max(1, total - WARMUP)
    return LR * 0.5 * (1 + math.cos(math.pi * p))


class SelfModelXL(nn.Module):
    def __init__(self, n_char: int, V: int, d: int = D_XL, n_layers: int = L_XL, rarity: torch.Tensor | None = None, surprise_temp: bool = False):
        super().__init__()
        self.arc_enc = s177.ArcEncoder(n_char, d=d)
        self.fast = s177.ArcTransformer(d=d, n_layers=n_layers)
        self.slow = s187.SurpriseWriter(d, d)
        self.head = nn.Linear(2 * d, V, bias=False)
        self.surprise_temp = surprise_temp
        if surprise_temp:
            self.temp_w = nn.Parameter(torch.tensor(4.0))
            self.temp_b = nn.Parameter(torch.tensor(-2.0))
        if rarity is not None:
            self.register_buffer("rarity", rarity)  # [V] z-scored novelty
            self.rar_proj = nn.Linear(1, d)
        else:
            self.rarity = None

    def _arcs(self, char_ids, ids=None):
        arcs = self.arc_enc(char_ids)
        if self.rarity is not None and ids is not None:
            arcs = arcs + self.rar_proj(self.rarity[ids].unsqueeze(-1))
        return arcs

    def forward_all(self, char_ids, pad, ids=None):
        arcs = self._arcs(char_ids, ids)
        fast = self.fast(arcs, pad_mask=pad)
        slow, surprise, pred_loss = self.slow(arcs, pad)
        logits = self.head(torch.cat([fast, slow], dim=-1))
        if self.surprise_temp:
            T = 1.0 + F.softplus(self.temp_w * surprise.detach() + self.temp_b).unsqueeze(-1)
            logits = logits / T
        return logits, surprise, pred_loss


def make_logits_fn(model, char_table, pad_id, id_aware: bool):
    """expose .logits(char_ids,pad) compatible with s185.span_logprob via wrapper obj"""

    class W:
        def eval(self):
            model.eval()
            return self

        def logits(self, char_ids, pad, shuffle_tape=False):
            return model.forward_all(char_ids, pad, ids=None)[0]

    return W()


@torch.no_grad()
def span_logprob_x(model, char_table, pad_id, ctx_ids, cand_ids, device) -> float:
    seq = (ctx_ids + cand_ids)[-MAX_ARCS:]
    n_ctx = len(seq) - len(cand_ids)
    x = torch.tensor([seq], dtype=torch.long, device=device)
    pad = x == pad_id
    logits = model.forward_all(char_table[x], pad, ids=x)[0][0]
    logp = F.log_softmax(logits, dim=-1)
    return sum(float(logp[n_ctx + k - 1, tid]) for k, tid in enumerate(cand_ids)) / max(1, len(cand_ids))


@torch.no_grad()
def score_items(scorer, items, only_type=None) -> dict:
    acc = {}
    for it in items:
        t = it["type"]
        if only_type and t != only_type:
            continue
        scores = [scorer(it["ctx_ids"], c) for c in it["cand_ids"]]
        ok, n = acc.get(t, (0, 0))
        acc[t] = (ok + int(int(np.argmax(scores)) == it["gold_idx"]), n + 1)
    return {f"{t}_acc": ok / max(1, n) for t, (ok, n) in acc.items()} | {f"{t}_n": n for t, (ok, n) in acc.items()}


def train_curve(tag, model, flat, off, char_table, pad_id, items_mid, device, steps, budget_s, smoke):
    if smoke:
        steps = 120
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    t0 = time.time()
    best = -1.0
    best_step = 0
    flat_evals = 0
    running = None
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        pad = ids == pad_id
        logits, surprise, pred_loss = model.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(ce) if running is None else 0.95 * running + 0.05 * float(ce)
        if step % (40 if smoke else EVAL_EVERY) == 0 or step == steps:
            model.eval()
            mid = score_items(lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), items_mid, "next_tok")
            acc = mid.get("next_tok_acc", 0)
            el = time.time() - t0
            log(f"  [{tag}] step {step}/{steps}: ce~{running:.3f} next_tok(mid)={acc:.3f} ({el:.0f}s)")
            if acc > best + 1e-6:
                best, best_step, flat_evals = acc, step, 0
                torch.save({"model": model.state_dict(), "step": step, "mid": acc}, CKPT / f"stage191_{tag}.pt")
            else:
                flat_evals += 1
            model.train()
            if el > budget_s:
                log(f"  [{tag}] budget hit, stop")
                break
            if flat_evals >= 2 and step >= steps // 2:
                log(f"  [{tag}] early stop (flat)")
                break
    ck = torch.load(CKPT / f"stage191_{tag}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return {"best_mid": best, "best_step": best_step, "ce": running, "wall_s": time.time() - t0}


def phase1(smoke, device):
    if pj("p1").exists():
        log("P1 done, skip")
        return
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
    items_mid = [it for it in items if it["type"] == "next_tok"][:N_MID]

    torch.manual_seed(SEED)
    model = SelfModelXL(n_char, V).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"P1 curve-XL d{D_XL}/L{L_XL} params={n_par/1e6:.1f}M")
    tr = train_curve("p1_curve", model, flat, off, char_table, pad_id, items_mid, device, STEPS["p1"], BUDGET["p1"], smoke)
    full = score_items(lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), items)
    log(f"  P1 FINAL: next_tok={full.get('next_tok_acc',0):.3f} entity={full.get('entity_acc',0):.3f} ood={full.get('ood_acc',0):.3f}")
    save_phase("p1", {"train": tr, "exam": full, "params_m": n_par / 1e6})


def phase2(smoke, device):
    if pj("p2").exists():
        log("P2 done, skip")
        return
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
    items_mid = [it for it in items if it["type"] == "next_tok"][:N_MID]

    conf = GPT2Config(vocab_size=V, n_positions=MAX_ARCS, n_embd=D_XL, n_layer=L_XL, n_head=8, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1)
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(conf).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"P2 gpt-XL params={n_par/1e6:.1f}M")

    @torch.no_grad()
    def gpt_span(ctx, cand):
        seq = (ctx + cand)[-MAX_ARCS:]
        n_ctx = len(seq) - len(cand)
        x = torch.tensor([seq], device=device)
        logp = F.log_softmax(model(input_ids=x).logits[0], dim=-1)
        return sum(float(logp[n_ctx + k - 1, tid]) for k, tid in enumerate(cand)) / max(1, len(cand))

    steps = 120 if smoke else STEPS["p2"]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    t0 = time.time()
    best, best_step, flat_evals, running = -1.0, 0, 0, None
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, rng, pad_id).to(device)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(loss) if running is None else 0.95 * running + 0.05 * float(loss)
        if step % (40 if smoke else EVAL_EVERY) == 0 or step == steps:
            model.eval()
            mid = score_items(gpt_span, items_mid, "next_tok")
            acc = mid.get("next_tok_acc", 0)
            el = time.time() - t0
            log(f"  [p2_gpt] step {step}/{steps}: ce~{running:.3f} next_tok(mid)={acc:.3f} ({el:.0f}s)")
            if acc > best + 1e-6:
                best, best_step, flat_evals = acc, step, 0
                torch.save({"model": model.state_dict(), "conf": conf.to_dict(), "step": step}, CKPT / "stage191_p2_gpt.pt")
            else:
                flat_evals += 1
            model.train()
            if el > BUDGET["p2"] or (flat_evals >= 2 and step >= steps // 2):
                log("  [p2_gpt] stop")
                break
    ck = torch.load(CKPT / "stage191_p2_gpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    full = score_items(gpt_span, items)
    log(f"  P2 FINAL: next_tok={full.get('next_tok_acc',0):.3f} entity={full.get('entity_acc',0):.3f} ood={full.get('ood_acc',0):.3f}")
    save_phase("p2", {"best_mid": best, "best_step": best_step, "exam": full, "params_m": n_par / 1e6})


def build_rarity(tok: Tokenizer, V: int, flat, device) -> torch.Tensor:
    """z-scored char-trigram novelty per token id."""
    # trigram counts from a sample of decoded corpus pieces
    from collections import Counter

    cnt: Counter = Counter()
    sample = flat[: min(len(flat), 3_000_000)]
    pieces = [tok.decode([int(t)], skip_special_tokens=False) or "" for t in np.unique(sample)]
    # counts weighted by frequency
    freq = np.bincount(sample, minlength=V)
    for tid in np.unique(sample):
        p = tok.decode([int(tid)], skip_special_tokens=False) or ""
        w = int(freq[tid])
        for i in range(len(p) - 2):
            cnt[p[i : i + 3]] += w
    total = sum(cnt.values()) or 1
    novelty = np.zeros(V, dtype=np.float32)
    for tid in range(V):
        p = tok.decode([int(tid)], skip_special_tokens=False) or ""
        tris = [p[i : i + 3] for i in range(len(p) - 2)]
        if not tris:
            novelty[tid] = 0.0
            continue
        novelty[tid] = float(np.mean([-np.log((cnt.get(t, 0) + 1) / (total + 1)) for t in tris]))
    m, s = novelty.mean(), novelty.std() + 1e-6
    return torch.tensor((novelty - m) / s, device=device)


def phase3(smoke, device):
    if pj("p3").exists():
        log("P3 done, skip")
        return
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
    items_mid = [it for it in items if it["type"] == "next_tok"][:N_MID]

    log("P3: building rarity table …")
    rarity = build_rarity(tok, V, flat, device)
    torch.manual_seed(SEED)
    model = SelfModelXL(n_char, V, rarity=rarity, surprise_temp=True).to(device)
    tr = train_curve("p3_rarity", model, flat, off, char_table, pad_id, items_mid, device, STEPS["p3"], BUDGET["p3"], smoke)
    full = score_items(lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), items)

    # G3 battery: entropy + surprise on real vs fake spans
    ent_items = [it for it in items if it["type"] == "entity"][:80]
    rngf = random.Random(3)
    e_real, e_fake, s_real, s_fake = [], [], [], []

    @torch.no_grad()
    def probe(ctx, span):
        seq = (ctx + span)[-MAX_ARCS:]
        n_ctx = len(seq) - len(span)
        x = torch.tensor([seq], dtype=torch.long, device=device)
        pad = x == pad_id
        logits, surprise, _ = model.forward_all(char_table[x], pad, ids=x)
        p = F.softmax(logits[0, len(seq) - 1], dim=-1)
        return float(-(p * torch.log(p + 1e-9)).sum()), float(surprise[0, n_ctx:].mean())

    for it in ent_items:
        gold_ids = it["cand_ids"][it["gold_idx"]]
        fake = FAKES[rngf.randint(0, len(FAKES) - 1)]
        fake_ids = [i for i in tok.encode(" " + fake).ids if i != pad_id]
        er, sr = probe(it["ctx_ids"], gold_ids)
        ef, sf = probe(it["ctx_ids"], fake_ids)
        e_real.append(er)
        e_fake.append(ef)
        s_real.append(sr)
        s_fake.append(sf)
    g3 = {
        "entropy_real": float(np.mean(e_real)),
        "entropy_fake": float(np.mean(e_fake)),
        "surprise_real": float(np.mean(s_real)),
        "surprise_fake": float(np.mean(s_fake)),
        "entropy_ok": float(np.mean(e_fake)) > float(np.mean(e_real)),
        "surprise_ok": float(np.mean(s_fake)) > float(np.mean(s_real)),
    }
    log(f"  P3 FINAL: next_tok={full.get('next_tok_acc',0):.3f} | G3 {g3}")
    save_phase("p3", {"train": tr, "exam": full, "g3": g3})


def phase4(smoke, device):
    if pj("p4").exists():
        log("P4 done, skip")
        return
    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    def curve_z(model):
        @torch.no_grad()
        def z_of_ids(ids_list):
            x = torch.tensor([ids_list[-MAX_ARCS:]], dtype=torch.long, device=device)
            pad = x == pad_id
            arcs = model._arcs(char_table[x], x)
            slow, _, _ = model.slow(arcs, pad)
            ln = int((~pad).sum())
            return slow[0, ln - 1]

        return z_of_ids

    @torch.no_grad()
    def gate_B(z_of_ids):
        def z_text(t):
            return z_of_ids([i for i in tok.encode(t).ids if i != pad_id])

        cos = lambda a, b: float(F.cosine_similarity(a, b, dim=-1))
        para = [cos(z_text(a), z_text(b)) for a, b in s179.PARAPHRASE_PAIRS]
        hard = [cos(z_text(a), z_text(b)) for a, b in s179.HARD_PAIRS]
        return {"para": float(np.mean(para)), "hard": float(np.mean(hard)), "gap": float(np.mean(hard) - np.mean(para))}

    @torch.no_grad()
    def doclink(z_of_ids, n=80):
        rng = random.Random(7)
        n_docs = len(off) - 1
        ok = 0
        for _ in range(n):
            d1, d2 = rng.randint(0, n_docs - 1), rng.randint(0, n_docs - 1)
            s1, e1 = off[d1], off[d1 + 1]
            s2, e2 = off[d2], off[d2 + 1]
            if e1 - s1 < MAX_ARCS + 16 or e2 - s2 < MAX_ARCS:
                continue
            half = (s1 + e1) // 2
            a = flat[s1 : min(s1 + MAX_ARCS, half)].tolist()
            b = flat[half : half + MAX_ARCS].tolist()
            c = flat[s2 : s2 + MAX_ARCS].tolist()
            za, zb, zc = z_of_ids(a), z_of_ids(b), z_of_ids(c)
            ok += int(float(F.cosine_similarity(za, zb, dim=-1)) > float(F.cosine_similarity(za, zc, dim=-1)))
        return ok / max(1, n)

    out = {}
    for tag, path, kwargs in (
        ("p1_curve", CKPT / "stage191_p1_curve.pt", {}),
        ("p3_rarity", CKPT / "stage191_p3_rarity.pt", {"surprise_temp": True, "need_rarity": True}),
    ):
        if not path.exists():
            continue
        rarity = build_rarity(tok, V, flat, device) if kwargs.get("need_rarity") else None
        m = SelfModelXL(n_char, V, rarity=rarity, surprise_temp=kwargs.get("surprise_temp", False)).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
        m.eval()
        z = curve_z(m)
        out[tag] = {"gateB": gate_B(z), "doclink": doclink(z)}
        log(f"  P4 {tag}: {out[tag]}")

    # GPT XL
    gpath = CKPT / "stage191_p2_gpt.pt"
    if gpath.exists():
        ck = torch.load(gpath, map_location=device, weights_only=False)
        gm = GPT2LMHeadModel(GPT2Config(**ck["conf"])).to(device)
        gm.load_state_dict(ck["model"])
        gm.eval()

        @torch.no_grad()
        def gpt_z(ids_list):
            x = torch.tensor([ids_list[-MAX_ARCS:]], device=device)
            h = gm.transformer(input_ids=x).last_hidden_state[0]
            return h.mean(dim=0)

        out["p2_gpt"] = {"gateB": gate_B(gpt_z), "doclink": doclink(gpt_z)}
        log(f"  P4 p2_gpt: {out['p2_gpt']}")

    # old 187 (d128, 2M-char training) for scale comparison
    p187 = CKPT / "stage187_self_model.pt"
    if p187.exists() and not smoke:
        import _stage170_curve_dynamics as s170

        text20 = s170.load_corpus(max_chars=20_000_000)
        chars_old = sorted(set(text20) | {" "})
        stoi_old = {c: i + 1 for i, c in enumerate(chars_old)}
        table_old = s185.build_char_table(tok, stoi_old, pad_id, V).to(device)
        mo = s187.SelfModel(len(chars_old) + 1, V).to(device)
        mo.load_state_dict(torch.load(p187, map_location=device, weights_only=False)["model"])
        mo.eval()

        @torch.no_grad()
        def z_old(ids_list):
            x = torch.tensor([ids_list[-MAX_ARCS:]], dtype=torch.long, device=device)
            pad = x == pad_id
            arcs = mo.arc_enc(table_old[x])
            slow, _, _ = mo.slow(arcs, pad)
            ln = int((~pad).sum())
            return slow[0, ln - 1]

        out["old_187_d128_2M"] = {"gateB": gate_B(z_old), "doclink": doclink(z_old)}
        log(f"  P4 old_187: {out['old_187_d128_2M']}")

    save_phase("p4", out)


def phase5(smoke):
    p = {k: json.loads(pj(k).read_text(encoding="utf-8")) for k in ("p0", "p1", "p2", "p3", "p4") if pj(k).exists()}
    verdicts = []
    if "p1" in p and "p2" in p:
        c, g = p["p1"]["exam"].get("next_tok_acc", 0), p["p2"]["exam"].get("next_tok_acc", 0)
        d = c - g
        verdicts.append("NIGHT_PARITY_HELD" if abs(d) <= 0.03 else ("NIGHT_CURVE_AHEAD" if d > 0 else "NIGHT_GPT_AHEAD"))
    if "p3" in p:
        g3 = p["p3"]["g3"]
        if g3["entropy_ok"] and g3["surprise_ok"]:
            verdicts.append("NIGHT_G3_FIXED")
        elif g3["surprise_ok"]:
            verdicts.append("NIGHT_G3_SURPRISE_ONLY")
    if "p4" in p and "old_187_d128_2M" in p["p4"]:
        gaps = {k: v["gateB"]["gap"] for k, v in p["p4"].items() if isinstance(v, dict) and "gateB" in v}
        old_gap = gaps.get("old_187_d128_2M")
        if old_gap is not None and any(v < old_gap - 0.01 for k, v in gaps.items() if k != "old_187_d128_2M"):
            verdicts.append("NIGHT_MEANING_MOVES")
    lines = [f"# Stage191 night report ({datetime.now(timezone.utc).isoformat()})", "", f"**Verdicts:** {', '.join(verdicts) or 'incomplete'}", ""]
    for k, v in p.items():
        lines.append(f"## {k}")
        lines.append("```json")
        lines.append(json.dumps(v, indent=2, ensure_ascii=False)[:3000])
        lines.append("```")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    save_phase("p5", {"verdicts": verdicts})
    log(f"[191] NIGHT DONE: {', '.join(verdicts) or 'incomplete'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    log(f"Stage191 start {datetime.now(timezone.utc).isoformat()} phase={args.phase} smoke={args.smoke}")
    phases = ["p0", "p1", "p2", "p3", "p4", "p5"] if args.phase == "all" else [args.phase]
    for ph in phases:
        if ph == "p0":
            phase0(args.smoke)
        elif ph == "p1":
            phase1(args.smoke, device)
        elif ph == "p2":
            phase2(args.smoke, device)
        elif ph == "p3":
            phase3(args.smoke, device)
        elif ph == "p4":
            phase4(args.smoke, device)
        elif ph == "p5":
            phase5(args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
