"""
Stage 211 — beyond-window: internal slow tape vs endpoint (pre-publish frontier).

Cross-doc: fact written while reading document A; query on document B (subject S never in B).
Compare retrieval of latest fake value F among 4 candidates.

Methods:
  internal_tape  — surprise-gated (key=ctx_fp local, value=slow_t) logged during read(A)
  endpoint_only  — slow_T after read(A) only (no addressable slots)
  external_slots — explicit fp slots from A (194/198 style, reference ceiling)
  gpt_incontext  — GPT sees only B tail + cue (beyond-window structural fail)
  doc_id_oracle  — key = doc embedding (metadata control; breaks under wrong id + noise)

Gates: see results/pre_publish_frontier.md §211

  python _stage211_internal_tape.py
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank, build_memory
from _stage196_tapelm import load_gpt
from _stage204_noise_robustness import noisy

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage211_decision.json"
MINI = RES / "stage211_mini.md"
LOG = RES / "_stage211_log.txt"

SEED = 211
CORPUS_CHARS = 150_000_000
MID_START = 70_000_000
MID_CHARS = 4_000_000
N_TASKS = 100
MAX_SLOTS = 48
SUR_MIN = 0.22
NOISE_P = 0.30
CHANCE = 0.25


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def encode_ids(tok: Tokenizer, pad_id: int, text: str) -> list[int]:
    return [i for i in tok.encode(text).ids if i != pad_id][-MAX_ARCS:]


@torch.no_grad()
def forward_slow(model, char_table, ids: list[int], device):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pad = x == pad_id_global
    arcs = model._arcs(char_table[x], x)
    fast = model.fast(arcs, pad_mask=pad)
    slow, surprise, _ = model.slow(arcs, pad)
    return slow[0], surprise[0], fast[0]


pad_id_global = 0


@torch.no_grad()
def build_internal_tape(model, char_table, bank: FpBank, text: str, device):
    ids = encode_ids(tok_global, pad_id_global, text)
    if len(ids) < 8:
        return None, None, None
    slow, surp, _ = forward_slow(model, char_table, ids, device)
    T = slow.size(0)
    idxs = torch.topk(surp, k=min(MAX_SLOTS, T)).indices
    keys, vals = [], []
    for t in idxs.tolist():
        if float(surp[t]) < SUR_MIN:
            continue
        lo = int((t / max(1, T - 1)) * max(1, len(text) - CTX_WIN))
        hi = min(len(text), lo + CTX_WIN)
        chunk = text[lo:hi]
        k = bank.ctx_fp(chunk)
        if k is None:
            continue
        keys.append(k)
        vals.append(F.normalize(slow[t], dim=-1))
    if not keys:
        return slow[-1], None, None
    return slow[-1], torch.stack(keys), torch.stack(vals)


@torch.no_grad()
def retrieve_soft(K: torch.Tensor, V: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    w = torch.softmax((K @ q) * 10.0, dim=0)
    v = (w.unsqueeze(0) @ V).squeeze(0)
    return F.normalize(v, dim=-1)


@torch.no_grad()
def score_candidates(vec: torch.Tensor, bank: FpBank, cands: list[str]) -> list[float]:
    fps = bank.fp(cands)
    return [float(vec @ fps[i]) for i in range(len(cands))]


def four_way_acc(scores: list[float], gold: int) -> bool:
    return int(np.argmax(scores)) == gold


def build_tasks(paras: list[str], fakes: list[str], rng: random.Random):
    tasks = []
    fi = 0
    pool = [p for p in paras if 120 < len(p) < 900]
    rng.shuffle(pool)
    for i, p in enumerate(pool):
        if len(tasks) >= N_TASKS or fi >= len(fakes):
            break
        m = ENT_RE.search(p)
        if not m:
            continue
        S = m.group(1)
        if len(S) < 4:
            continue
        Fv = fakes[fi]
        fi += 1
        lo, hi = max(0, m.start() - CTX_WIN), min(len(p), m.end() + CTX_WIN)
        text_a = p[lo:hi].replace(S, S + " " + Fv, 1)
        text_b = None
        for _ in range(40):
            pb = pool[rng.randint(0, len(pool) - 1)]
            if S not in pb:
                text_b = pb[: min(600, len(pb))]
                break
        if text_b is None:
            continue
        tasks.append({"S": S, "value": Fv, "text_a": text_a, "text_b": text_b, "doc_id": i})
    return tasks


def eval_internal(tasks, bank, model, char_table, device, rng, noise_p: float = 0.0):
    ok = 0
    allv = list({t["value"] for t in tasks})
    nr = random.Random(SEED + 99)
    for t in tasks:
        ta = t["text_a"]
        if noise_p > 0:
            ta = "".join(noisy(c, noise_p, nr) if c.isalpha() else c for c in ta)
        ep, K, V = build_internal_tape(model, char_table, bank, ta, device)
        q = bank.fp([t["S"]])[0]
        if K is not None and V is not None:
            vec = retrieve_soft(K, V, q)
        else:
            vec = F.normalize(ep + q, dim=-1)
        others = [x for x in allv if x != t["value"]]
        rng.shuffle(others)
        cands = [t["value"]] + others[:3]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        gold = order.index(0)
        sc = score_candidates(vec, bank, shuf)
        ok += four_way_acc(sc, gold)
    return ok / max(1, len(tasks))


def eval_endpoint(tasks, bank, model, char_table, device, rng, noise_p: float = 0.0):
    ok = 0
    allv = list({t["value"] for t in tasks})
    nr = random.Random(SEED + 100)
    for t in tasks:
        ta = t["text_a"]
        if noise_p > 0:
            ta = "".join(noisy(c, noise_p, nr) if c.isalpha() else c for c in ta)
        ep, _, _ = build_internal_tape(model, char_table, bank, ta, device)
        q = bank.fp([t["S"]])[0]
        vec = F.normalize(ep + q, dim=-1)
        others = [x for x in allv if x != t["value"]]
        rng.shuffle(others)
        cands = [t["value"]] + others[:3]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        gold = order.index(0)
        sc = score_candidates(vec, bank, shuf)
        ok += four_way_acc(sc, gold)
    return ok / max(1, len(tasks))


def eval_external(tasks, bank, rng, noise_p: float = 0.0):
    ok = 0
    allv = list({t["value"] for t in tasks})
    nr = random.Random(SEED + 101)
    for t in tasks:
        ta = t["text_a"]
        if noise_p > 0:
            ta = "".join(noisy(c, noise_p, nr) if c.isalpha() else c for c in ta)
        K, vals = build_memory([ta], bank, "211")
        q = bank.fp([t["S"]])[0]
        sc_map = {}
        for i, v in enumerate(vals):
            s = float(K[i] @ q)
            sc_map[v] = max(sc_map.get(v, -9.0), s)
        others = [x for x in allv if x != t["value"]]
        rng.shuffle(others)
        cands = [t["value"]] + others[:3]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        gold = order.index(0)
        sc = [sc_map.get(c, -9.0) for c in shuf]
        ok += four_way_acc(sc, gold)
    return ok / max(1, len(tasks))


def eval_doc_id_oracle(tasks, bank, doc_vecs: dict[int, torch.Tensor], rng, wrong_id: bool):
    """Global doc-id keyed store; query uses doc embedding (wrong id on noisy test)."""
    keys, vals = [], []
    for t in tasks:
        keys.append(doc_vecs[t["doc_id"]])
        vals.append(bank.fp([t["value"]])[0])
    K = torch.stack(keys)
    V = torch.stack(vals)
    ok = 0
    allv = list({t["value"] for t in tasks})
    for t in tasks:
        qid = t["doc_id"] + (999 if wrong_id else 0)
        q = doc_vecs.get(qid, doc_vecs[t["doc_id"]])
        vec = retrieve_soft(K, V, q)
        others = [x for x in allv if x != t["value"]]
        rng.shuffle(others)
        cands = [t["value"]] + others[:3]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        gold = order.index(0)
        sc = score_candidates(vec, bank, shuf)
        ok += four_way_acc(sc, gold)
    return ok / max(1, len(tasks))


@torch.no_grad()
def eval_gpt_ic(tasks, gm, tok, pad_id, device, rng):
    ok = 0
    allv = list({t["value"] for t in tasks})
    for t in tasks:
        ids_b = encode_ids(tok, pad_id, t["text_b"])
        cue = [i for i in tok.encode(" " + t["S"] + " is").ids if i != pad_id]
        ctx = (ids_b + cue)[-MAX_ARCS:]
        others = [x for x in allv if x != t["value"]]
        rng.shuffle(others)
        cands = [t["value"]] + others[:3]
        order = list(range(4))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        gold = order.index(0)
        sc = []
        for c in shuf:
            cid = [i for i in tok.encode(" " + c).ids if i != pad_id]
            seq = (ctx + cid)[-MAX_ARCS:]
            nc = len(seq) - len(cid)
            x = torch.tensor([seq], device=device)
            logp = F.log_softmax(gm(input_ids=x).logits[0], dim=-1)
            sc.append(sum(float(logp[nc + k - 1, tid]) for k, tid in enumerate(cid)) / max(1, len(cid)))
        ok += four_way_acc(sc, gold)
    return ok / max(1, len(tasks))


tok_global = None


def main() -> int:
    global pad_id_global, tok_global
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage211 start {datetime.now(timezone.utc).isoformat()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    tok_global = tok
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    pad_id_global = pad_id
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    enc_sig = sum(float(p.abs().sum()) for p in model.arc_enc.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    gm = load_gpt(device)
    bank = FpBank(model, stoi, device)
    log(f"P1 frozen ({time.time()-t0:.0f}s)")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    mid = text[MID_START : MID_START + MID_CHARS]
    paras = [p.strip() for p in mid.split("\n") if 120 < len(p.strip()) < 1000]
    words = re.findall(r"[A-Za-z][a-z]+", text)
    del text
    fakes = gen_fakes(set(words), rng, N_TASKS + 20)
    tasks = build_tasks(paras, fakes, rng)
    log(f"tasks={len(tasks)} ({time.time()-t0:.0f}s)")

    doc_vecs = {}
    for t in tasks:
        if t["doc_id"] not in doc_vecs:
            doc_vecs[t["doc_id"]] = F.normalize(torch.randn(model.head.in_features // 2, device=device), dim=-1)
    doc_vecs[t["doc_id"] + 999] = F.normalize(torch.randn(model.head.in_features // 2, device=device), dim=-1)

    ev = random.Random(SEED + 1)
    internal = eval_internal(tasks, bank, model, char_table, device, ev, 0.0)
    endpoint = eval_endpoint(tasks, bank, model, char_table, device, random.Random(SEED + 2), 0.0)
    external = eval_external(tasks, bank, random.Random(SEED + 3), 0.0)
    gpt_ic = eval_gpt_ic(tasks, gm, tok, pad_id, device, random.Random(SEED + 4))
    log(f"clean internal={internal:.3f} endpoint={endpoint:.3f} external={external:.3f} gpt_ic={gpt_ic:.3f}")

    internal_n = eval_internal(tasks, bank, model, char_table, device, random.Random(SEED + 5), NOISE_P)
    oracle_n = eval_doc_id_oracle(tasks, bank, doc_vecs, random.Random(SEED + 6), wrong_id=True)
    log(f"noisy p={NOISE_P} internal={internal_n:.3f} doc_id_wrong={oracle_n:.3f}")

    enc_sig2 = sum(float(p.abs().sum()) for p in model.arc_enc.parameters())
    anticf = abs(enc_sig - enc_sig2) < 1e-3

    g1 = internal - endpoint >= 0.25
    g2 = internal >= external - 0.10
    g3 = gpt_ic <= 0.35
    g4 = internal_n >= oracle_n - 0.05
    g5 = anticf

    if g1 and g2 and g3 and g4 and g5:
        overall = "THESIS_YES"
    elif g1 and g2 and g5:
        overall = "ENGINEERING_ONLY"
    else:
        overall = "THESIS_NO"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "internal_slow_tape_211",
        "overall": overall,
        "clean": {
            "internal_tape": internal,
            "endpoint_only": endpoint,
            "external_slots": external,
            "gpt_incontext": gpt_ic,
        },
        "noisy": {"p": NOISE_P, "internal_tape": internal_n, "doc_id_oracle_wrong_id": oracle_n},
        "gates": {"g1_beats_endpoint": g1, "g2_near_external": g2, "g3_beyond_window": g3, "g4_not_metadata": g4, "g5_anticf": g5},
        "n_tasks": len(tasks),
        "chance": CHANCE,
        "anticf_encoder_frozen": anticf,
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage211 — internal slow tape vs endpoint (cross-doc)",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- clean: internal **{internal:.3f}** endpoint {endpoint:.3f} external {external:.3f} gpt_ic {gpt_ic:.3f}",
                f"- noisy: internal **{internal_n:.3f}** doc_id_wrong {oracle_n:.3f} (p={NOISE_P})",
                f"- gates: {out['gates']}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[211] {overall} | internal={internal:.3f} endpoint={endpoint:.3f} ext={external:.3f} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
