"""
Stage 184 — Exam calibration via LOG-PROB (fix 183's broken cos scorer).

Two cloze flavors:
  1) next_tok  — next-token multiple choice. CALIBRATION: an LM must beat chance.
                 If GPT fails this, the harness lies → stop, fix harness.
  2) entity    — mask a content span (entity/number); candidates are real spans;
                 score by log-prob of the span in place (dataset-answer question).
  + OOD entity — gold span never in corpus; should stay ~chance.

Scoring = length-normalized log-prob of the candidate given context.
Systems with a vocab head: ce_gpt_181, hybrid_182. (dual_180 has no head → deferred to S2.)
random_uniform baseline = chance.

  python _stage184_exam_logprob.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage180_dual_channel as s180
import _stage182_slow_ce_tape as s182

RES = Path("results")
DATA = Path("data")
CKPT = Path("checkpoints")
LOG = RES / "_stage184_log.txt"
EXAM = DATA / "stage184_exam.jsonl"
DECISION = RES / "stage184_decision.json"
MINI = RES / "stage184_mini.md"
PLAN = RES / "plan_curve_dynamics.md"
TOK_PATH = s177.TOK_PATH

SEED = 184
CTX_TOK = 40          # context length in tokens
N_NEXT = 120          # next-token MC items (calibration)
N_ENTITY = 100        # entity cloze
N_OOD = 60
N_CAND = 4            # 4-way MC → chance 0.25
EXAM_CHARS = 800_000
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


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_text_fast(max_chars: int) -> str:
    path = s170.WIKI
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read(max_chars)


ENT_RE = re.compile(r"\b([A-Z][a-z]{3,}|\d{3,4})\b")


def build_exam(text: str, tok: Tokenizer, pad_id: int, rng: random.Random) -> list[dict]:
    # wikitext uses single newlines (no blank-line paragraphs) → split on \n, keep sizable lines
    paras = [p.strip() for p in text.split("\n") if 120 < len(p.strip()) < 1000][:1200]
    n = len(paras)
    hold = paras[int(0.8 * n) :]
    train = paras[: int(0.8 * n)]
    train_blob = " ".join(train)[:200_000].lower()
    log(f"  paras={n} hold={len(hold)}")

    # vocab of real tokens for next-tok distractors (exclude specials/pad)
    all_ids = set()
    for p in train[:300]:
        all_ids.update(tok.encode(p).ids)
    all_ids.discard(pad_id)
    id_pool = [i for i in all_ids if i != pad_id]

    items = []

    # ---- next-token MC (calibration) ----
    n_next = 0
    for p in hold:
        if n_next >= N_NEXT:
            break
        ids = [i for i in tok.encode(p).ids if i != pad_id]
        if len(ids) < CTX_TOK + 2:
            continue
        pos = rng.randint(CTX_TOK, len(ids) - 2)
        ctx_ids = ids[max(0, pos - CTX_TOK) : pos]
        gold = ids[pos]
        distractors = []
        for _ in range(30):
            d = id_pool[rng.randint(0, len(id_pool) - 1)]
            if d != gold and d not in distractors:
                distractors.append(d)
            if len(distractors) >= N_CAND - 1:
                break
        if len(distractors) < N_CAND - 1:
            continue
        cands = [[gold]] + [[d] for d in distractors]
        gi = 0
        order = list(range(len(cands)))
        rng.shuffle(order)
        cands = [cands[k] for k in order]
        gi = order.index(0)
        items.append({"type": "next_tok", "ctx_ids": ctx_ids, "cand_ids": cands, "gold_idx": gi})
        n_next += 1

    # ---- entity cloze (in-corpus) ----
    n_ent = 0
    entity_strings = []
    for p in hold:
        for m in ENT_RE.finditer(p):
            entity_strings.append(m.group(1))
    entity_strings = list(dict.fromkeys(entity_strings))
    for p in hold:
        if n_ent >= N_ENTITY:
            break
        m = ENT_RE.search(p, 60)  # not at very start → real context
        if not m:
            continue
        gold_str = m.group(1)
        ctx_text = p[: m.start()]
        ctx_ids = [i for i in tok.encode(ctx_text).ids if i != pad_id][-CTX_TOK:]
        if len(ctx_ids) < 8:
            continue
        # candidate token ids: encode " <ent>" (leading space, ByteLevel)
        gold_ids = [i for i in tok.encode(" " + gold_str).ids if i != pad_id]
        dset = []
        for _ in range(40):
            cand = entity_strings[rng.randint(0, len(entity_strings) - 1)]
            if cand != gold_str and cand not in dset:
                dset.append(cand)
            if len(dset) >= N_CAND - 1:
                break
        if len(dset) < N_CAND - 1 or not gold_ids:
            continue
        cand_ids = [gold_ids] + [[i for i in tok.encode(" " + d).ids if i != pad_id] for d in dset]
        order = list(range(len(cand_ids)))
        rng.shuffle(order)
        cand_ids = [cand_ids[k] for k in order]
        gi = order.index(0)
        items.append({"type": "entity", "ctx_ids": ctx_ids, "cand_ids": cand_ids, "gold_idx": gi})
        n_ent += 1

    # ---- OOD entity (gold never in corpus) ----
    fake_pool = [f for f in ["Zorblax", "Quenith", "Marbune", "Xaldera", "Kessari", "Vornak", "Talmidex", "Orsiphon", "Pholmar", "Girenth"] if f.lower() not in train_blob]
    n_ood = 0
    for p in hold:
        if n_ood >= N_OOD or len(fake_pool) < N_CAND:
            break
        m = ENT_RE.search(p, 60)
        if not m:
            continue
        ctx_ids = [i for i in tok.encode(p[: m.start()]).ids if i != pad_id][-CTX_TOK:]
        if len(ctx_ids) < 8:
            continue
        picks = rng.sample(fake_pool, N_CAND)
        cand_ids = [[i for i in tok.encode(" " + w).ids if i != pad_id] for w in picks]
        gi = rng.randint(0, N_CAND - 1)  # arbitrary gold; none is in corpus
        items.append({"type": "ood", "ctx_ids": ctx_ids, "cand_ids": cand_ids, "gold_idx": gi})
        n_ood += 1

    return items


# ---------- scorers ----------


@torch.no_grad()
def gpt_span_logprob(model, ctx_ids: list[int], cand_ids: list[int], device) -> float:
    seq = ctx_ids + cand_ids
    x = torch.tensor([seq], device=device)
    logits = model(input_ids=x).logits[0]  # [T,V]
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for k, tid in enumerate(cand_ids):
        pos = len(ctx_ids) + k - 1  # logits at pos predict token pos+1
        total += float(logp[pos, tid])
    return total / max(1, len(cand_ids))


@torch.no_grad()
def hybrid_span_logprob(model, tok, stoi, pad_id, ctx_ids: list[int], cand_ids: list[int], device) -> float:
    seq = ctx_ids + cand_ids
    x = torch.tensor([seq], device=device)
    pad = x == pad_id
    char_ids = s182.ids_to_char_batch(tok, x, stoi, pad_id).to(device)
    _, _, slow = model.forward_channels(char_ids, pad)
    logits = model.logits_from_slow(slow[0])  # [T,V]; logits[i] predicts id[i+1]
    logp = F.log_softmax(logits, dim=-1)
    total = 0.0
    for k, tid in enumerate(cand_ids):
        pos = len(ctx_ids) + k - 1
        total += float(logp[pos, tid])
    return total / max(1, len(cand_ids))


def score_system(name, scorer, items) -> dict:
    acc = {"next_tok": [0, 0], "entity": [0, 0], "ood": [0, 0]}
    for i, it in enumerate(items):
        if i % 50 == 0:
            log(f"    {name} {i}/{len(items)}")
        scores = [scorer(it["ctx_ids"], c) for c in it["cand_ids"]]
        pred = int(np.argmax(scores))
        t = it["type"]
        acc[t][1] += 1
        acc[t][0] += int(pred == it["gold_idx"])
    out = {}
    for t, (ok, n) in acc.items():
        out[f"{t}_acc"] = ok / max(1, n)
        out[f"{t}_n"] = n
    out["chance"] = 1.0 / N_CAND
    return out


def load_gpt(device):
    ck = torch.load(CKPT / "stage181_ce_control.pt", map_location=device, weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ck["conf"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def load_hybrid(device):
    path = CKPT / "stage182_slow_ce_tape.pt"
    if not path.exists():
        return None, None
    ck = torch.load(path, map_location=device, weights_only=False)
    stoi = ck.get("stoi")
    if not stoi:
        # 182 ckpt didn't save stoi; borrow from 180 (same charset construction)
        ck180 = torch.load(CKPT / "stage180_dual_channel.pt", map_location=device, weights_only=False)
        stoi = ck180["stoi"]
    n_char = max(stoi.values()) + 1
    V = Tokenizer.from_file(str(TOK_PATH)).get_vocab_size()
    model = s182.DualSlowCE(n_char, V).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, stoi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage184 start {datetime.now(timezone.utc).isoformat()}")
    log("Exam calibration via log-prob; GPT-beats-chance gate")

    rng = random.Random(SEED)
    device = torch.device(args.device)
    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0

    text = load_text_fast(EXAM_CHARS)
    log(f"text_chars={len(text)}")
    items = build_exam(text, tok, pad_id, rng)
    with EXAM.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    cn = sum(1 for i in items if i["type"] == "next_tok")
    ce = sum(1 for i in items if i["type"] == "entity")
    co = sum(1 for i in items if i["type"] == "ood")
    log(f"exam n={len(items)} next_tok={cn} entity={ce} ood={co}")

    results = {}

    log("load ce_gpt_181 …")
    gpt = load_gpt(device)
    results["ce_gpt_181"] = score_system("gpt", lambda c, cd: gpt_span_logprob(gpt, c, cd, device), items)

    hyb, stoi = load_hybrid(device)
    if hyb is not None:
        log("load hybrid_182 …")
        results["hybrid_182"] = score_system(
            "hybrid", lambda c, cd: hybrid_span_logprob(hyb, tok, stoi, pad_id, c, cd, device), items
        )
    else:
        log("skip hybrid_182 (no ckpt)")

    # random baseline
    rb = random.Random(0)
    results["random"] = score_system("random", lambda c, cd: rb.random(), items)

    for name, r in results.items():
        log(
            f"  {name}: next_tok={r['next_tok_acc']:.3f} entity={r['entity_acc']:.3f} "
            f"ood={r['ood_acc']:.3f} (chance={r['chance']:.2f})"
        )

    # ---- calibration gate ----
    gpt_next = results["ce_gpt_181"]["next_tok_acc"]
    chance = 1.0 / N_CAND
    calibrated = gpt_next >= chance + 0.20  # GPT must clearly beat chance on next-token
    if not calibrated:
        overall = "HARNESS_STILL_BROKEN"
    else:
        # among systems, any dataset-answer signal on entity (with OOD staying ~chance)?
        sig = []
        for name, r in results.items():
            if name == "random":
                continue
            if r["entity_acc"] >= chance + 0.10 and r["ood_acc"] <= chance + 0.10:
                sig.append(name)
        overall = "CALIBRATED_ENTITY_SIGNAL:" + ",".join(sig) if sig else "CALIBRATED_NO_ENTITY_SIGNAL"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "exam_logprob_184",
        "overall": overall,
        "calibrated": calibrated,
        "gate": "GPT next_tok >= chance+0.20",
        "chance": chance,
        "counts": {"next_tok": cn, "entity": ce, "ood": co},
        "results": results,
        "note": "next_tok = harness calibration (LM must win). entity = dataset-answer. ood must stay ~chance.",
        "next": (
            "If HARNESS_STILL_BROKEN: fix scorer/data before any curve claim. "
            "If CALIBRATED: entity_acc is now a trustworthy dataset-answer number; proceed to S2 (addressable tape)."
        ),
    }
    write_json(DECISION, out)
    lines = ["# Stage184 — log-prob exam calibration", "", f"**Overall:** `{overall}`", "", f"chance={chance:.2f}", ""]
    for name, r in results.items():
        lines.append(f"- `{name}`: next_tok={r['next_tok_acc']:.3f} entity={r['entity_acc']:.3f} ood={r['ood_acc']:.3f}")
    lines += ["", out["next"], ""]
    MINI.write_text("\n".join(lines), encoding="utf-8")
    log(f"[184] {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
