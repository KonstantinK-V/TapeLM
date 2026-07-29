"""
Stage 183 — Fast dataset-answer exam (not LM ceiling).

Build small cloze / doc-link / OOD packs from wiki hold.
Score frozen backbones by embedding similarity (no long probe train):
  - cloze: pick candidate that best matches context state (cos)
  - doc-link: same-doc pairs should be closer than cross-doc
  - OOD: cloze with answers never in train — should stay near chance

Systems: ce_gpt_181, dual_180, hybrid_182 (if ckpt).

Speed: N small, encode once, vectorized scoring.

  python _stage183_dataset_answer_exam.py
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
LOG = RES / "_stage183_log.txt"
EXAM = DATA / "stage183_exam.jsonl"
DECISION = RES / "stage183_decision.json"
MINI = RES / "stage183_mini.md"
PLAN = RES / "plan_stage183_dataset_answer.md"
TOK_PATH = s177.TOK_PATH

SEED = 183
MAX_LEN = 48  # shorter = faster
N_CLOZE = 40
N_DOC = 24
N_OOD = 16
PAD = "[PAD]"
EXAM_CHARS = 800_000  # enough for exam; charset from ckpt


def load_text_fast(max_chars: int) -> str:
    """Avoid read_text() of full multi‑100MB wiki."""
    path = s170.WIKI
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read(max_chars)


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


ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
NUM_RE = re.compile(r"\b(\d{3,4})\b")


def split_paras(text: str) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    return paras


def pick_answer(sent: str) -> tuple[str, str] | None:
    """Return (context_with_mask, answer) or None."""
    for rx in (ENTITY_RE, NUM_RE):
        ms = list(rx.finditer(sent))
        if not ms:
            continue
        # prefer mid-sentence rarer-ish: longest match
        m = max(ms, key=lambda x: len(x.group(1)))
        ans = m.group(1)
        if len(ans) < 3:
            continue
        ctx = sent[: m.start()] + " [MASK] " + sent[m.end() :]
        if len(ctx) < 20:
            continue
        return ctx.strip(), ans
    return None


def build_exam(text: str, rng: random.Random) -> list[dict]:
    paras = [p.strip() for p in text.split("\n\n") if 100 < len(p.strip()) < 800][:500]
    if len(paras) < 40:
        paras = [text[i : i + 300] for i in range(0, min(len(text), 120_000), 300)]
    n = len(paras)
    hold = paras[int(0.8 * n) :]
    train = paras[: int(0.8 * n)]
    train_blob = "\n".join(train[:100])[:50_000].lower()
    log(f"  paras={n} hold={len(hold)}")

    cloze_raw, answers = [], []
    for p in hold:
        sent = p.replace("\n", " ")[:200]
        # cheap answer: first Capitalized token length>=4
        m = re.search(r"\b([A-Z][a-z]{3,})\b", sent)
        if not m:
            m = re.search(r"\b(\d{3,4})\b", sent)
        if not m:
            continue
        ans = m.group(1)
        ctx = (sent[: m.start()] + " [MASK] " + sent[m.end() :]).strip()
        cloze_raw.append({"context": ctx, "gold": ans})
        answers.append(ans)
        if len(cloze_raw) >= N_CLOZE * 2:
            break
    rng.shuffle(cloze_raw)
    log(f"  cloze_raw={len(cloze_raw)}")

    items = []
    n_in = 0
    for row in cloze_raw:
        if n_in >= N_CLOZE:
            break
        gold = row["gold"]
        distractors = []
        for _ in range(20):
            d = answers[rng.randint(0, len(answers) - 1)]
            if d != gold and d not in distractors:
                distractors.append(d)
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            continue
        cands = [gold] + distractors
        rng.shuffle(cands)
        items.append({"type": "cloze", "context": row["context"], "gold": gold, "candidates": cands, "ood": False})
        n_in += 1

    fake_pool = [f for f in ["Zorblax", "Quenith", "Marbune", "Xaldera", "9191", "Kessari", "Vornak", "Talmidex"] if f.lower() not in train_blob]
    for i in range(min(N_OOD, len(cloze_raw), max(0, len(fake_pool) - 3))):
        gold = fake_pool[i]
        distractors = fake_pool[i + 1 : i + 4]
        cands = [gold] + distractors
        rng.shuffle(cands)
        items.append({"type": "cloze", "context": cloze_raw[-(i + 1)]["context"], "gold": gold, "candidates": cands, "ood": True})

    for _ in range(N_DOC):
        i = rng.randint(0, len(hold) - 1)
        p = hold[i]
        a, b = p[: len(p) // 3], p[len(p) // 2 : 2 * len(p) // 3]
        if len(a) < 30 or len(b) < 30:
            continue
        items.append({"type": "doc_link", "text_a": a, "text_b": b, "same": True})
        j = (i + 1 + rng.randint(0, max(1, len(hold) - 2))) % len(hold)
        items.append({"type": "doc_link", "text_a": a, "text_b": hold[j][: len(a)], "same": False})
    return items


# ---------- encoders ----------


@torch.no_grad()
def enc_gpt(model, tok, text: str, device, pad_id: int) -> torch.Tensor:
    ids = [i for i in tok.encode(text).ids if i != pad_id][-MAX_LEN:] or [1]
    x = torch.tensor([ids], device=device)
    h = model.transformer(input_ids=x).last_hidden_state[0]
    return F.normalize(torch.cat([h[-1], h.mean(0)], 0), dim=0)


@torch.no_grad()
def enc_dual(model_backbone, tok, stoi, text: str, device, mode: str = "slow") -> torch.Tensor:
    pcs = s177.encode_pieces(tok, text)[-MAX_LEN:] or ["."]
    char_ids = s177.pieces_to_char_ids(pcs, stoi).unsqueeze(0).to(device)
    pad = torch.zeros(1, len(pcs), dtype=torch.bool, device=device)
    _, fast, slow = model_backbone.forward_channels(char_ids, pad)
    h = slow[0] if mode == "slow" else (0.5 * fast[0] + 0.5 * slow[0])
    return F.normalize(torch.cat([h[-1], h.mean(0)], 0), dim=0)


def cos(a, b) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def score_cloze(encode_fn, items: list[dict], cache: dict | None = None) -> dict:
    cache = cache if cache is not None else {}

    def enc(t: str):
        if t not in cache:
            cache[t] = encode_fn(t)
        return cache[t]

    in_ok = in_n = ood_ok = ood_n = 0
    cloze_items = [it for it in items if it["type"] == "cloze"]
    for i, it in enumerate(cloze_items):
        if i % 20 == 0:
            log(f"    cloze {i}/{len(cloze_items)}")
        ctx = enc(it["context"])
        scores = []
        for c in it["candidates"]:
            filled = it["context"].replace("[MASK]", c)
            scores.append(cos(ctx, enc(filled)))
        pred = it["candidates"][int(np.argmax(scores))]
        if it.get("ood"):
            ood_n += 1
            ood_ok += int(pred == it["gold"])
        else:
            in_n += 1
            in_ok += int(pred == it["gold"])
    return {
        "cloze_in_acc": in_ok / max(1, in_n),
        "cloze_in_n": in_n,
        "cloze_ood_acc": ood_ok / max(1, ood_n),
        "cloze_ood_n": ood_n,
        "chance": 0.25,
    }


def score_doclink(encode_fn, items: list[dict], cache: dict | None = None) -> dict:
    cache = cache if cache is not None else {}

    def enc(t: str):
        if t not in cache:
            cache[t] = encode_fn(t)
        return cache[t]

    same, diff = [], []
    for it in items:
        if it["type"] != "doc_link":
            continue
        s = cos(enc(it["text_a"]), enc(it["text_b"]))
        (same if it["same"] else diff).append(s)
    if not same or not diff:
        return {"doc_acc": 0.0, "doc_n": 0, "gap_same_minus_diff": 0.0}
    thr = 0.5 * (float(np.mean(same)) + float(np.mean(diff)))
    correct = sum(1 for s in same if s >= thr) + sum(1 for s in diff if s < thr)
    n = len(same) + len(diff)
    return {
        "doc_acc": correct / n,
        "doc_n": n,
        "mean_same": float(np.mean(same)),
        "mean_diff": float(np.mean(diff)),
        "gap_same_minus_diff": float(np.mean(same) - np.mean(diff)),
    }


def verdict_for(scores: dict) -> str:
    cin = scores["cloze_in_acc"]
    ood = scores["cloze_ood_acc"]
    chance = scores["chance"]
    if cin >= chance + 0.10 and ood <= chance + 0.05:
        return "DATASET_ANSWER_SIGNAL"
    if cin >= chance + 0.10 and ood > chance + 0.10:
        return "LEAK_OR_PRIOR"
    if scores.get("doc_acc", 0) >= 0.65 and scores.get("gap_same_minus_diff", 0) > 0.05:
        return "DOC_BINDING_ONLY"
    return "NO_DATASET_ANSWER"


def load_gpt(device):
    path = CKPT / "stage181_ce_control.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    conf = GPT2Config(**ck["conf"])
    model = GPT2LMHeadModel(conf).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def load_dual180(device):
    path = CKPT / "stage180_dual_channel.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    stoi = ck.get("stoi")
    if not stoi:
        raise RuntimeError("stage180 ckpt missing stoi")
    n_char = max(stoi.values()) + 1
    model = s180.DualChannel(n_char).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, stoi


def load_hybrid182(device, stoi):
    path = CKPT / "stage182_slow_ce_tape.pt"
    if not path.exists():
        return None
    n_char = max(stoi.values()) + 1
    tok = Tokenizer.from_file(str(TOK_PATH))
    V = tok.get_vocab_size()
    model = s182.DualSlowCE(n_char, V).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rebuild-exam", action="store_true")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage183 start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan={PLAN} | fast embedding exam (no long probe)")

    rng = random.Random(SEED)
    log("loading wiki slice …")
    text = load_text_fast(EXAM_CHARS)
    log(f"text_chars={len(text)}")
    if args.rebuild_exam or not EXAM.exists():
        log("building exam …")
        items = build_exam(text, rng)
        with EXAM.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        log(f"exam built → {EXAM} n={len(items)}")
    else:
        items = [json.loads(l) for l in EXAM.read_text(encoding="utf-8").splitlines() if l.strip()]
        log(f"exam loaded n={len(items)}")

    n_cloze = sum(1 for i in items if i["type"] == "cloze" and not i.get("ood"))
    n_ood = sum(1 for i in items if i["type"] == "cloze" and i.get("ood"))
    n_doc = sum(1 for i in items if i["type"] == "doc_link")
    log(f"counts cloze_in={n_cloze} ood={n_ood} doc_link={n_doc}")

    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    device = torch.device(args.device)

    systems = {}

    log("load ce_gpt_181 …")
    gpt = load_gpt(device)
    systems["ce_gpt_181"] = lambda t: enc_gpt(gpt, tok, t, device, pad_id)

    log("load dual_180 …")
    d180, stoi = load_dual180(device)
    systems["dual_180"] = lambda t: enc_dual(d180, tok, stoi, t, device, mode="slow")

    h182 = load_hybrid182(device, stoi)
    if h182 is not None:
        log("load hybrid_182 …")
        systems["hybrid_182"] = lambda t: enc_dual(h182.backbone, tok, stoi, t, device, mode="slow")
    else:
        log("skip hybrid_182 (no ckpt)")

    def enc_rand(t: str):
        g = torch.Generator().manual_seed(hash(t) % (2**31 - 1))
        return F.normalize(torch.randn(256, generator=g), dim=0)

    systems["random_hash"] = enc_rand

    results = {}
    for name, enc in systems.items():
        log(f"score {name} …")
        cache: dict = {}
        cloze = score_cloze(enc, items, cache)
        doc = score_doclink(enc, items, cache)
        merged = {**cloze, **doc}
        merged["verdict"] = verdict_for(merged)
        results[name] = merged
        log(
            f"  {name}: cloze_in={merged['cloze_in_acc']:.3f} ood={merged['cloze_ood_acc']:.3f} "
            f"doc={merged['doc_acc']:.3f} gap={merged.get('gap_same_minus_diff', 0):.3f} → {merged['verdict']}"
        )

    # overall: any non-random SIGNAL?
    signals = [k for k, v in results.items() if k != "random_hash" and v["verdict"] == "DATASET_ANSWER_SIGNAL"]
    if signals:
        overall = "EXAM_SIGNAL_PRESENT"
    elif any(v["verdict"] == "DOC_BINDING_ONLY" for k, v in results.items() if k != "random_hash"):
        overall = "EXAM_DOC_ONLY"
    else:
        overall = "EXAM_NO_SIGNAL_YET"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "dataset_answer_exam_183_fast",
        "overall": overall,
        "exam": str(EXAM),
        "counts": {"cloze_in": n_cloze, "ood": n_ood, "doc_link": n_doc},
        "results": results,
        "note": "Win = cloze_in≫chance and OOD~chance. LM ceiling (CE/ablation) is NOT this exam.",
        "next": "If NO_SIGNAL on all: strengthen fact-oriented non-text teacher or richer exam. If only GPT signals: principle gap. If dual/hybrid signal: north star alive.",
    }
    write_json(DECISION, out)
    lines = [
        "# Stage183 — dataset-answer exam (fast)",
        "",
        f"**Overall:** `{overall}`",
        "",
    ]
    for k, v in results.items():
        lines.append(
            f"- `{k}`: cloze_in={v['cloze_in_acc']:.3f} ood={v['cloze_ood_acc']:.3f} "
            f"doc={v['doc_acc']:.3f} → **{v['verdict']}**"
        )
    lines += ["", out["next"], ""]
    MINI.write_text("\n".join(lines), encoding="utf-8")
    log(f"[183] {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
