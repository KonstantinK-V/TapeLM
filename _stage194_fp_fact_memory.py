"""
Stage 194 — FP fact memory: entity recall via episodic fingerprint slots.

North-star attack: entity cloze is at chance for ALL systems (LM path = weight
memorization = needs billions of params). FP path (old SOTE SoftPhraseMemory,
hop1 only): while READING text, write slots
    key = normalize(mean fp(context words around entity)),  value = entity
At exam: query = normalize(mean fp(question context words));
score(candidate) = max cos(query, key) over slots whose value == candidate.

Zero training; fp = frozen 191-P1 arc encoder (as 192/193).

Gates:
  G1 acc >= 0.50 on entity items (chance 0.25; night models 0.27-0.30)
  G2 falsification control: memory built WITHOUT the read tail → acc <= 0.35
     (proves answers come from reading, not priors)

  python _stage194_fp_fact_memory.py
"""
from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data

RES = Path("results")
DATA = Path("data")
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage194_decision.json"
MINI = RES / "stage194_mini.md"
LOG = RES / "_stage194_log.txt"
EXAM_V3 = DATA / "stage191_exam_v3.jsonl"

SEED = 194
MAX_CHARS = s177.MAX_CHARS_PER_ARC
CORPUS_CHARS = 150_000_000
EXAM_TAIL_CHARS = 3_000_000
from _tape_index import CTX_WIN, context_words
ENT_RE = re.compile(r"\b([A-Z][a-z]{3,}|\d{3,4})\b")
WORD_RE = re.compile(r"[A-Za-z][a-z]{2,}")


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class FpBank:
    def __init__(self, model, stoi, device):
        self.model = model
        self.stoi = stoi
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def fp(self, ws: list[str]) -> torch.Tensor:
        todo = [w for w in ws if w not in self.cache]
        if todo:
            rows = torch.zeros(len(todo), 1, MAX_CHARS, dtype=torch.long)
            for i, w in enumerate(todo):
                for j, c in enumerate(w[:MAX_CHARS]):
                    rows[i, 0, j] = self.stoi.get(c, 0)
            out = F.normalize(self.model.arc_enc(rows.to(self.device))[:, 0], dim=-1)
            for w, v in zip(todo, out):
                self.cache[w] = v
        return torch.stack([self.cache[w] for w in ws], 0)

    @torch.no_grad()
    def ctx_fp(self, text: str, exclude: str | None = None) -> torch.Tensor | None:
        from _tape_index import CONTEXT_WORD_MIN, context_words

        ws = context_words(text, exclude=exclude)
        if len(ws) < CONTEXT_WORD_MIN:
            return None
        return F.normalize(self.fp(ws).mean(0), dim=-1)


def build_memory(paras: list[str], bank: FpBank, tag: str) -> tuple[torch.Tensor, list[str]]:
    keys, vals = [], []
    t0 = time.time()
    for p in paras:
        for m in ENT_RE.finditer(p):
            ent = m.group(1)
            lo, hi = max(0, m.start() - CTX_WIN), min(len(p), m.end() + CTX_WIN)
            k = bank.ctx_fp(p[lo:hi], exclude=ent)
            if k is not None:
                keys.append(k)
                vals.append(ent)
    K = torch.stack(keys, 0) if keys else torch.zeros(0, 256, device=bank.device)
    log(f"  memory[{tag}]: slots={len(vals)} ({time.time()-t0:.0f}s)")
    return K, vals


def score_entity_items(items, tok, pad_id, bank: FpBank, K: torch.Tensor, vals: list[str]) -> dict:
    by_ent: dict[str, list[int]] = {}
    for i, v in enumerate(vals):
        by_ent.setdefault(v, []).append(i)
    ok, n, abstain = 0, 0, 0
    for it in items:
        if it["type"] != "entity":
            continue
        ctx_text = tok.decode(it["ctx_ids"], skip_special_tokens=False)
        q = bank.ctx_fp(ctx_text)
        if q is None:
            continue
        sims = (K @ q) if len(vals) else torch.zeros(0)
        scores = []
        for cand_ids in it["cand_ids"]:
            cand = tok.decode(cand_ids, skip_special_tokens=False).strip()
            cand_w = ENT_RE.search(cand)
            cand = cand_w.group(1) if cand_w else cand
            idxs = by_ent.get(cand, [])
            scores.append(float(sims[idxs].max()) if idxs else -1.0)
        if max(scores) <= -1.0:
            abstain += 1
            continue
        ok += int(int(np.argmax(scores)) == it["gold_idx"])
        n += 1
    return {"acc": ok / max(1, n), "n": n, "abstain": abstain}


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage194 start {datetime.now(timezone.utc).isoformat()}")
    log("FP fact memory (episodic slots, zero training) vs entity cloze")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model.eval()
    bank = FpBank(model, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    tail = text[-EXAM_TAIL_CHARS:]
    tail_paras = [p.strip() for p in tail.split("\n") if 120 < len(p.strip()) < 1000][:1200]
    # control region: same-size slice from the middle of the corpus (never in exam)
    mid = text[60_000_000 : 60_000_000 + EXAM_TAIL_CHARS]
    ctrl_paras = [p.strip() for p in mid.split("\n") if 120 < len(p.strip()) < 1000][:1200]
    del text
    log(f"tail paras={len(tail_paras)} ctrl paras={len(ctrl_paras)} ({time.time()-t0:.0f}s)")

    items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
    ent_n = sum(1 for it in items if it["type"] == "entity")
    log(f"entity items={ent_n}")

    K_tail, vals_tail = build_memory(tail_paras, bank, "read-tail")
    res_read = score_entity_items(items, tok, pad_id, bank, K_tail, vals_tail)
    log(f"  READ memory: acc={res_read['acc']:.3f} n={res_read['n']} abstain={res_read['abstain']}")

    K_ctrl, vals_ctrl = build_memory(ctrl_paras, bank, "control-unread")
    res_ctrl = score_entity_items(items, tok, pad_id, bank, K_ctrl, vals_ctrl)
    log(f"  CONTROL memory: acc={res_ctrl['acc']:.3f} n={res_ctrl['n']} abstain={res_ctrl['abstain']}")

    p1_entity = None
    p1j = RES / "stage191_p1.json"
    if p1j.exists():
        p1_entity = json.loads(p1j.read_text(encoding="utf-8"))["exam"].get("entity_acc")

    g1 = res_read["acc"] >= 0.50 and res_read["n"] >= 50
    g2 = res_ctrl["acc"] <= 0.35 or res_ctrl["n"] < 20
    if g1 and g2:
        overall = "FACT_MEMORY_YES"
    elif res_read["acc"] >= 0.35 and g2:
        overall = "FACT_MEMORY_WEAK"
    else:
        overall = "FACT_MEMORY_NO"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "fp_fact_memory_194",
        "overall": overall,
        "read_memory": res_read,
        "control_unread_memory": res_ctrl,
        "chance": 0.25,
        "lm_baseline_p1_entity": p1_entity,
        "slots": {"read": len(vals_tail), "control": len(vals_ctrl)},
        "note": "zero training; fp = frozen P1 arc_enc; hop1 retrieval (old SOTE SoftPhraseMemory style)",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage194 — FP fact memory (entity recall)",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- READ memory: acc={res_read['acc']:.3f} (n={res_read['n']}, abstain={res_read['abstain']}) — chance 0.25, LM baseline {p1_entity}",
                f"- CONTROL (unread) memory: acc={res_ctrl['acc']:.3f} (n={res_ctrl['n']}, abstain={res_ctrl['abstain']})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[194] {overall} | read={res_read['acc']:.3f} ctrl={res_ctrl['acc']:.3f} chance=0.25 lm={p1_entity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
