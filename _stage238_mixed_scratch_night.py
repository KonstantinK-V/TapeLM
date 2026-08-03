"""
Stage 238 — Mixed multi-domain L1 **from scratch** (hypothesis check).

Train two SelfModelXL from random init (matched steps):
  A) prose/wiki-only
  B) interleaved wiki+code

Then: write facts in each arm's own fp; apply code arc_enc shift; fit qmap W;
compare W-recall and next_tok.

Does **not** touch `stage191_p1_curve.pt`.
Writes: `checkpoints/stage238_prose_scratch.pt`, `checkpoints/stage238_mixed_scratch.pt`

  python _stage238_mixed_scratch_night.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage221_fp_remap_adapter as s221
import _stage227_canonical_slots as s227
from _stage191_night import (
    EVAL_EVERY,
    EXAM_V3,
    LR,
    MICRO,
    PAD,
    SelfModelXL,
    W_SELF,
    load_data,
    lr_at,
    sample_windows,
    score_items,
    span_logprob_x,
)
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, mean_core_cos

RES = Path("results")
CKPT = Path("checkpoints")
CKPT_PROSE = CKPT / "stage238_prose_scratch.pt"
CKPT_MIXED = CKPT / "stage238_mixed_scratch.pt"
DECISION = RES / "stage238_decision.json"
MINI = RES / "stage238_mini.md"
LOG = RES / "_stage238_log.txt"
WIKI = Path("data/_wikitext103_train.txt")
SEED = 238
STEPS_FULL = 10_000
BUDGET_S = 3.6 * 3600  # per arm


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def build_mixed_flat(tok, pad_id: int, max_lines: int, smoke: bool):
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_lines = [ln for ln in f.read(8_000_000 if smoke else 40_000_000).splitlines() if ln.strip()][
            :max_lines
        ]
    code = s227.ensure_code(random.Random(SEED + 1), smoke)
    cl = [ln for ln in code.splitlines() if ln.strip()][:max_lines]
    mix: list[str] = []
    for i in range(max(len(wiki_lines), len(cl))):
        if i < len(wiki_lines):
            mix.append(wiki_lines[i])
        if i < len(cl):
            mix.append(cl[i])
    return s213.build_flat_from_text("\n".join(mix), tok, pad_id, max_lines=len(mix), min_line_len=20)


def train_from_scratch(
    tag: str,
    flat,
    off,
    char_table,
    pad_id: int,
    n_char: int,
    V: int,
    items_mid: list,
    device: torch.device,
    steps: int,
    budget_s: float,
    seed: int,
) -> tuple[SelfModelXL, dict]:
    torch.manual_seed(seed)
    model = SelfModelXL(n_char, V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(seed)
    t0 = time.time()
    best, best_step, flat_evals = -1.0, 0, 0
    running = None
    model.train()
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
        if step % (40 if steps <= 200 else EVAL_EVERY) == 0 or step == steps:
            model.eval()
            mid = score_items(
                lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), items_mid, "next_tok"
            )
            acc = mid.get("next_tok_acc", 0)
            el = time.time() - t0
            log(f"  [{tag}] step {step}/{steps}: ce~{running:.3f} next_tok(mid)={acc:.3f} ({el:.0f}s)")
            if acc > best + 1e-6:
                best, best_step, flat_evals = acc, step, 0
                torch.save({"model": model.state_dict(), "step": step, "mid": acc, "tag": tag}, CKPT / f"_tmp_238_{tag}.pt")
            else:
                flat_evals += 1
            model.train()
            if el > budget_s:
                log(f"  [{tag}] budget hit")
                break
            if flat_evals >= 2 and step >= steps // 2:
                log(f"  [{tag}] early stop (flat)")
                break
    tmp = CKPT / f"_tmp_238_{tag}.pt"
    if tmp.exists():
        ck = torch.load(tmp, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()
    return model, {"best_mid": best, "best_step": best_step, "ce": running, "wall_s": time.time() - t0}


def arm_memory_exam(
    model: SelfModelXL,
    stoi,
    char_table,
    pad_id: int,
    device: torch.device,
    core: list[str],
    subs: list[str],
    vals: list[str],
    flat_c,
    off_c,
    arc_probe: int,
    w_steps: int,
    rng: random.Random,
    seed_shift: int,
) -> dict:
    bank = FpBank(model, stoi, device)
    F0 = s221.fp_matrix(bank, core)
    K, V = s221.build_fact_bank(bank, subs, vals, rng)
    model_c = s221.finetune_arc_enc(model, flat_c, off_c, char_table, pad_id, device, arc_probe, seed_shift)
    bank_c = FpBank(model_c, stoi, device)
    cos = mean_core_cos(bank, bank_c, core)
    W, align = s221.train_remap(
        DomainAdapter(256).to(device), s221.fp_matrix(bank_c, core), F0, rng, w_steps, device
    )
    acc_w = s227.recall(K, V, bank_c, subs, vals, rng, query_x=s227.w_apply(W))
    acc_raw = s227.recall(K, V, bank_c, subs, vals, rng)
    return {
        "mean_cos_after_code_shift": cos,
        "W_align": align,
        "recall_W": acc_w,
        "recall_no_W": acc_raw,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    try:
        LOG.write_text("", encoding="utf-8")
    except OSError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 120 if args.smoke else STEPS_FULL
    budget = 600 if args.smoke else BUDGET_S
    max_lines = 400 if args.smoke else 20_000
    core_n = 60 if args.smoke else 400
    n_facts = 10 if args.smoke else 50
    arc_probe = 40 if args.smoke else 400
    w_steps = 60 if args.smoke else 800
    rng = random.Random(SEED)
    log(f"Stage238 start {datetime.now(timezone.utc).isoformat()} device={device} steps/arm={steps}")

    flat_wiki, off_wiki, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    flat_mix, off_mix = build_mixed_flat(tok, pad_id, max_lines, args.smoke)

    items = []
    if EXAM_V3.exists():
        with EXAM_V3.open(encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                if it.get("type") == "next_tok":
                    items.append(it)
                if len(items) >= (40 if args.smoke else 80):
                    break

    log("train prose/wiki from scratch …")
    model_p, tr_p = train_from_scratch(
        "prose", flat_wiki, off_wiki, char_table, pad_id, n_char, V, items, device, steps, budget, SEED + 2
    )
    log("train mixed wiki+code from scratch …")
    model_m, tr_m = train_from_scratch(
        "mixed", flat_mix, off_mix, char_table, pad_id, n_char, V, items, device, steps, budget, SEED + 3
    )

    if not args.smoke:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model_p.state_dict(), "train": tr_p, "stage": 238}, CKPT_PROSE)
        torch.save({"model": model_m.state_dict(), "train": tr_m, "stage": 238}, CKPT_MIXED)

    acc_p = float(
        score_items(lambda c, cd: span_logprob_x(model_p, char_table, pad_id, c, cd, device), items, "next_tok").get(
            "next_tok_acc", 0
        )
    )
    acc_m = float(
        score_items(lambda c, cd: span_logprob_x(model_m, char_table, pad_id, c, cd, device), items, "next_tok").get(
            "next_tok_acc", 0
        )
    )
    log(f"next_tok prose={acc_p:.3f} mixed={acc_m:.3f}")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(2_000_000)
    core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:core_n]
    wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]

    flat_c, off_c = s213.build_flat_from_text(
        s227.ensure_code(random.Random(SEED + 9), args.smoke), tok, pad_id, max_lines=max_lines, min_line_len=20
    )
    log("memory exam prose arm …")
    mem_p = arm_memory_exam(
        model_p, stoi, char_table, pad_id, device, core, subs, vals, flat_c, off_c, arc_probe, w_steps, rng, SEED + 11
    )
    log("memory exam mixed arm …")
    mem_m = arm_memory_exam(
        model_m, stoi, char_table, pad_id, device, core, subs, vals, flat_c, off_c, arc_probe, w_steps, rng, SEED + 12
    )
    log(f"mem prose={json.dumps(mem_p)} mixed={json.dumps(mem_m)}")

    g_nt = acc_m >= acc_p - 0.03
    g_w = mem_m["recall_W"] >= mem_p["recall_W"] + 0.12
    g_floor = mem_m["recall_W"] >= 0.55
    g_cos = mem_m["mean_cos_after_code_shift"] >= mem_p["mean_cos_after_code_shift"] + 0.05
    overall = (
        "MIXED_SCRATCH_OK"
        if g_nt and g_w and g_floor
        else (
            "MIXED_SCRATCH_PARTIAL"
            if g_nt and (g_w or g_cos or mem_m["recall_W"] >= mem_p["recall_W"])
            else "MIXED_SCRATCH_NO"
        )
    )

    out = {
        "stage": 238,
        "overall": overall,
        "gates": {
            "G_mixed_next_tok_not_worse": g_nt,
            "G_mixed_W_beats_prose_by_0p12": g_w,
            "G_mixed_W_floor_0p55": g_floor,
            "G_mixed_cos_more_stable": g_cos,
        },
        "steps_per_arm": steps,
        "train_prose": tr_p,
        "train_mixed": tr_m,
        "next_tok_prose": acc_p,
        "next_tok_mixed": acc_m,
        "memory_prose": mem_p,
        "memory_mixed": mem_m,
        "margin_W_mixed_minus_prose": mem_m["recall_W"] - mem_p["recall_W"],
        "hypothesis": "mixed-from-scratch improves post-code-shift W recall vs prose-from-scratch",
        "note": "Does not replace stage191_p1_curve.pt",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 238 mixed scratch\n\n**{overall}** nt_m={acc_m:.3f} W_m={mem_m['recall_W']:.3f} "
        f"W_p={mem_p['recall_W']:.3f} Δ={mem_m['recall_W'] - mem_p['recall_W']:+.3f}\n",
        encoding="utf-8",
    )
    log(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
