"""
Stage 237 — Overnight mixed-domain L1 step (not a 191 replacement).

From frozen P1: continue CE training on interleaved prose+code for several hours,
then measure next_tok mid + post-hoc code-shift W recall vs a prose-only continue control.

Does **not** overwrite `stage191_p1_curve.pt`. Writes `checkpoints/stage237_mixed_l1.pt`.

  python _stage237_mixed_l1_night.py [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
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
    load_data,
    lr_at,
    sample_windows,
    score_items,
    span_logprob_x,
)
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tapelm_ext import DomainAdapter, mean_core_cos

# remove unused s186 import if present

RES = Path("results")
CKPT_IN = Path("checkpoints/stage191_p1_curve.pt")
CKPT_OUT = Path("checkpoints/stage237_mixed_l1.pt")
CKPT_PROSE = Path("checkpoints/stage237_prose_continue.pt")
DECISION = RES / "stage237_decision.json"
MINI = RES / "stage237_mini.md"
LOG = RES / "_stage237_log.txt"
STORIES = Path("data/external_tinystories_100k_85.txt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 237


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
    pl = STORIES.read_text(encoding="utf-8", errors="ignore").splitlines()[:max_lines]
    code = s227.ensure_code(random.Random(SEED + 1), smoke)
    cl = code.splitlines()[:max_lines]
    mix: list[str] = []
    for i in range(max(len(pl), len(cl))):
        if i < len(pl) and pl[i].strip():
            mix.append(pl[i])
        if i < len(cl) and cl[i].strip():
            mix.append(cl[i])
    return s213.build_flat_from_text("\n".join(mix), tok, pad_id, max_lines=max_lines * 2, min_line_len=20)


def continue_train(model, flat, off, char_table, pad_id, device, steps: int, seed: int, tag: str):
    m = copy.deepcopy(model)
    m.train()
    params = list(m.parameters())
    opt = torch.optim.AdamW(params, lr=LR * 0.5)
    r2 = random.Random(seed)
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        ids = sample_windows(flat, off, MICRO, r2, pad_id).to(device)
        pad = ids == pad_id
        logits, _, pred_loss = m.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + 0.1 * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        running = 0.95 * running + 0.05 * float(ce.detach()) if step > 1 else float(ce.detach())
        if step % max(1, EVAL_EVERY // 2 if steps > 500 else 40) == 0 or step == steps:
            log(f"  [{tag}] step {step}/{steps} ce~{running:.3f} ({time.time() - t0:.0f}s)")
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 120 if args.smoke else 6000
    max_lines = 400 if args.smoke else 12000
    core_n = 60 if args.smoke else 400
    n_facts = 10 if args.smoke else 50
    arc_probe = 40 if args.smoke else 400
    w_steps = 60 if args.smoke else 800
    rng = random.Random(SEED)
    log(f"Stage237 start {datetime.now(timezone.utc).isoformat()} device={device} steps={steps}")

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size()).to(device)

    model0 = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    blob = torch.load(CKPT_IN, map_location=device, weights_only=False)
    model0.load_state_dict(blob["model"])
    model0.eval()

    flat_s, off_s = s213.build_flat_from_text(
        STORIES.read_text(encoding="utf-8", errors="ignore"), tok, pad_id, max_lines=max_lines
    )
    flat_m, off_m = build_mixed_flat(tok, pad_id, max_lines, args.smoke)

    log("continue prose …")
    model_p = continue_train(model0, flat_s, off_s, char_table, pad_id, device, steps, SEED + 2, "prose")
    log("continue mixed …")
    model_m = continue_train(model0, flat_m, off_m, char_table, pad_id, device, steps, SEED + 3, "mixed")

    if not args.smoke:
        CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model_m.state_dict(), "stage": 237, "steps": steps}, CKPT_OUT)
        torch.save({"model": model_p.state_dict(), "stage": 237, "tag": "prose_continue", "steps": steps}, CKPT_PROSE)

    items = []
    if EXAM_V3.exists():
        with EXAM_V3.open(encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                if it.get("type") == "next_tok":
                    items.append(it)
                if len(items) >= (40 if args.smoke else 120):
                    break

    def next_tok_acc(model) -> float:
        if not items:
            return float("nan")
        mid = score_items(lambda c, cd: span_logprob_x(model, char_table, pad_id, c, cd, device), items, "next_tok")
        return float(mid.get("next_tok_acc", 0.0))

    acc_p = next_tok_acc(model_p)
    acc_m = next_tok_acc(model_m)
    log(f"next_tok mid prose={acc_p:.3f} mixed={acc_m:.3f}")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(2_000_000)
    core = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z][a-z]{2,}", text) if len(w) <= 14))[:core_n]
    bank0 = FpBank(model0, stoi, device)
    bank_p = FpBank(model_p, stoi, device)
    bank_m = FpBank(model_m, stoi, device)
    F0 = s221.fp_matrix(bank0, core)

    flat_c, off_c = s213.build_flat_from_text(
        s227.ensure_code(random.Random(SEED + 7), args.smoke), tok, pad_id, max_lines=max_lines, min_line_len=20
    )
    # Code-shift from each continue checkpoint (probe domain drift after continue)
    model_p_code = s221.finetune_arc_enc(model_p, flat_c, off_c, char_table, pad_id, device, arc_probe, SEED + 11)
    model_m_code = s221.finetune_arc_enc(model_m, flat_c, off_c, char_table, pad_id, device, arc_probe, SEED + 12)
    bank_pc = FpBank(model_p_code, stoi, device)
    bank_mc = FpBank(model_m_code, stoi, device)

    cos_p = mean_core_cos(bank0, bank_pc, core)
    cos_m = mean_core_cos(bank0, bank_mc, core)

    W_p, _ = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_pc, core), F0, rng, w_steps, device)
    W_m, _ = s221.train_remap(DomainAdapter(256).to(device), s221.fp_matrix(bank_mc, core), F0, rng, w_steps, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wiki_words = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(f.read(2_000_000)) if len(m.group(1)) >= 5))
    subs = gen_fakes(set(wiki_words), rng, n_facts + 10)[:n_facts]
    vals = wiki_words[:n_facts]
    K0, V = s221.build_fact_bank(bank0, subs, vals, rng)
    acc_wp = s227.recall(K0, V, bank_pc, subs, vals, rng, query_x=s227.w_apply(W_p))
    acc_wm = s227.recall(K0, V, bank_mc, subs, vals, rng, query_x=s227.w_apply(W_m))

    g_nt = (not (acc_m != acc_m)) and (acc_m + 1e-9 >= acc_p - 0.03)  # nan-safe
    g_cos = cos_m >= cos_p + 0.02
    g_w = acc_wm >= acc_wp - 0.05
    overall = (
        "MIXED_L1_NIGHT_OK"
        if g_nt and (g_cos or g_w) and acc_wm >= 0.70
        else ("MIXED_L1_NIGHT_PARTIAL" if g_nt or g_w else "MIXED_L1_NIGHT_NO")
    )

    out = {
        "stage": 237,
        "overall": overall,
        "gates": {
            "G_mixed_next_tok_not_worse": bool(g_nt),
            "G_mixed_code_shift_closer_to_can": bool(g_cos),
            "G_mixed_W_recall_not_worse": bool(g_w),
        },
        "steps": steps,
        "next_tok_prose_continue": acc_p,
        "next_tok_mixed_continue": acc_m,
        "mean_cos_can_after_code_shift_prose": cos_p,
        "mean_cos_can_after_code_shift_mixed": cos_m,
        "recall_W_after_code_prose": acc_wp,
        "recall_W_after_code_mixed": acc_wm,
        "ckpt": str(CKPT_OUT) if CKPT_OUT.exists() else None,
        "note": "Does not replace stage191_p1_curve.pt; overnight scale step toward multi-domain L1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 237 mixed L1 night\n\n**{overall}** nt_m={acc_m:.3f} W_m={acc_wm:.3f} cos_m={cos_m:.3f}\n",
        encoding="utf-8",
    )
    log(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
