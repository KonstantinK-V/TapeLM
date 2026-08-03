"""
Stage 252 — Joint CE + lambda*CPC on the upper trunk (fix for 251).

251 read: CPC alone moved `fast` while `slow`/`head` kept the old representation
(head sees cat([fast, slow])), so holdout PPL blew up 72 -> 245. Fix = train both
objectives on the same steps so downstream re-aligns under the drift.

Arms: lambda in {0.0, 0.05, 0.2}; lambda=0 is the CAL control (pure CE) measured
in-run, so gates compare against it instead of a number from a previous stage.

Per arm:
  loss = CE(next token) + W_SELF*pred_loss + lambda * CPC(consequence prediction)
  CPC negatives: in-batch + one hard negative per anchor from the SAME document
  budget counted in CE tokens; probes on held-out docs; keep best-by-holdout-CE snapshot

Metrics: exam next_tok, holdout CE/PPL, 179 para/hard/gap, uniformity (collapse check),
slot recall, parametric leak.

  python _stage252_joint_cpc.py [--smoke] [--token-budget N] [--lambdas 0,0.05,0.2]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
import _stage251_cpc_understand as s251
from _stage191_night import MICRO, PAD, SelfModelXL, W_SELF, load_data
from _stage194_fp_fact_memory import FpBank

RES = Path("results")
DECISION = RES / "stage252_decision.json"
MINI = RES / "stage252_mini.md"
LOG = RES / "_stage252_log.txt"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
SEED = 252

CPC_BATCH = 64
CPC_L = s251.CPC_L
CPC_TEMP = s251.CPC_TEMP
JOINT_LR = 2e-4
N_PROBES = 8
STOP_PATIENCE = 2
STOP_MARGIN = 0.15


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def cpc_draw_hard(flat, off, docs, rng: random.Random, pad_id: int, batch: int, Lw: int):
    """anchor xa, continuation xb, and a hard negative xn from the same document."""
    xa = np.full((batch, Lw), pad_id, np.int64)
    xb = np.full((batch, Lw), pad_id, np.int64)
    xn = np.full((batch, Lw), pad_id, np.int64)
    long_docs = [d for d in docs if (off[d + 1] - off[d]) >= 4 * Lw]
    pool = long_docs or [d for d in docs if (off[d + 1] - off[d]) >= 2 * Lw] or docs
    for b in range(batch):
        d = pool[rng.randint(0, len(pool) - 1)]
        s, e = off[d], off[d + 1]
        span = e - s
        st = s + rng.randint(0, max(0, span - 2 * Lw))
        xa[b] = flat[st : st + Lw]
        xb[b] = flat[st + Lw : st + 2 * Lw]
        # hard negative: another window in the same doc, disjoint from anchor+positive
        placed = False
        if span >= 4 * Lw:
            for _ in range(6):
                sn = s + rng.randint(0, span - Lw)
                if sn + Lw <= st or sn >= st + 2 * Lw:
                    xn[b] = flat[sn : sn + Lw]
                    placed = True
                    break
        if not placed:
            d2 = pool[rng.randint(0, len(pool) - 1)]
            s2, e2 = off[d2], off[d2 + 1]
            sn = s2 + rng.randint(0, max(0, (e2 - s2) - Lw))
            xn[b] = flat[sn : sn + Lw]
    return torch.from_numpy(xa), torch.from_numpy(xb), torch.from_numpy(xn)


def make_hold_batches(flat, off, hold_docs, pad_id, n_batches: int, seed: int) -> list[torch.Tensor]:
    """One fixed held-out window set reused by every probe and every arm."""
    rng = random.Random(seed)
    return [s251.sample_windows_docs(flat, off, MICRO, rng, pad_id, hold_docs) for _ in range(n_batches)]


@torch.no_grad()
def fixed_hold_ce(model, batches: list[torch.Tensor], char_table, pad_id, device) -> float:
    tot, n = 0.0, 0
    for b in batches:
        ids = b.to(device)
        pad = ids == pad_id
        logits, _, _ = model.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        if valid.any():
            tot += float(F.cross_entropy(logits[:, :-1][valid], target[valid]))
            n += 1
    return tot / max(1, n)


@torch.no_grad()
def uniformity(model, flat, off, char_table, pad_id, device, docs, n: int, seed: int) -> float:
    """Mean pairwise cosine over random windows; high value = representation collapse."""
    rng = random.Random(seed)
    ids = s251.sample_windows_docs(flat, off, n, rng, pad_id, docs).to(device)
    z = F.normalize(s251.pooled_fast_train(model, char_table, ids, pad_id), dim=-1)
    sim = z @ z.T
    off_diag = sim[~torch.eye(n, dtype=torch.bool, device=device)]
    return float(off_diag.mean())


def train_joint(
    base,
    flat,
    off,
    char_table,
    pad_id,
    device,
    token_budget,
    lam: float,
    seed: int,
    tag: str,
    train_docs,
    hold_batches,
    items_probe,
    early_stop: bool = True,
    n_probes: int = N_PROBES,
):
    m = copy.deepcopy(base)
    s213.set_train_mode(m, "upper")
    d = m.head.in_features // 2
    pred = s251.Predictor(d).to(device)
    params = [p for p in m.parameters() if p.requires_grad]
    opt_params = params + (list(pred.parameters()) if lam > 0 else [])
    opt = torch.optim.AdamW(opt_params, lr=JOINT_LR, weight_decay=0.01)
    rng = random.Random(seed)
    crng = random.Random(seed + 7)

    probe_every = max(1, token_budget // n_probes)
    next_probe = probe_every
    tokens_ce = 0
    tokens_cpc = 0
    step = 0
    curve = []
    best = {"hold_ce": float("inf"), "sd": None, "tokens": 0}
    bad = 0
    stopped = None
    last = {"ce": 0.0, "cpc": 0.0}

    while tokens_ce < token_budget:
        ids = s251.sample_windows_docs(flat, off, MICRO, rng, pad_id, train_docs).to(device)
        tokens_ce += s251.count_valid_tokens(ids, pad_id)
        pad = ids == pad_id
        logits, _, pred_loss = m.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        cpc_val = float("nan")

        if lam > 0:
            xa, xb, xn = cpc_draw_hard(flat, off, train_docs, crng, pad_id, CPC_BATCH, CPC_L)
            tokens_cpc += s251.count_valid_tokens(xa, pad_id) + s251.count_valid_tokens(xb, pad_id)
            xa, xb, xn = xa.to(device), xb.to(device), xn.to(device)
            za = F.normalize(s251.pooled_fast_train(m, char_table, xa, pad_id), dim=-1)
            zb = F.normalize(s251.pooled_fast_train(m, char_table, xb, pad_id), dim=-1)
            zn = F.normalize(s251.pooled_fast_train(m, char_table, xn, pad_id), dim=-1)
            pa = pred(za)
            cand = torch.cat([pred(zb), pred(zn)], dim=0)  # B positives + B hard negatives
            cpc_logits = pa @ cand.T / CPC_TEMP
            labels = torch.arange(xa.size(0), device=device)
            cpc = F.cross_entropy(cpc_logits, labels)
            loss = loss + lam * cpc
            cpc_val = float(cpc)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        opt.step()
        step += 1
        last = {"ce": float(ce), "cpc": cpc_val}

        if tokens_ce >= next_probe:
            next_probe += probe_every
            m.eval()
            hold = fixed_hold_ce(m, hold_batches, char_table, pad_id, device)
            nt = s251.next_tok_acc(m, char_table, pad_id, items_probe, device)
            curve.append({"tokens_ce": tokens_ce, "hold_ce": hold, "next_tok": nt, "ce": last["ce"], "cpc": last["cpc"]})
            log(
                f"  {tag} tok={tokens_ce}/{token_budget} ce={last['ce']:.3f} "
                f"cpc={last['cpc']:.3f} hold={hold:.3f} nt={nt:.3f}"
            )
            if hold < best["hold_ce"]:
                best = {
                    "hold_ce": hold,
                    "sd": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()},
                    "tokens": tokens_ce,
                }
                bad = 0
            else:
                bad += 1
                if early_stop and bad >= STOP_PATIENCE and hold > best["hold_ce"] + STOP_MARGIN:
                    stopped = f"holdout_ce_rose_at_{tokens_ce}"
                    log(f"  {tag} early stop: hold {hold:.3f} > best {best['hold_ce']:.3f} + {STOP_MARGIN}")
                    s213.set_train_mode(m, "upper")
                    break
            s213.set_train_mode(m, "upper")

    m.eval()
    final_hold = fixed_hold_ce(m, hold_batches, char_table, pad_id, device)
    if best["sd"] is not None and best["hold_ce"] < final_hold - 1e-6:
        m.load_state_dict({k: v.to(device) for k, v in best["sd"].items()})
        restored = True
    else:
        restored = False
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    meta = {
        "tokens_ce": tokens_ce,
        "tokens_cpc": tokens_cpc,
        "steps": step,
        "curve": curve,
        "early_stop": stopped,
        "restored_best": restored,
        "best_hold_ce": best["hold_ce"],
        "best_at_tokens": best["tokens"],
        "final_hold_ce_before_restore": final_hold,
    }
    return m, meta


def evaluate(
    m, char_table, pad_id, tok, stoi, device, flat, off, hold_docs, hold_batches, items, facts, all_values, K, Vlist
):
    bank = FpBank(m, stoi, device)
    ce = fixed_hold_ce(m, hold_batches, char_table, pad_id, device)
    return {
        "next_tok": s251.next_tok_acc(m, char_table, pad_id, items, device),
        "hold_ce": ce,
        "hold_ppl": math.exp(min(ce, 20)),
        "inversion": s251.inversion_fast(m, char_table, pad_id, tok, device),
        "uniformity": uniformity(m, flat, off, char_table, pad_id, device, hold_docs, 48, SEED + 32),
        "slot_mem": L.tape_recall(facts, all_values, bank, K, Vlist, SEED),
        "param_leak": s251.curve_param_recall(m, char_table, pad_id, tok, facts, all_values, device, SEED + 33),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--token-budget", type=int, default=0)
    ap.add_argument("--lambdas", type=str, default="0,0.05,0.2")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    tb = args.token_budget or (120_000 if args.smoke else 4_000_000)
    lams = [float(x) for x in args.lambdas.split(",") if x.strip() != ""]
    n_facts = 8 if args.smoke else 20
    n_exam = 40 if args.smoke else 120
    n_probe = 24 if args.smoke else 60
    n_hold = 8 if args.smoke else 24

    log(f"Stage252 start {datetime.now(timezone.utc).isoformat()} budget={tb} lambdas={lams}")
    flat, off, stoi, n_char = load_data()
    train_docs, hold_docs = s251.split_train_hold(off)
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model0 = SelfModelXL(n_char, V).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model0.eval()
    for p in model0.parameters():
        p.requires_grad_(False)
    log(f"corpus docs train={len(train_docs)} hold={len(hold_docs)}")

    facts, all_values = s251.make_facts(rng, n_facts, args.smoke)
    bank0 = FpBank(model0, stoi, device)
    K, Vlist = L.write_tape_bank(bank0, facts)
    items = s251.load_exam_next(n_exam)
    items_probe = items[:n_probe]
    hold_batches = make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)
    log(f"fixed holdout set: {n_hold} batches x {MICRO} windows")

    base_eval = evaluate(
        model0, char_table, pad_id, tok, stoi, device, flat, off, hold_docs, hold_batches,
        items, facts, all_values, K, Vlist,
    )
    log(
        f"baseline nt={base_eval['next_tok']:.3f} hold_ce={base_eval['hold_ce']:.3f} "
        f"gap={base_eval['inversion']['gap_hard_minus_para']:+.3f} unif={base_eval['uniformity']:.3f} "
        f"mem={base_eval['slot_mem']:.3f} leak={base_eval['param_leak']:.3f}"
    )

    arms = {}
    for lam in lams:
        tag = f"lam{lam:g}"
        log(f"arm {tag}: joint CE + {lam:g}*CPC")
        m, meta = train_joint(
            model0, flat, off, char_table, pad_id, device, tb, lam, SEED + int(lam * 1000) + 1,
            tag, train_docs, hold_batches, items_probe,
        )
        ev = evaluate(
            m, char_table, pad_id, tok, stoi, device, flat, off, hold_docs, hold_batches,
            items, facts, all_values, K, Vlist,
        )
        arms[tag] = {"lambda": lam, **meta, **ev}
        log(
            f"  {tag} DONE nt={ev['next_tok']:.3f} hold_ce={ev['hold_ce']:.3f} "
            f"gap={ev['inversion']['gap_hard_minus_para']:+.3f} unif={ev['uniformity']:.3f} "
            f"mem={ev['slot_mem']:.3f} leak={ev['param_leak']:.3f} ({time.time()-t0:.0f}s)"
        )
        if lam == 0.0:
            Path("checkpoints").mkdir(exist_ok=True)
            if not args.smoke:
                torch.save({"model": m.state_dict(), "stage": 252, "lambda": 0.0}, "checkpoints/stage252_ce_upper.pt")

    ctrl = arms.get("lam0")
    ref_nt = ctrl["next_tok"] if ctrl else base_eval["next_tok"]
    ref_ce = ctrl["hold_ce"] if ctrl else base_eval["hold_ce"]
    base_gap = base_eval["inversion"]["gap_hard_minus_para"]
    base_unif = base_eval["uniformity"]

    scored = []
    for tag, a in arms.items():
        if a["lambda"] == 0.0:
            continue
        g_lang = a["next_tok"] >= ref_nt - 0.01 and a["hold_ce"] <= ref_ce + 0.05
        g_mean = a["inversion"]["gap_hard_minus_para"] <= base_gap - 0.02 or a["inversion"]["inversion"]
        g_collapse = a["uniformity"] <= base_unif + 0.10
        g_mem = a["slot_mem"] >= 0.75 and a["param_leak"] <= 0.40
        a["gates"] = {
            "G_language_kept": g_lang,
            "G_meaning_gain": g_mean,
            "G_no_collapse": g_collapse,
            "G_memory_clean": g_mem,
        }
        scored.append((tag, g_lang and g_mean and g_collapse and g_mem, g_lang, g_mean, g_collapse))

    winners = [t for t, ok, *_ in scored if ok]
    any_lang = any(gl for _, _, gl, _, _ in scored)
    any_mean = any(gm for _, _, _, gm, _ in scored)
    ctrl_improves = bool(ctrl and ctrl["next_tok"] >= base_eval["next_tok"] + 0.01)

    if winners:
        overall = "JOINT_CPC_OK"
    elif any_lang and any_mean:
        overall = "JOINT_CPC_PARTIAL"
    elif ctrl_improves:
        overall = "JOINT_CPC_NO_CE_ONLY_WINS"
    else:
        overall = "JOINT_CPC_NO"

    if winners:
        fork = "SCALE_JOINT_TOKENS"
    elif any_lang and not any_mean:
        fork = "TRY_PAWS_202_SUPERVISION"
    elif any_mean and not any_lang:
        fork = "LOWER_LAMBDA_OR_LR"
    else:
        fork = "SCALE_CE_ONLY_16M"

    out = {
        "stage": 252,
        "overall": overall,
        "fork_next": fork,
        "token_budget_per_arm": tb,
        "joint_lr": JOINT_LR,
        "cpc": {"batch": CPC_BATCH, "L": CPC_L, "temp": CPC_TEMP, "hard_negatives": "same-document window"},
        "winners": winners,
        "reference": {"control_arm": "lam0", "ref_next_tok": ref_nt, "ref_hold_ce": ref_ce},
        "baseline": base_eval,
        "arms": arms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
        "note": "251 fix: CPC alone drifted `fast` while slow/head stayed; joint CE re-aligns downstream. "
        "lambda=0 arm is the in-run CAL control; gates compare against it.",
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# Stage 252 joint CE + CPC", "", f"**{overall}** fork={fork} budget={tb} tok/arm", ""]
    lines.append(f"- baseline: nt={base_eval['next_tok']:.3f} hold_ce={base_eval['hold_ce']:.3f} gap={base_gap:+.3f} unif={base_unif:.3f}")
    for tag, a in arms.items():
        lines.append(
            f"- {tag} (λ={a['lambda']:g}): nt={a['next_tok']:.3f} hold_ce={a['hold_ce']:.3f} "
            f"gap={a['inversion']['gap_hard_minus_para']:+.3f} unif={a['uniformity']:.3f} "
            f"mem={a['slot_mem']:.3f} leak={a['param_leak']:.3f}"
            + (f" [early stop: {a['early_stop']}]" if a.get("early_stop") else "")
        )
    MINI.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(json.dumps({"overall": overall, "fork": fork, "winners": winners}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
