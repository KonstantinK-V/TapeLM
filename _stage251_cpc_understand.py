"""
Stage 251 — Understanding ladder: calibrate instrument, then CPC-upper on real corpus.

Plan (ingest-forks follow-up):
  0) CAL: upper CE on unmasked load_data() for a fixed *token budget* (is exam next_tok movable?)
  1) CPC: upper + frozen arc_enc; CPC on load_data windows; facts live in slots only (no binding CE)
  2) Fixed eval: exam next_tok + held-out wiki mean CE/PPL + 179 inversion + slot mem + parametric leak

Corpus: load_data() flat/offsets (multi-document), NOT single-doc join from 248/250.

  python _stage251_cpc_understand.py [--smoke] [--cal-only] [--skip-cal]
  python _stage251_cpc_understand.py --token-budget 4000000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
from _stage191_night import MICRO, MAX_ARCS, PAD, SelfModelXL, W_SELF, load_data, sample_windows, span_logprob_x
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank

RES = Path("results")
DECISION = RES / "stage251_decision.json"
MINI = RES / "stage251_mini.md"
LOG = RES / "_stage251_log.txt"
CKPT = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
EXAM = Path("data/stage191_exam_v3.jsonl")
SEED = 251

CPC_L = 32
CPC_BATCH = 64
CPC_TEMP = 0.07
CPC_LR = 3e-4
CE_LR = 3e-4
HOLD_FRAC = 0.05

FILLER_TEMPLATES = [
    "The passage continues with other institutional details.",
    "Later sections discuss unrelated regional history.",
    "The narrative shifts to broader economic context.",
    "A following paragraph covers administrative procedure.",
    "The article mentions general background without specifics.",
    "Subsequent lines treat a different topic entirely.",
]


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def strip_fact_bindings(text: str, facts, rng: random.Random) -> str:
    """Remove binding sentences; per-fact placeholders (no repeated single stub)."""
    out = text
    for i, f in enumerate(facts):
        tpl = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        out = out.replace(f["sent"], tpl)
        out = out.replace(f["S"], f"PersonAlpha{i}")
        out = out.replace(f["value"], f"RegionBeta{i}")
    return out


def split_train_hold(off, hold_frac: float = HOLD_FRAC) -> tuple[list[int], list[int]]:
    n_docs = len(off) - 1
    n_hold = max(12, int(n_docs * hold_frac))
    train = list(range(n_docs - n_hold))
    hold = list(range(n_docs - n_hold, n_docs))
    return train, hold


def sample_windows_docs(flat, off, batch, rng: random.Random, pad_id: int, doc_ids: list[int]) -> torch.Tensor:
    xs = np.full((batch, MAX_ARCS), pad_id, dtype=np.int64)
    for b in range(batch):
        d = doc_ids[rng.randint(0, len(doc_ids) - 1)]
        s, e = off[d], off[d + 1]
        ln = e - s
        if ln <= MAX_ARCS:
            xs[b, :ln] = flat[s:e]
        else:
            st = s + rng.randint(0, ln - MAX_ARCS)
            xs[b] = flat[st : st + MAX_ARCS]
    return torch.from_numpy(xs)


def count_valid_tokens(ids: torch.Tensor, pad_id: int) -> int:
    return int((ids != pad_id).sum().item())


class Predictor(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


def load_exam_next(n: int) -> list:
    items = []
    if not EXAM.exists():
        return items
    with EXAM.open(encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if it.get("type") == "next_tok":
                items.append(it)
            if len(items) >= n:
                break
    return items


def next_tok_acc(model, char_table, pad_id, items, device) -> float:
    if not items:
        return float("nan")
    ok = 0
    for it in items:
        sc = [span_logprob_x(model, char_table, pad_id, it["ctx_ids"], c, device) for c in it["cand_ids"]]
        ok += int(int(np.argmax(sc)) == it["gold_idx"])
    return ok / len(items)


def curve_param_recall(model, char_table, pad_id, tok, facts, all_values, device, seed: int) -> float:
    qrng = random.Random(seed + 11)
    ok = 0
    for f in facts:
        ctx = [
            i
            for i in tok.encode(f"In the report {f['S']} was linked to the organization of").ids
            if i != pad_id
        ]
        others = [x for x in all_values if x != f["value"]]
        qrng.shuffle(others)
        cands = [f["value"]] + others[:3]
        order = list(range(4))
        qrng.shuffle(order)
        shuf = [cands[i] for i in order]
        sc = [
            span_logprob_x(
                model, char_table, pad_id, ctx, [i for i in tok.encode(" " + c).ids if i != pad_id], device
            )
            for c in shuf
        ]
        ok += int(int(np.argmax(sc)) == order.index(0))
    return ok / max(1, len(facts))


@torch.no_grad()
def wiki_mean_ce(model, flat, off, char_table, pad_id, device, doc_ids: list[int], n_win: int, seed: int) -> float:
    rng = random.Random(seed)
    ces = []
    for _ in range(n_win):
        ids = sample_windows_docs(flat, off, 1, rng, pad_id, doc_ids).to(device)
        pad = ids == pad_id
        logits, _, _ = model.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        if valid.any():
            ces.append(float(F.cross_entropy(logits[:, :-1][valid], target[valid])))
    return float(np.mean(ces)) if ces else float("nan")


def inversion_fast(model, char_table, pad_id, tok, device) -> dict:
    @torch.no_grad()
    def z(t: str) -> torch.Tensor:
        ids = [i for i in tok.encode(t).ids if i != pad_id][:MAX_ARCS]
        x = torch.tensor([ids], device=device)
        pad = x == pad_id
        arcs = model._arcs(char_table[x], x)
        fast = model.fast(arcs, pad_mask=pad)
        m = (~pad).float().unsqueeze(-1)
        p = (fast * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(p[0], dim=-1)

    para = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.PARAPHRASE_PAIRS]))
    hard = float(np.mean([float(F.cosine_similarity(z(a), z(b), dim=-1)) for a, b in s179.HARD_PAIRS]))
    return {"para": para, "hard": hard, "gap_hard_minus_para": hard - para, "inversion": para > hard}


def train_ce_token_budget(
    base, flat, off, char_table, pad_id, device, token_budget, seed, tag, train_docs
) -> tuple[SelfModelXL, int]:
    m = copy.deepcopy(base)
    s213.set_train_mode(m, "upper")
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CE_LR, weight_decay=0.01)
    rng = random.Random(seed)
    tokens_seen = 0
    step = 0
    last_ce = 0.0
    while tokens_seen < token_budget:
        ids = sample_windows_docs(flat, off, MICRO, rng, pad_id, train_docs).to(device)
        tokens_seen += count_valid_tokens(ids, pad_id)
        pad = ids == pad_id
        logits, _, pred_loss = m.forward_all(char_table[ids], pad, ids=ids)
        target = ids[:, 1:]
        valid = (~pad[:, :-1]) & (~pad[:, 1:])
        ce = F.cross_entropy(logits[:, :-1][valid], target[valid])
        loss = ce + W_SELF * pred_loss[~pad].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        last_ce = float(ce)
        if step % max(50, token_budget // (MICRO * MAX_ARCS * 20)) == 0:
            log(f"  {tag} tokens={tokens_seen}/{token_budget} step={step} ce={last_ce:.3f}")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    log(f"  {tag} done tokens={tokens_seen} steps={step} ce={last_ce:.3f}")
    return m, tokens_seen


def pooled_fast_train(model, char_table, idb: torch.Tensor, pad_id: int) -> torch.Tensor:
    pad = idb == pad_id
    arcs = model._arcs(char_table[idb], idb)
    fast = model.fast(arcs, pad_mask=pad)
    m = (~pad).float().unsqueeze(-1)
    return (fast * m).sum(1) / m.sum(1).clamp(min=1.0)


def cpc_draw(flat, off, docs, rng: random.Random, pad_id: int, batch: int, L: int):
    xa = np.full((batch, L), pad_id, np.int64)
    xb = np.full((batch, L), pad_id, np.int64)
    eligible = [d for d in docs if (off[d + 1] - off[d]) >= 2 * L]
    if not eligible:
        eligible = docs
    for b in range(batch):
        d = eligible[rng.randint(0, len(eligible) - 1)]
        s, e = off[d], off[d + 1]
        st = s + rng.randint(0, max(0, (e - s) - 2 * L))
        xa[b] = flat[st : st + L]
        xb[b] = flat[st + L : st + 2 * L]
    return torch.from_numpy(xa), torch.from_numpy(xb)


def train_cpc_token_budget(
    base, flat, off, char_table, pad_id, device, token_budget, seed, tag, train_docs
) -> tuple[SelfModelXL, int]:
    m = copy.deepcopy(base)
    s213.set_train_mode(m, "upper")
    d = m.head.in_features // 2
    pred = Predictor(d).to(device)
    params = [p for p in m.parameters() if p.requires_grad] + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=CPC_LR, weight_decay=0.01)
    rng = random.Random(seed)
    tokens_seen = 0
    step = 0
    last_loss = 0.0
    while tokens_seen < token_budget:
        xa, xb = cpc_draw(flat, off, train_docs, rng, pad_id, CPC_BATCH, CPC_L)
        tokens_seen += count_valid_tokens(xa, pad_id) + count_valid_tokens(xb, pad_id)
        xa, xb = xa.to(device), xb.to(device)
        za = F.normalize(pooled_fast_train(m, char_table, xa, pad_id), dim=-1)
        zb = F.normalize(pooled_fast_train(m, char_table, xb, pad_id), dim=-1)
        pa = pred(za)
        pb = pred(zb)
        logits = pa @ pb.T / CPC_TEMP
        labels = torch.arange(xa.size(0), device=device)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        last_loss = float(loss)
        if step % max(40, token_budget // (CPC_BATCH * CPC_L * 30)) == 0:
            log(f"  {tag} tokens={tokens_seen}/{token_budget} step={step} cpc={last_loss:.3f}")
    m.eval()
    pred.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    log(f"  {tag} done tokens={tokens_seen} steps={step} cpc={last_loss:.3f}")
    return m, tokens_seen


def make_facts(rng: random.Random, n_facts: int, smoke: bool) -> tuple[list, list]:
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(2_000_000 if smoke else 8_000_000)
    values_pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5))
    rng.shuffle(values_pool)
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 20) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[i]
        facts.append({"S": S, "value": Vv, "sent": f"{S} was appointed director of {Vv} in 1987 .", "fid": i})
    all_values = [f["value"] for f in facts] + values_pool[n_facts : n_facts + 40]
    return facts, all_values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cal-only", action="store_true")
    ap.add_argument("--skip-cal", action="store_true")
    ap.add_argument("--token-budget", type=int, default=0, help="per phase (cal and cpc); 0 = preset")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    tb = args.token_budget or (120_000 if args.smoke else 2_000_000)
    n_facts = 8 if args.smoke else 20
    n_exam = 40 if args.smoke else 120
    n_ppl = 24 if args.smoke else 80

    log(f"Stage251 start {datetime.now(timezone.utc).isoformat()} token_budget/phase={tb} smoke={args.smoke}")
    flat, off, stoi, n_char = load_data()
    train_docs, hold_docs = split_train_hold(off)
    log(f"corpus docs train={len(train_docs)} hold={len(hold_docs)} tokens~{len(flat)}")

    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model0 = SelfModelXL(n_char, V).to(device)
    model0.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model"])
    model0.eval()
    for p in model0.parameters():
        p.requires_grad_(False)

    facts, all_values = make_facts(rng, n_facts, args.smoke)
    bank0 = FpBank(model0, stoi, device)
    K, Vlist = L.write_tape_bank(bank0, facts)
    items = load_exam_next(n_exam)

    mem0 = L.tape_recall(facts, all_values, bank0, K, Vlist, SEED)
    nt0 = next_tok_acc(model0, char_table, pad_id, items, device)
    leak0 = curve_param_recall(model0, char_table, pad_id, tok, facts, all_values, device, SEED)
    inv0 = inversion_fast(model0, char_table, pad_id, tok, device)
    ce0 = wiki_mean_ce(model0, flat, off, char_table, pad_id, device, hold_docs, n_ppl, SEED + 1)
    ppl0 = math.exp(min(ce0, 20))

    log(
        f"baseline nt={nt0:.3f} mem={mem0:.3f} leak={leak0:.3f} "
        f"inv={inv0['inversion']} gap={inv0['gap_hard_minus_para']:+.3f} hold_ce={ce0:.3f} ppl~{ppl0:.1f}"
    )

    cal_model = model0
    cal_tokens = 0
    nt_cal = nt0
    inv_cal = inv0
    ce_cal = ce0

    if not args.skip_cal:
        log("phase0 CAL: upper CE on unmasked load_data (instrument)")
        cal_model, cal_tokens = train_ce_token_budget(
            model0, flat, off, char_table, pad_id, device, tb, SEED + 10, "cal_ce", train_docs
        )
        nt_cal = next_tok_acc(cal_model, char_table, pad_id, items, device)
        inv_cal = inversion_fast(cal_model, char_table, pad_id, tok, device)
        ce_cal = wiki_mean_ce(cal_model, flat, off, char_table, pad_id, device, hold_docs, n_ppl, SEED + 11)
        log(f"  CAL nt {nt0:.3f}->{nt_cal:.3f} delta={nt_cal - nt0:+.3f} hold_ce={ce_cal:.3f}")

    cal_delta = nt_cal - nt0
    if cal_delta >= 0.015:
        cal_verdict = "CAL_MOVES_EXAM"
    elif cal_delta <= 0.005 and cal_delta >= -0.005:
        cal_verdict = "CAL_CEILING"
    elif cal_delta < -0.02:
        cal_verdict = "CAL_HURTS"
    else:
        cal_verdict = "CAL_FLAT"

    cpc_model = cal_model
    cpc_tokens = 0
    nt_cpc = nt_cal
    inv_cpc = inv_cal
    ce_cpc = ce_cal
    mem_cpc = mem0
    leak_cpc = leak0

    if not args.cal_only:
        log("phase1 CPC: upper on load_data; facts in slots only")
        cpc_model, cpc_tokens = train_cpc_token_budget(
            model0, flat, off, char_table, pad_id, device, tb, SEED + 20, "cpc_upper", train_docs
        )
        nt_cpc = next_tok_acc(cpc_model, char_table, pad_id, items, device)
        inv_cpc = inversion_fast(cpc_model, char_table, pad_id, tok, device)
        ce_cpc = wiki_mean_ce(cpc_model, flat, off, char_table, pad_id, device, hold_docs, n_ppl, SEED + 21)
        bank_c = FpBank(cpc_model, stoi, device)
        mem_cpc = L.tape_recall(facts, all_values, bank_c, K, Vlist, SEED)
        leak_cpc = curve_param_recall(cpc_model, char_table, pad_id, tok, facts, all_values, device, SEED + 22)

    # Mask hygiene spot-check (248/250 bug class): joined stream vs load_data doc count
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        snippet = f.read(500_000)
    paras = [p.strip() for p in snippet.split("\n") if len(p.strip()) > 160][: n_facts + 5]
    bad_stream = " ".join(paras[:20])
    _, off_bad = s213.build_flat_from_text(bad_stream, tok, pad_id, max_lines=500, min_line_len=16)
    mask_note = {
        "load_data_docs": len(off) - 1,
        "bad_joined_docs": len(off_bad) - 1,
        "248_250_single_doc_risk": len(off_bad) - 1 <= 2,
    }

    g_mem = mem_cpc >= 0.75
    g_leak = leak_cpc <= 0.40
    g_ppl = ce_cpc <= ce0 * 1.12 + 0.05
    g_nt_hold = nt_cpc >= nt0 - 0.03
    g_nt_gain = nt_cpc >= nt0 + 0.01
    g_inv_gain = inv_cpc["gap_hard_minus_para"] < inv0["gap_hard_minus_para"] - 0.01 or inv_cpc["inversion"]
    g_cal_instrument = cal_verdict in ("CAL_MOVES_EXAM", "CAL_CEILING")

    fork = None
    if g_inv_gain and not g_nt_gain and cal_verdict == "CAL_CEILING":
        fork = "TRY_ARC_UNFREEZE_202B"
    elif cal_verdict == "CAL_HURTS":
        fork = "ADD_KL_REPLAY_ANCHOR"
    elif g_nt_gain and g_mem and g_leak:
        fork = "SCALE_CPC_TOKENS"
    elif not g_inv_gain and not g_nt_gain:
        fork = "TRY_PAWS_202"

    if args.cal_only:
        overall = "CAL_" + cal_verdict.replace("CAL_", "") if cal_verdict.startswith("CAL_") else cal_verdict
    elif g_mem and g_leak and g_ppl and g_nt_hold and (g_nt_gain or g_inv_gain):
        overall = "CPC_UNDERSTAND_OK"
    elif g_mem and g_leak and g_nt_hold and (g_nt_gain or g_inv_gain or g_inv_gain):
        overall = "CPC_UNDERSTAND_PARTIAL"
    elif g_cal_instrument and not g_nt_gain and not g_inv_gain:
        overall = "CPC_UNDERSTAND_NO"
    else:
        overall = "CPC_UNDERSTAND_PARTIAL"

    out = {
        "stage": 251,
        "overall": overall,
        "token_budget_per_phase": tb,
        "cal_verdict": cal_verdict,
        "fork_next": fork,
        "mask_corpus_note": mask_note,
        "gates": {
            "G_mem": g_mem,
            "G_no_param_leak": g_leak,
            "G_holdout_ppl_ok": g_ppl,
            "G_exam_not_worse": g_nt_hold,
            "G_exam_gain": g_nt_gain,
            "G_inversion_gain": g_inv_gain,
            "G_cal_instrument": g_cal_instrument,
        },
        "baseline": {
            "next_tok": nt0,
            "slot_mem": mem0,
            "param_leak": leak0,
            "inversion": inv0,
            "hold_ce": ce0,
            "hold_ppl": ppl0,
        },
        "cal_ce": {
            "tokens": cal_tokens,
            "next_tok": nt_cal,
            "delta_nt": cal_delta,
            "inversion": inv_cal,
            "hold_ce": ce_cal,
        },
        "cpc_upper": {
            "tokens": cpc_tokens,
            "next_tok": nt_cpc,
            "delta_nt_vs_base": nt_cpc - nt0,
            "slot_mem": mem_cpc,
            "param_leak": leak_cpc,
            "inversion": inv_cpc,
            "hold_ce": ce_cpc,
            "hold_ppl": math.exp(min(ce_cpc, 20)),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
        "note": "Facts in slots only; train corpus = load_data(). CAL=unmasked CE; CPC=consequence prediction on upper.",
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 251 CPC understand\n\n**{overall}** cal={cal_verdict} fork={fork}\n"
        f"nt {nt0:.3f}->{nt_cal:.3f}(cal)->{nt_cpc:.3f}(cpc) mem={mem_cpc:.3f} leak={leak_cpc:.3f}\n"
        f"inv gap {inv0['gap_hard_minus_para']:+.3f}->{inv_cpc['gap_hard_minus_para']:+.3f}\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "cal_verdict": cal_verdict, "fork": fork}, indent=2))

    if not args.smoke and tb >= 500_000 and not args.cal_only:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save({"model": cpc_model.state_dict(), "stage": 251, "tokens": cpc_tokens}, "checkpoints/stage251_cpc_upper.pt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
