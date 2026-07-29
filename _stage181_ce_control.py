"""
Stage 181 — Matched CE-Transformer control (dataset context ceiling).

Same ByteLevel BPE (Stage177) + same wiki chunk as curve stages.
Train a small GPT-2 with standard next-token CE.
Probe the SAME gates as curve:
  A) same last piece / different prefix → hidden-state wipe?
  B) paraphrase vs hard spelling (micro-signal: para↑, gap hard-para↓)
  Ablation) CE loss with natural vs prefix-shuffled context

Question: does THIS dataset+scale support any context signal under ordinary LM training?
If CE control also flat on B-micro → don't call a curve wall.
If CE control shows A/B-micro and curve doesn't → curve objective/arch lag.

  python _stage181_ce_control.py
  python _stage181_ce_control.py --steps 10000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage179_curve_harden_B as s179

RES = Path("results")
CKPT_DIR = Path("checkpoints")
LOG = RES / "_stage181_log.txt"
DECISION = RES / "stage181_decision.json"
MINI = RES / "stage181_mini.md"
CKPT_OUT = CKPT_DIR / "stage181_ce_control.pt"
TOK_PATH = s177.TOK_PATH
PLAN = RES / "plan_curve_dynamics.md"
DEC180 = RES / "stage180_decision.json"

SEED = 181
D = 128
N_LAYER = 4
N_HEAD = 4
MAX_LEN = 64  # match curve MAX_ARCS
MICRO = 24
LR = 3e-4
EVAL_EVERY = 1500
DEFAULT_STEPS = 10_000
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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(F.normalize(a.float(), dim=0), F.normalize(b.float(), dim=0), dim=0))


def build_id_docs(tok: Tokenizer, text: str, max_docs: int = 4000) -> list[list[int]]:
    pad_id = tok.token_to_id(PAD)
    docs = []
    for block in text.split("\n\n"):
        block = block.strip()
        if len(block) < 40:
            continue
        ids = [i for i in tok.encode(block).ids if i != pad_id]
        if len(ids) >= 16:
            docs.append(ids)
        if len(docs) >= max_docs:
            break
    if len(docs) < 50:
        ids = [i for i in tok.encode(text[:2_000_000]).ids if i != pad_id]
        for i in range(0, max(1, len(ids) - 64), 48):
            docs.append(ids[i : i + 128])
            if len(docs) >= max_docs:
                break
    return docs


def sample_batch(docs: list[list[int]], batch: int, rng: random.Random, device, pad_id: int):
    xs = []
    for _ in range(batch):
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < 8:
            doc = doc * 4
        max_start = max(0, len(doc) - MAX_LEN)
        s = rng.randint(0, max_start) if max_start > 0 else 0
        window = doc[s : s + MAX_LEN]
        if len(window) < MAX_LEN:
            window = window + [pad_id] * (MAX_LEN - len(window))
        xs.append(window)
    x = torch.tensor(xs, dtype=torch.long, device=device)
    # labels: ignore pad
    y = x.clone()
    y[y == pad_id] = -100
    # shift for LM: predict next — GPT2 handles internally if labels=x
    return x


@torch.no_grad()
def hidden_last(model: GPT2LMHeadModel, ids: list[int], device, pad_id: int) -> torch.Tensor:
    ids = ids[-MAX_LEN:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    attn = (x != pad_id).long()
    out = model.transformer(input_ids=x, attention_mask=attn)
    h = out.last_hidden_state[0]  # [T,d]
    # last non-pad
    length = int(attn[0].sum().item())
    return h[length - 1]


@torch.no_grad()
def hidden_summary(model, ids: list[int], device, pad_id: int) -> torch.Tensor:
    ids = ids[-MAX_LEN:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    attn = (x != pad_id).long()
    h = model.transformer(input_ids=x, attention_mask=attn).last_hidden_state[0]
    length = int(attn[0].sum().item())
    h = h[:length]
    return F.normalize(torch.cat([h[-1], h.mean(0)], 0), dim=0)


def gate_A(model, docs: list[list[int]], device, pad_id: int, rng: random.Random, n_pairs: int = 80) -> dict:
    by_last: dict[int, list[list[int]]] = defaultdict(list)
    for doc in docs:
        if len(doc) < 12:
            continue
        for i in range(8, min(len(doc), 80)):
            last = doc[i]
            seq = doc[max(0, i - (MAX_LEN - 1)) : i + 1]
            if len(by_last[last]) < 40:
                pref = tuple(seq[:-1])
                if all(tuple(s[:-1]) != pref for s in by_last[last]):
                    by_last[last].append(seq)
    pairs_same = []
    for last, seqs in by_last.items():
        uniq = {}
        for s in seqs:
            key = tuple(s[:-1])
            if key not in uniq:
                uniq[key] = s
            if len(uniq) >= 2:
                break
        if len(uniq) >= 2:
            vals = list(uniq.values())
            pairs_same.append((vals[0], vals[1]))
        if len(pairs_same) >= n_pairs:
            break
    rng.shuffle(pairs_same)
    pairs_same = pairs_same[:n_pairs]
    flat = [s for seqs in list(by_last.values())[:200] for s in seqs[:3]]
    pairs_diff = []
    for _ in range(n_pairs * 4):
        if len(flat) < 2:
            break
        a, b = rng.sample(flat, 2)
        if a[-1] != b[-1]:
            pairs_diff.append((a, b))
        if len(pairs_diff) >= n_pairs:
            break

    cos_same = [
        cos(hidden_last(model, a, device, pad_id), hidden_last(model, b, device, pad_id)) for a, b in pairs_same
    ]
    cos_diff = [
        cos(hidden_last(model, a, device, pad_id), hidden_last(model, b, device, pad_id)) for a, b in pairs_diff
    ]
    m_same = float(np.mean(cos_same)) if cos_same else 1.0
    m_diff = float(np.mean(cos_diff)) if cos_diff else 0.0
    if m_same >= 0.98:
        verdict = "A_FAIL_LAST_TOKEN_WIPES"
    elif m_same < 0.90 and (m_same - m_diff) < 0.35:
        verdict = "A_PASS_PREFIX_VISIBLE"
    else:
        verdict = "A_WEAK_PARTIAL"
    return {
        "verdict": verdict,
        "mean_cos_same_last_piece": m_same,
        "mean_cos_diff_last_piece": m_diff,
        "n_same": len(cos_same),
        "n_diff": len(cos_diff),
    }


def gate_B(model, tok: Tokenizer, device, pad_id: int, rng: random.Random) -> dict:
    def enc(text: str):
        ids = [i for i in tok.encode(text).ids if i != pad_id]
        return hidden_summary(model, ids or [tok.token_to_id("a") or 1], device, pad_id)

    para = [cos(enc(a), enc(b)) for a, b in s179.PARAPHRASE_PAIRS]
    hard = [cos(enc(a), enc(b)) for a, b in s179.HARD_PAIRS]
    flat = []
    for a, b in s179.PARAPHRASE_PAIRS:
        flat.extend([enc(a), enc(b)])
    rand = []
    for _ in range(len(para) * 4):
        i, j = rng.sample(range(len(flat)), 2)
        rand.append(cos(flat[i], flat[j]))
    m_para, m_rand, m_hard = float(np.mean(para)), float(np.mean(rand)), float(np.mean(hard))
    gap = m_hard - m_para
    lift_r = m_para - m_rand
    if lift_r > 0.05 and (m_para - m_hard) > 0.03:
        verdict = "B_PASS_MEANING_STRUCTURE"
    elif gap > 0.05 and lift_r <= 0.05:
        verdict = "B_FORM_DOMINANT"  # level label, not wall
    elif lift_r <= 0.02:
        verdict = "B_NO_PARA_CLUSTER"
    else:
        verdict = "B_MICRO_MIXED"
    return {
        "verdict": verdict,
        "mean_cos_paraphrase": m_para,
        "mean_cos_random": m_rand,
        "mean_cos_hard_spelling": m_hard,
        "gap_hard_minus_para": gap,
        "lift_vs_random": lift_r,
        "n_para": len(para),
    }


@torch.no_grad()
def prefix_ablation(model, docs, device, pad_id, rng, n: int = 40) -> dict:
    """CE on last positions: natural vs shuffled prefix (same suffix)."""
    losses_nat, losses_shuf = [], []
    for _ in range(n):
        doc = docs[rng.randint(0, len(docs) - 1)]
        if len(doc) < MAX_LEN:
            continue
        s = rng.randint(0, len(doc) - MAX_LEN)
        window = doc[s : s + MAX_LEN]
        suf = max(8, MAX_LEN // 3)
        prefix, suffix = window[:-suf], window[-suf:]
        shuf_prefix = prefix.copy()
        rng.shuffle(shuf_prefix)
        for ids, bucket in ((window, losses_nat), (shuf_prefix + suffix, losses_shuf)):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids=x, labels=x)
            # focus on suffix tokens loss roughly via full CE (proxy)
            bucket.append(float(out.loss))
    if not losses_nat:
        return {"delta_shuf_minus_nat": 0.0, "nat": 0.0, "shuf": 0.0}
    nat, shuf = float(np.mean(losses_nat)), float(np.mean(losses_shuf))
    return {
        "mean_ce_natural": nat,
        "mean_ce_prefix_shuffled": shuf,
        "delta_shuf_minus_nat": shuf - nat,
        "n": len(losses_nat),
        "note": "positive delta => model used prefix (context helps CE)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage181 start {datetime.now(timezone.utc).isoformat()}")
    log("Matched CE GPT-2 control — dataset context ceiling vs curve stages")
    log(f"match: BPE={TOK_PATH} d={D} L={N_LAYER} H={N_HEAD} T={MAX_LEN} steps={args.steps}")

    if not TOK_PATH.exists():
        raise FileNotFoundError(TOK_PATH)
    tok = Tokenizer.from_file(str(TOK_PATH))
    pad_id = tok.token_to_id(PAD)
    if pad_id is None:
        pad_id = 0
    V = tok.get_vocab_size()
    text = s170.load_corpus(max_chars=20_000_000)
    docs = build_id_docs(tok, text)
    hold = docs[int(0.8 * len(docs)) :] or docs[-100:]
    train = docs[: int(0.8 * len(docs))] or docs
    log(f"docs={len(docs)} V={V} pad={pad_id}")

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    conf = GPT2Config(
        vocab_size=V,
        n_positions=MAX_LEN,
        n_embd=D,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=4 * D,
        bos_token_id=pad_id,
        eos_token_id=pad_id,
        pad_token_id=pad_id,
    )
    model = GPT2LMHeadModel(conf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)

    model.eval()
    A0 = gate_A(model, hold, device, pad_id, random.Random(SEED))
    B0 = gate_B(model, tok, device, pad_id, random.Random(SEED + 1))
    Ab0 = prefix_ablation(model, hold, device, pad_id, random.Random(SEED + 2))
    log(
        f"  init A: same={A0['mean_cos_same_last_piece']:.3f} diff={A0['mean_cos_diff_last_piece']:.3f} → {A0['verdict']}"
    )
    log(
        f"  init B: para={B0['mean_cos_paraphrase']:.3f} hard={B0['mean_cos_hard_spelling']:.3f} "
        f"gap={B0['gap_hard_minus_para']:.3f} lift_r={B0['lift_vs_random']:+.3f} → {B0['verdict']}"
    )
    log(f"  init ablation Δ(shuf-nat)={Ab0['delta_shuf_minus_nat']:+.4f}")

    history = []
    Af, Bf, Abf = A0, B0, Ab0
    running = None
    model.train()
    for step in range(1, args.steps + 1):
        x = sample_batch(train, MICRO, rng, device, pad_id)
        out = model(input_ids=x, labels=x)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(loss.detach()) if running is None else 0.95 * running + 0.05 * float(loss.detach())

        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            Af = gate_A(model, hold, device, pad_id, random.Random(SEED + step))
            Bf = gate_B(model, tok, device, pad_id, random.Random(SEED + step + 3))
            Abf = prefix_ablation(model, hold, device, pad_id, random.Random(SEED + step + 5))
            row = {
                "step": step,
                "ce": running,
                "A_same": Af["mean_cos_same_last_piece"],
                "A": Af["verdict"],
                "para": Bf["mean_cos_paraphrase"],
                "hard": Bf["mean_cos_hard_spelling"],
                "gap": Bf["gap_hard_minus_para"],
                "lift_r": Bf["lift_vs_random"],
                "B": Bf["verdict"],
                "ablation_delta": Abf["delta_shuf_minus_nat"],
            }
            history.append(row)
            log(
                f"  step {step}: ce~{running:.3f} A_same={row['A_same']:.3f}→{row['A']} | "
                f"para={row['para']:.3f} hard={row['hard']:.3f} gap={row['gap']:.3f}→{row['B']} | "
                f"ablΔ={row['ablation_delta']:+.4f}"
            )
            model.train()
            torch.save({"model": model.state_dict(), "conf": conf.to_dict(), "step": step, "A": Af, "B": Bf}, CKPT_OUT)

    # compare to 180 peak if present
    cmp180 = None
    if DEC180.exists():
        d180 = json.loads(DEC180.read_text(encoding="utf-8"))
        traj = d180.get("trajectory", {})
        peak = traj.get("4500", {})
        cmp180 = {
            "curve_peak_para": peak.get("para"),
            "curve_peak_gap": peak.get("gap"),
            "curve_A_slow_best": traj.get("1500", {}).get("A_slow"),
            "ce_final_para": Bf["mean_cos_paraphrase"],
            "ce_final_gap": Bf["gap_hard_minus_para"],
            "ce_final_A": Af["mean_cos_same_last_piece"],
            "ce_ablation_delta": Abf["delta_shuf_minus_nat"],
        }

    # micro trend on B
    if len(history) >= 2:
        para_delta = history[-1]["para"] - history[0]["para"]
        gap_delta = history[-1]["gap"] - history[0]["gap"]
    else:
        para_delta = gap_delta = 0.0

    a_ok = Af["mean_cos_same_last_piece"] < 0.90
    abl_ok = Abf["delta_shuf_minus_nat"] > 0.02
    micro_ok = para_delta > 0.01 or gap_delta < -0.01
    if a_ok and (abl_ok or micro_ok):
        overall = "CE_CONTROL_CONTEXT_SIGNAL_YES"
    elif a_ok:
        overall = "CE_CONTROL_A_ONLY"
    elif abl_ok:
        overall = "CE_CONTROL_ABLATION_ONLY"
    else:
        overall = "CE_CONTROL_FLAT"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "ce_transformer_control_181",
        "overall": overall,
        "matched": {
            "tokenizer": str(TOK_PATH),
            "corpus_chars": 20_000_000,
            "d": D,
            "layers": N_LAYER,
            "heads": N_HEAD,
            "seq": MAX_LEN,
            "steps": args.steps,
            "objective": "GPT2 next-token CE",
        },
        "init_A": A0,
        "init_B": B0,
        "init_ablation": Ab0,
        "final_A": Af,
        "final_B": Bf,
        "final_ablation": Abf,
        "b_micro": {"para_delta_first_to_last_eval": para_delta, "gap_delta_first_to_last_eval": gap_delta},
        "history": history,
        "vs_180": cmp180,
        "interpretation": {
            "FLAT": "this dataset+scale barely teaches context even for CE — don't wall the curve",
            "A_ONLY": "prefix lives in state; meaning micro weak — same regime as curve",
            "CONTEXT_SIGNAL_YES": "data supports context under CE — curve should chase this ceiling",
        },
        "next": "Use CE control as ceiling. If FLAT/A_ONLY: scale data before blaming curve. If YES: put semantic pressure on slow channel.",
    }
    write_json(DECISION, out)
    MINI.write_text(
        "\n".join(
            [
                "# Stage181 — CE Transformer control",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- A: {Af['verdict']} same={Af['mean_cos_same_last_piece']:.3f}",
                f"- B: {Bf['verdict']} para={Bf['mean_cos_paraphrase']:.3f} hard={Bf['mean_cos_hard_spelling']:.3f} gap={Bf['gap_hard_minus_para']:.3f}",
                f"- ablation Δ={Abf['delta_shuf_minus_nat']:+.4f}",
                f"- B micro: paraΔ={para_delta:+.3f} gapΔ={gap_delta:+.3f}",
                f"- {out['next']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[181] {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
