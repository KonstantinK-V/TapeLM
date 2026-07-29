"""
Stage 179 — Harden retention (178) + Gate B (meaning vs form).

Continue the retention objective from Stage178, longer train.
Track:
  A) same last BPE piece / different prefix (must stay PASS-ish)
  B) paraphrase proximity vs random vs hard spelling cousins
     (174-style: is z about meaning or just form?)

Optional mid anneal of retention weight to see if A holds without constant push.

  python _stage179_curve_harden_B.py
  python _stage179_curve_harden_B.py --steps 10000
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage170_curve_dynamics as s170
import _stage177_curve_bpe as s177
import _stage178_curve_retention as s178

RES = Path("results")
CKPT_DIR = Path("checkpoints")
LOG = RES / "_stage179_log.txt"
DECISION = RES / "stage179_decision.json"
MINI = RES / "stage179_mini.md"
CKPT_IN = s178.CKPT_OUT
CKPT_OUT = CKPT_DIR / "stage179_curve_harden.pt"
TOK_PATH = s177.TOK_PATH
PLAN = RES / "plan_curve_dynamics.md"

SEED = 179
D = s177.D
MAX_ARCS = s177.MAX_ARCS
MICRO = 16
RET_PAIRS = 8
LR = 2e-4
EVAL_EVERY = 1500
DEFAULT_STEPS = 10_000
ANNEAL_AT = 4500  # soften retention after harden phase
W_RET_START = 1.5
W_RET_AFTER = 0.4


PARAPHRASE_PAIRS = [
    ("The cat sat on the mat.", "A cat was sitting on the mat."),
    ("She quickly opened the door.", "She opened the door quickly."),
    ("He bought a new car yesterday.", "Yesterday he purchased a new automobile."),
    ("The weather is very cold today.", "It is extremely chilly outside today."),
    ("Children are playing in the park.", "Kids are playing at the park."),
    ("I need to finish this work soon.", "I must complete this task shortly."),
    ("The book was written by a famous author.", "A famous writer wrote the book."),
    ("They arrived at the station early.", "They got to the station early."),
    ("Please close the window.", "Could you shut the window?"),
    ("The train leaves at noon.", "The train departs at midday."),
    ("He is afraid of spiders.", "Spiders scare him."),
    ("She teaches mathematics at school.", "She is a math teacher at the school."),
    ("The film was long and boring.", "The movie was lengthy and dull."),
    ("We should start the meeting now.", "Let's begin the meeting now."),
    ("The river flows into the sea.", "The river runs into the ocean."),
    ("His answer was completely wrong.", "His reply was totally incorrect."),
    ("The store opens at nine.", "The shop opens at 9."),
    ("Birds fly south in winter.", "In winter birds migrate south."),
    ("She drank a cup of tea.", "She had a cup of tea."),
    ("The problem is difficult to solve.", "Solving the problem is hard."),
    ("He forgot his keys at home.", "He left his keys at home."),
    ("The sun rises in the east.", "In the east the sun comes up."),
    ("The bridge connects the two cities.", "The two cities are linked by the bridge."),
    ("Water boils at one hundred degrees.", "Water boils at 100 degrees."),
    ("The dog chased the ball across the yard.", "Across the yard the dog ran after the ball."),
]

HARD_PAIRS = [
    ("The cat sat on the mat.", "The car sat on the mat."),
    ("She opened the door quickly.", "She opened the book quickly."),
    ("He bought a new car yesterday.", "He bought a new cat yesterday."),
    ("The weather is very cold today.", "The weather is very warm today."),
    ("Children are playing in the park.", "Children are studying in the park."),
    ("The train leaves at noon.", "The plane leaves at noon."),
    ("Water boils at one hundred degrees.", "Oil boils at one hundred degrees."),
    ("She teaches mathematics at school.", "She teaches history at school."),
]


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


class GateWrap(nn.Module):
    def __init__(self, m: s178.RetentionModel):
        super().__init__()
        self.m = m

    def forward_states(self, char_ids, pad_mask=None):
        return self.m.forward_states(self.m.encode_arcs(char_ids), pad_mask)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(F.normalize(a, dim=0), F.normalize(b, dim=0), dim=0))


@torch.no_grad()
def encode_text_states(model: GateWrap, tok: Tokenizer, text: str, stoi: dict, device) -> torch.Tensor:
    pcs = s177.encode_pieces(tok, text)
    if not pcs:
        pcs = ["."]
    pcs = pcs[-MAX_ARCS:]
    char_ids = s177.pieces_to_char_ids(pcs, stoi).unsqueeze(0).to(device)
    pad = torch.zeros(1, len(pcs), dtype=torch.bool, device=device)
    return model.forward_states(char_ids, pad_mask=pad)[0]  # [A,d]


def z_summary(states: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.cat([states[-1], states.mean(0)], 0), dim=0)


@torch.no_grad()
def gate_B(model: GateWrap, tok: Tokenizer, stoi: dict, device, rng: random.Random) -> dict:
    para = []
    flat = []
    for a, b in PARAPHRASE_PAIRS:
        za = z_summary(encode_text_states(model, tok, a, stoi, device))
        zb = z_summary(encode_text_states(model, tok, b, stoi, device))
        para.append(cos(za, zb))
        flat.extend([za, zb])
    hard = [
        cos(
            z_summary(encode_text_states(model, tok, a, stoi, device)),
            z_summary(encode_text_states(model, tok, b, stoi, device)),
        )
        for a, b in HARD_PAIRS
    ]
    rand = []
    for _ in range(len(para) * 4):
        i, j = rng.sample(range(len(flat)), 2)
        rand.append(cos(flat[i], flat[j]))
    m_para, m_rand, m_hard = float(np.mean(para)), float(np.mean(rand)), float(np.mean(hard))
    lift_r, lift_h = m_para - m_rand, m_para - m_hard
    if lift_r > 0.05 and lift_h > 0.03:
        verdict = "B_PASS_MEANING_STRUCTURE"
    elif lift_r > 0.03 and lift_h <= 0.02:
        verdict = "B_FAIL_FORM_NOT_MEANING"
    elif lift_h <= 0.0 and lift_r > 0.02:
        verdict = "B_FAIL_FORM_NOT_MEANING"
    elif lift_r <= 0.02:
        verdict = "B_FAIL_NO_PARAPHRASE_CLUSTER"
    else:
        verdict = "B_WEAK_MIXED"
    return {
        "verdict": verdict,
        "mean_cos_paraphrase": m_para,
        "mean_cos_random": m_rand,
        "mean_cos_hard_spelling": m_hard,
        "lift_vs_random": lift_r,
        "lift_vs_hard": lift_h,
        "n_para": len(para),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-scratch", action="store_true")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage179 start {datetime.now(timezone.utc).isoformat()}")
    log("Harden 178 retention + gate B (paraphrase vs hard spelling)")
    log(f"plan={PLAN}")

    if not TOK_PATH.exists():
        raise FileNotFoundError(TOK_PATH)
    tok = Tokenizer.from_file(str(TOK_PATH))
    text = s170.load_corpus(max_chars=20_000_000)
    chars = sorted(set(text) | {" "})
    itos = ["<pad>"] + chars
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    docs = s177.build_piece_docs(tok, text)
    hold = docs[int(0.8 * len(docs)) :] or docs[-100:]
    train = docs[: int(0.8 * len(docs))] or docs
    index = s178.build_same_last_index(train)
    log(f"docs={len(docs)} same-last={len(index)} V={tok.get_vocab_size()}")

    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = s178.RetentionModel(len(itos)).to(device)
    start_step = 0
    if CKPT_IN.exists() and not args.from_scratch:
        ck = torch.load(CKPT_IN, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        start_step = int(ck.get("step", 0))
        log(f"loaded {CKPT_IN} step={start_step}")
    else:
        log("training from scratch (no 178 ckpt)")

    gmodel = GateWrap(model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = random.Random(SEED)

    # patch retention weight on module used by s178.retention_loss via global — instead scale manually
    model.eval()
    A0 = s177.gate_A(gmodel, hold, stoi, device, random.Random(SEED))
    B0 = gate_B(gmodel, tok, stoi, device, random.Random(SEED + 1))
    log(
        f"  init A: same={A0['mean_cos_same_last_piece']:.3f} diff={A0['mean_cos_diff_last_piece']:.3f} → {A0['verdict']}"
    )
    log(
        f"  init B: para={B0['mean_cos_paraphrase']:.3f} rand={B0['mean_cos_random']:.3f} "
        f"hard={B0['mean_cos_hard_spelling']:.3f} lift_r={B0['lift_vs_random']:+.3f} "
        f"lift_h={B0['lift_vs_hard']:+.3f} → {B0['verdict']}"
    )

    history = []
    Af, Bf = A0, B0
    running = None
    model.train()
    for step in range(1, args.steps + 1):
        w_ret = W_RET_START if step < ANNEAL_AT else W_RET_AFTER
        # temporarily scale: call losses and reweight ret
        x, pad = s177.sample_batch(train, stoi, MICRO, rng, device)
        loss_d, st = s178.dynamics_bundle(model, x, pad)
        ret_batch = s178.sample_retention_pair_batch(index, stoi, RET_PAIRS, rng, device)
        if ret_batch is not None:
            # s178.retention_loss already multiplies W_RET; rescale to desired w_ret
            loss_r, st_r = s178.retention_loss(model, ret_batch)
            loss_r = loss_r * (w_ret / s178.W_RET)
            loss = loss_d + loss_r
            st.update(st_r)
        else:
            loss = loss_d
            st["ret_cos"] = 1.0

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = float(loss.detach()) if running is None else 0.95 * running + 0.05 * float(loss.detach())

        if step % EVAL_EVERY == 0 or step == args.steps:
            model.eval()
            Af = s177.gate_A(gmodel, hold, stoi, device, random.Random(SEED + step))
            Bf = gate_B(gmodel, tok, stoi, device, random.Random(SEED + step + 7))
            row = {
                "step": step,
                "w_ret": w_ret,
                "A_same": Af["mean_cos_same_last_piece"],
                "A_diff": Af["mean_cos_diff_last_piece"],
                "A": Af["verdict"],
                "B_para": Bf["mean_cos_paraphrase"],
                "B_rand": Bf["mean_cos_random"],
                "B_hard": Bf["mean_cos_hard_spelling"],
                "B": Bf["verdict"],
            }
            history.append(row)
            log(
                f"  step {step}: loss~{running:.3f} w_ret={w_ret:.2f} "
                f"ret_cos={st.get('ret_cos',0):.3f} past={st.get('cos_past',0):.3f} inst={st.get('cos_inst',0):.3f} "
                f"A_same={row['A_same']:.3f} A_diff={row['A_diff']:.3f}→{row['A']} | "
                f"B para={row['B_para']:.3f} rand={row['B_rand']:.3f} hard={row['B_hard']:.3f}→{row['B']}"
            )
            model.train()
            torch.save(
                {"model": model.state_dict(), "stoi": stoi, "step": start_step + step, "A": Af, "B": Bf},
                CKPT_OUT,
            )

    # overall
    a_ok = Af["mean_cos_same_last_piece"] < 0.90 and "FAIL" not in Af["verdict"]
    if "PASS" in Bf["verdict"] and a_ok:
        overall = "HARDEN_A_YES_B_YES"
    elif a_ok and "WEAK" in Bf["verdict"]:
        overall = "HARDEN_A_YES_B_WEAK"
    elif a_ok:
        overall = "HARDEN_A_YES_B_FAIL"
    elif "PASS" in Bf["verdict"]:
        overall = "HARDEN_A_FAIL_B_YES"  # weird
    else:
        overall = "HARDEN_A_FAIL_B_FAIL"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "curve_harden_B_179",
        "overall": overall,
        "steps": args.steps,
        "loaded_178": CKPT_IN.exists() and not args.from_scratch,
        "anneal": {"at": ANNEAL_AT, "w_ret_before": W_RET_START, "w_ret_after": W_RET_AFTER},
        "init_A": A0,
        "init_B": B0,
        "final_A": Af,
        "final_B": Bf,
        "history": history,
        "note": "A=prefix visible; B=paraphrase vs hard spelling (meaning vs form).",
        "next": (
            "If A_YES_B_FAIL: retention≠meaning — need semantic/instance channel, not more A soak. "
            "If B_YES: scale carefully + longer context probes. "
            "If A collapses after anneal: retention still a crutch."
        ),
    }
    write_json(DECISION, out)
    MINI.write_text(
        "\n".join(
            [
                "# Stage179 — harden + gate B",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- A init→final: {A0['mean_cos_same_last_piece']:.3f}→{Af['mean_cos_same_last_piece']:.3f} ({Af['verdict']})",
                f"- B: {Bf['verdict']} para={Bf['mean_cos_paraphrase']:.3f} rand={Bf['mean_cos_random']:.3f} hard={Bf['mean_cos_hard_spelling']:.3f}",
                f"- lift_rand={Bf['lift_vs_random']:+.3f} lift_hard={Bf['lift_vs_hard']:+.3f}",
                f"- {out['next']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[179] {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
