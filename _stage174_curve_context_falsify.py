"""
Stage 174 — Early context/meaning falsify on frozen curve (no dyn retrain).

A) same letter-suffix, different prefix → does z still differ?
B) paraphrase pairs vs random pairs → is z closer for same meaning?
C) sentence-order shuffle (local orthography kept, discourse broken) → does z/dyn care?

  python _stage174_curve_context_falsify.py
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _stage170_curve_dynamics as s170
import _stage172_curve_scale as s172

RES = Path("results")
CKPT_PEN = Path("checkpoints/stage170_curve.pt")
CKPT_DYN = Path("checkpoints/stage172_curve.pt")
LOG = RES / "_stage174_log.txt"
DECISION = RES / "stage174_context_falsify_decision.json"
MINI = RES / "stage174_context_falsify_mini.md"

SEED = 174
SUFFIX_LEN = 24
PREFIX_LEN = 96
N_SAME_SUFFIX_PAIRS = 120
N_DIFF_SUFFIX_PAIRS = 120


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def to_ids(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long)


@torch.no_grad()
def encode_z(model, text: str, stoi: dict, device) -> torch.Tensor:
    x = to_ids(text, stoi).unsqueeze(0).to(device)
    return model.encode(x)[0]  # [T,d]


def z_summary(z: torch.Tensor) -> torch.Tensor:
    """Endpoint + mean pool — fixed vector for comparisons."""
    return F.normalize(torch.cat([z[-1], z.mean(0)], dim=0), dim=0)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ---------- A: same suffix ----------

def mine_same_suffix_pairs(text: str, rng: random.Random):
    """Windows ending with identical last SUFFIX_LEN chars, different earlier PREFIX_LEN."""
    L = PREFIX_LEN + SUFFIX_LEN
    by_suf = defaultdict(list)
    step = 17
    for i in range(0, len(text) - L - 1, step):
        win = text[i : i + L]
        suf = win[-SUFFIX_LEN:]
        pref = win[:-SUFFIX_LEN]
        by_suf[suf].append((i, pref, win))

    pairs = []
    for suf, items in by_suf.items():
        if len(items) < 2:
            continue
        # distinct prefixes
        uniq = {}
        for i, pref, win in items:
            if pref not in uniq:
                uniq[pref] = win
            if len(uniq) >= 2:
                break
        if len(uniq) < 2:
            continue
        ws = list(uniq.values())
        pairs.append((ws[0], ws[1], suf))
        if len(pairs) >= N_SAME_SUFFIX_PAIRS * 2:
            break
    rng.shuffle(pairs)
    return pairs[:N_SAME_SUFFIX_PAIRS]


def mine_diff_suffix_pairs(text: str, rng: random.Random, n: int):
    L = PREFIX_LEN + SUFFIX_LEN
    pairs = []
    for _ in range(n * 3):
        i = rng.randint(0, len(text) - L - 1)
        j = rng.randint(0, len(text) - L - 1)
        a, b = text[i : i + L], text[j : j + L]
        if a[-SUFFIX_LEN:] == b[-SUFFIX_LEN:]:
            continue
        pairs.append((a, b))
        if len(pairs) >= n:
            break
    return pairs


@torch.no_grad()
def test_A(model, text, stoi, device, rng) -> dict:
    log("### A) same letter-suffix, different prefix")
    same = mine_same_suffix_pairs(text, rng)
    diff = mine_diff_suffix_pairs(text, rng, N_DIFF_SUFFIX_PAIRS)
    log(f"  pairs same_suffix={len(same)} diff_suffix={len(diff)}")

    cos_same, cos_diff = [], []
    for a, b, suf in same:
        za, zb = encode_z(model, a, stoi, device), encode_z(model, b, stoi, device)
        # endpoint only (most sensitive to wipe by suffix) + summary
        cos_same.append(cos(F.normalize(za[-1], dim=0), F.normalize(zb[-1], dim=0)))
    for a, b in diff:
        za, zb = encode_z(model, a, stoi, device), encode_z(model, b, stoi, device)
        cos_diff.append(cos(F.normalize(za[-1], dim=0), F.normalize(zb[-1], dim=0)))

    # Also: does earlier context leave a trace BEFORE the suffix dominates?
    # Compare z at position -SUFFIX_LEN-1 (end of prefix) for same-suffix pairs
    cos_at_prefix_end = []
    for a, b, suf in same[:80]:
        za, zb = encode_z(model, a, stoi, device), encode_z(model, b, stoi, device)
        # index of last prefix char
        i = PREFIX_LEN - 1
        cos_at_prefix_end.append(cos(F.normalize(za[i], dim=0), F.normalize(zb[i], dim=0)))

    m_same = float(np.mean(cos_same)) if cos_same else 0.0
    m_diff = float(np.mean(cos_diff)) if cos_diff else 0.0
    m_pref = float(np.mean(cos_at_prefix_end)) if cos_at_prefix_end else 0.0
    # Context wipe: same-suffix endpoints much closer than diff-suffix,
    # AND much closer than they were at prefix-end
    wipe = (m_same - m_diff) > 0.15 and (m_same - m_pref) > 0.10
    # Context retained: same-suffix endpoints still clearly separated like random
    retain = (m_same - m_diff) < 0.05

    if wipe:
        verdict = "A_FAIL_CONTEXT_WIPED_BY_SUFFIX"
    elif retain:
        verdict = "A_PASS_PREFIX_STILL_VISIBLE"
    else:
        verdict = "A_WEAK_PARTIAL_TRACE"

    out = {
        "verdict": verdict,
        "n_same": len(cos_same),
        "n_diff": len(cos_diff),
        "mean_cos_endpoint_same_suffix": m_same,
        "mean_cos_endpoint_diff_suffix": m_diff,
        "mean_cos_at_prefix_end_same_suf_pairs": m_pref,
        "delta_same_minus_diff": m_same - m_diff,
    }
    log(
        f"  endpoint cos same_suf={m_same:.3f} diff_suf={m_diff:.3f} "
        f"at_prefix_end={m_pref:.3f} → {verdict}"
    )
    return out


# ---------- B: paraphrases ----------

PARAPHRASE_PAIRS = [
    ("The cat sat on the mat.", "A cat was sitting on the mat."),
    ("She quickly opened the door.", "She opened the door quickly."),
    ("He bought a new car yesterday.", "Yesterday he purchased a new automobile."),
    ("The weather is very cold today.", "It is extremely chilly outside today."),
    ("Children are playing in the park.", "Kids are playing at the park."),
    ("I need to finish this work soon.", "I must complete this task shortly."),
    ("The book was written by a famous author.", "A famous writer wrote the book."),
    ("They arrived at the station early.", "They got to the station early."),
    ("Water boils at one hundred degrees.", "Water boils at 100 degrees."),
    ("The dog chased the ball across the yard.", "Across the yard the dog ran after the ball."),
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
    ("The bridge connects the two cities.", "The two cities are linked by the bridge."),
    ("She drank a cup of tea.", "She had a cup of tea."),
    ("The problem is difficult to solve.", "Solving the problem is hard."),
    ("He forgot his keys at home.", "He left his keys at home."),
    ("The sun rises in the east.", "In the east the sun comes up."),
]


@torch.no_grad()
def test_B(model, stoi, device, rng) -> dict:
    log("### B) paraphrase proximity vs random pairs")
    vecs = []
    for a, b in PARAPHRASE_PAIRS:
        za = z_summary(encode_z(model, a, stoi, device))
        zb = z_summary(encode_z(model, b, stoi, device))
        vecs.append((za, zb, cos(za, zb)))

    para_cos = [c for _, _, c in vecs]
    # random cross pairs (different meanings)
    flat = [za for za, zb, _ in vecs] + [zb for za, zb, _ in vecs]
    rand_cos = []
    for _ in range(len(para_cos) * 4):
        i, j = rng.sample(range(len(flat)), 2)
        # avoid exact paraphrase partners roughly by index distance
        rand_cos.append(cos(flat[i], flat[j]))

    # Also: surface-similar but different meaning (hard negative)
    hard = [
        ("The cat sat on the mat.", "The car sat on the mat."),
        ("She opened the door quickly.", "She opened the book quickly."),
        ("He bought a new car yesterday.", "He bought a new cat yesterday."),
        ("The weather is very cold today.", "The weather is very warm today."),
        ("Children are playing in the park.", "Children are studying in the park."),
        ("The train leaves at noon.", "The plane leaves at noon."),
        ("Water boils at one hundred degrees.", "Oil boils at one hundred degrees."),
        ("She teaches mathematics at school.", "She teaches history at school."),
    ]
    hard_cos = []
    for a, b in hard:
        hard_cos.append(cos(z_summary(encode_z(model, a, stoi, device)), z_summary(encode_z(model, b, stoi, device))))

    m_para = float(np.mean(para_cos))
    m_rand = float(np.mean(rand_cos))
    m_hard = float(np.mean(hard_cos))
    lift_rand = m_para - m_rand
    lift_hard = m_para - m_hard

    # PASS if paraphrases clearly closer than random AND not closer than hard negs only by spelling
    # FAIL if paraphrases ≈ random, or paraphrases ≈ hard (meaning-blind, form-driven)
    if lift_rand > 0.05 and lift_hard > 0.03:
        verdict = "B_PASS_MEANING_STRUCTURE"
    elif lift_rand > 0.03 and lift_hard <= 0.02:
        verdict = "B_FAIL_FORM_NOT_MEANING"  # paraphrases closer than random but ~ hard spelling cousins
    elif lift_rand <= 0.02:
        verdict = "B_FAIL_NO_PARAPHRASE_CLUSTER"
    else:
        verdict = "B_WEAK_MIXED"

    out = {
        "verdict": verdict,
        "mean_cos_paraphrase": m_para,
        "mean_cos_random": m_rand,
        "mean_cos_hard_spelling": m_hard,
        "lift_vs_random": lift_rand,
        "lift_vs_hard": lift_hard,
        "n_para": len(para_cos),
    }
    log(
        f"  para={m_para:.3f} random={m_rand:.3f} hard_spell={m_hard:.3f} "
        f"lift_rand={lift_rand:+.3f} lift_hard={lift_hard:+.3f} → {verdict}"
    )
    return out


# ---------- C: sentence shuffle ----------

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p) > 20]


@torch.no_grad()
def test_C(model, text, stoi, device, rng) -> dict:
    log("### C) sentence-order shuffle (discourse break, local orthography kept)")
    # build paragraphs with >= 4 sentences
    chunks = []
    buf = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if buf:
                chunks.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        chunks.append(" ".join(buf))

    samples = []
    for ch in chunks:
        sents = split_sentences(ch)
        if len(sents) >= 4:
            samples.append(sents)
        if len(samples) >= 80:
            break

    cos_nat_shuf = []
    cos_nat_wordshuf = []
    dyn_drop = []  # optional: k1 cos on natural vs sentence-shuffled stream via dyn — skip heavy

    for sents in samples:
        nat = " ".join(sents)
        shuf_s = sents[:]
        rng.shuffle(shuf_s)
        # ensure actually different order
        if shuf_s == sents:
            continue
        sent_shuf = " ".join(shuf_s)
        # word shuffle: destroy local order more
        words = nat.split()
        rng.shuffle(words)
        word_shuf = " ".join(words)

        # truncate to similar length for fair encode
        L = min(400, len(nat), len(sent_shuf), len(word_shuf))
        nat, sent_shuf, word_shuf = nat[:L], sent_shuf[:L], word_shuf[:L]

        zn = z_summary(encode_z(model, nat, stoi, device))
        zs = z_summary(encode_z(model, sent_shuf, stoi, device))
        zw = z_summary(encode_z(model, word_shuf, stoi, device))
        cos_nat_shuf.append(cos(zn, zs))
        cos_nat_wordshuf.append(cos(zn, zw))

    m_sent = float(np.mean(cos_nat_shuf)) if cos_nat_shuf else 0.0
    m_word = float(np.mean(cos_nat_wordshuf)) if cos_nat_wordshuf else 0.0

    # If sentence shuffle barely moves z (high cos) but word shuffle does → discourse-blind, local-ortho
    # If sentence shuffle moves z a lot → some long-range sensitivity (not necessarily meaning)
    if m_sent > 0.90 and (m_sent - m_word) > 0.08:
        verdict = "C_FAIL_DISCOURSE_BLIND_LOCAL_ORTHO"
    elif m_sent < 0.75:
        verdict = "C_PASS_ORDER_SENSITIVE"
    else:
        verdict = "C_WEAK_PARTIAL_ORDER"

    out = {
        "verdict": verdict,
        "n": len(cos_nat_shuf),
        "mean_cos_natural_vs_sentence_shuffle": m_sent,
        "mean_cos_natural_vs_word_shuffle": m_word,
        "gap_sent_minus_word": m_sent - m_word,
    }
    log(f"  cos(nat,sent_shuf)={m_sent:.3f} cos(nat,word_shuf)={m_word:.3f} → {verdict}")
    return out


def combine_verdict(a, b, c) -> tuple[str, str]:
    fails = sum(1 for v in (a, b, c) if "FAIL" in v)
    passes = sum(1 for v in (a, b, c) if "PASS" in v)
    if fails >= 2 and passes == 0:
        return (
            "CONTEXT_WALL_ON_CURVE",
            "Curve holds letter-path form but fails early context/meaning probes — do not scale dyn.",
        )
    if passes >= 2:
        return (
            "CONTEXT_SIGNAL_POSSIBLE",
            "Enough signal to justify careful dyn/pen work aimed at these probes.",
        )
    return (
        "CONTEXT_UNCLEAR_MIXED",
        "Mixed A/B/C — one more targeted probe before any long soak.",
    )


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage174 start {datetime.now(timezone.utc).isoformat()}")
    log("Early context/meaning falsify on frozen pen@170 + dyn@172 (no retrain)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pen_blob = torch.load(CKPT_PEN, map_location="cpu", weights_only=False)
    dyn_blob = torch.load(CKPT_DYN, map_location="cpu", weights_only=False)
    stoi, itos = pen_blob["stoi"], pen_blob["itos"]

    model = s172.ScaleModel(len(itos)).to(device)
    model.load_state_dict(dyn_blob["model"], strict=False)
    pen_sd = {k[len("pen.") :]: v for k, v in pen_blob["model"].items() if k.startswith("pen.")}
    model.pen.load_state_dict(pen_sd, strict=True)
    for p in model.pen.parameters():
        p.requires_grad_(False)
    model.eval()
    log(f"device={device} dyn_step={dyn_blob.get('step')}")

    text = s170.load_corpus(max_chars=8_000_000)
    # mid hold-ish slice
    text = text[2_000_000:6_000_000]
    rng = random.Random(SEED)

    A = test_A(model, text, stoi, device, rng)
    B = test_B(model, stoi, device, rng)
    C = test_C(model, text, stoi, device, rng)

    overall, detail = combine_verdict(A["verdict"], B["verdict"], C["verdict"])
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "curve_context_falsify_174",
        "overall": overall,
        "detail": detail,
        "A_same_suffix": A,
        "B_paraphrase": B,
        "C_sentence_shuffle": C,
        "recommendation": (
            "STOP scaling dyn; redesign pen/object or accept script-engine role"
            if overall == "CONTEXT_WALL_ON_CURVE"
            else (
                "Proceed carefully with probes as gates"
                if overall == "CONTEXT_SIGNAL_POSSIBLE"
                else "One more falsify before investment"
            )
        ),
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    bullets = [
        f"overall `{overall}`",
        detail,
        f"A: {A['verdict']} (same_suf_end={A['mean_cos_endpoint_same_suffix']:.3f} diff={A['mean_cos_endpoint_diff_suffix']:.3f} pref_end={A['mean_cos_at_prefix_end_same_suf_pairs']:.3f})",
        f"B: {B['verdict']} (para={B['mean_cos_paraphrase']:.3f} rand={B['mean_cos_random']:.3f} hard={B['mean_cos_hard_spelling']:.3f})",
        f"C: {C['verdict']} (nat~sent_shuf={C['mean_cos_natural_vs_sentence_shuffle']:.3f} nat~word_shuf={C['mean_cos_natural_vs_word_shuffle']:.3f})",
        f"recommendation: {out['recommendation']}",
    ]
    MINI.write_text(
        "\n".join(
            ["# Stage174 — context/meaning early falsify", "", f"**Overall:** `{overall}`", ""]
            + [f"- {b}" for b in bullets]
            + [""]
        ),
        encoding="utf-8",
    )
    log(f"[174] {overall}")
    log(detail)
    log(out["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
