"""
Stage 204 — W1: substrate robustness to noise / OOV (the axis where BPE is weak BY CONSTRUCTION).

Two fair, space-agnostic tests (ranking-based, so curve-cos and GPT-cos never compared directly):

  A. identity retrieval under char noise: query = typo(w) -> retrieve clean w among a pool.
     reported separately for SEEN (real corpus entities) and OOV (novel pronounceable fakes).
  B. downstream fact recall with NOISY queries: planted facts, subject-anchored keys,
     curve fp memory vs fair GPT+RAG mirror (identical key/query recipe, only encoder differs).
  C. mechanism stat: BPE pieces per word, clean vs noisy (why BPE breaks).

Gates:
  G_idA     curve identity acc at p=0.2 & 0.3 beats GPT by >= +0.10
  G_factB   curve fact recall at p=0.3 beats fair RAG by >= +0.10
  G_degrade curve relative drop (p0 -> p0.3) < RAG relative drop

  python _stage204_noise_robustness.py
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
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, WORD_RE, FpBank
from _stage196_tapelm import load_gpt

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage204_decision.json"
MINI = RES / "stage204_mini.md"
LOG = RES / "_stage204_log.txt"

SEED = 204
CORPUS_CHARS = 30_000_000
MAX_ARCS = 64
N_SEEN = 700
N_OOV = 300
N_FACTS = 150
N_FILLER = 400
CTX_WIN = 100
LEVELS = [0.0, 0.1, 0.2, 0.3]


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


ALPHA = "abcdefghijklmnopqrstuvwxyz"


def noisy(w: str, p: float, rng: random.Random) -> str:
    """Char noise (sub/del/ins/swap). Position 0 is preserved so we measure spelling noise,
    not a capitalization artifact of the tokenizer."""
    if p <= 0 or len(w) < 4:
        return w
    out = [w[0]]
    edited = False
    for c in w[1:]:
        if rng.random() < p:
            edited = True
            op = rng.choice(("sub", "del", "ins", "swap"))
            if op == "sub":
                out.append(rng.choice(ALPHA))
            elif op == "del":
                continue
            elif op == "ins":
                out.append(c)
                out.append(rng.choice(ALPHA))
            else:  # swap with previous
                if len(out) >= 1:
                    out.insert(len(out) - 1, c)
                else:
                    out.append(c)
        else:
            out.append(c)
    if not edited:  # guarantee at least one edit when p > 0
        i = rng.randrange(1, len(w))
        out = list(w[:i]) + [rng.choice(ALPHA)] + list(w[i + 1 :])
    s = "".join(out)
    return s if len(s) >= 3 else w


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage204 start {datetime.now(timezone.utc).isoformat()}")
    log("W1: substrate robustness to noise/OOV — curve fp vs fair GPT mirror")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bank = FpBank(model, stoi, device)
    gm = load_gpt(device)
    log(f"models loaded ({time.time()-t0:.0f}s)")

    @torch.no_grad()
    def gpt_emb(ids):
        ids = [i for i in ids if i != pad_id][-MAX_ARCS:]
        if not ids:
            return None
        h = gm.transformer(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0].mean(0)
        return F.normalize(h, dim=-1)

    def gpt_word(w):
        return gpt_emb(tok.encode(" " + w).ids)

    def gpt_ctx(textv, exclude=None):
        ws = [x for x in WORD_RE.findall(textv) if x != exclude][:40]
        return gpt_emb(tok.encode(" ".join(ws)).ids) if len(ws) >= 3 else None

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 300]
    seen = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(text) if len(m.group(1)) >= 5))
    rng.shuffle(seen)
    seen = seen[:N_SEEN]
    oov = [w for w in gen_fakes(set(seen), rng, N_OOV + 60) if len(w) >= 5][:N_OOV]
    pool = seen + oov
    log(f"pool: seen={len(seen)} oov={len(oov)} paras={len(paras)} ({time.time()-t0:.0f}s)")

    # ---------- A. identity retrieval under noise (rank-based, fair across spaces) ----------
    pool_curve = bank.fp(pool)
    pool_gpt = torch.stack([gpt_word(w) for w in pool], 0)
    log(f"pool encoded both spaces ({time.time()-t0:.0f}s)")

    A = {"curve_seen": {}, "gpt_seen": {}, "curve_oov": {}, "gpt_oov": {}}
    for p in LEVELS:
        nrng = random.Random(SEED + int(p * 100))
        qs = [noisy(w, p, nrng) for w in pool]
        qc = bank.fp(qs)
        pred_c = (qc @ pool_curve.T).argmax(dim=-1).tolist()
        qg = torch.stack([gpt_word(w) for w in qs], 0)
        pred_g = (qg @ pool_gpt.T).argmax(dim=-1).tolist()
        for tag, lo, hi in (("seen", 0, len(seen)), ("oov", len(seen), len(pool))):
            idx = range(lo, hi)
            A[f"curve_{tag}"][p] = sum(int(pred_c[i] == i) for i in idx) / max(1, hi - lo)
            A[f"gpt_{tag}"][p] = sum(int(pred_g[i] == i) for i in idx) / max(1, hi - lo)
        log(
            f"  A p={p:.1f}: curve seen={A['curve_seen'][p]:.3f} oov={A['curve_oov'][p]:.3f} | "
            f"gpt seen={A['gpt_seen'][p]:.3f} oov={A['gpt_oov'][p]:.3f} ({time.time()-t0:.0f}s)"
        )

    # ---------- B. fact recall with noisy queries (curve fp memory vs fair GPT+RAG) ----------
    subjects = [w for w in gen_fakes(set(seen), random.Random(SEED + 7), N_FACTS + 80) if len(w) >= 5][:N_FACTS]
    values = seen[: N_FACTS]
    facts = []
    for S, Vv in zip(subjects, values):
        filler = paras[rng.randrange(len(paras))][:200]
        facts.append({"S": S, "value": Vv, "text": f"{filler} {S} was appointed director of {Vv} in 1987 ."})

    rows_c, rows_g, slot_val = [], [], []
    for fdict in facts:  # subject-anchored keys (197 M3 recipe), identical for both encoders
        kc = bank.fp([fdict["S"]])[0]
        cc = bank.ctx_fp(fdict["text"], exclude=fdict["value"])
        rows_c.append(F.normalize(kc + cc, dim=-1) if cc is not None else kc)
        kg = gpt_word(fdict["S"])
        cg = gpt_ctx(fdict["text"], exclude=fdict["value"])
        rows_g.append(F.normalize(kg + cg, dim=-1) if cg is not None else kg)
        slot_val.append(fdict["value"])
    for para in paras[:N_FILLER]:  # distractor slots from real text
        m = ENT_RE.search(para)
        if not m:
            continue
        ent = m.group(1)
        lo, hi = max(0, m.start() - CTX_WIN), min(len(para), m.end() + CTX_WIN)
        cc = bank.ctx_fp(para[lo:hi], exclude=ent)
        cg = gpt_ctx(para[lo:hi], exclude=ent)
        if cc is None or cg is None:
            continue
        rows_c.append(F.normalize(bank.fp([ent])[0] + cc, dim=-1))
        rows_g.append(F.normalize(gpt_word(ent) + cg, dim=-1))
        slot_val.append(ent)
    Kc = torch.stack(rows_c, 0)
    Kg = torch.stack(rows_g, 0)
    log(f"  memory slots={len(slot_val)} (facts={len(facts)}) ({time.time()-t0:.0f}s)")

    allv = list(dict.fromkeys(slot_val))

    def score_recall(Kmat, qfn, p):
        nrng = random.Random(SEED + 31 + int(p * 100))
        ok = 0
        for fdict in facts:
            q = qfn(noisy(fdict["S"], p, nrng))
            sc = {}
            for j, s in enumerate((Kmat @ q).tolist()):
                sc[slot_val[j]] = max(sc.get(slot_val[j], -9.9), s)
            others = [x for x in allv if x != fdict["value"]]
            nrng.shuffle(others)
            cands = [fdict["value"]] + others[:3]
            order = list(range(4))
            nrng.shuffle(order)
            shuf = [cands[i] for i in order]
            ok += int(int(np.argmax([sc.get(c, -9.9) for c in shuf])) == order.index(0))
        return ok / len(facts)

    B = {"curve": {}, "rag": {}}
    for p in LEVELS:
        B["curve"][p] = score_recall(Kc, lambda w: bank.fp([w])[0], p)
        B["rag"][p] = score_recall(Kg, lambda w: gpt_word(w), p)
        log(f"  B p={p:.1f}: curve={B['curve'][p]:.3f} rag={B['rag'][p]:.3f} ({time.time()-t0:.0f}s)")

    # ---------- B2 (hardened): noise in the STORED text too, 8-way, independent noise realizations ----------
    B2_LEVELS = [0.0, 0.2, 0.3]
    N_FILLER_B2 = 400
    WAYS2 = 8

    def noise_text(t, p, nrng):
        return WORD_RE.sub(lambda m: noisy(m.group(0), p, nrng), t)

    B2 = {"curve": {}, "rag": {}}
    for p in B2_LEVELS:
        nrng = random.Random(SEED + 71 + int(p * 100))
        rc, rg, sv = [], [], []
        for fdict in facts:  # stored text is corrupted; anchor is the corrupted subject in that text
            S_store = noisy(fdict["S"], p, nrng)
            txt = noise_text(fdict["text"].replace(fdict["S"], S_store), p, nrng)
            cc = bank.ctx_fp(txt, exclude=fdict["value"])
            cg = gpt_ctx(txt, exclude=fdict["value"])
            kc, kg = bank.fp([S_store])[0], gpt_word(S_store)
            rc.append(F.normalize(kc + cc, dim=-1) if cc is not None else kc)
            rg.append(F.normalize(kg + cg, dim=-1) if cg is not None else kg)
            sv.append(fdict["value"])
        for para in paras[:N_FILLER_B2]:
            m = ENT_RE.search(para)
            if not m:
                continue
            ent = m.group(1)
            lo, hi = max(0, m.start() - CTX_WIN), min(len(para), m.end() + CTX_WIN)
            txt = noise_text(para[lo:hi], p, nrng)
            cc = bank.ctx_fp(txt, exclude=ent)
            cg = gpt_ctx(txt, exclude=ent)
            if cc is None or cg is None:
                continue
            rc.append(F.normalize(bank.fp([noisy(ent, p, nrng)])[0] + cc, dim=-1))
            rg.append(F.normalize(gpt_word(noisy(ent, p, nrng)) + cg, dim=-1))
            sv.append(ent)
        K2c, K2g = torch.stack(rc, 0), torch.stack(rg, 0)
        allv2 = list(dict.fromkeys(sv))

        def score2(Kmat, qfn, seed_off):
            qrng = random.Random(SEED + 91 + seed_off + int(p * 100))
            ok = 0
            for fdict in facts:
                q = qfn(noisy(fdict["S"], p, qrng))  # independent noise realization
                sc = {}
                for j, s in enumerate((Kmat @ q).tolist()):
                    sc[sv[j]] = max(sc.get(sv[j], -9.9), s)
                others = [x for x in allv2 if x != fdict["value"]]
                qrng.shuffle(others)
                cands = [fdict["value"]] + others[: WAYS2 - 1]
                order = list(range(len(cands)))
                qrng.shuffle(order)
                shuf = [cands[i] for i in order]
                ok += int(int(np.argmax([sc.get(c, -9.9) for c in shuf])) == order.index(0))
            return ok / len(facts)

        B2["curve"][p] = score2(K2c, lambda w: bank.fp([w])[0], 0)
        B2["rag"][p] = score2(K2g, lambda w: gpt_word(w), 1)
        log(f"  B2 p={p:.1f} (noisy corpus+query, {WAYS2}-way): curve={B2['curve'][p]:.3f} rag={B2['rag'][p]:.3f} ({time.time()-t0:.0f}s)")

    # ---------- C. mechanism: BPE fragmentation ----------
    frag = {}
    for p in LEVELS:
        nrng = random.Random(SEED + 5 + int(p * 100))
        pieces = [len(tok.encode(" " + noisy(w, p, nrng)).ids) for w in pool]
        frag[p] = float(np.mean(pieces))
    log(f"  C BPE pieces/word: {[f'{p}:{frag[p]:.2f}' for p in LEVELS]}")

    def drop(d):
        return (d[0.0] - d[0.3]) / max(1e-6, d[0.0])

    g_idA = all(A["curve_seen"][p] >= A["gpt_seen"][p] + 0.10 for p in (0.2, 0.3))
    g_oov = all(A["curve_oov"][p] >= A["gpt_oov"][p] + 0.10 for p in (0.2, 0.3))
    g_factB = B["curve"][0.3] >= B["rag"][0.3] + 0.10
    g_degrade = drop(B["curve"]) < drop(B["rag"])
    g_hard = B2["curve"][0.3] >= B2["rag"][0.3] + 0.10
    if (g_idA or g_oov) and g_factB and g_degrade and g_hard:
        overall = "NOISE_ROBUST_WIN"
    elif g_idA or g_oov or g_factB:
        overall = "NOISE_ROBUST_PARTIAL"
    else:
        overall = "NOISE_ROBUST_NO"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "noise_oov_robustness_204",
        "overall": overall,
        "A_identity_retrieval": {k: {str(p): v for p, v in d.items()} for k, d in A.items()},
        "B_fact_recall_noisy_query": {k: {str(p): v for p, v in d.items()} for k, d in B.items()},
        "B_relative_drop_p0_to_p03": {"curve": drop(B["curve"]), "rag": drop(B["rag"])},
        "B2_hardened_noisy_corpus_and_query_8way": {k: {str(p): v for p, v in d.items()} for k, d in B2.items()},
        "C_bpe_pieces_per_word": {str(p): v for p, v in frag.items()},
        "gates": {
            "g_id_seen": g_idA,
            "g_id_oov": g_oov,
            "g_fact": g_factB,
            "g_degrade": g_degrade,
            "g_hard_B2": g_hard,
        },
        "pool": {"seen": len(seen), "oov": len(oov)},
        "slots": len(slot_val),
        "chance_B": 0.25,
        "note": "rank-based metrics only (no cross-space cosine comparison); GPT+RAG uses the identical "
        "subject-anchored key/query recipe, only the encoder differs",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage204 — W1 noise/OOV robustness",
                "",
                f"**Overall:** `{overall}`",
                "",
                "| noise p | A id seen curve/gpt | A id OOV curve/gpt | B recall curve/rag | BPE pieces/word |",
                "|---------|---------------------|--------------------|--------------------|-----------------|",
            ]
            + [
                f"| {p:.1f} | {A['curve_seen'][p]:.3f} / {A['gpt_seen'][p]:.3f} "
                f"| {A['curve_oov'][p]:.3f} / {A['gpt_oov'][p]:.3f} "
                f"| {B['curve'][p]:.3f} / {B['rag'][p]:.3f} | {frag[p]:.2f} |"
                for p in LEVELS
            ]
            + [
                "",
                f"- relative drop p0→p0.3 (fact recall): curve {drop(B['curve']):.3f} vs rag {drop(B['rag']):.3f}",
                "- **B2 hardened (noise in stored corpus AND query, 8-way, chance 0.125):** "
                + " · ".join(f"p={p:.1f} curve {B2['curve'][p]:.3f} / rag {B2['rag'][p]:.3f}" for p in B2_LEVELS),
                f"- gates: id_seen={g_idA} id_oov={g_oov} fact={g_factB} degrade={g_degrade} hard_B2={g_hard}",
                f"- slots={len(slot_val)}, chance B=0.25",
            ]
        ),
        encoding="utf-8",
    )
    log(
        f"[204] {overall} | A0.3 curve seen={A['curve_seen'][0.3]:.3f} vs gpt={A['gpt_seen'][0.3]:.3f} | "
        f"B0.3 curve={B['curve'][0.3]:.3f} vs rag={B['rag'][0.3]:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
