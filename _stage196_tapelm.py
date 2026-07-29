"""
Stage 196 — TapeLM: assemble variant A into ONE inference stack on the frozen
191-P1 curve encoder, and enforce the anti-clone (distinguishability) gate.

Single object `TapeLM` over frozen P1 does all three on ONE held-out slice (exam v3):
  1. GENERATE  : next-piece CE log-prob  (parity = entry ticket, must TIE GPT)
  2. RECALL    : FP episodic fact memory  (194)   — win axis
  3. CALIBRATE : FP-lexicon lexical surprise (192) — win axis (OOD "I don't know")
  4. EDIT      : one-shot knowledge write at read time — win axis (GPT structurally can't)

Anti-clone rule (plan North-star v2): parity is the entry ticket, NOT a win.
Win only counts where BPE-GPT is structurally weak, and there curve must BEAT, not tie.
Nearest real rival = GPT+RAG (same retrieval math, GPT's own embedding as key).

Controls:
  - GPT-XL (191-P2): next_tok parity, parametric entity recall, native BPE OOD surprisal.
  - GPT+RAG        : identical retrieval over same read paras, keyed by GPT mean-pool embedding.

Verdicts:
  parity_hold        = |curve_next - gpt_next| <= 0.03
  recall_win         = curve_recall >= 0.50 and curve_recall > gpt_param + 0.15
  recall_beats_rag   = curve_recall >= gpt_rag - 0.03            (not worse than RAG)
  calib_win          = curve_lexAUC > gpt_bpeAUC and curve_lexAUC >= 0.80
  edit_win           = curve_edit >= 0.50 and curve_edit > gpt_edit + 0.20
  overall:
    parity_hold & recall_win & recall_beats_rag & calib_win & edit_win -> TAPELM_COMPOSES_AND_DISTINCT
    parity_hold & recall_win & calib_win & edit_win & !recall_beats_rag -> TAPELM_DISTINCT_RECALL_RAG_EQUIV
    parity_hold & !(any strict beat over BOTH gpt & gpt+rag)           -> TAPELM_CLONE_RISK
    else                                                              -> TAPELM_PARTIAL

  python _stage196_tapelm.py
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data, score_items, span_logprob_x
from _stage192_fp_lexicon import auc, gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank, build_memory, score_entity_items

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_GPT = Path("checkpoints/stage191_p2_gpt.pt")
WIKI = Path("data/_wikitext103_train.txt")
EXAM_V3 = Path("data/stage191_exam_v3.jsonl")
DECISION = RES / "stage196_decision.json"
MINI = RES / "stage196_mini.md"
LOG = RES / "_stage196_log.txt"

SEED = 196
CORPUS_CHARS = 150_000_000
EXAM_TAIL_CHARS = 3_000_000
MIN_COUNT = 2
MAX_LEX = 200_000
N_FAKE = 150
N_EDIT = 60
MAX_CHARS = s177.MAX_CHARS_PER_ARC


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


# ------------------------------------------------------------------ TapeLM
class TapeLM:
    """One inference object over the frozen P1 curve encoder: generate / recall / calibrate / edit."""

    def __init__(self, model, stoi, tok, char_table, pad_id, device):
        self.model = model
        self.tok = tok
        self.char_table = char_table
        self.pad_id = pad_id
        self.device = device
        self.bank = FpBank(model, stoi, device)  # shared fp-space for ALL of recall/calib/edit
        self.K = None
        self.vals = None
        self.lex = None
        self.lex_words = None

    # ---- 1. generation (parity entry ticket) ----
    def next_tok_score(self, ctx_ids, cand_ids) -> float:
        return span_logprob_x(self.model, self.char_table, self.pad_id, ctx_ids, cand_ids, self.device)

    # ---- 2. episodic fact memory (read then recall) ----
    def read(self, paras, tag="read"):
        self.K, self.vals = build_memory(paras, self.bank, tag)

    # ---- 3. lexical calibration ----
    @torch.no_grad()
    def build_lexicon(self, lex_words):
        self.lex_words = lex_words
        fps = []
        for i in range(0, len(lex_words), 4096):
            fps.append(self.bank.fp(lex_words[i : i + 4096]))
        self.lex = torch.cat(fps, 0)

    @torch.no_grad()
    def lex_surprise(self, words) -> np.ndarray:
        fps = self.bank.fp(words)
        best = (fps @ self.lex.T).max(dim=-1).values
        return (1.0 - best).cpu().numpy()

    # ---- 4. one-shot knowledge edit ----
    @torch.no_grad()
    def write_fact(self, ctx_text, value, exclude=None):
        k = self.bank.ctx_fp(ctx_text, exclude=exclude)
        if k is None:
            return False
        self.edit_K = getattr(self, "edit_K", [])
        self.edit_V = getattr(self, "edit_V", [])
        self.edit_K.append(k)
        self.edit_V.append(value)
        return True

    @torch.no_grad()
    def recall_fact(self, query_text, cands, exclude=None):
        q = self.bank.ctx_fp(query_text, exclude=exclude)
        if q is None or not getattr(self, "edit_K", []):
            return None
        Kx = torch.stack(self.edit_K, 0)
        sims = Kx @ q
        best = {}
        for i, v in enumerate(self.edit_V):
            best[v] = max(best.get(v, -1.0), float(sims[i]))
        return int(np.argmax([best.get(c, -1.0) for c in cands]))


# ------------------------------------------------------------------ GPT bank (RAG control)
class GptBank:
    """GPT+RAG control: same retrieval math as FpBank, key = GPT mean-pool embedding."""

    def __init__(self, gm, tok, pad_id, device):
        self.gm = gm
        self.tok = tok
        self.pad_id = pad_id
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def ctx_fp(self, text, exclude=None):
        ws = [w for w in WORD_RE.findall(text) if w != exclude][:40]
        if len(ws) < 3:
            return None
        key = " ".join(ws)
        if key in self.cache:
            return self.cache[key]
        ids = [i for i in self.tok.encode(key).ids if i != self.pad_id][-MAX_ARCS:]
        if not ids:
            return None
        x = torch.tensor([ids], device=self.device)
        h = self.gm.transformer(input_ids=x).last_hidden_state[0].mean(0)
        v = F.normalize(h, dim=-1)
        self.cache[key] = v
        return v


def load_gpt(device):
    ck = torch.load(CKPT_GPT, map_location=device, weights_only=False)
    gm = GPT2LMHeadModel(GPT2Config(**ck["conf"])).to(device)
    gm.load_state_dict(ck["model"])
    gm.eval()
    return gm


def gpt_span(gm, device, ctx, cand) -> float:
    seq = (ctx + cand)[-MAX_ARCS:]
    n_ctx = len(seq) - len(cand)
    x = torch.tensor([seq], device=device)
    with torch.no_grad():
        logp = F.log_softmax(gm(input_ids=x).logits[0], dim=-1)
    return sum(float(logp[n_ctx + k - 1, tid]) for k, tid in enumerate(cand)) / max(1, len(cand))


@torch.no_grad()
def gpt_word_surprisal(gm, tok, device, words) -> np.ndarray:
    """GPT's native OOD signal: mean per-piece surprisal of the word given a neutral prefix."""
    prefix = [i for i in tok.encode(" The ").ids][-8:]
    out = []
    for w in words:
        wid = [i for i in tok.encode(" " + w).ids]
        if not wid:
            out.append(0.0)
            continue
        seq = prefix + wid
        x = torch.tensor([seq], device=device)
        logp = F.log_softmax(gm(input_ids=x).logits[0], dim=-1)
        n = len(prefix)
        s = -np.mean([float(logp[n + k - 1, tid]) for k, tid in enumerate(wid)])
        out.append(float(s))
    return np.array(out)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage196 start {datetime.now(timezone.utc).isoformat()}")
    log("TapeLM assembled stack + anti-clone distinguishability gate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    gm = load_gpt(device)
    log(f"models loaded ({time.time()-t0:.0f}s)")

    tape = TapeLM(model, stoi, tok, char_table, pad_id, device)

    items = [json.loads(l) for l in EXAM_V3.read_text(encoding="utf-8").splitlines()]
    nt_items = [it for it in items if it["type"] == "next_tok"]
    ent_items = [it for it in items if it["type"] == "entity"]
    log(f"exam: next_tok={len(nt_items)} entity={len(ent_items)}")

    # corpus slices
    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    tail = text[-EXAM_TAIL_CHARS:]
    tail_paras = [p.strip() for p in tail.split("\n") if 120 < len(p.strip()) < 1000][:1200]
    mid = text[60_000_000 : 60_000_000 + EXAM_TAIL_CHARS]
    mid_paras = [p.strip() for p in mid.split("\n") if 120 < len(p.strip()) < 1000][:600]

    import re
    from collections import Counter

    words = re.findall(r"[A-Za-z][a-z]+", text[:CORPUS_CHARS])
    del text
    cnt = Counter(words)
    word_set = set(cnt.keys())
    lex_words = [w for w, c in cnt.most_common(MAX_LEX) if c >= MIN_COUNT]
    log(f"tail_paras={len(tail_paras)} mid_paras={len(mid_paras)} lexicon={len(lex_words)} ({time.time()-t0:.0f}s)")

    R = {}

    # ---------- AXIS 1: GENERATION PARITY (entry ticket) ----------
    curve_nt = score_items(tape.next_tok_score, nt_items, "next_tok")["next_tok_acc"]
    gpt_nt = score_items(lambda c, cd: gpt_span(gm, device, c, cd), nt_items, "next_tok")["next_tok_acc"]
    R["parity"] = {"curve_next_tok": curve_nt, "gpt_next_tok": gpt_nt, "delta": curve_nt - gpt_nt}
    log(f"[1 parity] curve={curve_nt:.3f} gpt={gpt_nt:.3f} d={curve_nt-gpt_nt:+.3f} ({time.time()-t0:.0f}s)")

    # ---------- AXIS 2: FACT RECALL (curve vs GPT-parametric vs GPT+RAG) ----------
    tape.read(tail_paras, "read-tail")
    curve_rec = score_entity_items(items, tok, pad_id, tape.bank, tape.K, tape.vals)["acc"]
    gpt_param = score_items(lambda c, cd: gpt_span(gm, device, c, cd), ent_items, "entity")["entity_acc"]
    gbank = GptBank(gm, tok, pad_id, device)
    K_g, vals_g = build_memory(tail_paras, gbank, "gpt-rag")
    gpt_rag = score_entity_items(items, tok, pad_id, gbank, K_g, vals_g)["acc"]
    R["recall"] = {"curve_fp": curve_rec, "gpt_parametric": gpt_param, "gpt_rag": gpt_rag, "chance": 0.25}
    log(f"[2 recall] curve_fp={curve_rec:.3f} gpt_param={gpt_param:.3f} gpt_rag={gpt_rag:.3f} ({time.time()-t0:.0f}s)")

    # ---------- AXIS 3: OOD CALIBRATION (curve fp-lexicon vs GPT BPE surprisal) ----------
    tape.build_lexicon(lex_words)
    real_words = []
    for it in ent_items:
        s = tok.decode(it["cand_ids"][it["gold_idx"]], skip_special_tokens=False).strip()
        w = re.findall(r"[A-Za-z][a-z]+", s)
        if w and w[0] in cnt:
            real_words.append(w[0])
    real_words = list(dict.fromkeys(real_words))
    fakes = gen_fakes(word_set, rng, N_FAKE)
    curve_auc = auc(tape.lex_surprise(fakes), tape.lex_surprise(real_words))
    gpt_auc = auc(gpt_word_surprisal(gm, tok, device, fakes), gpt_word_surprisal(gm, tok, device, real_words))
    R["calibration"] = {"curve_lex_auc": curve_auc, "gpt_bpe_auc": gpt_auc, "n_real": len(real_words), "n_fake": len(fakes)}
    log(f"[3 calib] curve_lexAUC={curve_auc:.3f} gpt_bpeAUC={gpt_auc:.3f} ({time.time()-t0:.0f}s)")

    # ---------- AXIS 4: ONE-SHOT KNOWLEDGE EDIT (curve acquires at read time; GPT cannot) ----------
    edit_items = []
    used_fakes = gen_fakes(word_set, rng, N_EDIT * 2)
    fi = 0
    for p in mid_paras:
        if len(edit_items) >= N_EDIT or fi >= len(used_fakes):
            break
        m = ENT_RE.search(p)
        if not m:
            continue
        ent = m.group(1)
        F_new = used_fakes[fi]
        fi += 1
        passage = p.replace(ent, F_new)
        half = len(passage) // 2
        w1 = passage[:half]
        w2 = passage[half:]
        if F_new not in w1 or len(WORD_RE.findall(w2)) < 4:
            continue
        edit_items.append({"write": w1, "query": w2, "F": F_new})
    log(f"knowledge-edit items={len(edit_items)}")

    # curve: one-shot write from w1, retrieve from disjoint w2
    tape.edit_K, tape.edit_V = [], []
    all_F = [e["F"] for e in edit_items]
    for e in edit_items:
        tape.write_fact(e["write"], e["F"], exclude=e["F"])
    ok_c = n_c = 0
    for e in edit_items:
        others = [f for f in all_F if f != e["F"]]
        rng.shuffle(others)
        cands = [e["F"]] + others[:3]
        order = list(range(len(cands)))
        rng.shuffle(order)
        shuffled = [cands[k] for k in order]
        gold = order.index(0)
        pred = tape.recall_fact(e["query"], shuffled, exclude=e["F"])
        if pred is None:
            continue
        ok_c += int(pred == gold)
        n_c += 1
    curve_edit = ok_c / max(1, n_c)

    # GPT: cannot ingest the write; score fake candidates by logprob given w2 (structurally chance)
    ok_g = n_g = 0
    for e in edit_items:
        others = [f for f in all_F if f != e["F"]]
        rng.shuffle(others)
        cands = [e["F"]] + others[:3]
        order = list(range(len(cands)))
        rng.shuffle(order)
        shuffled = [cands[k] for k in order]
        gold = order.index(0)
        ctx_ids = [i for i in tok.encode(e["query"]).ids if i != pad_id][-MAX_ARCS:]
        sc = [gpt_span(gm, device, ctx_ids, [i for i in tok.encode(" " + c).ids if i != pad_id]) for c in shuffled]
        ok_g += int(int(np.argmax(sc)) == gold)
        n_g += 1
    gpt_edit = ok_g / max(1, n_g)
    R["edit"] = {"curve": curve_edit, "gpt": gpt_edit, "n": n_c, "chance": 0.25}
    log(f"[4 edit] curve={curve_edit:.3f} gpt={gpt_edit:.3f} (n={n_c}) ({time.time()-t0:.0f}s)")

    # ---------- VERDICT ----------
    parity_hold = abs(R["parity"]["delta"]) <= 0.03
    recall_win = curve_rec >= 0.50 and curve_rec > gpt_param + 0.15
    recall_beats_rag = curve_rec >= gpt_rag - 0.03
    calib_win = curve_auc > gpt_auc and curve_auc >= 0.80
    edit_win = curve_edit >= 0.50 and curve_edit > gpt_edit + 0.20
    strict_beat_any = (curve_rec > gpt_param + 0.15 and curve_rec > gpt_rag) or (curve_auc > gpt_auc) or (curve_edit > gpt_edit + 0.20)

    if parity_hold and recall_win and recall_beats_rag and calib_win and edit_win:
        overall = "TAPELM_COMPOSES_AND_DISTINCT"
    elif parity_hold and recall_win and calib_win and edit_win and not recall_beats_rag:
        overall = "TAPELM_DISTINCT_RECALL_RAG_EQUIV"
    elif parity_hold and not strict_beat_any:
        overall = "TAPELM_CLONE_RISK"
    else:
        overall = "TAPELM_PARTIAL"

    gates = {
        "parity_hold": parity_hold,
        "recall_win": recall_win,
        "recall_beats_rag": recall_beats_rag,
        "calib_win": calib_win,
        "edit_win": edit_win,
    }
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "tapelm_196",
        "overall": overall,
        "gates": gates,
        "axes": R,
        "note": "one frozen P1 encoder; generation + fp memory + fp lexicon + one-shot edit share one fp-space; "
        "anti-clone: win only where GPT structurally weak, curve must beat GPT AND GPT+RAG",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage196 — TapeLM assembled stack + anti-clone gate",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- **parity (entry ticket):** curve {curve_nt:.3f} vs gpt {gpt_nt:.3f} (Δ{curve_nt-gpt_nt:+.3f}) — hold={parity_hold}",
                f"- **recall (win):** curve_fp {curve_rec:.3f} vs gpt_param {gpt_param:.3f} vs **gpt+rag {gpt_rag:.3f}** (chance 0.25)",
                f"- **calibration (win):** curve_lexAUC {curve_auc:.3f} vs gpt_bpeAUC {gpt_auc:.3f}",
                f"- **one-shot edit (win):** curve {curve_edit:.3f} vs gpt {gpt_edit:.3f} (n={n_c}, chance 0.25)",
                "",
                f"gates: {gates}",
                "",
                "One frozen curve encoder serves generation + fact memory + lexical calibration + one-shot edit "
                "from a single shared fp-space. Win counted only where BPE-GPT is structurally weak (recall/calib/edit), "
                "with GPT+RAG as the nearest rival control.",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[196] {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
