"""
Stage 198 — streaming, beyond-window, update-heavy memory under budget.
The last piece for AND_DISTINCT: does online in-space write beat a rebuilt RAG index?

Regime: a long stream of write-events over time (>> attention window). Some subject
entities S get an initial fake value F1, then a LATER update F2 (overwrite). Many
boilerplate events (real entities) create budget pressure. Memory budget B < stream.
At end, query each fact entity by its subject anchor; want the CURRENT (latest) value.

Systems (all query by subject-anchor fp(S), best mech from 197):
  tape_gated   : online write, admission by fp-lexicon surprise (in-space, free), recency for updates
  rag_uniform  : GPT-embedding store, admission by recency/ingestion order (no in-space novelty prior)
  rag_novelty  : GPT-embedding store, admission by the SAME fp-surprise (bolted-on) — honesty control
  gpt_incontext: vanilla GPT sees only last window of the concatenated stream (beyond-window fail)

Gates:
  G_beyond  tape - gpt_incontext >= 0.30     (beats in-context; needs external memory)
  G_budget  tape - rag_uniform  >= 0.15      (in-space write policy > ingestion order at same budget)
  G_update  tape latest-value acc (updated) >= 0.60
  distinctness type:
    tape > rag_novelty + 0.05  -> STREAM_CAPABILITY_DISTINCT (RAG can't match even when handed the signal)
    else                       -> STREAM_ARCHITECTURAL_DISTINCT (capability reachable, but only by bolting
                                   the tape's own signal onto RAG; tape gets it built-in/free)

  python _stage198_stream_update.py
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import CTX_WIN, ENT_RE, WORD_RE, FpBank
from _stage196_tapelm import GptBank, load_gpt

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
DECISION = RES / "stage198_decision.json"
MINI = RES / "stage198_mini.md"
LOG = RES / "_stage198_log.txt"

SEED = 198
CORPUS_CHARS = 150_000_000
MID_START = 70_000_000
MID_CHARS = 4_000_000
MIN_COUNT = 2
MAX_LEX = 200_000
N_FACTS = 90
UPDATE_FRAC = 0.5
BOILER_MULT = 4
MAX_ARCS = 64


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def build_stream(paras, rng, fakes):
    """events: list of dicts {t, S, value, text, novel}; facts: {S -> latest value}."""
    subjects = []
    for p in paras:
        occ = defaultdict(list)
        for m in ENT_RE.finditer(p):
            occ[m.group(1)].append((m.start(), m.end()))
        for S, ps in occ.items():
            if len(S) >= 4 and len(ps) >= 1:
                s0, e0 = ps[0]
                win = p[max(0, s0 - CTX_WIN) : min(len(p), e0 + CTX_WIN)]
                if S in win and len(WORD_RE.findall(win)) >= 4:
                    subjects.append((S, win))
    rng.shuffle(subjects)
    subjects = subjects[:N_FACTS]

    fi = 0
    events, latest = [], {}
    for S, win in subjects:
        F1 = fakes[fi]
        fi += 1
        events.append({"S": S, "value": F1, "text": win.replace(S, S + " " + F1, 1), "novel": True})
        latest[S] = F1
    n_upd = int(len(subjects) * UPDATE_FRAC)
    for S, win in subjects[:n_upd]:
        F2 = fakes[fi]
        fi += 1
        events.append({"S": S, "value": F2, "text": win.replace(S, S + " " + F2, 1), "novel": True})
        latest[S] = F2

    # boilerplate: real-entity events (low fp-surprise), create budget pressure
    boiler = []
    for p in paras:
        if len(boiler) >= len(events) * BOILER_MULT:
            break
        m = ENT_RE.search(p)
        if not m:
            continue
        ent = m.group(1)
        lo, hi = max(0, m.start() - CTX_WIN), min(len(p), m.end() + CTX_WIN)
        boiler.append({"S": ent, "value": ent, "text": p[lo:hi], "novel": False})
    stream = events + boiler
    rng.shuffle(stream)
    for t, ev in enumerate(stream):
        ev["t"] = t
    return stream, latest, n_upd


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"Stage198 start {datetime.now(timezone.utc).isoformat()}")
    log("streaming beyond-window update-heavy memory under budget")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    t0 = time.time()

    flat, off, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    gm = load_gpt(device)
    bank = FpBank(model, stoi, device)
    gbank = GptBank(gm, tok, pad_id, device)
    log(f"models loaded ({time.time()-t0:.0f}s)")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read(CORPUS_CHARS)
    mid = text[MID_START : MID_START + MID_CHARS]
    mid_paras = [p.strip() for p in mid.split("\n") if 120 < len(p.strip()) < 1000][:2000]
    words = re.findall(r"[A-Za-z][a-z]+", text)
    del text
    cnt = Counter(words)
    word_set = set(cnt.keys())
    lex_words = [w for w, c in cnt.most_common(MAX_LEX) if c >= MIN_COUNT]
    fps = []
    for i in range(0, len(lex_words), 4096):
        fps.append(bank.fp(lex_words[i : i + 4096]))
    lex = torch.cat(fps, 0)
    log(f"mid_paras={len(mid_paras)} lexicon={len(lex_words)} ({time.time()-t0:.0f}s)")

    fakes = gen_fakes(word_set, rng, N_FACTS * 3)
    stream, latest, n_upd = build_stream(mid_paras, rng, fakes)
    budget = sum(1 for e in stream if e["novel"])  # room only for the fact-events if chosen well
    log(f"stream={len(stream)} fact_events(budget)={budget} updated_entities={n_upd} query_entities={len(latest)}")

    # fp-surprise of each event's value word (admission signal)
    vals = [e["value"] for e in stream]
    sur = (1.0 - (bank.fp(vals) @ lex.T).max(dim=-1).values).cpu().numpy()
    for e, s in zip(stream, sur):
        e["sur"] = float(s)

    # subject-anchor blend key/query (M3, best in 197)
    def key_curve(e):
        c = bank.ctx_fp(e["text"], exclude=e["value"])
        base = bank.fp([e["S"]])[0]
        return F.normalize(base + c, dim=-1) if c is not None else base

    def query_curve(S):
        return bank.fp([S])[0]

    # ---- admission policies ----
    def admit(policy):
        if policy == "gated":
            keep = sorted(stream, key=lambda e: -e["sur"])[:budget]
        elif policy == "uniform":  # recency / ingestion order
            keep = sorted(stream, key=lambda e: -e["t"])[:budget]
        return keep

    def eval_curve(policy):
        keep = admit(policy)
        krows = [(key_curve(e), e["value"], e["t"]) for e in keep]
        ok = n = 0
        allv = list({e["value"] for e in stream if e["novel"]})
        for S, gold in latest.items():
            q = query_curve(S)
            sc = {}
            for k, v, t in krows:
                s = float(k @ q) + 1e-3 * (t / len(stream))  # recency tiebreak
                sc[v] = max(sc.get(v, -9.9), s)
            others = [x for x in allv if x != gold]
            rng.shuffle(others)
            cands = [gold] + others[:3]
            order = list(range(len(cands)))
            rng.shuffle(order)
            shuf = [cands[i] for i in order]
            g = order.index(0)
            ok += int(int(np.argmax([sc.get(c, -9.9) for c in shuf])) == g)
            n += 1
        return ok / max(1, n)

    # fair GPT+RAG: mirror the tape's key/query EXACTLY, only the encoder differs (GPT vs curve)
    @torch.no_grad()
    def gpt_emb(ids):
        ids = [i for i in ids if i != pad_id][-MAX_ARCS:]
        if not ids:
            return None
        x = torch.tensor([ids], device=device)
        h = gm.transformer(input_ids=x).last_hidden_state[0].mean(0)
        return F.normalize(h, dim=-1)

    def gpt_word_fp(w):
        return gpt_emb(tok.encode(" " + w).ids)

    def gpt_ctx_fp(textv, exclude=None):
        ws = [w for w in WORD_RE.findall(textv) if w != exclude][:40]
        if len(ws) < 3:
            return None
        return gpt_emb(tok.encode(" ".join(ws)).ids)

    def key_rag(e):
        base = gpt_word_fp(e["S"])
        c = gpt_ctx_fp(e["text"], exclude=e["value"])
        if base is None:
            return c
        return F.normalize(base + c, dim=-1) if c is not None else base

    def query_rag(S):
        return gpt_word_fp(S)

    def eval_rag(policy):
        keep = admit(policy)
        krows = []
        for e in keep:
            k = key_rag(e)
            if k is not None:
                krows.append((k, e["value"], e["t"]))
        ok = n = 0
        allv = list({e["value"] for e in stream if e["novel"]})
        for S, gold in latest.items():
            q = query_rag(S)
            if q is None:
                continue
            sc = {}
            for k, v, t in krows:
                s = float(k @ q) + 1e-3 * (t / len(stream))
                sc[v] = max(sc.get(v, -9.9), s)
            others = [x for x in allv if x != gold]
            rng.shuffle(others)
            cands = [gold] + others[:3]
            order = list(range(len(cands)))
            rng.shuffle(order)
            shuf = [cands[i] for i in order]
            g = order.index(0)
            ok += int(int(np.argmax([sc.get(c, -9.9) for c in shuf])) == g)
            n += 1
        return ok / max(1, n)

    @torch.no_grad()
    def eval_gpt_incontext():
        # vanilla GPT: sees only the last window of the concatenated stream text
        big = " ".join(e["text"] for e in sorted(stream, key=lambda e: e["t"]))
        tail_ids = [i for i in tok.encode(big).ids if i != pad_id][-MAX_ARCS + 6 :]
        ok = n = 0
        allv = list({e["value"] for e in stream if e["novel"]})
        for S, gold in latest.items():
            cue = [i for i in tok.encode(" " + S + " is").ids if i != pad_id]
            ctx = (tail_ids + cue)[-MAX_ARCS:]
            others = [x for x in allv if x != gold]
            rng.shuffle(others)
            cands = [gold] + others[:3]
            order = list(range(len(cands)))
            rng.shuffle(order)
            shuf = [cands[i] for i in order]
            g = order.index(0)
            sc = []
            for c in shuf:
                cid = [i for i in tok.encode(" " + c).ids if i != pad_id]
                seq = (ctx + cid)[-MAX_ARCS:]
                nc = len(seq) - len(cid)
                x = torch.tensor([seq], device=device)
                logp = F.log_softmax(gm(input_ids=x).logits[0], dim=-1)
                sc.append(sum(float(logp[nc + k - 1, tt]) for k, tt in enumerate(cid)) / max(1, len(cid)))
            ok += int(int(np.argmax(sc)) == g)
            n += 1
        return ok / max(1, n)

    tape_gated = eval_curve("gated")
    rag_uniform = eval_rag("uniform")
    rag_novelty = eval_rag("gated")  # RAG handed the SAME fp-surprise admission (honesty control)
    gpt_ic = eval_gpt_incontext()
    log(f"  tape_gated={tape_gated:.3f} rag_uniform={rag_uniform:.3f} rag_novelty={rag_novelty:.3f} gpt_incontext={gpt_ic:.3f} ({time.time()-t0:.0f}s)")

    # update-handling: latest-value acc on UPDATED entities only (tape)
    updated_S = [e["S"] for e in stream if e["novel"]]
    updated_S = [S for S in latest if updated_S.count(S) >= 2 or True][:n_upd]  # first n_upd updated
    # recompute tape acc restricted to updated entities
    keep = admit("gated")
    krows = [(key_curve(e), e["value"], e["t"]) for e in keep]
    upd_set = set()
    seen = defaultdict(int)
    for e in sorted(stream, key=lambda e: e["t"]):
        if e["novel"]:
            seen[e["S"]] += 1
    upd_entities = [S for S, c in seen.items() if c >= 2]
    ok = n = 0
    allv = list({e["value"] for e in stream if e["novel"]})
    for S in upd_entities:
        gold = latest[S]
        q = query_curve(S)
        sc = {}
        for k, v, t in krows:
            s = float(k @ q) + 1e-3 * (t / len(stream))
            sc[v] = max(sc.get(v, -9.9), s)
        others = [x for x in allv if x != gold]
        rng.shuffle(others)
        cands = [gold] + others[:3]
        order = list(range(len(cands)))
        rng.shuffle(order)
        shuf = [cands[i] for i in order]
        g = order.index(0)
        ok += int(int(np.argmax([sc.get(c, -9.9) for c in shuf])) == g)
        n += 1
    tape_update = ok / max(1, n)
    log(f"  tape latest-value acc on updated entities={tape_update:.3f} (n={n})")

    g_beyond = tape_gated - gpt_ic >= 0.30
    g_budget = tape_gated - rag_uniform >= 0.15
    g_update = tape_update >= 0.60
    if g_beyond and g_budget and g_update:
        if tape_gated > rag_novelty + 0.05:
            overall = "STREAM_CAPABILITY_DISTINCT"
        else:
            overall = "STREAM_ARCHITECTURAL_DISTINCT"
    elif g_beyond and g_update:
        overall = "STREAM_BEYOND_WINDOW_ONLY"
    else:
        overall = "STREAM_PARTIAL"

    gates = {"g_beyond": g_beyond, "g_budget": g_budget, "g_update": g_update}
    R = {
        "tape_gated": tape_gated,
        "rag_uniform": rag_uniform,
        "rag_novelty_bolted": rag_novelty,
        "gpt_incontext": gpt_ic,
        "tape_update_latest": tape_update,
        "budget": budget,
        "stream_len": len(stream),
        "updated_entities": len(upd_entities),
        "chance": 0.25,
    }
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "stream_update_198",
        "overall": overall,
        "gates": gates,
        "results": R,
        "note": "online in-space write (fp-surprise admission + subject-anchor + recency) vs rebuilt RAG index; "
        "rag_novelty control quantifies whether distinctness is capability or architecture",
    }
    DECISION.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    MINI.write_text(
        "\n".join(
            [
                "# Stage198 — streaming / beyond-window / update-heavy under budget",
                "",
                f"**Overall:** `{overall}`",
                "",
                f"- tape (gated, in-space): **{tape_gated:.3f}**",
                f"- rag_uniform (ingestion order): {rag_uniform:.3f}",
                f"- rag_novelty (RAG + bolted fp-surprise): {rag_novelty:.3f}",
                f"- gpt_incontext (beyond-window): {gpt_ic:.3f}  (chance 0.25)",
                f"- tape latest-value on updated entities: {tape_update:.3f}",
                "",
                f"budget={budget}, stream={len(stream)}, updated={len(upd_entities)}",
                f"gates: {gates}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[198] {overall} | tape={tape_gated:.3f} rag_u={rag_uniform:.3f} rag_n={rag_novelty:.3f} gpt={gpt_ic:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
