"""
Stage 267 — Read, refine, read again: the mind's job is the loop, not the projection.

266 closed the single-shot question. A learned query vector loses to plain words on every trunk
(top1 0.062 vs 0.199); LLM keywords do not beat the question's own words; blind paraphrase wakes
4 silent queries out of 176. The cause was named in 266's own data: the model invents words that
exist somewhere on the tape (novel_on_tape 0.756) but not beside the fact being asked for. It
cannot bridge because it does not know what is written.

But after one retrieval it does know. That is the whole idea here:

    hop 0   question words          -> votes -> top-k slots
    read    the write-contexts of those slots, verbatim tape text
    hop 1   words chosen while LOOKING at that text -> votes again

The new words are taken from the tape rather than guessed at it. This is 257's mechanism — the
second hop anchored on what the first hop returned — carried over from the composition exam to
the open-bank query. Nothing in memory changes: same bank, same postings, same idf, zero trained
parameters at read time.

Arms, so that "the loop helps" cannot be confused with "prompting helps":

    A  hop0            surface words only                    (= 264's votes arm, validity anchor)
    B  refine_grounded refine while reading the RETRIEVED passages
    C  refine_random   refine while reading RANDOM passages   (causal control for grounding)
    D  refine_selective grounded, but only where hop0 is uncertain
    E  refine_blind    refine with no passages at all         (= 266's paraphrase, reference)

C is the gate that matters. If B ≈ C the loop is not reading the tape, it is just being prompted,
and 267 is a NO however good B looks. D exists because 266 taught that adding words to an already
healthy query hurts it — union lost 0.199 -> 0.165 — so refinement has to be spent only where hop0
is in trouble, and "in trouble" must be computable without the gold slot.

Silence is scored on the gold vote mass, and rank counts a silent gold as last, not first: 266's
paraphrase arm read top1 0.477 purely because 71 empty answers left an empty score table and the
old formula called that rank 1.

  python _stage267_read_refine.py [--smoke]
  python _stage267_read_refine.py --k-read 5 --hops 2
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage24x_lib as L
from _stage191_night import SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE
from _stage261_nl_query import WORD_RE, collect, ctx_words, jaccard
from _stage262_trunk_swap import ExternalTrunk
from _tape_index import context_words

RES = Path("results")
DECISION = RES / "stage267_decision.json"
MINI = RES / "stage267_mini.md"
LOG = RES / "_stage267_log.txt"
DECISION_264 = RES / "stage264_decision.json"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 261  # same exam construction as 261/266

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

REFINE_PROMPT = (
    "Text fragment:\n{q}\n\n"
    "Passages retrieved from an index (some may be irrelevant):\n{passages}\n\n"
    "List 3-8 English content words that would most likely appear in the passage that actually "
    "continues the fragment. Prefer words you can see in the passages above when they look "
    "relevant to the fragment. Comma-separated words only — no sentences."
)
BLIND_PROMPT = (
    "Text fragment:\n{q}\n\n"
    "List 3-8 English content words that would most likely appear in the passage that actually "
    "continues the fragment. Comma-separated words only — no sentences."
)


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def ensure_short_hf_home() -> str | None:
    """Windows + long HF hub paths → OSError Errno 22. Prefer a short HF_HOME."""
    if os.name != "nt":
        return None
    cur = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cur and len(Path(cur).resolve().as_posix()) < 40:
        return cur
    short = Path(os.environ.get("SOTE_HF_HOME", r"C:\hf"))
    try:
        short.mkdir(parents=True, exist_ok=True)
    except OSError:
        short = Path.cwd() / "hf"
        short.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(short)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(short / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(short / "transformers"))
    return str(short)


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chat_wrap(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_word_list(ext: ExternalTrunk, body: str, *, max_new: int = 48) -> list[str]:
    prompt = chat_wrap(ext.tok, body)
    enc = ext.tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(ext.device) for k, v in enc.items()}
    n_in = int(enc["input_ids"].shape[1])
    out = ext.model.generate(
        **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=ext.tok.eos_token_id
    )
    raw = ext.tok.decode(out[0][n_in:], skip_special_tokens=True)
    words, seen = [], set()
    for part in re.split(r"[,;\n|/]+", raw):
        for w in WORD_RE.findall(part):
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                words.append(lw)
    return words


# --------------------------------------------------------------------------------------
# voting, with silence scored as last place
# --------------------------------------------------------------------------------------
def vote(words, postings, idf) -> dict[int, float]:
    sc: dict[int, float] = defaultdict(float)
    for w in words:
        for cid in postings.get(w, ()):
            sc[cid] += idf.get(w, 0.0)
    return sc


def rank_of(sc: dict[int, float], gold: int, n_slots: int) -> int:
    """Silence is last, not first.

    266's paraphrase arm scored top1 0.477 because 71 queries produced no words, left `sc` empty,
    and `1 + len({v > 0})` called that rank 1. A gold slot with no votes is tied with every other
    unvoted slot; the pessimistic reading is the only honest one.
    """
    g = sc.get(gold, 0.0)
    if g <= 0.0:
        return n_slots
    return 1 + sum(1 for v in sc.values() if v > g)


def nway_strict(gold_score: float, others) -> bool:
    return all(gold_score > o for o in others)


def hop0_uncertainty(sc: dict[int, float]) -> tuple[bool, float]:
    """Is this query in trouble? Computable without the gold slot.

    Silent (nothing voted) or a flat top — those are the queries worth spending a refinement on.
    266's union arm lost top1 precisely by spending words on queries that were already fine.
    """
    if not sc:
        return True, 0.0
    top = sorted(sc.values(), reverse=True)
    if len(top) == 1:
        return False, 1.0
    margin = (top[0] - top[1]) / max(top[0], 1e-9)
    return margin < 0.15, margin


def score_arm(items, words_fn, postings, idf, n_slots, med, *, n_way=20) -> dict:
    wrng = random.Random(SEED + 5)
    ranks, nway, gold_scores, lows = [], [], [], []
    for it in items:
        sc = vote(words_fn(it), postings, idf)
        g = sc.get(it["slot"], 0.0)
        r = rank_of(sc, it["slot"], n_slots)
        ranks.append(r)
        gold_scores.append(g)
        pool = [j for j in wrng.sample(range(n_slots), min(n_way * 3, n_slots))
                if j != it["slot"]][: n_way - 1]
        nway.append(int(nway_strict(g, (sc.get(j, 0.0) for j in pool))))
        lows.append(it["overlap"] <= med)
    r = np.asarray(ranks, dtype=np.float64)
    silent = np.asarray([g <= 0.0 for g in gold_scores])
    low = np.asarray(lows)
    hit = r == 1

    def _m(mask, arr):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    return {
        "top1": float(hit.mean()),
        "mrr": float(np.mean(1.0 / r)),
        "median_rank": float(np.median(r)),
        "acc_20way": float(np.mean(nway)),
        "chance_20way": 1.0 / n_way,
        "tie_at_zero_frac": float(silent.mean()),
        "tie_at_zero_frac_low_overlap": _m(low, silent),
        "tie_at_zero_frac_high_overlap": _m(~low, silent),
        "top1_low_overlap": _m(low, hit),
        "top1_high_overlap": _m(~low, hit),
        "top1_low_overlap_given_vote": _m(low & ~silent, hit),
        "n": len(ranks),
        "n_tie_at_zero": int(silent.sum()),
        "n_low_overlap": int(low.sum()),
    }


def woken_frac(items, base_fn, new_fn, postings, idf) -> dict:
    """Queries the tape was silent on that the refinement actually gave vote mass to.

    This is the metric the mind has to move. It is not tie_at_zero on its own: waking a query with
    wrong words converts "no answer" into "wrong answer", so the rank of the woken ones is
    reported beside the count.
    """
    woken, woken_hit, woken_ranks = 0, 0, []
    n_slots = None
    for it in items:
        b = vote(base_fn(it), postings, idf)
        if b.get(it["slot"], 0.0) > 0.0:
            continue
        a = vote(new_fn(it), postings, idf)
        if a.get(it["slot"], 0.0) > 0.0:
            woken += 1
            r = 1 + sum(1 for v in a.values() if v > a[it["slot"]])
            woken_ranks.append(r)
            woken_hit += int(r == 1)
    n = max(1, len(items))
    return {
        "woken_n": woken,
        "woken_frac": woken / n,
        "woken_top1": (woken_hit / woken) if woken else float("nan"),
        "woken_median_rank": float(np.median(woken_ranks)) if woken_ranks else float("nan"),
    }


# --------------------------------------------------------------------------------------
# exam: identical construction to 261/266
# --------------------------------------------------------------------------------------
def build_exam(bank, lines, n_ent, n_dist, rng):
    cands = collect(lines, bank)
    ents = sorted(cands)[:n_ent]
    rng.shuffle(ents)
    vals, ctxw, items, write_ctxs = [], [], [], []

    for e in ents:
        occ = cands[e]
        a, b = occ[0], occ[1]
        wctx = a["line"][max(0, a["start"] - 140) : min(len(a["line"]), a["end"] + 140)]
        qtext = b["line"][max(0, b["start"] - 200) : b["start"]].strip()
        ws = context_words(wctx, exclude=e)
        qs = context_words(qtext, exclude=e)
        if len(ws) < 4 or len(qs) < 4:
            continue
        items.append({
            "ent": e, "slot": len(vals), "qtext": qtext, "wctx": wctx, "qwords": qs,
            "overlap": jaccard(ctx_words(wctx, e), ctx_words(qtext, e)),
        })
        vals.append(e)
        ctxw.append(ws)
        write_ctxs.append(wctx)

    n_exam = len(vals)
    used = set(vals)
    for ln in lines:
        if len(vals) >= n_exam + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = max(0, m.start() - 140), min(len(ln), m.end() + 140)
            ws = context_words(ln[lo:hi], exclude=e)
            if len(ws) < 4:
                continue
            vals.append(e)
            ctxw.append(ws)
            write_ctxs.append(ln[lo:hi])
            used.add(e)
            if len(vals) >= n_exam + n_dist:
                break

    postings: dict[str, list[int]] = defaultdict(list)
    for cid, ws in enumerate(ctxw):
        for w in ws:
            postings[w].append(cid)
    idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in postings}
    return vals, write_ctxs, items, n_exam, postings, idf


def passages_for(sc: dict[int, float], write_ctxs, k: int, rng=None, n_slots=0) -> list[str]:
    """Top-k retrieved write-contexts, or k random ones for the grounding control."""
    if rng is not None:
        ids = [rng.randrange(n_slots) for _ in range(k)]
    else:
        ids = [cid for cid, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]]
    return [write_ctxs[i][:300] for i in ids]


def published_264_surface() -> dict | None:
    if not DECISION_264.is_file():
        return None
    try:
        d = json.loads(DECISION_264.read_text(encoding="utf-8"))
        return (d.get("summary") or {}).get("retrieval", {}).get("votes")
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--entities", type=int, default=0)
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--k-read", type=int, default=5, help="passages shown to the model at hop 1")
    ap.add_argument("--n-way", type=int, default=20)
    ap.add_argument("--margin-thresh", type=float, default=0.15,
                    help="hop0 top1/top2 margin below which the selective arm refines")
    ap.add_argument("--no-blind", action="store_true", help="skip the E reference arm")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    hf_home = ensure_short_hf_home()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_ent = args.entities or (60 if args.smoke else 400)
    n_dist = args.distractor_slots or (400 if args.smoke else 4000)
    max_lines = 3000 if args.smoke else 25000

    log(f"Stage267 read-refine start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"model={args.model} k_read={args.k_read}" + (f" HF_HOME={hf_home}" if hf_home else ""))

    _, _, stoi, n_char = load_data()
    V = Tokenizer.from_file(str(s177.TOK_PATH)).get_vocab_size()
    curve = SelfModelXL(n_char, V).to(device)
    curve.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    curve.eval()
    for p in curve.parameters():
        p.requires_grad_(False)
    bank = FpBank(curve, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(3_000_000 if args.smoke else 20_000_000)
    lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400][:max_lines]
    vals, write_ctxs, items, n_exam, postings, idf = build_exam(bank, lines, n_ent, n_dist, rng)
    if len(items) < 16:
        log("  not enough exam pairs")
        return 1
    n_slots = len(vals)
    med = float(np.median([it["overlap"] for it in items]))
    log(f"  exam={n_exam} bank={n_slots} vocab={len(postings)} "
        f"postings={sum(len(v) for v in postings.values())} overlap_med={med:.3f}")

    # curve is only needed for the exam construction; the loop below is pure text + postings
    del curve
    free_cuda()

    # ---- A: hop 0 ----
    A = score_arm(items, lambda it: it["qwords"], postings, idf, n_slots, med, n_way=args.n_way)
    log(f"  A hop0 surface: {json.dumps(A)}")

    hop0 = {}
    for it in items:
        sc = vote(it["qwords"], postings, idf)
        unc, margin = hop0_uncertainty(sc)
        hop0[it["slot"]] = {"sc": sc, "uncertain": unc, "margin": margin}
    n_unc = sum(1 for v in hop0.values() if v["uncertain"])
    log(f"  hop0 uncertain (silent or margin<{args.margin_thresh}): {n_unc}/{len(items)}")

    log(f"\n== loading {args.model} ==")
    try:
        ext = ExternalTrunk(args.model, device)
    except Exception as e:  # noqa: BLE001
        log(f"  LOAD FAIL: {type(e).__name__}: {e}")
        DECISION.write_text(json.dumps({
            "stage": 267, "overall": "READ_REFINE_INVALID",
            "error": f"{type(e).__name__}: {e}", "model": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        return 1
    if not getattr(ext.tok, "chat_template", None):
        log("  FATAL: model has no chat_template — an Instruct run without it is a base run")
        DECISION.write_text(json.dumps({
            "stage": 267, "overall": "READ_REFINE_INVALID", "error": "missing_chat_template",
            "model": args.model, "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        return 1

    ctrl_rng = random.Random(SEED + 77)
    grounded: dict[int, list[str]] = {}
    randomed: dict[int, list[str]] = {}
    blind: dict[int, list[str]] = {}
    n_empty = {"grounded": 0, "random": 0, "blind": 0}
    samples = []

    for i, it in enumerate(items):
        sc = hop0[it["slot"]]["sc"]
        pas = passages_for(sc, write_ctxs, args.k_read)
        body = REFINE_PROMPT.format(q=it["qtext"], passages="\n---\n".join(pas))
        g = generate_word_list(ext, body)
        grounded[it["slot"]] = g
        n_empty["grounded"] += int(not g)

        pas_r = passages_for(None, write_ctxs, args.k_read, rng=ctrl_rng, n_slots=n_slots)
        body_r = REFINE_PROMPT.format(q=it["qtext"], passages="\n---\n".join(pas_r))
        r_ = generate_word_list(ext, body_r)
        randomed[it["slot"]] = r_
        n_empty["random"] += int(not r_)

        if not args.no_blind:
            b = generate_word_list(ext, BLIND_PROMPT.format(q=it["qtext"]))
            blind[it["slot"]] = b
            n_empty["blind"] += int(not b)

        if len(samples) < 5:
            samples.append({
                "qtext": it["qtext"][:120],
                "hop0_uncertain": hop0[it["slot"]]["uncertain"],
                "passages_head": [p[:90] for p in pas[:2]],
                "grounded": g[:8],
                "random_ctrl": r_[:8],
                "from_passages": [w for w in g if any(w in p.lower() for p in pas)][:8],
            })
        if (i + 1) % 25 == 0:
            log(f"    refined {i+1}/{len(items)} ({time.time()-t0:.0f}s)")

    del ext
    free_cuda()

    def _union(base, extra_by_slot):
        return lambda it: list(dict.fromkeys(list(it["qwords"]) + list(extra_by_slot.get(it["slot"], []))))

    def _selective(extra_by_slot):
        def f(it):
            if hop0[it["slot"]]["uncertain"]:
                return list(dict.fromkeys(list(it["qwords"]) + list(extra_by_slot.get(it["slot"], []))))
            return list(it["qwords"])
        return f

    B_fn = _union(None, grounded)
    C_fn = _union(None, randomed)
    D_fn = _selective(grounded)
    B = score_arm(items, B_fn, postings, idf, n_slots, med, n_way=args.n_way)
    C = score_arm(items, C_fn, postings, idf, n_slots, med, n_way=args.n_way)
    D = score_arm(items, D_fn, postings, idf, n_slots, med, n_way=args.n_way)
    E = None
    if not args.no_blind:
        E = score_arm(items, _union(None, blind), postings, idf, n_slots, med, n_way=args.n_way)

    wk_B = woken_frac(items, lambda it: it["qwords"], B_fn, postings, idf)
    wk_C = woken_frac(items, lambda it: it["qwords"], C_fn, postings, idf)
    wk_D = woken_frac(items, lambda it: it["qwords"], D_fn, postings, idf)

    # how much of what the model said it actually copied out of the passages it was shown
    copied, total = 0, 0
    for it in items:
        pas = " ".join(passages_for(hop0[it["slot"]]["sc"], write_ctxs, args.k_read)).lower()
        for w in grounded.get(it["slot"], []):
            total += 1
            copied += int(w in pas)
    copy_rate = copied / max(1, total)

    log(f"  B grounded : {json.dumps(B)}\n  woken {json.dumps(wk_B)}")
    log(f"  C random   : {json.dumps(C)}\n  woken {json.dumps(wk_C)}")
    log(f"  D selective: {json.dumps(D)}\n  woken {json.dumps(wk_D)}")
    if E:
        log(f"  E blind    : {json.dumps(E)}")
    log(f"  copy_rate (refined words seen in the shown passages): {copy_rate:.3f}")

    # ---- gates ----
    def headline_beats(ch, base):
        return bool(ch["top1"] >= base["top1"] + 0.03 or ch["acc_20way"] >= base["acc_20way"] + 0.05)

    ref264 = published_264_surface()
    g_reproduces_264 = (
        ref264 is None or abs(A["top1"] - float(ref264.get("top1", A["top1"]))) <= 0.05
    )
    best_name, best = max(
        [("B_grounded", B), ("D_selective", D)], key=lambda kv: kv[1]["top1"]
    )
    g_refine_beats_hop0 = headline_beats(best, A)
    g_grounding_causal = bool(
        best["top1"] >= C["top1"] + 0.05 or best["acc_20way"] >= C["acc_20way"] + 0.05
    )
    g_silence_reduced = bool(A["tie_at_zero_frac"] - best["tie_at_zero_frac"] >= 0.05)
    g_woken_useful = bool(
        (wk_B if best_name == "B_grounded" else wk_D)["woken_frac"] >= 0.05
        and best["top1"] >= A["top1"] - 0.02
    )
    g_selective_beats_always = bool(D["top1"] >= B["top1"] + 0.03)
    g_reads_passages = bool(copy_rate >= 0.30)

    if not g_reproduces_264:
        overall = "READ_REFINE_INVALID"
    elif g_refine_beats_hop0 and g_grounding_causal:
        overall = "READ_REFINE_OK"
    elif g_grounding_causal and (g_silence_reduced or g_woken_useful):
        overall = "READ_REFINE_PARTIAL"
    elif g_refine_beats_hop0 and not g_grounding_causal:
        overall = "PROMPTING_NOT_READING"
    else:
        overall = "READ_REFINE_NO"

    out = {
        "stage": 267,
        "overall": overall,
        "model": args.model,
        "seed": SEED,
        "smoke": args.smoke,
        "bank_slots": n_slots,
        "exam_slots": n_exam,
        "n_eval": len(items),
        "k_read": args.k_read,
        "margin_thresh": args.margin_thresh,
        "overlap_median": med,
        "trained_parameters": 0,
        "fp_version": getattr(L, "canonical_fp_version", lambda: CKPT_P1.name)(),
        "hop0_uncertain_n": n_unc,
        "empty_generations": n_empty,
        "copy_rate_from_passages": copy_rate,
        "gates": {
            "G_hop0_reproduces_264": g_reproduces_264,
            "G_refine_beats_hop0": g_refine_beats_hop0,
            "G_grounding_causal": g_grounding_causal,
            "G_silence_reduced": g_silence_reduced,
            "G_woken_useful": g_woken_useful,
            "G_selective_beats_always": g_selective_beats_always,
            "G_reads_passages": g_reads_passages,
            "best_arm": best_name,
        },
        "arms": {
            "A_hop0_surface": A,
            "B_refine_grounded": B,
            "C_refine_random_passages": C,
            "D_refine_selective": D,
            "E_refine_blind": E,
        },
        "woken": {"B_grounded": wk_B, "C_random": wk_C, "D_selective": wk_D},
        "reference_264_votes": ref264,
        "samples": samples,
        "note": (
            "The loop, not the projection: hop0 votes retrieve passages, the model picks words "
            "while LOOKING at them, hop1 votes again. C shows the same model the same number of "
            "RANDOM passages — if B does not beat C the loop is prompting, not reading, and the "
            "verdict is PROMPTING_NOT_READING however good B looks. D spends refinement only "
            "where hop0 is uncertain, because 266 showed extra words hurt an already-healthy "
            "query. Rank counts a silent gold as last: 266's paraphrase arm read top1 0.477 only "
            "because empty answers left an empty score table. Memory is untouched — same bank, "
            "same postings, zero trained parameters."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def row(name, r):
        if not r:
            return f"| {name} | n/a | n/a | n/a | n/a |\n"
        return (f"| {name} | {r['top1']:.3f} | {r['median_rank']:.1f} | "
                f"{r['acc_20way']:.3f} | {r['tie_at_zero_frac']:.3f} |\n")

    MINI.write_text(
        f"# Stage 267 read-refine\n\n**{overall}** · model={args.model} · bank={n_slots} "
        f"· eval={len(items)} · trained params **0**\n\n"
        f"| arm | top1 | median | 20-way | silence |\n|---|---:|---:|---:|---:|\n"
        + row("A hop0 surface", A)
        + row("B refine grounded", B)
        + row("C refine RANDOM passages", C)
        + row("D refine selective", D)
        + row("E refine blind", E)
        + f"\n- woken (grounded): **{wk_B['woken_n']}** queries, top1 among them "
          f"{wk_B['woken_top1']:.3f}; random control woke {wk_C['woken_n']}\n"
        f"- copy rate from shown passages: **{copy_rate:.3f}**\n"
        f"- hop0 uncertain: {n_unc}/{len(items)}\n\n"
        f"## Gates\n\n"
        + "".join(f"- {k}: **{v}**\n" for k, v in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
