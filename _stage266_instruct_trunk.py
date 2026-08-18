"""
Stage 266 — Is 261's NO a trunk-capacity wall or an interface wall?

Same open-bank NL exam as 261. Only the frozen trunk changes, via ExternalTrunk (262).
W_q stays in fp space (256); SemQuery.in_dim is read off each trunk. Keys never see the
external tokenizer.

Matched-size control is the point: Qwen2.5-0.5B base vs Qwen2.5-0.5B-Instruct — same
parameter count, only the tuning differs. Instruct without chat template is an invalid
run (bare text ≈ base). Ladder then steps size (1.5B / 3B Instruct) to separate
"ability appears" from "size pulls".

Fourth arm (zero train, strongest thesis check): Instruct emits keywords in words;
those feed 261f-style word votes. "Mind formulates the query, tape answers."

Gates:
  G_instruct_beats_base_matched — 0.5B-Instruct vs 0.5B base on a live semantic channel
                                  (if Instruct alpha≈0, compare fp_only — not fp+sem≡fp)
  G_ladder_monotone             — 20-way nondecreasing along the size ladder; null if
                                  Instruct alphas collapsed (trunk out of the loop)
  G_prompted_query              — keyword→votes beat trained W_sem (matched Instruct)
  G_prompted_beats_surface      — Instruct keywords beat raw question words on headlines
                                  (top1 / 20-way only; median is G_prompted_median_better)
  G_union_beats_surface         — surface ∪ keywords beats surface on headlines
  G_mind_refines                — union wins headlines + preserves coverage vs surface

Verdict remaps 261:
  NO_AT_TRUNK_SCALE       — matched Instruct does not beat base → capacity at this scale
  NL_QUERY_NO            — honest interface NO (Instruct helps or prompted wins, open still fails)
  INSTRUCT_TRUNK_OK      — Instruct beats base and open-domain signal moves
  PROMPTED_QUERY_SIGNAL  — keywords→votes beat W_sem; interface still broken
  WORDS_FORMULATE_QUERY  — words crush learned query vector; remap QUERY_MUST_BE_WORDS
  MIND_REFINES_QUERY     — union (surface ∪ keywords) beats surface on headlines; mind helps

  python _stage266_instruct_trunk.py [--smoke]
  python _stage266_instruct_trunk.py --include-3b
  python _stage266_instruct_trunk.py --prompted-only   # cheap; merges into prior decision
  python _stage266_instruct_trunk.py --smoke --only qwen05_base --verify-seed
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
from _stage261_nl_query import (
    SemQuery,
    WORD_RE,
    collect,
    ctx_words,
    fp_raw,
    jaccard,
)
from _stage262_trunk_swap import ExternalTrunk
from _tape_index import context_words, nway_strict, vote_arm_fields, vote_rank

RES = Path("results")
DECISION = RES / "stage266_decision.json"
MINI = RES / "stage266_mini.md"
LOG = RES / "_stage266_log.txt"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 261  # identical exam construction to 261

# Matched pair first; then size ladder. 3B optional (VRAM).
LADDER = (
    {"id": "qwen05_base", "model": "Qwen/Qwen2.5-0.5B", "instruct": False, "rung": 0},
    {"id": "qwen05_instruct", "model": "Qwen/Qwen2.5-0.5B-Instruct", "instruct": True, "rung": 1},
    {"id": "qwen15_instruct", "model": "Qwen/Qwen2.5-1.5B-Instruct", "instruct": True, "rung": 2},
    {"id": "qwen3_instruct", "model": "Qwen/Qwen2.5-3B-Instruct", "instruct": True, "rung": 3},
)

KW_PROMPT = (
    "Extract 3-8 content keywords (nouns and proper names) from the text below. "
    "Reply with comma-separated words only — no sentences.\n\n{text}"
)

# Not "better words" — different words: absent from the question, hopefully present on tape.
PARA_PROMPT = (
    "The text below is a question fragment. List 3-8 English content words that do NOT "
    "appear in the text, but that a related Wikipedia article on the same topic would "
    "likely contain (synonyms, related people, places, technical terms). "
    "Comma-separated words only — no sentences.\n\n{text}"
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
    """Windows + long HF hub paths → OSError Errno 22. Prefer short HF_HOME (e.g. C:\\hf)."""
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
    """Instruct models without this are base models wearing an Instruct name."""
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def trunk_state(ext: ExternalTrunk, text: str, *, instruct: bool) -> torch.Tensor | None:
    feed = chat_wrap(ext.tok, text) if instruct else text
    return ext.state(feed)


@torch.no_grad()
def generate_word_list(ext: ExternalTrunk, prompt_body: str, *, max_new: int = 48) -> list[str]:
    prompt = chat_wrap(ext.tok, prompt_body)
    ids = ext.tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    ids = {k: v.to(ext.device) for k, v in ids.items()}
    n_in = int(ids["input_ids"].shape[1])
    out = ext.model.generate(
        **ids, max_new_tokens=max_new, do_sample=False, pad_token_id=ext.tok.eos_token_id,
    )
    raw = ext.tok.decode(out[0][n_in:], skip_special_tokens=True)
    words = []
    seen = set()
    for part in re.split(r"[,;\n|/]+", raw):
        for w in WORD_RE.findall(part):
            lw = w.lower()
            if lw in seen:
                continue
            seen.add(lw)
            words.append(lw)
    return words


@torch.no_grad()
def extract_keywords(ext: ExternalTrunk, text: str, *, max_new: int = 48) -> list[str]:
    return generate_word_list(ext, KW_PROMPT.format(text=text), max_new=max_new)


@torch.no_grad()
def extract_paraphrase(ext: ExternalTrunk, text: str, *, max_new: int = 48) -> list[str]:
    """Words absent from the question — bridge candidates for tape postings."""
    qset = {w.lower() for w in WORD_RE.findall(text)}
    raw = generate_word_list(ext, PARA_PROMPT.format(text=text), max_new=max_new)
    return [w for w in raw if w not in qset]


def silence_beats(challenger: dict, baseline: dict, *, margin: float = 0.05) -> bool:
    """Primary mind-chance metric: reduce tie_at_zero (overall or low-overlap)."""
    c_sil = challenger.get("silence") or {}
    b_sil = baseline.get("silence") or {}
    return bool(
        float(challenger.get("tie_at_zero_frac", 1.0))
        <= float(baseline.get("tie_at_zero_frac", 0.0)) - margin
        or float(c_sil.get("tie_at_zero_frac_low_overlap", 1.0))
        <= float(b_sil.get("tie_at_zero_frac_low_overlap", 0.0)) - margin
    )


def paraphrase_bridge_diag(
    items_eval, para_by_slot: dict, postings: dict,
) -> dict:
    """How often novel words actually exist on the tape index."""
    n_novel = n_on_tape = n_q_with_bridge = 0
    for it in items_eval:
        novel = para_by_slot.get(it["slot"], [])
        hits = [w for w in novel if w in postings]
        n_novel += len(novel)
        n_on_tape += len(hits)
        if hits:
            n_q_with_bridge += 1
    n = max(1, len(items_eval))
    return {
        "n_queries": len(items_eval),
        "n_novel_words": n_novel,
        "n_novel_on_tape": n_on_tape,
        "novel_on_tape_frac": float(n_on_tape / max(1, n_novel)),
        "queries_with_bridge_word_frac": float(n_q_with_bridge / n),
    }


def build_exam(bank, lines, n_ent, n_dist, rng):
    """Keys + natural (write, ask) pairs — same seed/logic as 261.

    Harvest = remaining multi-mention entities (≥2 sentences), same recipe as the exam:
    write from mention A, ask from mention B. NOT same-sentence prefixes of noise slots
    (that would teach a different invariance than the open NL exam).
    """
    cands = collect(lines, bank)
    ents = sorted(cands)[:n_ent]
    rng.shuffle(ents)
    keys, vals, items = [], [], []
    write_ctxs: list[str] = []

    def try_pair(e, occ):
        a, b = occ[0], occ[1]
        wctx = a["line"][max(0, a["start"] - 140) : min(len(a["line"]), a["end"] + 140)]
        k = bank.ctx_fp(wctx, exclude=e)
        if k is None:
            return None
        qtext = b["line"][max(0, b["start"] - 200) : b["start"]].strip()
        if len(WORD_RE.findall(qtext)) < 4:
            return None
        raw = fp_raw(bank, qtext, use_anchor=True)
        if raw is None:
            return None
        return {
            "key": F.normalize(bank.fp([a["anchor"]])[0] + k, dim=-1),
            "wctx": wctx,
            "qtext": qtext,
            "raw": raw,
            "qwords": context_words(qtext, exclude=e),
            "overlap": jaccard(ctx_words(wctx, e), ctx_words(qtext, e)),
        }

    for e in ents:
        got = try_pair(e, cands[e])
        if got is None:
            continue
        keys.append(got["key"])
        write_ctxs.append(got["wctx"])
        items.append({
            "ent": e, "slot": len(vals), "qtext": got["qtext"], "raw": got["raw"],
            "wctx": got["wctx"], "qwords": got["qwords"], "overlap": got["overlap"],
        })
        vals.append(e)

    n_exam = len(keys)
    used = {it["ent"] for it in items}

    # Cross-mention harvest: leftover multi-mention entities (same dist as exam).
    # Cap at n_dist so bank stays ≈ n_exam+n_dist — else PRE_HARVEST (from 4352) is invalid.
    harvest: list[dict] = []
    rest = [e for e in sorted(cands) if e not in used]
    rng2 = random.Random(SEED + 11)
    rng2.shuffle(rest)
    for e in rest:
        if len(harvest) >= n_dist:
            break
        got = try_pair(e, cands[e])
        if got is None:
            continue
        slot_id = len(vals)
        keys.append(got["key"])
        write_ctxs.append(got["wctx"])
        vals.append(e)
        used.add(e)
        harvest.append({
            "ent": e, "slot": slot_id, "qtext": got["qtext"], "raw": got["raw"],
            "wctx": got["wctx"], "qwords": got["qwords"], "overlap": got["overlap"],
            "harvest": True,
        })

    # Extra single-mention wiki noise if harvest alone is short of the open-bank size.
    n_target = n_exam + n_dist
    for ln in lines:
        if len(keys) >= n_target:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = max(0, m.start() - 140), min(len(ln), m.end() + 140)
            c = bank.ctx_fp(ln[lo:hi], exclude=e)
            if c is None:
                continue
            an = [w for w in ANCHOR_RE.findall(ln[lo : m.start()]) if w != e]
            if not an:
                continue
            keys.append(F.normalize(bank.fp([an[-1]])[0] + c, dim=-1))
            write_ctxs.append(ln[lo:hi])
            vals.append(e)
            used.add(e)
            if len(keys) >= n_target:
                break

    postings: dict[str, list[int]] = defaultdict(list)
    for cid, (e, wctx) in enumerate(zip(vals, write_ctxs)):
        for w in context_words(wctx, exclude=e):
            postings[w].append(cid)
    idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in postings}
    return keys, vals, items, n_exam, postings, idf, harvest


# Pre-harvest 266 collapse (same exam construction) — relative HARVEST_FIXES baseline.
PRE_HARVEST = {
    "qwen05_base": {
        "eval_top1_sem": 0.0, "eval_20way_sem": 0.074,
        "eval_20way_fp": 0.153, "fit_20way_fp": 0.301, "fit_top1_sem": 1.0,
    },
    "qwen05_instruct": {
        "eval_top1_sem": 0.0, "eval_20way_sem": 0.119,
        "eval_20way_fp": 0.210, "fit_20way_fp": 0.443, "fit_top1_sem": 1.0,
    },
    "qwen15_instruct": {
        "eval_top1_sem": 0.0, "eval_20way_sem": 0.085,
        "eval_20way_fp": 0.188, "fit_20way_fp": 0.528, "fit_top1_sem": 1.0,
    },
}


def harvest_improved(r: dict, rid: str) -> bool:
    """Relative to PRE_HARVEST. Requires eval-side lift — fit drop alone is undertraining."""
    pre = PRE_HARVEST.get(rid)
    if not pre or "top1_sem" not in r:
        return False
    ev_top1 = float(r.get("top1_sem", 0.0))
    ev_sem = float(r.get("acc_20way_sem", 0.0))
    ev_fp = float(r.get("acc_20way_fp", 0.0))
    fit_fp = float(r.get("fit_acc_20way_fp", pre["fit_20way_fp"]))

    eval_lift = (
        ev_top1 >= 0.02
        or ev_sem >= 1.5 * pre["eval_20way_sem"]
        or ev_fp >= pre["eval_20way_fp"] + 0.05
    )
    if not eval_lift:
        return False

    # Optional supporting evidence: memorization broke *and* eval moved
    # (fit_top1 < 0.90 alone is NOT enough — large pool + fixed steps → underfit)
    return True


def still_collapsed(r: dict) -> bool:
    """Absolute memorize signature: perfect fit, dead open top1, weak 20-way."""
    return bool(
        float(r.get("fit_top1_sem", 0)) >= 0.90
        and float(r.get("top1_sem", 1)) <= 0.05
        and float(r.get("acc_20way_sem", 1)) < 0.20
    )


def fill_h(items, ext: ExternalTrunk, *, instruct: bool) -> int:
    n_ok = 0
    for it in items:
        old = it.pop("h", None)
        del old
        h = trunk_state(ext, it["qtext"], instruct=instruct)
        # keep on CPU between trunks — GPU copies of prior model states would OOM the ladder
        it["h"] = None if h is None else h.detach().cpu()
        if it["h"] is not None:
            n_ok += 1
    return n_ok


def clear_h(items) -> None:
    for it in items:
        it.pop("h", None)


def train_and_score(train_pool, fit, ev, K, steps, tau, device, med: float, n_way=20):
    """Train W_q+SemQuery on bank-wide harvest (+ fit exam); score fit vs eval.

    256 lesson: fitting adapters on a handful of exam pairs memorizes slots. Train pool
    must be harvested (prefix→slot) pairs from the bank — noise + fit, never eval.
    """
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    train_pool = [it for it in train_pool if it.get("h") is not None]
    fit = [it for it in fit if it.get("h") is not None]
    ev = [it for it in ev if it.get("h") is not None]
    if len(train_pool) < 16 or len(fit) < 8 or len(ev) < 8:
        return None

    h_dim = int(train_pool[0]["h"].numel())
    semq = SemQuery(h_dim, device)
    W_q = L.init_query_adapter(device)
    Rq = torch.stack([it["raw"] for it in train_pool]).to(device).float()
    Hq = torch.stack([it["h"].to(device) for it in train_pool]).to(device).float()
    Gq = torch.tensor([it["slot"] for it in train_pool], device=device, dtype=torch.long)
    n_train = Rq.size(0)
    log(f"    train_pool={n_train} (harvest+fit) fit_diag={len(fit)} eval={len(ev)}")

    opt = torch.optim.AdamW(list(semq.parameters()) + list(W_q.parameters()), lr=2e-3, weight_decay=0.01)
    for step in range(1, steps + 1):
        sel = torch.randint(0, n_train, (min(32, n_train),), device=device)
        q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
        a = semq.a(Hq[sel], q_fp, K, tau).reshape(-1, 1)
        q = F.normalize((1 - a) * q_fp + a * semq.q(Hq[sel]), dim=-1)
        loss = F.cross_entropy((q @ K.t()) / tau, Gq[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(semq.parameters()) + list(W_q.parameters()), 1.0)
        opt.step()
        if step == 40 or step % max(1, steps // 5) == 0:
            log(f"    step {step}/{steps} loss={float(loss):.3f} a={float(a.mean()):.3f}")
    semq.eval()

    @torch.no_grad()
    def score(items_, use_sem, Kmat=K):
        wrng = random.Random(SEED + 5)
        ranks, alphas, lo, hi, nway = [], [], [], [], []
        for it in items_:
            q_fp = F.normalize(W_q(it["raw"].unsqueeze(0)), dim=-1)[0]
            h = it["h"].to(device)
            if use_sem:
                af = float(semq.a(h, q_fp, Kmat, tau).reshape(-1)[0])
                alphas.append(af)
                q_sem = F.normalize(semq.q(h.unsqueeze(0)), dim=-1).reshape(-1)
                q = F.normalize((1 - af) * q_fp + af * q_sem, dim=-1)
                sims = Kmat @ q
            else:
                sims = Kmat @ q_fp
            r = 1 + int((sims > sims[it["slot"]]).sum())
            ranks.append(r)
            (hi if it["overlap"] > med else lo).append(int(r == 1))
            pool = [j for j in wrng.sample(range(Kmat.size(0)), min(n_way * 3, Kmat.size(0)))
                    if j != it["slot"]][: n_way - 1]
            nway.append(int(all(float(sims[it["slot"]]) > float(sims[j]) for j in pool)))
        r = np.asarray(ranks, dtype=np.float64)
        return {
            "top1": float(np.mean(r == 1)), "mrr": float(np.mean(1.0 / r)),
            "median_rank": float(np.median(r)),
            "top1_low_overlap": float(np.mean(lo)) if lo else float("nan"),
            "top1_high_overlap": float(np.mean(hi)) if hi else float("nan"),
            "alpha": float(np.mean(alphas)) if alphas else 0.0, "n": len(ranks),
            f"acc_{n_way}way": float(np.mean(nway)),
            f"chance_{n_way}way": 1.0 / n_way,
        }

    fp_only, sem = score(ev, False), score(ev, True)
    fp_fit, sem_fit = score(fit, False), score(fit, True)
    perm = torch.randperm(K.size(0), generator=torch.Generator().manual_seed(SEED + 1))
    shuf = score(ev, True, Kmat=K[perm.to(K.device)])
    overfit = still_collapsed({
        "fit_top1_sem": sem_fit["top1"], "top1_sem": sem["top1"],
        "acc_20way_sem": sem["acc_20way"],
    })
    log(f"    FIT  fp 20-way={fp_fit['acc_20way']:.3f} sem 20-way={sem_fit['acc_20way']:.3f} "
        f"top1_sem={sem_fit['top1']:.3f} a={sem_fit['alpha']:.3f}")
    log(f"    EVAL fp 20-way={fp_only['acc_20way']:.3f} sem 20-way={sem['acc_20way']:.3f} "
        f"top1_sem={sem['top1']:.3f} a={sem['alpha']:.3f}  mixer_overfit={overfit}")
    return {
        "h_dim": h_dim, "n_fit": len(fit), "n_eval": len(ev),
        "n_train_pool": n_train, "overlap_median": med,
        "fp_only": fp_only, "fp_plus_sem": sem, "shuffled_keys": shuf,
        "fit_fp_only": fp_fit, "fit_fp_plus_sem": sem_fit,
        "mixer_overfit": overfit,
        "acc_20way_fp": fp_only["acc_20way"], "acc_20way_sem": sem["acc_20way"],
        "top1_sem": sem["top1"], "alpha": sem["alpha"],
        "fit_top1_sem": sem_fit["top1"], "fit_acc_20way_sem": sem_fit["acc_20way"],
        "fit_acc_20way_fp": fp_fit["acc_20way"],
    }


def score_votes(items_eval, postings, idf, qwords_fn, n_slots, n_way=20, med=0.0):
    wrng = random.Random(SEED + 5)
    ranks, nway, rows = [], [], []
    for it in items_eval:
        qw = qwords_fn(it)
        sc: dict[int, float] = defaultdict(float)
        for w in qw:
            for cid in postings.get(w, ()):
                sc[cid] += idf.get(w, 0.0)
        gold, rank = vote_rank(sc, it["slot"], n_slots)
        ranks.append(rank)
        pool = [j for j in wrng.sample(range(n_slots), min(n_way * 3, n_slots))
                if j != it["slot"]][: n_way - 1]
        nway.append(int(nway_strict(gold, (sc.get(j, 0.0) for j in pool))))
        rows.append({"gold_score": float(gold), "rank": rank, "low_overlap": it["overlap"] <= med})
    silence = vote_arm_fields(rows)
    r = np.asarray(ranks, dtype=np.float64)
    return {
        "top1": float(np.mean(r == 1)), "mrr": float(np.mean(1.0 / r)),
        "median_rank": float(np.median(r)),
        "top1_low_overlap": silence["top1_low_overlap"],
        "top1_high_overlap": silence["top1_high_overlap"],
        "tie_at_zero_frac": silence["tie_at_zero_frac"],
        "silence": silence,
        "n": len(ranks),
        f"acc_{n_way}way": float(np.mean(nway)),
        f"chance_{n_way}way": 1.0 / n_way,
    }


def headline_beats(challenger: dict, baseline: dict) -> bool:
    """Headline only: top1 or 20-way. Median rank is a separate diagnostic."""
    return bool(
        challenger["top1"] >= baseline["top1"] + 0.03
        or challenger["acc_20way"] >= baseline["acc_20way"] + 0.05
    )


def median_rank_better(challenger: dict, baseline: dict, *, margin: float = 10.0) -> bool:
    return bool(challenger["median_rank"] <= baseline["median_rank"] - margin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--entities", type=int, default=0)
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--skip-3b", dest="skip_3b", action="store_true",
                    help="skip Qwen2.5-3B-Instruct (default)")
    ap.add_argument("--include-3b", dest="skip_3b", action="store_false",
                    help="try 3B Instruct; OOM → skip that rung")
    ap.set_defaults(skip_3b=True)
    ap.add_argument("--prompt-model", type=str, default="",
                    help="HF id for keyword arm; default = matched 0.5B-Instruct")
    ap.add_argument("--verify-seed", action="store_true",
                    help="train base twice on same h; confirm identical 20-way")
    ap.add_argument("--no-prompted", action="store_true",
                    help="skip keyword→votes / paraphrase arms")
    ap.add_argument("--prompted-only", action="store_true",
                    help="skip trunk ladder; run keyword→votes (+ union + paraphrase) and merge prior decision")
    ap.add_argument("--paraphrase-only", action="store_true",
                    help="skip trunk ladder + keywords; run paraphrase silence arm and merge prior")
    ap.add_argument("--only", type=str, default="",
                    help="comma rung ids to run (e.g. qwen05_base); empty = full ladder")
    args = ap.parse_args()
    skip_3b = bool(args.skip_3b)
    if args.prompted_only and args.no_prompted:
        log("  --prompted-only and --no-prompted are mutually exclusive")
        return 1
    if args.paraphrase_only and args.no_prompted:
        log("  --paraphrase-only and --no-prompted are mutually exclusive")
        return 1
    merge_prior_arms = bool(args.prompted_only or args.paraphrase_only)

    LOG.write_text("", encoding="utf-8")
    hf_home = ensure_short_hf_home()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = args.steps or (150 if args.smoke else 800)
    n_ent = args.entities or (60 if args.smoke else 400)
    n_dist = args.distractor_slots or (400 if args.smoke else 4000)
    max_lines = 3000 if args.smoke else 25000

    log(f"Stage266 instruct trunk start {datetime.now(timezone.utc).isoformat()} "
        f"device={device} steps={steps} smoke={args.smoke} skip_3b={skip_3b}"
        + (f" HF_HOME={hf_home}" if hf_home else ""))

    _, _, stoi, n_char = load_data()
    V = Tokenizer.from_file(str(s177.TOK_PATH)).get_vocab_size()
    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank = FpBank(model_can, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(3_000_000 if args.smoke else 20_000_000)
    lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400][:max_lines]
    keys, vals, items, n_exam, postings, idf, harvest = build_exam(
        bank, lines, n_ent, n_dist, rng,
    )
    if len(items) < 16:
        log("  not enough exam pairs")
        return 1
    K = torch.stack(keys).to(device).float()
    med = float(np.median([it["overlap"] for it in items]))
    n_harv = len(harvest)
    n_pure = len(vals) - n_exam - n_harv
    log(f"  exam={n_exam} bank={len(vals)} harvest_cross={n_harv} pure_noise={n_pure} "
        f"postings={sum(len(v) for v in postings.values())} overlap_med={med:.3f}")

    order = list(range(len(items)))
    random.Random(SEED).shuffle(order)
    mid = len(order) // 2
    fit_items = [items[i] for i in order[:mid]]
    ev_items = [items[i] for i in order[mid:]]
    # Cross-mention harvest (+ fit exam) — never eval exam slots
    train_pool = list(fit_items) + list(harvest)
    log(f"  train_pool={len(train_pool)} (fit_exam={len(fit_items)} + harvest_cross={len(harvest)}) "
        f"eval_exam={len(ev_items)}")
    surface_votes = score_votes(
        ev_items, postings, idf, lambda it: it["qwords"], len(vals), med=med,
    )
    log(f"  surface votes (eval n={len(ev_items)}): {json.dumps(surface_votes)}")

    ladder = [r for r in LADDER if not (skip_3b and r["id"] == "qwen3_instruct")]
    if args.prompted_only or args.paraphrase_only:
        ladder = []
        log(f"  --{'paraphrase' if args.paraphrase_only else 'prompted'}-only: skipping trunk ladder")
    elif args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        ladder = [r for r in ladder if r["id"] in want]
        if not ladder:
            log(f"  --only {want} matched no rungs")
            return 1
    trunk_results: dict[str, dict] = {}
    prior: dict | None = None
    if merge_prior_arms and DECISION.exists():
        try:
            prior = json.loads(DECISION.read_text(encoding="utf-8"))
            for rid, row in (prior.get("ladder") or {}).items():
                if isinstance(row, dict):
                    trunk_results[rid] = row
            log(f"  merged prior ladder from {DECISION} "
                f"({sum(1 for r in trunk_results.values() if 'acc_20way_sem' in r)} scored trunks)")
        except Exception as e:  # noqa: BLE001
            log(f"  prior decision load fail: {e}")
            prior = None

    for rung in ladder:
        log(f"\n== trunk {rung['id']} model={rung['model']} instruct={rung['instruct']} ==")
        free_cuda()
        try:
            ext = ExternalTrunk(rung["model"], device)
        except Exception as e:  # noqa: BLE001
            log(f"  LOAD FAIL: {type(e).__name__}: {e}")
            trunk_results[rung["id"]] = {"error": f"{type(e).__name__}: {e}", **rung}
            continue
        if rung["instruct"] and not getattr(ext.tok, "chat_template", None):
            log("  FATAL: Instruct model has no chat_template — refuse to run bare text")
            trunk_results[rung["id"]] = {"error": "missing_chat_template", **rung}
            del ext
            free_cuda()
            continue
        log(f"  hidden={ext.dim} chat_template={'yes' if rung['instruct'] else 'n/a (base)'}")
        clear_h(items)
        clear_h(harvest)
        free_cuda()
        n_ok = fill_h(items, ext, instruct=rung["instruct"])
        n_hv = fill_h(harvest, ext, instruct=rung["instruct"])
        log(f"  h filled exam {n_ok}/{len(items)} harvest {n_hv}/{len(harvest)}")
        try:
            res = train_and_score(
                train_pool, fit_items, ev_items, K, steps, args.tau, device, med,
            )
        except torch.cuda.OutOfMemoryError as e:
            log(f"  OOM during train: {e}")
            trunk_results[rung["id"]] = {"error": "OOM_train", **rung, "hidden": ext.dim}
            del ext
            clear_h(items)
            clear_h(harvest)
            free_cuda()
            continue
        if res is None:
            trunk_results[rung["id"]] = {"error": "too_few_h", **rung}
        else:
            trunk_results[rung["id"]] = {
                **{k: rung[k] for k in ("id", "model", "instruct", "rung")},
                "hidden": ext.dim,
                "chat_template_used": bool(rung["instruct"]),
                **res,
            }
            log(f"  fp 20-way={res['acc_20way_fp']:.3f} sem 20-way={res['acc_20way_sem']:.3f} "
                f"top1_sem={res['top1_sem']:.3f} a={res['alpha']:.3f} "
                f"FIT top1_sem={res['fit_top1_sem']:.3f} pool={res['n_train_pool']}")
            if args.verify_seed and rung["id"] == "qwen05_base":
                res2 = train_and_score(
                    train_pool, fit_items, ev_items, K, steps, args.tau, device, med,
                )
                same = (
                    res2 is not None
                    and abs(res2["acc_20way_sem"] - res["acc_20way_sem"]) < 1e-12
                    and abs(res2["acc_20way_fp"] - res["acc_20way_fp"]) < 1e-12
                    and abs(res2["top1_sem"] - res["top1_sem"]) < 1e-12
                )
                trunk_results[rung["id"]]["verify_seed"] = {
                    "second_acc_20way_sem": None if res2 is None else res2["acc_20way_sem"],
                    "second_acc_20way_fp": None if res2 is None else res2["acc_20way_fp"],
                    "identical": same,
                }
                s2 = "None" if res2 is None else f"{res2['acc_20way_sem']:.6f}"
                log(f"  verify_seed identical={same} sem {res['acc_20way_sem']:.6f} vs {s2}")
        # Always unload before next rung — reload matched Instruct only for prompted arm.
        del ext
        clear_h(items)
        clear_h(harvest)
        free_cuda()

    # ---- word arms: keywords / union / paraphrase (zero train) ----
    prompted = None
    union_votes = None
    paraphrase = None
    paraphrase_union = None
    paraphrase_diag = None
    prompt_ext: ExternalTrunk | None = None
    prompt_rung: dict | None = None
    skip_word_arms = bool(
        args.verify_seed or args.no_prompted
        or (
            args.only
            and not args.prompted_only
            and not args.paraphrase_only
            and "qwen05_instruct" not in args.only
            and "prompt" not in args.only
        )
    )
    run_keywords = (not skip_word_arms) and (not args.paraphrase_only)
    run_paraphrase = not skip_word_arms
    if skip_word_arms:
        log("\n== word arms: skipped (verify-seed / --only / --no-prompted) ==")
    else:
        mid_name = args.prompt_model or "Qwen/Qwen2.5-0.5B-Instruct"
        log(f"\n== word arms: loading {mid_name} ==")
        try:
            free_cuda()
            prompt_ext = ExternalTrunk(mid_name, device)
            prompt_rung = {"id": "prompt", "model": mid_name, "instruct": True}
            if not getattr(prompt_ext.tok, "chat_template", None):
                log("  FATAL: prompt model missing chat_template")
                prompted = {"error": "missing_chat_template"}
                del prompt_ext
                prompt_ext = None
        except Exception as e:  # noqa: BLE001
            log(f"  prompt model load fail: {e}")
            prompt_ext = None

    # Restore prior keyword/union when paraphrase-only.
    if args.paraphrase_only and prior:
        if isinstance(prior.get("prompted_query"), dict) and "acc_20way" in prior["prompted_query"]:
            prompted = prior["prompted_query"]
        if isinstance(prior.get("union_votes"), dict) and "acc_20way" in prior["union_votes"]:
            union_votes = prior["union_votes"]

    oom = False
    if run_keywords and prompt_ext is not None and prompt_rung is not None and (
        prompted is None or "acc_20way" not in (prompted or {})
    ):
        log(f"\n== prompted keywords → votes via {prompt_rung.get('model')} ==")
        kw_cache: list[list[str]] = []
        n_empty = 0
        for it in ev_items:
            try:
                kws = extract_keywords(prompt_ext, it["qtext"])
            except torch.cuda.OutOfMemoryError:
                log("  OOM on keyword generate — abort keyword arm")
                prompted = {"error": "OOM_generate"}
                oom = True
                break
            if not kws:
                n_empty += 1
                kws = list(it["qwords"])
            kw_cache.append(kws)
        if not oom:
            kw_by_slot = {ev_items[i]["slot"]: kw_cache[i] for i in range(len(ev_items))}
            prompted = score_votes(
                ev_items, postings, idf,
                lambda it: kw_by_slot[it["slot"]],
                len(vals), med=med,
            )
            prompted.update({
                "model": prompt_rung.get("model"),
                "trained_parameters": 0,
                "empty_keyword_fallback_n": n_empty,
                "examples": [
                    {"qtext": ev_items[i]["qtext"][:120], "keywords": kw_cache[i][:8]}
                    for i in range(min(5, len(ev_items)))
                ],
            })
            log(f"  prompted votes: {json.dumps({k: v for k, v in prompted.items() if k != 'examples'})}")

            union_by_slot = {
                ev_items[i]["slot"]: list(dict.fromkeys(
                    list(ev_items[i]["qwords"]) + list(kw_cache[i])
                ))
                for i in range(len(ev_items))
            }
            union_votes = score_votes(
                ev_items, postings, idf,
                lambda it: union_by_slot[it["slot"]],
                len(vals), med=med,
            )
            union_votes["trained_parameters"] = 0
            union_votes["kind"] = "surface_union_keywords"
            log(f"  union votes: {json.dumps(union_votes)}")

    if run_paraphrase and prompt_ext is not None and prompt_rung is not None and not oom:
        log(f"\n== paraphrase (novel words) → votes via {prompt_rung.get('model')} ==")
        para_cache: list[list[str]] = []
        n_empty_p = 0
        for it in ev_items:
            try:
                pws = extract_paraphrase(prompt_ext, it["qtext"])
            except torch.cuda.OutOfMemoryError:
                log("  OOM on paraphrase generate — abort paraphrase arm")
                paraphrase = {"error": "OOM_generate"}
                oom = True
                break
            if not pws:
                n_empty_p += 1
            para_cache.append(pws)
        if not oom:
            para_by_slot = {ev_items[i]["slot"]: para_cache[i] for i in range(len(ev_items))}
            paraphrase_diag = paraphrase_bridge_diag(ev_items, para_by_slot, postings)
            paraphrase = score_votes(
                ev_items, postings, idf,
                lambda it: para_by_slot[it["slot"]],
                len(vals), med=med,
            )
            paraphrase.update({
                "model": prompt_rung.get("model"),
                "trained_parameters": 0,
                "kind": "paraphrase_novel",
                "empty_paraphrase_n": n_empty_p,
                "bridge": paraphrase_diag,
                "examples": [
                    {
                        "qtext": ev_items[i]["qtext"][:120],
                        "paraphrase": para_cache[i][:8],
                        "on_tape": [w for w in para_cache[i][:8] if w in postings],
                    }
                    for i in range(min(5, len(ev_items)))
                ],
            })
            log(f"  paraphrase votes: {json.dumps({k: v for k, v in paraphrase.items() if k != 'examples'})}")

            # Honest silence test: keep surface coverage, add novel bridge words.
            pu_by_slot = {
                ev_items[i]["slot"]: list(dict.fromkeys(
                    list(ev_items[i]["qwords"]) + list(para_cache[i])
                ))
                for i in range(len(ev_items))
            }
            paraphrase_union = score_votes(
                ev_items, postings, idf,
                lambda it: pu_by_slot[it["slot"]],
                len(vals), med=med,
            )
            paraphrase_union["trained_parameters"] = 0
            paraphrase_union["kind"] = "surface_union_paraphrase"
            paraphrase_union["bridge"] = paraphrase_diag
            woken = 0
            for it in ev_items:
                def _gold(words, slot=it["slot"]):
                    sc = 0.0
                    for w in words:
                        if slot in postings.get(w, ()):
                            sc += idf.get(w, 0.0)
                    return sc
                if _gold(it["qwords"]) <= 0.0 and _gold(pu_by_slot[it["slot"]]) > 0.0:
                    woken += 1
            paraphrase_union["surface_silent_woken_frac"] = float(woken / max(1, len(ev_items)))
            paraphrase_union["surface_silent_woken_n"] = woken
            log(
                "  paraphrase∪surface: "
                + json.dumps({k: v for k, v in paraphrase_union.items() if k != "examples"})
            )

    if prompt_ext is not None:
        del prompt_ext
        prompt_ext = None
        free_cuda()

    # ---- gates ----
    skip_prompted = skip_word_arms
    base = trunk_results.get("qwen05_base")
    inst = trunk_results.get("qwen05_instruct")
    base_ok = base and "acc_20way_sem" in base
    inst_ok = inst and "acc_20way_sem" in inst

    def _alpha_off(r: dict | None, thr: float = 1e-5) -> bool:
        return bool(r and "alpha" in r and abs(float(r["alpha"])) < thr)

    if _alpha_off(inst):
        g_instruct_beats = bool(
            base_ok and inst_ok
            and inst["acc_20way_fp"] >= base["acc_20way_fp"] + 0.05
        )
        g_instruct_note = "alpha_collapsed_compare_fp_only"
    else:
        g_instruct_beats = bool(
            base_ok and inst_ok
            and inst["acc_20way_sem"] >= base["acc_20way_sem"] + 0.05
        )
        g_instruct_note = "compare_fp_plus_sem"

    ladder_scores = []
    ladder_alphas = []
    for rid in ("qwen05_instruct", "qwen15_instruct", "qwen3_instruct"):
        r = trunk_results.get(rid)
        if r and "acc_20way_sem" in r:
            ladder_scores.append(r["acc_20way_sem"])
            ladder_alphas.append(float(r.get("alpha", 1.0)))
    if len(ladder_scores) >= 2 and all(abs(a) < 1e-5 for a in ladder_alphas):
        g_ladder = None
        g_ladder_note = "alphas_collapsed_trunk_out"
    else:
        g_ladder = bool(
            len(ladder_scores) >= 2
            and all(ladder_scores[i] + 1e-9 >= ladder_scores[i - 1] - 0.02
                    for i in range(1, len(ladder_scores)))
            and ladder_scores[-1] >= ladder_scores[0] + 0.03
        )
        g_ladder_note = "scored"

    sem_matched = inst["acc_20way_sem"] if inst_ok else float("nan")
    prompted_ok = isinstance(prompted, dict) and "acc_20way" in prompted
    union_ok = isinstance(union_votes, dict) and "acc_20way" in union_votes
    para_ok = isinstance(paraphrase, dict) and "acc_20way" in paraphrase
    para_u_ok = isinstance(paraphrase_union, dict) and "acc_20way" in paraphrase_union
    if skip_word_arms or not prompted_ok:
        g_prompted = None
        g_prompted_beats_surface = None
        g_prompted_median_better = None
    else:
        w_top1 = float(inst.get("top1_sem", 0.0)) if inst_ok else 0.0
        g_prompted = bool(
            prompted["top1"] >= w_top1 + 0.05
            or (inst_ok and prompted["acc_20way"] >= sem_matched + 0.05)
            or prompted["top1"] >= 0.12
        )
        g_prompted_beats_surface = headline_beats(prompted, surface_votes)
        g_prompted_median_better = median_rank_better(prompted, surface_votes)

    if not union_ok:
        g_union_beats_surface = None
        g_union_median_better = None
        g_mind_refines = None
    else:
        g_union_beats_surface = headline_beats(union_votes, surface_votes)
        g_union_median_better = median_rank_better(union_votes, surface_votes)
        sil_u = float(union_votes.get("tie_at_zero_frac", 1.0))
        sil_s = float(surface_votes.get("tie_at_zero_frac", 0.0))
        g_mind_refines = bool(
            g_union_beats_surface
            and sil_u <= sil_s + 0.03
            and (
                union_votes["top1"] >= prompted["top1"] - 1e-9
                if prompted_ok else True
            )
        )

    if not para_u_ok:
        g_paraphrase_breaks_silence = None
        g_paraphrase_novel_on_tape = None
    else:
        g_paraphrase_breaks_silence = silence_beats(paraphrase_union, surface_votes)
        g_paraphrase_novel_on_tape = bool(
            paraphrase_diag is not None
            and float(paraphrase_diag.get("novel_on_tape_frac", 0.0)) >= 0.25
        )

    helped_ids = [
        rid for rid in ("qwen05_base", "qwen05_instruct", "qwen15_instruct", "qwen3_instruct")
        if harvest_improved(trunk_results.get(rid) or {}, rid)
    ]
    collapsed_ids = [
        rid for rid in ("qwen05_base", "qwen05_instruct", "qwen15_instruct", "qwen3_instruct")
        if still_collapsed(trunk_results.get(rid) or {})
    ]
    harvest_helped = bool(helped_ids)
    any_overfit = bool(collapsed_ids) and not harvest_helped

    words_crush_learned = bool(
        inst_ok
        and (
            surface_votes["top1"] >= float(inst.get("top1_sem", 0.0)) + 0.08
            or (prompted_ok and prompted["top1"] >= float(inst.get("top1_sem", 0.0)) + 0.08)
        )
    )

    if not (base_ok and inst_ok) and not (merge_prior_arms and prior and prior.get("ladder")):
        overall = "INSTRUCT_TRUNK_INVALID"
        remap_261 = None
    elif g_paraphrase_breaks_silence:
        overall = "PARAPHRASE_BREAKS_SILENCE"
        remap_261 = "QUERY_MUST_BE_WORDS"
    elif g_mind_refines:
        overall = "MIND_REFINES_QUERY"
        remap_261 = "QUERY_MUST_BE_WORDS"
    elif words_crush_learned or (prompted_ok and g_prompted):
        overall = "WORDS_FORMULATE_QUERY"
        remap_261 = "QUERY_MUST_BE_WORDS"
    elif harvest_helped:
        overall = "HARVEST_FIXES_MIXER"
        remap_261 = "MIXER_WAS_DATA"
    elif any_overfit or bool(collapsed_ids):
        overall = "MIXER_OVERFIT"
        remap_261 = "MIXER_DEFECT"
    elif g_instruct_beats and (g_ladder or g_prompted) and (
        (inst_ok and inst["top1_sem"] >= 0.15)
        or (prompted_ok and prompted.get("top1", 0) >= 0.15)
    ):
        overall = "INSTRUCT_TRUNK_OK"
        remap_261 = "NO_WAS_TRUNK_SCALE"
    elif not g_instruct_beats and base_ok and inst_ok:
        overall = "NO_AT_TRUNK_SCALE"
        remap_261 = "NO_AT_TRUNK_SCALE"
    elif g_prompted and not g_instruct_beats:
        overall = "PROMPTED_QUERY_SIGNAL"
        remap_261 = "NL_QUERY_NO"
    else:
        overall = "NL_QUERY_NO"
        remap_261 = "NL_QUERY_NO"

    any_words_suffice = bool(
        union_ok and words_crush_learned and not g_union_beats_surface
    )
    paraphrase_useless = bool(para_u_ok and not g_paraphrase_breaks_silence)

    ref261 = RES / "stage261_decision.json"
    exam_parity = {"ref": str(ref261), "matched": None}
    if ref261.exists():
        try:
            r261 = json.loads(ref261.read_text(encoding="utf-8"))
            exam_parity = {
                "ref_slots": r261.get("slots"),
                "ref_exam_slots": r261.get("exam_slots"),
                "ref_n_eval": r261.get("n_eval"),
                "this_slots": len(vals),
                "this_exam_slots": n_exam,
                "this_n_eval": len(ev_items),
                "slots_match": r261.get("slots") == len(vals),
                "exam_slots_match": r261.get("exam_slots") == n_exam,
                "note": (
                    "Same construction seed=261; count mismatch → trunk ladder still "
                    "internally valid, but remap_261 needs a caveat."
                    if r261.get("slots") != len(vals) or r261.get("exam_slots") != n_exam
                    else "counts match published 261 decision"
                ),
            }
            log(f"  exam_parity vs 261: {json.dumps(exam_parity)}")
        except Exception as e:  # noqa: BLE001
            exam_parity = {"error": str(e)}

    out = {
        "stage": 266,
        "overall": overall,
        "remap_261": remap_261,
        "smoke": args.smoke,
        "seed": SEED,
        "train_reseed_per_trunk": True,
        "steps": steps,
        "slots": len(vals),
        "exam_slots": n_exam,
        "noise_slots": len(vals) - n_exam,
        "harvest_noise_pairs": len(harvest),
        "harvest_kind": "cross_mention",
        "harvest_capped_at_n_dist": True,
        "train_pool_n": len(train_pool),
        "bank_matches_pre_harvest": abs(len(vals) - 4352) <= 64,
        "pre_harvest_baseline": PRE_HARVEST,
        "overlap_median": med,
        "exam_parity_261": exam_parity,
        "fp_version": L.canonical_fp_version(),
        "gates": {
            "G_instruct_beats_base_matched": g_instruct_beats,
            "G_instruct_compare": g_instruct_note,
            "G_ladder_monotone": g_ladder,
            "G_ladder_note": g_ladder_note,
            "G_prompted_query": g_prompted,
            "G_prompted_beats_surface": g_prompted_beats_surface,
            "G_prompted_median_better": g_prompted_median_better,
            "G_union_beats_surface": g_union_beats_surface,
            "G_union_median_better": g_union_median_better,
            "G_mind_refines": g_mind_refines,
            "G_paraphrase_breaks_silence": g_paraphrase_breaks_silence,
            "G_paraphrase_novel_on_tape": g_paraphrase_novel_on_tape,
            "G_paraphrase_useless": paraphrase_useless,
            "G_words_crush_learned": words_crush_learned,
            "G_any_words_suffice": any_words_suffice,
            "G_mixer_overfit": bool(collapsed_ids) and not harvest_helped,
            "G_harvest_helped": harvest_helped,
            "harvest_helped_ids": helped_ids,
            "collapsed_ids": collapsed_ids,
            "instruct_alpha_collapsed": _alpha_off(inst),
            "headline_beats_means": "top1+0.03 or 20-way+0.05; median rank is separate",
            "silence_beats_means": "tie_at_zero −0.05 overall or low-overlap",
        },
        "fit_diag": {
            rid: {
                "fit_top1_sem": (trunk_results.get(rid) or {}).get("fit_top1_sem"),
                "eval_top1_sem": (trunk_results.get(rid) or {}).get("top1_sem"),
                "fit_acc_20way_sem": (trunk_results.get(rid) or {}).get("fit_acc_20way_sem"),
                "eval_acc_20way_sem": (trunk_results.get(rid) or {}).get("acc_20way_sem"),
                "eval_acc_20way_fp": (trunk_results.get(rid) or {}).get("acc_20way_fp"),
                "mixer_overfit": (trunk_results.get(rid) or {}).get("mixer_overfit"),
                "alpha": (trunk_results.get(rid) or {}).get("alpha"),
            }
            for rid in ("qwen05_base", "qwen05_instruct", "qwen15_instruct", "qwen3_instruct")
            if trunk_results.get(rid) and "top1_sem" in (trunk_results.get(rid) or {})
        },
        "matched_pair": {
            "base": (
                {k: base.get(k) for k in (
                    "model", "hidden", "acc_20way_fp", "acc_20way_sem", "top1_sem", "alpha",
                    "fit_top1_sem", "mixer_overfit",
                )} | {"verify_seed": base.get("verify_seed")}
            ) if base_ok else base,
            "instruct": {k: inst.get(k) for k in (
                "model", "hidden", "acc_20way_fp", "acc_20way_sem", "top1_sem", "alpha",
                "chat_template_used", "fit_top1_sem", "mixer_overfit",
            )} if inst_ok else inst,
            "delta_20way_sem": (
                (inst["acc_20way_sem"] - base["acc_20way_sem"]) if base_ok and inst_ok else None
            ),
            "delta_20way_fp_only": (
                (inst["acc_20way_fp"] - base["acc_20way_fp"]) if base_ok and inst_ok else None
            ),
        },
        "ladder": {rid: trunk_results.get(rid) for rid in (
            "qwen05_base", "qwen05_instruct", "qwen15_instruct", "qwen3_instruct",
        )},
        "surface_votes": surface_votes,
        "prompted_query": prompted,
        "union_votes": union_votes,
        "paraphrase": paraphrase,
        "paraphrase_union": paraphrase_union,
        "paraphrase_bridge": paraphrase_diag,
        "secondary": {
            "harvest_fixes_mixer": harvest_helped,
            "remap_if_harvest_only": "MIXER_WAS_DATA",
            "any_words_suffice": any_words_suffice,
            "paraphrase_useless": paraphrase_useless,
        },
        "note": (
            "Mind's only remaining chance is silence: novel paraphrase words that exist on "
            "tape (bridge), measured by tie_at_zero_frac — not top1. Keywords subtract noise "
            "(median better, coverage worse). Union∪keywords failed headlines → any words "
            "suffice. G_prompted_beats_surface is headline-only. Instruct alpha≈0 → "
            "fp+sem≡fp_only. QUERY_MUST_BE_WORDS when words crush trained W."
        ),
        "prompted_only_run": bool(args.prompted_only),
        "paraphrase_only_run": bool(args.paraphrase_only),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")

    def fmt_arm(r):
        if not r or "acc_20way_sem" not in r:
            return r.get("error", "n/a") if isinstance(r, dict) else "n/a"
        fit_t = r.get("fit_top1_sem")
        fit_s = f" / FIT top1_sem **{fit_t:.3f}**" if fit_t is not None else ""
        return (f"eval sem **{r['acc_20way_sem']:.3f}** / fp {r['acc_20way_fp']:.3f} "
                f"/ top1 {r['top1_sem']:.3f} / a={r['alpha']:.2f}{fit_s}")

    fit_rows = ""
    for rid in ("qwen05_base", "qwen05_instruct", "qwen15_instruct", "qwen3_instruct"):
        r = trunk_results.get(rid)
        if not r or "fit_top1_sem" not in r:
            continue
        fit_rows += (
            f"| {rid} | **{r['fit_top1_sem']:.3f}** | {r['top1_sem']:.3f} | "
            f"{r['fit_acc_20way_sem']:.3f} | {r['acc_20way_sem']:.3f} | "
            f"{r['acc_20way_fp']:.3f} | {r['mixer_overfit']} |\n"
        )

    mini = (
        f"# Stage 266 instruct trunk ladder\n\n"
        f"**{overall}** · remap_261=`{remap_261}` · bank={len(vals)} exam={n_exam}"
        f"{' · SMOKE' if args.smoke else ''}\n\n"
        f"## Fit vs eval (mixer diagnostic)\n\n"
        f"| trunk | FIT top1_sem | EVAL top1_sem | FIT 20way_sem | EVAL 20way_sem | EVAL fp | overfit |\n"
        f"|-------|-------------:|--------------:|--------------:|---------------:|--------:|:-------:|\n"
        f"{fit_rows}\n"
        f"## Matched pair\n\n"
        f"| trunk | metric |\n|-------|--------|\n"
        f"| 0.5B base | {fmt_arm(base)} |\n"
        f"| 0.5B-Instruct + chat template | {fmt_arm(inst)} |\n"
        f"| Δ sem 20-way | {out['matched_pair']['delta_20way_sem']} |\n"
        f"| Δ fp_only 20-way | {out['matched_pair'].get('delta_20way_fp_only')}\n\n"
        f"## Ladder\n\n"
        f"| id | result |\n|----|--------|\n"
        + "".join(
            f"| {rid} | {fmt_arm(trunk_results.get(rid))} |\n"
            for rid in ("qwen05_base", "qwen05_instruct", "qwen15_instruct", "qwen3_instruct")
        )
        + "\n## Word-vote arms (0 train)\n\n"
        + "| arm | top1 | median | 20-way | silence | low-ov silence | low-ov\\|vote |\n"
        + "|-----|-----:|-------:|-------:|--------:|---------------:|-------------:|\n"
        + (
            f"| surface words | {surface_votes['top1']:.3f} | {surface_votes['median_rank']:.1f} | "
            f"{surface_votes['acc_20way']:.3f} | {surface_votes['tie_at_zero_frac']:.3f} | "
            f"{surface_votes['silence'].get('tie_at_zero_frac_low_overlap', float('nan')):.3f} | "
            f"{surface_votes['silence'].get('top1_low_overlap_given_vote', float('nan')):.3f} |\n"
        )
        + (
            f"| prompted keywords | {prompted['top1']:.3f} | {prompted['median_rank']:.1f} | "
            f"{prompted['acc_20way']:.3f} | {prompted['tie_at_zero_frac']:.3f} | "
            f"{prompted['silence'].get('tie_at_zero_frac_low_overlap', float('nan')):.3f} | "
            f"{prompted['silence'].get('top1_low_overlap_given_vote', float('nan')):.3f} |\n"
            if prompted_ok else "| prompted keywords | n/a | n/a | n/a | n/a | n/a | n/a |\n"
        )
        + (
            f"| surface ∪ keywords | {union_votes['top1']:.3f} | {union_votes['median_rank']:.1f} | "
            f"{union_votes['acc_20way']:.3f} | {union_votes['tie_at_zero_frac']:.3f} | "
            f"{union_votes['silence'].get('tie_at_zero_frac_low_overlap', float('nan')):.3f} | "
            f"{union_votes['silence'].get('top1_low_overlap_given_vote', float('nan')):.3f} |\n"
            if union_ok else "| surface ∪ keywords | n/a | n/a | n/a | n/a | n/a | n/a |\n"
        )
        + (
            f"| paraphrase novel | {paraphrase['top1']:.3f} | {paraphrase['median_rank']:.1f} | "
            f"{paraphrase['acc_20way']:.3f} | {paraphrase['tie_at_zero_frac']:.3f} | "
            f"{paraphrase['silence'].get('tie_at_zero_frac_low_overlap', float('nan')):.3f} | "
            f"{paraphrase['silence'].get('top1_low_overlap_given_vote', float('nan')):.3f} |\n"
            if para_ok else "| paraphrase novel | n/a | n/a | n/a | n/a | n/a | n/a |\n"
        )
        + (
            f"| surface ∪ paraphrase | {paraphrase_union['top1']:.3f} | "
            f"{paraphrase_union['median_rank']:.1f} | {paraphrase_union['acc_20way']:.3f} | "
            f"{paraphrase_union['tie_at_zero_frac']:.3f} | "
            f"{paraphrase_union['silence'].get('tie_at_zero_frac_low_overlap', float('nan')):.3f} | "
            f"{paraphrase_union['silence'].get('top1_low_overlap_given_vote', float('nan')):.3f} |\n"
            if para_u_ok else "| surface ∪ paraphrase | n/a | n/a | n/a | n/a | n/a | n/a |\n"
        )
        + (
            f"| trained W (matched Instruct) | {inst.get('top1_sem', float('nan')):.3f} | — | "
            f"{sem_matched:.3f} | — | — | — |\n"
            if inst_ok else ""
        )
        + (
            f"\nParaphrase bridge: novel_on_tape="
            f"{(paraphrase_diag or {}).get('novel_on_tape_frac', float('nan')):.3f}, "
            f"queries_with_bridge="
            f"{(paraphrase_diag or {}).get('queries_with_bridge_word_frac', float('nan')):.3f}, "
            f"woken_silent="
            f"{(paraphrase_union or {}).get('surface_silent_woken_frac', float('nan')) if para_u_ok else float('nan'):.3f}\n"
            if para_ok or para_u_ok else ""
        )
        + f"\n## Gates\n\n"
        f"- G_instruct_beats_base_matched: **{g_instruct_beats}** ({g_instruct_note})\n"
        f"- G_ladder_monotone: **{g_ladder}** ({g_ladder_note})\n"
        f"- G_prompted_query: **{g_prompted}**\n"
        f"- G_prompted_beats_surface: **{g_prompted_beats_surface}** (headline: top1/20-way)\n"
        f"- G_prompted_median_better: **{g_prompted_median_better}**\n"
        f"- G_union_beats_surface: **{g_union_beats_surface}**\n"
        f"- G_mind_refines: **{g_mind_refines}**\n"
        f"- G_paraphrase_breaks_silence: **{g_paraphrase_breaks_silence}**\n"
        f"- G_paraphrase_novel_on_tape: **{g_paraphrase_novel_on_tape}**\n"
        f"- G_paraphrase_useless: **{paraphrase_useless}**\n"
        f"- G_words_crush_learned: **{words_crush_learned}**\n"
        f"- G_any_words_suffice: **{any_words_suffice}**\n"
        f"- G_mixer_overfit: **{out['gates']['G_mixer_overfit']}**\n"
        f"- G_harvest_helped: **{harvest_helped}**\n"
    )
    MINI.write_text(mini, encoding="utf-8")
    log(json.dumps({"overall": overall, "remap_261": remap_261, "gates": out["gates"]}, indent=2))
    log(f"wrote {DECISION} wall={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
