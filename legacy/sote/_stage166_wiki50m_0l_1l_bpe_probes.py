"""
Stage 166 — Wiki ~50M tokens: word 1L d128 h2 vs 0L vs BPE + understanding probes.

Plan: results/plan_wiki50m_probes_10h.md
Budget: ~10h. Waits for Stage165.

Steps:
  0) WikiText-103 → SOTE filter → 50M tokens → ≤8-word windows
  1) word_1L_d128_h2 @80k
  1b) word_1L_d256_h2 @80k  (underfit probe @50M)
  2) word_0L_last_d128 @80k
  3) bpe_1L_d128_h2 @80k
  4) HOLD + order-shuffle + same-last probes → decision

Run:
  python _stage166_wiki50m_0l_1l_bpe_probes.py
"""
from __future__ import annotations

import io
import json
import random
import re
import sys
import time
import traceback
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CHAR2ID,
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    _warmup_then_constant,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, _subsample  # noqa: E402
from _stage111_112_follow import _encode_words, eval_bpe_word_holds  # noqa: E402
from _stage150_155_clean_compare_pipeline import make_opt, metrics_pack  # noqa: E402
from _stage164_zerolayer_control import ZeroLayerWordLM  # noqa: E402

DEC165 = RES / "stage165_hard_floor_d16_d8_decision.json"
DEC = RES / "stage166_wiki50m_0l_1l_bpe_probes_decision.json"
LOG = RES / "_stage166_wiki50m_log.txt"
PLAN = RES / "plan_wiki50m_probes_10h.md"
RAW_ZIP = ROOT / "data" / "_wikitext103_v1.zip"
RAW_TRAIN = ROOT / "data" / "_wikitext103_train.txt"
CORPUS = ROOT / "data" / "external_wikitext103_50m_tokens_85.txt"
BPE_TOK = RES / "stage166_wiki50m_bpe_tokenizer.json"
META_DATA = RES / "stage166_wiki50m_data_meta.json"

TARGET_TOKENS = 50_000_000
MAX_WORD_LEN = 24
MIN_WORDS, MAX_WORDS = 3, 8
V_WORD_CAP = 40_000
V_BPE = 8000
FT_STEPS = 80_000
# Stretch / long-soak default when 166 finishes early
STRETCH_STEPS = 150_000
WARMUP = 200
EVAL_EVERY = 4000
SEED = 166
UNK = "<unk>"

TRUNK = {
    "batch": 8,
    "lr": 1e-3,
    "opt": "Adam",
    "wd": 0.0,
    "d": 128,
    "n_layer": 1,
    "n_head": 2,
    "fat_frac": 0.25,
    "word_max_len": 16,
    "bpe_max_len": 48,
    "n_positions": 64,
}


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF


def wait_165(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage165 ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC165.exists():
            d = json.loads(DEC165.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 165 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 165 ... {int(time.time()-t0)}s")
    raise TimeoutError("165 not ready")


def already_done(path: Path) -> dict | None:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("verdict"):
            log(f"[skip] {path.name}")
            return d
    return None


def write_dec(path: Path, out: dict):
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[write] {path.name}")


# ---------- data ----------

def ensure_wikitext103_train() -> Path:
    if RAW_TRAIN.exists() and RAW_TRAIN.stat().st_size > 10_000_000:
        log(f"[data] reuse {RAW_TRAIN.name} ({RAW_TRAIN.stat().st_size/1e6:.1f}MB)")
        return RAW_TRAIN
    RAW_TRAIN.parent.mkdir(exist_ok=True)
    urls = [
        "https://wikitext.smerity.com/wikitext-103-v1.zip",
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
    ]
    if not RAW_ZIP.exists() or RAW_ZIP.stat().st_size < 100_000:
        last_err = None
        for url in urls:
            try:
                log(f"[data] download WikiText-103 zip ... {url[:70]}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                # follow redirects manually if needed
                with urllib.request.urlopen(req, timeout=1200) as resp:
                    data = resp.read()
                if len(data) < 1_000_000:
                    raise RuntimeError(f"zip too small: {len(data)} bytes")
                RAW_ZIP.write_bytes(data)
                log(f"[data] wrote zip {RAW_ZIP.stat().st_size/1e6:.1f}MB")
                last_err = None
                break
            except Exception as e:
                last_err = e
                log(f"[data] download fail: {e}")
        if last_err and (not RAW_ZIP.exists() or RAW_ZIP.stat().st_size < 100_000):
            raise RuntimeError(f"WikiText-103 download failed: {last_err}")
    log("[data] extract wiki.train.tokens ...")
    with zipfile.ZipFile(RAW_ZIP, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith("wiki.train.tokens")]
        if not names:
            raise RuntimeError(f"wiki.train.tokens not in zip: {zf.namelist()[:20]}")
        with zf.open(names[0]) as src, RAW_TRAIN.open("wb") as dst:
            dst.write(src.read())
    log(f"[data] wrote {RAW_TRAIN} ({RAW_TRAIN.stat().st_size/1e6:.1f}MB)")
    return RAW_TRAIN


def build_wiki50m_corpus() -> tuple[list[str], dict]:
    if CORPUS.exists() and META_DATA.exists():
        meta = json.loads(META_DATA.read_text(encoding="utf-8"))
        phrases = [
            ln.strip()
            for ln in CORPUS.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if meta.get("target_tokens") == TARGET_TOKENS and len(phrases) > 10000:
            log(f"[data] reuse corpus n_phrases={len(phrases)} meta_tokens={meta.get('n_tokens_used')}")
            return phrases, meta

    raw_path = ensure_wikitext103_train()
    log(f"[data] filter SOTE charset + window; stop at {TARGET_TOKENS/1e6:.0f}M tokens ...")
    n_tokens = 0
    phrases = []
    seen = set()
    # stream by lines to limit RAM
    buf_words: list[str] = []
    with raw_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if n_tokens >= TARGET_TOKENS and len(phrases) > 50000:
                break
            text = line.lower()
            text = re.sub(r"[^a-z0-9\s]+", " ", text)
            words = [
                w
                for w in text.split()
                if w and all(c in CHAR2ID and c != " " for c in w) and len(w) <= MAX_WORD_LEN
            ]
            if not words:
                continue
            n_tokens += len(words)
            buf_words.extend(words)
            # emit sliding windows from buffer
            while len(buf_words) >= MIN_WORDS:
                take = min(MAX_WORDS, len(buf_words))
                win = buf_words[:take]
                # advance by ~half window to reduce exact dups but keep coverage
                step = max(1, take // 2)
                buf_words = buf_words[step:]
                if len(win) < MIN_WORDS:
                    continue
                ph = " ".join(win)
                if ph in seen:
                    continue
                seen.add(ph)
                phrases.append(ph)
                if n_tokens >= TARGET_TOKENS and len(phrases) >= 200000:
                    # enough windows once token budget hit
                    if len(phrases) >= 500000:
                        break
            if n_tokens >= TARGET_TOKENS and len(phrases) >= 500000:
                break

    # Cap phrase count for 10h trainability while keeping token claim in meta
    max_phrases = 800_000
    if len(phrases) > max_phrases:
        rng = random.Random(SEED)
        phrases = phrases[:]
        rng.shuffle(phrases)
        phrases = phrases[:max_phrases]

    meta = {
        "source": "wikitext-103-train",
        "target_tokens": TARGET_TOKENS,
        "n_tokens_used": min(n_tokens, TARGET_TOKENS) if n_tokens else n_tokens,
        "n_tokens_streamed": n_tokens,
        "n_phrases": len(phrases),
        "max_word_len": MAX_WORD_LEN,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "phrase_cap": max_phrases,
    }
    header = [
        "# SOTE WikiText-103 ~50M tokens (a-z0-9 space; Stage166)",
        f"# meta: {json.dumps(meta)}",
    ]
    CORPUS.write_text("\n".join(header + phrases) + "\n", encoding="utf-8")
    META_DATA.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[data] wrote {CORPUS.name} phrases={len(phrases)} tokens_streamed={n_tokens}")
    return phrases, meta


def wiki_split(phrases: list[str], seed: int = SEED):
    """Natural split (no TS fat path triples). Light fat = frequent lines copies."""
    rng = random.Random(seed)
    ph = phrases[:]
    rng.shuffle(ph)
    n = len(ph)
    n_hold = max(2000, int(0.12 * n))
    hold = ph[:n_hold]
    train_raw = ph[n_hold:]
    # light fat: top bigram-bearing lines duplicated
    ctr = Counter(train_raw)
    top = [p for p, _ in ctr.most_common(500)]
    fat_lines = []
    for p in top:
        for _ in range(8):
            fat_lines.append({"words": p.split(), "phrase": p, "bucket": "fat_train", "split": "fat"})
    train = [{"words": p.split(), "phrase": p, "bucket": "story_train", "split": "train"} for p in train_raw]
    hold_lines = [{"words": p.split(), "phrase": p, "bucket": "hold", "split": "hold"} for p in hold]
    # split hold into seen-like / story-like by overlap with train unigram
    train_words = Counter(w for ln in train for w in ln["words"])
    hold_seen, hold_story = [], []
    for ln in hold_lines:
        rare = sum(1 for w in ln["words"] if train_words[w] <= 2)
        if rare == 0:
            hold_seen.append(ln)
        else:
            hold_story.append(ln)
    if not hold_story:
        hold_story = hold_lines[len(hold_lines) // 2 :]
        hold_seen = hold_lines[: len(hold_lines) // 2]
    if not hold_seen:
        hold_seen = hold_story[: max(1, len(hold_story) // 5)]
    for ln in train + fat_lines + hold_seen + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
    meta = {
        "n_train": len(train),
        "n_fat": len(fat_lines),
        "n_hold_seen": len(hold_seen),
        "n_hold_story": len(hold_story),
    }
    return train + fat_lines, hold_seen, hold_story, meta


def build_stoi(train_lines, hold_lines, cap=V_WORD_CAP):
    ctr = Counter(w for ln in train_lines for w in ln["words"])
    vocab = [UNK] + [w for w, _ in ctr.most_common(cap) if w != UNK]
    stoi = {w: i for i, w in enumerate(vocab)}
    n_train_tok = sum(len(ln["words"]) for ln in train_lines)
    n_cov = sum(1 for ln in train_lines for w in ln["words"] if w in stoi and w != UNK)
    # also map hold OOV → UNK at encode time
    return vocab, stoi, {"V": len(vocab), "train_coverage": n_cov / max(n_train_tok, 1), "n_types_raw": len(ctr)}


def map_ids(words, stoi):
    return [stoi.get(w, stoi[UNK]) for w in words]


# ---------- probes ----------

@torch.no_grad()
def probe_order_and_same_last(model, hold_lines, surfaces, stoi, device, max_lines=800, seed=SEED):
    rng = random.Random(seed)
    lines = hold_lines if len(hold_lines) <= max_lines else rng.sample(hold_lines, max_lines)

    clean_n = shuf_n = clean_h = shuf_h = 0
    # same-last buckets
    by_last = defaultdict(list)  # last -> list of (prefix_words, gold)

    for ln in lines:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            pref = ws[:t]
            by_last[pref[-1]].append((pref, gold))

            # clean
            ids = map_ids(pref, stoi)[-model.max_len :]
            if not ids:
                continue
            pred = surfaces[model.pred_id(ids)]
            # decode UNK surface
            ok = int(pred == gold or (pred == UNK and gold not in stoi))
            # stricter: only exact gold surface match when gold in vocab
            if gold in stoi and stoi[gold] != stoi[UNK]:
                ok = int(pred == gold)
            else:
                ok = 0
            clean_n += 1
            clean_h += ok

            # shuffle prefix (keep length); last token fixed so bigram cue remains —
            # stronger test: shuffle *all* including last? For order sensitivity of context,
            # shuffle all but last (classic: full shuffle of prefix).
            shuf = pref[:]
            if len(shuf) >= 2:
                body = shuf[:-1]
                rng.shuffle(body)
                shuf = body + [shuf[-1]]
            ids_s = map_ids(shuf, stoi)[-model.max_len :]
            pred_s = surfaces[model.pred_id(ids_s)]
            ok_s = int(pred_s == gold) if gold in stoi and stoi[gold] != stoi[UNK] else 0
            shuf_n += 1
            shuf_h += ok_s

    clean_acc = clean_h / max(clean_n, 1)
    shuf_acc = shuf_h / max(shuf_n, 1)
    order_drop = clean_acc - shuf_acc

    # same-last: only lasts with ≥2 distinct golds
    sl_n = sl_h = 0
    maj_h = 0
    amb_lasts = 0
    for last, items in by_last.items():
        golds = [g for _, g in items]
        if len(set(golds)) < 2:
            continue
        amb_lasts += 1
        maj = Counter(golds).most_common(1)[0][0]
        for pref, gold in items:
            if gold not in stoi or stoi[gold] == stoi[UNK]:
                continue
            ids = map_ids(pref, stoi)[-model.max_len :]
            pred = surfaces[model.pred_id(ids)]
            sl_n += 1
            sl_h += int(pred == gold)
            maj_h += int(maj == gold)

    same_last_acc = sl_h / max(sl_n, 1)
    majority_acc = maj_h / max(sl_n, 1)
    same_last_lift = same_last_acc - majority_acc

    return {
        "clean_acc": clean_acc,
        "shuffle_acc": shuf_acc,
        "order_drop": order_drop,
        "n_order": clean_n,
        "same_last_acc": same_last_acc,
        "majority_last_acc": majority_acc,
        "same_last_lift_vs_majority": same_last_lift,
        "n_same_last": sl_n,
        "n_ambiguous_lasts": amb_lasts,
    }


# ---------- train arms ----------

def train_word_arm(tag, phrases, device, cfg, stack, *, n_layer, n_head, d, steps=FT_STEPS):
    log(f"\n======== WORD {tag} {n_layer}L/{n_head}H d={d} steps={steps} fat={TRUNK['fat_frac']} ========")
    train, hold_seen, hold_story, meta = wiki_split(phrases)
    surfaces, stoi, vmeta = build_stoi(train, hold_seen + hold_story)
    log(f"[vocab] {vmeta}")

    # fp init; UNK = mean of known fps noise
    fps = []
    for w in surfaces:
        if w == UNK:
            fps.append(torch.zeros(256, device=device))
        else:
            try:
                fps.append(F.normalize(stack.w(w).detach(), dim=-1))
            except Exception:
                fps.append(torch.zeros(256, device=device))
    fps_t = torch.stack(fps, 0)
    if (surfaces[0] == UNK) or UNK in stoi:
        known = fps_t[1:] if surfaces[0] == UNK else fps_t
        known = known[known.norm(dim=-1) > 0]
        if len(known):
            fps_t[stoi[UNK]] = F.normalize(known.mean(0) + 0.01 * torch.randn_like(known[0]), dim=-1)

    model = WordIdTransformer(len(surfaces), d, n_head, n_layer, TRUNK["word_max_len"], 0.1).to(device)
    model.init_from_fps(fps_t)
    if d > fps_t.shape[1]:
        with torch.no_grad():
            model.tok.weight[: len(surfaces), fps_t.shape[1] :].normal_(std=0.02)
            model.tok.weight[: len(surfaces)] = F.normalize(model.tok.weight[: len(surfaces)], dim=-1)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[word] V={len(surfaces)} params={n_params/1e6:.2f}M")

    # remap pairs through UNK
    def pairs_of(lines, max_n=None, seed=0):
        use = lines if max_n is None else _subsample(lines, max_n, seed)
        out = []
        for li, ln in enumerate(use):
            ids = map_ids(ln["words"], stoi)
            for t in range(1, len(ids)):
                out.append(
                    {
                        "line_i": li,
                        "prefix_word_ids": ids[:t],
                        "target_word": surfaces[ids[t]],
                        "target_word_id": ids[t],
                        "prefix_len": t,
                        "split": ln.get("split", "x"),
                        "phrase": ln.get("phrase", ""),
                    }
                )
        return out, use

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p, _ = pairs_of(fat)
    story_p, _ = pairs_of(story, 80000, seed=stable_seed(tag, "story"))
    if not fat_p:
        fat_p = story_p

    ev_seen = _subsample(hold_seen, min(400, len(hold_seen)), stable_seed(tag, "evs"))
    ev_story = _subsample(hold_story, min(400, len(hold_story)), stable_seed(tag, "evy"))
    fin_seen = _subsample(hold_seen, min(800, len(hold_seen)), stable_seed(tag, "fins"))
    fin_story = _subsample(hold_story, min(800, len(hold_story)), stable_seed(tag, "finy"))

    # eval helpers need hold lines with words; use original words (pred compared to gold surface)
    opt = make_opt(model.parameters(), TRUNK)
    rr = random.Random(stable_seed(tag, "train"))
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []
    batch, fat_frac = TRUNK["batch"], TRUNK["fat_frac"]

    def snap(step):
        model.eval()
        # build temporary suite using mapped ids via custom loop on roles ALL/obj
        st = _eval_mapped(model, ev_story, surfaces, stoi, device)
        s = _eval_mapped(model, ev_seen, surfaces, stoi, device)
        sall, obj = st["all"], s["obj"]
        curve.append({"step": step, "story_all": sall, "obj": obj})
        log(f"  [{tag}] step {step}: SEEN obj={100*obj:.1f}% | STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(story_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.train()

    model.train()
    snap(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, TRUNK["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac)))
        batch_ex = [rr.choice(fat_p) for _ in range(n_fat)] + [rr.choice(story_p) for _ in range(batch - n_fat)]
        packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    fs = _eval_mapped(model, fin_seen, surfaces, stoi, device)
    ft = _eval_mapped(model, fin_story, surfaces, stoi, device)
    probes = probe_order_and_same_last(model, fin_story, surfaces, stoi, device)
    ck = CKPT / f"stage166_{tag}.pt"
    torch.save({"word_tf": best["state"], "surfaces": surfaces, "stoi": stoi, "tag": tag}, ck)
    return {
        "arm": tag,
        "unit": "word_fp",
        "arch": {"d": d, "n_layer": n_layer, "n_head": n_head},
        "V": len(surfaces),
        "vocab_meta": vmeta,
        "params": n_params,
        "steps": steps,
        "mix_meta": meta,
        "curve": curve,
        "ckpt": str(ck),
        "seen_obj": fs["obj"],
        "story_all": ft["all"],
        "story_obj": ft["obj"],
        "probes": probes,
        "trunk": TRUNK,
    }


@torch.no_grad()
def _eval_mapped(model, hold_lines, surfaces, stoi, device):
    from train import _role, RELS

    roles = defaultdict(lambda: {"n": 0, "h": 0})
    obj = {"n": 0, "h": 0}
    for ln in hold_lines:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            if gold not in stoi or stoi[gold] == stoi.get(UNK, -1):
                continue
            ids = map_ids(ws[:t], stoi)[-model.max_len :]
            pred = surfaces[model.pred_id(ids)]
            ok = int(pred == gold)
            ex = {"target_word": gold, "prefix_len": t}
            role = _role(ex, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h"] += ok
            if t >= 1 and ws[t - 1] in RELS:
                obj["n"] += 1
                obj["h"] += ok
    return {
        "all": roles["ALL"]["h"] / max(roles["ALL"]["n"], 1),
        "obj": obj["h"] / max(obj["n"], 1),
        "roles": {k: v["h"] / max(v["n"], 1) for k, v in roles.items()},
    }


def train_zero_arm(tag, phrases, device, cfg, stack, *, d=128, steps=FT_STEPS):
    log(f"\n======== ZERO {tag} d={d} steps={steps} ========")
    train, hold_seen, hold_story, meta = wiki_split(phrases, seed=SEED + 1)
    surfaces, stoi, vmeta = build_stoi(train, hold_seen + hold_story)
    log(f"[vocab] {vmeta}")
    fps = []
    for w in surfaces:
        if w == UNK:
            fps.append(torch.zeros(256, device=device))
        else:
            try:
                fps.append(F.normalize(stack.w(w).detach(), dim=-1))
            except Exception:
                fps.append(torch.zeros(256, device=device))
    fps_t = torch.stack(fps, 0)
    model = ZeroLayerWordLM(len(surfaces), d_model=d, max_len=TRUNK["word_max_len"], mode="last").to(device)
    model.init_from_fps(fps_t)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[zero] V={len(surfaces)} params={n_params/1e6:.2f}M")

    def pairs_of(lines, max_n=None, seed=0):
        use = lines if max_n is None else _subsample(lines, max_n, seed)
        out = []
        for li, ln in enumerate(use):
            ids = map_ids(ln["words"], stoi)
            for t in range(1, len(ids)):
                out.append(
                    {
                        "line_i": li,
                        "prefix_word_ids": ids[:t],
                        "target_word": surfaces[ids[t]],
                        "prefix_len": t,
                        "split": ln.get("split", "x"),
                        "phrase": ln.get("phrase", ""),
                    }
                )
        return out

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = pairs_of(fat)
    story_p = pairs_of(story, 80000, seed=3)
    if not fat_p:
        fat_p = story_p
    ev_story = _subsample(hold_story, min(400, len(hold_story)), 11)
    fin_seen = _subsample(hold_seen, min(800, len(hold_seen)), 20)
    fin_story = _subsample(hold_story, min(800, len(hold_story)), 21)
    opt = make_opt(model.parameters(), TRUNK)
    rr = random.Random(7)
    best = {"story_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    curve = []
    batch, fat_frac = TRUNK["batch"], TRUNK["fat_frac"]

    def snap(step):
        model.eval()
        st = _eval_mapped(model, ev_story, surfaces, stoi, device)
        s = _eval_mapped(model, fin_seen[:200] if fin_seen else ev_story, surfaces, stoi, device)
        curve.append({"step": step, "story_all": st["all"], "obj": s["obj"]})
        log(f"  [{tag}] step {step}: STORY={100*st['all']:.1f}%")
        if (st["all"], s["obj"]) >= (best["story_all"], best["obj"]):
            best.update(story_all=st["all"], obj=s["obj"], state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.train()

    model.train()
    snap(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, TRUNK["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac)))
        batch_ex = [rr.choice(fat_p) for _ in range(n_fat)] + [rr.choice(story_p) for _ in range(batch - n_fat)]
        packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    fs = _eval_mapped(model, fin_seen, surfaces, stoi, device)
    ft = _eval_mapped(model, fin_story, surfaces, stoi, device)
    probes = probe_order_and_same_last(model, fin_story, surfaces, stoi, device)
    ck = CKPT / f"stage166_{tag}.pt"
    torch.save({"zero_lm": best["state"], "surfaces": surfaces, "stoi": stoi}, ck)
    return {
        "arm": tag,
        "unit": "word_0L_last",
        "arch": {"d": d, "n_layer": 0, "n_head": 0},
        "V": len(surfaces),
        "vocab_meta": vmeta,
        "params": n_params,
        "steps": steps,
        "mix_meta": meta,
        "curve": curve,
        "ckpt": str(ck),
        "seen_obj": fs["obj"],
        "story_all": ft["all"],
        "story_obj": ft["obj"],
        "probes": probes,
        "trunk": TRUNK,
    }


def train_bpe_arm(tag, phrases, device, cfg, *, steps=FT_STEPS):
    log(f"\n======== BPE {tag} 1L/2H d=128 steps={steps} ========")
    if BPE_TOK.exists():
        tok = Tokenizer.from_file(str(BPE_TOK))
        log(f"[bpe] reuse {BPE_TOK.name}")
    else:
        log(f"[bpe] train V={V_BPE} on n={len(phrases)}")
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=V_BPE,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            show_progress=False,
        )
        tok.train_from_iterator(phrases, trainer=trainer)
        tok.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", tok.token_to_id("[BOS]")), ("[EOS]", tok.token_to_id("[EOS]"))],
        )
        tok.save(str(BPE_TOK))

    train, hold_seen, hold_story, meta = wiki_split(phrases, seed=SEED + 2)
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()
    fat_seqs, story_seqs = [], []
    bmax = TRUNK["bpe_max_len"]
    for ln in train:
        ids = _encode_words(tok, ln["words"], max_len=bmax, bos=bos, eos=eos, pad=pad)
        if len(ids) < 3:
            continue
        if ln.get("bucket") == "fat_train":
            fat_seqs.append(ids)
        else:
            story_seqs.append(ids)
    if not fat_seqs:
        fat_seqs = story_seqs
    conf = GPT2Config(
        vocab_size=V,
        n_positions=max(TRUNK["n_positions"], bmax + 2),
        n_embd=128,
        n_layer=1,
        n_head=2,
        n_inner=4 * 128,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    model = GPT2LMHeadModel(conf).to(device)
    with torch.no_grad():
        model.transformer.wte.weight.normal_(std=0.02)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[bpe] V={V} params={n_params/1e6:.2f}M fat={len(fat_seqs)} story={len(story_seqs)}")
    opt = make_opt(model.parameters(), TRUNK)
    rr = random.Random(9)
    ev_story = _subsample(hold_story, min(400, len(hold_story)), 11)
    fin_story = _subsample(hold_story, min(800, len(hold_story)), 21)
    fin_seen = _subsample(hold_seen, min(800, len(hold_seen)), 20)
    best = {"story_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    curve = []
    batch, fat_frac = TRUNK["batch"], TRUNK["fat_frac"]

    def snap(step):
        model.eval()
        st = eval_bpe_word_holds(model, tok, ev_story, device, max_n_lines=len(ev_story), encode_max_len=bmax)
        s = eval_bpe_word_holds(model, tok, fin_seen[:200] if fin_seen else ev_story, device, max_n_lines=min(200, len(fin_seen) or 1), encode_max_len=bmax)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        obj = s["obj"]["hit1"]
        curve.append({"step": step, "story_all": sall, "obj": obj})
        log(f"  [{tag}] step {step}: STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(story_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.train()

    model.train()
    snap(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, TRUNK["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac)))
        seqs = [rr.choice(fat_seqs) for _ in range(n_fat)] + [rr.choice(story_seqs) for _ in range(batch - n_fat)]
        maxlen = max(len(s) for s in seqs)
        x = torch.full((batch, maxlen), pad, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        labels = x.clone()
        labels[labels == pad] = -100
        loss = model(x, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    ft = eval_bpe_word_holds(model, tok, fin_story, device, max_n_lines=len(fin_story), encode_max_len=bmax)
    fs = eval_bpe_word_holds(model, tok, fin_seen, device, max_n_lines=len(fin_seen), encode_max_len=bmax)
    # BPE order probe: approximate via word-hold clean vs shuffled words then encode
    probes = _bpe_order_probe(model, tok, fin_story, device, bmax)
    ck = CKPT / f"stage166_{tag}.pt"
    torch.save({"gpt2": best["state"], "tag": tag}, ck)
    return {
        "arm": tag,
        "unit": "ws_bpe",
        "arch": {"d": 128, "n_layer": 1, "n_head": 2},
        "V": V,
        "params": n_params,
        "steps": steps,
        "mix_meta": meta,
        "curve": curve,
        "ckpt": str(ck),
        "seen_obj": fs["obj"]["hit1"],
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "probes": probes,
        "trunk": TRUNK,
    }


@torch.no_grad()
def _bpe_order_probe(model, tok, hold_lines, device, bmax, max_lines=400, seed=SEED):
    """Word-level clean vs prefix-shuffle then BPE-encode (approx order_drop)."""
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    rng = random.Random(seed)
    lines = hold_lines if len(hold_lines) <= max_lines else rng.sample(hold_lines, max_lines)
    clean = eval_bpe_word_holds(model, tok, lines, device, max_n_lines=len(lines), encode_max_len=bmax)
    # shuffled copies
    shuf_lines = []
    for ln in lines:
        ws = ln["words"][:]
        if len(ws) >= 3:
            body = ws[:-1]
            rng.shuffle(body)
            ws = body + [ws[-1]]
        shuf_lines.append({**ln, "words": ws})
    shuf = eval_bpe_word_holds(model, tok, shuf_lines, device, max_n_lines=len(shuf_lines), encode_max_len=bmax)
    c = clean["roles"].get("ALL", {}).get("hit1", 0.0)
    s = shuf["roles"].get("ALL", {}).get("hit1", 0.0)
    return {
        "clean_acc": c,
        "shuffle_acc": s,
        "order_drop": c - s,
        "same_last_acc": None,
        "majority_last_acc": None,
        "same_last_lift_vs_majority": None,
        "note": "BPE same-last not implemented; order via word-shuffle then encode",
    }


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"166 wiki50m start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan={PLAN}")
    if already_done(DEC):
        return 0
    try:
        up = wait_165()
        t_wall0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = Config()
        cfg.max_word_len = MAX_WORD_LEN
        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

        log("\n##### STEP 0: data #####")
        phrases, data_meta = build_wiki50m_corpus()
        log(f"[step0] phrases={len(phrases)} tokens_meta={data_meta}")

        arms = {}
        log("\n##### STEP 1: word 1L d128 h2 #####")
        arms["word_1L_d128_h2"] = train_word_arm(
            "word_1L_d128_h2", phrases, device, cfg, stack, n_layer=1, n_head=2, d=128
        )

        log("\n##### STEP 1b: word 1L d256 h2 (underfit probe) #####")
        arms["word_1L_d256_h2"] = train_word_arm(
            "word_1L_d256_h2", phrases, device, cfg, stack, n_layer=1, n_head=2, d=256
        )

        log("\n##### STEP 2: word 0L last d128 #####")
        arms["word_0L_last_d128"] = train_zero_arm("word_0L_last_d128", phrases, device, cfg, stack, d=128)

        log("\n##### STEP 3: BPE 1L d128 h2 #####")
        arms["bpe_1L_d128_h2"] = train_bpe_arm("bpe_1L_d128_h2", phrases, device, cfg)

        w1 = arms["word_1L_d128_h2"]
        w256 = arms["word_1L_d256_h2"]
        z0 = arms["word_0L_last_d128"]
        bp = arms["bpe_1L_d128_h2"]
        gap_all = w1["story_all"] - z0["story_all"]
        gap_ord = (w1["probes"]["order_drop"] - z0["probes"]["order_drop"])
        gap_sl = (w1["probes"].get("same_last_lift_vs_majority") or 0) - (
            z0["probes"].get("same_last_lift_vs_majority") or 0
        )
        unit_gap = bp["story_all"] - w1["story_all"]
        width_gap = w256["story_all"] - w1["story_all"]
        underfit = width_gap >= 0.02

        if gap_all >= 0.03 or gap_ord >= 0.05 or gap_sl >= 0.05:
            verdict = "ATTENTION_MATTERS_ON_WIKI"
        elif underfit:
            verdict = "UNDERFIT_AT_50M"
        elif gap_all < 0.015 and gap_ord < 0.03 and gap_sl < 0.03:
            verdict = "STILL_REDUNDANT"
        else:
            verdict = "MIXED"
        # If both attention matters and underfit, prefer compound tag
        if (gap_all >= 0.03 or gap_ord >= 0.05 or gap_sl >= 0.05) and underfit:
            verdict = "ATTENTION_AND_UNDERFIT"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "wiki50m_0l_1l_bpe_probes",
            "plan": str(PLAN),
            "data": data_meta,
            "trunk": TRUNK,
            "steps": FT_STEPS,
            "arms": arms,
            "gaps": {
                "word_1L_d128_minus_0L_all": gap_all,
                "word_1L_d256_minus_d128_all": width_gap,
                "order_drop_1L_d128_minus_0L": gap_ord,
                "same_last_lift_1L_d128_minus_0L": gap_sl,
                "bpe_minus_word_1L_d128_all": unit_gap,
            },
            "underfit_at_50m": underfit,
            "verdict": verdict,
            "wall_hours": (time.time() - t_wall0) / 3600,
            "upstream_165": up.get("verdict"),
            "ref100": REF100,
            "ref_ts_164": "ATTENTION_REDUNDANT on TinyStories; this dig tests wiki harder regime + d256 underfit",
        }
        write_dec(DEC, out)
        log(
            f"[166] {verdict} d128={100*w1['story_all']:.1f}% d256={100*w256['story_all']:.1f}% "
            f"0L={100*z0['story_all']:.1f}% BPE={100*bp['story_all']:.1f}% | "
            f"width={100*width_gap:+.1f}pp gap_all={100*gap_all:+.1f}pp "
            f"ord={100*gap_ord:+.1f}pp | wall={out['wall_hours']:.2f}h"
        )
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
