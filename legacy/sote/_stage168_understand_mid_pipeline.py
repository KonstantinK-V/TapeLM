"""
Stage 168 — SOTE understanding mid-path (CP0–CP4).

Plan: results/plan_sote_understand_mid_30h.md
Cue (CP3): dual-channel separate embedding (not special-token-in-seq).

Usage:
  python _stage168_understand_mid_pipeline.py --cp 0
  python _stage168_understand_mid_pipeline.py --cp 1
  ...
  python _stage168_understand_mid_pipeline.py --cp all
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    _warmup_then_constant,
    collate_word_id_batch,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, _subsample  # noqa: E402
from _stage150_155_clean_compare_pipeline import make_opt  # noqa: E402
from _stage164_zerolayer_control import ZeroLayerWordLM  # noqa: E402
import _stage166_wiki50m_0l_1l_bpe_probes as s166  # noqa: E402

PLAN = RES / "plan_sote_understand_mid_30h.md"
RAW_ZIP = ROOT / "data" / "_wikitext103_v1.zip"
RAW_TRAIN = ROOT / "data" / "_wikitext103_train.txt"
CORPUS = ROOT / "data" / "external_wikitext103_rich_understand_168.txt"
BATTERY = RES / "stage168_understanding_battery.jsonl"
META0 = RES / "stage168_cp0_data_meta.json"

TARGET_TOKENS = 30_000_000  # mid-path: 30M rich (3050-friendly)
MAX_WORD_LEN = 24
MIN_WORDS, MAX_WORDS = 4, 16
V_CAP = 40_000
UNK = "<unk>"
SEED = 168
WARMUP = 200
EVAL_EVERY = 4000
FT_STEPS_CP1 = 60_000
FT_STEPS_CP2_THIN = 60_000
FT_STEPS_CP2_DEEP = 80_000
FT_STEPS_CP3 = 80_000
DEEP_LAYERS = 4
DEEP_HEADS = 4

# Keep letters/digits/punct/specials; drop most other unicode
TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[.,!?;:\"'()\[\]{}\-%/$]"
)
PUNCT_CHARS = set(".,!?;:\"'()[]{}-%/$")

TRUNK = {
    "batch": 8,
    "lr": 1e-3,
    "opt": "Adam",
    "wd": 0.0,
    "d": 128,
    "n_layer": 1,
    "n_head": 2,
    "fat_frac": 0.15,  # light — understanding dig, not fat race
    "word_max_len": 16,
}


def log_to(path: Path, msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_mini(path: Path, title: str, bullets: list[str], verdict: str):
    body = [f"# {title}", "", f"**Verdict:** `{verdict}`", ""] + [f"- {b}" for b in bullets] + [""]
    path.write_text("\n".join(body), encoding="utf-8")


def stable_seed(*parts) -> int:
    import zlib

    s = str(SEED) + "|" + "|".join(str(p) for p in parts)
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


# ----- tokenization / orthography -----

def tokenize_rich(line: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(line) if t]


def case_tag(tok: str) -> str:
    if tok in PUNCT_CHARS or all(c in PUNCT_CHARS for c in tok):
        return "punct"
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return "other"
    if all(c.isupper() for c in letters):
        return "upper"
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "title"
    if all(c.islower() for c in letters):
        return "lower"
    return "mixed"


def alnum_core(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


# ----- CP0 -----

def ensure_raw_wiki(log) -> Path:
    if RAW_TRAIN.exists() and RAW_TRAIN.stat().st_size > 10_000_000:
        log(f"[data] reuse {RAW_TRAIN.name}")
        return RAW_TRAIN
    return s166.ensure_wikitext103_train()


def build_rich_corpus(log) -> tuple[list[list[str]], dict]:
    if CORPUS.exists() and META0.exists():
        meta = json.loads(META0.read_text(encoding="utf-8"))
        phrases = []
        for ln in CORPUS.read_text(encoding="utf-8").splitlines():
            if ln.strip() and not ln.startswith("#"):
                phrases.append(ln.strip().split("\t")[0].split())
        if meta.get("n_tokens_used", 0) >= TARGET_TOKENS // 2 and len(phrases) > 5000:
            log(f"[data] reuse rich corpus phrases={len(phrases)} tokens={meta.get('n_tokens_used')}")
            return phrases, meta

    raw = ensure_raw_wiki(log)
    log(f"[data] rich tokenize (case+punct), target {TARGET_TOKENS/1e6:.0f}M tokens ...")
    n_tokens = 0
    phrases: list[list[str]] = []
    seen = set()
    buf: list[str] = []
    case_counts = Counter()
    punct_counts = Counter()

    with raw.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if n_tokens >= TARGET_TOKENS and len(phrases) > 20000:
                break
            if line.startswith(" = ") or not line.strip():
                continue
            toks = tokenize_rich(line)
            if not toks:
                continue
            # filter overlong alnum pieces
            clean = []
            for t in toks:
                if t in PUNCT_CHARS or len(alnum_core(t)) <= MAX_WORD_LEN:
                    clean.append(t)
                    case_counts[case_tag(t)] += 1
                    if t in PUNCT_CHARS:
                        punct_counts[t] += 1
            if not clean:
                continue
            n_tokens += len(clean)
            buf.extend(clean)
            # emit non-overlapping-ish windows of length MAX_WORDS, plus a mid length
            while len(buf) >= MAX_WORDS:
                for L in (MAX_WORDS, max(MIN_WORDS, MAX_WORDS // 2)):
                    ph = buf[:L]
                    key = " ".join(ph)
                    if key not in seen:
                        seen.add(key)
                        phrases.append(ph)
                buf = buf[MAX_WORDS:]
                if len(phrases) >= 500_000:
                    break
            if len(phrases) >= 500_000:
                break

    # write corpus as space-joined (punct already separate tokens)
    CORPUS.parent.mkdir(exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as out:
        out.write("# Stage168 rich wiki (case+punct preserved as tokens)\n")
        for ph in phrases:
            out.write(" ".join(ph) + "\n")

    meta = {
        "source": "wikitext-103-train",
        "charset": "case+punct+digits+basic_specials",
        "target_tokens": TARGET_TOKENS,
        "n_tokens_used": n_tokens,
        "n_phrases": len(phrases),
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "case_tag_counts": dict(case_counts),
        "punct_counts": dict(punct_counts.most_common(30)),
        "token_re": TOKEN_RE.pattern,
    }
    write_json(META0, meta)
    log(f"[data] wrote {CORPUS.name} phrases={len(phrases)} tokens_streamed~{n_tokens}")
    return phrases, meta


def build_battery(phrases: list[list[str]], log, max_items=8000) -> list[dict]:
    """Forced same-last disambiguation items from corpus statistics."""
    if BATTERY.exists() and BATTERY.stat().st_size > 10000:
        items = []
        for ln in BATTERY.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                items.append(json.loads(ln))
        if len(items) >= 500:
            log(f"[battery] reuse n={len(items)}")
            return items

    next_by_last: dict[str, Counter] = defaultdict(Counter)
    examples_by_last: dict[str, list] = defaultdict(list)
    for ph in phrases:
        for t in range(1, len(ph)):
            last, nxt = ph[t - 1], ph[t]
            next_by_last[last][nxt] += 1
            if len(examples_by_last[last]) < 40:
                examples_by_last[last].append({"prefix": ph[:t], "gold": nxt})

    items = []
    for last, ctr in next_by_last.items():
        commons = [(w, c) for w, c in ctr.most_common(8) if c >= 3]
        if len(commons) < 2:
            continue
        maj_w, maj_c = commons[0]
        # need at least one non-majority gold with support
        alts = [(w, c) for w, c in commons[1:] if w != maj_w]
        if not alts:
            continue
        exs = examples_by_last[last]
        # pick up to 2 examples whose gold is not majority, and 1 majority
        nonmaj = [e for e in exs if e["gold"] != maj_w][:3]
        majex = [e for e in exs if e["gold"] == maj_w][:2]
        for e in nonmaj + majex:
            items.append(
                {
                    "last": last,
                    "prefix": e["prefix"],
                    "gold": e["gold"],
                    "majority_next": maj_w,
                    "majority_frac": maj_c / max(sum(ctr.values()), 1),
                    "n_next_types": len(ctr),
                }
            )
        if len(items) >= max_items:
            break

    rr = random.Random(SEED)
    rr.shuffle(items)
    items = items[:max_items]
    with BATTERY.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_amb = sum(1 for it in items if it["gold"] != it["majority_next"])
    log(f"[battery] wrote n={len(items)} nonmajority_gold={n_amb}")
    return items


def run_cp0() -> int:
    logp = RES / "_stage168_cp0_log.txt"
    if not logp.exists():
        logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    log(f"CP0 start {datetime.now(timezone.utc).isoformat()}")
    try:
        phrases, meta = build_rich_corpus(log)
        battery = build_battery(phrases, log)
        n_amb = sum(1 for it in battery if it["gold"] != it["majority_next"])
        maj_acc = sum(1 for it in battery if it["gold"] == it["majority_next"]) / max(len(battery), 1)
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cp": 0,
            "protocol": "understand_mid_cp0_infra",
            "plan": str(PLAN),
            "data": meta,
            "battery": {
                "n": len(battery),
                "n_nonmajority_gold": n_amb,
                "majority_baseline_acc_on_battery": maj_acc,
                "path": str(BATTERY),
            },
            "cue_plan_cp3": "dual_channel_separate_embedding",
            "verdict": "CP0_READY",
            "next": "CP1 plain CE rich thin 1L vs 0L",
        }
        write_json(RES / "stage168_cp0_decision.json", out)
        write_mini(
            RES / "stage168_cp0_mini.md",
            "Stage168 CP0 — infra mini report",
            [
                f"Rich corpus: {meta.get('n_phrases')} phrases, ~{meta.get('n_tokens_used')} tokens streamed",
                f"Case tags: {meta.get('case_tag_counts')}",
                f"Top punct: {list((meta.get('punct_counts') or {}).items())[:8]}",
                f"Battery n={len(battery)}, non-majority gold={n_amb}, majority baseline≈{100*maj_acc:.1f}%",
                "CP3 cue = separate dual-channel embedding (not special token)",
                "Next: CP1 plain CE baseline (orthography alone)",
            ],
            "CP0_READY",
        )
        log(f"[CP0] READY battery={len(battery)} maj_base={100*maj_acc:.1f}%")
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


# ----- shared LM helpers -----

def freeze_stack(device):
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
    return stack, cfg


def split_phrases(phrases: list[list[str]], seed=SEED):
    rr = random.Random(seed)
    idx = list(range(len(phrases)))
    rr.shuffle(idx)
    n_hold = max(3000, int(0.1 * len(phrases)))
    hold_i = set(idx[:n_hold])
    train, hold = [], []
    for i, ph in enumerate(phrases):
        row = {"words": ph, "phrase": " ".join(ph), "bucket": "story"}
        (hold if i in hold_i else train).append(row)
    # light fat = short windows
    for ln in train:
        if len(ln["words"]) <= 6:
            ln["bucket"] = "fat_train"
    return train, hold


def build_stoi(train_rows, hold_rows):
    ctr = Counter()
    for ln in train_rows:
        ctr.update(ln["words"])
    tops = [w for w, _ in ctr.most_common(V_CAP - 1)]
    surfaces = [UNK] + tops
    stoi = {w: i for i, w in enumerate(surfaces)}
    cov = sum(ctr[w] for w in tops) / max(sum(ctr.values()), 1)
    return surfaces, stoi, {"V": len(surfaces), "train_coverage": cov, "n_types_raw": len(ctr)}


def map_ids(words, stoi):
    return [stoi.get(w, stoi[UNK]) for w in words]


@torch.no_grad()
def init_fps(surfaces, stack, device):
    """Letter-fp on alnum core; punct/case get dedicated offsets."""
    dim = 256
    case_basis = {
        "lower": torch.zeros(dim, device=device),
        "title": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.05,
        "upper": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.08,
        "mixed": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.06,
        "punct": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.1,
        "other": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.04,
    }
    punct_bank = {}
    fps = []
    for w in surfaces:
        if w == UNK:
            fps.append(torch.zeros(dim, device=device))
            continue
        tag = case_tag(w)
        if tag == "punct" or w in PUNCT_CHARS:
            if w not in punct_bank:
                punct_bank[w] = F.normalize(torch.randn(dim, device=device), dim=-1)
            fps.append(F.normalize(punct_bank[w] + case_basis["punct"], dim=-1))
            continue
        core = alnum_core(w)
        try:
            base = F.normalize(stack.w(core).detach(), dim=-1) if core else torch.zeros(dim, device=device)
        except Exception:
            base = torch.zeros(dim, device=device)
        fps.append(F.normalize(base + case_basis.get(tag, case_basis["other"]), dim=-1))
    return torch.stack(fps, 0)


def pairs_from_rows(rows, surfaces, stoi, max_n=None, seed=0, ambig_only=False, next_by_last=None):
    use = rows if max_n is None else _subsample(rows, max_n, seed)
    out = []
    for li, ln in enumerate(use):
        ids = map_ids(ln["words"], stoi)
        words = ln["words"]
        for t in range(1, len(ids)):
            last = words[t - 1]
            if ambig_only and next_by_last is not None:
                if len(next_by_last.get(last, {})) < 2:
                    continue
            out.append(
                {
                    "line_i": li,
                    "prefix_word_ids": ids[:t],
                    "target_word": surfaces[ids[t]] if ids[t] < len(surfaces) else UNK,
                    "target_word_id": ids[t],
                    "prefix_len": t,
                    "last_surface": last,
                    "phrase": ln.get("phrase", ""),
                }
            )
    return out


def eval_all(model, rows, surfaces, stoi, device, max_n=400):
    model.eval()
    rows = _subsample(rows, min(max_n, len(rows)), 1)
    n = h = 0
    with torch.no_grad():
        for ln in rows:
            ids = map_ids(ln["words"], stoi)
            for t in range(1, len(ids)):
                pref = ids[:t]
                gold = ids[t]
                # pack single
                ex = {
                    "prefix_word_ids": pref,
                    "target_word_id": gold,
                    "target_word": surfaces[gold] if gold < len(surfaces) else UNK,
                }
                packed = collate_word_id_batch(
                    [ex], stoi, model.max_len, model.pad_id, device
                )
                if packed is None:
                    continue
                x, mask, tgt = packed
                pred = model.logits_last_from_batch(x, mask).argmax(-1).item()
                n += 1
                h += int(pred == tgt.item())
    return h / max(n, 1)


def eval_battery(model, battery, stoi, surfaces, device, model_max_len):
    model.eval()
    n = h = maj_h = 0
    with torch.no_grad():
        for it in battery:
            ids = map_ids(it["prefix"], stoi)
            gold = stoi.get(it["gold"], stoi[UNK])
            maj = stoi.get(it["majority_next"], stoi[UNK])
            ex = {
                "prefix_word_ids": ids,
                "target_word_id": gold,
                "target_word": it["gold"],
            }
            packed = collate_word_id_batch([ex], stoi, model_max_len, model.pad_id, device)
            if packed is None:
                continue
            x, mask, tgt = packed
            pred = model.logits_last_from_batch(x, mask).argmax(-1).item()
            n += 1
            h += int(pred == gold)
            maj_h += int(maj == gold)
    acc = h / max(n, 1)
    maj_acc = maj_h / max(n, 1)
    return {
        "battery_acc": acc,
        "majority_acc": maj_acc,
        "lift_vs_majority": acc - maj_acc,
        "n": n,
    }


def order_probe(model, rows, stoi, surfaces, device, max_n=300):
    model.eval()
    rows = _subsample(rows, min(max_n, len(rows)), 2)
    rr = random.Random(42)
    clean_n = clean_h = shuf_n = shuf_h = 0
    with torch.no_grad():
        for ln in rows:
            words = ln["words"]
            if len(words) < 3:
                continue
            ids = map_ids(words, stoi)
            for t in range(2, len(ids)):
                pref = ids[:t]
                gold = ids[t]
                ex = {"prefix_word_ids": pref, "target_word_id": gold, "target_word": ""}
                packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
                if packed is None:
                    continue
                x, mask, tgt = packed
                pred = model.logits_last_from_batch(x, mask).argmax(-1).item()
                clean_n += 1
                clean_h += int(pred == gold)
                # shuffle prefix except last
                mid = pref[:-1][:]
                rr.shuffle(mid)
                shuf = mid + [pref[-1]]
                ex2 = {"prefix_word_ids": shuf, "target_word_id": gold, "target_word": ""}
                packed2 = collate_word_id_batch([ex2], stoi, model.max_len, model.pad_id, device)
                if packed2 is None:
                    continue
                x2, m2, _ = packed2
                pred2 = model.logits_last_from_batch(x2, m2).argmax(-1).item()
                shuf_n += 1
                shuf_h += int(pred2 == gold)
    c = clean_h / max(clean_n, 1)
    s = shuf_h / max(shuf_n, 1)
    return {"clean_acc": c, "shuffle_acc": s, "order_drop": c - s, "n": clean_n}


def train_plain_arm(tag, model, train_rows, hold_rows, surfaces, stoi, device, steps, log, ambig_only=False):
    next_by_last = defaultdict(Counter)
    for ln in train_rows:
        w = ln["words"]
        for t in range(1, len(w)):
            next_by_last[w[t - 1]][w[t]] += 1

    fat = [ln for ln in train_rows if ln.get("bucket") == "fat_train"] or train_rows
    story = [ln for ln in train_rows if ln.get("bucket") != "fat_train"] or train_rows
    fat_p = pairs_from_rows(fat, surfaces, stoi, seed=stable_seed(tag, "fat"), ambig_only=ambig_only, next_by_last=next_by_last)
    story_p = pairs_from_rows(
        story, surfaces, stoi, max_n=100000, seed=stable_seed(tag, "story"),
        ambig_only=ambig_only, next_by_last=next_by_last,
    )
    if not fat_p:
        fat_p = story_p
    if not story_p:
        raise RuntimeError(f"no pairs for {tag}")

    opt = make_opt(model.parameters(), TRUNK)
    rr = random.Random(stable_seed(tag, "train"))
    best = {"story_all": -1.0, "state": None}
    curve = []
    batch, fat_frac = TRUNK["batch"], TRUNK["fat_frac"]

    def snap(step):
        model.eval()
        sall = eval_all(model, hold_rows, surfaces, stoi, device, 400)
        curve.append({"step": step, "story_all": sall})
        log(f"  [{tag}] step {step}: HOLD ALL={100*sall:.1f}%")
        if sall >= best["story_all"]:
            best["story_all"] = sall
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()

    model.train()
    snap(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, TRUNK["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac))) if fat_frac > 0 else 0
        batch_ex = [rr.choice(fat_p) for _ in range(n_fat)] + [
            rr.choice(story_p) for _ in range(batch - n_fat)
        ]
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

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"curve": curve, "best_story_all": best["story_all"]}


def run_cp1() -> int:
    logp = RES / "_stage168_cp1_log.txt"
    if not logp.exists():
        logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    dec0 = RES / "stage168_cp0_decision.json"
    if not dec0.exists():
        log("[cp1] CP0 missing — running CP0 first")
        rc = run_cp0()
        if rc != 0:
            return rc

    log(f"CP1 start {datetime.now(timezone.utc).isoformat()}")
    t0 = time.time()
    try:
        phrases = []
        for ln in CORPUS.read_text(encoding="utf-8").splitlines():
            if ln.strip() and not ln.startswith("#"):
                phrases.append(ln.split())
        battery = [json.loads(l) for l in BATTERY.read_text(encoding="utf-8").splitlines() if l.strip()]
        # hold-out battery split
        rr = random.Random(SEED)
        rr.shuffle(battery)
        bat_hold = battery[: max(800, len(battery) // 5)]
        train_rows, hold_rows = split_phrases(phrases)
        surfaces, stoi, vmeta = build_stoi(train_rows, hold_rows)
        log(f"[vocab] {vmeta}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stack, _ = freeze_stack(device)
        fps = init_fps(surfaces, stack, device)

        # --- 1L ---
        log("\n##### CP1: word_1L_d128_h2 plain CE rich #####")
        m1 = WordIdTransformer(len(surfaces), TRUNK["d"], TRUNK["n_head"], TRUNK["n_layer"], TRUNK["word_max_len"], 0.1).to(device)
        m1.init_from_fps(fps)
        tr1 = train_plain_arm("word_1L_rich", m1, train_rows, hold_rows, surfaces, stoi, device, FT_STEPS_CP1, log)
        b1 = eval_battery(m1, bat_hold, stoi, surfaces, device, m1.max_len)
        o1 = order_probe(m1, hold_rows, stoi, surfaces, device)
        a1 = eval_all(m1, hold_rows, surfaces, stoi, device, 800)

        # --- 0L ---
        log("\n##### CP1: word_0L_last_d128 plain CE rich #####")
        m0 = ZeroLayerWordLM(len(surfaces), d_model=TRUNK["d"], max_len=TRUNK["word_max_len"], mode="last").to(device)
        m0.init_from_fps(fps)
        tr0 = train_plain_arm("word_0L_rich", m0, train_rows, hold_rows, surfaces, stoi, device, FT_STEPS_CP1, log)
        b0 = eval_battery(m0, bat_hold, stoi, surfaces, device, m0.max_len)
        o0 = order_probe(m0, hold_rows, stoi, surfaces, device)
        a0 = eval_all(m0, hold_rows, surfaces, stoi, device, 800)

        lift_b = b1["battery_acc"] - b0["battery_acc"]
        lift_maj = b1["lift_vs_majority"]
        ord_g = o1["order_drop"] - o0["order_drop"]

        if lift_b >= 0.05 or (lift_maj >= 0.05 and ord_g >= 0.05):
            verdict = "CP1_ORTHO_PROMISING"
        elif abs(lift_b) < 0.02 and b1["lift_vs_majority"] < 0.02:
            verdict = "CP1_ORTHO_ALONE_WEAK"
        else:
            verdict = "CP1_MIXED"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cp": 1,
            "protocol": "understand_mid_cp1_plain_ce_rich",
            "steps": FT_STEPS_CP1,
            "trunk": TRUNK,
            "vocab": vmeta,
            "arms": {
                "word_1L": {"all": a1, "battery": b1, "order": o1, "train": tr1},
                "word_0L": {"all": a0, "battery": b0, "order": o0, "train": tr0},
            },
            "gaps": {
                "1L_minus_0L_battery": lift_b,
                "1L_minus_0L_all": a1 - a0,
                "1L_lift_vs_majority": lift_maj,
                "1L_minus_0L_order_drop": ord_g,
            },
            "verdict": verdict,
            "wall_hours": (time.time() - t0) / 3600,
            "next": "CP2 CE_ambig + contrast (implementation change)",
            "ref100": REF100,
        }
        # save ckpts
        CKPT.mkdir(exist_ok=True)
        torch.save({"model": m1.state_dict(), "stoi": stoi, "surfaces": surfaces}, CKPT / "stage168_cp1_1L.pt")
        torch.save({"model": m0.state_dict(), "stoi": stoi, "surfaces": surfaces}, CKPT / "stage168_cp1_0L.pt")
        write_json(RES / "stage168_cp1_decision.json", out)
        write_mini(
            RES / "stage168_cp1_mini.md",
            "Stage168 CP1 — plain CE rich mini report",
            [
                f"1L ALL={100*a1:.1f}% | 0L ALL={100*a0:.1f}% | gap={100*(a1-a0):+.1f}pp",
                f"Battery 1L={100*b1['battery_acc']:.1f}% (lift vs maj {100*b1['lift_vs_majority']:+.1f}pp) | 0L={100*b0['battery_acc']:.1f}%",
                f"1L−0L battery={100*lift_b:+.1f}pp | order_drop gap={100*ord_g:+.1f}pp",
                f"Verdict {verdict} — orthography alone; next = ambig CE + contrast",
            ],
            verdict,
        )
        log(f"[CP1] {verdict}")
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


def train_ambig_contrast(
    tag,
    model,
    ambig_pairs,
    by_last_pairs,
    bat_hold,
    hold_rows,
    surfaces,
    stoi,
    device,
    steps,
    log,
    seed_tag,
):
    opt = make_opt(model.parameters(), TRUNK)
    rng = random.Random(stable_seed(seed_tag))
    best = {"battery_lift": -99.0, "state": None, "battery_acc": 0.0}
    curve = []
    batch = TRUNK["batch"]

    def snap(step):
        model.eval()
        b = eval_battery(model, bat_hold, stoi, surfaces, device, model.max_len)
        a = eval_all(model, hold_rows, surfaces, stoi, device, 400)
        curve.append({"step": step, **b, "all": a})
        log(
            f"  [{tag}] step {step}: ALL={100*a:.1f}% bat={100*b['battery_acc']:.1f}% "
            f"lift_maj={100*b['lift_vs_majority']:+.1f}pp"
        )
        if b["lift_vs_majority"] >= best["battery_lift"]:
            best["battery_lift"] = b["lift_vs_majority"]
            best["battery_acc"] = b["battery_acc"]
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()

    model.train()
    snap(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, TRUNK["lr"], WARMUP)
        batch_ex = [rng.choice(ambig_pairs) for _ in range(batch)]
        packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        logits = model.logits_last_from_batch(ids, mask)
        loss_ce = F.cross_entropy(logits, tgt)
        loss_c = torch.tensor(0.0, device=device)
        n_c = 0
        for ex in batch_ex:
            wrongs = [
                c
                for c in (by_last_pairs.get(ex["last_surface"]) or [])
                if c["target_word_id"] != ex["target_word_id"]
            ]
            if not wrongs:
                continue
            wex = rng.choice(wrongs)
            packed_g = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed_g is None:
                continue
            xg, mg, tg = packed_g
            lg = model.logits_last_from_batch(xg, mg)[0]
            loss_c = loss_c + F.relu(0.2 + lg[wex["target_word_id"]] - lg[tg[0]])
            n_c += 1
        if n_c:
            loss_c = loss_c / n_c
        loss = loss_ce + 0.5 * loss_c
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            snap(step)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    b = eval_battery(model, bat_hold, stoi, surfaces, device, model.max_len)
    o = order_probe(model, hold_rows, stoi, surfaces, device)
    a = eval_all(model, hold_rows, surfaces, stoi, device, 800)
    return {"curve": curve, "battery": b, "order": o, "all": a, "best_lift": best["battery_lift"]}


def run_cp2() -> int:
    """CE_ambig + contrast: thin 1L control + deep 4L h4 under understanding-gate."""
    logp = RES / "_stage168_cp2_log.txt"
    # fresh log for restarted deep-arm run
    logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    if not (RES / "stage168_cp1_decision.json").exists():
        log("[cp2] need CP1 first")
        rc = run_cp1()
        if rc != 0:
            return rc

    # clear stale decision from aborted thin-only run
    for p in (RES / "stage168_cp2_decision.json", RES / "stage168_cp2_mini.md"):
        if p.exists():
            p.unlink()

    log(f"CP2 start {datetime.now(timezone.utc).isoformat()} (1L + deep 4L under battery gate)")
    t0 = time.time()
    try:
        phrases = [
            ln.split()
            for ln in CORPUS.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        battery = [
            json.loads(l) for l in BATTERY.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        rr = random.Random(SEED + 2)
        rr.shuffle(battery)
        bat_hold = battery[: max(800, len(battery) // 5)]
        train_rows, hold_rows = split_phrases(phrases, seed=SEED + 2)
        surfaces, stoi, vmeta = build_stoi(train_rows, hold_rows)
        log(f"[vocab] {vmeta}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stack, _ = freeze_stack(device)
        fps = init_fps(surfaces, stack, device)

        next_by_last = defaultdict(Counter)
        for ln in train_rows:
            w = ln["words"]
            for t in range(1, len(w)):
                next_by_last[w[t - 1]][w[t]] += 1

        by_last_pairs = defaultdict(list)
        ambig_pairs = pairs_from_rows(
            train_rows,
            surfaces,
            stoi,
            max_n=120000,
            seed=3,
            ambig_only=True,
            next_by_last=next_by_last,
        )
        for p in ambig_pairs:
            by_last_pairs[p["last_surface"]].append(p)
        log(f"[data] ambig_pairs={len(ambig_pairs)}")

        arms = {}
        cp1 = json.loads((RES / "stage168_cp1_decision.json").read_text(encoding="utf-8"))
        cp1_lift = cp1["arms"]["word_1L"]["battery"]["lift_vs_majority"]
        cp1_0l_bat = cp1["arms"]["word_0L"]["battery"]["battery_acc"]

        # --- thin 1L ---
        log("\n##### CP2: 1L_ambig (thin control) #####")
        m1 = WordIdTransformer(
            len(surfaces), TRUNK["d"], TRUNK["n_head"], 1, TRUNK["word_max_len"], 0.1
        ).to(device)
        m1.init_from_fps(fps)
        ck1 = CKPT / "stage168_cp1_1L.pt"
        if ck1.exists():
            blob = torch.load(ck1, map_location="cpu", weights_only=False)
            if len(blob.get("surfaces", [])) == len(surfaces):
                try:
                    m1.load_state_dict(blob["model"], strict=False)
                    log("[cp2] warm-start 1L from CP1")
                except Exception as e:
                    log(f"[cp2] 1L warm-start skip: {e}")
        arms["1L_ambig"] = train_ambig_contrast(
            "1L_ambig",
            m1,
            ambig_pairs,
            by_last_pairs,
            bat_hold,
            hold_rows,
            surfaces,
            stoi,
            device,
            FT_STEPS_CP2_THIN,
            log,
            "cp2_1l",
        )
        arms["1L_ambig"]["arch"] = {"n_layer": 1, "n_head": TRUNK["n_head"], "d": TRUNK["d"]}
        torch.save(
            {"model": m1.state_dict(), "stoi": stoi, "surfaces": surfaces, "arch": arms["1L_ambig"]["arch"]},
            CKPT / "stage168_cp2_1L.pt",
        )

        # --- deep 4L h4 ---
        log("\n##### CP2: 4L_h4_ambig (deep arm — understanding gate) #####")
        m4 = WordIdTransformer(
            len(surfaces), TRUNK["d"], DEEP_HEADS, DEEP_LAYERS, TRUNK["word_max_len"], 0.1
        ).to(device)
        m4.init_from_fps(fps)
        arms["4L_h4_ambig"] = train_ambig_contrast(
            "4L_h4_ambig",
            m4,
            ambig_pairs,
            by_last_pairs,
            bat_hold,
            hold_rows,
            surfaces,
            stoi,
            device,
            FT_STEPS_CP2_DEEP,
            log,
            "cp2_4l",
        )
        arms["4L_h4_ambig"]["arch"] = {
            "n_layer": DEEP_LAYERS,
            "n_head": DEEP_HEADS,
            "d": TRUNK["d"],
        }
        torch.save(
            {
                "model": m4.state_dict(),
                "stoi": stoi,
                "surfaces": surfaces,
                "arch": arms["4L_h4_ambig"]["arch"],
            },
            CKPT / "stage168_cp2_4L.pt",
        )

        # pick best by battery lift vs majority, then vs 0L battery
        def score(name):
            b = arms[name]["battery"]
            return (b["lift_vs_majority"], b["battery_acc"] - cp1_0l_bat, b["battery_acc"])

        best_name = max(arms.keys(), key=score)
        best = arms[best_name]
        deltas = {
            name: arms[name]["battery"]["lift_vs_majority"] - cp1_lift for name in arms
        }
        deep_vs_thin = (
            arms["4L_h4_ambig"]["battery"]["lift_vs_majority"]
            - arms["1L_ambig"]["battery"]["lift_vs_majority"]
        )

        any_help = any(deltas[n] >= 0.05 and arms[n]["battery"]["lift_vs_majority"] >= 0.05 for n in arms)
        any_weak = any(deltas[n] >= 0.02 for n in arms)
        if any_help:
            verdict = "CP2_LOSS_HELPS"
        elif deep_vs_thin >= 0.03 and arms["4L_h4_ambig"]["battery"]["lift_vs_majority"] > arms["1L_ambig"]["battery"]["lift_vs_majority"]:
            verdict = "CP2_DEEP_BETTER_WEAK"
        elif any_weak:
            verdict = "CP2_LOSS_WEAK"
        else:
            verdict = "CP2_LOSS_NULL"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cp": 2,
            "protocol": "understand_mid_cp2_ambig_ce_contrast_thin_vs_deep",
            "steps_thin": FT_STEPS_CP2_THIN,
            "steps_deep": FT_STEPS_CP2_DEEP,
            "arms": {
                k: {
                    "arch": v["arch"],
                    "all": v["all"],
                    "battery": v["battery"],
                    "order": v["order"],
                    "vs_cp1_lift_delta": deltas[k],
                    "curve": v["curve"],
                }
                for k, v in arms.items()
            },
            "best_arm": best_name,
            "best_arch": best["arch"],
            "deep_minus_thin_lift_maj": deep_vs_thin,
            "cp1_0L_battery": cp1_0l_bat,
            "verdict": verdict,
            "wall_hours": (time.time() - t0) / 3600,
            "next": "CP3 dual-channel CueChannel on best_arm trunk",
            "note": "Depth re-opened for understanding-gate only; STORY-thin lock does not apply here",
        }
        # legacy alias for CP3 warm-start
        write_json(RES / "stage168_cp2_decision.json", out)
        write_mini(
            RES / "stage168_cp2_mini.md",
            "Stage168 CP2 — ambig+contrast thin vs deep mini report",
            [
                f"1L bat={100*arms['1L_ambig']['battery']['battery_acc']:.1f}% lift_maj={100*arms['1L_ambig']['battery']['lift_vs_majority']:+.1f}pp (ΔCP1 {100*deltas['1L_ambig']:+.1f})",
                f"4L bat={100*arms['4L_h4_ambig']['battery']['battery_acc']:.1f}% lift_maj={100*arms['4L_h4_ambig']['battery']['lift_vs_majority']:+.1f}pp (ΔCP1 {100*deltas['4L_h4_ambig']:+.1f})",
                f"deep−thin lift={100*deep_vs_thin:+.1f}pp | best={best_name} {best['arch']}",
                f"vs CP1 0L battery ({100*cp1_0l_bat:.1f}%): 1L {100*(arms['1L_ambig']['battery']['battery_acc']-cp1_0l_bat):+.1f}pp | 4L {100*(arms['4L_h4_ambig']['battery']['battery_acc']-cp1_0l_bat):+.1f}pp",
                f"Verdict {verdict} — next CP3 cue on best trunk",
            ],
            verdict,
        )
        log(f"[CP2] {verdict} best={best_name}")
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


class DualChannelWordLM(nn.Module):
    """Language trunk (WordIdTransformer) + separate CueChannel over prefix structure."""

    def __init__(self, n_vocab, d, n_head, n_layer, max_len, dropout=0.1):
        super().__init__()
        self.trunk = WordIdTransformer(n_vocab, d, n_head, n_layer, max_len, dropout)
        self.n_vocab = n_vocab
        self.max_len = max_len
        self.pad_id = self.trunk.pad_id
        self.cue_emb = nn.Embedding(n_vocab + 1, d, padding_idx=self.pad_id)
        self.cue_pos = nn.Embedding(max_len, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=max(1, n_head),
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.cue_enc = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.fuse = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))

    def init_from_fps(self, fps):
        self.trunk.init_from_fps(fps)
        with torch.no_grad():
            self.cue_emb.weight.copy_(self.trunk.tok.weight.detach())

    def logits_last_from_batch(self, ids, mask):
        h_lang = self._trunk_hidden(ids, mask)
        h_cue = self._cue_hidden(ids, mask)
        fused = self.fuse(torch.cat([h_lang, h_cue], dim=-1))
        return F.linear(fused, self.trunk.tok.weight[: self.n_vocab])

    def _trunk_hidden(self, ids, mask):
        bsz, tmax = ids.shape
        pos = torch.arange(tmax, device=ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.trunk.tok(ids) + self.trunk.pos(pos)
        causal = torch.triu(torch.ones(tmax, tmax, device=ids.device, dtype=torch.bool), diagonal=1)
        h = self.trunk.tr(x, mask=causal, src_key_padding_mask=mask)
        return h[:, -1, :]

    def _cue_hidden(self, ids, mask):
        """Cue channel: blank last token so channel must use earlier context + orthography."""
        ids_cue = ids.clone()
        # left-padded: last column is prefix end
        ids_cue[:, -1] = self.pad_id
        pos_ids = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.cue_emb(ids_cue) + self.cue_pos(pos_ids)
        pad = ids_cue == self.pad_id
        x = self.cue_enc(x, src_key_padding_mask=pad)
        w = (~pad).float().unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


def run_cp3() -> int:
    logp = RES / "_stage168_cp3_log.txt"
    if not logp.exists():
        logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    if not (RES / "stage168_cp2_decision.json").exists():
        log("[cp3] need CP2 first")
        rc = run_cp2()
        if rc != 0:
            return rc

    log(f"CP3 start {datetime.now(timezone.utc).isoformat()}")
    t0 = time.time()
    try:
        phrases = [ln.split() for ln in CORPUS.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        battery = [json.loads(l) for l in BATTERY.read_text(encoding="utf-8").splitlines() if l.strip()]
        rr = random.Random(SEED + 3)
        rr.shuffle(battery)
        bat_hold = battery[: max(800, len(battery) // 5)]
        train_rows, hold_rows = split_phrases(phrases, seed=SEED + 3)
        surfaces, stoi, vmeta = build_stoi(train_rows, hold_rows)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stack, _ = freeze_stack(device)
        fps = init_fps(surfaces, stack, device)

        next_by_last = defaultdict(Counter)
        for ln in train_rows:
            for t in range(1, len(ln["words"])):
                next_by_last[ln["words"][t - 1]][ln["words"][t]] += 1
        ambig_pairs = pairs_from_rows(
            train_rows, surfaces, stoi, max_n=120000, seed=5, ambig_only=True, next_by_last=next_by_last
        )
        by_last = defaultdict(list)
        for p in ambig_pairs:
            by_last[p["last_surface"]].append(p)

        cp2 = json.loads((RES / "stage168_cp2_decision.json").read_text(encoding="utf-8"))
        best_arch = cp2.get("best_arch") or {"n_layer": 1, "n_head": TRUNK["n_head"], "d": TRUNK["d"]}
        best_name = cp2.get("best_arm") or "1L_ambig"
        n_layer = int(best_arch.get("n_layer", 1))
        n_head = int(best_arch.get("n_head", TRUNK["n_head"]))
        log(f"[cp3] using best CP2 trunk {best_name} arch={best_arch}")

        log("\n##### CP3: DualChannelWordLM (trunk + CueChannel) #####")
        model = DualChannelWordLM(
            len(surfaces), TRUNK["d"], n_head, n_layer, TRUNK["word_max_len"]
        ).to(device)
        model.init_from_fps(fps)
        ck2 = CKPT / ("stage168_cp2_4L.pt" if n_layer >= 4 else "stage168_cp2_1L.pt")
        if not ck2.exists():
            ck2 = CKPT / "stage168_cp2_1L.pt"
        if ck2.exists():
            blob = torch.load(ck2, map_location="cpu", weights_only=False)
            try:
                model.trunk.load_state_dict(blob["model"], strict=False)
                log(f"[cp3] warm-start trunk from {ck2.name}")
            except Exception as e:
                log(f"[cp3] trunk warm-start skip: {e}")

        opt = make_opt(model.parameters(), TRUNK)
        rng = random.Random(stable_seed("cp3"))
        best = {"lift": -99.0, "state": None}
        curve = []

        def snap(step):
            model.eval()
            b = eval_battery(model, bat_hold, stoi, surfaces, device, model.max_len)
            a = eval_all(model, hold_rows, surfaces, stoi, device, 400)
            curve.append({"step": step, **b, "all": a})
            log(
                f"  [cp3] step {step}: ALL={100*a:.1f}% bat={100*b['battery_acc']:.1f}% "
                f"lift_maj={100*b['lift_vs_majority']:+.1f}pp"
            )
            if b["lift_vs_majority"] >= best["lift"]:
                best["lift"] = b["lift_vs_majority"]
                best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        model.train()
        snap(0)
        batch = TRUNK["batch"]
        for step in range(1, FT_STEPS_CP3 + 1):
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, TRUNK["lr"], WARMUP)
            batch_ex = [rng.choice(ambig_pairs) for _ in range(batch)]
            packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            logits = model.logits_last_from_batch(ids, mask)
            loss_ce = F.cross_entropy(logits, tgt)
            loss_c = torch.tensor(0.0, device=device)
            n_c = 0
            for ex in batch_ex:
                wrongs = [c for c in by_last.get(ex["last_surface"], []) if c["target_word_id"] != ex["target_word_id"]]
                if not wrongs:
                    continue
                wex = rng.choice(wrongs)
                packed_g = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
                if packed_g is None:
                    continue
                xg, mg, tg = packed_g
                lg = model.logits_last_from_batch(xg, mg)[0]
                loss_c = loss_c + F.relu(0.2 + lg[wex["target_word_id"]] - lg[tg[0]])
                n_c += 1
            if n_c:
                loss_c = loss_c / n_c
            loss = loss_ce + 0.5 * loss_c
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0 or step == FT_STEPS_CP3:
                snap(step)

        if best["state"] is not None:
            model.load_state_dict(best["state"])
        b = eval_battery(model, bat_hold, stoi, surfaces, device, model.max_len)
        o = order_probe(model, hold_rows, stoi, surfaces, device)
        a = eval_all(model, hold_rows, surfaces, stoi, device, 800)
        cp2 = json.loads((RES / "stage168_cp2_decision.json").read_text(encoding="utf-8"))
        if "arms" in cp2 and cp2.get("best_arm"):
            cp2_bat = cp2["arms"][cp2["best_arm"]]["battery"]
        else:
            cp2_bat = cp2.get("battery") or {"lift_vs_majority": 0.0}
        delta = b["lift_vs_majority"] - cp2_bat.get("lift_vs_majority", 0.0)

        if b["lift_vs_majority"] >= 0.08 and b["battery_acc"] - b["majority_acc"] >= 0.05:
            verdict = "CP3_CUE_HELPS"
        elif delta >= 0.03:
            verdict = "CP3_CUE_WEAK"
        else:
            verdict = "CP3_CUE_NULL"

        torch.save(
            {
                "model": model.state_dict(),
                "stoi": stoi,
                "surfaces": surfaces,
                "arch": best_arch,
                "best_arm": best_name,
            },
            CKPT / "stage168_cp3_dual.pt",
        )
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cp": 3,
            "protocol": "understand_mid_cp3_dual_channel_cue",
            "cue": "separate_CueChannel_blank_last_encode_fuse",
            "trunk_from": best_name,
            "trunk_arch": best_arch,
            "all": a,
            "battery": b,
            "order": o,
            "vs_cp2_lift_delta": delta,
            "curve": curve,
            "verdict": verdict,
            "wall_hours": (time.time() - t0) / 3600,
            "next": "CP4 final rollup",
        }
        write_json(RES / "stage168_cp3_decision.json", out)
        write_mini(
            RES / "stage168_cp3_mini.md",
            "Stage168 CP3 — dual-channel cue mini report",
            [
                f"ALL={100*a:.1f}% bat={100*b['battery_acc']:.1f}% lift_maj={100*b['lift_vs_majority']:+.1f}pp",
                f"Δ vs CP2 lift = {100*delta:+.1f}pp | order_drop={100*o['order_drop']:+.1f}pp",
                f"Cue = separate emb channel (last blanked) fused with trunk",
                f"Verdict {verdict}",
            ],
            verdict,
        )
        log(f"[CP3] {verdict}")
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


def run_cp4() -> int:
    logp = RES / "_stage168_cp4_log.txt"
    log = lambda m: log_to(logp, m)
    parts = {}
    for i in range(4):
        p = RES / f"stage168_cp{i}_decision.json"
        if p.exists():
            parts[f"cp{i}"] = json.loads(p.read_text(encoding="utf-8"))
    if "cp3" not in parts:
        return run_cp3() or run_cp4()

    c1 = parts.get("cp1", {})
    c2 = parts.get("cp2", {})
    c3 = parts.get("cp3", {})
    lift = (c3.get("battery") or {}).get("lift_vs_majority", 0)
    v3 = c3.get("verdict", "")
    v2 = c2.get("verdict", "")

    if lift >= 0.08 and "HELPS" in v3:
        final = "UNDERSTANDS"
    elif "HELPS" in v2 or "HELPS" in v3 or "WEAK" in v3:
        final = "PARTIAL_" + ("CUE" if "CUE" in v3 else "LOSS")
    else:
        final = "STILL_LAST_TOKEN"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cp": 4,
        "protocol": "understand_mid_cp4_rollup",
        "parts": {k: {"verdict": parts[k].get("verdict")} for k in parts},
        "final_verdict": final,
        "note": "Primary=battery/same-last style lifts; ALL secondary. Dual-channel cue at CP3.",
    }
    write_json(RES / "stage168_cp4_decision.json", out)
    write_mini(
        RES / "stage168_cp4_mini.md",
        "Stage168 CP4 — final understanding rollup",
        [
            f"CP1 {c1.get('verdict')} | CP2 {c2.get('verdict')} | CP3 {c3.get('verdict')}",
            f"Final battery lift_maj (CP3)={100*lift:+.1f}pp",
            f"**{final}**",
            "If STILL_LAST_TOKEN → next research = harder instance/handwriting channel, not more CE steps",
        ],
        final,
    )
    log(f"[CP4] {final}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", default="0", help="0|1|2|3|4|all")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    seq = ["0", "1", "2", "3", "4"] if args.cp == "all" else [args.cp]
    runners = {"0": run_cp0, "1": run_cp1, "2": run_cp2, "3": run_cp3, "4": run_cp4}
    for c in seq:
        rc = runners[c]()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
