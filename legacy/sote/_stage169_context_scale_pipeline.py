"""
Stage 169 вЂ” SOTE context dig @ 0.5BвЂ“1B (real scale).

Plan: results/plan_sote_context_0p5b_1b.md

  python _stage169_context_scale_pipeline.py --step 0   # pretok
  python _stage169_context_scale_pipeline.py --step 1   # VRAM smoke
  python _stage169_context_scale_pipeline.py --step 2   # full primary
  python _stage169_context_scale_pipeline.py --step 3   # matched 0L
  python _stage169_context_scale_pipeline.py --step 4   # probes + verdict
  python _stage169_context_scale_pipeline.py --step all # S0в†’S4 chain
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    _warmup_then_constant,
    load_foundation_85,
)
from _stage164_zerolayer_control import ZeroLayerWordLM  # noqa: E402
import _stage166_wiki50m_0l_1l_bpe_probes as s166  # noqa: E402

PLAN = RES / "plan_sote_context_0p5b_1b.md"
RAW_TRAIN = ROOT / "data" / "_wikitext103_train.txt"
DATA = ROOT / "data"
IDS_BIN = DATA / "sote_ids_0p5b.bin"
CODEBOOK = DATA / "sote_codebook_169.json"
BATTERY = RES / "stage169_understanding_battery.jsonl"
RITUAL_PATH = RES / "stage169_ritual.json"

TARGET_TOKENS = 500_000_000
V_MIN, V_MAX = 32_000, 64_000
COVERAGE_MIN = 0.95
UNK = "<unk>"
SEED = 169
MAX_WORD_LEN = 24
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[.,!?;:\"'()\[\]{}\-%/$]")
PUNCT_CHARS = set(".,!?;:\"'()[]{}-%/$")

# Train ritual (plan lock)
DEFAULT_RITUAL = {
    "d": 256,
    "n_layer": 4,
    "n_head": 4,
    "seq_len": 512,  # may drop to 256 after S1
    "micro_batch": 2,
    "effective_batch": 64,
    "lr": 1e-3,
    "lr_min": 1e-3,
    "lr_schedule": "warmup_constant",
    "opt": "Adam",
    "wd": 0.0,
    "warmup": 3000,
    "ckpt_every": 10_000,
    "smoke_steps": 20_000,
    "full_steps": 300_000,
    "eval_every": 10_000,
}


def log_to(path: Path, msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, torch.Tensor):
            return None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def sanitize_train_result(tr: dict) -> dict:
    """Drop non-JSON bits (model state tensors) for decision files."""
    best = tr.get("best") or {}
    return {
        "curve": tr.get("curve") or [],
        "best": {"lift": float(best.get("lift", -99)), "bat": float(best.get("bat", 0))},
        "wall_hours": float(tr.get("wall_hours") or 0),
    }


def write_mini(path: Path, title: str, bullets: list[str], verdict: str):
    body = [f"# {title}", "", f"**Verdict:** `{verdict}`", ""] + [f"- {b}" for b in bullets] + [""]
    path.write_text("\n".join(body), encoding="utf-8")


def tokenize_rich(line: str) -> list[str]:
    out = []
    for t in TOKEN_RE.findall(line):
        if t in PUNCT_CHARS:
            out.append(t)
        elif len(re.sub(r"[^a-z0-9]", "", t.lower())) <= MAX_WORD_LEN:
            out.append(t)
    return out


def load_ritual() -> dict:
    if RITUAL_PATH.exists():
        r = json.loads(RITUAL_PATH.read_text(encoding="utf-8"))
        out = dict(DEFAULT_RITUAL)
        out.update(r)
        return out
    return dict(DEFAULT_RITUAL)


def save_ritual(r: dict):
    write_json(RITUAL_PATH, r)


# ---------- data sources ----------

def iter_wikitext_tokens(log):
    path = s166.ensure_wikitext103_train()
    log(f"[src] WikiText-103 {path}")
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(" = ") or not line.strip():
                continue
            toks = tokenize_rich(line)
            if toks:
                n += len(toks)
                yield "wikitext103", toks
    log(f"[src] WikiText-103 streamed ~{n} tokens")


OWT_CACHE = DATA / "_owt_tokens_cache.txt"
OWT_META = DATA / "_owt_tokens_cache.txt.meta"
OWT_WORKER = ROOT / "_stage169_owt_worker.py"


def _owt_cache_tokens() -> int:
    if not OWT_META.exists():
        return 0
    try:
        return int(OWT_META.read_text(encoding="utf-8").strip().split()[0])
    except Exception:
        return 0


def ensure_owt_cache(log, max_tokens: int) -> Path:
    """Build local OWT token cache via subprocess (avoids in-process datasets crash)."""
    have = _owt_cache_tokens()
    if have >= max_tokens and OWT_CACHE.exists():
        log(f"[src] OWT cache ready have={have} >= need={max_tokens}")
        return OWT_CACHE
    log(f"[src] spawning OWT worker need={max_tokens} have={have} ...")
    import subprocess

    cmd = [
        sys.executable,
        "-u",
        str(OWT_WORKER),
        "--out",
        str(OWT_CACHE),
        "--max-tokens",
        str(max_tokens),
    ]
    # Stream worker logs into our log
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip("\n"))
    rc = proc.wait()
    have = _owt_cache_tokens()
    if rc != 0 and have < max(1, max_tokens // 10):
        raise RuntimeError(f"OWT worker failed rc={rc} have={have}")
    log(f"[src] OWT cache have={have} rc={rc}")
    return OWT_CACHE


def iter_openwebtext_tokens(log, max_tokens: int):
    """Yield tokens from local OWT cache (built by worker if needed)."""
    if max_tokens <= 0:
        return
    cache = ensure_owt_cache(log, max_tokens)
    n = 0
    with cache.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if n >= max_tokens:
                break
            toks = line.split()
            if not toks:
                continue
            left = max_tokens - n
            if len(toks) > left:
                toks = toks[:left]
            n += len(toks)
            yield "openwebtext", toks
    log(f"[src] OpenWebText cache read n={n}")


def iter_all_token_docs(log, target: int):
    """Yield (source, toks) until ~target tokens counted."""
    total = 0
    by_src = Counter()
    for src, toks in iter_wikitext_tokens(log):
        yield src, toks
        total += len(toks)
        by_src[src] += len(toks)
        if total >= target:
            log(f"[src] reached target from WT alone total={total}")
            return by_src, total
    need = target - total
    log(f"[src] WT contributed {total}; need +{need} from OpenWebText")
    for src, toks in iter_openwebtext_tokens(log, need):
        yield src, toks
        total += len(toks)
        by_src[src] += len(toks)
    return by_src, total


# ---------- S0 ----------

def run_s0() -> int:
    logp = RES / "_stage169_s0_log.txt"
    logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    log(f"S0 start {datetime.now(timezone.utc).isoformat()}")
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    try:
        # Pass 1: count frequencies (streaming; prune hapaxes to bound RAM)
        log("##### PASS 1: count token frequencies #####")
        ctr: Counter = Counter()
        total = 0
        by_src = Counter()
        for src, toks in _stream_until(log, TARGET_TOKENS):
            ctr.update(toks)
            total += len(toks)
            by_src[src] += len(toks)
            # Bound vocabulary RAM. Hapax-only prune stalls once almost all
            # remaining types have count>=2 (~800k floor) — escalate to top-N.
            if len(ctr) > 800_000:
                before = len(ctr)
                pruned = Counter({w: c for w, c in ctr.items() if c >= 2})
                if len(pruned) > 750_000:
                    min_c = 3
                    while True:
                        nxt = Counter({w: c for w, c in pruned.items() if c >= min_c})
                        if len(nxt) <= 750_000 or min_c >= 20:
                            pruned = nxt if len(nxt) <= 750_000 else Counter(dict(pruned.most_common(700_000)))
                            break
                        pruned = nxt
                        min_c += 1
                ctr = pruned
                if len(ctr) < before:
                    log(f"  [pass1] prune types {before}->{len(ctr)} (min_c escalate)")
            if total % 5_000_000 < len(toks):
                log(f"  [pass1] tokens={total} types={len(ctr)} src={dict(by_src)}")

        log(f"[pass1] done tokens={total} types={len(ctr)} by_src={dict(by_src)}")
        if total < TARGET_TOKENS * 0.5:
            log(f"[WARN] only reached {total} << {TARGET_TOKENS}")

        # Choose V with coverage >= 95%
        V = V_MIN
        surfaces = None
        coverage = 0.0
        while V <= V_MAX:
            tops = [w for w, _ in ctr.most_common(V - 1)]
            surfaces = [UNK] + tops
            covered = sum(ctr[w] for w in tops)
            coverage = covered / max(total, 1)
            unk_rate = 1.0 - coverage
            log(f"[vocab] try V={V} coverage={100*coverage:.2f}% unk_rate={100*unk_rate:.2f}%")
            if coverage >= COVERAGE_MIN:
                break
            V = min(V_MAX, V + 8000)
            if V == V_MAX and coverage < COVERAGE_MIN:
                # last try at V_MAX
                tops = [w for w, _ in ctr.most_common(V_MAX - 1)]
                surfaces = [UNK] + tops
                coverage = sum(ctr[w] for w in tops) / max(total, 1)
                log(f"[vocab] V_MAX={V_MAX} coverage={100*coverage:.2f}% (may be <95%)")
                break

        assert surfaces is not None
        stoi = {w: i for i, w in enumerate(surfaces)}
        V = len(surfaces)
        dtype = np.uint16 if V <= 65535 else np.uint32
        log(f"[vocab] final V={V} dtype={dtype} coverage={100*coverage:.2f}%")

        codebook = {
            "V": V,
            "unk": UNK,
            "surfaces": surfaces,
            "coverage": coverage,
            "unk_rate": 1.0 - coverage,
            "n_types_raw": len(ctr),
            "n_tokens_counted": total,
            "by_src_pass1": dict(by_src),
            "token_re": TOKEN_RE.pattern,
            "dtype": "uint16" if dtype == np.uint16 else "uint32",
        }
        write_json(CODEBOOK, codebook)

        # Pass 2: write ids memmap + battery stats
        log("##### PASS 2: write memmap ids + battery stats #####")
        n_write = min(total, TARGET_TOKENS)
        # Use exact pass1 total if we couldn't hit target
        n_write = total if total < TARGET_TOKENS else TARGET_TOKENS
        ids = np.memmap(IDS_BIN, mode="w+", dtype=dtype, shape=(n_write,))
        next_by_last: dict[str, Counter] = defaultdict(Counter)
        examples_by_last: dict[str, list] = defaultdict(list)
        pos = 0
        by_src2 = Counter()
        unk_n = 0
        prev = None
        prefix_buf: list[str] = []

        for src, toks in _stream_until(log, n_write):
            for t in toks:
                if pos >= n_write:
                    break
                tid = stoi.get(t, stoi[UNK])
                if tid == stoi[UNK]:
                    unk_n += 1
                ids[pos] = tid
                pos += 1
                by_src2[src] += 1
                # battery bookkeeping on surfaces (mapped)
                surf = surfaces[tid]
                if prev is not None:
                    next_by_last[prev][surf] += 1
                    if len(examples_by_last[prev]) < 30:
                        # prefix ends with prev (last); gold = next surf
                        pref = (prefix_buf + [prev])[-32:]
                        examples_by_last[prev].append({"prefix": pref, "gold": surf})
                prefix_buf.append(surf)
                if len(prefix_buf) > 64:
                    prefix_buf = prefix_buf[-64:]
                prev = surf
            if pos >= n_write:
                break
            if pos % 5_000_000 < 10000:
                log(f"  [pass2] wrote={pos}/{n_write}")

        ids.flush()
        del ids
        log(f"[pass2] wrote {pos} ids to {IDS_BIN.name} unk_rate_write={unk_n/max(pos,1):.4f}")

        # Build battery
        log("##### BATTERY from large stream #####")
        items = []
        for last, ctr_n in next_by_last.items():
            commons = [(w, c) for w, c in ctr_n.most_common(8) if c >= 5]
            if len(commons) < 2:
                continue
            maj_w, maj_c = commons[0]
            alts = [w for w, _ in commons[1:]]
            if not alts:
                continue
            exs = examples_by_last.get(last) or []
            nonmaj = [e for e in exs if e["gold"] != maj_w][:3]
            majex = [e for e in exs if e["gold"] == maj_w][:2]
            for e in nonmaj + majex:
                if len(e["prefix"]) < 2:
                    continue
                items.append(
                    {
                        "last": last,
                        "prefix": e["prefix"],
                        "gold": e["gold"],
                        "majority_next": maj_w,
                        "majority_frac": maj_c / max(sum(ctr_n.values()), 1),
                        "n_next_types": len(ctr_n),
                    }
                )
            if len(items) >= 12000:
                break
        rr = random.Random(SEED)
        rr.shuffle(items)
        items = items[:10000]
        with BATTERY.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        n_amb = sum(1 for it in items if it["gold"] != it["majority_next"])
        maj_base = sum(1 for it in items if it["gold"] == it["majority_next"]) / max(len(items), 1)
        log(f"[battery] n={len(items)} nonmaj={n_amb} maj_base={100*maj_base:.1f}%")

        save_ritual(DEFAULT_RITUAL)
        reached = pos
        verdict = "S0_READY" if reached >= int(0.9 * TARGET_TOKENS) else "S0_PARTIAL"
        if coverage < COVERAGE_MIN:
            verdict = "S0_LOW_COVERAGE"
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 0,
            "protocol": "context_scale_s0_pretok",
            "plan": str(PLAN),
            "target_tokens": TARGET_TOKENS,
            "n_tokens_written": reached,
            "by_src": dict(by_src2),
            "vocab": {
                "V": V,
                "coverage": coverage,
                "unk_rate": 1.0 - coverage,
                "dtype": codebook["dtype"],
            },
            "paths": {
                "ids": str(IDS_BIN),
                "codebook": str(CODEBOOK),
                "battery": str(BATTERY),
            },
            "battery": {
                "n": len(items),
                "n_nonmajority_gold": n_amb,
                "majority_baseline": maj_base,
            },
            "verdict": verdict,
            "wall_hours": (time.time() - t0) / 3600,
            "next": "S1 VRAM smoke 4L/d256 seq512",
        }
        write_json(RES / "stage169_s0_decision.json", out)
        write_mini(
            RES / "stage169_s0_mini.md",
            "Stage169 S0 вЂ” pretok mini report",
            [
                f"Tokens written: {reached}/{TARGET_TOKENS} ({100*reached/TARGET_TOKENS:.1f}%)",
                f"Sources: {dict(by_src2)}",
                f"V={V} coverage={100*coverage:.2f}% unk_rate={100*(1-coverage):.2f}%",
                f"Battery n={len(items)} maj_base={100*maj_base:.1f}%",
                f"Ids: {IDS_BIN.name} dtype={codebook['dtype']}",
                f"Next: S1 smoke (seq 512 в†’ fallback 256 on OOM)",
            ],
            verdict,
        )
        log(f"[S0] {verdict} wall={(time.time()-t0)/3600:.2f}h")
        return 0 if verdict in ("S0_READY", "S0_PARTIAL", "S0_LOW_COVERAGE") else 1
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


def _stream_until(log, target: int):
    """Yield (src, toks) until cumulative tokens would exceed target (caller truncates)."""
    total = 0
    # WT first
    path = s166.ensure_wikitext103_train()
    log(f"[stream] WikiText-103 ...")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if total >= target:
                return
            if line.startswith(" = ") or not line.strip():
                continue
            toks = tokenize_rich(line)
            if not toks:
                continue
            if total + len(toks) > target:
                toks = toks[: target - total]
            if not toks:
                return
            total += len(toks)
            yield "wikitext103", toks
    if total >= target:
        return
    need = target - total
    log(f"[stream] need +{need} from OpenWebText (have {total})")
    yield from iter_openwebtext_tokens(log, need)


# ---------- shared train / eval ----------

def load_codebook():
    cb = json.loads(CODEBOOK.read_text(encoding="utf-8"))
    surfaces = cb["surfaces"]
    stoi = {w: i for i, w in enumerate(surfaces)}
    return cb, surfaces, stoi


def open_ids():
    cb, _, _ = load_codebook()
    dtype = np.uint16 if cb.get("dtype", "uint16") == "uint16" else np.uint32
    ids = np.memmap(IDS_BIN, mode="r", dtype=dtype)
    return ids, cb


def alnum_core(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def init_fps(surfaces, stack, device):
    dim = 256
    case_basis = {
        "lower": torch.zeros(dim, device=device),
        "title": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.05,
        "upper": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.08,
        "mixed": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.06,
        "punct": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.1,
        "other": F.normalize(torch.randn(dim, device=device), dim=-1) * 0.04,
    }

    def case_tag(tok: str) -> str:
        if tok in PUNCT_CHARS:
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

    punct_bank = {}
    fps = []
    for w in surfaces:
        if w == UNK:
            fps.append(torch.zeros(dim, device=device))
            continue
        tag = case_tag(w)
        if tag == "punct":
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
    return stack


def sample_batch(ids: np.memmap, seq_len: int, batch: int, rng: random.Random, device, pad_id: int):
    """Causal LM batch: input ids[:, :-1], target ids[:, 1:] from contiguous windows."""
    n = len(ids)
    max_start = n - seq_len - 1
    if max_start < 1:
        raise RuntimeError("ids too short")
    windows = []
    for _ in range(batch):
        s = rng.randint(0, max_start)
        windows.append(ids[s : s + seq_len + 1].astype(np.int64))
    arr = np.stack(windows, 0)
    x = torch.tensor(arr[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(arr[:, 1:], dtype=torch.long, device=device)
    return x, y


def lm_loss(model, x, y):
    """Full-sequence CE for WordIdTransformer / Dual-style with forward(ids)->logits [B,T,V]."""
    # WordIdTransformer.forward expects optional key_padding_mask
    logits = model.forward(x)  # [B,T,V]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def _pred_from_prefix(model, x):
    """x: [1,T] в†’ predicted next id. Supports WordIdTransformer [B,T,V] and 0L [B,V]."""
    out = model.forward(x)
    if out.dim() == 3:
        return int(out[0, -1].argmax().item())
    return int(out[0].argmax().item())


@torch.no_grad()
def eval_hold_acc(model, ids, seq_len, device, n_windows=200, seed=1):
    model.eval()
    rng = random.Random(seed)
    n = h = 0
    max_start = len(ids) - seq_len - 1
    hold0 = int(0.8 * len(ids))
    for _ in range(n_windows):
        s = hold0 + rng.randint(0, max(1, int(0.2 * len(ids)) - seq_len - 2))
        s = min(s, max_start)
        window = ids[s : s + seq_len + 1].astype(np.int64)
        x = torch.tensor(window[:-1], dtype=torch.long, device=device).unsqueeze(0)
        gold = int(window[-1])
        pred = _pred_from_prefix(model, x)
        n += 1
        h += int(pred == gold)
    return h / max(n, 1)


@torch.no_grad()
def eval_battery(model, battery, stoi, surfaces, device, max_len):
    model.eval()
    n = h = maj_h = 0
    for it in battery:
        pref = [stoi.get(w, stoi[UNK]) for w in it["prefix"]][-max_len:]
        if not pref:
            continue
        gold = stoi.get(it["gold"], stoi[UNK])
        maj = stoi.get(it["majority_next"], stoi[UNK])
        x = torch.tensor([pref], dtype=torch.long, device=device)
        pred = _pred_from_prefix(model, x)
        n += 1
        h += int(pred == gold)
        maj_h += int(maj == gold)
    acc = h / max(n, 1)
    maj = maj_h / max(n, 1)
    return {"battery_acc": acc, "majority_acc": maj, "lift_vs_majority": acc - maj, "n": n}


@torch.no_grad()
def eval_battery_0l(model, battery, stoi, device, max_len):
    """0L: use only last token embedding path via model API."""
    return eval_battery(model, battery, stoi, None, device, max_len)


def make_opt(model, ritual):
    """AdamW (+ optional Adam) with WD on weights; biases/norms get wd=0."""
    wd = float(ritual.get("wd", 0.0))
    lr = float(ritual["lr"])
    opt_name = str(ritual.get("opt", "AdamW")).lower()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if opt_name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    return torch.optim.AdamW(groups, lr=lr)


def lr_at_step(step: int, ritual: dict, total_steps: int) -> float:
    """Linear warmup then cosine decay to lr_min (or constant if schedule says so)."""
    warmup = int(ritual.get("warmup", 0))
    lr = float(ritual["lr"])
    lr_min = float(ritual.get("lr_min", lr * 0.1))
    schedule = str(ritual.get("lr_schedule", "warmup_cosine")).lower()
    if step <= warmup:
        return lr * step / max(warmup, 1)
    if schedule in ("constant", "warmup_constant"):
        return lr
    t = (step - warmup) / max(total_steps - warmup, 1)
    t = min(max(t, 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def train_lm(
    tag,
    model,
    ids,
    ritual,
    steps,
    device,
    log,
    battery_hold,
    stoi,
    surfaces,
    ckpt_path: Path,
    zero_layer: bool = False,
    allow_early_stop: bool = False,
    start_step: int = 0,
):
    seq_len = ritual["seq_len"]
    micro = ritual["micro_batch"]
    eff = ritual["effective_batch"]
    accum = max(1, eff // micro)
    warmup = ritual["warmup"]
    ckpt_every = ritual["ckpt_every"]
    eval_every = ritual["eval_every"]
    start_step = max(0, int(start_step))
    opt = make_opt(model, ritual)
    log(
        f"  [{tag}] opt={ritual.get('opt')} wd={ritual.get('wd')} "
        f"lr={ritual.get('lr')}->{ritual.get('lr_min', ritual.get('lr'))} "
        f"schedule={ritual.get('lr_schedule', 'warmup_constant')} "
        f"warmup={warmup} steps={steps} resume_from={start_step}"
    )
    rng = random.Random(SEED + hash(tag) % 10000)
    curve = []
    best = {"lift": -99.0, "state": None, "bat": 0.0}
    model.train()
    t0 = time.time()
    running_loss = 0.0
    opt.zero_grad(set_to_none=True)
    # After resume, require more post-resume evals before early-stop can fire
    early_stop_min_step = max(int(0.3 * steps), start_step + 4 * eval_every)

    def snap(step):
        model.eval()
        hold = eval_hold_acc(model, ids, min(seq_len, 128), device, n_windows=150, seed=step)
        bat = eval_battery(model, battery_hold, stoi, surfaces, device, seq_len)
        cur_lr = opt.param_groups[0]["lr"]
        curve.append({"step": step, "hold_next": hold, **bat, "loss_ema": running_loss, "lr": cur_lr})
        log(
            f"  [{tag}] step {step}: hold@last={100*hold:.1f}% bat={100*bat['battery_acc']:.1f}% "
            f"lift_maj={100*bat['lift_vs_majority']:+.1f}pp loss~{running_loss:.3f} lr={cur_lr:.2e}"
        )
        if bat["lift_vs_majority"] >= best["lift"]:
            best["lift"] = bat["lift_vs_majority"]
            best["bat"] = bat["battery_acc"]
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        return bat

    if start_step <= 0:
        snap(0)
    else:
        log(f"  [{tag}] resume: skip step0 eval, continue from {start_step + 1}")
    for step in range(start_step + 1, steps + 1):
        cur_lr = lr_at_step(step, ritual, steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = sample_batch(ids, seq_len, micro, rng, device, model.pad_id)
        if zero_layer:
            # 0L trained as predict each position from last-only is wrong for full CE;
            # use last-position CE only: take prefix windows ending at each... simpler: CE on last token of window
            logits = model.logits_last_from_batch(x, key_padding_mask=(x == model.pad_id))
            loss = F.cross_entropy(logits, y[:, -1]) / accum
        else:
            loss = lm_loss(model, x, y) / accum
        loss.backward()
        running_loss = 0.95 * running_loss + 0.05 * float(loss.item() * accum) if running_loss else float(loss.item() * accum)
        if step % accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % eval_every == 0 or step == steps:
            bat = snap(step)
            if allow_early_stop and step >= early_stop_min_step:
                bats = [c["battery_acc"] for c in curve if c["step"] > 0]
                lifts = [c["lift_vs_majority"] for c in curve if c["step"] > 0]
                climbing = False
                if len(bats) >= 3:
                    climbing = bats[-1] > bats[-3] + 0.002 or lifts[-1] > lifts[-3] + 0.002
                if len(bats) >= 4 and bat["lift_vs_majority"] < 0:
                    recent_flat = max(bats[-3:]) - min(bats[-3:]) < 0.005
                    if recent_flat and not climbing:
                        log(f"[{tag}] EARLY_STOP @ {step}: lift<0 and flat")
                        break
        if step % ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "ritual": ritual, "tag": tag},
                ckpt_path,
            )
            log(f"  [{tag}] ckpt {ckpt_path.name} @ {step}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save(
        {"model": model.state_dict(), "ritual": ritual, "tag": tag, "curve": curve, "best": best},
        ckpt_path,
    )
    return {"curve": curve, "best": best, "wall_hours": (time.time() - t0) / 3600}


def load_battery_hold(frac=0.2):
    items = [json.loads(l) for l in BATTERY.read_text(encoding="utf-8").splitlines() if l.strip()]
    rr = random.Random(SEED)
    rr.shuffle(items)
    n = max(800, int(frac * len(items)))
    return items[:n], items[n:]


# ---------- S1 ----------

def _vram_mib(bytes_: int) -> float:
    return round(bytes_ / (1024 * 1024), 1)


def _cuda_mem_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_mib": _vram_mib(total_b),
        "free_mib": _vram_mib(free_b),
        "alloc_mib": _vram_mib(torch.cuda.memory_allocated()),
        "reserved_mib": _vram_mib(torch.cuda.memory_reserved()),
        "peak_alloc_mib": _vram_mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": _vram_mib(torch.cuda.max_memory_reserved()),
    }


def probe_train_step_vram(
    V: int,
    d: int,
    n_head: int,
    n_layer: int,
    seq: int,
    micro: int,
    ids: np.memmap,
    fps: torch.Tensor,
    device,
    zero_layer: bool = False,
) -> dict:
    """One forward+backward; returns peak alloc MiB or OOM error."""
    if not torch.cuda.is_available():
        return {"ok": False, "error": "no_cuda", "seq": seq, "micro": micro, "d": d, "n_layer": n_layer}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_mem_snapshot()
    try:
        if zero_layer:
            m = ZeroLayerWordLM(V, d_model=d, max_len=seq, mode="last").to(device)
            m.init_from_fps(fps)
            x, y = sample_batch(ids, seq, micro, random.Random(0), device, m.pad_id)
            logits = m.logits_last_from_batch(x, key_padding_mask=(x == m.pad_id))
            loss = F.cross_entropy(logits, y[:, -1])
        else:
            m = WordIdTransformer(V, d, n_head, n_layer, seq, 0.1).to(device)
            m.init_from_fps(fps)
            x, y = sample_batch(ids, seq, micro, random.Random(0), device, m.pad_id)
            loss = lm_loss(m, x, y)
        loss.backward()
        peak = _cuda_mem_snapshot()
        ok = True
        err = None
        del m, x, y, loss
    except RuntimeError as e:
        ok = False
        err = str(e).split("\n")[0][:240]
        peak = _cuda_mem_snapshot()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {
        "ok": ok,
        "error": err,
        "n_layer": n_layer,
        "d": d,
        "n_head": n_head,
        "seq": seq,
        "micro": micro,
        "zero_layer": zero_layer,
        "before_free_mib": before.get("free_mib"),
        "peak_alloc_mib": peak.get("peak_alloc_mib"),
        "peak_reserved_mib": peak.get("peak_reserved_mib"),
        "total_mib": before.get("total_mib"),
    }


def vram_capacity_sweep(ids, ritual, fps, device, log) -> dict:
    """Matrix of fwd+bwd peaks — what fits on this GPU."""
    V = fps.size(0)
    gpu = _cuda_mem_snapshot()
    log(f"[vram] GPU={gpu.get('device')} total={gpu.get('total_mib')} MiB free={gpu.get('free_mib')} MiB")
    configs = []
    # Primary 4L/d256 grid
    for seq in (128, 256, 512):
        for micro in (1, 2, 4, 8):
            configs.append((ritual["n_layer"], ritual["d"], ritual["n_head"], seq, micro, False))
    # Stretch: deeper/wider if cheap probes pass
    for seq, micro in ((256, 2), (512, 1), (512, 2)):
        configs.append((4, 384, 6, seq, micro, False))
        configs.append((6, 256, 4, seq, micro, False))
        configs.append((4, 512, 8, seq, micro, False))
    # 0L reference @ ritual seq candidates
    for seq in (256, 512):
        for micro in (2, 4, 8):
            configs.append((0, ritual["d"], ritual["n_head"], seq, micro, True))

    rows = []
    for n_layer, d, n_head, seq, micro, zl in configs:
        tag = f"{'0L' if zl else f'{n_layer}L'}/d{d} seq={seq} micro={micro}"
        log(f"[vram] probe {tag} ...")
        row = probe_train_step_vram(V, d, n_head, n_layer if not zl else 0, seq, micro, ids, fps, device, zl)
        status = "OK" if row["ok"] else "OOM"
        log(
            f"[vram] {status} {tag} peak_alloc={row.get('peak_alloc_mib')} "
            f"peak_reserved={row.get('peak_reserved_mib')} MiB"
            + (f" | {row['error']}" if row.get("error") else "")
        )
        rows.append(row)

    # Lock: longest seq among OK primary 4L/d256, then largest micro with ~15% headroom
    total = float(gpu.get("total_mib") or 4096)
    headroom_ok = lambda r: (r.get("peak_alloc_mib") or 1e9) < 0.85 * total

    def pick(rows_ok):
        if not rows_ok:
            return None
        # prefer headroom; else any OK
        pool = [r for r in rows_ok if headroom_ok(r)] or rows_ok
        pool.sort(key=lambda r: (r["seq"], r["micro"]), reverse=True)
        best = pool[0]
        return {
            "seq_len": best["seq"],
            "micro_batch": best["micro"],
            "d": best["d"],
            "n_layer": best["n_layer"],
            "n_head": best.get("n_head"),
            "peak_alloc_mib": best.get("peak_alloc_mib"),
        }

    primary_ok = [
        r
        for r in rows
        if r["ok"]
        and not r["zero_layer"]
        and r["n_layer"] == ritual["n_layer"]
        and r["d"] == ritual["d"]
    ]
    lock = pick(primary_ok)
    if lock is None:
        any_ok = [r for r in rows if r["ok"] and not r["zero_layer"]]
        any_ok.sort(key=lambda r: (r["d"] == ritual["d"], r["n_layer"] == ritual["n_layer"]), reverse=True)
        lock = pick(any_ok)

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "probes": rows,
        "lock_suggestion": lock,
        "ok_count": sum(1 for r in rows if r["ok"]),
        "oom_count": sum(1 for r in rows if not r["ok"]),
    }
    write_json(RES / "stage169_s1_vram.json", out)
    log(f"[vram] wrote stage169_s1_vram.json lock={lock}")
    return out


def run_s1() -> int:
    logp = RES / "_stage169_s1_log.txt"
    logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    if not (RES / "stage169_s0_decision.json").exists():
        rc = run_s0()
        if rc != 0:
            return rc
    log(f"S1 start {datetime.now(timezone.utc).isoformat()}")
    ritual = load_ritual()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, cb = open_ids()
    _, surfaces, stoi = load_codebook()
    bat_hold, _ = load_battery_hold()
    stack = freeze_stack(device)
    fps = init_fps(surfaces, stack, device)
    del stack
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Full VRAM capacity matrix (peaks) before smoke
    log("##### S1 VRAM capacity sweep #####")
    vram = vram_capacity_sweep(ids, ritual, fps, device, log)
    lock = vram.get("lock_suggestion")
    if not lock:
        log("[vram] FATAL: nothing fits")
        write_json(
            RES / "stage169_s1_decision.json",
            {"verdict": "S1_VRAM_FAIL", "vram": vram, "next": "shrink d or seq further"},
        )
        return 1

    ritual["seq_len"] = int(lock["seq_len"])
    ritual["micro_batch"] = int(lock["micro_batch"])
    ritual["d"] = int(lock["d"])
    ritual["n_layer"] = int(lock["n_layer"])
    if lock.get("n_head"):
        ritual["n_head"] = int(lock["n_head"])
    elif ritual["d"] % ritual["n_head"] != 0:
        ritual["n_head"] = 4 if ritual["d"] % 4 == 0 else 8
    save_ritual(ritual)
    ok_512 = any(
        r["ok"] and not r["zero_layer"] and r["n_layer"] == ritual["n_layer"] and r["d"] == ritual["d"] and r["seq"] == 512
        for r in vram["probes"]
    )
    log(f"[vram] LOCK ritual seq={ritual['seq_len']} micro={ritual['micro_batch']} d={ritual['d']} L={ritual['n_layer']}")

    # smoke train primary — also record peak during a few steps
    log(f"##### S1 smoke primary {ritual['n_layer']}L d{ritual['d']} seq={ritual['seq_len']} micro={ritual['micro_batch']} #####")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = WordIdTransformer(
        len(surfaces), ritual["d"], ritual["n_head"], ritual["n_layer"], ritual["seq_len"], 0.1
    ).to(device)
    model.init_from_fps(fps)
    tr = train_lm(
        "s1_primary",
        model,
        ids,
        ritual,
        ritual["smoke_steps"],
        device,
        log,
        bat_hold,
        stoi,
        surfaces,
        CKPT / "stage169_s1_primary.pt",
    )
    smoke_peak = _cuda_mem_snapshot() if torch.cuda.is_available() else {}
    log(f"[vram] smoke primary peak_alloc={smoke_peak.get('peak_alloc_mib')} MiB")

    # smoke 0L short
    log("##### S1 smoke 0L #####")
    m0 = ZeroLayerWordLM(len(surfaces), d_model=ritual["d"], max_len=ritual["seq_len"], mode="last").to(device)
    m0.init_from_fps(fps)
    tr0 = train_lm(
        "s1_0L",
        m0,
        ids,
        {**ritual, "smoke_steps": ritual["smoke_steps"]},
        min(ritual["smoke_steps"], 10000),
        device,
        log,
        bat_hold,
        stoi,
        surfaces,
        CKPT / "stage169_s1_0L.pt",
        zero_layer=True,
    )

    # compact peak table for mini report
    peak_lines = []
    for r in vram["probes"]:
        if r["zero_layer"]:
            continue
        if r["d"] != ritual["d"] or r["n_layer"] != ritual["n_layer"]:
            if not r["ok"]:
                continue
        mark = "OK" if r["ok"] else "OOM"
        peak_lines.append(
            f"{r['n_layer']}L/d{r['d']} seq{r['seq']}×{r['micro']}: {mark} peak={r.get('peak_alloc_mib')} MiB"
        )

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": 1,
        "ritual": ritual,
        "seq512_ok": ok_512,
        "vram": vram,
        "smoke_peak_alloc_mib": smoke_peak.get("peak_alloc_mib"),
        "primary": sanitize_train_result(tr),
        "zero": sanitize_train_result(tr0),
        "verdict": "S1_SMOKE_OK",
        "next": "S2 full primary",
    }
    write_json(RES / "stage169_s1_decision.json", out)
    write_mini(
        RES / "stage169_s1_mini.md",
        "Stage169 S1 — smoke + VRAM peaks",
        [
            f"GPU: {vram['gpu'].get('device')} total={vram['gpu'].get('total_mib')} MiB",
            f"LOCK: seq={ritual['seq_len']} micro={ritual['micro_batch']} d={ritual['d']} L={ritual['n_layer']}",
            f"seq512_ok={ok_512}; smoke_peak_alloc={smoke_peak.get('peak_alloc_mib')} MiB",
            "Peaks (primary grid + stretch OK):",
            *peak_lines[:24],
            f"primary best lift_maj={100*tr['best']['lift']:+.1f}pp bat={100*tr['best']['bat']:.1f}%",
            f"0L best lift_maj={100*tr0['best']['lift']:+.1f}pp",
            "Next: S2 full primary train",
        ],
        "S1_SMOKE_OK",
    )
    log("[S1] S1_SMOKE_OK")
    return 0


def run_s2() -> int:
    logp = RES / "_stage169_s2_log.txt"
    if not logp.exists():
        logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    if not (RES / "stage169_s1_decision.json").exists():
        rc = run_s1()
        if rc != 0:
            return rc
    log(f"S2 start {datetime.now(timezone.utc).isoformat()}")
    ritual = load_ritual()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, _ = open_ids()
    _, surfaces, stoi = load_codebook()
    bat_hold, _ = load_battery_hold()
    stack = freeze_stack(device)
    fps = init_fps(surfaces, stack, device)
    del stack
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = WordIdTransformer(
        len(surfaces), ritual["d"], ritual["n_head"], ritual["n_layer"], ritual["seq_len"], 0.1
    ).to(device)
    model.init_from_fps(fps)
    start_step = 0
    ck2 = CKPT / "stage169_s2_primary.pt"
    if ck2.exists():
        blob = torch.load(ck2, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(blob["model"], strict=False)
            start_step = int(blob.get("step") or 0)
            log(f"[s2] resume from {ck2.name} @ step={start_step}")
        except Exception as e:
            log(f"[s2] resume load failed, try S1: {e}")
            start_step = 0
    if start_step <= 0:
        ck1 = CKPT / "stage169_s1_primary.pt"
        if ck1.exists():
            blob = torch.load(ck1, map_location="cpu", weights_only=False)
            try:
                model.load_state_dict(blob["model"], strict=False)
                log("[s2] warm-start from S1")
            except Exception as e:
                log(f"[s2] warm-start skip: {e}")

    log(
        f"##### S2 full primary {ritual['n_layer']}L d{ritual['d']} "
        f"seq={ritual['seq_len']} micro={ritual['micro_batch']} steps={ritual['full_steps']} "
        f"opt={ritual.get('opt')} wd={ritual.get('wd')} "
        f"schedule={ritual.get('lr_schedule', 'warmup_constant')} resume={start_step} #####"
    )
    tr = train_lm(
        "s2_primary",
        model,
        ids,
        ritual,
        ritual["full_steps"],
        device,
        log,
        bat_hold,
        stoi,
        surfaces,
        ck2,
        allow_early_stop=True,
        start_step=start_step,
    )
    # verdict from curve
    lifts = [c["lift_vs_majority"] for c in tr["curve"] if c["step"] > 0]
    final_lift = lifts[-1] if lifts else tr["best"]["lift"]
    bats = [c["battery_acc"] for c in tr["curve"] if c["step"] > 0]
    climbing = len(bats) >= 3 and bats[-1] > bats[max(0, len(bats)//3)] + 0.005
    if final_lift < 0 and not climbing and len(lifts) >= 3:
        verdict = "S2_EARLY_OR_NULL"
    elif final_lift >= 0.05:
        verdict = "S2_PROMISING"
    else:
        verdict = "S2_CONTINUE"
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": 2,
        "ritual": ritual,
        "train": {**sanitize_train_result(tr), "curve_tail": (tr.get("curve") or [])[-5:]},
        "final_lift_maj": final_lift,
        "verdict": verdict,
        "next": "S3 matched 0L",
    }
    write_json(RES / "stage169_s2_decision.json", out)
    write_mini(
        RES / "stage169_s2_mini.md",
        "Stage169 S2 вЂ” primary full train mini report",
        [
            f"steps budget={ritual['full_steps']} wall={tr['wall_hours']:.2f}h",
            f"best lift_maj={100*tr['best']['lift']:+.1f}pp bat={100*tr['best']['bat']:.1f}%",
            f"final lift_maj={100*final_lift:+.1f}pp climbing={climbing}",
            f"Verdict {verdict}",
        ],
        verdict,
    )
    log(f"[S2] {verdict}")
    return 0


def run_s3() -> int:
    logp = RES / "_stage169_s3_log.txt"
    logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    if not (RES / "stage169_s2_decision.json").exists():
        rc = run_s2()
        if rc != 0:
            return rc
    log(f"S3 start {datetime.now(timezone.utc).isoformat()}")
    ritual = load_ritual()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, _ = open_ids()
    _, surfaces, stoi = load_codebook()
    bat_hold, _ = load_battery_hold()
    stack = freeze_stack(device)
    fps = init_fps(surfaces, stack, device)
    m0 = ZeroLayerWordLM(len(surfaces), d_model=ritual["d"], max_len=ritual["seq_len"], mode="last").to(device)
    m0.init_from_fps(fps)
    tr = train_lm(
        "s3_0L",
        m0,
        ids,
        ritual,
        ritual["full_steps"],
        device,
        log,
        bat_hold,
        stoi,
        surfaces,
        CKPT / "stage169_s3_0L.pt",
        zero_layer=True,
    )
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": 3,
        "train": sanitize_train_result(tr),
        "verdict": "S3_0L_DONE",
        "next": "S4 probes",
    }
    write_json(RES / "stage169_s3_decision.json", out)
    write_mini(
        RES / "stage169_s3_mini.md",
        "Stage169 S3 вЂ” 0L matched mini report",
        [
            f"0L best lift_maj={100*tr['best']['lift']:+.1f}pp bat={100*tr['best']['bat']:.1f}%",
            f"wall={tr['wall_hours']:.2f}h",
        ],
        "S3_0L_DONE",
    )
    log("[S3] S3_0L_DONE")
    return 0


def run_s4() -> int:
    logp = RES / "_stage169_s4_log.txt"
    logp.write_text("", encoding="utf-8")
    log = lambda m: log_to(logp, m)
    for need in ("s2", "s3"):
        if not (RES / f"stage169_{need}_decision.json").exists():
            log(f"[s4] missing {need}")
            return 1
    log(f"S4 start {datetime.now(timezone.utc).isoformat()}")
    ritual = load_ritual()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, surfaces, stoi = load_codebook()
    bat_hold, _ = load_battery_hold(frac=0.25)
    # load models
    primary = WordIdTransformer(
        len(surfaces), ritual["d"], ritual["n_head"], ritual["n_layer"], ritual["seq_len"], 0.1
    ).to(device)
    blob = torch.load(CKPT / "stage169_s2_primary.pt", map_location="cpu", weights_only=False)
    primary.load_state_dict(blob["model"], strict=False)
    z = ZeroLayerWordLM(len(surfaces), d_model=ritual["d"], max_len=ritual["seq_len"], mode="last").to(device)
    blob0 = torch.load(CKPT / "stage169_s3_0L.pt", map_location="cpu", weights_only=False)
    z.load_state_dict(blob0["model"], strict=False)

    b1 = eval_battery(primary, bat_hold, stoi, surfaces, device, ritual["seq_len"])
    b0 = eval_battery(z, bat_hold, stoi, surfaces, device, ritual["seq_len"])
    gap = b1["battery_acc"] - b0["battery_acc"]
    lift = b1["lift_vs_majority"]

    if gap >= 0.05 and lift >= 0.05:
        final = "CONTEXT_STEP_YES"
    elif gap >= 0.02 or lift >= 0.02:
        final = "MIXED"
    else:
        final = "NO_CONTEXT_AT_0p5B_CE"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": 4,
        "primary_battery": b1,
        "zero_battery": b0,
        "gaps": {"primary_minus_0L_battery": gap, "primary_lift_vs_majority": lift},
        "verdict": final,
        "ritual": ritual,
        "note": "BPE secondary skipped unless time; ALL not primary gate",
    }
    write_json(RES / "stage169_context_0p5b_decision.json", out)
    write_mini(
        RES / "stage169_s4_mini.md",
        "Stage169 S4 вЂ” context verdict",
        [
            f"Primary bat={100*b1['battery_acc']:.1f}% lift_maj={100*lift:+.1f}pp",
            f"0L bat={100*b0['battery_acc']:.1f}%",
            f"primaryв€’0L battery={100*gap:+.1f}pp",
            f"**{final}**",
            "If MIXED/climbing в†’ S5 stretch to 1B; if NO_CONTEXT в†’ not more 50M tricks",
        ],
        final,
    )
    log(f"[S4] {final}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="0", help="0|1|2|3|4|all")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    runners = {"0": run_s0, "1": run_s1, "2": run_s2, "3": run_s3, "4": run_s4}
    seq = ["0", "1", "2", "3", "4"] if args.step == "all" else [args.step]
    for s in seq:
        rc = runners[s]()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

