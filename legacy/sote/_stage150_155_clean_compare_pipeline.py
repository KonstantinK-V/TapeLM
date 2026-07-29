"""
Stages 150-155 — clean compare pipeline (post-149).

Plan: results/plan_150_plus_clean_compare.md

  150  trunks G / S / S+  x  word_fp + ws_bpe_rand   (G reused from 149)
  151  orthogonal 1-factor on G and S+ (word_fp only)
  152  context/budget match on S+ (+152c fat_frac=0.45)
  153  scale x batch grids (word_fp; G and S+)
  154  capacity x data factorial (S+ ritual; N=100k/500k; d pairs)
  155  eval-surface note from existing ckpts (no new train)

Rules: exact@1, hops OUT, resume if stage decision exists, matched d per trunk.
Follow-on (queued via `_run_queue_150_158.py`): `_stage156_157_morph_codebook_pipeline.py`
  156 shared morph → 157 rare morph-only → 158 ComposeLayer preprocess+LM.

Run (waits for 149):
  python _stage150_155_clean_compare_pipeline.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    _warmup_then_constant,
    build_ts_repeat_mix,
    collate_word_id_batch,
    compose_plain,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage111_112_follow import _encode_words, eval_bpe_word_holds  # noqa: E402

DEC149 = RES / "stage149_matched_hparams_decision.json"
LOG = RES / "_stage150_155_log.txt"
PIPELINE_DEC = RES / "stage150_155_clean_compare_pipeline_decision.json"
BPE_TOK = RES / "stage150_155_bpe_tokenizer.json"
PLAN = RES / "plan_150_plus_clean_compare.md"

CORPUS_100K = ROOT / "data" / "external_tinystories_100k_85.txt"
CORPUS_500K = ROOT / "data" / "external_tinystories_500k_85.txt"

FT_STEPS = 40_000
FT_STEPS_SMALL_N = 25_000
WARMUP = 200
EVAL_EVERY_WORD = 4000
EVAL_EVERY_BPE = 5000
V_BPE = 8000
SEED_MIX = 272

# Matched hold sizes (word vs BPE). Unequal N was a STORY-noise confound.
EV_SEEN_N = 400
EV_STORY_N = 300
FIN_SEEN_N = 800
FIN_STORY_N = 500


def stable_seed(*parts) -> int:
    """Process-stable seed (avoid Python's salted hash())."""
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF

# --- locked trunks (full recipes; never mix levers across trunks) ---
TRUNKS = {
    "G": {
        "batch": 16,
        "lr": 3e-4,
        "opt": "AdamW",
        "wd": 0.01,
        "d": 256,
        "n_layer": 4,
        "n_head": 4,
        "fat_frac": 0.75,
        "word_max_len": 16,
        "bpe_max_len": 48,
        "n_positions": 64,
    },
    "S": {
        "batch": 8,
        "lr": 1e-3,
        "opt": "Adam",
        "wd": 0.0,
        "d": 256,
        "n_layer": 2,
        "n_head": 4,
        "fat_frac": 0.75,
        "word_max_len": 16,
        "bpe_max_len": 48,
        "n_positions": 64,
    },
    "Splus": {
        "batch": 8,
        "lr": 1e-3,
        "opt": "Adam",
        "wd": 0.0,
        "d": 512,
        "n_layer": 4,
        "n_head": 8,
        "fat_frac": 0.75,
        "word_max_len": 16,
        "bpe_max_len": 48,
        "n_positions": 64,
    },
}

# 154 capacity pairs under S+ ritual (opt/batch/lr fixed to S+)
CAP_PAIRS = {
    "hist": {"d": 256, "n_layer": 2, "n_head": 4},
    "headroom": {"d": 512, "n_layer": 4, "n_head": 8},
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


def wait_json(path: Path, label: str, timeout_s=48 * 3600, poll=60):
    log(f"[wait] for {path.name} ({label}) ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] {label} done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still {label} ... {int(time.time()-t0)}s")
    raise TimeoutError(f"{label} not ready: {path}")


def done(path: Path) -> dict | None:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("verdict"):
            log(f"[skip] {path.name} already done verdict={d.get('verdict')}")
            return d
    return None


def write_dec(path: Path, out: dict):
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[write] {path.name}")


def make_opt(params, trunk: dict):
    if trunk["opt"] == "AdamW":
        return torch.optim.AdamW(params, lr=trunk["lr"], weight_decay=trunk["wd"])
    return torch.optim.Adam(params, lr=trunk["lr"])


def load_phrases(n_target: int | None = None) -> list[str]:
    if n_target is None or n_target <= 100_000:
        phrases = ensure_100k()
        if n_target and n_target < len(phrases):
            rr = random.Random(153)
            phrases = phrases[:]
            rr.shuffle(phrases)
            phrases = phrases[:n_target]
        return phrases
    src = CORPUS_500K if CORPUS_500K.exists() else CORPUS_100K
    raw = [
        ln.strip()
        for ln in src.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if n_target and n_target < len(raw):
        rr = random.Random(154)
        raw = raw[:]
        rr.shuffle(raw)
        raw = raw[:n_target]
    log(f"[data] loaded n={len(raw)} from {src.name}")
    return raw


def setup_stack(device):
    cfg = Config()
    cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
    cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
    cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
    parent = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent.exists():
        parent = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent)
    for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
        for p in mod.parameters():
            p.requires_grad_(False)
        mod.eval()
    return cfg, stack


def make_mix(phrases, cfg, seed=SEED_MIX):
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=seed)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    return train, hold_seen, hold_rare, hold_story, meta


def train_or_load_bpe(phrases: list[str], cache: Path | None = None) -> Tokenizer:
    cache = cache or BPE_TOK
    if cache.exists():
        log(f"[bpe] reuse {cache.name}")
        return Tokenizer.from_file(str(cache))
    # prefer 149 tok only for the default 100k cache
    alt = RES / "stage149_matched_bpe_tokenizer.json"
    if cache == BPE_TOK and alt.exists():
        log(f"[bpe] reuse {alt.name}")
        tok = Tokenizer.from_file(str(alt))
        tok.save(str(cache))
        return tok
    log(f"[bpe] train whitespace-BPE V={V_BPE} -> {cache.name} on n={len(phrases)}")
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
    cache.parent.mkdir(exist_ok=True)
    tok.save(str(cache))
    return tok


def piece_fp(stack, piece: str, dim: int, device):
    s = "".join(c for c in piece if c.isalnum() or c == " ").strip()
    if not s:
        return torch.zeros(dim, device=device)
    try:
        return F.normalize(
            compose_plain(stack.encoder, stack.composer, s.replace(" ", ""), device).detach(),
            dim=-1,
        )
    except Exception:
        try:
            return F.normalize(stack.w(s.split()[0]).detach(), dim=-1)
        except Exception:
            return torch.zeros(dim, device=device)


def init_gpt2_emb_rand(model):
    with torch.no_grad():
        model.transformer.wte.weight.normal_(std=0.02)


def metrics_pack(seen, story):
    return {
        "seen_obj": seen["obj"]["hit1"],
        "seen_rel": seen["roles"].get("rel", {}).get("hit1", 0.0),
        "story_all": story["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": story["obj"]["hit1"],
    }


def train_word(
    tag: str,
    trunk: dict,
    phrases: list[str],
    device,
    cfg,
    stack,
    *,
    steps: int | None = None,
    seed: int = 150,
    ckpt_name: str | None = None,
) -> dict:
    steps = steps or FT_STEPS
    log(
        f"\n======== WORD {tag} batch={trunk['batch']} lr={trunk['lr']} "
        f"{trunk['opt']} d={trunk['d']} {trunk['n_layer']}L/{trunk['n_head']}H "
        f"fat={trunk['fat_frac']} steps={steps} ========"
    )
    train, hold_seen, hold_rare, hold_story, meta = make_mix(phrases, cfg)
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    model = WordIdTransformer(
        len(words), trunk["d"], trunk["n_head"], trunk["n_layer"], trunk["word_max_len"], 0.1
    ).to(device)
    model.init_from_fps(fps)
    # Foundation fp is d=256; for d>256 fill remaining dims with small noise (not dead zeros).
    if trunk["d"] > fps.shape[1]:
        with torch.no_grad():
            extra = trunk["d"] - fps.shape[1]
            model.tok.weight[: len(words), fps.shape[1] :].normal_(std=0.02)
            model.tok.weight[: len(words)] = F.normalize(model.tok.weight[: len(words)], dim=-1)
            log(f"[word] padded emb +{extra} dims with N(0,0.02) then renorm")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[word] V={len(words)} params={n_params/1e6:.2f}M")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 60000, seed + 1), stoi)
    if not fat_p:
        fat_p = story_p

    ev_seen = _subsample(hold_seen, EV_SEEN_N, seed + 10)
    ev_story = _subsample(hold_story, EV_STORY_N, seed + 11)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), FIN_SEEN_N), seed + 20)
    fin_story = _subsample(hold_story, min(len(hold_story), FIN_STORY_N), seed + 21)

    opt = make_opt(model.parameters(), trunk)
    rr = random.Random(seed)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []
    batch = trunk["batch"]
    fat_frac = trunk["fat_frac"]

    def snap(step):
        model.eval()
        s = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        st = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
        obj = s["obj"]["hit1"]
        rel = s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(f"  [{tag}] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(
                story_all=sall,
                obj=obj,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        model.train()

    model.train()
    snap(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, trunk["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac))) if fat_frac > 0 else 0
        n_fat = min(n_fat, batch)
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
        if step % EVAL_EVERY_WORD == 0 or step == steps:
            snap(step)

    last_story = curve[-1]["story_all"] if curve else -1.0
    last_obj = curve[-1]["obj"] if curve else -1.0
    model.load_state_dict(best["state"])
    model.eval()
    fs = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    ft = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    ck = CKPT / (ckpt_name or f"stage150_155_{tag}.pt")
    torch.save({"word_tf": best["state"], "surfaces": words, "trunk": trunk, "tag": tag}, ck)
    out = {
        "arm": tag,
        "unit": "word_fp",
        "trunk": {k: trunk[k] for k in ("batch", "lr", "opt", "wd", "d", "n_layer", "n_head", "fat_frac")},
        "V": len(words),
        "params": n_params,
        "steps": steps,
        "mix_meta": {k: meta.get(k) for k in ("n_train", "n_hold_seen", "n_story_hold") if k in meta},
        "curve": curve,
        "ckpt": str(ck),
        "story_all_last": last_story,
        "seen_obj_last": last_obj,
        "ckpt_select": "best_ev_story_then_obj",
        **metrics_pack(fs, ft),
    }
    return out


def train_bpe(
    tag: str,
    trunk: dict,
    phrases: list[str],
    tok: Tokenizer,
    device,
    cfg,
    *,
    steps: int | None = None,
    seed: int = 150,
    ckpt_name: str | None = None,
    bpe_max_len: int | None = None,
    n_positions: int | None = None,
) -> dict:
    steps = steps or FT_STEPS
    bpe_max_len = bpe_max_len if bpe_max_len is not None else trunk["bpe_max_len"]
    n_positions = n_positions if n_positions is not None else trunk["n_positions"]
    fat_frac = float(trunk.get("fat_frac", 0.75))
    log(
        f"\n======== BPE {tag} batch={trunk['batch']} lr={trunk['lr']} "
        f"{trunk['opt']} d={trunk['d']} {trunk['n_layer']}L/{trunk['n_head']}H "
        f"maxlen={bpe_max_len} npos={n_positions} fat={fat_frac} steps={steps} ========"
    )
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()
    train, hold_seen, hold_rare, hold_story, meta = make_mix(phrases, cfg)
    fat_seqs, story_seqs = [], []
    for ln in train:
        ids = _encode_words(tok, ln["words"], max_len=bpe_max_len, bos=bos, eos=eos, pad=pad)
        if len(ids) < 3:
            continue
        if ln.get("bucket") == "fat_train":
            fat_seqs.append(ids)
        else:
            story_seqs.append(ids)
    if not fat_seqs:
        fat_seqs = story_seqs
    if not story_seqs:
        story_seqs = fat_seqs
    log(f"[bpe] fat_seqs={len(fat_seqs)} story_seqs={len(story_seqs)} V={V}")

    conf = GPT2Config(
        vocab_size=V,
        n_positions=max(n_positions, bpe_max_len + 2),
        n_embd=trunk["d"],
        n_layer=trunk["n_layer"],
        n_head=trunk["n_head"],
        n_inner=4 * trunk["d"],
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    model = GPT2LMHeadModel(conf).to(device)
    init_gpt2_emb_rand(model)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[bpe] params={n_params/1e6:.2f}M")

    opt = make_opt(model.parameters(), trunk)
    rr = random.Random(seed)
    ev_seen = _subsample(hold_seen, EV_SEEN_N, seed + 10)
    ev_story = _subsample(hold_story, EV_STORY_N, seed + 11)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), FIN_SEEN_N), seed + 20)
    fin_story = _subsample(hold_story, min(len(hold_story), FIN_STORY_N), seed + 21)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []
    batch = trunk["batch"]

    def snap(step):
        model.eval()
        s = eval_bpe_word_holds(
            model, tok, ev_seen, device, max_n_lines=len(ev_seen), encode_max_len=bpe_max_len
        )
        st = eval_bpe_word_holds(
            model, tok, ev_story, device, max_n_lines=len(ev_story), encode_max_len=bpe_max_len
        )
        obj = s["obj"]["hit1"]
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "story_all": sall})
        log(f"  [{tag}] step {step}: SEEN obj={100*obj:.1f}% | STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(
                story_all=sall,
                obj=obj,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        model.train()

    model.train()
    snap(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, trunk["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac))) if fat_frac > 0 else 0
        n_fat = min(n_fat, batch)
        batch_seqs = [rr.choice(fat_seqs) for _ in range(n_fat)] + [
            rr.choice(story_seqs) for _ in range(batch - n_fat)
        ]
        maxlen = max(len(s) for s in batch_seqs)
        x = torch.full((batch, maxlen), pad, dtype=torch.long, device=device)
        for i, s in enumerate(batch_seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        labels = x.clone()
        labels[labels == pad] = -100
        loss = model(x, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY_BPE == 0 or step == steps:
            snap(step)

    last_story = curve[-1]["story_all"] if curve else -1.0
    last_obj = curve[-1]["obj"] if curve else -1.0
    model.load_state_dict(best["state"])
    model.eval()
    fs = eval_bpe_word_holds(
        model, tok, fin_seen, device, max_n_lines=len(fin_seen), encode_max_len=bpe_max_len
    )
    ft = eval_bpe_word_holds(
        model, tok, fin_story, device, max_n_lines=len(fin_story), encode_max_len=bpe_max_len
    )
    ck = CKPT / (ckpt_name or f"stage150_155_{tag}.pt")
    torch.save({"gpt2": best["state"], "tag": tag, "trunk": trunk}, ck)
    out = {
        "arm": tag,
        "unit": "ws_bpe_rand",
        "trunk": {k: trunk[k] for k in ("batch", "lr", "opt", "wd", "d", "n_layer", "n_head", "fat_frac")},
        "bpe_max_len": bpe_max_len,
        "n_positions": conf.n_positions,
        "V": V,
        "params": n_params,
        "steps": steps,
        "curve": curve,
        "ckpt": str(ck),
        "story_all_last": last_story,
        "seen_obj_last": last_obj,
        "ckpt_select": "best_ev_story_then_obj",
        **metrics_pack(fs, ft),
    }
    return out


def slim(arm: dict) -> dict:
    keys = (
        "arm", "unit", "trunk", "V", "params", "steps", "seen_obj", "seen_rel",
        "story_all", "story_obj", "story_all_last", "seen_obj_last", "ckpt_select",
        "ckpt", "bpe_max_len", "n_positions", "note", "reuse_from", "n_phrases",
    )
    return {k: arm[k] for k in keys if k in arm}


# -------------------- Stage 150 --------------------
def run_150(device, cfg, stack, phrases, tok, up149: dict) -> dict:
    dec = RES / "stage150_dual_trunk_decision.json"
    prev = done(dec)
    if prev:
        return prev
    log("\n########## STAGE 150 dual-trunk + S+ headroom ##########")
    log(
        "[150] Retrain G under same protocol as S/S+ (fat lock on BPE, matched hold N). "
        "149 kept as prior GPT-only table; not copied into 150 gaps."
    )
    arms = {}
    for tname in ("G", "S", "Splus"):
        tr = dict(TRUNKS[tname])
        arms[f"{tname}_word_fp"] = train_word(
            f"{tname}_word_fp",
            tr,
            phrases,
            device,
            cfg,
            stack,
            seed=stable_seed(150, tname, "word"),
        )
        arms[f"{tname}_ws_bpe_rand"] = train_bpe(
            f"{tname}_ws_bpe_rand",
            tr,
            phrases,
            tok,
            device,
            cfg,
            seed=stable_seed(150, tname, "bpe"),
        )

    def gap(a, b):
        if a is None or b is None:
            return None
        return a - b

    table = {
        t: {
            "word_story": (arms.get(f"{t}_word_fp") or {}).get("story_all"),
            "bpe_story": (arms.get(f"{t}_ws_bpe_rand") or {}).get("story_all"),
            "word_story_last": (arms.get(f"{t}_word_fp") or {}).get("story_all_last"),
            "bpe_story_last": (arms.get(f"{t}_ws_bpe_rand") or {}).get("story_all_last"),
            "unit_gap_bpe_minus_word": gap(
                (arms.get(f"{t}_ws_bpe_rand") or {}).get("story_all"),
                (arms.get(f"{t}_word_fp") or {}).get("story_all"),
            ),
            "unit_gap_last": gap(
                (arms.get(f"{t}_ws_bpe_rand") or {}).get("story_all_last"),
                (arms.get(f"{t}_word_fp") or {}).get("story_all_last"),
            ),
        }
        for t in ("G", "S", "Splus")
    }
    s_story = table["S"]["word_story"] or 0.0
    sp_story = table["Splus"]["word_story"] or 0.0
    headroom_delta = sp_story - s_story
    primary = "Splus"
    if sp_story + 1e-9 < s_story - 0.02:
        primary = "S"
        note_primary = "S+ STORY worse than S by >2pp @100k; primary=S but dim NOT closed"
    else:
        note_primary = "S+ is default SOTE primary (headroom); flat@100k does not close dim"

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "dual_trunk_plus_headroom",
        "plan": str(PLAN),
        "arms": {k: slim(v) for k, v in arms.items()},
        "table_story": table,
        "headroom_Splus_minus_S_word_story": headroom_delta,
        "default_primary_trunk": primary,
        "note_primary": note_primary,
        "story_protocol_notes": [
            "BPE uses same fat_frac mix as word (was uniform — STORY confound).",
            "Matched EV/FIN hold sizes word=BPE.",
            "ckpt = best on ev_story; also report story_all_last for optimism check.",
            "149 G not reused (149 BPE lacked fat lock).",
            f"149_ref_word_story={(up149.get('word_fp') or {}).get('story_all')}",
            f"149_ref_bpe_story={(up149.get('ws_bpe_rand') or {}).get('story_all')}",
        ],
        "ref100": REF100,
        "verdict": "DUAL_TRUNK_TABLE",
    }
    write_dec(dec, out)
    (RES / "stage150_DUAL_TRUNK_TABLE.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(
        f"[150] G gap={100*(table['G']['unit_gap_bpe_minus_word'] or 0):+.1f}pp "
        f"S gap={100*(table['S']['unit_gap_bpe_minus_word'] or 0):+.1f}pp "
        f"S+ gap={100*(table['Splus']['unit_gap_bpe_minus_word'] or 0):+.1f}pp | "
        f"headroom_d={100*headroom_delta:+.1f}pp primary={primary}"
    )
    return out


# -------------------- Stage 151 --------------------
def run_151(device, cfg, stack, phrases) -> dict:
    dec = RES / "stage151_one_factor_decision.json"
    prev = done(dec)
    if prev:
        return prev
    log("\n########## STAGE 151 orthogonal 1-factor @G and @S+ ##########")
    results = {}

    # Base locks (word only)
    bases = {
        "G": dict(TRUNKS["G"]),
        "Splus": dict(TRUNKS["Splus"]),
    }
    # Train baselines for comparison (may overlap 150 — retrain for clean curve under same seed)
    for bname, base in bases.items():
        tag = f"151_{bname}_base"
        results[tag] = train_word(
            tag, base, phrases, device, cfg, stack, seed=stable_seed(151, bname, "base")
        )

    factors = []
    # G factors
    g = dict(TRUNKS["G"])
    factors.append(("G", "heads8", {**g, "n_head": 8}))
    factors.append(("G", "layers2", {**g, "n_layer": 2}))
    factors.append(("G", "batch8", {**g, "batch": 8}))
    factors.append(("G", "lr1e3", {**g, "lr": 1e-3}))
    factors.append(("G", "Adam", {**g, "opt": "Adam", "wd": 0.0}))
    # S+ factors
    s = dict(TRUNKS["Splus"])
    factors.append(("Splus", "heads4", {**s, "n_head": 4}))  # down control
    factors.append(("Splus", "layers2", {**s, "n_layer": 2}))
    factors.append(("Splus", "batch16", {**s, "batch": 16}))
    factors.append(("Splus", "lr3e4", {**s, "lr": 3e-4}))
    factors.append(("Splus", "AdamW", {**s, "opt": "AdamW", "wd": 0.01}))

    for bname, fname, tr in factors:
        tag = f"151_{bname}_{fname}"
        results[tag] = train_word(
            tag, tr, phrases, device, cfg, stack, seed=stable_seed(151, bname, fname)
        )

    deltas = {}
    for bname in ("G", "Splus"):
        base_key = f"151_{bname}_base"
        base_s = results[base_key]["story_all"]
        deltas[bname] = {}
        for k, v in results.items():
            if k.startswith(f"151_{bname}_") and k != base_key:
                deltas[bname][k] = v["story_all"] - base_s

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "orthogonal_one_factor_word",
        "arms": {k: slim(v) for k, v in results.items()},
        "deltas_story_vs_base": deltas,
        "verdict": "ONE_FACTOR_TABLE",
        "note": "Each row = exactly one axis vs locked base; dim not re-litigated here.",
    }
    write_dec(dec, out)
    log(f"[151] wrote one-factor table; G deltas={ {k: round(100*x,1) for k,x in deltas['G'].items()} }")
    return out


# -------------------- Stage 152 --------------------
def run_152(device, cfg, stack, phrases, tok) -> dict:
    dec = RES / "stage152_context_budget_decision.json"
    prev = done(dec)
    if prev:
        return prev
    log("\n########## STAGE 152 context/budget on S+ ##########")
    tr = dict(TRUNKS["Splus"])
    arms = {}

    # residual baseline already in 150; retrain thin matched variants
    arms["152a_word"] = train_word("152a_word", tr, phrases, device, cfg, stack, seed=1521)
    # BPE truncated to same content budget as word slots (~16).
    # n_positions=32 is only pos-table headroom (>= train/eval seq); NOT a longer window.
    # Density hypothesis: equal #CE tokens, not equal n_positions.
    arms["152a_bpe_maxlen16"] = train_bpe(
        "152a_bpe_maxlen16",
        tr,
        phrases,
        tok,
        device,
        cfg,
        seed=1522,
        bpe_max_len=16,
        n_positions=32,
    )

    # 152b: token-budget — truncate BPE to mean word length scale; log tpw
    sample = phrases[:2000]
    nw = sum(len(s.split()) for s in sample)
    nb = sum(len(tok.encode(s).ids) for s in sample)
    tpw = nb / max(nw, 1)
    budget = 16
    arms["152b_bpe_budget16"] = train_bpe(
        "152b_bpe_budget16",
        tr,
        phrases,
        tok,
        device,
        cfg,
        seed=1523,
        bpe_max_len=budget,
        n_positions=budget + 8,
    )
    arms["152b_word"] = arms["152a_word"]  # same word arm

    # 152c fat_frac=0.45 (one change)
    tr_fat = {**tr, "fat_frac": 0.45}
    arms["152c_word_fat045"] = train_word(
        "152c_word_fat045", tr_fat, phrases, device, cfg, stack, seed=1524
    )
    arms["152c_bpe_fat045"] = train_bpe(
        "152c_bpe_fat045", tr_fat, phrases, tok, device, cfg, seed=1525
    )

    gap_a = arms["152a_bpe_maxlen16"]["story_all"] - arms["152a_word"]["story_all"]
    gap_b = arms["152b_bpe_budget16"]["story_all"] - arms["152b_word"]["story_all"]
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "context_budget_fat_Splus",
        "tpw_sample": tpw,
        "arms": {k: slim(v) for k, v in arms.items()},
        "gaps_bpe_minus_word_story": {"152a_maxlen16": gap_a, "152b_budget16": gap_b},
        "verdict": "CONTEXT_BUDGET_TABLE",
        "note": "Claim on word HOLD only; piece-greedy diagnostic inside eval_bpe_word_holds.",
    }
    write_dec(dec, out)
    log(f"[152] tpw={tpw:.3f} gap_a={100*gap_a:+.1f}pp gap_b={100*gap_b:+.1f}pp")
    return out


# -------------------- Stage 153 --------------------
def run_153(device, cfg, stack) -> dict:
    dec = RES / "stage153_scale_batch_decision.json"
    prev = done(dec)
    if prev:
        return prev
    log("\n########## STAGE 153 scale x batch (word only) ##########")
    arms = {}
    grid = []
    for tname in ("G", "Splus"):
        for n_ph in (20_000, 100_000):
            for batch in (8, 16):
                grid.append((tname, n_ph, batch))

    for tname, n_ph, batch in grid:
        tr = {**TRUNKS[tname], "batch": batch}
        phrases = load_phrases(n_ph)
        steps = FT_STEPS_SMALL_N if n_ph <= 20_000 else FT_STEPS
        tag = f"153_{tname}_N{n_ph}_B{batch}"
        arms[tag] = train_word(
            tag, tr, phrases, device, cfg, stack, steps=steps, seed=1530 + n_ph // 1000 + batch
        )
        arms[tag]["n_phrases"] = len(phrases)

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "scale_x_batch_word",
        "arms": {k: slim(v) for k, v in arms.items()},
        "verdict": "SCALE_BATCH_TABLE",
        "note": "Thesis: batch8 may be small-N/fat specific; may interact with S+ capacity.",
    }
    write_dec(dec, out)
    log("[153] scale x batch table done")
    return out


# -------------------- Stage 154 --------------------
def run_154(device, cfg, stack, tok) -> dict:
    dec = RES / "stage154_capacity_data_decision.json"
    prev = done(dec)
    if prev:
        return prev
    log("\n########## STAGE 154 capacity x data (S+ ritual) ##########")
    arms = {}
    # S+ ritual = opt/batch/lr from Splus; capacity from CAP_PAIRS
    ritual = {k: TRUNKS["Splus"][k] for k in ("batch", "lr", "opt", "wd", "fat_frac", "word_max_len", "bpe_max_len", "n_positions")}

    for n_ph, nlab in ((100_000, "100k"), (460_000, "460k")):
        phrases = load_phrases(n_ph)
        # Retrain BPE tok on the same N (100k-tok on 460k story = UNK/STORY confound).
        tok_n = train_or_load_bpe(
            phrases, cache=RES / f"stage154_bpe_tokenizer_{nlab}.json"
        )
        for pname, cap in CAP_PAIRS.items():
            tr = {**ritual, **cap}
            wtag = f"154_{nlab}_{pname}_word"
            btag = f"154_{nlab}_{pname}_bpe"
            arms[wtag] = train_word(
                wtag, tr, phrases, device, cfg, stack, seed=stable_seed(154, nlab, pname, "word")
            )
            arms[wtag]["n_phrases"] = len(phrases)
            arms[btag] = train_bpe(
                btag, tr, phrases, tok_n, device, cfg, seed=stable_seed(154, nlab, pname, "bpe")
            )
            arms[btag]["n_phrases"] = len(phrases)

    def story(tag):
        return (arms.get(tag) or {}).get("story_all")

    interp = {
        "word_headroom_lift_100k": (story("154_100k_headroom_word") or 0) - (story("154_100k_hist_word") or 0),
        "word_headroom_lift_460k": (story("154_460k_headroom_word") or 0) - (story("154_460k_hist_word") or 0),
        "bpe_headroom_lift_100k": (story("154_100k_headroom_bpe") or 0) - (story("154_100k_hist_bpe") or 0),
        "bpe_headroom_lift_460k": (story("154_460k_headroom_bpe") or 0) - (story("154_460k_hist_bpe") or 0),
    }
    # Premature-close test
    if interp["word_headroom_lift_100k"] < 0.01 and interp["word_headroom_lift_460k"] >= 0.02:
        premature = True
        note = "Flat@100k but lift@460k => old dim close was premature; keep S+."
    elif interp["word_headroom_lift_100k"] < 0.01 and interp["word_headroom_lift_460k"] < 0.01:
        premature = False
        note = "Flat at both N under this V/window; capacity not the bottleneck (may thin S+)."
    else:
        premature = None
        note = "Mixed/positive@100k; see lifts; do not close dim from a single slice."

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "capacity_x_data_Splus_ritual",
        "cap_pairs": CAP_PAIRS,
        "arms": {k: slim(v) for k, v in arms.items()},
        "lifts_story": interp,
        "premature_close_evidence": premature,
        "verdict": "CAPACITY_DATA_TABLE",
        "note": note,
    }
    write_dec(dec, out)
    log(
        f"[154] word lift 100k={100*interp['word_headroom_lift_100k']:+.1f}pp "
        f"460k={100*interp['word_headroom_lift_460k']:+.1f}pp | {note}"
    )
    return out


# -------------------- Stage 155 --------------------
def run_155(d150: dict, d154: dict) -> dict:
    dec = RES / "stage155_eval_surface_decision.json"
    prev = done(dec)
    if prev:
        return prev
    log("\n########## STAGE 155 eval surface (no new train) ##########")
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "eval_surface_note",
        "verdict": "EVAL_NOTE",
        "note": (
            "Word arms: word-argmax exact@1. BPE arms: greedy piece->word HOLD (same as 112/149). "
            "No oracle word-boundary CE implemented; if unit gap collapses under 152 matched "
            "context but remains under residual 48-len BPE, residual is decode/context not atom."
        ),
        "pointers": {
            "stage150": "stage150_dual_trunk_decision.json",
            "stage152": "stage152_context_budget_decision.json",
            "stage154": "stage154_capacity_data_decision.json",
            "primary_trunk": (d150 or {}).get("default_primary_trunk"),
            "capacity_note": (d154 or {}).get("note"),
        },
    }
    write_dec(dec, out)
    return out


def append_replay(summary: str):
    replay = RES / "sote_v2_path_replay.md"
    if not replay.exists():
        return
    txt = replay.read_text(encoding="utf-8")
    if "Stage 150-155 clean compare" in txt:
        return
    block = f"\n**Stage 150-155 clean compare pipeline:** {summary} See `plan_150_plus_clean_compare.md` + `stage150_155_clean_compare_pipeline_decision.json`.\n"
    if "**F85 dual-channel FREEZE:**" in txt:
        txt = txt.replace("**F85 dual-channel FREEZE:**", block + "\n**F85 dual-channel FREEZE:**")
    else:
        txt = txt.rstrip() + "\n" + block
    replay.write_text(txt, encoding="utf-8")


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"150-155 clean compare pipeline start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan={PLAN if PLAN.exists() else 'MISSING'}")
    try:
        up149 = wait_json(DEC149, "149")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg, stack = setup_stack(device)
        phrases100 = load_phrases(100_000)
        tok = train_or_load_bpe(phrases100)

        d150 = run_150(device, cfg, stack, phrases100, tok, up149)
        d151 = run_151(device, cfg, stack, phrases100)
        d152 = run_152(device, cfg, stack, phrases100, tok)
        d153 = run_153(device, cfg, stack)
        d154 = run_154(device, cfg, stack, tok)
        d155 = run_155(d150, d154)

        pipe = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "clean_compare_150_155",
            "plan": str(PLAN),
            "upstream_149_verdict": up149.get("verdict"),
            "stages": {
                "150": d150.get("verdict"),
                "151": d151.get("verdict"),
                "152": d152.get("verdict"),
                "153": d153.get("verdict"),
                "154": d154.get("verdict"),
                "155": d155.get("verdict"),
            },
            "default_primary_trunk": d150.get("default_primary_trunk"),
            "headroom_delta_100k": d150.get("headroom_Splus_minus_S_word_story"),
            "capacity_premature_close": d154.get("premature_close_evidence"),
            "verdict": "PIPELINE_DONE",
            "decisions": [
                "stage150_dual_trunk_decision.json",
                "stage151_one_factor_decision.json",
                "stage152_context_budget_decision.json",
                "stage153_scale_batch_decision.json",
                "stage154_capacity_data_decision.json",
                "stage155_eval_surface_decision.json",
            ],
        }
        write_dec(PIPELINE_DEC, pipe)
        append_replay(
            f"primary={d150.get('default_primary_trunk')} "
            f"headroom@100k={100*(d150.get('headroom_Splus_minus_S_word_story') or 0):+.1f}pp "
            f"premature_close={d154.get('premature_close_evidence')}"
        )
        log("DONE 150-155 pipeline")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
