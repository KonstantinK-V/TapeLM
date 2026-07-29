"""
Stage 140 — pack context by BPE-token budget; inside SOTE keep WORD atoms.

User variant: "cut like BPE by letter/token count — e.g. BPE 32 tokens may
span ~60 words — but read them as unified word tokens."

Protocol (after 139; hops OUT):
  1) Train/reuse corpus BPE (V=8k) on TS-100k phrases.
  2) Rebuild windows by *BPE id budget* B=32 (not max_words=8):
       take consecutive words until bpe_len(words) >= B; emit that word span.
  3) Arm A word: WordIdTransformer on these longer word sequences
       (each word = one atom / letter-fp init).
  4) Arm B BPE: GPT2-mini on same spans tokenized to <=B(+bos/eos) ids.
  5) Eval word-level exact@1 on hold spans (greedy BPE→word like 112).

Also logs space/punct contract vs BPE (see decision.note).

Gate: report gap BPE_ALL − word_ALL on HOLD; compare to Stage112 (~+14pp)
and Stage136 wiki gap. Not a PASS on absolute STORY — fairness dig.

Run:
  python _stage140_bpe_budget_word_atoms.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import traceback
from collections import defaultdict
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
    RELS,
    RES,
    Config,
    WordIdTransformer,
    _role,
    _warmup_then_constant,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import ensure_100k, _subsample  # noqa: E402
from _stage111_112_follow import _encode_words, eval_bpe_word_holds  # noqa: E402

DEC139 = RES / "stage139_bpe_tail_cut_words_decision.json"
LOG = RES / "_stage140_log.txt"
DEC = RES / "stage140_bpe_budget_word_atoms_decision.json"
BPE_TOK = RES / "stage140_budget_bpe_tokenizer.json"
CORPUS_BUDGET = ROOT / "data" / "external_tinystories_bpebudget32_85.txt"

BPE_BUDGET = 32
FT_STEPS = 40_000
EVAL_EVERY_WORD = 2000
EVAL_EVERY_BPE = 5000
BATCH = 8
MAX_WORDS_CAP = 64  # safety


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_139(timeout_s=20 * 3600, poll=60):
    log(f"[wait] for {DEC139.name} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC139.exists():
            d = json.loads(DEC139.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 139 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("139 not ready")


SPACE_PUNCT_CONTRACT = {
    "sote_filter": (
        "lowercase; keep a-z 0-9 space and .!?; other punct -> space; "
        "split sentences on .!?/newline then DROP the punct chars; "
        "word boundary = whitespace; no punct tokens in vocab."
    ),
    "bpe_here": (
        "Whitespace pre_tokenizer: splits on spaces; spaces are NOT model tokens; "
        "BPE merges only inside whitespace-separated pieces; "
        "no dedicated '.' ',' tokens in this pipeline (punct already stripped upstream)."
    ),
    "vs_gpt_tiktoken": (
        "Real GPT BPE often keeps punct as separate tokens and may attach spaces "
        "via byte-level rules (Ġ). Our mini-BPE is whitespace-BPE on already-cleaned "
        "SOTE text — fair within contract, not identical to tiktoken."
    ),
}


def train_or_load_bpe(phrases: list[str], vocab_size: int = 8000) -> Tokenizer:
    if BPE_TOK.exists():
        log(f"[bpe] reuse {BPE_TOK.name}")
        return Tokenizer.from_file(str(BPE_TOK))
    log(f"[bpe] train vocab={vocab_size} on {len(phrases)} phrases")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=False,
    )
    tok.train_from_iterator(phrases, trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tok.token_to_id("[BOS]")), ("[EOS]", tok.token_to_id("[EOS]"))],
    )
    tok.save(str(BPE_TOK))
    return tok


def bpe_n_tokens(tok: Tokenizer, words: list[str]) -> int:
    """Count BPE ids for word span WITHOUT bos/eos specials when possible."""
    text = " ".join(words)
    # encode without post-processor if available
    try:
        enc = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(text).ids
        bos, eos = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]")
        ids = [i for i in ids if i not in (bos, eos)]
        return len(ids)
    return len(enc.ids)


def pack_by_bpe_budget(phrases: list[str], tok: Tokenizer, budget: int, seed: int = 140) -> list[str]:
    """
    Slide over phrase words; emit spans whose BPE length first reaches `budget`
    (or max words cap). Each emitted line = space-joined words (SOTE atoms).
    """
    rr = random.Random(seed)
    out = []
    seen = set()
    # shuffle source phrases then pack
    ph = list(phrases)
    rr.shuffle(ph)
    for line in ph:
        ws = line.split()
        if len(ws) < 2:
            continue
        i = 0
        while i < len(ws):
            j = i + 1
            while j <= len(ws) and (j - i) <= MAX_WORDS_CAP:
                n = bpe_n_tokens(tok, ws[i:j])
                if n >= budget or j == len(ws) or (j - i) == MAX_WORDS_CAP:
                    span = ws[i:j]
                    if len(span) >= 2:
                        s = " ".join(span)
                        if s not in seen:
                            seen.add(s)
                            out.append(s)
                    i = j
                    break
                j += 1
            else:
                break
        if len(out) >= 120_000:
            break
    return out


def load_phrases(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def ensure_budget_corpus(base_phrases: list[str], tok: Tokenizer) -> list[str]:
    if CORPUS_BUDGET.exists():
        ph = load_phrases(CORPUS_BUDGET)
        if len(ph) >= 20_000:
            log(f"[data] reuse {CORPUS_BUDGET.name} n={len(ph)}")
            return ph
    log(f"[data] pack windows by BPE budget={BPE_BUDGET} ...")
    packed = pack_by_bpe_budget(base_phrases, tok, BPE_BUDGET)
    # stats
    rr = random.Random(140)
    sample = packed if len(packed) <= 2000 else [packed[i] for i in rr.sample(range(len(packed)), 2000)]
    n_words = [len(s.split()) for s in sample]
    n_bpe = [bpe_n_tokens(tok, s.split()) for s in sample]
    meta = {
        "budget_bpe": BPE_BUDGET,
        "n_out": len(packed),
        "mean_words": sum(n_words) / max(len(n_words), 1),
        "mean_bpe": sum(n_bpe) / max(len(n_bpe), 1),
        "p50_words": sorted(n_words)[len(n_words) // 2] if n_words else 0,
    }
    header = [
        f"# SOTE windows packed to ~{BPE_BUDGET} BPE tokens; atoms=words",
        f"# meta: {json.dumps(meta)}",
    ]
    CORPUS_BUDGET.write_text("\n".join(header + packed) + "\n", encoding="utf-8")
    log(f"[data] wrote n={len(packed)} mean_words={meta['mean_words']:.1f} mean_bpe={meta['mean_bpe']:.1f}")
    return packed


def train_word(phrases, device, cfg):
    log("\n======== A word atoms on BPE-budget windows ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=140)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    parent = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent.exists():
        parent = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent)
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    # longer contexts — max_len up to 64
    max_len = min(64, max(16, max(len(ln["words"]) for ln in train[:500]) + 2))
    model = WordIdTransformer(len(words), 256, 4, 2, max_len, 0.1).to(device)
    model.init_from_fps(fps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[word] V={len(words)} max_len={max_len} params={n_params/1e6:.2f}M")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 60000, 20), stoi)
    if not fat_p:
        fat_p = story_p

    ev_seen = _subsample(hold_seen, 400, 1401)
    ev_story = _subsample(hold_story, 300, 1403)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), 2401)
    fin_story = _subsample(hold_story, min(len(hold_story), 500), 2403)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(140)
    best = {"story_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    curve = []

    def snap(step):
        model.eval()
        s = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        st = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
        obj, rel = s["obj"]["hit1"], s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(f"  [word] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | HOLD ALL={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(story_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.train()

    model.train()
    snap(0)
    for step in range(1, FT_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        n_fat = max(1, int(round(BATCH * 0.75)))
        batch = [rr.choice(fat_p) for _ in range(n_fat)] + [rr.choice(story_p) for _ in range(BATCH - n_fat)]
        packed = collate_word_id_batch(batch, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY_WORD == 0 or step == FT_STEPS:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    f_story = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    ck = CKPT / "stage140_budget_word.pt"
    torch.save({"word_tf": best["state"], "surfaces": words, "max_len": max_len}, ck)
    return {
        "arm": "word",
        "V": len(words),
        "params": n_params,
        "max_len": max_len,
        "seen_obj": f_seen["obj"]["hit1"],
        "seen_rel": f_seen["roles"].get("rel", {}).get("hit1", 0.0),
        "hold_all": f_story["roles"].get("ALL", {}).get("hit1", 0.0),
        "hold_obj": f_story["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
        "mix": {k: meta[k] for k in meta if k != "top_triple_freq"},
    }


def train_bpe(phrases, tok, device, cfg):
    log("\n======== B BPE on same BPE-budget windows ========")
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=140)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    seqs = []
    for ln in train:
        ids = _encode_words(tok, ln["words"], max_len=BPE_BUDGET + 4, bos=bos, eos=eos, pad=pad)
        if len(ids) >= 3:
            seqs.append(ids)
    log(f"[bpe] seqs={len(seqs)} V={V}")
    conf = GPT2Config(
        vocab_size=V,
        n_positions=max(64, BPE_BUDGET + 8),
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_inner=1024,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    model = GPT2LMHeadModel(conf).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rr = random.Random(140)
    ev_seen = _subsample(hold_seen, 200, 1401)
    ev_story = _subsample(hold_story, 150, 1403)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 400), 2401)
    fin_story = _subsample(hold_story, min(len(hold_story), 300), 2403)
    best = {"hold_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    curve = []

    def snap(step):
        model.eval()
        s = eval_bpe_word_holds(model, tok, ev_seen, device, max_n_lines=len(ev_seen))
        st = eval_bpe_word_holds(model, tok, ev_story, device, max_n_lines=len(ev_story))
        obj = s["obj"]["hit1"]
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "story_all": sall})
        log(f"  [bpe] step {step}: SEEN obj={100*obj:.1f}% | HOLD ALL={100*sall:.1f}%")
        if (sall, obj) >= (best["hold_all"], best["obj"]):
            best.update(hold_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.train()

    model.train()
    snap(0)
    for step in range(1, FT_STEPS + 1):
        batch_seqs = [rr.choice(seqs) for _ in range(BATCH)]
        maxlen = max(len(s) for s in batch_seqs)
        x = torch.full((BATCH, maxlen), pad, dtype=torch.long, device=device)
        for i, s in enumerate(batch_seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        labels = x.clone()
        labels[labels == pad] = -100
        loss = model(x, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY_BPE == 0 or step == FT_STEPS:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = eval_bpe_word_holds(model, tok, fin_seen, device, max_n_lines=len(fin_seen))
    f_story = eval_bpe_word_holds(model, tok, fin_story, device, max_n_lines=len(fin_story))
    ck = CKPT / "stage140_budget_bpe.pt"
    torch.save({"gpt2": best["state"], "tokenizer": str(BPE_TOK)}, ck)
    return {
        "arm": "bpe",
        "V": V,
        "params": n_params,
        "seen_obj": f_seen["obj"]["hit1"],
        "hold_all": f_story["roles"].get("ALL", {}).get("hit1", 0.0),
        "hold_obj": f_story["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"140 BPE-budget windows / word atoms start {datetime.now(timezone.utc).isoformat()}")
    log(f"Pack to ~{BPE_BUDGET} BPE tokens; SOTE reads words as unified atoms.")
    try:
        upstream = wait_139()
        base = ensure_100k()
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tok = train_or_load_bpe(base)
        phrases = ensure_budget_corpus(base, tok)
        # density diag on packed
        rr = random.Random(140)
        sample = phrases if len(phrases) <= 1500 else [phrases[i] for i in rr.sample(range(len(phrases)), 1500)]
        mw = sum(len(s.split()) for s in sample) / len(sample)
        mb = sum(bpe_n_tokens(tok, s.split()) for s in sample) / len(sample)
        log(f"[diag] packed mean_words={mw:.1f} mean_bpe={mb:.1f} (target budget={BPE_BUDGET})")

        word = train_word(phrases, device, cfg)
        bpe = train_bpe(phrases, tok, device, cfg)
        gap = bpe["hold_all"] - word["hold_all"]
        if gap >= 0.25:
            verdict = "GAP_LARGER"
        elif gap >= 0.10:
            verdict = "GAP_SIMILAR"
        else:
            verdict = "GAP_SMALLER"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "bpe_budget_pack_word_atoms_inside",
            "upstream_139": upstream.get("verdict"),
            "bpe_budget": BPE_BUDGET,
            "diag": {"mean_words": mw, "mean_bpe": mb},
            "space_punct_contract": SPACE_PUNCT_CONTRACT,
            "word": word,
            "bpe": bpe,
            "gap_hold_all_pp": gap,
            "verdict": verdict,
            "note": (
                "Same character/BPE-token budget for context packing; SOTE still one-id-per-word. "
                "Punct/spaces: see space_punct_contract (SOTE strips punct; mini-BPE is whitespace-BPE)."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage140_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(
            f"[140] {verdict} word_HOLD={100*word['hold_all']:.1f}% "
            f"bpe_HOLD={100*bpe['hold_all']:.1f}% gap={100*gap:+.1f}pp"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 140 BPE-budget pack / word atoms:** {verdict} "
                f"word={100*word['hold_all']:.1f}% bpe={100*bpe['hold_all']:.1f}% "
                f"gap={100*gap:+.1f}pp (budget={BPE_BUDGET}). "
                f"`stage140_bpe_budget_word_atoms_decision.json`.\n"
            )
            if "Stage 140 BPE-budget" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 140")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
