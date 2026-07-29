"""
Stage 136 — word-atom vs BPE on a DIFFERENT large corpus (WikiText-2).

Motivation: TinyStories BPE gap ~+14–18pp STORY may understate the unit effect
(kids text is morph-simple; C' head≈1.0 tpw). WikiText is harder / denser BPE.

Protocol (matched width; hops OUT):
  1) Build SOTE-charset WikiText-2 windows (~200k, max 8 words, max_word_len24)
  2) Diagnostic: mean BPE toks/word vs TinyStories-100k
  3) A word: WordIdTransformer d=256, 4L/4H, SOTE fp-init, recipe98 CE (no hops)
  4) B BPE: corpus BPE V=8k + GPT2 mini n_embd=256, 4L/4H (same as Stage112 width/depth)
  5) Gate = word-level exact@1 on natural HOLD (ALL + obj)

Verdict:
  GAP_LARGER  if (BPE_ALL − word_ALL) ≥ 0.25  (+25pp)
  GAP_SIMILAR if gap in [0.10, 0.25)
  GAP_SMALLER if gap < 0.10
  (also report vs Stage112 gap ~0.14)

Run:
  python _stage136_wikitext_word_vs_bpe.py
"""
from __future__ import annotations

import json
import random
import re
import sys
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
    CHAR2ID,
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
    filter_tinystories_chunk,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import _subsample  # noqa: E402
from _stage111_112_follow import (  # noqa: E402
    _encode_words,
    eval_bpe_word_holds,
    train_bpe_tokenizer as _train_bpe_tok_112,
)

LOG = RES / "_stage136_log.txt"
DEC = RES / "stage136_wikitext_word_vs_bpe_decision.json"
RAW_WIKI = ROOT / "data" / "_wikitext2_raw.txt"
CORPUS_WIKI = ROOT / "data" / "external_wikitext2_200k_85.txt"
CORPUS_TS = ROOT / "data" / "external_tinystories_100k_85.txt"
BPE_TOK = RES / "stage136_wikitext_bpe_tokenizer.json"

N_PHRASES = 200_000
FT_STEPS = 40_000
EVAL_EVERY_WORD = 2000
EVAL_EVERY_BPE = 5000
BATCH = 8
REF_GAP_112 = 0.1435  # Stage112 story_lift approx


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def load_phrases(path: Path) -> list[str]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def ensure_wikitext_raw() -> Path:
    if RAW_WIKI.exists() and RAW_WIKI.stat().st_size > 100_000:
        log(f"[data] reuse raw {RAW_WIKI.name} ({RAW_WIKI.stat().st_size} bytes)")
        return RAW_WIKI
    # Prefer plain files (HF `datasets` crashed on this machine with access violation).
    import urllib.request

    log("[data] download WikiText-2 (pytorch/examples word_language_model) ...")
    base = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"
    parts = []
    for name in ("train.txt", "valid.txt", "test.txt"):
        req = urllib.request.Request(base + name, headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=180).read().decode("utf-8", errors="ignore")
        parts.append(txt)
        log(f"  fetched {name} chars={len(txt)}")
    text = "\n".join(parts).replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    RAW_WIKI.parent.mkdir(exist_ok=True)
    RAW_WIKI.write_text(text + "\n", encoding="utf-8")
    log(f"[data] wrote {RAW_WIKI} chars={RAW_WIKI.stat().st_size}")
    return RAW_WIKI


def ensure_wikitext_corpus(cfg: Config) -> list[str]:
    if CORPUS_WIKI.exists():
        ph = load_phrases(CORPUS_WIKI)
        if len(ph) >= N_PHRASES * 0.8:
            log(f"[data] reuse {CORPUS_WIKI.name} n={len(ph)}")
            return ph
    raw = ensure_wikitext_raw()
    log(f"[data] filter WikiText -> {CORPUS_WIKI.name} max_lines={N_PHRASES}")
    phrases, meta = filter_tinystories_chunk(
        raw,
        CORPUS_WIKI,
        max_lines=N_PHRASES,
        min_words=3,
        max_words=8,
        seed=136,
        max_word_len=int(cfg.max_word_len),
    )
    # rewrite header to say WikiText
    body = [ln for ln in CORPUS_WIKI.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    meta = dict(meta)
    meta["corpus"] = "wikitext-2-raw-v1"
    header = [
        "# SOTE WikiText-2 (a-z + digits + space; Stage85+ charset)",
        f"# meta: {json.dumps(meta)}",
    ]
    CORPUS_WIKI.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    log(f"[data] wiki phrases={len(phrases)} meta={meta}")
    return phrases


def mean_bpe_tpw(phrases: list[str], vocab_size: int = 8000, sample_n: int = 5000, seed: int = 136) -> dict:
    """Train a throwaway BPE on sample; mean tokens per whitespace word."""
    rng = random.Random(seed)
    sample = phrases if len(phrases) <= sample_n else [phrases[i] for i in rng.sample(range(len(phrases)), sample_n)]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=False,
    )
    tok.train_from_iterator(sample, trainer=trainer)
    n_tok = 0
    n_word = 0
    for ph in sample:
        ws = ph.split()
        if not ws:
            continue
        ids = tok.encode(ph).ids
        n_tok += len(ids)
        n_word += len(ws)
    return {
        "mean_tpw": n_tok / max(n_word, 1),
        "n_word": n_word,
        "n_tok": n_tok,
        "sample_n": len(sample),
        "V": tok.get_vocab_size(),
    }


def train_bpe_tokenizer_wiki(phrases: list[str], vocab_size: int = 8000) -> Tokenizer:
    if BPE_TOK.exists():
        log(f"[bpe] reuse tokenizer {BPE_TOK}")
        return Tokenizer.from_file(str(BPE_TOK))
    log(f"[bpe] train BPE vocab={vocab_size} on {len(phrases)} wiki phrases")
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
    BPE_TOK.parent.mkdir(exist_ok=True)
    tok.save(str(BPE_TOK))
    return tok


def train_word_arm(phrases: list[str], device, cfg: Config) -> dict:
    log("\n======== A word-atom (CE only, 4L/256, SOTE fp-init) ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=136)
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

    # match BPE depth (4L) + width 256
    model = WordIdTransformer(len(words), 256, 4, 4, 16, 0.1).to(device)
    model.init_from_fps(fps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[word] V={len(words)} params={n_params/1e6:.2f}M mix={ {k: meta[k] for k in meta if k != 'top_triple_freq'} }")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 80000, 12), stoi)
    if not fat_p:
        fat_p = story_p

    ev_seen = _subsample(hold_seen, 400, 1361)
    ev_story = _subsample(hold_story, 300, 1363)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), 2361)
    fin_story = _subsample(hold_story, min(len(hold_story), 600), 2363)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(136)
    fat_frac = 0.75
    best = {
        "obj": -1.0,
        "story_all": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "seen": None,
        "story": None,
    }
    curve = []

    def snap(step):
        model.eval()
        s = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        st = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
        obj = s["obj"]["hit1"]
        rel = s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(f"  [word] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | HOLD ALL={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(
                obj=obj,
                story_all=sall,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                seen=s,
                story=st,
            )
        model.train()

    model.train()
    snap(0)
    for step in range(1, FT_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        n_fat = max(1, int(round(BATCH * fat_frac)))
        batch_ex = [rr.choice(fat_p) for _ in range(n_fat)]
        batch_ex += [rr.choice(story_p) for _ in range(BATCH - n_fat)]
        packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        logits = model.logits_last_from_batch(ids, mask)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY_WORD == 0 or step == FT_STEPS:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    f_story = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    ck = CKPT / "stage136_wikitext_word.pt"
    torch.save({"word_tf": best["state"], "surfaces": words, "n_layers": 4}, ck)
    return {
        "arm": "word",
        "V": len(words),
        "params": n_params,
        "n_layers": 4,
        "d_model": 256,
        "seen_obj": f_seen["obj"]["hit1"],
        "seen_rel": f_seen["roles"].get("rel", {}).get("hit1", 0.0),
        "seen_all": f_seen["roles"].get("ALL", {}).get("hit1", 0.0),
        "hold_all": f_story["roles"].get("ALL", {}).get("hit1", 0.0),
        "hold_obj": f_story["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
        "best_step_proxy": {"obj": best["obj"], "hold_all": best["story_all"]},
    }


def train_bpe_arm(phrases: list[str], device, cfg: Config) -> dict:
    log("\n======== B BPE+GPT2 mini (4L/256) ========")
    tok = train_bpe_tokenizer_wiki(phrases, vocab_size=8000)
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()

    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=136)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story = [ln for ln in train if ln.get("bucket") != "fat_train"]
    seqs = []
    for ln in fat + story:
        ids = _encode_words(tok, ln["words"], max_len=48, bos=bos, eos=eos, pad=pad)
        if len(ids) >= 3:
            seqs.append(ids)
    log(f"[bpe] seqs={len(seqs)} V={V}")

    conf = GPT2Config(
        vocab_size=V,
        n_positions=64,
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
    log(f"[bpe] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rr = random.Random(136)
    ev_seen = _subsample(hold_seen, 200, 1361)
    ev_story = _subsample(hold_story, 150, 1363)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 400), 2361)
    fin_story = _subsample(hold_story, min(len(hold_story), 300), 2363)

    best = {
        "obj": -1.0,
        "hold_all": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        s = eval_bpe_word_holds(model, tok, ev_seen, device, max_n_lines=len(ev_seen))
        st = eval_bpe_word_holds(model, tok, ev_story, device, max_n_lines=len(ev_story))
        obj = s["obj"]["hit1"]
        rel = s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(f"  [bpe] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | HOLD ALL={100*sall:.1f}%")
        if (sall, obj) >= (best["hold_all"], best["obj"]):
            best.update(
                obj=obj,
                hold_all=sall,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
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
    ck = CKPT / "stage136_wikitext_bpe.pt"
    torch.save({"gpt2": best["state"], "tokenizer": str(BPE_TOK)}, ck)
    return {
        "arm": "bpe",
        "V": V,
        "params": n_params,
        "n_layers": 4,
        "d_model": 256,
        "seen_obj": f_seen["obj"]["hit1"],
        "seen_rel": f_seen["roles"].get("rel", {}).get("hit1", 0.0),
        "seen_all": f_seen["roles"].get("ALL", {}).get("hit1", 0.0),
        "hold_all": f_story["roles"].get("ALL", {}).get("hit1", 0.0),
        "hold_obj": f_story["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
        "best_step_proxy": {"obj": best["obj"], "hold_all": best["hold_all"]},
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"136 WikiText word vs BPE start {datetime.now(timezone.utc).isoformat()}")
    log("Corpus: WikiText-2 (different from TinyStories). Hops OUT. Matched d=256, 4L.")
    try:
        cfg = Config()
        # Stage85+ word length
        if not hasattr(cfg, "max_word_len") or int(cfg.max_word_len) < 24:
            cfg.max_word_len = 24
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[device] {device}")

        phrases = ensure_wikitext_corpus(cfg)
        # density diagnostic vs TinyStories
        log("[diag] BPE toks/word on Wiki vs TinyStories-100k ...")
        diag_wiki = mean_bpe_tpw(phrases)
        diag_ts = None
        if CORPUS_TS.exists():
            diag_ts = mean_bpe_tpw(load_phrases(CORPUS_TS))
        log(f"  wiki mean_tpw={diag_wiki['mean_tpw']:.3f}")
        if diag_ts:
            log(f"  tinystories mean_tpw={diag_ts['mean_tpw']:.3f}")

        word = train_word_arm(phrases, device, cfg)
        bpe = train_bpe_arm(phrases, device, cfg)

        gap_all = bpe["hold_all"] - word["hold_all"]
        gap_obj = bpe["hold_obj"] - word["hold_obj"]
        if gap_all >= 0.25:
            verdict = "GAP_LARGER"
        elif gap_all >= 0.10:
            verdict = "GAP_SIMILAR"
        else:
            verdict = "GAP_SMALLER"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "wikitext2_word_vs_bpe_matched_256_4L",
            "corpus": str(CORPUS_WIKI),
            "n_phrases": len(phrases),
            "ft_steps": FT_STEPS,
            "diag_tpw": {"wiki": diag_wiki, "tinystories_100k": diag_ts},
            "word": word,
            "bpe": bpe,
            "gap_hold_all_pp": gap_all,
            "gap_hold_obj_pp": gap_obj,
            "ref_gap_112_story_pp": REF_GAP_112,
            "verdict": verdict,
            "note": (
                "HOLD ALL = natural wiki windows (story_keep hold). "
                "Not TinyStories; hops not used. Width matched to Stage112 BPE."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage136_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(
            f"[136] {verdict} word_HOLD={100*word['hold_all']:.1f}% "
            f"bpe_HOLD={100*bpe['hold_all']:.1f}% gap={100*gap_all:+.1f}pp "
            f"(112 ref ~+{100*REF_GAP_112:.0f}pp)"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 136 WikiText word vs BPE:** {verdict} "
                f"word={100*word['hold_all']:.1f}% bpe={100*bpe['hold_all']:.1f}% "
                f"gap={100*gap_all:+.1f}pp. `stage136_wikitext_word_vs_bpe_decision.json`.\n"
            )
            if "Stage 136 WikiText" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 136")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
