"""
Stage 148 — Alphabet-BPE (SOTE charset) + letter-fp piece emb; word confirm.

Never done before: BPE *merge algorithm* over our alphabet (a-z0-9+space),
NOT morph stems, NOT whitespace-word BPE with random emb, NOT raw char LM.

Protocol:
  1) Filter phrases to SOTE charset; train BPE from *character* pretok
     (Split '.') so merges are letter-pair freq like classic BPE.
  2) Piece emb = compose letter→fp (F85 encoder+composer) for each vocab string.
  3) Train causal TF next-piece CE (d=256, 4L/4H to match mini-GPT width/depth).
  4) Eval gate = word-level exact@1 (greedy decode pieces until whitespace).
  5) Ctrl = word-id CE (2L) on same mix — report gap vs alphabet-BPE.

Hops OUT. Gate exact@1.

Run:
  python _stage148_alphabet_bpe_fp.py
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
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, Regex
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
    compose_plain,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402

LOG = RES / "_stage148_log.txt"
DEC = RES / "stage148_alphabet_bpe_fp_decision.json"
BPE_TOK = RES / "stage148_alphabet_bpe_tokenizer.json"

FT_STEPS = 40_000
EVAL_EVERY_WORD = 2000
EVAL_EVERY_BPE = 5000
BATCH = 8
VOCAB = 8000
MAX_PIECE_LEN = 64


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def to_alpha_text(words: list[str]) -> str:
    """Join words with space; drop chars outside SOTE CHAR2ID (keep space)."""
    parts = []
    for w in words:
        ww = "".join(c for c in w.lower() if c in CHAR2ID and c != " ")
        if ww:
            parts.append(ww)
    return " ".join(parts)


def train_alphabet_bpe(phrases: list[str], vocab_size: int = VOCAB) -> Tokenizer:
    if BPE_TOK.exists():
        log(f"[abpe] reuse {BPE_TOK.name}")
        return Tokenizer.from_file(str(BPE_TOK))
    # corpus lines already charset-ish; normalize
    lines = []
    for ph in phrases:
        t = to_alpha_text(ph.split())
        if t:
            lines.append(t)
    log(f"[abpe] train alphabet-BPE V={vocab_size} on {len(lines)} lines (char pretok)")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    # every character is an initial symbol — merges over our alphabet
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"."), behavior="isolated")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=False,
        initial_alphabet=sorted({c for c in CHAR2ID if c != " "} | {" "}),
    )
    tok.train_from_iterator(lines, trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tok.token_to_id("[BOS]")), ("[EOS]", tok.token_to_id("[EOS]"))],
    )
    # decode: pieces concatenate (no extra spaces from pretok)
    BPE_TOK.parent.mkdir(exist_ok=True)
    tok.save(str(BPE_TOK))
    return tok


def piece_fp(stack, piece: str, dim: int, device):
    """Compose letter→fp for a BPE piece string (may include spaces)."""
    s = "".join(c for c in piece if c in CHAR2ID)
    if not s:
        return torch.zeros(dim, device=device)
    # spaces inside piece: compose non-space chunks then mean
    chunks = [c for c in s.split(" ") if c]
    if not chunks:
        # pure spaces
        return torch.zeros(dim, device=device)
    fps = []
    for ch in chunks:
        try:
            fps.append(compose_plain(stack.encoder, stack.composer, ch, device).detach())
        except Exception:
            try:
                fps.append(stack.w(ch).detach())
            except Exception:
                fps.append(torch.zeros(dim, device=device))
    return F.normalize(torch.stack(fps, 0).mean(0), dim=-1)


def init_gpt2_from_piece_fps(model, tok, stack, device, dim=256):
    V = tok.get_vocab_size()
    with torch.no_grad():
        emb = model.transformer.wte.weight
        n = 0
        for i in range(V):
            piece = tok.id_to_token(i) or ""
            # tokenizers may store Ġ etc — strip meta
            piece = piece.replace("Ġ", " ").replace("▁", " ")
            if piece in ("[PAD]", "[UNK]", "[BOS]", "[EOS]", "<pad>", "<unk>"):
                emb[i].normal_(std=0.02)
                continue
            fp = piece_fp(stack, piece, dim, device)
            d = min(dim, fp.numel())
            emb[i].zero_()
            emb[i, :d] = fp[:d]
            n += 1
        # tie lm_head if present
        if hasattr(model, "lm_head") and model.lm_head.weight.data_ptr() != emb.data_ptr():
            model.lm_head.weight.copy_(emb)
    log(f"[abpe] init emb from letter-fp for ~{n}/{V} pieces")


def encode_alpha(tok, words, max_len, bos, eos, pad):
    text = to_alpha_text(words)
    if not text:
        return [bos, eos]
    ids = tok.encode(text).ids[:max_len]
    if not ids:
        ids = [bos, eos]
    return ids


@torch.no_grad()
def abpe_pred_next_word(model, tok, prefix_words, device, max_new=12):
    """Greedy piece decode until whitespace-delimited word completes."""
    bos = tok.token_to_id("[BOS]")
    eos = tok.token_to_id("[EOS]")
    pad = tok.token_to_id("[PAD]")
    ids = encode_alpha(tok, prefix_words, MAX_PIECE_LEN, bos, eos, pad)
    if ids and ids[-1] == eos:
        ids = ids[:-1]
    if not ids:
        ids = [bos]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits = model(x).logits[0, -1]
        nxt = int(logits.argmax())
        if nxt in (eos, pad, bos):
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        decoded = tok.decode(out)
        # word finished when we have a space (next word started) or pure word
        if " " in decoded.strip():
            return decoded.strip().split()[0]
    decoded = tok.decode(out).strip()
    if not decoded:
        return ""
    return decoded.split()[0] if decoded.split() else decoded


@torch.no_grad()
def eval_abpe_word_holds(model, tok, hold_lines, device, max_n_lines=400):
    lines = hold_lines if len(hold_lines) <= max_n_lines else _subsample(hold_lines, max_n_lines, 7)
    roles = defaultdict(lambda: {"n": 0, "h1": 0})
    obj = {"n": 0, "h1": 0}
    for ln in lines:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            pred = abpe_pred_next_word(model, tok, ws[:t], device)
            ok = int(pred == gold)
            ex = {"target_word": gold, "prefix_len": t}
            role = _role(ex, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h1"] += ok
            if t >= 1 and ws[t - 1] in RELS:
                obj["n"] += 1
                obj["h1"] += ok

    def pack(d):
        return {k: {"n": v["n"], "hit1": v["h1"] / max(v["n"], 1)} for k, v in d.items()}

    return {
        "roles": pack(roles),
        "obj": {"hit1": obj["h1"] / max(obj["n"], 1), "n": obj["n"]},
        "n_lines": len(lines),
    }


def train_word_ctrl(phrases, device, cfg):
    log("\n======== A word-id ctrl ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
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
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
    model.init_from_fps(fps)
    n_params = sum(p.numel() for p in model.parameters())
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 50000, 48), stoi)
    if not fat_p:
        fat_p = story_p
    ev_seen = _subsample(hold_seen, 400, 1481)
    ev_story = _subsample(hold_story, 300, 1483)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), 2481)
    fin_story = _subsample(hold_story, min(len(hold_story), 500), 2483)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(148)
    best = {"story_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    curve = []

    def snap(step):
        model.eval()
        s = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        st = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
        obj, rel = s["obj"]["hit1"], s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(f"  [word] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(story_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.train()

    log(f"[word] V={len(words)} params={n_params/1e6:.2f}M")
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
    fs = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    ft = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    return {
        "arm": "word",
        "V": len(words),
        "params": n_params,
        "seen_obj": fs["obj"]["hit1"],
        "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "curve": curve,
    }


def train_alphabet_bpe_arm(phrases, tok, device, cfg):
    log("\n======== B alphabet-BPE + letter-fp emb ========")
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()
    parent = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent.exists():
        parent = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent)
    for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
        for p in mod.parameters():
            p.requires_grad_(False)
        mod.eval()

    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    seqs = []
    for ln in train:
        ids = encode_alpha(tok, ln["words"], MAX_PIECE_LEN, bos, eos, pad)
        if len(ids) >= 3:
            seqs.append(ids)
    log(f"[abpe] seqs={len(seqs)} V={V}")

    conf = GPT2Config(
        vocab_size=V,
        n_positions=MAX_PIECE_LEN,
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_inner=1024,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    model = GPT2LMHeadModel(conf).to(device)
    init_gpt2_from_piece_fps(model, tok, stack, device, dim=cfg.dim)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[abpe] params={n_params/1e6:.2f}M")

    # mean pieces per word diag
    rr = random.Random(148)
    sample = _subsample(train, 800, 9)
    n_w = n_p = 0
    for ln in sample:
        ws = ln["words"]
        n_w += len(ws)
        ids = encode_alpha(tok, ws, MAX_PIECE_LEN, bos, eos, pad)
        # strip bos/eos
        ids = [i for i in ids if i not in (bos, eos, pad)]
        n_p += len(ids)
    mean_ppw = n_p / max(n_w, 1)
    log(f"[abpe] diag mean_pieces_per_word={mean_ppw:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    ev_seen = _subsample(hold_seen, 200, 1481)
    ev_story = _subsample(hold_story, 150, 1483)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 400), 2481)
    fin_story = _subsample(hold_story, min(len(hold_story), 300), 2483)
    best = {"story_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    curve = []

    def snap(step):
        model.eval()
        s = eval_abpe_word_holds(model, tok, ev_seen, device, max_n_lines=len(ev_seen))
        st = eval_abpe_word_holds(model, tok, ev_story, device, max_n_lines=len(ev_story))
        obj = s["obj"]["hit1"]
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "story_all": sall})
        log(f"  [abpe] step {step}: SEEN obj={100*obj:.1f}% | STORY ALL={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(story_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
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
    fs = eval_abpe_word_holds(model, tok, fin_seen, device, max_n_lines=len(fin_seen))
    ft = eval_abpe_word_holds(model, tok, fin_story, device, max_n_lines=len(fin_story))
    ck = CKPT / "stage148_alphabet_bpe_fp.pt"
    torch.save({"gpt2": best["state"], "tokenizer": str(BPE_TOK), "mean_ppw": mean_ppw}, ck)
    return {
        "arm": "alphabet_bpe_fp",
        "V": V,
        "params": n_params,
        "mean_pieces_per_word": mean_ppw,
        "seen_obj": fs["obj"]["hit1"],
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"148 alphabet-BPE + letter-fp start {datetime.now(timezone.utc).isoformat()}")
    log("BPE merges over SOTE alphabet; piece emb=compose(fp); gate=word exact@1")
    try:
        phrases = ensure_100k()
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tok = train_alphabet_bpe(phrases, VOCAB)
        # sample vocab pieces
        sample_pieces = [tok.id_to_token(i) for i in range(min(20, tok.get_vocab_size()))]
        log(f"[abpe] sample tokens: {sample_pieces}")

        word = train_word_ctrl(phrases, device, cfg)
        abpe = train_alphabet_bpe_arm(phrases, tok, device, cfg)

        lift = abpe["story_all"] - word["story_all"]
        obj_d = abpe["seen_obj"] - word["seen_obj"]
        if obj_d < -0.03 and lift < 0:
            verdict = "HARM"
        elif lift >= 0.03 and word["seen_rel"] >= 0.70:
            verdict = "PASS"
        elif lift >= 0.015:
            verdict = "PARTIAL"
        elif abs(lift) < 0.015:
            verdict = "PARITY"
        else:
            verdict = "PARITY"

        # also compare to Stage112 whitespace-BPE gap narrative
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "alphabet_bpe_merges_letter_fp_emb_word_confirm",
            "not_tested_as": [
                "112_whitespace_bpe_random_emb",
                "138_morph_piece_ce",
                "143_raw_char_stream",
            ],
            "word": word,
            "alphabet_bpe": abpe,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
            "ref100": REF100,
            "verdict": verdict,
            "note": (
                "True BPE merge over a-z0-9+space (char pretok). "
                "Piece rows initialized from F85 letter compose, then FT. "
                "Word exact@1 via greedy decode to whitespace."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage148_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(
            f"[148] {verdict} word_STORY={100*word['story_all']:.1f}% "
            f"abpe_STORY={100*abpe['story_all']:.1f}% lift={100*lift:+.1f}pp "
            f"ppw={abpe['mean_pieces_per_word']:.2f}"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 148 alphabet-BPE + letter-fp:** {verdict} "
                f"word={100*word['story_all']:.1f}% abpe={100*abpe['story_all']:.1f}% "
                f"lift={100*lift:+.1f}pp. `stage148_alphabet_bpe_fp_decision.json`.\n"
            )
            if "Stage 148 alphabet-BPE" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 148")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
