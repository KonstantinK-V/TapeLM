"""
Stage 149 — HPARAM-MATCHED clean unit comparison + 131-style under GPT recipe.

Problem (user): batch/lr/opt/depth often differed word vs BPE, so gaps were
partly recipe noise. Also batch "too big" depends on N.

LOCKED recipe for ALL arms (GPT-like, equal):
  batch=16, lr=3e-4, AdamW wd=0.01, d=256, 4L/4H, steps=40000, warmup=200
  same TinyStories-100k mix seed=272, hops OUT, exact@1 gate

Arms:
  A word_fp          — word-id TF, F85 fp init
  B ws_bpe_rand      — whitespace-BPE V=8k, random emb (112-style unit)
  C ws_bpe_fp        — same BPE cuts, piece emb = letter compose (alphabet identity)
  D func_bias_gpt    — A then +131-style func-bias FT under SAME GPT hparams
                       (not the old Adam 1e-3 / batch8 atom recipe)

Alphabet-BPE note: Stage148 char-pretok collapsed to V≈41 (no merges).
Here "alphabet" = BPE merges *inside* whitespace words (letter alphabet) +
letter-fp piece init — actual V~8k.

Waits for Stage148 decision (pipeline).

Run:
  python _stage149_matched_hparams.py
"""
from __future__ import annotations

import json
import random
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
    compose_plain,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage111_112_follow import _encode_words, eval_bpe_word_holds  # noqa: E402

DEC148 = RES / "stage148_alphabet_bpe_fp_decision.json"
LOG = RES / "_stage149_log.txt"
DEC = RES / "stage149_matched_hparams_decision.json"
BPE_TOK = RES / "stage149_matched_bpe_tokenizer.json"

# --- LOCKED GPT-like recipe ---
BATCH = 16
LR = 3e-4
WD = 0.01
D_MODEL = 256
N_LAYER = 4
N_HEAD = 4
FT_STEPS = 40_000
WARMUP = 200
FUNC_STEPS = 25_000
EVAL_EVERY_WORD = 2000
EVAL_EVERY_BPE = 5000
V_BPE = 8000

FUNC = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "as", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "it", "he", "she", "they", "we", "you", "i", "his", "her", "their", "this",
    "that", "these", "those", "not", "no", "so", "if", "then", "than", "too",
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


def wait_148(timeout_s=8 * 3600, poll=45):
    log(f"[wait] for {DEC148.name} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC148.exists():
            d = json.loads(DEC148.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 148 done verdict={d.get('verdict')} V_abpe={d.get('alphabet_bpe',{}).get('V')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("148 not ready")


def train_ws_bpe(phrases: list[str]) -> Tokenizer:
    if BPE_TOK.exists():
        log(f"[bpe] reuse {BPE_TOK.name}")
        return Tokenizer.from_file(str(BPE_TOK))
    log(f"[bpe] train whitespace-BPE V={V_BPE} (letter merges inside words)")
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
    return tok


def piece_fp(stack, piece: str, dim: int, device):
    s = "".join(c for c in piece if c.isalnum() or c == " ").strip()
    if not s:
        return torch.zeros(dim, device=device)
    try:
        return F.normalize(compose_plain(stack.encoder, stack.composer, s.replace(" ", ""), device).detach(), dim=-1)
    except Exception:
        try:
            return F.normalize(stack.w(s.split()[0]).detach(), dim=-1)
        except Exception:
            return torch.zeros(dim, device=device)


def init_gpt2_emb(model, tok, mode: str, stack, device, dim=256):
    """mode: rand | fp"""
    V = tok.get_vocab_size()
    with torch.no_grad():
        emb = model.transformer.wte.weight
        if mode == "rand":
            emb.normal_(std=0.02)
            return
        n = 0
        for i in range(V):
            piece = (tok.id_to_token(i) or "").replace("Ġ", "").replace("▁", "")
            if piece in ("[PAD]", "[UNK]", "[BOS]", "[EOS]"):
                emb[i].normal_(std=0.02)
                continue
            fp = piece_fp(stack, piece, dim, device)
            emb[i].zero_()
            d = min(dim, fp.numel())
            emb[i, :d] = fp[:d]
            n += 1
        if hasattr(model, "lm_head") and model.lm_head.weight.data_ptr() != emb.data_ptr():
            model.lm_head.weight.copy_(emb)
        log(f"[bpe] fp-init pieces~{n}/{V}")


def make_mix(phrases, cfg, seed=272):
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=seed)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    return train, hold_seen, hold_rare, hold_story, meta


def train_word_matched(phrases, device, cfg, stack):
    log("\n======== A word_fp (MATCHED GPT hparams) ========")
    train, hold_seen, hold_rare, hold_story, meta = make_mix(phrases, cfg)
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    model = WordIdTransformer(len(words), D_MODEL, N_HEAD, N_LAYER, 16, 0.1).to(device)
    model.init_from_fps(fps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[word] V={len(words)} params={n_params/1e6:.2f}M batch={BATCH} lr={LR} {N_LAYER}L")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 60000, 49), stoi)
    if not fat_p:
        fat_p = story_p

    ev_seen = _subsample(hold_seen, 400, 1491)
    ev_story = _subsample(hold_story, 300, 1493)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), 2491)
    fin_story = _subsample(hold_story, min(len(hold_story), 500), 2493)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rr = random.Random(149)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
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
        log(f"  [word] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(
                story_all=sall,
                obj=obj,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        model.train()

    model.train()
    snap(0)
    for step in range(1, FT_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, LR, WARMUP)
        n_fat = max(1, int(round(BATCH * 0.75)))
        batch = [rr.choice(fat_p) for _ in range(n_fat)] + [
            rr.choice(story_p) for _ in range(BATCH - n_fat)
        ]
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
    ck = CKPT / "stage149_word_matched.pt"
    torch.save({"word_tf": best["state"], "surfaces": words}, ck)
    out = {
        "arm": "word_fp",
        "V": len(words),
        "params": n_params,
        "seen_obj": fs["obj"]["hit1"],
        "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
    }
    return out, model, words, stoi, fps, train, hold_seen, hold_story, fat_p, story_p


def train_bpe_matched(phrases, tok, device, cfg, stack, emb_mode: str, tag: str):
    log(f"\n======== {tag} (MATCHED GPT hparams, emb={emb_mode}) ========")
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()
    train, hold_seen, hold_rare, hold_story, meta = make_mix(phrases, cfg)
    seqs = []
    for ln in train:
        ids = _encode_words(tok, ln["words"], max_len=48, bos=bos, eos=eos, pad=pad)
        if len(ids) >= 3:
            seqs.append(ids)
    log(f"[bpe] seqs={len(seqs)} V={V}")

    conf = GPT2Config(
        vocab_size=V,
        n_positions=64,
        n_embd=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=4 * D_MODEL,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )
    model = GPT2LMHeadModel(conf).to(device)
    init_gpt2_emb(model, tok, emb_mode, stack, device, dim=cfg.dim)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[bpe] params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rr = random.Random(149 + (0 if emb_mode == "rand" else 1))
    ev_seen = _subsample(hold_seen, 200, 1491)
    ev_story = _subsample(hold_story, 150, 1493)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 400), 2491)
    fin_story = _subsample(hold_story, min(len(hold_story), 300), 2493)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        s = eval_bpe_word_holds(model, tok, ev_seen, device, max_n_lines=len(ev_seen))
        st = eval_bpe_word_holds(model, tok, ev_story, device, max_n_lines=len(ev_story))
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
    for step in range(1, FT_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, LR, WARMUP)
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
    fs = eval_bpe_word_holds(model, tok, fin_seen, device, max_n_lines=len(fin_seen))
    ft = eval_bpe_word_holds(model, tok, fin_story, device, max_n_lines=len(fin_story))
    ck = CKPT / f"stage149_{tag}.pt"
    torch.save({"gpt2": best["state"], "emb_mode": emb_mode}, ck)
    return {
        "arm": tag,
        "emb_mode": emb_mode,
        "V": V,
        "params": n_params,
        "seen_obj": fs["obj"]["hit1"],
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
    }


def train_func_bias_gpt(
    model,
    words,
    stoi,
    train,
    hold_seen,
    hold_story,
    fat_p,
    story_p,
    device,
):
    """131-style func-bias, but LOCKED GPT hparams (lr=3e-4, batch=16, AdamW)."""
    log("\n======== D func_bias_gpt (131 loss, GPT hparams) ========")
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rr = random.Random(13149)
    ev_seen = _subsample(hold_seen, 400, 1491)
    ev_story = _subsample(hold_story, 300, 1493)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), 2491)
    fin_story = _subsample(hold_story, min(len(hold_story), 500), 2493)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        s = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        st = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
        obj = s["obj"]["hit1"]
        rel = s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        # func in top5 diag
        func_n = func_h = 0
        for ln in ev_story[:80]:
            ws = ln["words"]
            ids = [stoi[w] for w in ws if w in stoi]
            for t in range(1, min(len(ids), 6)):
                logits = model.logits_from_prefix(ids[:t])
                top = [words[int(i)] for i in logits.topk(5).indices.tolist()]
                func_n += 1
                func_h += sum(1 for w in top if w in FUNC) / 5
        curve.append(
            {
                "step": step,
                "obj": obj,
                "rel": rel,
                "story_all": sall,
                "func_top5": func_h / max(func_n, 1),
            }
        )
        log(
            f"  [func] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | "
            f"STORY={100*sall:.1f}% func5={100*func_h/max(func_n,1):.1f}%"
        )
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(
                story_all=sall,
                obj=obj,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        model.train()

    model.train()
    snap(0)
    for step in range(1, FUNC_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, LR, WARMUP)
        losses = []
        # story-heavy with func penalty (131)
        for _ in range(max(1, BATCH // 2)):
            ex = rr.choice(story_p) if story_p else rr.choice(fat_p)
            gold_w = ex["target_word"]
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            logits = model.logits_last_from_batch(ids, mask)
            ce = F.cross_entropy(logits, tgt)
            if gold_w not in FUNC:
                func_ids = [stoi[w] for w in FUNC if w in stoi]
                if func_ids:
                    fl = logits[0, func_ids].mean()
                    gl = logits[0, int(tgt[0])]
                    ce = ce + 0.15 * F.relu(fl + 0.5 - gl)
            losses.append(ce)
        for _ in range(max(1, BATCH // 2)):
            ex = rr.choice(fat_p)
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            losses.append(F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt))
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY_WORD == 0 or step == FUNC_STEPS:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    fs = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    ft = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    ck = CKPT / "stage149_func_bias_gpt.pt"
    torch.save({"word_tf": best["state"], "surfaces": words}, ck)
    return {
        "arm": "func_bias_gpt",
        "protocol": "131_func_penalty_under_GPT_hparams",
        "seen_obj": fs["obj"]["hit1"],
        "seen_rel": fs["roles"].get("rel", {}).get("hit1", 0.0),
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "curve": curve,
        "ckpt": str(ck),
        "ref131_story": 0.2446,
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"149 matched hparams start {datetime.now(timezone.utc).isoformat()}")
    log(f"LOCK batch={BATCH} lr={LR} AdamW wd={WD} {N_LAYER}L/{N_HEAD}H d={D_MODEL} steps={FT_STEPS}")
    try:
        up = wait_148()
        phrases = ensure_100k()
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

        tok = train_ws_bpe(phrases)
        log(f"[bpe] V={tok.get_vocab_size()} (148 char-BPE was V~41 broken)")

        word, model_w, words, stoi, fps, train, hold_seen, hold_story, fat_p, story_p = train_word_matched(
            phrases, device, cfg, stack
        )
        bpe_rand = train_bpe_matched(phrases, tok, device, cfg, stack, "rand", "ws_bpe_rand")
        bpe_fp = train_bpe_matched(phrases, tok, device, cfg, stack, "fp", "ws_bpe_fp")

        # reload best word weights for func-bias continuation
        model_w.load_state_dict(
            torch.load(word["ckpt"], map_location="cpu", weights_only=False)["word_tf"]
        )
        model_w = model_w.to(device)
        func = train_func_bias_gpt(
            model_w, words, stoi, train, hold_seen, hold_story, fat_p, story_p, device
        )

        gap_bpe_vs_word = bpe_rand["story_all"] - word["story_all"]
        gap_fp_vs_word = bpe_fp["story_all"] - word["story_all"]
        gap_fp_vs_rand = bpe_fp["story_all"] - bpe_rand["story_all"]
        func_lift = func["story_all"] - word["story_all"]

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "hparam_matched_unit_compare_plus_131_gpt",
            "locked_hparams": {
                "batch": BATCH,
                "lr": LR,
                "optimizer": "AdamW",
                "weight_decay": WD,
                "d_model": D_MODEL,
                "n_layer": N_LAYER,
                "n_head": N_HEAD,
                "ft_steps": FT_STEPS,
                "warmup": WARMUP,
                "func_steps": FUNC_STEPS,
            },
            "upstream_148": {
                "verdict": up.get("verdict"),
                "abpe_V": (up.get("alphabet_bpe") or {}).get("V"),
                "note": "148 char-pretok failed merges (V~41); 149 uses ws-BPE+fp instead",
            },
            "word_fp": word,
            "ws_bpe_rand": bpe_rand,
            "ws_bpe_fp": bpe_fp,
            "func_bias_gpt": func,
            "gaps_story": {
                "bpe_rand_minus_word": gap_bpe_vs_word,
                "bpe_fp_minus_word": gap_fp_vs_word,
                "bpe_fp_minus_bpe_rand": gap_fp_vs_rand,
                "func_minus_word": func_lift,
            },
            "ref100": REF100,
            "ref112_unmatched_gap": 0.1435,
            "ref131_story": 0.2446,
            "verdict": (
                "MATCHED_TABLE"
            ),
            "note": (
                "All A/B/C share identical batch/lr/opt/depth/steps. "
                "D = 131 func-bias continued from A under same GPT hparams."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / "stage149_MATCHED_TABLE.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(
            f"[149] word={100*word['story_all']:.1f}% bpe_rand={100*bpe_rand['story_all']:.1f}% "
            f"bpe_fp={100*bpe_fp['story_all']:.1f}% func={100*func['story_all']:.1f}% | "
            f"gap_bpe={100*gap_bpe_vs_word:+.1f}pp func_lift={100*func_lift:+.1f}pp"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 149 matched hparams:** word={100*word['story_all']:.1f}% "
                f"bpe_rand={100*bpe_rand['story_all']:.1f}% bpe_fp={100*bpe_fp['story_all']:.1f}% "
                f"func_gpt={100*func['story_all']:.1f}% "
                f"(batch={BATCH} lr={LR} 4L). `stage149_matched_hparams_decision.json`.\n"
            )
            if "Stage 149 matched" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 149")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
