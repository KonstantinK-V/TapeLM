"""
Stage 119 — after 118: use SOTE *advantages* (not chase STORY ALL).

SOTE edges vs BPE LM that we can bank in training:
  1) Word atoms → shorter sequences (8 words vs ~15–25 BPE toks) → fewer matmuls/step
  2) Closed V~5k vs BPE 8k–50k → cheaper softmax / decode
  3) F85 fp init → fewer steps to lock path rel (data efficiency)
  4) Known path roles (rel/right) → curriculum / sparse compute on hard slots
  5) Freeze encode+optional freeze emb early → tiny trainable head, fast FT

Digs:
  119a speed_bench (eval/train micro): tokens/sec & ms/batch SOTE word-TF vs
       reload Stage112-sized GPT2 on same phrases (wall-clock, CUDA sync).
  119b freeze_emb_ft: init from Stage100, FREEZE tok emb, train only Transformer
       (+pos); measure steps-to-rel≥90% and final SEEN/STORY. Speed + "geometry
       already good" thesis.
  119c role_curriculum: phase1 15k steps only rel+right pairs; phase2 25k full mix.
       Compare steps-to-SEEN-obj≥35% vs Stage100 flat mix.

Gates (not STORY-primary):
  SPEED if SOTE ms/batch ≤ 0.7× BPE on matched batch content
  EFFICIENT if 119b/c reach rel≥90% in ≤60% steps of ref curve
  PARITY otherwise

Run (waits for stage118_decision.json):
  python _stage119_sote_edges_follow.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from train import (  # noqa: E402
    CKPT,
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
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402

DEC_PRIOR = RES / "stage118_decision.json"
LOG = RES / "_stage119_log.txt"
DEC = RES / "stage119_decision.json"
ATOM100 = CKPT / "stage100_scale_100k.pt"
BPE_TOK = RES / "stage112_bpe_tokenizer.json"
BPE_CKPT = CKPT / "stage112_bpe_baseline.pt"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s: int = 10 * 3600, poll_s: int = 40) -> dict:
    log(f"[wait] for {DEC_PRIOR} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PRIOR.exists():
            d = json.loads(DEC_PRIOR.read_text(encoding="utf-8"))
            # any 118 key present
            if any(k.startswith("stage118") for k in d):
                log("[wait] 118 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("118 not ready")


def annotate(pairs, lines):
    for ex in pairs:
        ex["role"] = _role(ex, lines[ex["line_i"]]["words"])
    return pairs


def setup(phrases):
    cfg = Config()
    cfg.c87_n_fat = 200
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = 40
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = 0.50
    cfg.c87_hold_frac = 0.15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent85)
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    all_lines = train + hold_seen + hold_rare + hold_story
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in all_lines for w in ln["words"]})
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    stoi = {s: i for i, s in enumerate(words)}
    return {
        "cfg": cfg, "device": device, "stack": stack, "train": train,
        "hold_seen": hold_seen, "hold_story": hold_story, "hold_rare": hold_rare,
        "meta": meta, "surf": words, "stoi": stoi, "fps": fps, "phrases": phrases,
    }


def bench_speed(ctx) -> dict:
    """Wall-clock train step + forward for SOTE vs BPE GPT mini."""
    device = ctx["device"]
    surf, stoi = ctx["surf"], ctx["stoi"]
    train = ctx["train"]
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = annotate(lines_to_pairs(fat[:500], stoi), fat[:500])
    story_p = annotate(lines_to_pairs(_subsample(story, 2000, 1), stoi), _subsample(story, 2000, 1))
    rng = random.Random(0)
    batch = 8

    model = WordIdTransformer(len(surf), 256, 4, 2, 16, 0.1).to(device)
    model.init_from_fps(ctx["fps"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def sote_step():
        ex = [rng.choice(fat_p) for _ in range(6)] + [rng.choice(story_p) for _ in range(2)]
        packed = collate_word_id_batch(ex, stoi, 16, model.pad_id, device)
        if packed is None:
            return 0
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return ids.numel()

    # warmup
    for _ in range(20):
        sote_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ntok = 0
    nsteps = 200
    for _ in range(nsteps):
        ntok += sote_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sote_ms = elapsed * 1000 / nsteps
    sote_tok_s = ntok / elapsed

    bpe = {"available": False}
    if BPE_TOK.exists():
        tok = Tokenizer.from_file(str(BPE_TOK))
        V = tok.get_vocab_size()
        bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
        conf = GPT2Config(
            vocab_size=V, n_positions=64, n_embd=256, n_layer=4, n_head=4, n_inner=1024,
            bos_token_id=bos, eos_token_id=eos, pad_token_id=pad,
        )
        gpt = GPT2LMHeadModel(conf).to(device)
        if BPE_CKPT.exists():
            try:
                ck = torch.load(BPE_CKPT, map_location="cpu", weights_only=False)
                gpt.load_state_dict(ck["model"], strict=False)
            except Exception:
                pass
        gopt = torch.optim.AdamW(gpt.parameters(), lr=3e-4)
        seqs = []
        for ln in _subsample(train, 3000, 2):
            ids = tok.encode(" ".join(ln["words"])).ids[:48]
            if len(ids) >= 3:
                seqs.append(ids)

        def bpe_step():
            batch_seqs = [rng.choice(seqs) for _ in range(batch)]
            maxlen = max(len(s) for s in batch_seqs)
            x = torch.full((batch, maxlen), pad, dtype=torch.long, device=device)
            for i, s in enumerate(batch_seqs):
                x[i, : len(s)] = torch.tensor(s, device=device)
            labels = x.clone()
            labels[labels == pad] = -100
            loss = gpt(x, labels=labels).loss
            gopt.zero_grad(set_to_none=True)
            loss.backward()
            gopt.step()
            return int((labels != -100).sum())

        for _ in range(20):
            bpe_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ntok = 0
        for _ in range(nsteps):
            ntok += bpe_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        bpe = {
            "available": True,
            "ms_per_step": elapsed * 1000 / nsteps,
            "tok_per_s": ntok / elapsed,
            "params": sum(p.numel() for p in gpt.parameters()),
        }

    sote_params = sum(p.numel() for p in model.parameters())
    out = {
        "sote": {
            "ms_per_step": sote_ms,
            "tok_per_s": sote_tok_s,
            "params": sote_params,
            "unit": "word_positions_in_batch",
        },
        "bpe": bpe,
    }
    if bpe.get("available"):
        out["ratio_ms_sote_over_bpe"] = sote_ms / max(bpe["ms_per_step"], 1e-9)
        out["verdict"] = (
            "SPEED" if sote_ms <= 0.7 * bpe["ms_per_step"]
            else ("PARITY_SPEED" if sote_ms <= 1.1 * bpe["ms_per_step"] else "SLOWER")
        )
    else:
        out["verdict"] = "SOTE_ONLY"
    log(f"[119a] {json.dumps(out, indent=2)}")
    (RES / "stage119a_speed_bench_report.txt").write_text(
        f"Stage 119a speed_bench\n{json.dumps(out, indent=2)}\n", encoding="utf-8"
    )
    return out


def train_freeze_emb(ctx) -> dict:
    """Freeze word emb; train Transformer only — SOTE geometry as fixed codebook rows."""
    device, surf, stoi = ctx["device"], ctx["surf"], ctx["stoi"]
    train = ctx["train"]
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = annotate(lines_to_pairs(fat, stoi), fat)
    story_p = annotate(lines_to_pairs(_subsample(story, 40000, 3), stoi), _subsample(story, 40000, 3))
    ev_seen = _subsample(ctx["hold_seen"], 600, 1001)
    ev_story = _subsample(ctx["hold_story"], 400, 1003)

    model = WordIdTransformer(len(surf), 256, 4, 2, 16, 0.1).to(device)
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == surf:
            model.load_state_dict(ck["word_tf"], strict=True)
            log("[119b] loaded Stage100 then FREEZE emb")
        else:
            model.init_from_fps(ctx["fps"])
    else:
        model.init_from_fps(ctx["fps"])
    model.tok.weight.requires_grad_(False)
    # pos + transformer trainable
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"[119b] trainable={n_train/1e6:.2f}M / {n_all/1e6:.2f}M")

    rng = random.Random(272)
    steps_to_rel = None
    curve = []
    best = {"step": 0, "obj": 0.0, "rel": 0.0, "story_all": 0.0,
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    def snap(step):
        nonlocal steps_to_rel
        model.eval()
        seen = eval_id_capacity_suite(model, ev_seen, train, surf, stoi, device)
        story = eval_id_capacity_suite(model, ev_story, train, surf, stoi, device)
        obj, rel = seen["obj"]["hit1"], seen["roles"].get("rel", {}).get("hit1", 0.0)
        st = story["roles"].get("ALL", {}).get("hit1", 0.0)
        if steps_to_rel is None and rel >= 0.90:
            steps_to_rel = step
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": st})
        key = (1 if rel >= 0.5 else 0, obj, rel)
        bkey = (1 if best["rel"] >= 0.5 else 0, best["obj"], best["rel"])
        if key >= bkey:
            best.update(step=step, obj=obj, rel=rel, story_all=st,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        log(f"  [119b] step {step}: obj={obj*100:.1f}% rel={rel*100:.1f}% STORY={st*100:.1f}%")
        model.train()

    snap(0)
    model.train()
    ft_steps = 30000
    for step in range(1, ft_steps + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        ex = [rng.choice(fat_p) for _ in range(6)] + [rng.choice(story_p) for _ in range(2)]
        packed = collate_word_id_batch(ex, stoi, 16, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == ft_steps:
            snap(step)

    model.load_state_dict(best["state"])
    # ref: Stage100 locked rel early ~3–6k often; efficiency if steps_to_rel <= 0.6 * 8000
    ref_steps = 8000
    verdict = "EFFICIENT" if steps_to_rel is not None and steps_to_rel <= 0.6 * ref_steps else (
        "HOLD_REL" if best["rel"] >= 0.90 else "FAIL_REL"
    )
    if best["obj"] + 0.02 < REF100["obj"] and verdict == "EFFICIENT":
        verdict = "EFFICIENT_BUT_OBJ_DROP"
    out = {
        "stage": 119, "tag": "freeze_emb_ft", "verdict": verdict,
        "trainable_params": n_train, "steps_to_rel90": steps_to_rel,
        "seen": {"obj": best["obj"], "rel": best["rel"]},
        "story": {"all": best["story_all"]}, "curve": curve,
    }
    ck = CKPT / "stage119_freeze_emb_ft.pt"
    torch.save({"word_tf": best["state"], "surfaces": surf, "frozen_emb": True}, ck)
    out["ckpt"] = str(ck)
    (RES / f"stage119_freeze_emb_ft_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[119b] {verdict} steps_to_rel90={steps_to_rel}")
    return out


def train_curriculum(ctx) -> dict:
    """Phase1: only rel+right pairs; Phase2: full mix."""
    device, surf, stoi = ctx["device"], ctx["surf"], ctx["stoi"]
    train = ctx["train"]
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = annotate(lines_to_pairs(fat, stoi), fat)
    story_p = annotate(lines_to_pairs(_subsample(story, 40000, 4), stoi), _subsample(story, 40000, 4))
    path_p = [ex for ex in fat_p + story_p if ex.get("role") in ("rel", "right")]
    log(f"[119c] path_pairs={len(path_p)} all_story_pairs={len(story_p)}")

    ev_seen = _subsample(ctx["hold_seen"], 600, 1001)
    ev_story = _subsample(ctx["hold_story"], 400, 1003)
    model = WordIdTransformer(len(surf), 256, 4, 2, 16, 0.1).to(device)
    model.init_from_fps(ctx["fps"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(272)

    steps_to_obj = None
    steps_to_rel = None
    curve = []
    best = {"step": 0, "obj": 0.0, "rel": 0.0, "story_all": 0.0,
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    def snap(step):
        nonlocal steps_to_obj, steps_to_rel
        model.eval()
        seen = eval_id_capacity_suite(model, ev_seen, train, surf, stoi, device)
        story_e = eval_id_capacity_suite(model, ev_story, train, surf, stoi, device)
        obj, rel = seen["obj"]["hit1"], seen["roles"].get("rel", {}).get("hit1", 0.0)
        st = story_e["roles"].get("ALL", {}).get("hit1", 0.0)
        if steps_to_rel is None and rel >= 0.90:
            steps_to_rel = step
        if steps_to_obj is None and obj >= 0.35:
            steps_to_obj = step
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": st})
        key = (1 if rel >= 0.5 else 0, obj, rel)
        bkey = (1 if best["rel"] >= 0.5 else 0, best["obj"], best["rel"])
        if key >= bkey:
            best.update(step=step, obj=obj, rel=rel, story_all=st,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        log(f"  [119c] step {step}: obj={obj*100:.1f}% rel={rel*100:.1f}% STORY={st*100:.1f}%")
        model.train()

    def run_phase(n_steps, pool, start_step, fat_frac=0.75):
        model.train()
        for i in range(1, n_steps + 1):
            step = start_step + i
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, 1e-3, 200)
            if pool is path_p:
                ex = [rng.choice(pool) for _ in range(8)]
            else:
                n_fat = 6
                ex = [rng.choice(fat_p) for _ in range(n_fat)] + [rng.choice(story_p) for _ in range(8 - n_fat)]
            packed = collate_word_id_batch(ex, stoi, 16, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 1000 == 0:
                snap(step)
        return start_step + n_steps

    snap(0)
    log("[119c] phase1 path-only 15000")
    s = run_phase(15000, path_p, 0)
    log("[119c] phase2 full mix 25000")
    run_phase(25000, None, s)

    # efficiency vs typical ~10–15k to obj 35%
    verdict = "EFFICIENT" if steps_to_obj is not None and steps_to_obj <= 9000 else "PARITY"
    if best["rel"] < 0.70:
        verdict = "FAIL_REL"
    out = {
        "stage": 119, "tag": "role_curriculum", "verdict": verdict,
        "steps_to_rel90": steps_to_rel, "steps_to_obj35": steps_to_obj,
        "seen": {"obj": best["obj"], "rel": best["rel"]},
        "story": {"all": best["story_all"]}, "curve": curve,
    }
    ck = CKPT / "stage119_role_curriculum.pt"
    torch.save({"word_tf": best["state"], "surfaces": surf}, ck)
    out["ckpt"] = str(ck)
    (RES / f"stage119_role_curriculum_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[119c] {verdict} steps_to_obj35={steps_to_obj} steps_to_rel90={steps_to_rel}")
    return out


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage119 SOTE-edges start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_prior()
        phrases = ensure_100k()
        ctx = setup(phrases)
        r_a = bench_speed(ctx)
        r_b = train_freeze_emb(ctx)
        r_c = train_curriculum(ctx)
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thesis": (
                "SOTE advantages for training: shorter word seqs, smaller closed V, "
                "fp init, freezable emb codebook, role curriculum — chase efficiency/"
                "speed/path lock, not STORY ALL parity with BPE"
            ),
            "prior_118_keys": [k for k in prior if str(k).startswith("stage118")],
            "stage119a_speed": r_a,
            "stage119b_freeze_emb": {k: r_b[k] for k in r_b if k != "curve"},
            "stage119c_curriculum": {k: r_c[k] for k in r_c if k != "curve"},
            "curves": {"119b": r_b.get("curve"), "119c": r_c.get("curve")},
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 119 (SOTE edges):** speed {r_a.get('verdict')}; "
                f"freeze_emb {r_b['verdict']} (rel90@{r_b.get('steps_to_rel90')}); "
                f"curriculum {r_c['verdict']} (obj35@{r_c.get('steps_to_obj35')}). "
                f"`stage119_decision.json`.\n"
            )
            if "Stage 119 (SOTE edges)" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 119")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
