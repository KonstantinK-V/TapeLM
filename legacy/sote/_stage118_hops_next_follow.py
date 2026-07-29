"""
Stage 118 — auto after 116/117, branched on hops-prior result.

If 117a/b show PARITY and obj_prior≈obj (hop bonus inert):
  118a conflict_rerank: add hop bonus ONLY when atom top1-top2 logit gap < τ
  118b hard_mask: on right slot, restrict softmax to hop candidates ∪ {gold during train}
     (infer: hop candidates only; if empty → fall back to full V)

If 117 already PASS_OBJ:
  118 = hop2 joint diag only (no FT) + freeze note

Also always:
  118c phrase_mix: mix real fact-bank phrases into atom CE as extra story-like
     lines (path phrases as training text) — "bake phrases into next-word"
     without hop teacher. fat_frac 0.5 path_bank / 0.25 fat / 0.25 story.

Run (waits for stage116_117_decision.json):
  python _stage118_hops_next_follow.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    FACT_BANK_EXP_F,
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
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import (  # noqa: E402
    annotate_roles,
    build_path_memory,
    hop_candidates_for_prefix,
)

DEC_PRIOR = RES / "stage116_117_decision.json"
LOG = RES / "_stage118_log.txt"
DEC = RES / "stage118_decision.json"
ATOM100 = CKPT / "stage100_scale_100k.pt"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s: int = 8 * 3600, poll_s: int = 40) -> dict:
    log(f"[wait] for {DEC_PRIOR} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PRIOR.exists():
            d = json.loads(DEC_PRIOR.read_text(encoding="utf-8"))
            if d.get("stage117a_rerank") is not None and d.get("stage117b_loss") is not None:
                log("[wait] 116/117 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("116/117 not ready")


def load_bank_phrases() -> list[dict]:
    out = []
    if not FACT_BANK_EXP_F.exists():
        return out
    for raw in FACT_BANK_EXP_F.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        ws = s.split()
        if len(ws) >= 3 and ws[1] in RELS:
            ph = " ".join(ws[:3])
            out.append({
                "phrase": ph, "words": ws[:3], "bucket": "bank_path",
                "triple": (ws[0], ws[1], ws[2]), "subkind": "noun_rel",
                "split": "bank",
            })
    return out


def setup_common(phrases):
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
    bank = load_bank_phrases()
    all_lines = train + hold_seen + hold_rare + hold_story + bank
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    all_words = sorted({w for ln in all_lines for w in ln["words"]})
    word_fps = F.normalize(
        torch.stack([stack.w(w).detach() for w in all_words], 0), dim=-1
    ).to(device)
    surf = all_words
    stoi = {s: i for i, s in enumerate(surf)}
    mem, by_lr, by_rel = build_path_memory(stack, cfg, train + bank, device, FACT_BANK_EXP_F)
    return {
        "cfg": cfg, "device": device, "stack": stack, "train": train,
        "hold_seen": hold_seen, "hold_rare": hold_rare, "hold_story": hold_story,
        "bank": bank, "meta": meta, "surf": surf, "stoi": stoi, "word_fps": word_fps,
        "mem": mem, "by_lr": by_lr, "by_rel": by_rel,
    }


def init_model(ctx):
    model = WordIdTransformer(
        n_vocab=len(ctx["surf"]), d_model=256, n_heads=4, n_layers=2, max_len=16, dropout=0.1
    ).to(ctx["device"])
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == ctx["surf"]:
            model.load_state_dict(ck["word_tf"], strict=True)
            log(f"[init] {ATOM100.name}")
        else:
            model.init_from_fps(ctx["word_fps"])
            log("[init] fp (vocab mismatch)")
    else:
        model.init_from_fps(ctx["word_fps"])
    return model


def train_variant(
    *,
    stage: int,
    tag: str,
    ctx: dict,
    mode: str,  # conflict | hard_mask | phrase_mix
    ft_steps: int = 30000,
    eval_every: int = 1000,
    alpha: float = 5.0,
    tau: float = 1.0,
    fat_frac: float = 0.75,
    bank_frac: float = 0.0,
) -> dict:
    device = ctx["device"]
    stack, mem = ctx["stack"], ctx["mem"]
    by_lr, by_rel = ctx["by_lr"], ctx["by_rel"]
    stoi, surf = ctx["stoi"], ctx["surf"]
    train = ctx["train"]
    bank = ctx["bank"]

    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    story_src = story_lines if len(story_lines) <= 50000 else _subsample(story_lines, 50000, 12)
    fat_pairs = annotate_roles(lines_to_pairs(fat_lines, stoi), fat_lines)
    story_pairs = annotate_roles(lines_to_pairs(story_src, stoi), story_src)
    bank_pairs = annotate_roles(lines_to_pairs(bank, stoi), bank) if bank else []
    if not fat_pairs:
        fat_pairs = story_pairs

    ev_seen = _subsample(ctx["hold_seen"], 600, 1001)
    ev_story = _subsample(ctx["hold_story"], 400, 1003)
    fin_seen = _subsample(ctx["hold_seen"], min(len(ctx["hold_seen"]), 1200), 2001)
    fin_story = _subsample(ctx["hold_story"], min(len(ctx["hold_story"]), 800), 2003)

    model = init_model(ctx)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(272)
    batch = 8

    out_ckpt = CKPT / f"stage{stage}_{tag}.pt"
    log(f"\n======== Stage {stage} {tag} mode={mode} ========")

    def sample_batch():
        if mode == "phrase_mix" and bank_pairs:
            n_bank = max(1, int(round(batch * bank_frac)))
            n_fat = max(1, int(round(batch * fat_frac)))
            n_fat = min(n_fat, batch - n_bank)
            n_st = batch - n_bank - n_fat
            ex = [rng.choice(bank_pairs) for _ in range(n_bank)]
            ex += [rng.choice(fat_pairs) for _ in range(n_fat)]
            ex += [rng.choice(story_pairs) for _ in range(max(0, n_st))]
        else:
            n_fat = max(1, int(round(batch * fat_frac)))
            n_fat = min(n_fat, batch)
            ex = [rng.choice(fat_pairs) for _ in range(n_fat)]
            ex += [rng.choice(story_pairs) for _ in range(batch - n_fat)]
        rng.shuffle(ex)
        return ex

    def apply_prior(logits, exs_valid, train_tgt=None):
        """Modify logits in-place for right-slot rows."""
        for i, ex in enumerate(exs_valid):
            if ex.get("role") != "right":
                continue
            pref = ex["words"][: ex["prefix_len"]]
            cands = hop_candidates_for_prefix(pref, by_lr, by_rel, mem, stack, stoi)
            if not cands:
                continue
            if mode == "conflict":
                top2 = torch.topk(logits[i], k=2)
                gap = float(top2.values[0] - top2.values[1])
                if gap >= tau:
                    continue  # confident — hops silent
                for cid in cands:
                    logits[i, cid] = logits[i, cid] + alpha
            elif mode == "hard_mask":
                allow = set(cands)
                if train_tgt is not None:
                    allow.add(int(train_tgt[i]))
                mask = torch.full_like(logits[i], -1e4)
                for cid in allow:
                    mask[cid] = 0.0
                logits[i] = logits[i] + mask
        return logits

    @torch.no_grad()
    def obj_with_prior(hold):
        pairs = lines_to_pairs(hold, stoi)
        n = h = 0
        for ex in pairs:
            ws = hold[ex["line_i"]]["words"]
            if _role(ex, ws) != "right":
                continue
            ids = torch.tensor([ex["prefix_word_ids"][-model.max_len :]], dtype=torch.long, device=device)
            logits = model.forward(ids)[0, -1].unsqueeze(0)
            fake = [{
                "role": "right",
                "words": ws,
                "prefix_len": ex["prefix_len"],
            }]
            logits = apply_prior(logits.clone(), fake, train_tgt=None)
            pred = surf[int(logits[0].argmax())]
            n += 1
            h += int(pred == ex["target_word"])
        return h / max(n, 1)

    best = {
        "step": 0, "obj": 0.0, "obj_p": 0.0, "rel": 0.0, "story_all": 0.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        seen = eval_id_capacity_suite(model, ev_seen, train, surf, stoi, device)
        story = eval_id_capacity_suite(model, ev_story, train, surf, stoi, device)
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        obj_p = obj_with_prior(ev_seen) if mode in ("conflict", "hard_mask") else obj_s
        curve.append({"step": step, "obj": obj_s, "obj_p": obj_p, "rel": rel_s, "story_all": st_all})
        key = (1 if rel_s >= 0.70 else 0, obj_p, obj_s, st_all)
        bkey = (1 if best["rel"] >= 0.70 else 0, best["obj_p"], best["obj"], best["story_all"])
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, obj_p=obj_p, rel=rel_s, story_all=st_all,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        log(
            f"  step {step:5d}: obj={obj_s*100:.1f}% obj+hop={obj_p*100:.1f}% "
            f"rel={rel_s*100:.1f}% STORY={st_all*100:.1f}%"
        )
        model.train()

    snap(0)
    model.train()
    for step in range(1, ft_steps + 1):
        lr = _warmup_then_constant(step, 1e-3, 200)
        for g in opt.param_groups:
            g["lr"] = lr
        exs = sample_batch()
        exs_valid = [ex for ex in exs if ex["target_word"] in stoi and ex["prefix_word_ids"]]
        packed = collate_word_id_batch(exs_valid, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        exs_valid = exs_valid[: tgt.shape[0]]
        logits = model.logits_last_from_batch(ids, pad_mask)
        if mode in ("conflict", "hard_mask"):
            logits = apply_prior(logits, exs_valid, train_tgt=tgt)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == ft_steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = eval_id_capacity_suite(model, fin_seen, train, surf, stoi, device)
    f_story = eval_id_capacity_suite(model, fin_story, train, surf, stoi, device)
    obj_s = f_seen["obj"]["hit1"]
    rel_s = f_seen["roles"].get("rel", {}).get("hit1", 0.0)
    st_all = f_story["roles"].get("ALL", {}).get("hit1", 0.0)
    obj_p = obj_with_prior(fin_seen) if mode in ("conflict", "hard_mask") else obj_s
    obj_lift = obj_p - REF100["obj"]
    if rel_s >= 0.70 and obj_lift >= 0.03:
        verdict = "PASS_OBJ"
    elif rel_s >= 0.70 and obj_lift >= 0.015:
        verdict = "PARTIAL"
    elif rel_s < 0.70:
        verdict = "FAIL_REL"
    else:
        verdict = "PARITY"

    report = "\n".join([
        f"SOTE Stage {stage} — {tag}",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"mode={mode} alpha={alpha} tau={tau} bank_frac={bank_frac}",
        f"SEEN obj={obj_s*100:.1f}% obj+hop={obj_p*100:.1f}% rel={rel_s*100:.1f}%",
        f"STORY ALL={st_all*100:.1f}%",
        f"vs100 obj+hop {obj_lift*100:+.1f}pp",
        f"Verdict: {verdict}",
    ]) + "\n"
    log("\n" + report)
    (RES / f"stage{stage}_{tag}_report.txt").write_text(report, encoding="utf-8")
    (RES / f"stage{stage}_{tag}_{verdict}.txt").write_text(report, encoding="utf-8")
    result = {
        "stage": stage, "tag": tag, "verdict": verdict, "mode": mode,
        "seen": {"obj": obj_s, "obj_hop": obj_p, "rel": rel_s},
        "story": {"all": st_all}, "obj_lift_pp": obj_lift,
        "curve": curve, "ckpt": str(out_ckpt), "best_step": best["step"],
    }
    (RES / f"stage{stage}_{tag}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"stage": stage, "tag": tag, "word_tf": best["state"], "surfaces": surf}, out_ckpt)
    log(f"Saved {out_ckpt}")
    return result


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage118 start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_prior()
        r117a = prior["stage117a_rerank"]
        r117b = prior["stage117b_loss"]
        # detect inert prior: both parity/fail and tiny prior lift in curves if present
        inert = (
            r117a.get("verdict") in ("PARITY", "FAIL_REL", "PARTIAL")
            and r117b.get("verdict") in ("PARITY", "FAIL_REL", "PARTIAL")
            and abs(float(r117a.get("obj_lift_pp", 0))) < 0.025
            and abs(float(r117b.get("obj_lift_pp", 0))) < 0.025
        )
        passed = r117a.get("verdict") == "PASS_OBJ" or r117b.get("verdict") == "PASS_OBJ"
        log(f"[branch] inert_prior={inert} passed={passed} "
            f"117a={r117a.get('verdict')} 117b={r117b.get('verdict')}")

        phrases = ensure_100k()
        ctx = setup_common(phrases)
        results = {"prior_117": {"a": r117a.get("verdict"), "b": r117b.get("verdict")}}

        if passed and not inert:
            log("[branch] PASS path — skip aggressive hop; run phrase_mix only")
            results["stage118c"] = train_variant(
                stage=118, tag="phrase_mix_bank", ctx=ctx, mode="phrase_mix",
                ft_steps=30000, fat_frac=0.5, bank_frac=0.25,
            )
        else:
            log("[branch] inert/soft — conflict rerank + hard mask + phrase mix")
            results["stage118a"] = train_variant(
                stage=118, tag="hop_conflict_rerank", ctx=ctx, mode="conflict",
                ft_steps=30000, alpha=5.0, tau=1.0, fat_frac=0.75,
            )
            results["stage118b"] = train_variant(
                stage=118, tag="hop_hard_mask", ctx=ctx, mode="hard_mask",
                ft_steps=30000, fat_frac=0.75,
            )
            results["stage118c"] = train_variant(
                stage=118, tag="phrase_mix_bank", ctx=ctx, mode="phrase_mix",
                ft_steps=30000, fat_frac=0.5, bank_frac=0.25,
            )

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "branch": "pass_light" if passed and not inert else "inert_aggressive",
            **{k: v for k, v in results.items()},
            "note": (
                "118a=hop only when top1-top2 gap<τ; "
                "118b=hard mask to hop set on right; "
                "118c=mix fact-bank phrases into CE next-word"
            ),
        }
        # compact serializable
        def slim(x):
            if not isinstance(x, dict):
                return x
            if "verdict" in x:
                return {k: x[k] for k in ("verdict", "tag", "mode", "seen", "story", "obj_lift_pp", "ckpt") if k in x}
            return x
        out_slim = {k: slim(v) if k.startswith("stage") else v for k, v in out.items()}
        DEC.write_text(json.dumps(out_slim, indent=2), encoding="utf-8")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            bits = []
            for key in ("stage118a", "stage118b", "stage118c"):
                if key in results:
                    r = results[key]
                    bits.append(f"{r['tag']} {r['verdict']} obj+hop={r['seen']['obj_hop']*100:.1f}%")
            block = f"\n**Stage 118:** " + "; ".join(bits) + f". `{DEC.name}`.\n"
            if "Stage 118:" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 118")
        log(json.dumps(out_slim, indent=2))
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
