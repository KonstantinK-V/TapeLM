"""
Stage 113–114 after 111/112:

Hypothesis (post 107/108 PARITY):
  Soft CE downweight on saturated rel does not free capacity — it just weakens
  an already-solved head. Better: change *which* examples enter the batch.

  113 rel_undersample: sample rel-role pairs at 0.5× their natural pool rate
      (no CE class/slot weights). recipe98, SOTE init, fat_frac=0.75, ~100k.
  114 rare_long_sample: sample probability ∝ len(target)/(freq+1), renormalized
      (encourages long/rare targets; no explicit rel knob).

Gate vs Stage100 (obj 42.6% / STORY 19.6%):
  PASS if SEEN obj +3pp with rel>=0.70
  PARTIAL if +1.5pp obj or +2pp STORY with rel held

Run (waits for stage111_112_decision.json):
  python _stage113_114_follow.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

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
from _stage109_110_slot_baseline import (  # noqa: E402
    REF100,
    ensure_100k,
    _subsample,
)

DEC_PRIOR = RES / "stage111_112_decision.json"
LOG = RES / "_stage113_114_log.txt"
DEC = RES / "stage113_114_decision.json"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s: int = 12 * 3600, poll_s: int = 45) -> dict:
    log(f"[wait] for {DEC_PRIOR} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PRIOR.exists():
            d = json.loads(DEC_PRIOR.read_text(encoding="utf-8"))
            if d.get("stage111") is not None and d.get("stage112") is not None:
                log("[wait] 111/112 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("111/112 not ready")


def annotate_roles(pairs, lines):
    for ex in pairs:
        ex["role"] = _role(ex, lines[ex["line_i"]]["words"])
    return pairs


def train_sample_run(
    *,
    stage: int,
    tag: str,
    phrases: list[str],
    mode: str,  # "rel_under" | "rare_long"
    rel_rate: float = 0.5,
    ft_steps: int = 50000,
    eval_every: int = 1000,
    fat_frac: float = 0.75,
    ref: dict | None = None,
) -> dict:
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

    out_ckpt = CKPT / f"stage{stage}_{tag}.pt"
    out_txt = RES / f"stage{stage}_{tag}_report.txt"
    out_json = RES / f"stage{stage}_{tag}_metrics.json"

    log(f"\n======== Stage {stage} {tag} mode={mode} ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    all_lines = train + hold_seen + hold_rare + hold_story
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    all_words = sorted({w for ln in all_lines for w in ln["words"]})
    word_fps = F.normalize(
        torch.stack([stack.w(w).detach() for w in all_words], 0), dim=-1
    ).to(device)
    surf = all_words
    stoi = {s: i for i, s in enumerate(surf)}

    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    story_src = story_lines if len(story_lines) <= 60000 else _subsample(story_lines, 60000, 12)
    fat_pairs = annotate_roles(lines_to_pairs(fat_lines, stoi), fat_lines)
    story_pairs = annotate_roles(lines_to_pairs(story_src, stoi), story_src)
    if not fat_pairs:
        fat_pairs = story_pairs

    # unigram on train words for rare_long
    unigram = Counter()
    for ln in train:
        for w in ln["words"]:
            unigram[w] += 1

    def split_rel(pairs):
        rel, oth = [], []
        for ex in pairs:
            (rel if ex["role"] == "rel" else oth).append(ex)
        return rel, oth

    fat_rel, fat_oth = split_rel(fat_pairs)
    st_rel, st_oth = split_rel(story_pairs)
    nat_rel_frac = (len(fat_rel) + len(st_rel)) / max(len(fat_pairs) + len(story_pairs), 1)
    log(
        f"nat_rel_frac={nat_rel_frac:.3f} target_rel_frac={nat_rel_frac * rel_rate:.3f} "
        f"fat_rel={len(fat_rel)} st_rel={len(st_rel)}"
    )

    # rare_long weights on story+fat other/all
    def rare_long_weight(ex) -> float:
        w = ex["target_word"]
        return max(len(w), 1) / float(unigram.get(w, 0) + 1)

    if mode == "rare_long":
        all_pairs = fat_pairs + story_pairs
        raw_w = [rare_long_weight(ex) for ex in all_pairs]
        s = sum(raw_w) or 1.0
        probs = [x / s for x in raw_w]
        # alias-lite: cumulative
        cdf = []
        acc = 0.0
        for p in probs:
            acc += p
            cdf.append(acc)
        log(f"rare_long mean_w={s/len(raw_w):.4f} n_pairs={len(all_pairs)}")

    ev_seen = _subsample(hold_seen, 600, 1001)
    ev_rare = _subsample(hold_rare, 120, 1002)
    ev_story = _subsample(hold_story, 400, 1003)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1200), 2001)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), 160), 2002)
    fin_story = _subsample(hold_story, min(len(hold_story), 800), 2003)

    model = WordIdTransformer(
        n_vocab=len(surf), d_model=256, n_heads=4, n_layers=2, max_len=16, dropout=0.1
    ).to(device)
    model.init_from_fps(word_fps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params={n_params/1e6:.2f}M V={len(surf)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(272)
    batch = 8
    warmup = 200
    lr = 1e-3
    # diagnostics: fraction of rel in batches
    rel_in_batch = []

    def _set_lr(step):
        cur = _warmup_then_constant(step, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur

    def _pick(pool_rel, pool_oth, n, want_rel_frac):
        """Draw n examples with approximately want_rel_frac from rel pool."""
        n_rel = int(round(n * want_rel_frac)) if pool_rel else 0
        n_rel = min(n_rel, n)
        n_oth = n - n_rel
        out = []
        if pool_rel and n_rel:
            out += [rng.choice(pool_rel) for _ in range(n_rel)]
        src_oth = pool_oth if pool_oth else pool_rel
        out += [rng.choice(src_oth) for _ in range(n_oth)]
        rng.shuffle(out)
        return out

    def _sample_batch():
        n_fat = max(1, int(round(batch * fat_frac))) if fat_pairs else 0
        n_fat = min(n_fat, batch)
        n_st = batch - n_fat
        if mode == "rel_under":
            want = nat_rel_frac * rel_rate
            ex = _pick(fat_rel, fat_oth, n_fat, want) if n_fat else []
            ex += _pick(st_rel, st_oth, n_st, want) if n_st else []
        else:
            # rare_long: ignore fat/story split for target bias; still mix fat_frac of fat indices
            ex = []
            for _ in range(batch):
                u = rng.random()
                # binary search cdf
                lo, hi = 0, len(cdf) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if u <= cdf[mid]:
                        hi = mid
                    else:
                        lo = mid + 1
                ex.append(all_pairs[lo])
            # optional: force some fat presence
            if fat_pairs and fat_frac > 0:
                n_swap = max(1, int(round(batch * fat_frac * 0.5)))
                for i in range(n_swap):
                    ex[i] = rng.choice(fat_pairs)
                rng.shuffle(ex)
        rel_in_batch.append(sum(1 for e in ex if e.get("role") == "rel") / max(len(ex), 1))
        return ex

    def _eval(lines):
        return eval_id_capacity_suite(model, lines, train, surf, stoi, device)

    best = {
        "step": 0, "obj": 0.0, "rel": 0.0, "story_all": 0.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        seen, rare, story = _eval(ev_seen), _eval(ev_rare), _eval(ev_story)
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        rb = sum(rel_in_batch[-eval_every:]) / max(len(rel_in_batch[-eval_every:]), 1) if rel_in_batch else 0.0
        curve.append({
            "step": step, "obj": obj_s, "rel": rel_s, "story_all": st_all,
            "rare_obj": rare["obj"]["hit1"], "batch_rel_frac": rb,
        })
        key = (1 if rel_s >= 0.70 else 0, obj_s, st_all, rel_s)
        bkey = (1 if best["rel"] >= 0.70 else 0, best["obj"], best["story_all"], best["rel"])
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, rel=rel_s, story_all=st_all,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        log(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% | "
            f"STORY ALL={st_all*100:.1f}% | batch_rel~{rb*100:.1f}%"
        )
        model.train()

    log(f"=== FT sampling mode={mode} fat_frac={fat_frac} ===")
    snap(0)
    model.train()
    for step in range(1, ft_steps + 1):
        _set_lr(step)
        packed = collate_word_id_batch(_sample_batch(), stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, pad_mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == ft_steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen, f_rare, f_story = _eval(fin_seen), _eval(fin_rare), _eval(fin_story)
    obj_s = f_seen["obj"]["hit1"]
    rel_s = f_seen["roles"].get("rel", {}).get("hit1", 0.0)
    st_all = f_story["roles"].get("ALL", {}).get("hit1", 0.0)
    ref = ref or REF100
    obj_lift = obj_s - ref["obj"]
    st_lift = st_all - ref["story_all"]
    mean_batch_rel = sum(rel_in_batch) / max(len(rel_in_batch), 1)

    if rel_s >= 0.70 and obj_lift >= 0.03:
        verdict = "PASS_OBJ"
    elif rel_s >= 0.70 and (obj_lift >= 0.015 or st_lift >= 0.02):
        verdict = "PARTIAL"
    elif rel_s < 0.70:
        verdict = "FAIL_REL"
    else:
        verdict = "PARITY"

    report = "\n".join([
        f"SOTE Stage {stage} — {tag} (sample composition, not CE weight)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"mode={mode} rel_rate={rel_rate} nat_rel_frac={nat_rel_frac:.3f} "
        f"mean_batch_rel={mean_batch_rel:.3f} fat_frac={fat_frac}",
        f"best_step={best['step']} params={n_params}",
        f"SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}%",
        f"STORY ALL={st_all*100:.1f}% obj={f_story['obj']['hit1']*100:.1f}%",
        f"RARE obj={f_rare['obj']['hit1']*100:.1f}%",
        f"vs100: obj {obj_lift*100:+.1f}pp story {st_lift*100:+.1f}pp",
        f"Verdict: {verdict}",
        "Contrast 107/108: those reweighted CE; this changes batch mixture.",
    ]) + "\n"
    log("\n" + report)
    out_txt.write_text(report, encoding="utf-8")
    (RES / f"stage{stage}_{tag}_{verdict}.txt").write_text(report, encoding="utf-8")
    result = {
        "stage": stage, "tag": tag, "verdict": verdict, "mode": mode,
        "nat_rel_frac": nat_rel_frac, "rel_rate": rel_rate,
        "mean_batch_rel": mean_batch_rel,
        "seen": {"obj": obj_s, "rel": rel_s},
        "story": {"all": st_all, "obj": f_story["obj"]["hit1"]},
        "rare": {"obj": f_rare["obj"]["hit1"]},
        "obj_lift_pp": obj_lift, "story_lift_pp": st_lift,
        "curve": curve, "ckpt": str(out_ckpt), "best_step": best["step"],
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({
        "stage": stage, "tag": tag, "verdict": verdict, "mode": mode,
        "word_tf": best["state"], "surfaces": surf, "cfg": asdict(cfg),
    }, out_ckpt)
    log(f"Saved {out_ckpt}")
    return result


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage113/114 start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_prior()
        phrases = ensure_100k()
        r113 = train_sample_run(
            stage=113, tag="rel_undersample", phrases=phrases,
            mode="rel_under", rel_rate=0.5, ft_steps=50000, fat_frac=0.75,
            ref=REF100,
        )
        r114 = train_sample_run(
            stage=114, tag="rare_long_sample", phrases=phrases,
            mode="rare_long", rel_rate=1.0, ft_steps=50000, fat_frac=0.75,
            ref=REF100,
        )
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prior_111_112": {
                "111": prior.get("stage111", {}).get("verdict") if isinstance(prior.get("stage111"), dict) else None,
                "112": prior.get("stage112", {}).get("verdict") if isinstance(prior.get("stage112"), dict) else None,
            },
            "hypothesis": (
                "107/108 CE reweight failed because rel already saturated; "
                "undersampling changes which gradients dominate capacity"
            ),
            "stage113": r113,
            "stage114": r114,
            "ref100": REF100,
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stages 113–114 (sample composition):** rel@0.5× → obj "
                f"{r113['seen']['obj']*100:.1f}% ({r113['verdict']}); "
                f"rare_long → obj {r114['seen']['obj']*100:.1f}% ({r114['verdict']}). "
                f"`stage113_114_decision.json`.\n"
            )
            if "Stages 113–114" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 113/114")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
