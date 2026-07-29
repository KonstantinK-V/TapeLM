"""
Stage 107–108: SOTE word-level loss reweight digs (after followups 103–106).

Hypothesis (user):
  rel (on|to) hits 95%+ by ~10–15k and plateaus — easy/frequent targets dominate
  gradients; obj stays hard. BPE suffers similarly but can't tag word roles; SOTE can
  downweight function/path-rel tokens and upweight rare/long words via CE class weights.

  Stage107: class-weight CE on 500k / 2L4H / fat0.75
    w(target) = len_band * inv_freq * rel_down
      len: 1–2 -> 0.8; 3–5 -> 1.0; 6+ -> 1.2
      inv_freq: clip(sqrt(med/count), 0.5, 2.0)
      rel_down: on|to -> 0.35 else 1.0
  Stage108: stronger rel down only (on|to -> 0.15), len/freq = 1 — isolate knob

Gate vs Stage101 SEEN obj 40.7% / STORY 20.6%:
  PASS_OBJ if SEEN obj +4pp & rel still >=70% (don't kill path)
  PASS_BAL if STORY ALL +3pp with SEEN obj held (±2pp)

Run:
  python _overnight_reweight.py
"""
from __future__ import annotations

import json
import math
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
    RELS,
    WordIdTransformer,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
    _warmup_then_constant,
)
from _overnight_atom_scale import (  # noqa: E402
    CORPUS_500K,
    ensure_corpus,
)

FOLLOW = RES / "overnight_followups_decision.json"
# if followups not written yet, also accept overnight done
OVERNIGHT = RES / "overnight_atom_scale_decision.json"
LOG = RES / "_overnight_reweight_log.txt"
OUT_DEC = RES / "overnight_reweight_decision.json"

REF101 = {"obj": 0.407, "rel": 0.971, "story_all": 0.206, "rare_obj": 0.138}


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s: int = 10 * 3600, poll_s: int = 45) -> dict:
    """Prefer followups done; else if overnight done and followups absent > long, start anyway after overnight."""
    log(f"[wait] prefer {FOLLOW.name} (fallback {OVERNIGHT.name})")
    t0 = time.time()
    overnight_seen = False
    while time.time() - t0 < timeout_s:
        if FOLLOW.exists():
            d = json.loads(FOLLOW.read_text(encoding="utf-8"))
            log("[wait] followups done")
            return {"source": "followups", **d}
        if OVERNIGHT.exists():
            overnight_seen = True
            # if followups never started / crashed, don't block forever: wait 30min after overnight
            age = time.time() - OVERNIGHT.stat().st_mtime
            # also check followups process indirectly: if overnight older than 2h and no follow
            if age > 2 * 3600 and not FOLLOW.exists():
                log("[wait] overnight stale >2h without followups; proceed")
                return {"source": "overnight_fallback", **json.loads(OVERNIGHT.read_text(encoding="utf-8"))}
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s overnight_seen={overnight_seen}")
    raise TimeoutError("prior chain not ready")


def _subsample(lines, n, seed):
    if not lines or n <= 0 or len(lines) <= n:
        return list(lines)
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    return [lines[i] for i in idx[:n]]


def _pair_pool(lines, stoi, cap, seed):
    if not lines:
        return []
    if len(lines) > cap:
        lines = _subsample(lines, cap, seed)
    return lines_to_pairs(lines, stoi)


def corpus_rel_stats(phrases: list[str]) -> dict:
    n = len(phrases)
    n_on = n_to = n_either = 0
    tgt_rel = 0
    tgt_n = 0
    for p in phrases:
        ws = p.split()
        has_on = "on" in ws
        has_to = "to" in ws
        n_on += int(has_on)
        n_to += int(has_to)
        n_either += int(has_on or has_to)
        for i in range(1, len(ws)):
            tgt_n += 1
            if ws[i] in RELS:
                tgt_rel += 1
    return {
        "n_phrases": n,
        "frac_has_on": n_on / max(n, 1),
        "frac_has_to": n_to / max(n, 1),
        "frac_has_on_or_to": n_either / max(n, 1),
        "frac_targets_are_rel": tgt_rel / max(tgt_n, 1),
        "n_next_targets": tgt_n,
    }


def build_target_weights(
    surf: list[str],
    unigram: Counter,
    *,
    mode: str,
    rel_w: float,
) -> torch.Tensor:
    """Per-class CE weights indexed by vocab id."""
    counts = [max(unigram.get(w, 1), 1) for w in surf]
    med = sorted(counts)[len(counts) // 2]
    ws = []
    for w, c in zip(surf, counts):
        if mode == "full":
            L = len(w)
            if L <= 2:
                lw = 0.8
            elif L <= 5:
                lw = 1.0
            else:
                lw = 1.2
            fw = math.sqrt(med / float(c))
            fw = min(2.0, max(0.5, fw))
            rw = rel_w if w in RELS else 1.0
            ws.append(lw * fw * rw)
        elif mode == "rel_only":
            ws.append(rel_w if w in RELS else 1.0)
        else:
            ws.append(1.0)
    return torch.tensor(ws, dtype=torch.float32)


def train_reweight(
    *,
    stage: int,
    tag: str,
    phrases: list[str],
    mode: str,
    rel_w: float,
    ft_steps: int = 60000,
    eval_every: int = 2000,
    fat_frac: float = 0.75,
    ref: dict | None = None,
) -> dict:
    cfg = Config()
    cfg.c87_n_fat = 200
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = 40
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = 0.40
    cfg.c87_hold_frac = 0.12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent85)

    out_ckpt = CKPT / f"stage{stage}_{tag}.pt"
    out_txt = RES / f"stage{stage}_{tag}_report.txt"
    out_json = RES / f"stage{stage}_{tag}_metrics.json"

    log(f"\n======== Stage {stage} {tag} mode={mode} rel_w={rel_w} ========")
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

    unigram = Counter()
    for ln in train:
        for w in ln["words"]:
            unigram[w] += 1
    class_w = build_target_weights(surf, unigram, mode=mode, rel_w=rel_w).to(device)
    log(f"V={len(surf)} weight_mean={float(class_w.mean()):.3f}")
    if "on" in stoi and "to" in stoi:
        log(f"  w(on)={float(class_w[stoi['on']]):.3f} w(to)={float(class_w[stoi['to']]):.3f}")

    ev_seen = _subsample(hold_seen, 600, 1001)
    ev_rare = _subsample(hold_rare, 120, 1002)
    ev_story = _subsample(hold_story, 400, 1003)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1200), 2001)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), 160), 2002)
    fin_story = _subsample(hold_story, min(len(hold_story), 800), 2003)

    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    fat_pairs = _pair_pool(fat_lines, stoi, len(fat_lines), 11)
    story_pairs = _pair_pool(story_lines, stoi, 60000, 12)
    if not fat_pairs:
        fat_pairs = story_pairs

    model = WordIdTransformer(
        n_vocab=len(surf), d_model=256, n_heads=4, n_layers=2, max_len=16, dropout=0.1
    ).to(device)
    model.init_from_fps(word_fps)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(272)
    batch = 8
    warmup = 200
    lr = 1e-3

    def _set_lr(step):
        cur = _warmup_then_constant(step, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur

    def _sample_batch():
        n_fat = max(1, int(round(batch * fat_frac)))
        n_fat = min(n_fat, batch)
        ex = [rng.choice(fat_pairs) for _ in range(n_fat)]
        ex += [rng.choice(story_pairs) for _ in range(batch - n_fat)]
        rng.shuffle(ex)
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
        curve.append({"step": step, "obj": obj_s, "rel": rel_s, "story_all": st_all,
                      "rare_obj": rare["obj"]["hit1"]})
        # prefer joint: high obj with rel not collapsed
        key = (1 if rel_s >= 0.70 else 0, obj_s, st_all, rel_s)
        bkey = (1 if best["rel"] >= 0.70 else 0, best["obj"], best["story_all"], best["rel"])
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, rel=rel_s, story_all=st_all,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        log(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% | "
            f"STORY ALL={st_all*100:.1f}% | RARE obj={rare['obj']['hit1']*100:.1f}%"
        )
        model.train()

    log("=== FT weighted CE ===")
    snap(0)
    model.train()
    for step in range(1, ft_steps + 1):
        _set_lr(step)
        packed = collate_word_id_batch(_sample_batch(), stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        logits = model.logits_last_from_batch(ids, pad_mask)
        loss = F.cross_entropy(logits, tgt, weight=class_w)
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
    ref = ref or REF101
    obj_lift = obj_s - ref["obj"]
    st_lift = st_all - ref["story_all"]
    rel_ok = rel_s >= 0.70

    if rel_ok and obj_lift >= 0.04:
        verdict = "PASS_OBJ"
    elif rel_ok and st_lift >= 0.03 and abs(obj_lift) <= 0.02:
        verdict = "PASS_BAL"
    elif rel_ok and (obj_lift >= 0.02 or st_lift >= 0.02):
        verdict = "PARTIAL"
    elif not rel_ok:
        verdict = "FAIL_REL"
    else:
        verdict = "PARITY"

    report = "\n".join([
        f"SOTE Stage {stage} — {tag} (word-level CE reweight)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"mode={mode} rel_w={rel_w} fat_frac={fat_frac} steps={ft_steps}",
        f"RELS={list(RELS)}",
        f"best_step={best['step']}",
        f"SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}%",
        f"STORY ALL={st_all*100:.1f}% obj={f_story['obj']['hit1']*100:.1f}%",
        f"RARE obj={f_rare['obj']['hit1']*100:.1f}%",
        f"vs101: obj {obj_lift*100:+.1f}pp story {st_lift*100:+.1f}pp",
        f"Verdict: {verdict}",
        "Hypothesis: downweight easy rel targets frees gradient for obj/rare.",
    ]) + "\n"
    log("\n" + report)
    out_txt.write_text(report, encoding="utf-8")
    (RES / f"stage{stage}_{tag}_{verdict}.txt").write_text(report, encoding="utf-8")
    result = {
        "stage": stage, "tag": tag, "verdict": verdict, "mode": mode, "rel_w": rel_w,
        "seen": {"obj": obj_s, "rel": rel_s},
        "story": {"all": st_all, "obj": f_story["obj"]["hit1"]},
        "rare": {"obj": f_rare["obj"]["hit1"]},
        "obj_lift_pp": obj_lift, "story_lift_pp": st_lift,
        "curve": curve, "ckpt": str(out_ckpt), "best_step": best["step"],
        "weight_on": float(class_w[stoi["on"]]) if "on" in stoi else None,
        "weight_to": float(class_w[stoi["to"]]) if "to" in stoi else None,
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({
        "stage": stage, "tag": tag, "verdict": verdict, "word_tf": best["state"],
        "surfaces": surf, "class_weights": class_w.detach().cpu(),
        "mode": mode, "rel_w": rel_w, "cfg": asdict(cfg),
    }, out_ckpt)
    log(f"Saved {out_ckpt}")
    return result


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Reweight digs start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_prior()
        phrases = ensure_corpus(CORPUS_500K, 500_000, seed=272)
        stats = corpus_rel_stats(phrases)
        log(f"[stats] corpus on|to: {json.dumps(stats)}")

        r107 = train_reweight(
            stage=107, tag="ce_reweight_full", phrases=phrases,
            mode="full", rel_w=0.35, ft_steps=60000, ref=REF101,
        )
        r108 = train_reweight(
            stage=108, tag="ce_reweight_rel_only", phrases=phrases,
            mode="rel_only", rel_w=0.15, ft_steps=60000, ref=REF101,
        )

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prior": prior.get("source"),
            "corpus_stats": stats,
            "hypothesis": (
                "rel saturates early and steals gradient; SOTE can reweight word targets "
                "(len × inv-freq × on|to down) unlike opaque BPE pieces"
            ),
            "stage107": r107,
            "stage108": r108,
            "ref101": REF101,
        }
        OUT_DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stages 107–108 (CE reweight dig):** corpus on|to in "
                f"{stats['frac_has_on_or_to']*100:.0f}% windows; "
                f"107 full w(on)={r107.get('weight_on')} → SEEN obj "
                f"{r107['seen']['obj']*100:.1f}% ({r107['verdict']}); "
                f"108 rel_only w={r108['rel_w']} → obj {r108['seen']['obj']*100:.1f}% "
                f"({r108['verdict']}). `overnight_reweight_decision.json`.\n"
            )
            if "Stages 107–108" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE reweight digs")
        log(json.dumps(out, indent=2, default=str))
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
