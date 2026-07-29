"""
Stage 109–110: slot-weight CE (dynamic) + standard TF baseline.

109 slot_dyn: recipe98 on ~100k (Stage100 corpus), SOTE fp init.
  Per-example CE weight by _role(target):
    rel  -> w_rel (starts 0.25; decays toward 0.08 if eval rel>=0.90)
    right/obj -> w_obj (starts 2.0; rises to 3.0 if obj plateau)
    other/verb_ing/leftish -> 1.0
  Dynamic update every eval_every from SEEN hold metrics.

110 tf_baseline: same data/arch/steps, RANDOM emb (no init_from_fps).
  Isolates: is SOTE fp-init doing work vs plain word-id Transformer?

Gate vs Stage100 (obj 42.6% / rel 95.8% / STORY 19.6%):
  109 PASS if SEEN obj +3pp with rel>=0.70
  110 report parity/gap vs 100 (baseline expected weaker or similar)

Run:
  python _stage109_110_slot_baseline.py
"""
from __future__ import annotations

import json
import random
import sys
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
    filter_tinystories_chunk,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)

CORPUS_100K = ROOT / "data" / "external_tinystories_100k_85.txt"
RAW_SCALE = ROOT / "data" / "_tinystories_raw_scale.txt"
RAW_100 = ROOT / "data" / "_tinystories_raw_100k.txt"
LOG = RES / "_stage109_110_log.txt"
DEC = RES / "stage109_110_decision.json"

REF100 = {"obj": 0.426, "rel": 0.958, "story_all": 0.196, "rare_obj": 0.117}


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


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


def ensure_100k() -> list[str]:
    if CORPUS_100K.exists():
        phrases = [
            ln.strip()
            for ln in CORPUS_100K.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if len(phrases) >= 80000:
            log(f"[data] reuse {CORPUS_100K.name} n={len(phrases)}")
            return phrases
    cfg = Config()
    raw = RAW_SCALE if RAW_SCALE.exists() else RAW_100
    assert raw.exists(), f"need raw at {raw}"
    phrases, meta = filter_tinystories_chunk(
        raw, CORPUS_100K, max_lines=100000, max_word_len=cfg.max_word_len, seed=272
    )
    log(f"[data] filtered n={len(phrases)} meta={meta}")
    return phrases


def annotate_roles(pairs, lines):
    for ex in pairs:
        ws = lines[ex["line_i"]]["words"]
        ex["role"] = _role(ex, ws)
    return pairs


def train_run(
    *,
    stage: int,
    tag: str,
    phrases: list[str],
    use_sote_init: bool,
    slot_dynamic: bool,
    ft_steps: int = 50000,
    eval_every: int = 1000,
    fat_frac: float = 0.75,
    w_rel0: float = 0.25,
    w_obj0: float = 2.0,
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

    log(f"\n======== Stage {stage} {tag} sote_init={use_sote_init} slot_dyn={slot_dynamic} ========")
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

    # map pair index pools with roles — need line lists aligned
    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    # build pairs from capped story; roles need correct line_i into that list
    fat_src = fat_lines
    story_src = story_lines if len(story_lines) <= 60000 else _subsample(story_lines, 60000, 12)
    fat_pairs = annotate_roles(lines_to_pairs(fat_src, stoi), fat_src)
    story_pairs = annotate_roles(lines_to_pairs(story_src, stoi), story_src)
    if not fat_pairs:
        fat_pairs = story_pairs

    role_counts = Counter(ex["role"] for ex in fat_pairs + story_pairs)
    log(f"mix={ {k: meta[k] for k in meta if k != 'top_triple_freq'} }")
    log(f"role_counts_train_pool={dict(role_counts)}")

    ev_seen = _subsample(hold_seen, 600, 1001)
    ev_rare = _subsample(hold_rare, 120, 1002)
    ev_story = _subsample(hold_story, 400, 1003)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1200), 2001)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), 160), 2002)
    fin_story = _subsample(hold_story, min(len(hold_story), 800), 2003)

    model = WordIdTransformer(
        n_vocab=len(surf), d_model=256, n_heads=4, n_layers=2, max_len=16, dropout=0.1
    ).to(device)
    if use_sote_init:
        model.init_from_fps(word_fps)
    else:
        nn_init = torch.nn.init
        nn_init.normal_(model.tok.weight, std=0.02)
        with torch.no_grad():
            model.tok.weight[model.pad_id].zero_()
        log("RANDOM emb init (standard TF baseline)")

    n_params = sum(p.numel() for p in model.parameters())
    log(f"params={n_params/1e6:.2f}M V={len(surf)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(272)
    batch = 8
    warmup = 200
    lr = 1e-3

    # mutable slot weights
    w_rel = w_rel0
    w_obj = w_obj0
    slot_hist = []
    obj_hist = []

    def slot_w(role: str) -> float:
        if role == "rel":
            return w_rel
        if role == "right":
            return w_obj
        return 1.0

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
        nonlocal w_rel, w_obj
        model.eval()
        seen, rare, story = _eval(ev_seen), _eval(ev_rare), _eval(ev_story)
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({
            "step": step, "obj": obj_s, "rel": rel_s, "story_all": st_all,
            "rare_obj": rare["obj"]["hit1"], "w_rel": w_rel, "w_obj": w_obj,
        })
        if slot_dynamic and step > 0:
            # throttle rel when saturated
            if rel_s >= 0.90:
                w_rel = max(0.08, w_rel * 0.85)
            elif rel_s < 0.50:
                w_rel = min(w_rel0, w_rel * 1.05)
            # boost obj if plateau (last 3 evals within 1pp) and rel high
            obj_hist.append(obj_s)
            if len(obj_hist) >= 3 and rel_s >= 0.85:
                if max(obj_hist[-3:]) - min(obj_hist[-3:]) < 0.015:
                    w_obj = min(3.5, w_obj * 1.08)
            slot_hist.append({"step": step, "w_rel": w_rel, "w_obj": w_obj, "rel": rel_s, "obj": obj_s})
            log(f"  [dyn] w_rel={w_rel:.3f} w_obj={w_obj:.3f}")

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

    log(f"=== FT slot_w0 rel={w_rel0} obj={w_obj0} ===")
    snap(0)
    model.train()
    for step in range(1, ft_steps + 1):
        _set_lr(step)
        exs = _sample_batch()
        packed = collate_word_id_batch(exs, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        logits = model.logits_last_from_batch(ids, pad_mask)
        # per-example weights by role (aligned with collate order — collate may drop bad ex)
        # rebuild weights from successfully packed: re-filter like collate
        weights = []
        for ex in exs:
            if ex["target_word"] not in stoi:
                continue
            if not ex["prefix_word_ids"]:
                continue
            weights.append(slot_w(ex.get("role", "other")))
        if len(weights) != tgt.shape[0]:
            # fallback uniform if mismatch
            w = torch.ones(tgt.shape[0], device=device)
        else:
            w = torch.tensor(weights, dtype=torch.float32, device=device)
        per = F.cross_entropy(logits, tgt, reduction="none")
        loss = (per * w).sum() / w.sum().clamp_min(1e-6)
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

    if use_sote_init and slot_dynamic:
        if rel_s >= 0.70 and obj_lift >= 0.03:
            verdict = "PASS_OBJ"
        elif rel_s >= 0.70 and obj_lift >= 0.015:
            verdict = "PARTIAL"
        elif rel_s < 0.70:
            verdict = "FAIL_REL"
        else:
            verdict = "PARITY"
    else:
        # baseline: just compare
        if abs(obj_lift) < 0.02 and abs(st_lift) < 0.02:
            verdict = "PARITY_BASE"
        elif obj_s + 0.02 < ref["obj"]:
            verdict = "WEAKER_BASE"
        elif obj_lift >= 0.02:
            verdict = "STRONGER_BASE"
        else:
            verdict = "MIXED_BASE"

    report = "\n".join([
        f"SOTE Stage {stage} — {tag}",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"sote_init={use_sote_init} slot_dynamic={slot_dynamic}",
        f"w_rel0={w_rel0} w_obj0={w_obj0} final_w_rel={w_rel:.3f} final_w_obj={w_obj:.3f}",
        f"params={n_params} V={len(surf)} steps={ft_steps}",
        f"best_step={best['step']}",
        f"SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}%",
        f"STORY ALL={st_all*100:.1f}% obj={f_story['obj']['hit1']*100:.1f}%",
        f"RARE obj={f_rare['obj']['hit1']*100:.1f}%",
        f"vs100: obj {obj_lift*100:+.1f}pp story {st_lift*100:+.1f}pp",
        f"Verdict: {verdict}",
    ]) + "\n"
    log("\n" + report)
    out_txt.write_text(report, encoding="utf-8")
    (RES / f"stage{stage}_{tag}_{verdict}.txt").write_text(report, encoding="utf-8")
    result = {
        "stage": stage, "tag": tag, "verdict": verdict,
        "sote_init": use_sote_init, "slot_dynamic": slot_dynamic,
        "seen": {"obj": obj_s, "rel": rel_s},
        "story": {"all": st_all, "obj": f_story["obj"]["hit1"]},
        "rare": {"obj": f_rare["obj"]["hit1"]},
        "obj_lift_pp": obj_lift, "story_lift_pp": st_lift,
        "final_w_rel": w_rel, "final_w_obj": w_obj,
        "curve": curve, "slot_hist": slot_hist,
        "ckpt": str(out_ckpt), "best_step": best["step"], "params": n_params,
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({
        "stage": stage, "tag": tag, "verdict": verdict,
        "word_tf": best["state"], "surfaces": surf,
        "sote_init": use_sote_init, "slot_dynamic": slot_dynamic,
        "cfg": asdict(cfg),
    }, out_ckpt)
    log(f"Saved {out_ckpt}")
    return result


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage109/110 start {datetime.now(timezone.utc).isoformat()}")
    try:
        phrases = ensure_100k()
        r109 = train_run(
            stage=109, tag="slot_dyn_obj", phrases=phrases,
            use_sote_init=True, slot_dynamic=True,
            ft_steps=50000, eval_every=1000,
            w_rel0=0.25, w_obj0=2.0, ref=REF100,
        )
        r110 = train_run(
            stage=110, tag="tf_baseline_randemb", phrases=phrases,
            use_sote_init=False, slot_dynamic=False,
            ft_steps=50000, eval_every=1000,
            w_rel0=1.0, w_obj0=1.0, ref=REF100,
        )
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ref100": REF100,
            "stage109": r109,
            "stage110": r110,
            "note": (
                "109=dynamic slot CE (throttle rel, boost right/obj); "
                "110=standard word-id TF random emb baseline"
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stages 109–110:** slot-dyn CE → SEEN obj "
                f"{r109['seen']['obj']*100:.1f}% ({r109['verdict']}); "
                f"TF baseline (rand emb) obj {r110['seen']['obj']*100:.1f}% "
                f"({r110['verdict']}). `stage109_110_decision.json`.\n"
            )
            if "Stages 109–110" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 109/110")
        log(json.dumps({
            "109": {"verdict": r109["verdict"], "obj": r109["seen"]["obj"], "rel": r109["seen"]["rel"]},
            "110": {"verdict": r110["verdict"], "obj": r110["seen"]["obj"], "rel": r110["seen"]["rel"]},
        }, indent=2))
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
