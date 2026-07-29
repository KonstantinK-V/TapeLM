"""
Stage 115 after 113/114 — longer context dig.

Note: current TinyStories windows are max_words=8, so max_len=16 is already slack.
Raising max_len alone on 8-word data is a null dig. Real test = longer windows + matching max_len.

  115a long_win: re-filter ~100k with max_words=16, train WordIdTF max_len=24
       recipe98 (batch8, warmup200, lr1e-3, fat0.75), SOTE fp init.
  115b max_len_only (control): same 8-word corpus, max_len=32 — expect PARITY
       (isolates "just bump max_len" myth).

Gate vs Stage100 (obj 42.6% / STORY 19.6%):
  PASS if STORY ALL +3pp or SEEN obj +3pp with rel>=0.70

Run (waits for stage113_114_decision.json):
  python _stage115_context_follow.py
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

from train import (  # noqa: E402
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    _warmup_then_constant,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    filter_tinystories_chunk,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, _subsample  # noqa: E402

DEC_PRIOR = RES / "stage113_114_decision.json"
LOG = RES / "_stage115_context_log.txt"
DEC = RES / "stage115_context_decision.json"
CORPUS_8 = ROOT / "data" / "external_tinystories_100k_85.txt"
CORPUS_16 = ROOT / "data" / "external_tinystories_100k_w16.txt"
RAW_SCALE = ROOT / "data" / "_tinystories_raw_scale.txt"
RAW_100 = ROOT / "data" / "_tinystories_raw_100k.txt"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s: int = 14 * 3600, poll_s: int = 45) -> dict:
    log(f"[wait] for {DEC_PRIOR} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PRIOR.exists():
            d = json.loads(DEC_PRIOR.read_text(encoding="utf-8"))
            if d.get("stage113") is not None and d.get("stage114") is not None:
                log("[wait] 113/114 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("113/114 not ready")


def load_phrases(path: Path, max_lines: int, max_words: int, seed: int = 272) -> list[str]:
    if path.exists():
        phrases = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if len(phrases) >= int(max_lines * 0.8):
            log(f"[data] reuse {path.name} n={len(phrases)}")
            return phrases
    cfg = Config()
    raw = RAW_SCALE if RAW_SCALE.exists() else RAW_100
    assert raw.exists(), f"missing {raw}"
    log(f"[data] filter max_words={max_words} -> {path.name}")
    phrases, meta = filter_tinystories_chunk(
        raw, path,
        max_lines=max_lines,
        max_words=max_words,
        max_word_len=int(cfg.max_word_len),
        seed=seed,
    )
    log(f"[data] n={len(phrases)} meta={meta}")
    return phrases


def train_ctx(
    *,
    stage: int,
    tag: str,
    phrases: list[str],
    max_len: int,
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

    log(f"\n======== Stage {stage} {tag} max_len={max_len} n_phrases={len(phrases)} ========")
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
    fat_pairs = lines_to_pairs(fat_lines, stoi) if fat_lines else []
    story_pairs = lines_to_pairs(story_src, stoi)
    if not fat_pairs:
        fat_pairs = story_pairs

    # length stats
    pref_lens = [len(ex["prefix_word_ids"]) for ex in story_pairs[:5000]]
    log(
        f"mix={ {k: meta[k] for k in meta if k != 'top_triple_freq'} } "
        f"prefix_len mean={sum(pref_lens)/max(len(pref_lens),1):.2f} "
        f"max={max(pref_lens) if pref_lens else 0}"
    )

    ev_seen = _subsample(hold_seen, 600, 1001)
    ev_rare = _subsample(hold_rare, 120, 1002)
    ev_story = _subsample(hold_story, 400, 1003)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1200), 2001)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), 160), 2002)
    fin_story = _subsample(hold_story, min(len(hold_story), 800), 2003)

    model = WordIdTransformer(
        n_vocab=len(surf), d_model=256, n_heads=4, n_layers=2,
        max_len=max_len, dropout=0.1,
    ).to(device)
    model.init_from_fps(word_fps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params={n_params/1e6:.2f}M V={len(surf)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(272)
    batch = 8
    warmup = 200

    def _set_lr(step):
        cur = _warmup_then_constant(step, 1e-3, warmup)
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
        key = (1 if rel_s >= 0.70 else 0, obj_s, st_all, rel_s)
        bkey = (1 if best["rel"] >= 0.70 else 0, best["obj"], best["story_all"], best["rel"])
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, rel=rel_s, story_all=st_all,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        log(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% | "
            f"STORY ALL={st_all*100:.1f}%"
        )
        model.train()

    log("=== FT ===")
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

    if rel_s >= 0.70 and (obj_lift >= 0.03 or st_lift >= 0.03):
        verdict = "PASS"
    elif rel_s >= 0.70 and (obj_lift >= 0.015 or st_lift >= 0.015):
        verdict = "PARTIAL"
    elif rel_s < 0.70:
        verdict = "FAIL_REL"
    else:
        verdict = "PARITY"

    report = "\n".join([
        f"SOTE Stage {stage} — {tag}",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"max_len={max_len} n_phrases={len(phrases)} params={n_params}",
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
        "stage": stage, "tag": tag, "verdict": verdict, "max_len": max_len,
        "n_phrases": len(phrases), "params": n_params,
        "seen": {"obj": obj_s, "rel": rel_s},
        "story": {"all": st_all, "obj": f_story["obj"]["hit1"]},
        "rare": {"obj": f_rare["obj"]["hit1"]},
        "obj_lift_pp": obj_lift, "story_lift_pp": st_lift,
        "curve": curve, "ckpt": str(out_ckpt), "best_step": best["step"],
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({
        "stage": stage, "tag": tag, "verdict": verdict,
        "word_tf": best["state"], "surfaces": surf,
        "max_len": max_len, "cfg": asdict(cfg),
    }, out_ckpt)
    log(f"Saved {out_ckpt}")
    return result


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage115 context start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_prior()
        phrases16 = load_phrases(CORPUS_16, max_lines=100000, max_words=16, seed=275)
        phrases8 = load_phrases(CORPUS_8, max_lines=100000, max_words=8, seed=272)

        r115a = train_ctx(
            stage=115, tag="long_win16_maxlen24",
            phrases=phrases16, max_len=24,
            ft_steps=50000, ref=REF100,
        )
        # control: bump max_len only on short windows
        r115b = train_ctx(
            stage=115, tag="maxlen32_win8_ctrl",
            phrases=phrases8, max_len=32,
            ft_steps=40000, ref=REF100,
        )
        # mark control distinctly in nested keys
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": (
                "115a=longer windows(16)+max_len24; "
                "115b=control max_len32 on 8-word windows (expect null)"
            ),
            "prior_113_114": {
                "113": prior.get("stage113", {}).get("verdict") if isinstance(prior.get("stage113"), dict) else None,
                "114": prior.get("stage114", {}).get("verdict") if isinstance(prior.get("stage114"), dict) else None,
            },
            "stage115a": r115a,
            "stage115b_ctrl": r115b,
            "ref100": REF100,
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 115 (context):** win16/max24 → obj "
                f"{r115a['seen']['obj']*100:.1f}% STORY {r115a['story']['all']*100:.1f}% "
                f"({r115a['verdict']}); max32@win8 ctrl → obj "
                f"{r115b['seen']['obj']*100:.1f}% ({r115b['verdict']}). "
                f"`stage115_context_decision.json`.\n"
            )
            if "Stage 115 (context)" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 115")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
