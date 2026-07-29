"""
Overnight atom scale chain (no babysitting):

  Stage101: ~500k TinyStories, recipe98, same 3M TF (dim256/2L/4H)
  Decision vs Stage100:
    STORY lift >= +5pp & SEEN held  -> Stage102 data ~1M
    STORY flat (|lift|<3pp) & SEEN held -> Stage102 capacity A'
      (dim256 keep F85 init; n_layers 2->4, n_heads 4->8)
    else -> stop + report (SEEN collapse / ambiguous)

Run:
  python _overnight_atom_scale.py
"""
from __future__ import annotations

import json
import random
import sys
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
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    filter_tinystories_chunk,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
    _warmup_then_constant,
)

RAW_BIG = ROOT / "data" / "_tinystories_raw_scale.txt"
CORPUS_500K = ROOT / "data" / "external_tinystories_500k_85.txt"
CORPUS_1M = ROOT / "data" / "external_tinystories_1m_85.txt"
LOG = RES / "_overnight_atom_scale_log.txt"
DECISION = RES / "overnight_atom_scale_decision.json"

REF100 = {"obj": 0.426, "rel": 0.958, "story_all": 0.196, "rare_obj": 0.117}

# Decision thresholds
LIFT_GO_1M = 0.05
FLAT_ABS = 0.03
SEEN_OBJ_FLOOR = 0.30
SEEN_REL_FLOOR = 0.50


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def ensure_raw(char_budget: int = 120_000_000):
    if RAW_BIG.exists() and RAW_BIG.stat().st_size >= char_budget * 0.9:
        log(f"[data] reuse raw {RAW_BIG.name} size={RAW_BIG.stat().st_size}")
        return

    import urllib.request

    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStoriesV2-GPT4-train.txt"
    )
    log(f"[data] HTTP GET TinyStoriesV2-GPT4-train.txt -> {RAW_BIG.name} budget={char_budget}")
    req = urllib.request.Request(url, headers={"User-Agent": "sote-overnight/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        with RAW_BIG.open("wb") as f:
            got = 0
            while got < char_budget:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got % (20 * 1024 * 1024) < 1024 * 1024:
                    log(f"  wrote {got}")
    log(f"[data] raw ready size={RAW_BIG.stat().st_size}")


def ensure_corpus(path: Path, max_lines: int, seed: int = 272) -> list[str]:
    if path.exists():
        phrases = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if len(phrases) >= int(max_lines * 0.90):
            log(f"[data] reuse corpus {path.name} n={len(phrases)}")
            return phrases
        log(f"[data] corpus {path.name} too small n={len(phrases)}; rebuild")
    cfg = Config()
    assert RAW_BIG.exists(), f"missing {RAW_BIG}"
    log(f"[data] filter {RAW_BIG.name} -> {path.name} max_lines={max_lines}")
    phrases, meta = filter_tinystories_chunk(
        RAW_BIG,
        path,
        max_lines=max_lines,
        max_word_len=int(cfg.max_word_len),
        seed=seed,
    )
    log(f"[data] filtered n={len(phrases)} meta={meta}")
    return phrases


def _subsample(lines, n, seed):
    if not lines or n <= 0 or len(lines) <= n:
        return list(lines)
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    return [lines[i] for i in idx[:n]]


def _pair_pool(lines, stoi, cap: int, seed: int):
    """Build next-token pairs; cap source lines to bound memory."""
    if not lines:
        return []
    if len(lines) > cap:
        lines = _subsample(lines, cap, seed)
    return lines_to_pairs(lines, stoi)


def train_atom_scale(
    *,
    stage: int,
    tag: str,
    phrases: list[str],
    d_model: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    ft_steps: int = 80000,
    eval_every: int = 2000,
    batch: int = 8,
    lr: float = 1e-3,
    warmup: int = 200,
    fat_frac: float = 0.75,
    story_keep_frac: float = 0.40,
    hold_frac: float = 0.12,
    n_fat: int = 200,
    fat_copies: int = 40,
    story_pair_cap: int = 60000,
    ref: dict | None = None,
    story_lift_pass: float = 0.05,
) -> dict:
    cfg = Config()
    cfg.c87_n_fat = n_fat
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = fat_copies
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = story_keep_frac
    cfg.c87_hold_frac = hold_frac

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    stack, f85 = load_foundation_85(device, cfg, path=parent85)

    out_ckpt = CKPT / f"stage{stage}_{tag}.pt"
    out_txt = RES / f"stage{stage}_{tag}_report.txt"
    out_json = RES / f"stage{stage}_{tag}_metrics.json"

    log(f"\n======== Stage {stage} {tag} ========")
    log(f"device={device} foundation={parent85.name} dim_fp={f85.get('dim')}")
    log(
        f"arch d={d_model} L={n_layers} H={n_heads} steps={ft_steps} "
        f"batch={batch} lr={lr} warmup={warmup} fat_frac={fat_frac} n_phrases={len(phrases)}"
    )

    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    log(f"mix={ {k: meta[k] for k in meta if k != 'top_triple_freq'} }")

    all_lines = train + hold_seen + hold_rare + hold_story
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    all_words = sorted({w for ln in all_lines for w in ln["words"]})
    log(f"building fps V={len(all_words)} ...")
    # batch encode for speed
    word_fps_list = []
    bs = 256
    for i in range(0, len(all_words), bs):
        chunk = all_words[i : i + bs]
        word_fps_list.append(
            F.normalize(torch.stack([stack.w(w).detach() for w in chunk], 0), dim=-1)
        )
    word_fps = torch.cat(word_fps_list, 0).to(device)
    surf = all_words
    stoi = {s: i for i, s in enumerate(surf)}

    ev_seen = _subsample(hold_seen, 600, 1001)
    ev_rare = _subsample(hold_rare, 120, 1002)
    ev_story = _subsample(hold_story, 400, 1003)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1200), 2001)
    fin_rare = _subsample(hold_rare, min(len(hold_rare), 160), 2002)
    fin_story = _subsample(hold_story, min(len(hold_story), 800), 2003)

    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    fat_pairs = _pair_pool(fat_lines, stoi, cap=len(fat_lines), seed=11) if fat_lines else []
    story_pairs = _pair_pool(story_lines, stoi, cap=story_pair_cap, seed=12)
    if not fat_pairs:
        fat_pairs = story_pairs
    log(f"fat_pairs={len(fat_pairs)} story_pairs={len(story_pairs)} (story_cap={story_pair_cap})")

    model = WordIdTransformer(
        n_vocab=len(surf),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=16,
        dropout=0.1,
    ).to(device)
    model.init_from_fps(word_fps)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params={n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(272)

    def _set_lr(step: int) -> float:
        cur = _warmup_then_constant(step, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur
        return cur

    def _sample_batch():
        n_fat = max(1, int(round(batch * fat_frac))) if fat_pairs else 0
        n_fat = min(n_fat, batch)
        n_st = batch - n_fat
        ex = [rng.choice(fat_pairs) for _ in range(n_fat)]
        ex += [rng.choice(story_pairs) for _ in range(n_st)]
        rng.shuffle(ex)
        return ex

    def _eval_hold(lines):
        return eval_id_capacity_suite(model, lines, train, surf, stoi, device) if lines else None

    best = {
        "step": 0,
        "obj": 0.0,
        "rel": 0.0,
        "story_all": 0.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def _snap(step: int):
        model.eval()
        seen = _eval_hold(ev_seen)
        rare = _eval_hold(ev_rare)
        story = _eval_hold(ev_story)
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({
            "step": step,
            "obj_seen": obj_s,
            "rel_seen": rel_s,
            "story_all": st_all,
            "obj_rare": rare["obj"]["hit1"],
            "obj_story": story["obj"]["hit1"],
        })
        key = (1 if rel_s >= 0.50 else 0, st_all, obj_s, rel_s)
        bkey = (1 if best["rel"] >= 0.50 else 0, best["story_all"], best["obj"], best["rel"])
        if key >= bkey:
            best.update(
                step=step,
                obj=obj_s,
                rel=rel_s,
                story_all=st_all,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        log(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% | "
            f"STORY ALL={st_all*100:.1f}% obj={story['obj']['hit1']*100:.1f}% | "
            f"RARE obj={rare['obj']['hit1']*100:.1f}%"
        )
        model.train()

    log("=== FT ===")
    _snap(0)
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
            _snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = _eval_hold(fin_seen)
    f_rare = _eval_hold(fin_rare)
    f_story = _eval_hold(fin_story)
    obj_s = f_seen["obj"]["hit1"]
    rel_s = f_seen["roles"].get("rel", {}).get("hit1", 0.0)
    all_s = f_seen["roles"].get("ALL", {}).get("hit1", 0.0)
    st_all = f_story["roles"].get("ALL", {}).get("hit1", 0.0)
    st_obj = f_story["obj"]["hit1"]
    rare_obj = f_rare["obj"]["hit1"]

    ref = ref or REF100
    lift_story = st_all - float(ref["story_all"])
    seen_held = obj_s >= SEEN_OBJ_FLOOR and rel_s >= SEEN_REL_FLOOR

    if seen_held and lift_story >= story_lift_pass:
        verdict = "PASS_LIFT"
    elif seen_held and abs(lift_story) < FLAT_ABS:
        verdict = "CEILING"
    elif seen_held and lift_story >= 0.03:
        verdict = "PARTIAL_LIFT"
    elif seen_held:
        verdict = "PARITY"
    else:
        verdict = "FAIL"

    lines = [
        f"SOTE Stage {stage} — {tag}",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"device: {device}",
        f"foundation: {parent85.name}",
        f"n_phrases={len(phrases)} V={len(surf)} params={n_params}",
        f"arch: d={d_model} L={n_layers} H={n_heads}",
        f"train: batch={batch} steps={ft_steps} warmup={warmup} lr={lr} fat_frac={fat_frac}",
        f"mix: { {k: meta[k] for k in meta if k != 'top_triple_freq'} }",
        "NO soft@5. F85 FROZEN not overwritten.",
        "",
        "=== Final ===",
        f"  best_step={best['step']}",
        f"  SEEN  obj={obj_s*100:.1f}% rel={rel_s*100:.1f}% ALL={all_s*100:.1f}%",
        f"  RARE  obj={rare_obj*100:.1f}%",
        f"  STORY ALL={st_all*100:.1f}% obj={st_obj*100:.1f}%",
        "",
        f"=== vs ref {ref} ===",
        f"  STORY lift {lift_story*100:+.1f}pp  seen_held={seen_held}",
        "",
        f"=== Verdict: {verdict} ===",
    ]
    report = "\n".join(lines) + "\n"
    log("\n" + report)
    out_txt.write_text(report, encoding="utf-8")
    (RES / f"stage{stage}_{tag}_{verdict}.txt").write_text(report, encoding="utf-8")

    result = {
        "stage": stage,
        "tag": tag,
        "verdict": verdict,
        "best_step": best["step"],
        "n_phrases": len(phrases),
        "V": len(surf),
        "params": n_params,
        "arch": {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads},
        "seen": {"obj": obj_s, "rel": rel_s, "all": all_s},
        "rare": {"obj": rare_obj},
        "story": {"all": st_all, "obj": st_obj},
        "lift_story_pp": lift_story,
        "seen_held": seen_held,
        "ref": ref,
        "curve": curve,
        "ckpt": str(out_ckpt),
        "report": str(out_txt),
    }
    out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    torch.save(
        {
            "stage": stage,
            "tag": tag,
            "verdict": verdict,
            "word_tf": best["state"],
            "surfaces": surf,
            "word_fps": word_fps.detach().cpu(),
            "cfg": asdict(cfg),
            "arch": result["arch"],
            "metrics": {k: result[k] for k in ("seen", "rare", "story", "lift_story_pp", "verdict")},
        },
        out_ckpt,
    )
    log(f"Saved {out_ckpt}")
    return result


def decide_next(r101: dict) -> str:
    """Return '1m' | 'capacity' | 'stop'."""
    lift = float(r101["lift_story_pp"])
    held = bool(r101["seen_held"])
    if not held:
        return "stop"
    if lift >= LIFT_GO_1M:
        return "1m"
    if abs(lift) < FLAT_ABS:
        return "capacity"
    # weak lift 3-5pp: still try 1M once (data may need more)
    if lift >= 0.03:
        return "1m"
    return "capacity"


def update_docs(r101: dict, r102: dict | None, branch: str):
    replay = RES / "sote_v2_path_replay.md"
    contract = RES / "fp_language_contract.md"
    block101 = (
        f"\n**Stage 101 {r101['verdict']}:** ~{r101['n_phrases']//1000}k TinyStories "
        f"(recipe98) — SEEN obj **{r101['seen']['obj']*100:.1f}%** rel "
        f"**{r101['seen']['rel']*100:.1f}%**; STORY ALL **{r101['story']['all']*100:.1f}%** "
        f"(lift {r101['lift_story_pp']*100:+.1f}pp vs 100). "
        f"`stage101_{r101['tag']}_{r101['verdict']}.txt`.\n"
    )
    if replay.exists():
        txt = replay.read_text(encoding="utf-8")
        if "Stage 101" not in txt:
            txt = txt.replace(
                "**F85 dual-channel FREEZE:**",
                block101 + "\n**F85 dual-channel FREEZE:**",
            )
            replay.write_text(txt, encoding="utf-8")
    if r102 is not None and replay.exists():
        txt = replay.read_text(encoding="utf-8")
        block102 = (
            f"\n**Stage 102 {r102['verdict']} ({branch}):** "
            f"SEEN obj **{r102['seen']['obj']*100:.1f}%** rel "
            f"**{r102['seen']['rel']*100:.1f}%**; STORY ALL **{r102['story']['all']*100:.1f}%** "
            f"(lift {r102['lift_story_pp']*100:+.1f}pp vs ref). "
            f"`stage102_{r102['tag']}_{r102['verdict']}.txt`.\n"
        )
        if "Stage 102" not in txt:
            txt = txt.replace(
                "**F85 dual-channel FREEZE:**",
                block102 + "\n**F85 dual-channel FREEZE:**",
            )
            replay.write_text(txt, encoding="utf-8")
    if contract.exists():
        c = contract.read_text(encoding="utf-8")
        note = (
            f"- **Stage 101 {r101['verdict']}:** ~500k scale — STORY "
            f"{r101['story']['all']*100:.1f}% (lift {r101['lift_story_pp']*100:+.1f}pp vs 100); "
            f"branch→{branch}. `stage101_{r101['tag']}_{r101['verdict']}.txt`.\n"
        )
        if "Stage 101" not in c:
            c = c.replace(
                "Stage100 scale atom is a living dig",
                note + "Stage100 scale atom is a living dig",
            )
            if r102 is not None:
                note2 = (
                    f"- **Stage 102 {r102['verdict']} ({branch}):** STORY "
                    f"{r102['story']['all']*100:.1f}% "
                    f"(lift {r102['lift_story_pp']*100:+.1f}pp). "
                    f"`stage102_{r102['tag']}_{r102['verdict']}.txt`.\n"
                )
                c = c.replace(note, note + note2)
            contract.write_text(c, encoding="utf-8")


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Overnight atom scale start {datetime.now(timezone.utc).isoformat()}")

    try:
        ensure_raw(120_000_000)
        phrases_500 = ensure_corpus(CORPUS_500K, 500_000, seed=272)

        r101 = train_atom_scale(
            stage=101,
            tag="scale_500k",
            phrases=phrases_500,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ft_steps=80000,
            eval_every=2000,
            ref=REF100,
            story_lift_pass=LIFT_GO_1M,
        )

        branch = decide_next(r101)
        log(f"\n[decision] after 101: branch={branch} lift={r101['lift_story_pp']*100:+.1f}pp "
            f"seen_held={r101['seen_held']} verdict={r101['verdict']}")

        r102 = None
        if branch == "1m":
            phrases_1m = ensure_corpus(CORPUS_1M, 1_000_000, seed=273)
            r102 = train_atom_scale(
                stage=102,
                tag="scale_1m",
                phrases=phrases_1m,
                d_model=256,
                n_layers=2,
                n_heads=4,
                ft_steps=100000,
                eval_every=2500,
                ref={
                    "obj": r101["seen"]["obj"],
                    "rel": r101["seen"]["rel"],
                    "story_all": r101["story"]["all"],
                    "rare_obj": r101["rare"]["obj"],
                },
                story_lift_pass=0.04,
                story_keep_frac=0.35,
                story_pair_cap=70000,
            )
        elif branch == "capacity":
            # A': depth/width at dim256 (F85 init intact; no dead pad dims)
            r102 = train_atom_scale(
                stage=102,
                tag="capacity_4L8H",
                phrases=phrases_500,  # same 500k data
                d_model=256,
                n_layers=4,
                n_heads=8,
                ft_steps=80000,
                eval_every=2000,
                ref={
                    "obj": r101["seen"]["obj"],
                    "rel": r101["seen"]["rel"],
                    "story_all": r101["story"]["all"],
                    "rare_obj": r101["rare"]["obj"],
                },
                story_lift_pass=0.04,
            )
        else:
            log("[decision] STOP — SEEN not held; skip Stage102")

        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ref100": REF100,
            "stage101": {
                "verdict": r101["verdict"],
                "story_all": r101["story"]["all"],
                "lift_vs_100": r101["lift_story_pp"],
                "seen": r101["seen"],
                "ckpt": r101["ckpt"],
            },
            "branch": branch,
            "stage102": None
            if r102 is None
            else {
                "verdict": r102["verdict"],
                "tag": r102["tag"],
                "story_all": r102["story"]["all"],
                "lift_vs_ref": r102["lift_story_pp"],
                "seen": r102["seen"],
                "arch": r102["arch"],
                "ckpt": r102["ckpt"],
            },
        }
        DECISION.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        update_docs(r101, r102, branch)
        log(f"\nDONE branch={branch}")
        log(json.dumps(decision, indent=2))
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
