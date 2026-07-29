"""
Stage 135 — CE-only trunk; hops completely OUT of training AND eval path.

Answers: "можно прогнать так (hops за обучением)?"
  Yes — this IS the policy that already produced best atom numbers
  (Stage95→100→125). Here we re-measure with hops code path absent.

Also records why CE-only SOTE word-id is still not "close to GPT"
  (same learning game ≠ same system).

Run (eval-only, no FT):
  python _stage135_ce_only_no_hops.py
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    build_ts_repeat_mix,
    eval_id_capacity_suite,
    line_subkind,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402

ATOM125 = CKPT / "stage125_same_role_rank.pt"
ATOM100 = CKPT / "stage100_scale_100k.pt"
DEC112 = RES / "stage111_112_decision.json"
LOG = RES / "_stage135_log.txt"
DEC = RES / "stage135_ce_only_no_hops_decision.json"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def pack_suite(suite):
    roles = suite.get("roles") or {}
    return {
        "all": roles.get("ALL", {}).get("hit1", 0.0),
        "obj": suite.get("obj", {}).get("hit1")
        if isinstance(suite.get("obj"), dict)
        else roles.get("right", {}).get("hit1", 0.0),
        "rel": roles.get("rel", {}).get("hit1", 0.0),
        "roles": {
            k: {"n": v.get("n", 0), "hit1": v.get("hit1", 0.0)}
            for k, v in roles.items()
        },
    }


def load_bpe112_ref():
    if not DEC112.exists():
        return None
    d = json.loads(DEC112.read_text(encoding="utf-8"))
    s = d.get("stage112") or {}
    return {
        "verdict": s.get("verdict"),
        "seen_obj": (s.get("seen") or {}).get("obj"),
        "seen_rel": (s.get("seen") or {}).get("rel"),
        "story_all": (s.get("story") or {}).get("all"),
        "story_obj": ((s.get("story") or {}).get("roles") or {}).get("right", {}).get("hit1"),
        "V_bpe": s.get("V_bpe"),
        "params": s.get("params"),
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"135 CE-only / hops OUT start {datetime.now(timezone.utc).isoformat()}")
    log("No SoftPhraseMemory. No hop bias. No hop loss. Codebook + CE trunk only.")
    try:
        phrases = ensure_100k()
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # foundation only to build same word list / fps contract as atom ckpt
        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        del stack  # hops/memory never built

        train, hold_seen, hold_rare, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
        for ln in train + hold_seen + hold_rare + hold_story:
            ln.setdefault("subkind", line_subkind(ln))
            ln["split"] = ln.get("bucket", "x")
        words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
        stoi = {s: i for i, s in enumerate(words)}

        path = ATOM125 if ATOM125.exists() else ATOM100
        model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert list(ck["surfaces"]) == words
        model.load_state_dict(ck["word_tf"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        log(f"[init] CE trunk {path.name} V={len(words)} (hops path = ABSENT)")

        ev_seen = _subsample(hold_seen, 600, 1351)
        ev_story = _subsample(hold_story, 400, 1353)
        ev_rare = _subsample(hold_rare, 200, 1355) if hold_rare else []

        seen = pack_suite(eval_id_capacity_suite(model, ev_seen, train, words, stoi, device))
        story = pack_suite(eval_id_capacity_suite(model, ev_story, train, words, stoi, device))
        rare = (
            pack_suite(eval_id_capacity_suite(model, ev_rare, train, words, stoi, device))
            if ev_rare
            else None
        )

        bpe = load_bpe112_ref()
        story_gap_vs_bpe = None
        if bpe and bpe.get("story_all") is not None:
            story_gap_vs_bpe = float(bpe["story_all"]) - float(story["all"])

        # Policy confirmation vs ref100 / expected 125 band
        obj_lift_100 = seen["obj"] - REF100["obj"]
        if seen["rel"] >= 0.70 and abs(obj_lift_100) <= 0.04:
            verdict = "POLICY_OK"  # CE-only still in atom band; hops not required for LM
        elif seen["rel"] >= 0.70 and obj_lift_100 >= 0.03:
            verdict = "PASS"
        else:
            verdict = "PARITY"

        why_not_gpt = {
            "same_axis": "context → next-atom CE → decode (shared with GPT-style LMs)",
            "not_close_because": [
                {
                    "axis": "atom_unit",
                    "here": "whole-word codebook (~5k surfaces)",
                    "gpt_like": "BPE/subword (Stage112 V=8k on same windows → STORY~34%)",
                    "evidence_pp_story": story_gap_vs_bpe,
                },
                {
                    "axis": "scale",
                    "here": "2L/4H d=256, ~50k–100k TinyStories windows",
                    "gpt_like": "orders of magnitude more params + data",
                },
                {
                    "axis": "objective_surface",
                    "here": "exact@1 on short windows / roles",
                    "gpt_like": "token CE at web scale + long context KV",
                },
                {
                    "axis": "encode",
                    "here": "frozen letter→word_fp foundation + word-id head",
                    "gpt_like": "end-to-end learned token emb from scratch",
                },
                {
                    "axis": "memory",
                    "here": "hops OUT of train (this stage); optional separate API",
                    "gpt_like": "facts mostly absorbed into weights/KV of same trunk",
                },
            ],
            "punchline": (
                "Leaving hops out of training makes SOTE's LM trunk more GPT-like in "
                "training recipe, but atom=word + tiny scale + TinyStories keep it far "
                "from GPT performance — Stage112 already showed BPE alone lifts STORY "
                "~+14pp on the same task."
            ),
        }

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "ce_only_hops_completely_out",
            "atom_ckpt": path.name,
            "hops_in_train": False,
            "hops_in_eval": False,
            "seen": seen,
            "story": story,
            "rare": rare,
            "ref100": REF100,
            "bpe112_ref": bpe,
            "story_gap_vs_bpe_pp": story_gap_vs_bpe,
            "verdict": verdict,
            "why_not_gpt": why_not_gpt,
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage135_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")

        log(
            f"[135] {verdict} SEEN obj={100*seen['obj']:.1f}% rel={100*seen['rel']:.1f}% "
            f"STORY ALL={100*story['all']:.1f}% | vs BPE112 STORY gap="
            f"{100*story_gap_vs_bpe:+.1f}pp" if story_gap_vs_bpe is not None else
            f"[135] {verdict} SEEN obj={100*seen['obj']:.1f}% rel={100*seen['rel']:.1f}% "
            f"STORY ALL={100*story['all']:.1f}%"
        )
        log(why_not_gpt["punchline"])

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 135 CE-only (hops OUT):** {verdict} "
                f"SEEN obj={100*seen['obj']:.1f}% STORY={100*story['all']:.1f}% "
                f"(no hops in train/eval). `stage135_ce_only_no_hops_decision.json`.\n"
            )
            if "Stage 135 CE-only" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 135")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
