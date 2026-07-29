"""
Follow-ups after overnight 101/102 (queue while sleeping).

Waits for results/overnight_atom_scale_decision.json, then:

  Stage103: fat_frac=0.90 on 500k (2L4H) — joint obj dig (Stage98 leftover knob)
  Stage104: denser fat triples (n_fat=400 x60) fat_frac=0.85 — more SEEN frames
  Stage105: short FT fat_frac=0.95 (40k steps)
  Stage106: ~1M TinyStories × capacity 4L8H (~5M params)

Run:
  python _overnight_followups.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _overnight_atom_scale import (  # noqa: E402
    CORPUS_1M,
    CORPUS_500K,
    DECISION,
    REF100,
    RES,
    ensure_corpus,
    ensure_raw,
    log,
    train_atom_scale,
)

FOLLOW_LOG = RES / "_overnight_followups_log.txt"
FOLLOW_DECISION = RES / "overnight_followups_decision.json"


def _log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    with FOLLOW_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    # also mirror into main overnight log
    try:
        log(msg)
    except Exception:
        pass


def wait_overnight(timeout_s: int = 6 * 3600, poll_s: int = 30) -> dict:
    _log(f"[wait] for {DECISION} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DECISION.exists():
            d = json.loads(DECISION.read_text(encoding="utf-8"))
            # written only when overnight main() finishes
            if d.get("stage101") is not None:
                _log(f"[wait] overnight done branch={d.get('branch')}")
                return d
        time.sleep(poll_s)
        elapsed = int(time.time() - t0)
        if elapsed % 300 < poll_s:
            _log(f"[wait] still ... {elapsed}s")
    raise TimeoutError("overnight decision not ready")


def load_phrases():
    return ensure_corpus(CORPUS_500K, 500_000, seed=272)


def main():
    RES.mkdir(exist_ok=True)
    if FOLLOW_LOG.exists():
        FOLLOW_LOG.write_text("", encoding="utf-8")
    _log(f"Followups start {datetime.now(timezone.utc).isoformat()}")

    try:
        d = wait_overnight()
        phrases = load_phrases()
        r101 = d["stage101"]
        ref101 = {
            "obj": r101["seen"]["obj"],
            "rel": r101["seen"]["rel"],
            "story_all": r101["story_all"],
            "rare_obj": 0.138,
        }
        # if 102 exists, also keep as secondary ref for capacity verdict note
        r102 = d.get("stage102")

        # --- 103 fat_frac 0.90 ---
        r103 = train_atom_scale(
            stage=103,
            tag="fat90_obj",
            phrases=phrases,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ft_steps=60000,
            eval_every=2000,
            fat_frac=0.90,
            n_fat=200,
            fat_copies=40,
            ref=ref101,
            story_lift_pass=0.03,
        )
        # rewrite verdict emphasis on SEEN obj vs 101
        obj_lift = r103["seen"]["obj"] - ref101["obj"]
        if r103["seen_held"] and obj_lift >= 0.04:
            r103["verdict"] = "PASS_OBJ"
        elif r103["seen_held"] and obj_lift >= 0.02:
            r103["verdict"] = "PARTIAL_OBJ"
        elif r103["seen_held"] and abs(obj_lift) < 0.02:
            r103["verdict"] = "PARITY_OBJ"
        _log(f"[103] obj_lift={obj_lift*100:+.1f}pp verdict→{r103['verdict']}")

        # --- 104 denser fat pool ---
        r104 = train_atom_scale(
            stage=104,
            tag="fat_dense400",
            phrases=phrases,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ft_steps=60000,
            eval_every=2000,
            fat_frac=0.85,
            n_fat=400,
            fat_copies=60,
            ref=ref101,
            story_lift_pass=0.03,
        )
        obj_lift4 = r104["seen"]["obj"] - ref101["obj"]
        if r104["seen_held"] and obj_lift4 >= 0.04:
            r104["verdict"] = "PASS_OBJ"
        elif r104["seen_held"] and obj_lift4 >= 0.02:
            r104["verdict"] = "PARTIAL_OBJ"
        elif r104["seen_held"] and abs(obj_lift4) < 0.02:
            r104["verdict"] = "PARITY_OBJ"
        _log(f"[104] obj_lift={obj_lift4*100:+.1f}pp verdict→{r104['verdict']}")

        # --- 105: hottest fat_frac from scratch (short) ---
        r105 = train_atom_scale(
            stage=105,
            tag="fat95_push",
            phrases=phrases,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ft_steps=40000,
            eval_every=2000,
            fat_frac=0.95,
            n_fat=300,
            fat_copies=50,
            ref=ref101,
            story_lift_pass=0.03,
        )
        obj_lift5 = r105["seen"]["obj"] - ref101["obj"]
        if r105["seen_held"] and obj_lift5 >= 0.04:
            r105["verdict"] = "PASS_OBJ"
        elif r105["seen_held"] and obj_lift5 >= 0.02:
            r105["verdict"] = "PARTIAL_OBJ"
        elif r105["seen_held"] and abs(obj_lift5) < 0.02:
            r105["verdict"] = "PARITY_OBJ"
        _log(f"[105] obj_lift={obj_lift5*100:+.1f}pp verdict→{r105['verdict']}")

        # --- 106: 1M data × capacity 4L8H (~5M) ---
        _log("[106] ensure larger raw for ~1M windows ...")
        ensure_raw(220_000_000)
        phrases_1m = ensure_corpus(CORPUS_1M, 1_000_000, seed=273)
        ref_cap = ref101
        if r102 is not None and isinstance(r102.get("seen"), dict):
            # prefer capacity run as arch baseline if present
            ref_cap = {
                "obj": r102["seen"].get("obj", ref101["obj"]),
                "rel": r102["seen"].get("rel", ref101["rel"]),
                "story_all": r102.get("story_all", ref101["story_all"]),
                "rare_obj": ref101.get("rare_obj", 0.14),
            }
        r106 = train_atom_scale(
            stage=106,
            tag="scale_1m_4L8H",
            phrases=phrases_1m,
            d_model=256,
            n_layers=4,
            n_heads=8,
            ft_steps=100000,
            eval_every=2500,
            fat_frac=0.75,
            n_fat=200,
            fat_copies=40,
            story_keep_frac=0.35,
            story_pair_cap=70000,
            ref=ref_cap,
            story_lift_pass=0.04,
        )
        _log(
            f"[106] STORY={r106['story']['all']*100:.1f}% "
            f"SEEN obj={r106['seen']['obj']*100:.1f}% verdict={r106['verdict']}"
        )

        best = max(
            [
                ("101", ref101["obj"], None),
                ("103", r103["seen"]["obj"], r103),
                ("104", r104["seen"]["obj"], r104),
                ("105", r105["seen"]["obj"], r105),
                ("106", r106["seen"]["obj"], r106),
            ],
            key=lambda x: x[1],
        )

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overnight_branch": d.get("branch"),
            "stage102": r102,
            "ref101_obj": ref101["obj"],
            "stage103": {
                "verdict": r103["verdict"],
                "seen": r103["seen"],
                "story": r103["story"],
                "obj_lift_pp": obj_lift,
                "ckpt": r103["ckpt"],
            },
            "stage104": {
                "verdict": r104["verdict"],
                "seen": r104["seen"],
                "story": r104["story"],
                "obj_lift_pp": obj_lift4,
                "ckpt": r104["ckpt"],
            },
            "stage105": {
                "verdict": r105["verdict"],
                "seen": r105["seen"],
                "story": r105["story"],
                "obj_lift_pp": obj_lift5,
                "ckpt": r105["ckpt"],
            },
            "stage106": {
                "verdict": r106["verdict"],
                "tag": r106["tag"],
                "n_phrases": r106["n_phrases"],
                "params": r106["params"],
                "arch": r106["arch"],
                "seen": r106["seen"],
                "story": r106["story"],
                "lift_story_pp": r106["lift_story_pp"],
                "ckpt": r106["ckpt"],
            },
            "best_seen_obj": {"stage": best[0], "obj": best[1]},
            "note": "103-105 obj digs (fatter SEEN); 106 = 1M × 4L8H capacity",
        }
        FOLLOW_DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")

        # append short note to path replay
        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stages 103–106 (after 101 CEILING):** "
                f"fat0.90 obj={r103['seen']['obj']*100:.1f}% ({r103['verdict']}); "
                f"fat400×60 obj={r104['seen']['obj']*100:.1f}% ({r104['verdict']}); "
                f"fat0.95 obj={r105['seen']['obj']*100:.1f}% ({r105['verdict']}); "
                f"**1M×4L8H** STORY={r106['story']['all']*100:.1f}% "
                f"obj={r106['seen']['obj']*100:.1f}% ({r106['verdict']}). "
                f"Best SEEN obj: Stage {best[0]}={best[1]*100:.1f}%. "
                f"`overnight_followups_decision.json`.\n"
            )
            if "Stages 103–106" not in txt and "Stages 103–105" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
            elif "Stages 103–105" in txt and "Stages 103–106" not in txt:
                txt = txt.replace("Stages 103–105", "Stages 103–106 (incl 1M×4L8H)")
                replay.write_text(txt, encoding="utf-8")

        _log(f"DONE best_obj_stage={best[0]} obj={best[1]*100:.1f}%")
        _log(json.dumps(out, indent=2))
        return 0
    except Exception:
        _log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
