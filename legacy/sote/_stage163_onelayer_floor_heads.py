"""
Stage 163 — one-layer floor + heads (after 162).

After 1L d128@50k hit ~21.6% STORY: map how small we can go, and test h2.

Arms (N≈460k, word_fp, S★ ritual, 50k steps, hops OUT):
  1. shallow_1L_d128_h2   — heads probe at winner width (vs 162 h4)
  2. shallow_1L_d128_h1   — single head @ d128
  3. shallow_1L_d64_h2    — half of 128; head_dim=32
  4. shallow_1L_d64_h4    — d64 with 4H; head_dim=16
  5. shallow_1L_d32_h2    — hard floor; head_dim=16

Waits for stage162_onelayer_d128_50k_decision.json

Run:
  python _stage163_onelayer_floor_heads.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import CKPT, RES  # noqa: E402
from _stage109_110_slot_baseline import REF100  # noqa: E402
from _stage150_155_clean_compare_pipeline import (  # noqa: E402
    load_phrases,
    setup_stack,
    train_word,
)
from _stage160_geometry_500k import RITUAL  # noqa: E402

DEC162 = RES / "stage162_onelayer_d128_50k_decision.json"
DEC160 = RES / "stage160_geometry_500k_decision.json"
DEC = RES / "stage163_onelayer_floor_heads_decision.json"
LOG = RES / "_stage163_onelayer_floor_heads_log.txt"
N_PHRASES = 460_000
STEPS = 50_000

ARMS = {
    "shallow_1L_d128_h2": {
        "d": 128,
        "n_layer": 1,
        "n_head": 2,
        "note": "h2 @ winner width; head_dim=64",
    },
    "shallow_1L_d128_h1": {
        "d": 128,
        "n_layer": 1,
        "n_head": 1,
        "note": "single head @ d128; head_dim=128",
    },
    "shallow_1L_d64_h2": {
        "d": 64,
        "n_layer": 1,
        "n_head": 2,
        "note": "floor d64 h2; head_dim=32",
    },
    "shallow_1L_d64_h4": {
        "d": 64,
        "n_layer": 1,
        "n_head": 4,
        "note": "floor d64 h4; head_dim=16",
    },
    "shallow_1L_d32_h2": {
        "d": 32,
        "n_layer": 1,
        "n_head": 2,
        "note": "hard floor d32 h2; head_dim=16",
    },
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


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF


def wait_162(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage162_onelayer_d128_50k_decision.json ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC162.exists():
            d = json.loads(DEC162.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 162 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 162 ... {int(time.time()-t0)}s")
    raise TimeoutError("162 not ready")


def already_done(path: Path) -> dict | None:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("verdict"):
            log(f"[skip] {path.name}")
            return d
    return None


def write_dec(path: Path, out: dict):
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[write] {path.name}")


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")

    import _stage150_155_clean_compare_pipeline as p150

    _orig = p150.log

    def _tee(msg):
        _orig(msg)
        line = msg if msg.endswith("\n") else msg + "\n"
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)

    p150.log = _tee

    log(f"163 onelayer floor+heads start {datetime.now(timezone.utc).isoformat()}")
    if already_done(DEC):
        return 0
    try:
        up = wait_162()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg, stack = setup_stack(device)
        phrases = load_phrases(N_PHRASES)
        log(f"[data] n_phrases={len(phrases)} steps={STEPS}")

        arms = {}
        for name, spec in ARMS.items():
            trunk = dict(RITUAL)
            trunk.update(d=spec["d"], n_layer=spec["n_layer"], n_head=spec["n_head"])
            seed = stable_seed(163, name)
            log(f"\n########## {name}: {spec['note']} ##########")
            arm = train_word(
                name,
                trunk,
                phrases,
                device,
                cfg,
                stack,
                steps=STEPS,
                seed=seed,
                ckpt_name=f"stage163_{name}.pt",
            )
            arm["geom"] = spec
            arm["head_dim"] = spec["d"] // spec["n_head"]
            arm["n_phrases"] = len(phrases)
            arm["steps_requested"] = STEPS
            arms[name] = arm
            log(
                f"[done] {name} STORY={100*arm['story_all']:.1f}% "
                f"SEEN_obj={100*arm['seen_obj']:.1f}% params={arm['params']/1e6:.2f}M "
                f"head_dim={arm['head_dim']}"
            )

        ref_128_h4 = (up.get("arms") or {}).get("shallow_1L_d128_50k", {}).get("story_all")
        ref_base = None
        if DEC160.exists():
            d160 = json.loads(DEC160.read_text(encoding="utf-8"))
            ref_base = (d160.get("arms") or {}).get("base_2L_d256", {}).get("story_all")
        if ref_128_h4 is None:
            ref_128_h4 = 0.0

        deltas_vs_128h4 = {k: arms[k]["story_all"] - ref_128_h4 for k in arms}
        ranked = sorted(arms, key=lambda k: arms[k]["story_all"], reverse=True)
        best = ranked[0]

        # floor read: which d still within -2pp of 1L d128 h4
        survivors = [k for k, v in deltas_vs_128h4.items() if v >= -0.02]
        collapsed = [k for k, v in deltas_vs_128h4.items() if v < -0.05]

        # heads read at d128
        h2 = arms["shallow_1L_d128_h2"]["story_all"] - ref_128_h4
        h1 = arms["shallow_1L_d128_h1"]["story_all"] - ref_128_h4

        if deltas_vs_128h4[best] >= 0.015:
            verdict = "PASS_PROMOTE"
        elif survivors and collapsed:
            verdict = "FLOOR_MAPPED"
        elif survivors:
            verdict = "FLOOR_SOFT"
        elif collapsed:
            verdict = "FLOOR_HARD"
        else:
            verdict = "FLOOR_TABLE"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "onelayer_floor_and_heads_50k",
            "framing": (
                "Find decrease boundary below 1L d128 and test fewer heads (h2/h1). "
                "Same 50k / S★ / N≈460k as 162."
            ),
            "n_phrases": len(phrases),
            "steps": STEPS,
            "ritual": RITUAL,
            "arms": arms,
            "ref_1L_d128_h4_story": ref_128_h4,
            "ref_160_base_story": ref_base,
            "deltas_vs_1L_d128_h4": deltas_vs_128h4,
            "heads_delta_at_d128": {"h2_minus_h4": h2, "h1_minus_h4": h1},
            "survivors_within_2pp": survivors,
            "collapsed_below_5pp": collapsed,
            "best_arm": best,
            "best_story": arms[best]["story_all"],
            "ranked": ranked,
            "verdict": verdict,
            "upstream_162": up.get("verdict"),
            "ref100": REF100,
        }
        write_dec(DEC, out)
        log(
            f"[163] {verdict} best={best} STORY={100*arms[best]['story_all']:.1f}% "
            f"vs_128h4={100*deltas_vs_128h4[best]:+.1f}pp | "
            f"h2={100*h2:+.1f}pp h1={100*h1:+.1f}pp"
        )
        log("survivors@-2pp: " + (", ".join(survivors) if survivors else "none"))
        log("collapsed@-5pp: " + (", ".join(collapsed) if collapsed else "none"))
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
