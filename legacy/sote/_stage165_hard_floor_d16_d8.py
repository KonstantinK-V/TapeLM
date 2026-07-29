"""
Stage 165 — hard floor: 1L d16 / d8 @ 50k (after 164).

163 showed soft floor only (d32 still ~20.6%, within noise of d128).
Push width to d16 and d8 with 1 head to find real collapse.

Arms (N≈460k, word_fp, S★ ritual, 50k, hops OUT):
  1. shallow_1L_d16_h1  — head_dim=16
  2. shallow_1L_d8_h1   — head_dim=8
  3. shallow_1L_d16_h2  — optional: d16/2H head_dim=8 (same width, split)

Waits for stage164_zerolayer_control_decision.json

Run:
  python _stage165_hard_floor_d16_d8.py
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

DEC164 = RES / "stage164_zerolayer_control_decision.json"
DEC163 = RES / "stage163_onelayer_floor_heads_decision.json"
DEC162 = RES / "stage162_onelayer_d128_50k_decision.json"
DEC = RES / "stage165_hard_floor_d16_d8_decision.json"
LOG = RES / "_stage165_hard_floor_d16_d8_log.txt"
N_PHRASES = 460_000
STEPS = 50_000

ARMS = {
    "shallow_1L_d16_h1": {
        "d": 16,
        "n_layer": 1,
        "n_head": 1,
        "note": "hard floor d16 / 1H; head_dim=16",
    },
    "shallow_1L_d8_h1": {
        "d": 8,
        "n_layer": 1,
        "n_head": 1,
        "note": "hard floor d8 / 1H; head_dim=8",
    },
    "shallow_1L_d16_h2": {
        "d": 16,
        "n_layer": 1,
        "n_head": 2,
        "note": "d16 / 2H; head_dim=8 — split vs single",
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


def wait_164(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage164_zerolayer_control_decision.json ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC164.exists():
            d = json.loads(DEC164.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 164 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 164 ... {int(time.time()-t0)}s")
    raise TimeoutError("164 not ready")


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

    log(f"165 hard floor d16/d8 start {datetime.now(timezone.utc).isoformat()}")
    if already_done(DEC):
        return 0
    try:
        up = wait_164()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg, stack = setup_stack(device)
        phrases = load_phrases(N_PHRASES)
        log(f"[data] n_phrases={len(phrases)} steps={STEPS}")

        arms = {}
        for name, spec in ARMS.items():
            trunk = dict(RITUAL)
            trunk.update(d=spec["d"], n_layer=spec["n_layer"], n_head=spec["n_head"])
            seed = stable_seed(165, name)
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
                ckpt_name=f"stage165_{name}.pt",
            )
            arm["geom"] = spec
            arm["head_dim"] = spec["d"] // spec["n_head"]
            arm["n_phrases"] = len(phrases)
            arms[name] = arm
            log(
                f"[done] {name} STORY={100*arm['story_all']:.1f}% "
                f"SEEN_obj={100*arm['seen_obj']:.1f}% params={arm['params']/1e6:.2f}M "
                f"head_dim={arm['head_dim']}"
            )

        ref = None
        if DEC162.exists():
            d162 = json.loads(DEC162.read_text(encoding="utf-8"))
            ref = (d162.get("arms") or {}).get("shallow_1L_d128_50k", {}).get("story_all")
        if ref is None and DEC163.exists():
            d163 = json.loads(DEC163.read_text(encoding="utf-8"))
            ref = (d163.get("arms") or {}).get("shallow_1L_d128_h2", {}).get("story_all")
        if ref is None:
            ref = 0.0

        deltas = {k: arms[k]["story_all"] - ref for k in arms}
        ranked = sorted(arms, key=lambda k: arms[k]["story_all"], reverse=True)
        survivors = [k for k, v in deltas.items() if v >= -0.02]
        collapsed = [k for k, v in deltas.items() if v < -0.05]
        dead = [k for k, v in deltas.items() if v < -0.10]

        if dead:
            verdict = "HARD_FLOOR_FOUND"
        elif collapsed:
            verdict = "SOFT_COLLAPSE"
        elif survivors == list(arms.keys()):
            verdict = "STILL_NO_FLOOR"
        else:
            verdict = "FLOOR_TABLE"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "hard_floor_d16_d8_50k",
            "framing": (
                "Push 1L width to d16/d8 to find real STORY collapse. "
                "If still within ~2pp of d128, emb+1attn is insanely compressible on this task; "
                "if gap>=10pp, hard floor is between d32 and d16/d8."
            ),
            "n_phrases": len(phrases),
            "steps": STEPS,
            "ritual": RITUAL,
            "arms": arms,
            "ref_1L_d128_story": ref,
            "deltas_vs_1L_d128": deltas,
            "survivors_within_2pp": survivors,
            "collapsed_below_5pp": collapsed,
            "dead_below_10pp": dead,
            "best_arm": ranked[0],
            "ranked": ranked,
            "verdict": verdict,
            "upstream_164": up.get("verdict"),
            "ref100": REF100,
            "what_this_means": (
                "d = width of each word vector and of the single attention block. "
                "Tiny d means words are packed into a very low-dimensional codebook; "
                "attention mixes a thin signal. Collapse => task needs more geometric room "
                "than a near-bigram table in R^8/R^16. Survival => STORY here is carried by "
                "coarse clusters / transition structure, not rich syntax geometry."
            ),
        }
        write_dec(DEC, out)
        log(
            f"[165] {verdict} best={ranked[0]} "
            f"STORY={100*arms[ranked[0]]['story_all']:.1f}% "
            f"vs_d128={100*deltas[ranked[0]]:+.1f}pp"
        )
        log("dead@-10pp: " + (", ".join(dead) if dead else "none"))
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
