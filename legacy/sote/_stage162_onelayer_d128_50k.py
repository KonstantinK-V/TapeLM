"""
Stage 162 — one-layer LM dig: 1L d128 @ 50k steps (after 161).

User ask: try single-layer word LM at half-width, longer budget than 40k
(plateau caveat on shallow_1L_d256). Ritual = S★ otherwise.

Arms (N≈460k, word_fp, hops OUT):
  1. shallow_1L_d128_50k  — 1L / 4H / d128 / 50k steps  (primary)
  2. shallow_1L_d256_50k  — 1L / 4H / d256 / 50k steps  (matched longer-run ctrl)

Compare vs:
  - 160 base_2L_d256 (~19.1% @40k)
  - 161 shallow_1L_d256 (~19.3% @40k)
  - 161 thin_4L_d128 (~20.2% @40k)

Waits for stage161_extreme_geometry_500k_decision.json

Run:
  python _stage162_onelayer_d128_50k.py
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

DEC161 = RES / "stage161_extreme_geometry_500k_decision.json"
DEC160 = RES / "stage160_geometry_500k_decision.json"
DEC = RES / "stage162_onelayer_d128_50k_decision.json"
LOG = RES / "_stage162_onelayer_d128_50k_log.txt"
N_PHRASES = 460_000
STEPS = 50_000

ARMS = {
    "shallow_1L_d128_50k": {
        "d": 128,
        "n_layer": 1,
        "n_head": 4,
        "note": "1L half-width @50k — primary",
    },
    "shallow_1L_d256_50k": {
        "d": 256,
        "n_layer": 1,
        "n_head": 4,
        "note": "1L base-width @50k — longer-run ctrl vs 161@40k",
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


def wait_161(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage161_extreme_geometry_500k_decision.json ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC161.exists():
            d = json.loads(DEC161.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 161 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 161 ... {int(time.time()-t0)}s")
    raise TimeoutError("161 not ready")


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

    log(f"162 onelayer dig start {datetime.now(timezone.utc).isoformat()}")
    if already_done(DEC):
        return 0
    try:
        up = wait_161()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg, stack = setup_stack(device)
        phrases = load_phrases(N_PHRASES)
        log(f"[data] n_phrases={len(phrases)} steps={STEPS}")

        arms = {}
        for name, spec in ARMS.items():
            trunk = dict(RITUAL)
            trunk.update(d=spec["d"], n_layer=spec["n_layer"], n_head=spec["n_head"])
            seed = stable_seed(162, name)
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
                ckpt_name=f"stage162_{name}.pt",
            )
            arm["geom"] = spec
            arm["n_phrases"] = len(phrases)
            arm["steps_requested"] = STEPS
            arms[name] = arm
            log(
                f"[done] {name} STORY={100*arm['story_all']:.1f}% "
                f"SEEN_obj={100*arm['seen_obj']:.1f}% params={arm['params']/1e6:.2f}M "
                f"last_ev={100*arm.get('story_all_last', 0):.1f}%"
            )

        # refs
        ref = {}
        if DEC160.exists():
            d160 = json.loads(DEC160.read_text(encoding="utf-8"))
            ref["base_2L_d256"] = (d160.get("arms") or {}).get("base_2L_d256", {}).get("story_all")
        if DEC161.exists():
            d161 = json.loads(DEC161.read_text(encoding="utf-8"))
            a161 = d161.get("arms") or {}
            ref["shallow_1L_d256_40k"] = (a161.get("shallow_1L_d256") or {}).get("story_all")
            ref["thin_4L_d128_40k"] = (a161.get("thin_4L_d128") or {}).get("story_all")
            ref["thin_2L_d128_40k"] = (a161.get("thin_2L_d128") or {}).get("story_all")

        primary = arms["shallow_1L_d128_50k"]["story_all"]
        ctrl256 = arms["shallow_1L_d256_50k"]["story_all"]
        base = ref.get("base_2L_d256") or 0.0
        ref_1l40 = ref.get("shallow_1L_d256_40k")
        ref_thin4 = ref.get("thin_4L_d128_40k")

        lift_vs_base = primary - base
        lift_256_vs_40k = (ctrl256 - ref_1l40) if ref_1l40 is not None else None
        lift_128_vs_thin4 = (primary - ref_thin4) if ref_thin4 is not None else None

        # promote 1L d128 if clearly beats base and competitive with thin_4L
        if lift_vs_base >= 0.03:
            verdict = "PASS_PROMOTE"
        elif lift_vs_base >= 0.015:
            verdict = "PARTIAL"
        elif lift_vs_base >= -0.02:
            verdict = "PARITY"
        else:
            verdict = "HARM"

        # annotate longer-run read on d256 ctrl
        longer_note = "n/a"
        if lift_256_vs_40k is not None:
            if lift_256_vs_40k >= 0.015:
                longer_note = "1L_d256 still climbing @50k vs 40k — plateau hypothesis weakened"
            elif lift_256_vs_40k >= -0.01:
                longer_note = "1L_d256 flat 40k→50k — plateau-ish under this ritual"
            else:
                longer_note = "1L_d256 worse @50k — late overfit / noise"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "onelayer_d128_50k",
            "framing": (
                "Single-layer word LM at d128 with 50k steps; paired 1L d256@50k "
                "tests whether 40k was a false plateau. User preference: 1L LM path."
            ),
            "n_phrases": len(phrases),
            "steps": STEPS,
            "ritual": RITUAL,
            "arms": arms,
            "refs": ref,
            "primary_story": primary,
            "lift_1L_d128_vs_160_base": lift_vs_base,
            "lift_1L_d256_50k_vs_40k": lift_256_vs_40k,
            "lift_1L_d128_vs_thin4L_d128": lift_128_vs_thin4,
            "longer_run_note": longer_note,
            "verdict": verdict,
            "upstream_161": up.get("verdict"),
            "ref100": REF100,
        }
        write_dec(DEC, out)
        log(
            f"[162] {verdict} 1L_d128={100*primary:.1f}% "
            f"vs_base={100*lift_vs_base:+.1f}pp | 1L_d256_50k={100*ctrl256:.1f}% | {longer_note}"
        )
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
