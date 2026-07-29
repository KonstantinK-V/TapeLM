"""
Stage 161 — extreme geometries @ ~500k (after 160).

Push width/depth outside the S★ / S+ band to map the failure envelope.
Same S★ ritual lock; word_fp only; hops OUT; N≈460k.

Arms:
  thin_2L_d128        half-width, same depth as S★
  thin_4L_d128        thin-deep extreme
  ultra_deep_8L_d128  depth stress at d128
  deep_8L_d256        depth stress at base width
  shallow_1L_d256     extreme shallow @ base width
  shallow_1L_d512     extreme wide-shallow
  ultra_wide_2L_d768  wider than S+, still shallow

Waits for stage160_geometry_500k_decision.json

Run:
  python _stage161_extreme_geometry_500k.py
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
from _stage150_155_clean_compare_pipeline import load_phrases, setup_stack  # noqa: E402
from _stage160_geometry_500k import RITUAL, train_pyramid, train_uniform  # noqa: E402

DEC160 = RES / "stage160_geometry_500k_decision.json"
DEC = RES / "stage161_extreme_geometry_500k_decision.json"
LOG = RES / "_stage161_extreme_geometry_500k_log.txt"
N_PHRASES = 460_000

GEOMS = {
    "thin_2L_d128": {
        "kind": "uniform",
        "d": 128,
        "n_layer": 2,
        "n_head": 4,
        "note": "half-width vs S★; same depth",
    },
    "thin_4L_d128": {
        "kind": "uniform",
        "d": 128,
        "n_layer": 4,
        "n_head": 4,
        "note": "thin-deep extreme",
    },
    "ultra_deep_8L_d128": {
        "kind": "uniform",
        "d": 128,
        "n_layer": 8,
        "n_head": 4,
        "note": "depth stress at d128",
    },
    "deep_8L_d256": {
        "kind": "uniform",
        "d": 256,
        "n_layer": 8,
        "n_head": 4,
        "note": "depth stress at base width",
    },
    "shallow_1L_d256": {
        "kind": "uniform",
        "d": 256,
        "n_layer": 1,
        "n_head": 4,
        "note": "extreme shallow @ base width",
    },
    "shallow_1L_d512": {
        "kind": "uniform",
        "d": 512,
        "n_layer": 1,
        "n_head": 8,
        "note": "extreme wide-shallow",
    },
    "ultra_wide_2L_d768": {
        "kind": "uniform",
        "d": 768,
        "n_layer": 2,
        "n_head": 12,
        "note": "wider than S+, still shallow",
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


def wait_160(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage160_geometry_500k_decision.json ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC160.exists():
            d = json.loads(DEC160.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 160 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 160 ... {int(time.time()-t0)}s")
    raise TimeoutError("160 not ready")


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

    log(f"161 extreme geometry@500k start {datetime.now(timezone.utc).isoformat()}")
    if already_done(DEC):
        return 0
    try:
        up = wait_160()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg, stack = setup_stack(device)
        phrases = load_phrases(N_PHRASES)
        log(f"[data] n_phrases={len(phrases)}")

        arms = {}
        for name, geom in GEOMS.items():
            seed = stable_seed(161, name)
            log(f"\n########## EXTREME {name}: {geom.get('note')} ##########")
            if geom["kind"] == "uniform":
                arm = train_uniform(name, geom, phrases, device, cfg, stack, seed)
            else:
                arm = train_pyramid(name, geom, phrases, device, cfg, stack, seed)
            arm["geom"] = geom
            arm["n_phrases"] = len(phrases)
            arms[name] = arm
            log(
                f"[done] {name} STORY={100*arm['story_all']:.1f}% "
                f"SEEN_obj={100*arm['seen_obj']:.1f}% params={arm['params']/1e6:.2f}M"
            )

        base_s = None
        base_src = "stage160"
        if DEC160.exists():
            d160 = json.loads(DEC160.read_text(encoding="utf-8"))
            base_s = (d160.get("arms") or {}).get("base_2L_d256", {}).get("story_all")
        if base_s is None:
            base_s = 0.0
            base_src = "none"

        deltas = {k: arms[k]["story_all"] - base_s for k in arms}
        best_name = max(arms, key=lambda k: arms[k]["story_all"])
        ranked = sorted(arms, key=lambda k: arms[k]["story_all"], reverse=True)
        survivors = [k for k, v in deltas.items() if v >= -0.02]
        collapsed = [k for k, v in deltas.items() if v < -0.05]

        if any(deltas[k] >= 0.03 for k in arms):
            verdict = "PASS_PROMOTE"
        elif any(deltas[k] >= 0.015 for k in arms):
            verdict = "PARTIAL"
        elif not survivors and collapsed:
            verdict = "EXTREMES_COLLAPSE"
        elif survivors:
            verdict = "ENVELOPE_MAPPED"
        else:
            verdict = "EXTREME_TABLE"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "extreme_geometry_envelope_500k",
            "framing": (
                "Push width/depth outside S★/S+ band at N≈460k to map failure envelope. "
                "d128 / 1L / 8L / d768."
            ),
            "n_phrases": len(phrases),
            "ritual": RITUAL,
            "geoms": GEOMS,
            "arms": arms,
            "ref_base_story": base_s,
            "ref_base_src": base_src,
            "deltas_story_vs_160_base": deltas,
            "survivors_within_2pp": survivors,
            "collapsed_below_5pp": collapsed,
            "best_arm": best_name,
            "best_lift_vs_base": deltas[best_name],
            "ranked": ranked,
            "verdict": verdict,
            "upstream_160": up.get("verdict"),
            "ref100": REF100,
            "why_these_extremes": {
                "thin_2L_d128": "Is d256 necessary, or does half-width still work?",
                "thin_4L_d128": "Thin-deep extreme: low width + depth",
                "ultra_deep_8L_d128": "Does depth alone destroy thin models?",
                "deep_8L_d256": "Depth stress without shrinking width",
                "shallow_1L_d256": "Is 2L necessary @ base width?",
                "shallow_1L_d512": "Extreme wide-shallow: width without depth",
                "ultra_wide_2L_d768": "Beyond S+ width; still shallow — width ceiling?",
            },
        }
        write_dec(DEC, out)
        log(
            f"[161] {verdict} best={best_name} "
            f"STORY={100*arms[best_name]['story_all']:.1f}% "
            f"lift_vs_160base={100*deltas[best_name]:+.1f}pp"
        )
        log("survivors@-2pp: " + (", ".join(survivors) if survivors else "none"))
        log("collapsed@-5pp: " + (", ".join(collapsed) if collapsed else "none"))
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
