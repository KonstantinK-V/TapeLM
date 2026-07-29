#!/usr/bin/env python3
"""Run a TapeLM-line stage by number, e.g. 204 or 203. From repo root."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STAGE_SCRIPTS = {
    196: "_stage196_tapelm.py",
    203: "_stage203_internal_hops.py",
    204: "_stage204_noise_robustness.py",
    205: "_stage205_unlearn_provenance.py",
    206: "_stage206_latent_hops_budget.py",
    207: "_stage207_curve_thinking.py",
    209: "_stage209_sem_scaling.py",
    210: "_stage210_softfollow_forward.py",
    211: "_stage211_internal_tape.py",
    212: "_stage212_instance_channel.py",
}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python artifact/scripts/run_stage.py <stage_number>")
        print("Known:", ", ".join(str(k) for k in sorted(STAGE_SCRIPTS)))
        return 1
    num = sys.argv[1].replace("207max", "207_max")
    if num == "207_max":
        script = REPO / "_stage207_max.py"
    elif num.isdigit():
        name = STAGE_SCRIPTS.get(int(num))
        if not name:
            print(f"Stage {num} not in quick map. Run from root: python _stage{num}_….py")
            return 1
        script = REPO / name
    else:
        print(f"Unknown stage id: {num}")
        return 1
    if not script.exists():
        print(f"Missing {script}")
        return 1
    return subprocess.run([sys.executable, str(script)], cwd=str(REPO)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
