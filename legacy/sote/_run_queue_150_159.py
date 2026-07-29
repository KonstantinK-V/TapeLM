"""
Queue launcher: Stage 150 → 159 (after 149).

Order:
  1) _stage150_155_clean_compare_pipeline.py  (waits for 149)
  2) _stage156_157_morph_codebook_pipeline.py (156→157→158; waits for 150-155 rollup)
  3) _stage159_bpe_like_tails_codebook.py     (BPE-like tails; waits for 158)

Resume-safe: each stage skips if its decision.json already has a verdict.

Run:
  python _run_queue_150_159.py
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
LOG = RES / "_queue_150_159_log.txt"

STEPS = [
    ("150-155", ROOT / "_stage150_155_clean_compare_pipeline.py"),
    ("156-158", ROOT / "_stage156_157_morph_codebook_pipeline.py"),
    ("159", ROOT / "_stage159_bpe_like_tails_codebook.py"),
]


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main():
    RES.mkdir(exist_ok=True)
    log(f"queue 150-159 start {datetime.now(timezone.utc).isoformat()}")
    for name, script in STEPS:
        if not script.exists():
            log(f"[FAIL] missing {script.name}")
            return 1
        log(f"\n===== QUEUE STEP {name}: {script.name} =====")
        rc = subprocess.call([sys.executable, "-u", str(script)], cwd=str(ROOT))
        log(f"[queue] {name} exit={rc}")
        if rc != 0:
            log(f"[queue] STOPPED on {name}")
            return rc
    log("queue 150-159 DONE")
    (RES / "stage150_159_queue_decision.json").write_text(
        '{"verdict":"QUEUE_DONE","protocol":"150_to_159"}\n',
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
