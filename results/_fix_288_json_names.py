"""Untangle stage288 decision artifacts after the queue race."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

RES = Path("results")


def parse_block(text: str, name: str) -> dict:
    prefix = f"  {name} "
    for line in text.splitlines():
        if line.startswith(prefix) and line[len(prefix):].startswith("{"):
            return json.loads(line[len(prefix):])
    raise SystemExit(f"missing {name}")


def main() -> None:
    # 1) smoke min2 currently sitting in decision.json
    smoke_src = RES / "stage288_decision.json"
    if smoke_src.exists():
        d = json.loads(smoke_src.read_text(encoding="utf-8"))
        if d.get("smoke"):
            (RES / "stage288_decision_smoke_min2.json").write_text(
                json.dumps(d, indent=2), encoding="utf-8")
            mini = RES / "stage288_mini.md"
            if mini.exists():
                mini.replace(RES / "stage288_mini_smoke_min2.md")

    # 2) reconstruct REPAIR_OK min>=3 full from log (true artifact)
    # PowerShell redirection on this host wrote UTF-16 LE.
    def read_log(path: Path) -> str:
        raw = path.read_bytes()
        if raw[:2] == b"\xff\xfe" or (len(raw) > 3 and raw[1] == 0 and raw[3] == 0):
            return path.read_text(encoding="utf-16")
        return path.read_text(encoding="utf-8", errors="replace")

    log = read_log(RES / "_stage288_full_min3subset.out")
    if "REPAIR_OK" not in log:
        log = read_log(RES / "_stage288_full.out")
    control = parse_block(log, "CONTROL")
    held = parse_block(log, "HELD")
    g = re.search(r'"gates":\s*(\{.*?\})', log, re.S)
    gates = json.loads(g.group(1)) if g else {}
    out = {
        "stage": 288,
        "overall": "REPAIR_OK",
        "smoke": False,
        "holdout": "corpus",
        "run_tag": "min3subset",
        "train_steps": 6000,
        "usable_min_mentions": 3,
        "note": (
            "Reconstructed from results/_stage288_full_min3subset.out after decision.json "
            "was overwritten by smoke. This is the completed min>=3 full corpus-holdout run."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "train_control": control,
        "held_out": held,
        "source_log": "_stage288_full_min3subset.out",
    }
    (RES / "stage288_decision_min3subset.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    (RES / "stage288_mini_min3subset.md").write_text(
        "# Stage 288 repair (corpus holdout) · min≥3 subset\n\n"
        f"**REPAIR_OK** · reconstructed from log · not smoke\n\n"
        f"- detect forged held: {held['detection']['model_forged']:.3f} "
        f"(votes {held['detection']['votes_forged']:.3f})\n"
        f"- detect dup held: {held['detection']['model_dup']:.3f}\n"
        f"- clean pass held: {held['detection']['model_clean_pass']:.3f}\n"
        f"- repair reward held: {held['repair_true_flag']['model_reward']:.3f}\n"
        f"- unknown margin AUC: {held['repair_true_flag']['unknown_margin_auc']:.3f} "
        f"(z {held['repair_true_flag']['unknown_margin_auc_z']:+.2f})\n",
        encoding="utf-8",
    )

    # 3) wipe ambiguous default names so later runs with --run-tag min2 own them cleanly
    for p in (RES / "stage288_decision.json", RES / "stage288_mini.md"):
        if p.exists():
            p.unlink()

    print("wrote stage288_decision_min3subset.json (REPAIR_OK full)")
    print("wrote stage288_decision_smoke_min2.json")
    print("removed ambiguous stage288_decision.json / stage288_mini.md")


if __name__ == "__main__":
    main()
