"""Check of 526: wiki home k, foreign curve through 4800+."""
from __future__ import annotations

from pathlib import Path

import _audit526_openxfer as M

SRC = Path("_audit526_openxfer.py")


def props(src=None):
    src = SRC.read_text(encoding="utf-8") if src is None else src
    f = []
    if "from _contract_v1 import calibrate_k" not in src:
        f.append("1. home k calibration reuse missing")
    if "SIZES = (100, 400, 1200, 2400, 4800, 9600)" not in src:
        f.append("1. extended size curve missing")
    if "--home-corpus" not in src or "required=True" not in src:
        f.append("1. home + foreign corpus args missing")
    if "from _audit519_highcap import eval_cap" not in src:
        f.append("1. 519 walk missing")
    gate = src[src.find("gate ="):src.find("rec = dict")]
    if "mlast.get(\"d2\"" not in gate or "m100.get(\"d2\"" not in gate:
        f.append("2. GATE must compare last size vs 100")
    if 'hlast.get("d2", 99) < 1.0' not in gate:
        f.append("2. GATE missing high d2 < 1")
    return f


def main() -> int:
    fails = props()
    for x in fails:
        print("FAIL " + x)
    print(f"{len(fails)} failures" if fails else "all properties hold")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
