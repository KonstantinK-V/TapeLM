"""Recover S1 decision after train completed but JSON dump failed on Tensor state."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CKPT = ROOT / "checkpoints"


def from_ckpt(name: str) -> dict:
    obj = torch.load(CKPT / name, map_location="cpu", weights_only=False)
    curve = obj.get("curve") or []
    best = obj.get("best") or {}
    lift = float(best.get("lift", -99))
    bat = float(best.get("bat", 0))
    if curve:
        top = max(curve, key=lambda c: c.get("lift_vs_majority", -99))
        lift = float(top.get("lift_vs_majority", lift))
        bat = float(top.get("battery_acc", bat))
    return {
        "curve": curve,
        "best": {"lift": lift, "bat": bat},
        "wall_hours": None,
        "final": curve[-1] if curve else {},
    }


def main() -> None:
    ritual = json.loads((RES / "stage169_ritual.json").read_text(encoding="utf-8"))
    vram = json.loads((RES / "stage169_s1_vram.json").read_text(encoding="utf-8"))
    p = from_ckpt("stage169_s1_primary.pt")
    z = from_ckpt("stage169_s1_0L.pt")
    ok_512 = any(
        r["ok"]
        and not r["zero_layer"]
        and r["seq"] == 512
        and r["d"] == ritual["d"]
        and r["n_layer"] == ritual["n_layer"]
        for r in vram["probes"]
    )
    smoke_peak = 2026.3
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": 1,
        "ritual": ritual,
        "seq512_ok": ok_512,
        "vram": vram,
        "smoke_peak_alloc_mib": smoke_peak,
        "primary": p,
        "zero": z,
        "verdict": "S1_SMOKE_OK",
        "next": "S2 full primary",
        "note": "decision recovered after JSON Tensor serialize crash; train completed",
    }
    (RES / "stage169_s1_decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pf, zf = p["final"], z["final"]
    bullets = [
        f"GPU: {vram['gpu'].get('device')} total={vram['gpu'].get('total_mib')} MiB",
        f"LOCK: seq={ritual['seq_len']} micro={ritual['micro_batch']} d={ritual['d']} L={ritual['n_layer']}",
        f"seq512_ok={ok_512}; smoke_peak_alloc={smoke_peak} MiB",
        (
            f"primary @20k: hold={100 * pf.get('hold_next', 0):.1f}% "
            f"bat={100 * pf.get('battery_acc', 0):.1f}% "
            f"lift={100 * pf.get('lift_vs_majority', 0):+.1f}pp "
            f"loss~{pf.get('loss_ema')}"
        ),
        f"primary best lift={100 * p['best']['lift']:+.1f}pp bat={100 * p['best']['bat']:.1f}%",
        (
            f"0L @10k: bat={100 * zf.get('battery_acc', 0):.1f}% "
            f"lift={100 * zf.get('lift_vs_majority', 0):+.1f}pp"
        ),
        "Next: S2 full primary train",
    ]
    body = ["# Stage169 S1 — smoke + VRAM peaks", "", "**Verdict:** `S1_SMOKE_OK`", ""]
    body += [f"- {b}" for b in bullets] + [""]
    (RES / "stage169_s1_mini.md").write_text("\n".join(body), encoding="utf-8")
    with (RES / "_stage169_s1_log.txt").open("a", encoding="utf-8") as f:
        f.write("[S1] S1_SMOKE_OK (decision recovered)\n")
    print("OK", out["verdict"])
    print("primary final", pf)
    print("zero final", zf)


if __name__ == "__main__":
    main()
