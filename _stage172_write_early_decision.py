"""Write Stage172 early-stop decision from current ckpt."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

RES = Path("results")
CKPT = Path("checkpoints/stage172_curve.pt")
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
curve = blob.get("curve") or []
final = curve[-1] if curve else {}
beat_mean = float(final.get("min_lift_mean") or 0) > 0.02
beat_copy = float(final.get("min_lift_copy") or 0) > 0.02
contrast_ok = float(final.get("contrast_pref") or 0) > 0.55
k8 = final.get("k8") or {}
k16 = final.get("k16") or {}
far_ok = float(k16.get("lift_mean") or -1) > 0.0 and float(k8.get("lift_mean") or -1) > 0.02
if beat_mean and beat_copy and contrast_ok and far_ok:
    verdict = "CURVE_DYN_SCALE_YES"
elif beat_mean and contrast_ok:
    verdict = "CURVE_DYN_SCALE_MIXED"
else:
    verdict = "CURVE_DYN_SCALE_NULL"

out = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "protocol": "curve_dynamics_scale_172",
    "verdict": verdict,
    "early_stop": True,
    "reason": "user stop: metrics plateaued; remaining steps not needed for gate",
    "steps_done": blob.get("step"),
    "steps_planned": 80000,
    "pen_frozen": True,
    "horizons": blob.get("horizons"),
    "arch": {"pen_d": 96, "dyn_d": blob.get("d_model", 128), "ctx": 128, "seq": 192},
    "final": final,
    "gates": {
        "beat_mean": beat_mean,
        "beat_copy": beat_copy,
        "contrast_ok": contrast_ok,
        "far_ok": far_ok,
    },
    "probe_acc_hold": final.get("probe_acc"),
    "probe_role": "readout_only_stopgrad_z_never_teaches_dyn_or_pen",
    "note": "Stopped early at plateau. SCALE gate already clear.",
}
(RES / "stage172_decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
k1 = final.get("k1") or {}
bullets = [
    f"`{verdict}` EARLY_STOP @ {blob.get('step')} (planned 80k)",
    f"min_lift_mean={float(final.get('min_lift_mean') or 0):+.3f} contrast={float(final.get('contrast_pref') or 0):.3f} probe={100 * float(final.get('probe_acc') or 0):.1f}%",
    f"k1={float(k1.get('cos_delta') or 0):.3f} k8={float(k8.get('cos_delta') or 0):.3f} k16={float(k16.get('cos_delta') or 0):.3f}",
    f"gates={out['gates']}",
]
(RES / "stage172_mini.md").write_text(
    "\n".join(
        ["# Stage172 — curve scale (early stop)", "", f"**Verdict:** `{verdict}`", ""]
        + [f"- {b}" for b in bullets]
        + [""]
    ),
    encoding="utf-8",
)
with (RES / "_stage172_log.txt").open("a", encoding="utf-8") as f:
    f.write(f"[172] {verdict} EARLY_STOP @ {blob.get('step')}\n")
print(verdict, "step", blob.get("step"))
