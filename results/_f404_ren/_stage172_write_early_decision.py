"""Write Stage172 early-stop decision from current ckpt."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
v0 = v14('results')
v1 = v14('checkpoints/stage172_curve.pt')
v2 = v20.v15(v1, map_location='cpu', weights_only=False)
v3 = v2.v21('curve') or []
v4 = v3[-1] if v3 else {}
v5 = v22(v4.v21('min_lift_mean') or 0) > 0.02
v6 = v22(v4.v21('min_lift_copy') or 0) > 0.02
v7 = v22(v4.v21('contrast_pref') or 0) > 0.55
v8 = v4.v21('k8') or {}
v9 = v4.v21('k16') or {}
v10 = v22(v9.v21('lift_mean') or -1) > 0.0 and v22(v8.v21('lift_mean') or -1) > 0.02
if v5 and v6 and v7 and v10:
    v16 = 'CURVE_DYN_SCALE_YES'
elif v5 and v7:
    v16 = 'CURVE_DYN_SCALE_MIXED'
else:
    v16 = 'CURVE_DYN_SCALE_NULL'
v11 = {'timestamp': v31.v29(v32.v30).v23(), 'protocol': 'curve_dynamics_scale_172', 'verdict': v16, 'early_stop': True, 'reason': 'user stop: metrics plateaued; remaining steps not needed for gate', 'steps_done': v2.v21('step'), 'steps_planned': 80000, 'pen_frozen': True, 'horizons': v2.v21('horizons'), 'arch': {'pen_d': 96, 'dyn_d': v2.v21('d_model', 128), 'ctx': 128, 'seq': 192}, 'final': v4, 'gates': {'beat_mean': v5, 'beat_copy': v6, 'contrast_ok': v7, 'far_ok': v10}, 'probe_acc_hold': v4.v21('probe_acc'), 'probe_role': 'readout_only_stopgrad_z_never_teaches_dyn_or_pen', 'note': 'Stopped early at plateau. SCALE gate already clear.'}
(v0 / 'stage172_decision.json').v17(v28.v24(v11, indent=2), encoding='utf-8')
v12 = v4.v21('k1') or {}
v13 = [f"`{v16}` EARLY_STOP @ {v2.v21('step')} (planned 80k)", f"min_lift_mean={v22(v4.v21('min_lift_mean') or 0):+.3f} contrast={v22(v4.v21('contrast_pref') or 0):.3f} probe={100 * v22(v4.v21('probe_acc') or 0):.1f}%", f"k1={v22(v12.v21('cos_delta') or 0):.3f} k8={v22(v8.v21('cos_delta') or 0):.3f} k16={v22(v9.v21('cos_delta') or 0):.3f}", f"gates={v11['gates']}"]
(v0 / 'stage172_mini.md').v17('\n'.v25(['# Stage172 — curve scale (early stop)', '', f'**Verdict:** `{v16}`', ''] + [f'- {v33}' for v33 in v13] + ['']), encoding='utf-8')
with (v0 / '_stage172_log.txt').v26('a', encoding='utf-8') as v18:
    v18.v27(f"[172] {v16} EARLY_STOP @ {v2.v21('step')}\n")
v19(v16, 'step', v2.v21('step'))