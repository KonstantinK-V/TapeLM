import json
from pathlib import Path

d = json.loads(Path("results/_stage431_route.json").read_text(encoding="utf-8"))
print("seed  foreign  reach  Δrand   line/neigh/frame     VOID  GATE")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    print(
        f"{s}  {h.get('n_foreign')}/{h.get('n_comp_only')}  "
        f"{(h.get('reach_any_target') or 0):.3f}  "
        f"{(h.get('delta_vs_random') or 0):+.3f}  "
        f"{h.get('edges_line')}/{h.get('edges_neigh')}/{h.get('edges_frame')}  "
        f"{void}  {gate}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if nv == 3:
    print("VERDICT: VOID — no foreign / unreachable; no stepper")
elif ng == 3:
    print("VERDICT: ROUTE EXISTS — structural beats random; not GO policy")
elif ng == 0:
    print("VERDICT: NO ROUTE — №2 closed; do not train stepper")
else:
    print("VERDICT: mixed — read line/neigh/frame first")
