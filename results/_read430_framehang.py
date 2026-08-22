import json
from pathlib import Path

d = json.loads(Path("results/_stage430_framehang.json").read_text(encoding="utf-8"))
print("seed  co   move   d_hang   mean_fr  VOID  GATE")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    print(
        f"{s}  {h.get('n_comp_only')}  {(h.get('move_rate') or 0):.3f}  "
        f"{(h.get('d_hang') or 0):+.4f}  {(h.get('mean_frame') or 0):.2f}  "
        f"{void}  {gate}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if nv == 3:
    print("VERDICT: VOID — frame hang not a pair scorer")
elif ng == 3:
    print("VERDICT: GO ALGEBRA — joint in frame overlap; no net; no Phi")
elif ng == 0:
    print("VERDICT: STOP ALGEBRA — frames exist, hang blind; do not Phi (not 429)")
else:
    print("VERDICT: mixed")
