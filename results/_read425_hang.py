import json
from pathlib import Path

d = json.loads(Path("results/_stage425_hang.json").read_text(encoding="utf-8"))
print("seed  offered  comp  move   d_hang   VOID  GATE")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    print(
        f"{s}  {h.get('n_offered')}  {h.get('n_comp_only')}  "
        f"{(h.get('move_rate') or 0):.3f}  {(h.get('d_hang') or 0):+.4f}  "
        f"{void}  {gate}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if nv == 3:
    print("VERDICT: VOID — hang not a pair scorer; not a Phi gap")
elif ng == 3:
    print("VERDICT: GO ALGEBRA — joint in ctx graph; no net needed")
elif ng == 0:
    print("VERDICT: STOP ALGEBRA — arena live, hang blind; gap for Phi")
else:
    print("VERDICT: mixed")
