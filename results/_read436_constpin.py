import json
from pathlib import Path

d = json.loads(Path("results/_stage436_constpin.json").read_text(encoding="utf-8"))
print("seed  const_live  mixed_of_df2  const_hit  refuse_m  GATE  VOID")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    print(
        f"{s}  {(h.get('const_live') or 0):.3f}  {(h.get('mixed_of_df2') or 0):.3f}  "
        f"{(h.get('const_hit') or 0):.3f}  {(h.get('refuse_mixed') or 0):.3f}  "
        f"{gate}  {void}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if ng == 3:
    print("VERDICT: GO THINK — agree pin; differ refuse; mixed is counter")
elif nv == 3:
    print("VERDICT: VOID")
else:
    print("VERDICT: STOP or mixed")
