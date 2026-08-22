import json
from pathlib import Path

d = json.loads(Path("results/_stage427_hang.json").read_text(encoding="utf-8"))
print("seed  rare_types     move   d_hang   empty  VOID  GATE")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    rt = f"{h.get('n_rare_types')}/{h.get('n_types')}"
    print(
        f"{s}  {rt:14}  {(h.get('move_rate') or 0):.3f}  "
        f"{(h.get('d_hang') or 0):+.4f}  {(h.get('empty_true') or 0):.3f}  "
        f"{void}  {gate}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if any((d.get(s) or {}).get("n_rare_types") == 0 for s in ("1337", "8642", "2890")):
    print("NOTE: rare_types==0 on a seed — do not read hang there")
if nv == 3 and all((d.get(s) or {}).get("n_rare_types", 1) == 0 for s in ("1337", "8642", "2890")):
    print("VERDICT: VOID — rare set empty")
elif ng == 3:
    print("VERDICT: GO ALGEBRA — hapax-inclusive ctx has joint; no net")
elif nv == 3:
    print("VERDICT: VOID — hang does not move / thin arena")
elif ng == 0:
    print("VERDICT: STOP ALGEBRA — rare types exist, hang blind; gap for Phi")
else:
    print("VERDICT: mixed")
