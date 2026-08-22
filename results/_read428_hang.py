import json
from pathlib import Path

d = json.loads(Path("results/_stage428_hang.json").read_text(encoding="utf-8"))
print("seed  bridge        move   d_hang   empty  VOID  GATE")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    br = f"{h.get('n_bridge_types')}/{h.get('n_types')}"
    print(
        f"{s}  {br:14}  {(h.get('move_rate') or 0):.3f}  "
        f"{(h.get('d_hang') or 0):+.4f}  {(h.get('empty_true') or 0):.3f}  "
        f"{void}  {gate}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if all((d.get(s) or {}).get("n_bridge_types", 1) == 0 for s in ("1337", "8642", "2890")):
    print("VERDICT: VOID — no bridge types; do not read hang")
elif ng == 3:
    print("VERDICT: GO ALGEBRA — joint in 2..5-df overlap; no net")
elif nv == 3:
    print("VERDICT: VOID — hang does not move / thin; not pair scorer")
elif ng == 0:
    print("VERDICT: STOP ALGEBRA — bridge exists, hang blind; gap for Phi")
else:
    print("VERDICT: mixed")
print("Fourth threshold after this = shopping; do not run.")
