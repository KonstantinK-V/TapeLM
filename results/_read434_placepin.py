import json
from pathlib import Path

d = json.loads(Path("results/_stage434_placepin.json").read_text(encoding="utf-8"))
print("seed  live   mixed  d_mixed  d_const  VOID  GATE")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    print(
        f"{s}  {(h.get('live') or 0):.3f}  {h.get('n_mixed')}  "
        f"{(h.get('d_mixed') or 0):+.3f}  {(h.get('d_const') or 0):+.3f}  "
        f"{void}  {gate}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if nv == 3:
    print("VERDICT: VOID — place foreign cells do not hold v")
elif ng == 3:
    print("VERDICT: GO OFFER — mixed e(P) beats random; then train pick")
elif ng == 0:
    print("VERDICT: STOP or catalog — no train yet")
else:
    print("VERDICT: mixed")
