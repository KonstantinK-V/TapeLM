import json
from pathlib import Path

d = json.loads(Path("results/_stage432_place.json").read_text(encoding="utf-8"))
print("seed  foreign  mixed  best_d   VOID  GATE")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    print(
        f"{s}  {(h.get('foreign_nonempty') or 0):.3f}  "
        f"{(h.get('mixed') or 0):.3f}  {(h.get('best_d') or 0):+.3f}  "
        f"{void}  {gate}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if nv == 3:
    print("VERDICT: VOID — no foreign or no mixed; no second tape")
elif ng == 3:
    print("VERDICT: GO PLACE — structure beats maj on mixed; then second tape")
elif ng == 0:
    print("VERDICT: STOP PLACE — mixed exists, LINE/FRAME lose; no pair-Phi, no second tape")
else:
    print("VERDICT: mixed")
