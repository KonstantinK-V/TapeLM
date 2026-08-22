import json
from pathlib import Path

d = json.loads(Path("results/_stage435_placepick.json").read_text(encoding="utf-8"))
print("seed  live   mixed  mind   rnd    d_hit   GATE  VOID")
nv = ng = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void, gate = bool(h.get("void")), bool(h.get("gate"))
    nv += int(void)
    ng += int(gate)
    mh = h.get("mind_hit")
    rh = h.get("rnd_hit")
    dh = h.get("d_hit")
    print(
        f"{s}  {(h.get('live') or 0):.3f}  {h.get('n_mixed')}  "
        f"{(mh if mh is not None else float('nan')):.3f}  "
        f"{(rh if rh is not None else float('nan')):.3f}  "
        f"{(dh if dh is not None else float('nan')):+.3f}  "
        f"{gate}  {void}"
    )
print(f"VOID {nv}/3  GATE {ng}/3")
if ng == 3:
    print("VERDICT: GO PICK")
elif nv == 3:
    print("VERDICT: VOID")
elif ng == 0:
    print("VERDICT: STOP PICK — this scorer on w400; 432 idea stays open")
else:
    print("VERDICT: mixed")
