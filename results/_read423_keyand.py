import json
from pathlib import Path

d = json.loads(Path("results/_stage423_keyand.json").read_text(encoding="utf-8"))
print("seed  used  empty  AND    single  bag    rnd    AND-sng  GATE  VOID")
ng = nv = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    gate = bool(h.get("gate"))
    void = bool(h.get("void"))
    ng += int(gate)
    nv += int(void)
    print(
        f"{s}  {h.get('used')}  {(h.get('empty_and') or 0):.3f}  "
        f"{(h.get('and_live') or 0):.3f}  {(h.get('single_live') or 0):.3f}  "
        f"{(h.get('bag_live') or 0):.3f}  {(h.get('random_live') or 0):.3f}  "
        f"{(h.get('and_minus_single') or 0):+.3f}  {gate}  {void}"
    )
print(f"GATE {ng}/3   VOID {nv}/3")
if nv == 3:
    print("VERDICT: VOID — AND almost never fires")
elif ng == 3:
    print("VERDICT: GO — AND purer than better single; formation before Phi")
elif ng == 0:
    print("VERDICT: STOP — AND is thinner single; stop before Phi")
else:
    print("VERDICT: mixed — read per seed")
