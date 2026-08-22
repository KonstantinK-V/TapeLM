import json
from pathlib import Path

d = json.loads(Path("results/_stage417h_densepin.json").read_text(encoding="utf-8"))
print("seed  n    thin  live   ora    rnd    dlt    ref1   ref2   VOID  GATE  REF")
nv = ng = nr = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void = bool(h.get("void"))
    gate = bool(h.get("gate"))
    ref = bool(h.get("refuse_ok"))
    nv += int(not void)
    ng += int(gate)
    nr += int(ref)
    r1, r2 = h.get("refuse_df1"), h.get("refuse_df2")
    print(
        f"{s}  {h.get('n')}  {h.get('thin')}  "
        f"{(h.get('live') or 0):.3f}  {(h.get('oracle') or 0):.3f}  "
        f"{(h.get('random') or 0):.3f}  {(h.get('oracle_minus_random') or 0):+.3f}  "
        f"{r1 if r1 is None else f'{r1:.3f}'}  "
        f"{r2 if r2 is None else f'{r2:.3f}'}  "
        f"{void}  {gate}  {ref}"
    )
print(f"alive(not VOID) {nv}/3   GATE {ng}/3   refuse {nr}/3")
if nv == 0:
    print("VERDICT: VOID — 417 one-word OR is not this; do not CE on 417")
elif ng == 3:
    print("VERDICT: honest joint teacher exists — CE y may be wired")
else:
    print("VERDICT: mixed/coin — do not train Phi on these labels")
