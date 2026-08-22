import json
from pathlib import Path

d = json.loads(Path("results/_stage422_ordertie.json").read_text(encoding="utf-8"))
print("seed  n_live  n_tie  tie_rate  bag_t  ord_t  dlt    uniq_ag  VOID  GO  STOP")
nv = ng = ns = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void = bool(h.get("void"))
    go = bool(h.get("go"))
    stop = bool(h.get("stop"))
    nv += int(void)
    ng += int(go)
    ns += int(stop)
    ua = h.get("unique_agree")
    print(
        f"{s}  {h.get('n_live')}  {h.get('n_tie')}  "
        f"{(h.get('tie_rate') or 0):.3f}  "
        f"{(h.get('bag_tie') or 0):.3f}  {(h.get('ordered_tie') or 0):.3f}  "
        f"{(h.get('ordered_minus_bag_tie') or 0):+.3f}  "
        f"{(ua if ua is not None else float('nan')):.3f}  "
        f"{void}  {go}  {stop}"
    )
print(f"VOID {nv}/3   GO {ng}/3   STOP {ns}/3")
if nv == 3:
    print("VERDICT: VOID — almost no bag ties; 421 was bag; close order")
elif ng == 3:
    print("VERDICT: GO — order breaks ties to teacher; CE on sides next")
elif ns == 3:
    print("VERDICT: STOP — ties exist, order does not help; close order like 420")
else:
    print("VERDICT: mixed — read per seed")
