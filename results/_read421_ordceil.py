import json
from pathlib import Path

d = json.loads(Path("results/_stage421_ordceil.json").read_text(encoding="utf-8"))
print("seed  n_live  bag    ord    rnd    ord-bag  ord-rnd  VOID  GO  STOP")
ng = ns = nv = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    void = bool(h.get("void"))
    go = bool(h.get("go"))
    stop = bool(h.get("stop"))
    nv += int(void)
    ng += int(go)
    ns += int(stop)
    print(
        f"{s}  {h.get('n_live')}  "
        f"{(h.get('bag_live') or 0):.3f}  {(h.get('ordered_live') or 0):.3f}  "
        f"{(h.get('random_live') or 0):.3f}  "
        f"{(h.get('ordered_minus_bag') or 0):+.3f}  "
        f"{(h.get('ordered_minus_random') or 0):+.3f}  "
        f"{void}  {go}  {stop}"
    )
print(f"VOID {nv}/3   GO {ng}/3   STOP {ns}/3")
if nv == 3:
    print("VERDICT: VOID — no live pins")
elif ng == 3:
    print("VERDICT: GO — ordered beats bag and random; CE with ordered feats may follow")
elif ns == 3:
    print("VERDICT: STOP — ordered ≈ bag; do not feed order to Phi")
else:
    print("VERDICT: mixed — read per seed")
