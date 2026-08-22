import json
from pathlib import Path

d = json.loads(Path("results/_stage437_len.json").read_text(encoding="utf-8"))
print("seed  contract  const_live_span           mixed_span")
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    print(
        f"{s}  {h.get('contract_holds')}  {h.get('const_live_span')}  "
        f"{h.get('mixed_span')}"
    )
    bl = h.get("by_len") or {}
    for L in ("100", "400", "1600", "4000"):
        r = bl.get(L)
        if not r:
            continue
        print(
            f"  L={L:>4}  const_live {r['const_live']:.3f}  "
            f"mixed {r['mixed_of_df2']:.3f}  hit {r['const_hit']:.2f}  "
            f"ref {r['refuse_mixed']:.2f}  GATE {r['gate']}"
        )
