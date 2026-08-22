import json
from pathlib import Path

d = json.loads(Path("results/_stage429_hang.json").read_text(encoding="utf-8"))
print("seed  co   best_alg  go_alg  P(t>m)   GATE  VOID")
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    p = h.get("mind_gt_maj")
    ps = f"{p:.3f}" if p is not None else "—"
    print(
        f"{s}  {h.get('n_comp_only')}  {(h.get('best_algebra') or 0):+.4f}  "
        f"{h.get('go_algebra')}  {ps:7}  {h.get('gate')}  {h.get('void')}"
    )
