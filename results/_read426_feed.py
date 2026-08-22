import json
from pathlib import Path

d = json.loads(Path("results/_stage426_feed.json").read_text(encoding="utf-8"))
print("seed  co_n  both  cross>0  rare>0  hang>0  text_both")
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    co = h.get("comp_only") or {}
    print(
        f"{s}  {co.get('n')}  {(co.get('both_nonempty') or 0):.3f}  "
        f"{(co.get('share_cross_gt0') or 0):.3f}  "
        f"{(co.get('share_rare_gt0') or 0):.3f}  "
        f"{(co.get('share_hang_gt0') or 0):.3f}  "
        f"{(co.get('share_text_both') or 0):.3f}"
    )
