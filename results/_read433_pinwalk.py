import json
from pathlib import Path

d = json.loads(Path("results/_stage433_pinwalk.json").read_text(encoding="utf-8"))
print("seed  live   ora-rnd  void  ceil_go  mind-rnd  GATE")
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    dhit = h.get("d_hit")
    ds = f"{dhit:+.3f}" if dhit is not None else "—"
    print(
        f"{s}  {(h.get('live') or 0):.3f}  {(h.get('ceil_d') or 0):+.3f}  "
        f"{h.get('void')}  {h.get('ceiling_go')}  {ds}  {h.get('gate')}"
    )
