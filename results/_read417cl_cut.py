"""Read 417cl v2: primary MATCHED. PRIOR then SIGNAL then ROOM. Per seed."""
from __future__ import annotations

import json
from pathlib import Path

d = json.loads(Path("results/_stage417cl_cut.json").read_text(encoding="utf-8"))
print("seed  matched_best  rare  miss  local  +floor  PRIOR  SIGNAL  ROOM")
np_ = ns = nr = 0
for s in ("1337", "8642", "2890"):
    h = d.get(s) or {}
    m = h.get("matched") or {}
    prior = bool(h.get("prior_removed"))
    sig = bool(h.get("signal"))
    room = bool(h.get("room"))
    # If prior failed, audit forces signal/room false — report raw flags as stored
    np_ += int(prior)
    ns += int(sig)
    nr += int(room)
    print(
        f"{s}  {m.get('best'):.4f}  {m.get('rare'):.4f}  {m.get('miss'):.4f}  "
        f"{m.get('local'):.4f}  {m.get('best_over_floor'):+.4f}  "
        f"{prior}  {sig}  {room}"
    )
print(f"PRIOR {np_}/3   SIGNAL {ns}/3   ROOM {nr}/3")
if np_ == 3 and ns == 3 and nr == 3:
    print("VERDICT: matched gap exists — counts do not already solve cut")
elif np_ < 3:
    print("VERDICT: frequency prior still decides (399/400/411) — nothing else counts")
elif ns < 3:
    print("VERDICT: no signal on matched — arena empty")
else:
    print("VERDICT: no room — lookup already answers (38.3)")
